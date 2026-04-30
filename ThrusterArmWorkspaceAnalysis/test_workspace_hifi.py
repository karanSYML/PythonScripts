#!/usr/bin/env python3
"""
test_workspace_hifi.py
======================

Validates whether the fast erosion proxy (Σ cos^n(θ)/r²) correctly ranks
arm poses by erosion risk relative to the high-fidelity sputter_erosion
integrator.

Pipeline:
  1. Build a 12×12×12 joint-space grid (1728 cells)
  2. Compute static FK quantities (p_nozzle, t_hat) for every cell
  3. Apply F_kin (collision + joint limits), F_align (NSSK-N), F_CoG filters
  4. Compute the fast erosion proxy for every feasible cell
  5. Sample N_SAMPLE=20 poses spread across proxy-score quintiles
  6. Run the high-fidelity ErosionIntegrator on each sampled pose
  7. Print a ranked comparison table and Spearman rank correlation

Note: F_kin is computed with a Python loop over all cells.
      Expect ~30-90 s on a typical workstation for 1728 cells.
"""

import sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sputter_erosion"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm

from plume_impingement_pipeline import (
    RoboticArmGeometry, StackConfig, ThrusterParams,
    arm_has_collision,
)
from feasibility_cells import (
    FeasibilityConfig, SK_DIRECTIONS,
    build_joint_grid, compute_static_cell_quantities, compute_F_kin,
    compute_alpha, compute_F_align, compute_r_miss, compute_F_CoG,
)
from feasibility_map import compute_pivot
from sputter_erosion import (
    Vector3,
    ThrusterPlacement as _SE_ThrusterPlacement,
    SolarArray as _SE_SolarArray,
    Interconnect as _SE_Interconnect,
    SatelliteGeometry as _SE_SatelliteGeometry,
    SheathModel as _SE_SheathModel,
    HallThrusterPlume as _SE_HallThrusterPlume,
    SpeciesFractions as _SE_SpeciesFractions,
    EcksteinPreuss as _SE_EcksteinPreuss,
    EcksteinAngular as _SE_EcksteinAngular,
    ErosionIntegrator as _SE_ErosionIntegrator,
)
from sputter_erosion.yields import FullYield as _SE_FullYield
from scipy.stats import spearmanr, pearsonr


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SERVICER_YAW_DEG = -25.0
GRID_RES         = (12, 12, 12)   # 1728 cells — fast for a test
N_SAMPLES        = 20             # poses sampled across proxy quintiles
FIRING_SNAPSHOT_S = 3600.0        # 1-hour snapshot for ranking
PANEL_N_SPAN     = 12             # panel sample density (reduced for speed)
PANEL_N_CHORD    = 4

ARM   = RoboticArmGeometry()
STACK = StackConfig(
    servicer_mass=744.0,
    servicer_bus_x=0.9, servicer_bus_y=1.5, servicer_bus_z=0.8,
    client_mass=2800.0,
    client_bus_x=2.3, client_bus_y=3.0, client_bus_z=5.0,
    panel_span_one_side=16.0, panel_width=2.5,
    lar_offset_z=0.05,
)
THRUSTER_HW = ThrusterParams(
    discharge_voltage=300.0,
    mass_flow_rate=5e-6,
    beam_divergence_half_angle=20.0,
    plume_cosine_exponent=10.0,
    thrust_N=0.08,
    xe1_fraction=0.78,
    xe2_fraction=0.18,
    xe3_fraction=0.04,
    sheath_potential_V=20.0,
)
# NSSK North direction in LAR frame
D_HAT_NSSK_N = SK_DIRECTIONS["N"]

# Feasibility thresholds (use library defaults)
FEAS_CFG = FeasibilityConfig(
    eps_CoG_m=0.05,
    alpha_max_deg=5.0,
    grid_resolution=GRID_RES,
)


# ──────────────────────────────────────────────────────────────────────────────
# Inline panel-grid and proxy (avoid importing workspace_erosion_viz which
# pulls in matplotlib + plotly at module level)
# ──────────────────────────────────────────────────────────────────────────────

def _panel_grid_lar(stack: StackConfig, tracking_deg: float = 0.0) -> np.ndarray:
    """Solar panel sample points in LAR frame (both wings), shape (N, 3)."""
    track   = np.radians(tracking_deg)
    z_hinge = stack.client_bus_z / 2.0
    pts = []
    for side in [+1, -1]:
        x_hinge = side * stack.client_bus_x / 2.0
        for i in range(PANEL_N_SPAN):
            xi = x_hinge + side * ((i + 0.5) / PANEL_N_SPAN) * stack.panel_span_one_side
            for j in range(PANEL_N_CHORD):
                frac = (j + 0.5) / PANEL_N_CHORD - 0.5
                yi = frac * stack.panel_width
                pts.append([xi, yi * np.cos(track), z_hinge + yi * np.sin(track)])
    return np.array(pts)


def _erosion_proxy(
    p_nozzle: np.ndarray,   # (N, 3)
    plume_dir: np.ndarray,  # (N, 3) unit plume direction
    panel_pts: np.ndarray,  # (M, 3)
    n_exp: float,
) -> np.ndarray:
    """Σ_j cos^n(θ_ij)/r_ij²  shape (N,)"""
    dv   = panel_pts[np.newaxis, :, :] - p_nozzle[:, np.newaxis, :]  # (N,M,3)
    dist = np.linalg.norm(dv, axis=-1)
    dist = np.where(dist < 0.02, 0.02, dist)
    cos_th = np.einsum('nmi,ni->nm', dv / dist[..., np.newaxis], plume_dir)
    cos_th = np.clip(cos_th, 0.0, 1.0)
    flux = (cos_th ** n_exp) / dist ** 2
    return flux.sum(axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# High-fidelity integrator for a single arm pose
# ──────────────────────────────────────────────────────────────────────────────

_YIELD_MODEL = _SE_FullYield(
    energy_model=_SE_EcksteinPreuss(),
    angular_model=_SE_EcksteinAngular(),
    subthreshold_floor=0.0,
)
_INTEGRATOR = _SE_ErosionIntegrator(
    yield_model=_YIELD_MODEL,
    include_xe2=True,
    include_xe3=True,
    apply_sheath=True,
)


def _make_hall_plume() -> _SE_HallThrusterPlume:
    from plume_impingement_pipeline import ErosionEstimator, MaterialParams
    estimator = ErosionEstimator(THRUSTER_HW, MaterialParams(name="Ag"))
    return _SE_HallThrusterPlume(
        V_d=THRUSTER_HW.discharge_voltage,
        I_beam=estimator.beam_current_A(),
        mdot_neutral=THRUSTER_HW.mass_flow_rate,
        half_angle_90=np.deg2rad(THRUSTER_HW.beam_divergence_half_angle),
        cex_wing_amp=0.025,
        cex_wing_width=np.deg2rad(40.0),
        species=_SE_SpeciesFractions(
            THRUSTER_HW.xe1_fraction,
            THRUSTER_HW.xe2_fraction,
            THRUSTER_HW.xe3_fraction,
        ),
        sheath_potential=THRUSTER_HW.sheath_potential_V,
    )


def _hifi_for_pose(
    p_nozzle: np.ndarray,  # (3,) LAR
    plume_dir: np.ndarray, # (3,) unit, direction ion beam points
    panel_pts: np.ndarray, # (M, 3) LAR
    hall_plume: _SE_HallThrusterPlume,
    stack: StackConfig,
) -> float:
    """Return max interconnect thinning [nm] over FIRING_SNAPSHOT_S."""
    thruster = _SE_ThrusterPlacement(
        position_body=Vector3(*p_nozzle),
        fire_direction_body=Vector3(*plume_dir),
        plume=hall_plume,
    )

    # Panel geometry — same conventions as _build_sputter_geometry in pipeline
    panel_norm = np.array([0.0, 0.0, -1.0])   # anti-nadir face
    x_panel    = np.array([1.0, 0.0,  0.0])   # span axis
    y_panel    = np.cross(panel_norm, x_panel) # [0, -1, 0]
    y_panel   /= np.linalg.norm(y_panel)

    origin = panel_pts[0]
    n_pts  = len(panel_pts)
    interconnects = []
    for i, pt in enumerate(panel_pts):
        delta = pt - origin
        px = float(np.dot(delta, x_panel))
        py = float(np.dot(delta, y_panel))

        to_thr      = p_nozzle - pt
        to_thr_ip   = to_thr - np.dot(to_thr, panel_norm) * panel_norm
        ip_norm     = np.linalg.norm(to_thr_ip)
        edge_dir    = to_thr_ip / ip_norm if ip_norm > 1e-6 else x_panel
        en_x = float(np.dot(edge_dir, x_panel))
        en_y = float(np.dot(edge_dir, y_panel))
        en_z = float(np.dot(edge_dir, panel_norm))

        interconnects.append(_SE_Interconnect(
            position_local=(px, py),
            exposed_face_normal=Vector3(en_x, en_y, en_z),
            material_name="Ag",
            string_position=float(i) / max(n_pts - 1, 1),
            exposed_thickness=25e-6,
        ))

    solar_array = _SE_SolarArray(
        origin_body=Vector3(*origin),
        panel_normal_body=Vector3(*panel_norm),
        panel_x_body=Vector3(*x_panel),
        width=stack.panel_span_one_side * 2.0,
        height=stack.panel_width,
        interconnects=interconnects,
    )
    sheath = _SE_SheathModel(string_voltage=100.0,
                             floating_potential=-15.0, Te_local=2.0)
    sat_geo = _SE_SatelliteGeometry(
        thrusters=[thruster],
        solar_arrays=[solar_array],
        sheath=sheath,
    )

    results = _INTEGRATOR.evaluate(sat_geo, FIRING_SNAPSHOT_S)
    if not results:
        return 0.0
    return float(max(r.total_thinning_m for r in results)) * 1e9  # nm


# ──────────────────────────────────────────────────────────────────────────────
# Main test
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  test_workspace_hifi.py — proxy vs. high-fidelity erosion ranking   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Grid       : {GRID_RES[0]}×{GRID_RES[1]}×{GRID_RES[2]} = "
          f"{np.prod(GRID_RES):,} cells")
    print(f"  Sample size: {N_SAMPLES} poses across proxy quintiles")
    print(f"  Direction  : NSSK North  α_max={FEAS_CFG.alpha_max_deg}°  "
          f"ε_CoG={FEAS_CFG.eps_CoG_m} m")
    print()

    # ------------------------------------------------------------------
    # Step 1: Joint grid + static FK
    # ------------------------------------------------------------------
    t0 = time.time()
    pivot  = compute_pivot(ARM, STACK, SERVICER_YAW_DEG)
    n_hat  = ARM.n_hat_body
    q0g, q1g, q2g = build_joint_grid(ARM, resolution=GRID_RES)
    cq = compute_static_cell_quantities(
        ARM, pivot, n_hat, q0g, q1g, q2g,
        servicer_yaw_deg=SERVICER_YAW_DEG,
    )
    print(f"  FK completed in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: F_kin (joint limits + collision)
    # ------------------------------------------------------------------
    print(f"  Computing F_kin over {np.prod(GRID_RES):,} cells "
          f"(expect 30–90 s) ...", flush=True)
    t1 = time.time()
    F_kin = compute_F_kin(
        ARM, pivot, STACK, SERVICER_YAW_DEG,
        cq, q0g, q1g, q2g, verbose=True,
    )
    print(f"  F_kin done in {time.time()-t1:.1f}s  "
          f"({F_kin.sum()} / {np.prod(GRID_RES)} pass)")

    # ------------------------------------------------------------------
    # Step 3: F_align + F_CoG (NSSK North, BOL CoG)
    # ------------------------------------------------------------------
    p_CoG = STACK.stack_cog()
    alpha   = compute_alpha(cq['t_hat'], D_HAT_NSSK_N)
    r_miss  = compute_r_miss(cq['p_nozzle'], cq['t_hat'], p_CoG)
    F_align = compute_F_align(alpha, FEAS_CFG.alpha_max_rad)
    F_CoG   = compute_F_CoG(r_miss, FEAS_CFG.eps_CoG_m)
    F_total = F_kin & F_align & F_CoG
    n_feas  = int(F_total.sum())
    print(f"  F_align={F_align.sum()}  F_CoG={F_CoG.sum()}  "
          f"F_total (NSSK-N)={n_feas}")
    print()

    if n_feas == 0:
        print("  No feasible cells found for NSSK-N. "
              "Widening alpha_max to 15° for proxy demo.")
        F_align2 = compute_F_align(alpha, np.radians(15.0))
        F_total  = F_kin & F_align2
        n_feas   = int(F_total.sum())
        print(f"  F_total with α_max=15°: {n_feas} cells")

    if n_feas == 0:
        print("  Still no feasible cells. Using all F_kin-passing cells.")
        F_total = F_kin
        n_feas  = int(F_total.sum())

    # ------------------------------------------------------------------
    # Step 4: Erosion proxy for all feasible cells
    # ------------------------------------------------------------------
    panel_pts = _panel_grid_lar(STACK, tracking_deg=0.0)
    idx_feas  = np.argwhere(F_total)               # (n_feas, 3)

    p_noz_feas   = cq['p_nozzle'][F_total]         # (n_feas, 3)
    t_hat_feas   = cq['t_hat'][F_total]            # (n_feas, 3)
    plume_dir_feas = -t_hat_feas                   # plume exits opposite to thrust

    n_exp = THRUSTER_HW.plume_cosine_exponent
    print(f"  Computing proxy for {n_feas} feasible poses × {len(panel_pts)} panel pts ...",
          flush=True)
    t2 = time.time()
    proxy_scores = _erosion_proxy(p_noz_feas, plume_dir_feas, panel_pts, n_exp)
    print(f"  Proxy done in {time.time()-t2:.2f}s")
    print(f"  Proxy range: [{proxy_scores.min():.4e}, {proxy_scores.max():.4e}]")
    print()

    # ------------------------------------------------------------------
    # Step 5: Sample N_SAMPLES poses across proxy quintiles
    # ------------------------------------------------------------------
    n_actual = min(N_SAMPLES, n_feas)
    q_edges  = np.percentile(proxy_scores, np.linspace(0, 100, 6))
    sample_idxs = []  # indices into p_noz_feas / proxy_scores arrays

    per_quintile = max(1, n_actual // 5)
    for qi in range(5):
        lo, hi = q_edges[qi], q_edges[qi + 1]
        in_q = np.where((proxy_scores >= lo) & (proxy_scores <= hi))[0]
        if len(in_q) == 0:
            continue
        chosen = np.random.default_rng(seed=qi).choice(
            in_q, size=min(per_quintile, len(in_q)), replace=False
        )
        sample_idxs.extend(chosen.tolist())

    sample_idxs = sorted(set(sample_idxs))  # deduplicate, keep sorted
    print(f"  Sampled {len(sample_idxs)} poses from {5} proxy quintiles")
    print()

    # ------------------------------------------------------------------
    # Step 6: High-fidelity integrator for each sampled pose
    # ------------------------------------------------------------------
    hall_plume = _make_hall_plume()

    print(f"  Running HF integrator on {len(sample_idxs)} poses "
          f"({FIRING_SNAPSHOT_S/3600:.0f}-hr snapshot) ...", flush=True)
    t3 = time.time()
    hifi_values   = []
    proxy_sampled = []
    nozzle_positions = []

    for rank, si in enumerate(sample_idxs):
        p_noz = p_noz_feas[si]
        p_dir = plume_dir_feas[si]
        hifi  = _hifi_for_pose(p_noz, p_dir, panel_pts, hall_plume, STACK)
        hifi_values.append(hifi)
        proxy_sampled.append(proxy_scores[si])
        nozzle_positions.append(p_noz)
        if (rank + 1) % 5 == 0:
            print(f"    [{rank+1}/{len(sample_idxs)}] "
                  f"proxy={proxy_scores[si]:.3e}  hifi={hifi:.4f} nm",
                  flush=True)

    elapsed_hifi = time.time() - t3
    print(f"  HF done in {elapsed_hifi:.1f}s  "
          f"({elapsed_hifi/len(sample_idxs):.2f}s per pose)")
    print()

    # ------------------------------------------------------------------
    # Step 7: Comparison table + rank correlation
    # ------------------------------------------------------------------
    hifi_arr  = np.array(hifi_values)
    proxy_arr = np.array(proxy_sampled)

    # Rank by proxy (descending)
    order_by_proxy = np.argsort(-proxy_arr)

    print("  Ranked by proxy score (high → low):")
    print()
    hdr = (f"  {'#':>3}  {'p_noz_x':>8}  {'p_noz_y':>8}  {'p_noz_z':>8}"
           f"  {'proxy':>10}  {'hifi[nm]':>10}  {'q_bin':>6}")
    print(hdr)
    print(f"  {'-'*65}")

    for r, i in enumerate(order_by_proxy):
        q_bin = int(np.searchsorted(q_edges[1:], proxy_arr[i], side='right'))
        p = nozzle_positions[i]
        print(f"  {r+1:>3}  "
              f"{p[0]:>8.3f}  {p[1]:>8.3f}  {p[2]:>8.3f}"
              f"  {proxy_arr[i]:>10.4e}  {hifi_arr[i]:>10.4f}  Q{q_bin+1:>1}")

    print()

    # Statistics
    rho_s, pval_s = spearmanr(proxy_arr, hifi_arr)
    rho_p, pval_p = pearsonr(np.log1p(proxy_arr), np.log1p(hifi_arr))

    print(f"  Rank correlation (Spearman ρ):    {rho_s:+.4f}  (p={pval_s:.4f})")
    print(f"  Log-linear correlation (Pearson): {rho_p:+.4f}  (p={pval_p:.4f})")
    print()
    if rho_s >= 0.6:
        print("  Interpretation: proxy CORRECTLY ranks erosion risk "
              f"(ρ={rho_s:.2f} ≥ 0.6).")
    elif rho_s >= 0.3:
        print("  Interpretation: proxy has MODERATE rank correlation "
              f"(ρ={rho_s:.2f}). Useful as a first-pass filter.")
    else:
        print("  Interpretation: proxy has WEAK rank correlation "
              f"(ρ={rho_s:.2f}). Consider incorporating incidence-angle weighting.")

    print()

    # Quintile summary: mean hifi per proxy quintile
    print("  Mean HF erosion by proxy quintile:")
    print(f"  {'Quintile':>10}  {'proxy_lo':>10}  {'proxy_hi':>10}"
          f"  {'n_poses':>7}  {'mean_hifi_nm':>13}")
    print(f"  {'-'*55}")
    for qi in range(5):
        in_q  = [(proxy_arr[i], hifi_arr[i]) for i in range(len(proxy_arr))
                 if q_edges[qi] <= proxy_arr[i] <= q_edges[qi+1]]
        if in_q:
            hifi_q = [v[1] for v in in_q]
            print(f"  {'Q'+str(qi+1):>10}  "
                  f"{q_edges[qi]:>10.3e}  {q_edges[qi+1]:>10.3e}"
                  f"  {len(hifi_q):>7}  {np.mean(hifi_q):>13.4f}")

    print()

    # ------------------------------------------------------------------
    # Step 8: Plots
    # ------------------------------------------------------------------
    out_dir = os.path.join(HERE, "pipeline_runner_output")
    os.makedirs(out_dir, exist_ok=True)
    _plot_workspace_hifi(
        p_noz_feas, proxy_scores,
        np.array(nozzle_positions), proxy_arr, hifi_arr,
        q_edges, rho_s, rho_p,
        out_path=os.path.join(out_dir, "workspace_hifi_validation.png"),
    )
    print("Done.")


def _plot_workspace_hifi(
    p_noz_all, proxy_all,
    nozzle_sample, proxy_sample, hifi_sample,
    q_edges, rho_s, rho_p,
    out_path,
):
    """Three-panel figure: workspace coloured by proxy, proxy vs HF scatter,
    and per-quintile mean-HF bar chart."""
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(
        "Workspace Erosion Proxy vs. High-Fidelity Integrator — NSSK-N Feasible Poses\n"
        f"Spearman ρ={rho_s:.3f}   Pearson r={rho_p:.3f} (log-linear)",
        fontsize=11, fontweight="bold",
    )
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    cmap = cm.plasma
    proxy_log_all  = np.log10(proxy_all + 1e-12)
    vmin, vmax = proxy_log_all.min(), proxy_log_all.max()
    norm_proxy = Normalize(vmin=vmin, vmax=vmax)

    # ── Left: 3-D nozzle workspace coloured by proxy ─────────────────────────
    ax3d = fig.add_subplot(gs[0], projection="3d")
    sc = ax3d.scatter(
        p_noz_all[:, 0], p_noz_all[:, 1], p_noz_all[:, 2],
        c=proxy_log_all, cmap=cmap, norm=norm_proxy,
        s=8, alpha=0.6, linewidths=0,
    )
    # Overlay sampled poses with larger markers
    proxy_log_samp = np.log10(proxy_sample + 1e-12)
    ax3d.scatter(
        nozzle_sample[:, 0], nozzle_sample[:, 1], nozzle_sample[:, 2],
        c=proxy_log_samp, cmap=cmap, norm=norm_proxy,
        s=60, alpha=1.0, linewidths=0.5, edgecolors="k",
        zorder=5,
    )
    ax3d.set_xlabel("X [m]", fontsize=8)
    ax3d.set_ylabel("Y [m]", fontsize=8)
    ax3d.set_zlabel("Z [m]", fontsize=8)
    ax3d.set_title("Feasible Workspace\ncoloured by proxy (log₁₀)", fontsize=9)
    ax3d.tick_params(labelsize=7)
    cb = fig.colorbar(sc, ax=ax3d, shrink=0.6, pad=0.1)
    cb.set_label("log₁₀ proxy", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ── Middle: proxy vs HF scatter ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    q_colors = cm.RdYlGn_r(np.linspace(0.1, 0.9, 5))
    for qi in range(5):
        mask = (proxy_sample >= q_edges[qi]) & (proxy_sample <= q_edges[qi + 1])
        if mask.any():
            ax2.scatter(proxy_sample[mask], hifi_sample[mask],
                        s=70, color=q_colors[qi], label=f"Q{qi+1}", zorder=3,
                        edgecolors="k", linewidths=0.4)
    # Log-log fit line
    log_x = np.log1p(proxy_sample)
    log_y = np.log1p(hifi_sample)
    if len(log_x) > 2:
        c = np.polyfit(log_x, log_y, 1)
        x_fit = np.linspace(proxy_sample.min(), proxy_sample.max(), 80)
        y_fit = np.expm1(np.polyval(c, np.log1p(x_fit)))
        ax2.plot(x_fit, y_fit, "k--", lw=1.2, alpha=0.6, label="log-log fit")
    ax2.set_xlabel("Erosion proxy (Σ cosⁿθ/r²)")
    ax2.set_ylabel("HF thinning [nm/hr]")
    ax2.set_title(f"Proxy vs. HF Integrator\nSpearman ρ={rho_s:.3f}", fontsize=9)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(fontsize=7, title="Proxy quintile")
    ax2.grid(True, alpha=0.25)

    # ── Right: per-quintile mean HF bar chart ────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    q_means, q_stds, q_ns, q_labs = [], [], [], []
    for qi in range(5):
        mask = (proxy_sample >= q_edges[qi]) & (proxy_sample <= q_edges[qi + 1])
        vals = hifi_sample[mask]
        if len(vals) > 0:
            q_means.append(vals.mean())
            q_stds.append(vals.std() if len(vals) > 1 else 0.0)
            q_ns.append(len(vals))
            q_labs.append(f"Q{qi+1}\n(n={len(vals)})")
    x = np.arange(len(q_means))
    bars = ax3.bar(x, q_means, yerr=q_stds, capsize=5,
                   color=q_colors[:len(q_means)], alpha=0.85, ecolor="k", lw=0.8)
    for bar, val in zip(bars, q_means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(q_labs, fontsize=8)
    ax3.set_ylabel("Mean HF thinning [nm/hr]")
    ax3.set_title("Mean HF Erosion per Proxy Quintile\n(error bars = 1σ)", fontsize=9)
    ax3.grid(axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
