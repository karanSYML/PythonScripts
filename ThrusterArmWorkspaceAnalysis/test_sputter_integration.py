#!/usr/bin/env python3
"""
test_sputter_integration.py
===========================

Integration tests for the sputter_erosion high-fidelity path in
PlumePipeline.  Four test sections:

  1. Smoke tests  — library importable, pipelines construct cleanly
  2. Key parity   — hifi result dict contains all expected keys
  3. Sweep comparison — 8-case yaw sweep, analytical vs. high-fidelity
                        side-by-side with ratio column
  4. Monte Carlo  — Bayesian uncertainty on 3 worst HF cases
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Make sure the project root is on the path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plume_impingement_pipeline import (
    PlumePipeline, ThrusterParams, MaterialParams,
    CaseMatrixGenerator, OperationalParams,
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture data
# ──────────────────────────────────────────────────────────────────────────────

THRUSTER = ThrusterParams(
    name="SPT-100-like",
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

MATERIAL = MaterialParams(name="Ag", thickness_um=25.0)

FIXED = {
    "arm_reach_m": 3.0,
    "link_ratio": 0.5,
    "client_mass": 2500.0,
    "servicer_mass": 750.0,
    "panel_span_one_side": 16.0,
    "firing_duration_s": 15000.0,
    "mission_duration_yr": 5.0,
    "firings_per_day": 1.0,
    "panel_tracking_deg": 0.0,
}

YAW_SWEEP = [-30, -15, 0, 15, 30, 45, 60, 90]


def make_cases(yaw_values=None):
    gen = CaseMatrixGenerator()
    if yaw_values is None:
        yaw_values = YAW_SWEEP
    cases = []
    for yaw in yaw_values:
        c = dict(FIXED)
        c["shoulder_yaw_deg"] = float(yaw)
        cases.append(c)
    return cases


# ──────────────────────────────────────────────────────────────────────────────
# Section 1: Smoke tests
# ──────────────────────────────────────────────────────────────────────────────

def test_smoke():
    print("=" * 70)
    print("SECTION 1 — Smoke tests")
    print("=" * 70)

    # 1a: sputter_erosion importable
    sys.path.insert(0, os.path.join(HERE, "sputter_erosion"))
    from sputter_erosion import (
        HallThrusterPlume, ErosionIntegrator, MATERIALS,
    )
    print("  [PASS] sputter_erosion imports OK")
    print(f"         available materials: {sorted(MATERIALS.keys())}")

    # 1b: analytical pipeline constructs
    pipe_a = PlumePipeline(THRUSTER, MATERIAL, erosion_mode="analytical")
    assert pipe_a.erosion_mode == "analytical"
    print("  [PASS] PlumePipeline(analytical) constructed")

    # 1c: high-fidelity pipeline constructs (material alias Ag)
    pipe_h = PlumePipeline(THRUSTER, MATERIAL, erosion_mode="high_fidelity")
    assert pipe_h.erosion_mode == "high_fidelity"
    assert pipe_h._se_material_key == "Ag"
    print("  [PASS] PlumePipeline(high_fidelity) constructed, material_key='Ag'")

    # 1d: alias resolution — Silver_interconnect → Ag
    mat_alias = MaterialParams(name="Silver_interconnect", thickness_um=25.0)
    pipe_alias = PlumePipeline(THRUSTER, mat_alias, erosion_mode="high_fidelity")
    assert pipe_alias._se_material_key == "Ag"
    print("  [PASS] material alias 'Silver_interconnect' → 'Ag' resolved")

    # 1e: bad erosion_mode raises
    try:
        PlumePipeline(THRUSTER, MATERIAL, erosion_mode="magic")
        print("  [FAIL] expected ValueError not raised")
    except ValueError as exc:
        print(f"  [PASS] bad erosion_mode raises ValueError: {exc}")

    # 1f: run a single case to verify no runtime error
    cases1 = make_cases([0])
    r = pipe_h.run_sweep(cases1, verbose=False)
    assert len(r) == 1
    assert "max_erosion_um" in r[0]
    print(f"  [PASS] single-case HF sweep ran OK, "
          f"max_erosion={r[0]['max_erosion_um']:.4f} µm")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Section 2: Key parity
# ──────────────────────────────────────────────────────────────────────────────

def test_key_parity():
    print("=" * 70)
    print("SECTION 2 — Result-dict key parity")
    print("=" * 70)

    pipe_a = PlumePipeline(THRUSTER, MATERIAL, erosion_mode="analytical")
    pipe_h = PlumePipeline(THRUSTER, MATERIAL, erosion_mode="high_fidelity")
    cases1 = make_cases([0])

    r_a = pipe_a.run_sweep(cases1, verbose=False)[0]
    r_h = pipe_h.run_sweep(cases1, verbose=False)[0]

    # Keys that must be in both
    shared_keys = [
        "max_erosion_um", "mean_erosion_um", "erosion_fraction", "status",
        "worst_point_distance_m", "worst_point_offaxis_deg",
        "worst_point_incidence_deg",
        "nssk_deviation_deg", "ewsk_deviation_deg",
        "nssk_torque_Nm", "ewsk_torque_Nm",
        "ant_E1_erosion_um", "ant_W1_erosion_um",
        "cog_x", "cog_y", "cog_z",
    ]
    # Keys that must be exclusively in the HF result
    hifi_only_keys = [
        "hifi_mean_E_eV", "hifi_sheath_boost_eV",
        "hifi_max_fluence_ions_m2", "hifi_worst_incidence_deg",
        "hifi_worst_j_i",
    ]

    all_pass = True
    for k in shared_keys:
        ok = k in r_a and k in r_h
        mark = "[PASS]" if ok else "[FAIL]"
        if not ok:
            all_pass = False
        print(f"  {mark} shared key '{k}': analytic={k in r_a}, hifi={k in r_h}")

    for k in hifi_only_keys:
        ok = k in r_h and k not in r_a
        mark = "[PASS]" if ok else "[FAIL]"
        if not ok:
            all_pass = False
        print(f"  {mark} hifi-only key '{k}': hifi={k in r_h}, not_in_analytic={k not in r_a}")

    if all_pass:
        print("  All key parity checks passed.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Section 3: Yaw-sweep comparison
# ──────────────────────────────────────────────────────────────────────────────

def test_sweep_comparison():
    print("=" * 70)
    print("SECTION 3 — Yaw sweep: analytical vs. high-fidelity comparison")
    print("=" * 70)
    print(f"  Arm reach 3.0 m | Ag, 25 µm | 5-yr mission | 15000 s/day")
    print()

    cases = make_cases(YAW_SWEEP)

    pipe_a = PlumePipeline(THRUSTER, MATERIAL, erosion_mode="analytical")
    pipe_h = PlumePipeline(THRUSTER, MATERIAL, erosion_mode="high_fidelity")

    print("  Running analytical sweep ...", flush=True)
    res_a = pipe_a.run_sweep(cases, verbose=False)
    print("  Running high-fidelity sweep ...", flush=True)
    res_h = pipe_h.run_sweep(cases, verbose=True)

    print()
    hdr = (f"{'Yaw':>6}  {'Analytic':>10}  {'HiFi':>10}  {'HF/Ana':>7}"
           f"  {'mean_E':>8}  {'sheath':>7}  {'j_i':>10}  {'status_A':>8}  {'status_H':>8}")
    print(f"  {hdr}")
    print(f"  {'-'*96}")

    for i, yaw in enumerate(YAW_SWEEP):
        ra = res_a[i]
        rh = res_h[i]
        ratio = rh["max_erosion_um"] / ra["max_erosion_um"] if ra["max_erosion_um"] > 0 else float("nan")
        row = (f"{yaw:>6.0f}  "
               f"{ra['max_erosion_um']:>10.4f}  "
               f"{rh['max_erosion_um']:>10.4f}  "
               f"{ratio:>7.3f}  "
               f"{rh.get('hifi_mean_E_eV', 0):>8.1f}  "
               f"{rh.get('hifi_sheath_boost_eV', 0):>7.1f}  "
               f"{rh.get('hifi_worst_j_i', 0):>10.2e}  "
               f"{ra['status']:>8}  "
               f"{rh['status']:>8}")
        print(f"  {row}")

    print()
    erosions_a = [r["max_erosion_um"] for r in res_a]
    erosions_h = [r["max_erosion_um"] for r in res_h]
    print(f"  Analytical : max={max(erosions_a):.4f} µm  min={min(erosions_a):.4f} µm")
    print(f"  HF model   : max={max(erosions_h):.4f} µm  min={min(erosions_h):.4f} µm")
    print(f"  Median HF/analytic ratio: "
          f"{np.median([h/a if a > 0 else np.nan for h, a in zip(erosions_h, erosions_a)]):.3f}")

    # Rank correlation between the two sets
    from scipy.stats import spearmanr
    rho, pval = spearmanr(erosions_a, erosions_h)
    print(f"  Spearman ρ (rank correlation): {rho:.4f}  p={pval:.4f}")
    print()

    return res_a, res_h  # return for plotting and MC section


def plot_sweep_comparison(res_a, res_h, out_path):
    yaws = np.array(YAW_SWEEP, dtype=float)
    ea = np.array([r["max_erosion_um"] for r in res_a])
    eh = np.array([r["max_erosion_um"] for r in res_h])
    mean_e  = np.array([r.get("hifi_mean_E_eV", 0) for r in res_h])
    sheath  = np.array([r.get("hifi_sheath_boost_eV", 0) for r in res_h])

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle("Analytical vs. High-Fidelity Erosion — Shoulder-Yaw Sweep\n"
                 "SPT-100-like, Ag 25 µm, 5-yr, 15 000 s/day",
                 fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── Top-left: erosion bar chart ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    w = 2.8
    x = np.arange(len(yaws))
    b1 = ax1.bar(x - w/2, ea, w, label="Analytical", color="#4c72b0", alpha=0.85)
    b2 = ax1.bar(x + w/2, eh, w, label="High-fidelity", color="#dd8452", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{y:.0f}°" for y in yaws])
    ax1.set_xlabel("Shoulder yaw angle")
    ax1.set_ylabel("Max erosion [µm]")
    ax1.set_title("EOL Interconnect Erosion")
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")
    for bar, val in zip(b2, eh):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, color="#dd8452")

    # ── Top-right: HF/analytic ratio ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ratio = np.where(ea > 0, eh / ea, np.nan)
    colors = ["#c44e52" if r > 0.1 else "#55a868" for r in ratio]
    ax2.bar(x, ratio, color=colors, alpha=0.85)
    ax2.axhline(1.0, color="k", lw=0.8, ls="--", label="ratio=1 (equal)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{y:.0f}°" for y in yaws])
    ax2.set_xlabel("Shoulder yaw angle")
    ax2.set_ylabel("HF / Analytical ratio")
    ax2.set_title("Model Ratio (HF is lower → conservative analytical)")
    ax2.legend(fontsize=8)

    # ── Bottom-left: mean ion energy + sheath boost ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x - w/2, mean_e,  w, label="Mean E_ion [eV]",    color="#8172b3", alpha=0.85)
    ax3.bar(x + w/2, sheath,  w, label="Sheath boost [eV]", color="#c44e52", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{y:.0f}°" for y in yaws])
    ax3.set_xlabel("Shoulder yaw angle")
    ax3.set_ylabel("Energy [eV]")
    ax3.set_title("Ion Energy at Worst Interconnect")
    ax3.legend(fontsize=8)

    # ── Bottom-right: status classification ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    status_map  = {"SAFE": 0, "CAUTION": 1, "MARGINAL": 2, "FAIL": 3}
    status_cols = {"SAFE": "#55a868", "CAUTION": "#ccb974",
                   "MARGINAL": "#dd8452", "FAIL": "#c44e52"}
    for i, (ra, rh_) in enumerate(zip(res_a, res_h)):
        sa = status_map[ra["status"]]
        sh = status_map[rh_["status"]]
        ax4.scatter(i - 0.15, sa, s=80,
                    color=status_cols[ra["status"]], marker="o", zorder=3,
                    label="Analytical" if i == 0 else "")
        ax4.scatter(i + 0.15, sh, s=80,
                    color=status_cols[rh_["status"]], marker="^", zorder=3,
                    label="High-fidelity" if i == 0 else "")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{y:.0f}°" for y in yaws])
    ax4.set_yticks([0, 1, 2, 3])
    ax4.set_yticklabels(["SAFE", "CAUTION", "MARGINAL", "FAIL"])
    ax4.set_xlabel("Shoulder yaw angle")
    ax4.set_title("Status Classification (○=Analytic  △=HiFi)")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Section 4: Monte Carlo
# ──────────────────────────────────────────────────────────────────────────────

def test_monte_carlo(hf_results=None):
    print("=" * 70)
    print("SECTION 4 — Bayesian Monte Carlo uncertainty (3 worst cases, 200 samples)")
    print("=" * 70)

    pipe_h = PlumePipeline(THRUSTER, MATERIAL, erosion_mode="high_fidelity")

    if hf_results is None:
        cases = make_cases(YAW_SWEEP)
        print("  Re-running HF sweep for MC setup ...", flush=True)
        hf_results = pipe_h.run_sweep(cases, verbose=False)
    else:
        pipe_h.results = hf_results

    # Pick the 3 worst cases by max_erosion_um
    ranked = sorted(range(len(hf_results)),
                    key=lambda i: hf_results[i]["max_erosion_um"],
                    reverse=True)
    top3 = ranked[:3]
    print(f"  Worst 3 case indices by max HF erosion: {top3}")
    print(f"  (yaw degrees: "
          f"{[YAW_SWEEP[i] for i in top3]})")
    print()

    print("  Running MC (200 samples per case) ...", flush=True)
    mc_out = pipe_h.run_monte_carlo(case_indices=top3, n_samples=200, seed=42)

    print()
    print(f"  {'Case':>4}  {'Yaw':>5}  {'p5 µm':>8}  {'p50 µm':>8}  {'p95 µm':>8}  "
          f"{'mean µm':>9}  {'p95/p5':>7}  {'det µm':>9}")
    print(f"  {'-'*70}")
    for ci in top3:
        m = mc_out[ci]
        yaw = YAW_SWEEP[ci]
        det = hf_results[ci]["max_erosion_um"]
        spread = m["p95"] / m["p5"] if m["p5"] > 0 else float("nan")
        print(f"  {ci:>4}  {yaw:>5.0f}  "
              f"{m['p5']:>8.4f}  {m['p50']:>8.4f}  {m['p95']:>8.4f}  "
              f"{m['mean']:>9.4f}  {spread:>7.2f}  {det:>9.4f}")
    print()
    print("  Column legend:")
    print("    p5/p50/p95 = 5th/50th/95th percentile EOL thinning [µm]")
    print("    p95/p5     = uncertainty spread factor")
    print("    det        = deterministic HF (single run) value [µm]")
    print()

    return top3, mc_out  # noqa: consistent return for caller


def plot_monte_carlo(top3, mc_out, hf_results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Bayesian MC Uncertainty — Xe→Ag Yield Parameter Posterior\n"
                 "200 samples per case, worst 3 shoulder-yaw configurations",
                 fontsize=11, fontweight="bold")

    # ── Left: tornado / error-bar chart ─────────────────────────────────────
    ax = axes[0]
    yaw_labels = [f"yaw={YAW_SWEEP[ci]:+.0f}°" for ci in top3]
    x = np.arange(len(top3))
    for i, ci in enumerate(top3):
        m = mc_out[ci]
        p50 = m["p50"]
        err_lo = p50 - m["p5"]
        err_hi = m["p95"] - p50
        ax.barh(i, p50, xerr=[[err_lo], [err_hi]],
                height=0.5, color="#4c72b0", alpha=0.8,
                error_kw=dict(ecolor="#c44e52", capsize=6, lw=1.8))
        ax.scatter(hf_results[ci]["max_erosion_um"], i,
                   marker="|", s=200, color="k", zorder=5,
                   label="Deterministic HF" if i == 0 else "")
        ax.text(m["p95"] + 0.1, i, f"×{m['p95']/m['p5']:.1f}", va="center",
                fontsize=8, color="#c44e52")
    ax.set_yticks(x)
    ax.set_yticklabels(yaw_labels)
    ax.set_xlabel("EOL Interconnect Thinning [µm]")
    ax.set_title("p5 / p50 / p95 Lifetime Thinning\n(×N = p95/p5 spread)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # ── Right: CDF-style fan for worst case ──────────────────────────────────
    ax2 = axes[1]
    worst_ci = top3[0]
    m = mc_out[worst_ci]
    pctiles = np.array([5, 50, 95])
    vals    = np.array([m["p5"], m["p50"], m["p95"]])
    ax2.fill_betweenx([0, 1], [m["p5"], m["p5"]], [m["p95"], m["p95"]],
                      alpha=0.15, color="#4c72b0", label="p5–p95 band")
    ax2.fill_betweenx([0, 1], [m["p5"], m["p5"]], [m["p50"], m["p50"]],
                      alpha=0.25, color="#4c72b0")
    for pct, val, ls in zip(pctiles, vals, [":", "--", ":"]):
        ax2.axvline(val, color="#4c72b0", ls=ls, lw=1.5, label=f"p{pct}={val:.2f} µm")
    ax2.axvline(hf_results[worst_ci]["max_erosion_um"], color="k", lw=1.5,
                label=f"det={hf_results[worst_ci]['max_erosion_um']:.2f} µm")
    ax2.axvline(25.0, color="#c44e52", lw=1.2, ls="--", label="Ag thickness 25 µm")
    ax2.set_xlim(0, max(m["p95"] * 1.3, 30))
    ax2.set_xlabel("EOL Interconnect Thinning [µm]")
    ax2.set_title(f"Worst Case: yaw={YAW_SWEEP[worst_ci]:+.0f}°\n"
                  "Uncertainty Band")
    ax2.set_yticks([])
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR = os.path.join(HERE, "pipeline_runner_output")
    os.makedirs(OUT_DIR, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  sputter_erosion integration test suite                             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    test_smoke()
    test_key_parity()
    res_a, res_h = test_sweep_comparison()
    plot_sweep_comparison(
        res_a, res_h,
        out_path=os.path.join(OUT_DIR, "sputter_sweep_comparison.png"),
    )
    top3, mc_out = test_monte_carlo(res_h)
    plot_monte_carlo(
        top3, mc_out, res_h,
        out_path=os.path.join(OUT_DIR, "sputter_monte_carlo.png"),
    )

    print("All sections completed.")
