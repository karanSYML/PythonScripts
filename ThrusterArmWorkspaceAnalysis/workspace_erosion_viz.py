#!/usr/bin/env python3
"""
workspace_erosion_viz.py
========================
3D workspace visualization for the thruster arm, colored by plume-erosion
proxy at each reachable end-effector (nozzle) position.

For every collision-free arm pose on the joint-space grid:
  1. FK → nozzle position in LAR frame
  2. Thrust direction → composite CoG  (same convention as geometry_visualizer.py)
  3. Plume direction  = −thrust direction
  4. Erosion proxy   = Σ_{panel points} cos^n(θ_i) / r_i²
                       (integrated relative ion flux over both solar panel wings)

Color scale (plasma colormap, log-scaled):
  Dark purple = low erosion risk  →  Bright yellow = high erosion risk

Secondary panel: XY top-down view for azimuthal workspace coverage.

Usage
-----
    python workspace_erosion_viz.py           # interactive: matplotlib window + plotly browser tab
    python workspace_erosion_viz.py --save    # save PNG → workspace_erosion.png
                                              #      HTML → workspace_erosion.html
"""

import sys
import os
import json

import numpy as np

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plume_impingement_pipeline import (
    RoboticArmGeometry, StackConfig, ThrusterParams, arm_has_collision,
    ErosionEstimator, MaterialParams,
)
from feasibility_cells import build_joint_grid, compute_static_cell_quantities, compute_F_kin
from feasibility_map import compute_pivot

import sys as _sys
_SPUTTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sputter_erosion")
if _SPUTTER_PATH not in _sys.path:
    _sys.path.insert(0, _SPUTTER_PATH)

from sputter_erosion import (
    Vector3,
    ThrusterPlacement  as _SE_ThrusterPlacement,
    SolarArray         as _SE_SolarArray,
    Interconnect       as _SE_Interconnect,
    SatelliteGeometry  as _SE_SatelliteGeometry,
    SheathModel        as _SE_SheathModel,
    HallThrusterPlume  as _SE_HallThrusterPlume,
    SpeciesFractions   as _SE_SpeciesFractions,
    EcksteinPreuss     as _SE_EcksteinPreuss,
    EcksteinAngular    as _SE_EcksteinAngular,
    ErosionIntegrator  as _SE_ErosionIntegrator,
    GEOEnvironment     as _SE_GEOEnvironment,
    ThermalCycling     as _SE_ThermalCycling,
    LifetimeAnalysis   as _SE_LifetimeAnalysis,
    MissionProfile     as _SE_MissionProfile,
    FiringPhase        as _SE_FiringPhase,
)
from sputter_erosion.yields import FullYield as _SE_FullYield


# ---------------------------------------------------------------------------
# Scene configuration (mirrors geometry_visualizer.py defaults)
# ---------------------------------------------------------------------------

ARM   = RoboticArmGeometry()
STACK = StackConfig(
    servicer_mass=744.0,
    servicer_bus_x=0.9, servicer_bus_y=1.5, servicer_bus_z=0.8,
    client_mass=2800.0,
    client_bus_x=2.3, client_bus_y=3.0, client_bus_z=5.0,
    panel_span_one_side=16.0, panel_width=2.5,
    lar_offset_z=0.05,
)
THRUSTER          = ThrusterParams()
SERVICER_YAW_DEG  = -25.0
GRID_RESOLUTION   = (40, 40, 40)   # ~64k cells; ~20k after F_kin
PANEL_N_SPAN      = 25             # longitudinal sample density per wing
PANEL_N_CHORD     = 8             # chordwise (Y) sample density

# Per-species cosine-beam exponents (Xe+, Xe2+, Xe3+).
# Higher charge-state ions have modestly broader angular distributions.
_SPECIES_N_EXPS = (10.0, 8.0, 6.0)

# Charge-exchange (CEX) ion fraction of the total beam current.
# CEX ions scatter near-isotropically at low energy and contribute a
# direction-independent 1/r² floor to the panel flux.
_CEX_FRACTION = 0.10

# High-fidelity evaluation parameters
HF_TOP_K    = 20        # worst-proxy poses evaluated with full IEDF integrator
HF_FIRING_S = 3600.0   # 1-hour snapshot for ranking

# GEO environment scenarios used in the HF substorm comparison
_GEO_QUIESCENT       = _SE_GEOEnvironment()
_GEO_SUBSTORM        = _SE_GEOEnvironment(in_substorm=True)
_HF_THERMAL          = _SE_ThermalCycling()   # 90 cycles/yr, ΔT=120 K, Coffin-Manson exp=2
_HF_MISSION_YEARS    = 7.0                    # nominal GEO NSSK lifetime for CLF scaling
_HF_FIRINGS_PER_DAY  = 1.0                    # daily NSSK burns


def _load_cfg() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(here, "feasibility_inputs.json")
    try:
        with open(fpath) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Solar panel sample grid
# ---------------------------------------------------------------------------

def _panel_grid(stack: StackConfig, tracking_deg: float = 0.0):
    """Return (panel_pts, n_inc) for the solar panels in LAR frame.

    panel_pts : (N_pts, 3) — sample point positions
    n_inc     : (3,)       — inward surface normal of the irradiated face,
                             i.e. pointing toward incoming ions.

    At tracking_deg=0 the panels are horizontal at z = client_bus_z/2 and
    ions from the servicer (z < z_hinge) hit the bottom face, so n_inc = [0,0,1].
    For a tracking rotation φ around the X-aligned hinge, n_inc is rotated by
    Rx(φ): n_inc = [0, −sin(φ), cos(φ)].
    """
    track   = np.radians(tracking_deg)
    z_hinge = stack.client_bus_z / 2.0          # panel hinge height in LAR
    hw      = stack.panel_width / 2.0
    n_inc   = np.array([0.0, -np.sin(track), np.cos(track)])
    pts     = []
    for side in [+1, -1]:
        x_hinge = side * stack.client_bus_x / 2.0
        for i in range(PANEL_N_SPAN):
            xi = x_hinge + side * ((i + 0.5) / PANEL_N_SPAN) * stack.panel_span_one_side
            for j in range(PANEL_N_CHORD):
                # fractional position along chord: -0.5 → +0.5
                frac = (j + 0.5) / PANEL_N_CHORD - 0.5
                yi   = frac * stack.panel_width
                # When panel tracks (rotates about X-aligned hinge), Y → cos·Y, Z → sin·Y
                pts.append([xi,
                             yi * np.cos(track),
                             z_hinge + yi * np.sin(track)])
    return np.array(pts), n_inc


# ---------------------------------------------------------------------------
# Client bus AABB — used for line-of-sight occlusion
# ---------------------------------------------------------------------------

def _client_aabb(stack: StackConfig):
    """Return (aabb_min, aabb_max) for the client bus in LAR frame, each (3,).

    Client bus is centred at [0, 0, client_bus_z/2] with full dimensions
    [client_bus_x, client_bus_y, client_bus_z].
    """
    hx = stack.client_bus_x / 2.0
    hy = stack.client_bus_y / 2.0
    return (np.array([-hx, -hy, 0.0]),
            np.array([ hx,  hy, stack.client_bus_z]))


# ---------------------------------------------------------------------------
# High-fidelity single-pose evaluator (IEDF-integrated ErosionIntegrator)
# ---------------------------------------------------------------------------

_HF_YIELD_MODEL = _SE_FullYield(
    energy_model=_SE_EcksteinPreuss(),
    angular_model=_SE_EcksteinAngular(),
    subthreshold_floor=0.0,
)
_HF_INTEGRATOR = _SE_ErosionIntegrator(
    yield_model=_HF_YIELD_MODEL,
    include_xe2=True,
    include_xe3=True,
    apply_sheath=True,
)


def _make_hall_plume() -> _SE_HallThrusterPlume:
    """Build the HallThrusterPlume model from the module-level THRUSTER params."""
    estimator = ErosionEstimator(THRUSTER, MaterialParams(name="Ag"))
    return _SE_HallThrusterPlume(
        V_d=THRUSTER.discharge_voltage,
        I_beam=estimator.beam_current_A(),
        mdot_neutral=THRUSTER.mass_flow_rate,
        half_angle_90=np.deg2rad(THRUSTER.beam_divergence_half_angle),
        cex_wing_amp=0.025,
        cex_wing_width=np.deg2rad(40.0),
        species=_SE_SpeciesFractions(
            THRUSTER.xe1_fraction,
            THRUSTER.xe2_fraction,
            THRUSTER.xe3_fraction,
        ),
        sheath_potential=THRUSTER.sheath_potential_V,
    )


def _sheath_from_geo_env(
    geo_env: _SE_GEOEnvironment,
    string_voltage: float = 100.0,
    Te_local: float = 2.0,
) -> _SE_SheathModel:
    """Build SheathModel from a GEOEnvironment.

    Substorm: whole spacecraft at -8 kV.
    Quiescent: local near-thruster sheath set by Te_local (not ambient GEO Te).
    """
    V_f = geo_env.floating_potential() if geo_env.in_substorm else -3.5 * Te_local
    return _SE_SheathModel(string_voltage=string_voltage,
                           floating_potential=V_f, Te_local=Te_local)


def _build_hifi_sat_geo(
    p_nozzle:   np.ndarray,
    plume_dir:  np.ndarray,
    panel_pts:  np.ndarray,
    hall_plume: _SE_HallThrusterPlume,
    stack:      StackConfig,
    geo_env:    _SE_GEOEnvironment | None = None,
) -> _SE_SatelliteGeometry:
    """Assemble a SatelliteGeometry for one arm pose HF evaluation."""
    thruster = _SE_ThrusterPlacement(
        position_body=Vector3(*p_nozzle),
        fire_direction_body=Vector3(*plume_dir),
        plume=hall_plume,
    )

    panel_norm = np.array([0.0, 0.0, -1.0])
    x_panel    = np.array([1.0, 0.0,  0.0])
    y_panel    = np.cross(panel_norm, x_panel)
    y_panel   /= np.linalg.norm(y_panel)

    origin = panel_pts[0]
    n_pts  = len(panel_pts)
    interconnects = []
    for i, pt in enumerate(panel_pts):
        delta = pt - origin
        px = float(np.dot(delta, x_panel))
        py = float(np.dot(delta, y_panel))

        to_thr    = p_nozzle - pt
        to_thr_ip = to_thr - np.dot(to_thr, panel_norm) * panel_norm
        ip_norm   = np.linalg.norm(to_thr_ip)
        edge_dir  = to_thr_ip / ip_norm if ip_norm > 1e-6 else x_panel
        interconnects.append(_SE_Interconnect(
            position_local=(px, py),
            exposed_face_normal=Vector3(
                float(np.dot(edge_dir, x_panel)),
                float(np.dot(edge_dir, y_panel)),
                float(np.dot(edge_dir, panel_norm)),
            ),
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
    if geo_env is not None:
        sheath = _sheath_from_geo_env(geo_env)
    else:
        sheath = _SE_SheathModel(string_voltage=100.0,
                                 floating_potential=-15.0, Te_local=2.0)
    return _SE_SatelliteGeometry(
        thrusters=[thruster],
        solar_arrays=[solar_array],
        sheath=sheath,
    )


def _hifi_for_pose(
    p_nozzle:   np.ndarray,
    plume_dir:  np.ndarray,
    panel_pts:  np.ndarray,
    hall_plume: _SE_HallThrusterPlume,
    stack:      StackConfig,
    geo_env:    _SE_GEOEnvironment | None = None,
) -> float:
    """Return max interconnect thinning [nm] for a 1-hour firing snapshot.

    geo_env=None uses the legacy quiescent sheath (-15 V).
    Pass _GEO_QUIESCENT or _GEO_SUBSTORM to use GEO-physics-derived potentials.
    """
    sat_geo = _build_hifi_sat_geo(p_nozzle, plume_dir, panel_pts,
                                   hall_plume, stack, geo_env)
    results = _HF_INTEGRATOR.evaluate(sat_geo, HF_FIRING_S)
    if not results:
        return 0.0
    return float(max(r.total_thinning_m for r in results)) * 1e9  # nm


def _hifi_coupled_life_factor(
    p_nozzle:   np.ndarray,
    plume_dir:  np.ndarray,
    panel_pts:  np.ndarray,
    hall_plume: _SE_HallThrusterPlume,
    stack:      StackConfig,
) -> float:
    """Coupled life factor (0–1) for quiescent GEO over nominal mission.

    Builds a MissionProfile scaled to _HF_MISSION_YEARS × _HF_FIRINGS_PER_DAY
    daily NSSK burns of HF_FIRING_S each, then runs LifetimeAnalysis with
    _HF_THERMAL (Coffin-Manson fatigue) to get the worst-case interconnect CLF.
    CLF = fraction_remaining × thermal_life_factor; 1.0 = no degradation.
    """
    sat_geo = _build_hifi_sat_geo(p_nozzle, plume_dir, panel_pts,
                                   hall_plume, stack, _GEO_QUIESCENT)
    total_s = HF_FIRING_S * _HF_FIRINGS_PER_DAY * 365.25 * _HF_MISSION_YEARS
    profile = _SE_MissionProfile([_SE_FiringPhase("mission", sat_geo, total_s)])
    life = _SE_LifetimeAnalysis(_HF_INTEGRATOR, _HF_THERMAL).life_prediction(profile)
    if not life:
        return 1.0
    return float(min(v["coupled_life_factor"] for v in life.values()))


# ---------------------------------------------------------------------------
# Vectorized erosion proxy
# ---------------------------------------------------------------------------

def _erosion_proxy(
    p_nozzle:     np.ndarray,   # (N, 3) — nozzle positions for valid cells
    plume_dir:    np.ndarray,   # (N, 3) — unit plume direction per cell
    panel_pts:    np.ndarray,   # (M, 3) — panel sample points
    n_exp:        float,
    *,
    panel_normal: np.ndarray | None = None,  # (3,) — inward surface normal
    los_aabb=None,                           # None | (aabb_min, aabb_max)
    n_exps:       tuple | None = None,       # per-species exponents (n1, n2, n3)
    xe_fractions: tuple | None = None,       # per-species fractions  (f1, f2, f3)
    cex_coeff:    float = 0.0,               # isotropic CEX fraction added as c/r²
    chunk: int = 2048,
) -> np.ndarray:
    """Integrated relative plume flux on solar panels for each arm pose.

    φ(r, θ) ∝ [Σ_k f_k · cos^n_k(θ)] · cos(α_inc) / r²
             + cex_coeff / r²          (isotropic CEX wing, no angle dependence)
      θ       = off-axis angle from the plume beam axis
      α_inc   = angle of incidence on the panel surface (from surface normal)
      k       = ion species (Xe+, Xe2+, Xe3+)
    proxy[i] = Σ_j φ(r_{ij}, θ_{ij})   shape (N,)

    n_exps / xe_fractions: when both supplied, computes a species-weighted
        beam pattern instead of the single cos^n_exp term.
    panel_normal (3,): inward normal of the irradiated panel face (pointing
        toward incoming ions). When None, the incidence factor is omitted.
    los_aabb: when supplied, nozzle→panel segments passing through the AABB
        contribute zero flux (slab-method ray-AABB test, applied to both
        directed beam and CEX term).
    Processing is chunked to keep peak scratch-array memory ≲50 MB.
    """
    N = p_nozzle.shape[0]
    result = np.zeros(N, dtype=np.float64)
    n_blocked = 0
    multi_species = (n_exps is not None) and (xe_fractions is not None)

    if los_aabb is not None:
        aabb_min = np.asarray(los_aabb[0], dtype=np.float64)  # (3,)
        aabb_max = np.asarray(los_aabb[1], dtype=np.float64)

    for s in range(0, N, chunk):
        e  = min(s + chunk, N)
        pn = p_nozzle[s:e]                                           # (n, 3)
        pd = plume_dir[s:e]                                          # (n, 3)

        dv      = panel_pts[np.newaxis, :, :] - pn[:, np.newaxis, :]   # (n, M, 3)
        dist    = np.linalg.norm(dv, axis=-1)                           # (n, M)
        dist    = np.where(dist < 0.02, 0.02, dist)
        unit_dv = dv / dist[..., np.newaxis]                            # (n, M, 3)

        cos_th = np.einsum('nmi,ni->nm', unit_dv, pd)
        cos_th = np.clip(cos_th, 0.0, 1.0)                          # (n, M)

        # ── Species-weighted beam pattern ──────────────────────────────────
        if multi_species:
            beam = sum(f * (cos_th ** n) for f, n in zip(xe_fractions, n_exps))
        else:
            beam = cos_th ** n_exp
        flux = beam / dist ** 2                                      # (n, M)

        # ── Panel surface incidence angle ──────────────────────────────────
        if panel_normal is not None:
            # cos(α_inc) = dot(unit_ion_direction, n_inc)
            cos_inc = np.clip(np.einsum('nmi,i->nm', unit_dv, panel_normal),
                              0.0, 1.0)
            flux *= cos_inc

        # ── CEX isotropic wing ─────────────────────────────────────────────
        if cex_coeff > 0.0:
            flux += cex_coeff / dist ** 2

        # ── Line-of-sight occlusion (slab method) ─────────────────────────
        if los_aabb is not None:
            o     = pn[:, np.newaxis, :]                             # (n, 1, 3)
            EPS_D = 1e-10
            dv_s  = np.where(np.abs(dv) > EPS_D, dv,
                             EPS_D * np.sign(dv + EPS_D))
            t0    = (aabb_min - o) / dv_s                           # (n, M, 3)
            t1    = (aabb_max - o) / dv_s
            t_en  = np.max(np.minimum(t0, t1), axis=-1)             # (n, M)
            t_ex  = np.min(np.maximum(t0, t1), axis=-1)
            blocked = (t_en + 1e-6 < t_ex) & (t_ex > 1e-6) & (t_en < 1.0 - 1e-6)
            n_blocked += int(blocked.sum())
            flux[blocked] = 0.0

        result[s:e] = flux.sum(axis=-1)

    if los_aabb is not None:
        M   = panel_pts.shape[0]
        pct = 100.0 * n_blocked / (N * M)
        print(f"  LOS occlusion: {n_blocked:,} / {N * M:,} segments blocked ({pct:.1f}%)")

    return result


# ---------------------------------------------------------------------------
# Drawing helpers (minimal subset from geometry_visualizer.py)
# ---------------------------------------------------------------------------

def _Rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


_BOX_EDGES    = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
_BOX_FACE_IDX = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]


def _box_verts(ctr, dims):
    cx, cy, cz = ctr
    hx, hy, hz = dims[0]/2, dims[1]/2, dims[2]/2
    return np.array([[cx-hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz-hz],
                     [cx+hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz-hz],
                     [cx-hx,cy-hy,cz+hz],[cx+hx,cy-hy,cz+hz],
                     [cx+hx,cy+hy,cz+hz],[cx-hx,cy+hy,cz+hz]])


def _draw_box(ax, ctr, dims, ec, fc=None, fa=0.08, ea=0.40, lw=1.2, Rz=None):
    v = _box_verts(ctr, dims)
    if Rz is not None:
        o = np.array(ctr)
        v = (Rz @ (v - o).T).T + o
    if fc:
        ax.add_collection3d(Poly3DCollection(
            [[v[i] for i in f] for f in _BOX_FACE_IDX],
            alpha=fa, facecolor=fc, edgecolor="none"))
    for i, j in _BOX_EDGES:
        ax.plot3D(*zip(v[i], v[j]), color=ec, alpha=ea, lw=lw)


def _draw_context(ax, stack: StackConfig, pivot: np.ndarray,
                  cog: np.ndarray, panel_pts: np.ndarray):
    """Draw scene context: buses, LAR ring, panel outline, pivot, CoG."""
    Rz       = _Rz(SERVICER_YAW_DEG)
    serv_o   = stack.servicer_origin_in_lar_frame()

    # Client bus (light grey wireframe)
    _draw_box(ax, [0, 0, stack.client_bus_z/2],
              [stack.client_bus_x, stack.client_bus_y, stack.client_bus_z],
              ec="#5D6D7E", fc="#BDC3C7", fa=0.07, ea=0.28, lw=1.0)

    # Servicer bus (blue wireframe, yawed)
    _draw_box(ax, serv_o,
              [stack.servicer_bus_x, stack.servicer_bus_y, stack.servicer_bus_z],
              ec="#2471A3", fc="#5DADE2", fa=0.10, ea=0.30, lw=1.0, Rz=Rz)

    # Solar panel faces (translucent gold outlines)
    z_h = stack.client_bus_z / 2.0
    hw  = stack.panel_width / 2.0
    for side in [+1, -1]:
        x0 = side * stack.client_bus_x / 2.0
        x1 = x0 + side * stack.panel_span_one_side
        quad = np.array([[x0,-hw,z_h],[x1,-hw,z_h],[x1,hw,z_h],[x0,hw,z_h]])
        ax.add_collection3d(Poly3DCollection(
            [quad.tolist()], alpha=0.12,
            facecolor="#F39C12", edgecolor="#935116", lw=0.8))

    # Panel sample points (tiny dots for reference)
    ax.scatter(panel_pts[:,0], panel_pts[:,1], panel_pts[:,2],
               s=1.2, c="#935116", alpha=0.25, zorder=2, label="Panel sample pts")

    # LAR ring
    th = np.linspace(0, 2*np.pi, 64)
    ax.plot3D(0.6*np.cos(th), 0.6*np.sin(th), np.zeros_like(th),
              color="#17A589", lw=1.8, alpha=0.80, zorder=5)

    # Pivot (black square) and CoG (red diamond)
    ax.scatter(*pivot, s=130, c="#1A252F", marker="s",
               edgecolors="w", lw=1.2, zorder=10, label="Arm pivot")
    ax.scatter(*cog, s=180, c="#C0392B", marker="D",
               edgecolors="w", lw=1.3, zorder=10, label="Stack CoG")


def _draw_arm_at_pose(ax, pivot, p_elbow, p_wrist, p_nozzle, color, label=""):
    """Draw arm links + joint markers for a reference pose."""
    kw = dict(lw=3.5, solid_capstyle="round")
    ax.plot3D(*zip(pivot,    p_elbow),  color=color, **kw)
    ax.plot3D(*zip(p_elbow,  p_wrist),  color=color, **kw)
    ax.plot3D(*zip(p_wrist,  p_nozzle), color=color, lw=2.2,
              linestyle="--", solid_capstyle="round")
    ax.scatter(*p_nozzle, s=220, c=color, marker="*",
               edgecolors="k", lw=0.8, zorder=12, label=label)


# ---------------------------------------------------------------------------
# Top-down XY projection helper
# ---------------------------------------------------------------------------

def _xy_projection(ax2d, p_valid, erosion_log, vmin, vmax, pivot, cog, stack):
    """2D top-down (XY) view of workspace footprint colored by erosion."""
    sc = ax2d.scatter(p_valid[:,0], p_valid[:,1],
                      c=erosion_log, cmap="plasma", vmin=vmin, vmax=vmax,
                      s=4, alpha=0.55, rasterized=True)
    ax2d.scatter(*pivot[:2],  s=120, c="#1A252F", marker="s",
                 edgecolors="w", lw=1.0, zorder=8, label="Pivot")
    ax2d.scatter(*cog[:2],    s=140, c="#C0392B", marker="D",
                 edgecolors="w", lw=1.0, zorder=8, label="CoG")

    # Client bus footprint
    bx, by = stack.client_bus_x/2, stack.client_bus_y/2
    rect_x = [-bx, bx, bx, -bx, -bx]
    rect_y = [-by,-by, by,  by, -by]
    ax2d.fill(rect_x, rect_y, alpha=0.12, facecolor="#BDC3C7", edgecolor="#5D6D7E", lw=1.0)

    # Panel spans
    for side in [+1, -1]:
        x0 = side * bx
        x1 = x0 + side * stack.panel_span_one_side
        hw = stack.panel_width / 2
        ax2d.fill([x0,x1,x1,x0], [-hw,-hw,hw,hw],
                  alpha=0.12, facecolor="#F39C12", edgecolor="#935116", lw=0.7)

    ax2d.set_xlabel("X (North) [m]", fontsize=8)
    ax2d.set_ylabel("Y (East)  [m]", fontsize=8)
    ax2d.set_title("Top-down (XY) projection  [workspace ±4 m]",
                   fontsize=9, fontweight="bold")
    # Zoom to workspace region — panels extend to ±18 m but workspace is ±4 m
    ax2d.set_xlim(-4, 4)
    ax2d.set_ylim(-4, 4)
    ax2d.set_aspect("equal")
    ax2d.legend(fontsize=7, loc="upper right", framealpha=0.85)
    ax2d.grid(True, alpha=0.25, lw=0.5)
    return sc


# ---------------------------------------------------------------------------
# Plotly interactive figure
# ---------------------------------------------------------------------------

def _box_edges_plotly(ctr, dims, Rz=None):
    """Return (xs, ys, zs) with None separators for Scatter3d line mode (box wireframe)."""
    v = _box_verts(ctr, dims)
    if Rz is not None:
        o = np.array(ctr, dtype=float)
        v = (Rz @ (v - o).T).T + o
    xs, ys, zs = [], [], []
    for i, j in _BOX_EDGES:
        xs += [v[i, 0], v[j, 0], None]
        ys += [v[i, 1], v[j, 1], None]
        zs += [v[i, 2], v[j, 2], None]
    return xs, ys, zs


def _arm_trace_plotly(pivot, p_elbow, p_wrist, p_nozzle,
                       color: str, name: str, dash: bool = False):
    """Return a Scatter3d trace for one arm configuration."""
    pts = np.array([pivot, p_elbow, p_wrist, p_nozzle])
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="lines+markers",
        line=dict(color=color, width=6,
                  dash="dash" if dash else "solid"),
        marker=dict(size=[5, 5, 5, 10],
                    color=[color, color, color, color],
                    symbol=["circle", "circle", "circle", "diamond"],
                    line=dict(color="black", width=1)),
        name=name,
        legendgroup=name,
    )


def build_plotly_figure(
    p_valid:    np.ndarray,    # (N, 3) nozzle positions
    erosion:    np.ndarray,    # (N,)   raw erosion proxy
    log_erosion: np.ndarray,   # (N,)   log10(erosion)
    q0_valid:   np.ndarray,    # (N,)   joint angles [rad]
    q1_valid:   np.ndarray,
    q2_valid:   np.ndarray,
    pivot:      np.ndarray,
    cog:        np.ndarray,
    panel_pts:  np.ndarray,
    stack:      StackConfig,
    idx_min:    int,
    idx_max:    int,
    N_total:    int,
    hf_idx:     np.ndarray | None = None,   # (K,) indices into p_valid
    hf_nm:      np.ndarray | None = None,   # (K,) HF thinning [nm]
) -> go.Figure:
    """Build and return the interactive Plotly figure.

    Two sub-plots (side by side):
      Left  — full 3D workspace scatter + geometry context
      Right — XY top-down projection (2D scatter on a Scatter3d with z=0 plane)
    Both are in the same scene for a consistent color axis.
    """
    Rz = _Rz(SERVICER_YAW_DEG)
    serv_o = stack.servicer_origin_in_lar_frame()

    # Hover text: one entry per workspace point
    hover = [
        f"EE  ({p_valid[i,0]:+.3f}, {p_valid[i,1]:+.3f}, {p_valid[i,2]:+.3f}) m<br>"
        f"q0={np.degrees(q0_valid[i]):.1f}°  "
        f"q1={np.degrees(q1_valid[i]):.1f}°  "
        f"q2={np.degrees(q2_valid[i]):.1f}°<br>"
        f"Erosion proxy: {erosion[i]:.3e}<br>"
        f"log₁₀: {log_erosion[i]:.2f}"
        for i in range(len(p_valid))
    ]

    traces = []

    # ── Workspace scatter (main) ───────────────────────────────────────────
    traces.append(go.Scatter3d(
        x=p_valid[:, 0], y=p_valid[:, 1], z=p_valid[:, 2],
        mode="markers",
        marker=dict(
            size=3,
            color=log_erosion,
            colorscale="Plasma",
            colorbar=dict(
                title=dict(text="log₁₀(Σ plume flux)<br>← low   high →",
                           font=dict(size=11)),
                thickness=16, len=0.75, x=1.02,
                tickformat=".1f",
            ),
            opacity=0.75,
            showscale=True,
        ),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name=f"Workspace ({len(p_valid):,} poses)",
        legendgroup="ws",
    ))

    # ── Client bus wireframe ───────────────────────────────────────────────
    xs, ys, zs = _box_edges_plotly(
        [0, 0, stack.client_bus_z / 2],
        [stack.client_bus_x, stack.client_bus_y, stack.client_bus_z])
    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color="#5D6D7E", width=2),
        opacity=0.45, name="Client bus",
        hoverinfo="skip", legendgroup="geom",
    ))

    # ── Servicer bus wireframe (yawed) ────────────────────────────────────
    xs, ys, zs = _box_edges_plotly(
        serv_o,
        [stack.servicer_bus_x, stack.servicer_bus_y, stack.servicer_bus_z],
        Rz=Rz)
    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color="#2471A3", width=2),
        opacity=0.45, name="Servicer bus",
        hoverinfo="skip", legendgroup="geom",
    ))

    # ── Solar panel outlines (flat quads) ────────────────────────────────
    z_h = stack.client_bus_z / 2.0
    hw  = stack.panel_width / 2.0
    for side in [+1, -1]:
        x0 = side * stack.client_bus_x / 2.0
        x1 = x0 + side * stack.panel_span_one_side
        corners = np.array([[x0,-hw,z_h],[x1,-hw,z_h],
                             [x1, hw,z_h],[x0, hw,z_h],[x0,-hw,z_h]])
        traces.append(go.Scatter3d(
            x=corners[:,0], y=corners[:,1], z=corners[:,2],
            mode="lines",
            line=dict(color="#E67E22", width=2),
            opacity=0.50,
            name="Solar panels" if side == 1 else None,
            showlegend=(side == 1),
            hoverinfo="skip", legendgroup="geom",
        ))

    # ── Panel sample points ────────────────────────────────────────────────
    traces.append(go.Scatter3d(
        x=panel_pts[:,0], y=panel_pts[:,1], z=panel_pts[:,2],
        mode="markers",
        marker=dict(size=1.5, color="#935116", opacity=0.30),
        name="Panel sample pts",
        hoverinfo="skip", legendgroup="geom",
    ))

    # ── LAR ring ──────────────────────────────────────────────────────────
    th   = np.linspace(0, 2 * np.pi, 64)
    r_lar = 0.6
    traces.append(go.Scatter3d(
        x=r_lar * np.cos(th), y=r_lar * np.sin(th), z=np.zeros(64),
        mode="lines", line=dict(color="#17A589", width=3),
        opacity=0.80, name="LAR interface",
        hoverinfo="skip", legendgroup="geom",
    ))

    # ── Pivot & CoG markers ────────────────────────────────────────────────
    traces.append(go.Scatter3d(
        x=[pivot[0]], y=[pivot[1]], z=[pivot[2]],
        mode="markers",
        marker=dict(size=8, color="#1A252F", symbol="square",
                    line=dict(color="white", width=1)),
        name="Arm pivot", legendgroup="markers",
    ))
    traces.append(go.Scatter3d(
        x=[cog[0]], y=[cog[1]], z=[cog[2]],
        mode="markers",
        marker=dict(size=9, color="#C0392B", symbol="diamond",
                    line=dict(color="white", width=1)),
        name="Stack CoG", legendgroup="markers",
    ))

    # ── Min / Max erosion arm configurations ─────────────────────────────
    for idx, color, name_tag in [(idx_min, "#00E5FF", "Min-erosion pose"),
                                  (idx_max, "#FFD700", "Max-erosion pose")]:
        pe, pw, pn = ARM.forward_kinematics(
            pivot, q0_valid[idx], q1_valid[idx], q2_valid[idx],
            servicer_yaw_deg=SERVICER_YAW_DEG)
        traces.append(_arm_trace_plotly(pivot, pe, pw, pn, color, name_tag))

    # ── HF top-K overlay ──────────────────────────────────────────────────
    if hf_idx is not None and hf_nm is not None:
        hf_pts = p_valid[hf_idx]
        hf_hover = [
            f"HF thinning: {hf_nm[j]:.3f} nm (1 hr)<br>"
            f"proxy: {erosion[hf_idx[j]]:.3e}<br>"
            f"EE ({hf_pts[j,0]:+.3f}, {hf_pts[j,1]:+.3f}, {hf_pts[j,2]:+.3f}) m"
            for j in range(len(hf_idx))
        ]
        traces.append(go.Scatter3d(
            x=hf_pts[:, 0], y=hf_pts[:, 1], z=hf_pts[:, 2],
            mode="markers",
            marker=dict(
                size=7, symbol="diamond",
                color=hf_nm, colorscale="Hot",
                colorbar=dict(title=dict(text="HF thinning [nm]", font=dict(size=10)),
                              thickness=12, len=0.45, x=1.10),
                line=dict(color="black", width=1),
                showscale=True,
            ),
            text=hf_hover,
            hovertemplate="%{text}<extra></extra>",
            name=f"HF top-{len(hf_idx)} [nm]",
            legendgroup="hf",
        ))

    # ── Assemble figure ────────────────────────────────────────────────────
    n_valid = len(p_valid)
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(
                "Thruster Arm Workspace — Plume Erosion Proxy per EE Position<br>"
                f"<sup>Thrust: nozzle → CoG  |  "
                f"{n_valid:,} collision-free poses from {N_total:,}-cell grid  |  "
                f"Hover for joint angles & erosion value</sup>"
            ),
            x=0.5, font=dict(size=15),
        ),
        scene=dict(
            xaxis=dict(title="X — North [m]", range=[-3, 3]),
            yaxis=dict(title="Y — East [m]",  range=[-3, 3]),
            zaxis=dict(title="Z — nadir [m]", range=[-3, 5.5]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.4),
            camera=dict(eye=dict(x=-1.4, y=-1.6, z=0.8)),
        ),
        legend=dict(
            x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#BDC3C7", borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=0, r=120, t=80, b=0),
        paper_bgcolor="#F8F9FA",
        height=750,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Workspace Erosion Visualizer")
    print("=" * 60)

    cfg      = _load_cfg()
    n_hat_ee = np.array(cfg.get("nozzle_exit_direction_ee", ARM.n_hat_body))

    pivot = compute_pivot(ARM, STACK, SERVICER_YAW_DEG)
    cog   = STACK.stack_cog()
    print(f"Pivot : {np.round(pivot, 3)}")
    print(f"CoG   : {np.round(cog, 3)}")
    print(f"Grid  : {GRID_RESOLUTION}  →  {np.prod(GRID_RESOLUTION):,} cells")

    # ── 1. Joint grid + static cell quantities ─────────────────────────────
    print("\nBuilding grid and static cell quantities...")
    q0g, q1g, q2g = build_joint_grid(ARM, GRID_RESOLUTION)
    cq = compute_static_cell_quantities(ARM, pivot, n_hat_ee, q0g, q1g, q2g,
                                        servicer_yaw_deg=SERVICER_YAW_DEG)

    # ── 2. F_kin: collision + joint limit mask ─────────────────────────────
    print("Computing collision mask (F_kin)...")
    F_kin = compute_F_kin(ARM, pivot, STACK, SERVICER_YAW_DEG,
                          cq, q0g, q1g, q2g, verbose=True)

    # ── 3. Extract valid cells ─────────────────────────────────────────────
    p_nozzle_all = cq['p_nozzle']              # (N0,N1,N2,3)
    p_valid      = p_nozzle_all[F_kin]         # (N_valid, 3)
    N_valid      = p_valid.shape[0]
    print(f"Valid cells: {N_valid:,} / {F_kin.size:,}  ({100*N_valid/F_kin.size:.1f}%)")

    # Joint angle grids at valid cells (for arm drawing)
    q0_valid = q0g[F_kin]
    q1_valid = q1g[F_kin]
    q2_valid = q2g[F_kin]

    # ── 4. Plume direction from FK nozzle axis (CR3 @ n_hat_body per cell) ───
    # cq['t_hat'] = -(CR123 @ n_hat_body), already computed for every grid cell
    plume_d = -cq['t_hat'][F_kin]     # (N_valid, 3)  plasma exits along +n_hat

    # ── 5. Solar panel points + erosion proxy ─────────────────────────────
    # ── Solar panel tracking sweep — worst-case over orbit ───────────────────
    # Panels rotate about the X-aligned hinge to track the Sun.  We sample
    # _TRACKING_DEGS and take the element-wise maximum proxy per arm pose,
    # giving a conservative (worst-case) erosion estimate across the orbit.
    _TRACKING_DEGS = np.linspace(-30.0, 30.0, 7)   # ±30° in 6 steps
    aabb    = _client_aabb(STACK)
    xe_fracs = (THRUSTER.xe1_fraction, THRUSTER.xe2_fraction, THRUSTER.xe3_fraction)

    print(f"Computing erosion proxy (LOS + incidence + multi-species, "
          f"tracking sweep {_TRACKING_DEGS[0]:.0f}°→{_TRACKING_DEGS[-1]:.0f}°)...")
    erosion = np.zeros(p_valid.shape[0], dtype=np.float64)
    for t_deg in _TRACKING_DEGS:
        panel_pts, panel_normal = _panel_grid(STACK, tracking_deg=float(t_deg))
        e_t = _erosion_proxy(p_valid, plume_d, panel_pts,
                             n_exp=THRUSTER.plume_cosine_exponent,
                             panel_normal=panel_normal,
                             los_aabb=aabb,
                             n_exps=_SPECIES_N_EXPS,
                             xe_fractions=xe_fracs,
                             cex_coeff=_CEX_FRACTION)
        np.maximum(erosion, e_t, out=erosion)
        print(f"  tracking={t_deg:+.0f}°  max={e_t.max():.3e}  median={np.median(e_t):.3e}")
    # Use nominal tracking=0 panel_pts for HF evaluation and drawing
    panel_pts, _ = _panel_grid(STACK, tracking_deg=0.0)
    print(f"Erosion  min={erosion.min():.3e}  "
          f"max={erosion.max():.3e}  "
          f"median={np.median(erosion):.3e}")

    # ── HF evaluation on top-K worst-proxy poses ───────────────────────────
    top_k_idx = np.argsort(erosion)[::-1][:HF_TOP_K]
    print(f"\nRunning HF integrator on top-{HF_TOP_K} worst-proxy poses "
          f"({HF_FIRING_S/3600:.0f}-hr snapshot)  "
          f"[quiescent | substorm | CLF over {_HF_MISSION_YEARS:.0f} yr]...")
    hall_plume      = _make_hall_plume()
    hf_nm           = np.zeros(len(top_k_idx))   # quiescent thinning [nm]
    hf_nm_substorm  = np.zeros(len(top_k_idx))   # substorm thinning  [nm]
    hf_clf          = np.ones(len(top_k_idx))     # coupled life factor [0–1]
    for j, gi in enumerate(top_k_idx):
        hf_nm[j]          = _hifi_for_pose(
            p_valid[gi], plume_d[gi], panel_pts, hall_plume, STACK, _GEO_QUIESCENT)
        hf_nm_substorm[j] = _hifi_for_pose(
            p_valid[gi], plume_d[gi], panel_pts, hall_plume, STACK, _GEO_SUBSTORM)
        hf_clf[j]         = _hifi_coupled_life_factor(
            p_valid[gi], plume_d[gi], panel_pts, hall_plume, STACK)
        print(f"  [{j+1:2d}/{HF_TOP_K}]  proxy={erosion[gi]:.3e}  "
              f"quiescent={hf_nm[j]:.3f} nm  "
              f"substorm={hf_nm_substorm[j]:.3f} nm  "
              f"CLF={hf_clf[j]:.4f}  "
              f"EE=({p_valid[gi,0]:+.2f},{p_valid[gi,1]:+.2f},{p_valid[gi,2]:+.2f})")
    print(f"  Quiescent HF range : {hf_nm.min():.3f} – {hf_nm.max():.3f} nm")
    print(f"  Substorm  HF range : {hf_nm_substorm.min():.3f} – {hf_nm_substorm.max():.3f} nm")
    print(f"  CLF range          : {hf_clf.min():.4f} – {hf_clf.max():.4f}")

    # Log-scale for perceptual range
    log_erosion = np.log10(np.clip(erosion, 1e-30, None))
    vmin = float(np.percentile(log_erosion, 1))
    vmax = float(np.percentile(log_erosion, 99))

    # Best / worst poses (by erosion)
    idx_min = int(erosion.argmin())
    idx_max = int(erosion.argmax())
    for label, idx in [("Min-erosion", idx_min), ("Max-erosion", idx_max)]:
        pn = p_valid[idx]
        pe, pw, _ = ARM.forward_kinematics(pivot, q0_valid[idx], q1_valid[idx], q2_valid[idx],
                                           servicer_yaw_deg=SERVICER_YAW_DEG)
        print(f"  {label}: EE=({pn[0]:+.2f},{pn[1]:+.2f},{pn[2]:+.2f})  "
              f"q=({np.degrees(q0_valid[idx]):.1f}°,"
              f"{np.degrees(q1_valid[idx]):.1f}°,"
              f"{np.degrees(q2_valid[idx]):.1f}°)  "
              f"proxy={erosion[idx]:.3e}")

    # ── 6. Figure layout ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9), facecolor="#F8F9FA")
    fig.suptitle(
        "Thruster Arm Workspace  —  Plume Erosion Proxy per Reachable EE Position\n"
        f"Plume direction: FK nozzle axis (CR3 @ n̂_body)  |  "
        f"Erosion proxy: Σ cos^n(θ)·cos(α_inc)/r²  |  LOS-occluded  |  {len(panel_pts)} panel pts  |  "
        f"{N_valid:,} collision-free poses",
        fontsize=10, fontweight="bold", y=0.995, color="#1A252F",
    )

    # Axes: 3D scene (left), XY projection (top-right), colorbar + info (far right)
    ax3d  = fig.add_axes([0.01, 0.06, 0.56, 0.90], projection="3d")
    ax2d  = fig.add_axes([0.59, 0.38, 0.26, 0.56])
    ax_cb = fig.add_axes([0.87, 0.12, 0.025, 0.75])

    ax3d.set_facecolor("#EBF5FB")

    # ── 7. Draw static context ─────────────────────────────────────────────
    _draw_context(ax3d, STACK, pivot, cog, panel_pts)

    # ── 8. Workspace scatter (3D) ──────────────────────────────────────────
    ax3d.scatter(
        p_valid[:,0], p_valid[:,1], p_valid[:,2],
        c=log_erosion, cmap="plasma", vmin=vmin, vmax=vmax,
        s=5, alpha=0.65, depthshade=True, zorder=3,
    )

    # Highlight min/max erosion poses + draw the actual arm configuration
    for idx, color, tag in [(idx_min, "#00E5FF", "Min erosion"),
                             (idx_max, "#FFD700", "Max erosion")]:
        pe, pw, pn = ARM.forward_kinematics(
            pivot, q0_valid[idx], q1_valid[idx], q2_valid[idx],
            servicer_yaw_deg=SERVICER_YAW_DEG)
        _draw_arm_at_pose(ax3d, pivot, pe, pw, pn, color, tag)

    # Overlay top-K HF-evaluated poses: diamonds sized/colored by nm thinning
    hf_pts    = p_valid[top_k_idx]             # (K, 3)
    hf_norm   = plt.Normalize(vmin=hf_nm.min(), vmax=hf_nm.max())
    hf_colors = cm.hot(hf_norm(hf_nm))
    ax3d.scatter(
        hf_pts[:, 0], hf_pts[:, 1], hf_pts[:, 2],
        c=hf_nm, cmap="hot", norm=hf_norm,
        s=60, marker="D", edgecolors="k", linewidths=0.5,
        zorder=14, label=f"HF top-{HF_TOP_K} [nm]",
    )

    ax3d.set_xlim(-2.5, 2.5)
    ax3d.set_ylim(-2.5, 2.5)
    ax3d.set_zlim(-3.0, 5.5)
    ax3d.set_xlabel("X (N) [m]", fontsize=8, labelpad=2)
    ax3d.set_ylabel("Y (E) [m]", fontsize=8, labelpad=2)
    ax3d.set_zlabel("Z (nadir) [m]", fontsize=8, labelpad=2)
    ax3d.set_title("3D workspace  —  color = log₁₀(erosion proxy)",
                   fontsize=9, fontweight="bold", pad=5)
    ax3d.view_init(elev=22, azim=-115)
    ax3d.legend(fontsize=7.5, loc="upper left", framealpha=0.88,
                markerscale=0.9, labelspacing=0.4)

    # ── 9. XY top-down projection ──────────────────────────────────────────
    _xy_projection(ax2d, p_valid, log_erosion, vmin, vmax, pivot, cog, STACK)

    # ── 10. Colorbar ───────────────────────────────────────────────────────
    sm = cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap="plasma")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb, orientation="vertical")
    cbar.set_label("log₁₀(Σ plume flux)  →  erosion risk", fontsize=8.5, labelpad=8)
    tick_v = np.linspace(vmin, vmax, 7)
    cbar.set_ticks(tick_v)
    cbar.set_ticklabels([f"{10**v:.1e}" for v in tick_v], fontsize=7)

    # ── 11. Info annotation (below colorbar) ──────────────────────────────
    min_p, max_p = p_valid[idx_min], p_valid[idx_max]
    info = (
        f"Grid : {GRID_RESOLUTION[0]}³ = {np.prod(GRID_RESOLUTION):,}\n"
        f"Valid: {N_valid:,} ({100*N_valid/F_kin.size:.1f}%)\n"
        f"Panel pts: {len(panel_pts)}\n"
        f"n_exp: {THRUSTER.plume_cosine_exponent}\n"
        "\n"
        "── Min erosion ──\n"
        f" EE  ({min_p[0]:+.2f}, {min_p[1]:+.2f}, {min_p[2]:+.2f})\n"
        f" q0={np.degrees(q0_valid[idx_min]):.1f}° "
        f"q1={np.degrees(q1_valid[idx_min]):.1f}° "
        f"q2={np.degrees(q2_valid[idx_min]):.1f}°\n"
        f" proxy = {erosion[idx_min]:.2e}\n"
        "\n"
        "── Max erosion ──\n"
        f" EE  ({max_p[0]:+.2f}, {max_p[1]:+.2f}, {max_p[2]:+.2f})\n"
        f" q0={np.degrees(q0_valid[idx_max]):.1f}° "
        f"q1={np.degrees(q1_valid[idx_max]):.1f}° "
        f"q2={np.degrees(q2_valid[idx_max]):.1f}°\n"
        f" proxy = {erosion[idx_max]:.2e}"
    )
    fig.text(0.875, 0.085, info, fontsize=7, va="bottom", ha="left",
             fontfamily="monospace", color="#2C3E50",
             bbox=dict(facecolor="white", alpha=0.88,
                       edgecolor="#BDC3C7", lw=0.8, pad=4))

    # ── 12. Plotly interactive figure ──────────────────────────────────────
    print("Building Plotly figure...")
    fig_plotly = build_plotly_figure(
        p_valid, erosion, log_erosion,
        q0_valid, q1_valid, q2_valid,
        pivot, cog, panel_pts, STACK,
        idx_min, idx_max, F_kin.size,
        hf_idx=top_k_idx, hf_nm=hf_nm,
    )

    # ── 13. Output ─────────────────────────────────────────────────────────
    if "--save" in sys.argv:
        png_out  = "workspace_erosion.png"
        html_out = "workspace_erosion.html"
        fig.savefig(png_out, dpi=180, bbox_inches="tight", facecolor="#F8F9FA")
        fig_plotly.write_html(html_out, include_plotlyjs="cdn", full_html=True)
        print(f"Saved → {png_out}")
        print(f"Saved → {html_out}")
    else:
        # Show plotly first (opens browser tab, non-blocking)
        fig_plotly.show()
        # Then matplotlib (blocking until window is closed)
        plt.show()


if __name__ == "__main__":
    main()
