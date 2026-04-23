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
)
from feasibility_cells import build_joint_grid, compute_static_cell_quantities, compute_F_kin
from feasibility_map import compute_pivot


# ---------------------------------------------------------------------------
# Scene configuration (mirrors geometry_visualizer.py defaults)
# ---------------------------------------------------------------------------

ARM   = RoboticArmGeometry()
STACK = StackConfig(
    servicer_mass=750.0,
    servicer_bus_x=0.9, servicer_bus_y=1.5, servicer_bus_z=0.8,
    client_mass=2500.0,
    client_bus_x=2.3, client_bus_y=3.0, client_bus_z=5.0,
    panel_span_one_side=16.0, panel_width=2.5,
    lar_offset_z=0.05,
)
THRUSTER          = ThrusterParams()
SERVICER_YAW_DEG  = -25.0
GRID_RESOLUTION   = (40, 40, 40)   # ~64k cells; ~20k after F_kin
PANEL_N_SPAN      = 25             # longitudinal sample density per wing
PANEL_N_CHORD     = 8             # chordwise (Y) sample density


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

def _panel_grid(stack: StackConfig, tracking_deg: float = 0.0) -> np.ndarray:
    """Return panel sample points in LAR frame, shape (N_pts, 3).

    Panel hinge is at the ±X face of the client bus, at the bus mid-height
    (Z = client_bus_z / 2), consistent with draw_panel_faces() in
    geometry_visualizer.py. Each wing extends outward in ±X from the hinge.
    """
    track   = np.radians(tracking_deg)
    z_hinge = stack.client_bus_z / 2.0          # panel hinge height in LAR
    hw      = stack.panel_width / 2.0
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
    return np.array(pts)


# ---------------------------------------------------------------------------
# Vectorized erosion proxy
# ---------------------------------------------------------------------------

def _erosion_proxy(
    p_nozzle: np.ndarray,    # (N, 3) — nozzle positions for valid cells
    plume_dir: np.ndarray,   # (N, 3) — unit plume direction per cell
    panel_pts: np.ndarray,   # (M, 3) — panel sample points
    n_exp: float,
) -> np.ndarray:
    """Integrated relative plume flux on solar panels for each arm pose.

    φ(r, θ) ∝ cos^n(θ) / r²  where θ = off-axis angle from plume axis.
    proxy[i] = Σ_j φ(r_{ij}, θ_{ij})   shape (N,)
    """
    # dv[i,j] = panel_pts[j] - p_nozzle[i]   shape (N, M, 3)
    dv   = panel_pts[np.newaxis, :, :] - p_nozzle[:, np.newaxis, :]
    dist = np.linalg.norm(dv, axis=-1)                             # (N, M)
    dist = np.where(dist < 0.02, 0.02, dist)

    # cos(θ) = unit(dv) · plume_dir[i]
    cos_th = np.einsum('nmi,ni->nm', dv / dist[..., np.newaxis], plume_dir)
    cos_th = np.clip(cos_th, 0.0, 1.0)                            # (N, M)

    flux = (cos_th ** n_exp) / dist ** 2                           # (N, M)
    return flux.sum(axis=-1)                                        # (N,)


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
            pivot, q0_valid[idx], q1_valid[idx], q2_valid[idx])
        traces.append(_arm_trace_plotly(pivot, pe, pw, pn, color, name_tag))

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
    n_hat_ee = np.array(cfg.get("nozzle_exit_direction_ee", [0., 0., 1.]))

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

    # ── 4. Thrust direction: nozzle → CoG  ────────────────────────────────
    to_cog   = cog[np.newaxis, :] - p_valid
    dist_cog = np.linalg.norm(to_cog, axis=-1, keepdims=True)
    dist_cog = np.where(dist_cog < 1e-6, 1e-6, dist_cog)
    thrust_d = to_cog / dist_cog       # (N_valid, 3)
    plume_d  = -thrust_d               # plume exits opposite to thrust

    # ── 5. Solar panel points + erosion proxy ─────────────────────────────
    print("Computing erosion proxy (vectorized)...")
    panel_pts = _panel_grid(STACK, tracking_deg=0.0)
    erosion   = _erosion_proxy(p_valid, plume_d, panel_pts,
                               n_exp=THRUSTER.plume_cosine_exponent)
    print(f"Erosion  min={erosion.min():.3e}  "
          f"max={erosion.max():.3e}  "
          f"median={np.median(erosion):.3e}")

    # Log-scale for perceptual range
    log_erosion = np.log10(np.clip(erosion, 1e-30, None))
    vmin = float(np.percentile(log_erosion, 1))
    vmax = float(np.percentile(log_erosion, 99))

    # Best / worst poses (by erosion)
    idx_min = int(erosion.argmin())
    idx_max = int(erosion.argmax())
    for label, idx in [("Min-erosion", idx_min), ("Max-erosion", idx_max)]:
        pn = p_valid[idx]
        pe, pw, _ = ARM.forward_kinematics(pivot, q0_valid[idx], q1_valid[idx], q2_valid[idx])
        print(f"  {label}: EE=({pn[0]:+.2f},{pn[1]:+.2f},{pn[2]:+.2f})  "
              f"q=({np.degrees(q0_valid[idx]):.1f}°,"
              f"{np.degrees(q1_valid[idx]):.1f}°,"
              f"{np.degrees(q2_valid[idx]):.1f}°)  "
              f"proxy={erosion[idx]:.3e}")

    # ── 6. Figure layout ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9), facecolor="#F8F9FA")
    fig.suptitle(
        "Thruster Arm Workspace  —  Plume Erosion Proxy per Reachable EE Position\n"
        f"Thrust direction: nozzle → CoG  |  "
        f"Erosion proxy: Σ cos^n(θ)/r² over {len(panel_pts)} panel points  |  "
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
            pivot, q0_valid[idx], q1_valid[idx], q2_valid[idx])
        _draw_arm_at_pose(ax3d, pivot, pe, pw, pn, color, tag)

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
