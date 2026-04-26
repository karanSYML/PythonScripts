#!/usr/bin/env python3
"""
GeometryEngine Verification Visualizer
=======================================
Interactive 3D viewer for the thruster arm / plume geometry.

Joint angles are controlled directly — no IK target is computed.
Link lengths are fixed (ARM defaults).

Usage
-----
    python geometry_visualizer.py            # interactive window
    python geometry_visualizer.py --save     # save PNG instead of showing

Components shown
----------------
  ■ Client bus       – grey wireframe + translucent faces
  ■ Servicer bus     – steel-blue wireframe, docked below client (Z−)
  ■ LAR interface    – teal band at Z = 0
  ■ Solar panels     – gold translucent rectangles (±X side, sun-tracking aware)
  ■ Panel flux       – per-point flux intensity overlaid on panels (colour map)
  ■ Robotic arm      – three links, colour-coded:
                         GREEN  = within joint limits + no collision
                         ORANGE = joint limit exceeded
                         RED    = arm collides with any obstacle
                       link1 (shoulder→elbow) and link2 (elbow→wrist): solid thick
                       bracket (wrist→thruster): dashed thinner
  ■ Joint markers    – pivot (■), elbow (●), wrist (▲), thruster (★)
  ■ Stack COG        – red diamond (◆)
  ■ Thrust vector    – black arrow from thruster toward COG
  ■ Plume cone       – semi-transparent red cone (beam divergence half-angle)

Interactive controls
--------------------
  Sliders:
    Hinge 1  q0  [°]   [0, 270]   initial: stowed_joint_angles_deg[0]
    Hinge 2  q1  [°]   [0, 235]   initial: stowed_joint_angles_deg[1]
    Hinge 3  q2  [°]   [-36, 99]  initial: stowed_joint_angles_deg[2]
    Panel Track  α     [−90°, 90°]
    Client mass        [1500 kg, 6000 kg]
    Servicer mass      [700 / 750 / 800 kg]
  Checkbox:
    Flux overlay
  Config file (stowed_config.json):
    pivot_position_lar_m  – override computed pivot (null = auto)
    servicer_yaw_deg      – servicer docking yaw
    stowed_joint_angles_deg – (q0, q1, q2) stowed reference
    stowed_ee_unit_vector   – satellite-spec stowed EE direction
"""

import sys
import json
import os
from dataclasses import replace as _dc_replace
import numpy as np

# ── Backend must be set before pyplot is imported ──────────────────────────────
import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plume_impingement_pipeline import (
    RoboticArmGeometry,
    StackConfig,
    ThrusterParams,
    arm_has_collision,
)
from arm_kinematics import arm_cog_and_jacobian


# ─── Stowed configuration (loaded from JSON at startup) ───────────────────────

_STOWED_CFG: dict = {}

def load_stowed_config(path: str = "stowed_config.json") -> dict:
    """Load stowed_config.json from the script directory. Returns {} on failure."""
    here = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(here, path)
    try:
        with open(fpath) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ─── Default geometry (fixed for static scene; masses come from state) ─────────

_STACK_GEOM = StackConfig(
    servicer_mass=744.0,
    servicer_bus_x=0.9, servicer_bus_y=1.5, servicer_bus_z=0.8,
    client_mass=2800.0,
    client_bus_x=2.3,  client_bus_y=3.0,  client_bus_z=5.0,
    panel_span_one_side=16.0, panel_width=2.5,
    lar_offset_z=0.05,
)
THRUSTER = ThrusterParams()
ARM = RoboticArmGeometry()    # fixed link lengths / joint limits throughout

# Servicer docked yaw angle relative to client/LAR frame [deg, about +Z]
SERVICER_YAW_DEG: float = -25.0


def _Rz(deg: float) -> np.ndarray:
    """3×3 rotation matrix for a right-hand rotation about Z by *deg* degrees."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


# ─── VisScene ─────────────────────────────────────────────────────────────────

class VisScene:
    """Artist registry separating static (one-time) from dynamic (per-update) artists.

    Static artists are drawn once in init_static_scene() and never touched again.
    Dynamic artists are cleared and rebuilt on every slider event without calling
    ax.cla(), preserving the static geometry and the current view angle.
    """

    def __init__(self):
        self.static: list = []
        self.dynamic: list = []

    def add_static(self, artists):
        if isinstance(artists, list):
            self.static.extend(artists)
        elif artists is not None:
            self.static.append(artists)

    def add_dynamic(self, artists):
        if isinstance(artists, list):
            self.dynamic.extend(artists)
        elif artists is not None:
            self.dynamic.append(artists)

    def clear_dynamic(self):
        for a in self.dynamic:
            try:
                a.remove()
            except (ValueError, AttributeError):
                pass
        self.dynamic.clear()


# ─── Pure geometry helpers ────────────────────────────────────────────────────

def pivot_position(stack: StackConfig, arm: RoboticArmGeometry) -> np.ndarray:
    """Pivot point in LAR frame.

    If stowed_config.json supplies a non-null pivot_position_lar_m, that value
    is used directly.  Otherwise the pivot is computed from the servicer origin
    plus the body-frame offset rotated by servicer_yaw_deg from the config
    (falling back to the module-level SERVICER_YAW_DEG constant).
    """
    override = _STOWED_CFG.get("pivot_position_lar_m", [None, None, None])
    if override and all(v is not None for v in override):
        return np.array(override, dtype=float)
    yaw_deg = _STOWED_CFG.get("servicer_yaw_deg", SERVICER_YAW_DEG)
    origin = stack.servicer_origin_in_lar_frame()
    return origin + _Rz(yaw_deg) @ arm.arm_pivot_in_servicer_body()


def panel_grid(stack: StackConfig, tracking_deg: float = 0.0,
               n_span: int = 20, n_chord: int = 6):
    """Return (pts, normal) for the solar panel grid."""
    track = np.radians(tracking_deg)
    zi_base = -stack.client_bus_z / 3.0
    pts = []
    for side in [+1, -1]:
        base_x = side * stack.client_bus_x / 2.0
        for i in range(n_span):
            xi = base_x + side * ((i + 0.5) / n_span) * stack.panel_span_one_side
            for j in range(n_chord):
                yi = ((j + 0.5) / n_chord - 0.5) * stack.panel_width
                pts.append([xi, yi * np.cos(track), zi_base + yi * np.sin(track)])
    pts = np.array(pts)
    normal = np.array([0.0, -np.sin(track), np.cos(track)])
    normal /= np.linalg.norm(normal)
    return pts, normal


def relative_flux(thruster_pos: np.ndarray, plume_dir: np.ndarray,
                  pts: np.ndarray, n: float = 10.0) -> np.ndarray:
    """Unnormalised relative ion flux at each panel point (for colouring)."""
    dv = pts - thruster_pos[np.newaxis, :]
    dist = np.linalg.norm(dv, axis=1)
    dist = np.where(dist < 0.01, 0.01, dist)
    cos_off = np.clip(np.dot(dv / dist[:, np.newaxis], plume_dir), 0.0, 1.0)
    return (cos_off ** n) / dist ** 2


# ─── Drawing helpers (all return lists of artists) ────────────────────────────

def _box_verts(center, dims):
    cx, cy, cz = center
    hx, hy, hz = dims[0] / 2, dims[1] / 2, dims[2] / 2
    return np.array([
        [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
        [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
        [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
        [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
    ])

_BOX_EDGES = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4),
              (0,4),(1,5),(2,6),(3,7)]
_BOX_FACE_IDX = [
    [0,1,2,3], [4,5,6,7],
    [0,1,5,4], [2,3,7,6],
    [1,2,6,5], [0,3,7,4],
]


def draw_box(ax, center, dims, edge_color, face_color=None,
             face_alpha=0.13, edge_alpha=0.88, lw=1.8,
             Rz_mat: np.ndarray = None) -> list:
    """Draw a (optionally yaw-rotated) wireframe box; return list of artists.

    *Rz_mat*: optional 3×3 rotation matrix applied to vertices about *center*.
    """
    v = _box_verts(center, dims)
    if Rz_mat is not None:
        cx, cy, cz = center
        orig = np.array([cx, cy, cz])
        v = (Rz_mat @ (v - orig).T).T + orig
    artists = []
    if face_color is not None:
        faces = [[v[i] for i in idx] for idx in _BOX_FACE_IDX]
        poly = Poly3DCollection(
            faces, alpha=face_alpha, facecolor=face_color, edgecolor="none")
        ax.add_collection3d(poly)
        artists.append(poly)
    for i, j in _BOX_EDGES:
        artists.extend(ax.plot3D(*zip(v[i], v[j]), color=edge_color,
                                  alpha=edge_alpha, linewidth=lw))
    return artists


def draw_panel_faces(ax, stack: StackConfig, tracking_deg: float = 0.0) -> list:
    track = np.radians(tracking_deg)
    zi = stack.client_bus_z / 2.0
    hw = stack.panel_width / 2.0
    artists = []

    def corner(x, y):
        return [x, y * np.cos(track), zi + y * np.sin(track)]

    for side in [+1, -1]:
        x0 = side * (stack.client_bus_x / 2.0 + stack.panel_hinge_offset_y)
        x1 = x0 + side * stack.panel_span_one_side
        quad = [corner(x0, -hw), corner(x1, -hw),
                corner(x1,  hw), corner(x0,  hw)]
        poly = Poly3DCollection(
            [quad], alpha=0.22,
            facecolor="#F39C12", edgecolor="#935116", linewidth=0.8)
        ax.add_collection3d(poly)
        artists.append(poly)
    return artists


def draw_flux_overlay(ax, pts: np.ndarray, flux_vals: np.ndarray,
                      cmap: str = "plasma"):
    """Return (artist_list, norm) — artist_list is [] if pts is empty."""
    if len(pts) == 0:
        return [], None
    norm = plt.Normalize(vmin=0.0, vmax=np.percentile(flux_vals, 98) or 1e-30)
    colors = plt.get_cmap(cmap)(norm(flux_vals))
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=colors, s=6, zorder=4, depthshade=False)
    return [sc], norm


def draw_plume_cone(ax, tip: np.ndarray, direction: np.ndarray,
                    half_angle_deg: float = 20.0, length: float = 2.5) -> list:
    ha = np.radians(half_angle_deg)
    ref = np.array([0., 0., 1.]) if abs(direction[2]) < 0.9 else np.array([1., 0., 0.])
    p1 = np.cross(direction, ref);  p1 /= np.linalg.norm(p1)
    p2 = np.cross(direction, p1)
    u = np.linspace(0, 2 * np.pi, 36)
    artists = []
    for t in np.linspace(0, length, 7)[1:]:
        r = t * np.tan(ha)
        ring = (tip[:, None]
                + t * direction[:, None]
                + r * (np.outer(p1, np.cos(u)) + np.outer(p2, np.sin(u))))
        artists.extend(ax.plot3D(ring[0], ring[1], ring[2],
                                  color="#E74C3C", alpha=0.10, linewidth=0.7))
    for k in range(8):
        a = 2 * np.pi * k / 8
        edge = (tip + length * direction
                + length * np.tan(ha) * (np.cos(a) * p1 + np.sin(a) * p2))
        artists.extend(ax.plot3D(*zip(tip, edge), color="#E74C3C", alpha=0.22, linewidth=0.8))
    return artists


def draw_arm_links(ax, pivot: np.ndarray, p_elbow: np.ndarray,
                   p_wrist: np.ndarray, p_thruster: np.ndarray,
                   arm_color: str) -> list:
    """Draw three arm segments + four joint markers; return artist list.

    Markers:  pivot (■)  elbow (●)  wrist (▲)  thruster (★)
    """
    artists = []
    # Link 1: pivot → elbow  (solid, thick)
    artists.extend(ax.plot3D(*zip(pivot, p_elbow),
                              color=arm_color, lw=5.5, solid_capstyle="round"))
    # Link 2: elbow → wrist  (solid, thick)
    artists.extend(ax.plot3D(*zip(p_elbow, p_wrist),
                              color=arm_color, lw=5.5, solid_capstyle="round"))
    # Bracket: wrist → thruster  (dashed, lighter)
    artists.extend(ax.plot3D(*zip(p_wrist, p_thruster),
                              color=arm_color, lw=3.0, linestyle="--",
                              solid_capstyle="round"))
    artists.append(ax.scatter(*pivot,      s=140, c="#1A252F", marker="s",
                               edgecolors="w", linewidths=1.5, zorder=9))
    artists.append(ax.scatter(*p_elbow,    s=95,  c=arm_color, marker="o",
                               edgecolors="w", linewidths=1.5, zorder=9))
    artists.append(ax.scatter(*p_wrist,    s=95,  c=arm_color, marker="^",
                               edgecolors="w", linewidths=1.5, zorder=9))
    artists.append(ax.scatter(*p_thruster, s=230, c="#8E44AD",  marker="*",
                               edgecolors="w", linewidths=1.0, zorder=9))
    return artists


def draw_antenna_dishes(ax, stack: StackConfig) -> list:
    """Draw all four reflector antennas as nadir-facing (+Z normal) flat discs.

    East dishes (E1, E2) on the +Y bus face — 2.2 m diameter, light silver.
    West dishes (W1, W2) on the −Y bus face — 2.5 m diameter, slightly darker.
    Each disc is rendered as a filled Poly3DCollection in the XY plane, with a
    rim outline and a dish-name text label offset above the centre.
    """
    centers   = stack.antenna_centers_in_lar_frame()
    diameters = {
        "E1": stack.antenna_diameter_east,
        "E2": stack.antenna_diameter_east,
        "W1": stack.antenna_diameter_west,
        "W2": stack.antenna_diameter_west,
    }
    face_colors = {
        "E1": "#D5D8DC", "E2": "#D5D8DC",   # light silver – East
        "W1": "#AEB6BF", "W2": "#AEB6BF",   # mid silver  – West
    }
    edge_color = "#717D7E"
    theta = np.linspace(0, 2 * np.pi, 64)
    artists = []

    for name, center in centers.items():
        r = diameters[name] / 2.0
        xs = center[0] + r * np.cos(theta)
        ys = center[1] + r * np.sin(theta)
        zs = np.full_like(xs, center[2])

        # Filled aperture disc
        poly = Poly3DCollection(
            [list(zip(xs, ys, zs))],
            alpha=0.40, facecolor=face_colors[name],
            edgecolor=edge_color, linewidth=1.0)
        ax.add_collection3d(poly)
        artists.append(poly)

        # Rim outline
        artists.extend(ax.plot3D(
            np.append(xs, xs[0]),
            np.append(ys, ys[0]),
            np.append(zs, zs[0]),
            color=edge_color, linewidth=0.9, alpha=0.75))

        # Dish-name label just above the disc centre
        txt = ax.text(center[0], center[1], center[2] + 0.18, name,
                      fontsize=7.5, color="#4D5656", ha="center", va="bottom",
                      fontweight="bold")
        artists.append(txt)

    return artists


def draw_gripper_arm(ax, stack: StackConfig, serv_origin: np.ndarray,
                     Rz_mat: np.ndarray = None) -> list:
    """Draw the rigid docking adapter between the servicer top face and the LAR.

    The adapter fills the structural gap from the servicer +Z face up to the
    LAR bottom face.  It is drawn as a narrow box (60 % of servicer footprint)
    centred on the servicer XY origin, with the same yaw as the servicer body.
    """
    z_bot = serv_origin[2] + stack.servicer_bus_z / 2.0   # servicer top face
    z_top = -(stack.lar_offset_z)                           # LAR bottom face
    height = z_top - z_bot
    if height <= 0.0:
        return []
    center = np.array([serv_origin[0], serv_origin[1], z_bot + height / 2.0])
    w_x = stack.servicer_bus_x * 0.055
    w_y = stack.servicer_bus_y * 0.045
    return draw_box(ax, center, [w_x, w_y, height],
                    edge_color="#1A252F", face_color="#2C3E50",
                    face_alpha=0.22, edge_alpha=0.80, lw=1.4,
                    Rz_mat=Rz_mat)


def draw_servicer_panels(ax, stack: StackConfig, serv_origin: np.ndarray,
                          Rz_mat: np.ndarray = None) -> list:
    """Draw the servicer's own solar panels as flat wing-like quads.

    Each panel is a flat rectangle lying in the local XY plane (horizontal),
    extending from the servicer ±Y face outward.  Both panels are rotated by
    the same yaw angle as the servicer body.

    Panel footprint in servicer body frame
    ──────────────────────────────────────
      X : −servicer_bus_x/2  →  +servicer_bus_x/2   (servicer body width)
      Y : ±servicer_bus_y/2  →  ±(servicer_bus_y/2 + span)
      Z : 0  (at servicer centre height)
    """
    half_x = stack.servicer_bus_x / 2.0
    half_y = stack.servicer_bus_y / 2.0
    gap    = stack.servicer_panel_gap   # m gap between bus face and panel inner edge
    span   = stack.servicer_panel_span  # m panel span per side

    artists = []
    for side in [+1, -1]:
        # 4 corners in servicer body frame (flat, z = 0 → servicer centre height)
        body = np.array([
            [-half_x, side * (half_y + gap),          0.0],
            [ half_x, side * (half_y + gap),          0.0],
            [ half_x, side * (half_y + gap + span),   0.0],
            [-half_x, side * (half_y + gap + span),   0.0],
        ])
        # Rotate into LAR frame and translate
        if Rz_mat is not None:
            lar = (Rz_mat @ body.T).T + serv_origin
        else:
            lar = body + serv_origin

        poly = Poly3DCollection(
            [lar.tolist()], alpha=0.28,
            facecolor="#F0B27A", edgecolor="#935116", linewidth=0.9)
        ax.add_collection3d(poly)
        artists.append(poly)

        # Panel outline for clarity
        outline = np.vstack([lar, lar[0]])
        artists.extend(ax.plot3D(outline[:, 0], outline[:, 1], outline[:, 2],
                                  color="#935116", linewidth=0.9, alpha=0.70))
    return artists


def set_equal_aspect(ax, _pts=None):
    """Set fixed axis bounds suited to the full assembly + arm reach."""
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-3.0, 5.0)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass


# ─── Scene management ─────────────────────────────────────────────────────────

def _legend_elements() -> list:
    """Return the list of legend handle objects (Patch / Line2D)."""
    return [
        Patch(fc="#BDC3C7", ec="#5D6D7E", label="Client bus"),
        Patch(fc="#5DADE2", ec="#2471A3", label=f"Servicer bus  (yaw {SERVICER_YAW_DEG:.0f}\u00b0)"),
        Patch(fc="#2C3E50", ec="#1A252F", label="Gripper arm / docking adapter"),
        Patch(fc="#F0B27A", ec="#935116", label="Servicer solar panels"),
        Patch(fc="#76D7C4", ec="#17A589", label="LAR interface"),
        Patch(fc="#F39C12", ec="#935116", label="Client solar panels  (\u00b1X)"),
        Patch(fc="#D5D8DC", ec="#717D7E", label="Antenna reflectors  (\u00d74)"),
        Line2D([0], [0], color="#2471A3", marker="D", markersize=7, lw=0,
               markeredgecolor="w", label="Servicer origin"),
        Line2D([0], [0], color="#C0392B", marker="D", markersize=7, lw=0,
               markeredgecolor="w", label="Stack CoG  (arm + masses)"),
        Line2D([0], [0], color="#E74C3C", lw=2, alpha=0.5, label="Plume cone"),
        Line2D([0], [0], color="#1A252F", lw=2, label="Thrust vector"),
        Line2D([0], [0], color="#27AE60", lw=4, label="Arm \u2013 OK"),
        Line2D([0], [0], color="#E67E22", lw=4, label="Arm \u2013 Limit exceeded"),
        Line2D([0], [0], color="#E74C3C", lw=4, label="Arm \u2013 Collision / unreachable"),
        Line2D([0], [0], color="#F4D03F", lw=2.5, linestyle="--",
               label="Stowed EE dir (spec)"),
    ]


def init_static_scene(ax, stack: StackConfig, scene: VisScene):
    """Draw time-invariant geometry (buses, LAR) and set axes style.

    Called once at start-up.  Uses only geometric parameters of *stack*;
    mass parameters are ignored here.  The legend is drawn separately in a
    dedicated axes via draw_legend().
    """
    serv_origin = stack.servicer_origin_in_lar_frame()
    Rz = _Rz(SERVICER_YAW_DEG)    # servicer body is yawed 25° in LAR frame

    # Client bus (not rotated — aligned with LAR frame)
    scene.add_static(draw_box(
        ax, [0, 0, stack.client_bus_z / 2.0],
        [stack.client_bus_x, stack.client_bus_y, stack.client_bus_z],
        edge_color="#5D6D7E", face_color="#BDC3C7", face_alpha=0.13))

    # Servicer bus (rotated 25° about Z)
    scene.add_static(draw_box(
        ax, serv_origin,
        [stack.servicer_bus_x, stack.servicer_bus_y, stack.servicer_bus_z],
        edge_color="#2471A3", face_color="#5DADE2", face_alpha=0.20,
        Rz_mat=Rz))

    # LAR interface (aligned with LAR frame)
    lar_z = -(stack.lar_offset_z / 2.0)
    lar_w = min(stack.client_bus_x, stack.servicer_bus_x) * 0.65
    scene.add_static(draw_box(
        ax, [0, 0, lar_z],
        [lar_w, lar_w * 0.85, stack.lar_offset_z],
        edge_color="#17A589", face_color="#76D7C4", face_alpha=0.45, lw=1.2))

    # Rigid gripper arm — structural adapter from servicer top to LAR (rotated)
    scene.add_static(draw_gripper_arm(ax, stack, serv_origin, Rz_mat=Rz))

    # Servicer solar panels (rotated with servicer body)
    scene.add_static(draw_servicer_panels(ax, stack, serv_origin, Rz_mat=Rz))

    # Antenna reflectors (geometry-only, mass-independent → static)
    scene.add_static(draw_antenna_dishes(ax, stack))

    # Servicer origin marker — blue diamond, matches servicer bus colour
    scene.add_static([ax.scatter(
        *serv_origin, s=160, c="#2471A3", marker="D",
        edgecolors="w", linewidths=1.5, zorder=8)])

    # Stowed EE reference arrow from satellite spec (golden dashed)
    spec_vec = _STOWED_CFG.get("stowed_ee_unit_vector")
    if spec_vec is not None:
        sv = np.array(spec_vec, dtype=float)
        stowed_pivot = pivot_position(stack, ARM)
        arr_len = 1.5
        scene.add_static([ax.quiver(
            *stowed_pivot, *(sv * arr_len),
            color="#F4D03F", linewidth=2.0, arrow_length_ratio=0.18,
            linestyle="dashed")])

    ax.set_xlabel("X  [m]", fontsize=9, labelpad=3)
    ax.set_ylabel("Y  [m]", fontsize=9, labelpad=3)
    ax.set_zlabel("Z  [m]  (+Z = nadir)", fontsize=9, labelpad=3)
    ax.set_title("GeometryEngine 3D Verification", fontsize=11,
                 fontweight="bold", pad=7)
    set_equal_aspect(ax)


def draw_legend(ax_leg):
    """Populate a dedicated 2-D axes with the scene legend (no 3-D plot clutter)."""
    ax_leg.axis("off")
    leg = ax_leg.legend(
        handles=_legend_elements(),
        loc="upper left",
        fontsize=7.5,
        framealpha=0.92,
        edgecolor="#BDC3C7",
        handlelength=1.6,
        handleheight=1.1,
        borderpad=0.8,
        labelspacing=0.55,
        title="Legend",
        title_fontsize=8.5,
    )
    ax_leg.add_artist(leg)


def update_dynamic_scene(ax, ax_info, scene: VisScene, state: dict):
    """Clear dynamic artists and rebuild for current state; refresh status panel.

    Joint angles (q0_deg, q1_deg, q2_deg) are read directly from state and
    converted to radians — no IK is performed.
    Both masses are injected into a copy of _STACK_GEOM via dataclasses.replace.
    """
    scene.clear_dynamic()
    ax_info.cla()
    ax_info.axis("off")

    tracking_deg = state["tracking_deg"]
    show_flux    = state["show_flux"]

    # Build stack with current masses
    stack = _dc_replace(_STACK_GEOM,
                        client_mass=state["client_mass"],
                        servicer_mass=state["servicer_mass"])

    yaw_deg = _STOWED_CFG.get("servicer_yaw_deg", SERVICER_YAW_DEG)
    pivot   = pivot_position(stack, ARM)

    # ── Joint angles direct from sliders ──────────────────────────────────
    q0 = np.radians(state["q0_deg"])
    q1 = np.radians(state["q1_deg"])
    q2 = np.radians(state["q2_deg"])
    in_limits = ARM.within_joint_limits(q0, q1, q2)

    # ── Forward kinematics ────────────────────────────────────────────────
    p_elbow, p_wrist, p_thruster = ARM.forward_kinematics(
        pivot, q0, q1, q2, servicer_yaw_deg=yaw_deg)

    # ── CoG (analytical, from arm_kinematics) ─────────────────────────────
    base_cog   = stack.stack_cog()
    stack_mass = stack.client_mass + stack.servicer_mass
    arm_cog_3d, _ = arm_cog_and_jacobian(
        ARM, pivot, np.array([q0, q1, q2]), servicer_yaw_deg=yaw_deg)
    arm_mass = ARM.arm_mass()
    cog = (stack_mass * base_cog + arm_mass * arm_cog_3d) / (stack_mass + arm_mass)

    # ── Plume / thrust direction from FK nozzle axis (CR3 @ n_hat_body) ─────
    plume_dir  = ARM.nozzle_direction_lar(q0, q1, q2, servicer_yaw_deg=yaw_deg)
    thrust_dir = -plume_dir
    dist_to_cog = np.linalg.norm(cog - p_thruster)

    # ── Collision check (client bus + servicer box + panels + antennas) ─────
    collision = arm_has_collision(pivot, p_elbow, p_wrist, p_thruster,
                                  stack, yaw_deg)

    # ── Arm colour ────────────────────────────────────────────────────────
    if collision:
        arm_clr, arm_tag = "#E74C3C", "COLLISION"
    elif not in_limits:
        arm_clr, arm_tag = "#E67E22", "LIMIT EXCEEDED"
    else:
        arm_clr, arm_tag = "#27AE60", "OK"

    # ── Panel grid + flux ─────────────────────────────────────────────────
    panel_pts, _ = panel_grid(stack, tracking_deg)
    flux = relative_flux(p_thruster, plume_dir, panel_pts,
                         n=THRUSTER.plume_cosine_exponent)

    # ── Dynamic draws ──────────────────────────────────────────────────────

    # Solar panels (tracking_deg changes with slider)
    scene.add_dynamic(draw_panel_faces(ax, stack, tracking_deg))

    # Flux overlay
    flux_norm = None
    if show_flux:
        flux_artists, flux_norm = draw_flux_overlay(ax, panel_pts, flux)
        scene.add_dynamic(flux_artists)

    # Plume cone
    scene.add_dynamic(draw_plume_cone(
        ax, p_thruster, plume_dir,
        half_angle_deg=THRUSTER.beam_divergence_half_angle,
        length=min(ARM.link1_length + ARM.link2_length, 3.0) * 0.7))

    # Thrust vector arrow
    arrow_len = min((ARM.link1_length + ARM.link2_length) * 0.30, 1.4)
    scene.add_dynamic([ax.quiver(
        *p_thruster, *(thrust_dir * arrow_len),
        color="#1A252F", linewidth=2.5, arrow_length_ratio=0.30)])

    # Arm: three segments + four joint markers
    scene.add_dynamic(draw_arm_links(ax, pivot, p_elbow, p_wrist, p_thruster, arm_clr))

    # Stack CoG
    scene.add_dynamic([ax.scatter(
        *cog, s=210, c="#C0392B", marker="D",
        edgecolors="w", linewidths=1.5, zorder=10)])

    # ── Status panel ──────────────────────────────────────────────────────
    ok_c  = "#1E8449"
    bad_c = "#C0392B"
    lines = []

    def section(title):
        lines.append(("── " + title + " ──", True, "#1A252F", 9.5))

    def row(text, color="#2C3E50", bold=False):
        lines.append((text, bold, color, 9.0))

    def blank():
        lines.append(("", False, "white", 5.0))

    section("ARM STATUS")
    row(arm_tag, color=arm_clr, bold=True)
    blank()

    section("THRUSTER POS")
    row(f"X = {p_thruster[0]:+.3f} m")
    row(f"Y = {p_thruster[1]:+.3f} m")
    row(f"Z = {p_thruster[2]:+.3f} m")
    blank()

    section("STACK COG")
    row(f"X = {cog[0]:+.3f} m")
    row(f"Y = {cog[1]:+.3f} m")
    row(f"Z = {cog[2]:+.3f} m")
    blank()

    section("PIVOT POS")
    row(f"X = {pivot[0]:+.3f} m")
    row(f"Y = {pivot[1]:+.3f} m")
    row(f"Z = {pivot[2]:+.3f} m")
    blank()

    section("JOINT ANGLES")
    row(f"q0 hinge1 = {np.degrees(q0):+6.1f}°")
    row(f"q1 hinge2 = {np.degrees(q1):+6.1f}°")
    row(f"q2 hinge3 = {np.degrees(q2):+6.1f}°")
    blank()

    section("FEASIBILITY")
    row(f"Joint limits:  {'OK ' if in_limits  else 'EXCEEDED'}",
        color=ok_c if in_limits else bad_c)
    row(f"Bus collision: {'YES' if collision   else 'NO  '}",
        color=bad_c if collision else ok_c)
    blank()

    section("DISTANCES")
    row(f"|Thruster–COG|  = {dist_to_cog:.2f} m")
    if len(panel_pts) > 0:
        d_tp = np.linalg.norm(panel_pts - p_thruster[np.newaxis, :], axis=1)
        row(f"Min panel dist = {d_tp.min():.2f} m")
        row(f"Peak flux pt   = {flux.max():.3e}")

    section("MASSES")
    row(f"Servicer : {stack.servicer_mass:.0f} kg")
    row(f"Client   : {stack.client_mass:.0f} kg")
    row(f"Arm      : {arm_mass:.1f} kg")

    # ── Stowed verification ───────────────────────────────────────────────
    spec_vec = _STOWED_CFG.get("stowed_ee_unit_vector")
    if spec_vec is not None:
        sv = np.array(spec_vec, dtype=float)
        diff = p_thruster - p_wrist
        norm_diff = np.linalg.norm(diff)
        d_bracket = diff / norm_diff if norm_diff > 1e-9 else np.array([0., 0., 1.])
        cos_a = float(np.clip(np.dot(d_bracket, sv), -1.0, 1.0))
        ang_deg = np.degrees(np.arccos(cos_a))
        blank()
        section("STOWED VERIFY")
        row(f"Spec  [{sv[0]:+.4f},{sv[1]:+.4f},{sv[2]:+.4f}]")
        row(f"Actual[{d_bracket[0]:+.4f},{d_bracket[1]:+.4f},{d_bracket[2]:+.4f}]")
        row(f"Angular diff: {ang_deg:.1f}\u00b0",
            color=ok_c if ang_deg < 5.0 else bad_c, bold=True)

    y = 0.985
    dy_map = {True: 0.048, False: 0.043}
    ax_info.set_title("Status", fontsize=10, fontweight="bold", pad=4)
    for text, bold, color, size in lines:
        if text == "":
            y -= 0.016
            continue
        ax_info.text(0.04, y, text, transform=ax_info.transAxes,
                     fontsize=size, color=color, va="top",
                     fontweight="bold" if bold else "normal",
                     fontfamily="monospace")
        y -= dy_map[bold]

    # Flux colourbar
    if show_flux and flux_norm is not None:
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.get_cmap("plasma"))
        sm.set_array([])
        try:
            ax_info.figure.colorbar(sm, ax=ax_info, orientation="horizontal",
                                    fraction=0.06, pad=0.04,
                                    label="Rel. flux (norm.)")
        except Exception:
            pass


# ─── Figure setup + main ──────────────────────────────────────────────────────

def main():
    global _STOWED_CFG
    save_mode = "--save" in sys.argv

    # Load stowed config before anything else so pivot_position() can use it
    _STOWED_CFG = load_stowed_config()

    # Initial joint angles from stowed config (defaults 0,0,0)
    _stowed_q   = _STOWED_CFG.get("stowed_joint_angles_deg", [0.0, 0.0, 0.0])
    init_q0 = float(_stowed_q[0]) if len(_stowed_q) > 0 else 0.0
    init_q1 = float(_stowed_q[1]) if len(_stowed_q) > 1 else 0.0
    init_q2 = float(_stowed_q[2]) if len(_stowed_q) > 2 else 0.0

    fig = plt.figure(figsize=(18, 9), facecolor="#F8F9FA")
    fig.suptitle(
        "Thruster Arm Geometry Verifier  \u00b7  Servicer below client (Z\u2212)  \u00b7  "
        "Joint angles \u2192 FK  \u00b7  Drag to rotate",
        fontsize=11, fontweight="bold", y=0.998, color="#1A252F",
    )

    # ── Axes layout ───────────────────────────────────────────────────────
    #   [0.01, 0.26, 0.56, 0.71]  3D scene
    #   [0.59, 0.26, 0.19, 0.71]  Status panel
    #   [0.79, 0.26, 0.20, 0.71]  Legend (outside 3D plot, pushed right)
    ax3d    = fig.add_axes([0.01, 0.26, 0.56, 0.71], projection="3d")
    ax3d.set_facecolor("#EBF5FB")
    ax_info = fig.add_axes([0.59, 0.26, 0.19, 0.71])
    ax_leg  = fig.add_axes([0.79, 0.26, 0.20, 0.71])

    ax3d.view_init(elev=20, azim=-110)

    # ── Mutable shared state ───────────────────────────────────────────────
    state = {
        "q0_deg":        init_q0,
        "q1_deg":        init_q1,
        "q2_deg":        init_q2,
        "tracking_deg":  0.0,
        "client_mass":   2800.0,
        "servicer_mass": 744.0,
        "show_flux":     False,
    }

    # ── Static scene + legend (drawn once) ────────────────────────────────
    scene = VisScene()
    init_static_scene(ax3d, _STACK_GEOM, scene)
    draw_legend(ax_leg)

    def _redraw():
        update_dynamic_scene(ax3d, ax_info, scene, state)
        fig.canvas.draw_idle()

    if not save_mode:
        # ── Sliders ────────────────────────────────────────────────────────
        # Six rows; bottom y starts at 0.040, stepping 0.032 each row.
        sl_y = [0.205, 0.173, 0.141, 0.109, 0.077, 0.040]
        sl_w = 0.50
        sl_x = 0.07
        sl_h = 0.022

        ax_q0    = fig.add_axes([sl_x, sl_y[0], sl_w, sl_h])
        ax_q1    = fig.add_axes([sl_x, sl_y[1], sl_w, sl_h])
        ax_q2    = fig.add_axes([sl_x, sl_y[2], sl_w, sl_h])
        ax_track = fig.add_axes([sl_x, sl_y[3], sl_w, sl_h])
        ax_cmass = fig.add_axes([sl_x, sl_y[4], sl_w, sl_h])
        ax_smass = fig.add_axes([sl_x, sl_y[5], sl_w, sl_h])

        sl_q0    = Slider(ax_q0,    "Hinge 1  q0  [\u00b0]",    0.0, 270.0, valinit=init_q0, valstep=1.0)
        sl_q1    = Slider(ax_q1,    "Hinge 2  q1  [\u00b0]",    0.0, 235.0, valinit=init_q1, valstep=1.0)
        sl_q2    = Slider(ax_q2,    "Hinge 3  q2  [\u00b0]",  -36.0,  99.0, valinit=init_q2, valstep=1.0)
        sl_track = Slider(ax_track, "Panel Track  \u03b1 [\u00b0]", -90, 90, valinit=0,  valstep=5)
        sl_cmass = Slider(ax_cmass, "Client mass [kg]",     1500, 6000,  valinit=2800,   valstep=50)
        sl_smass = Slider(ax_smass, "Servicer mass [kg]",    700,  800,  valinit=744,    valstep=50)

        def _on_slider(_val):
            state["q0_deg"]        = sl_q0.val
            state["q1_deg"]        = sl_q1.val
            state["q2_deg"]        = sl_q2.val
            state["tracking_deg"]  = sl_track.val
            state["client_mass"]   = sl_cmass.val
            state["servicer_mass"] = sl_smass.val
            _redraw()

        for sl in (sl_q0, sl_q1, sl_q2, sl_track, sl_cmass, sl_smass):
            sl.on_changed(_on_slider)

        # ── Checkbox: flux overlay (shifted right, away from status panel) ──
        ax_check = fig.add_axes([0.85, 0.010, 0.12, 0.040])
        check = CheckButtons(ax_check, ["Flux overlay"], [False])

        def _on_check(_label):
            state["show_flux"] = check.get_status()[0]
            _redraw()

        check.on_clicked(_on_check)

    # ── Initial draw ───────────────────────────────────────────────────────
    update_dynamic_scene(ax3d, ax_info, scene, state)

    if save_mode:
        out = "geometry_verification.png"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="#F8F9FA")
        print(f"Saved \u2192 {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
