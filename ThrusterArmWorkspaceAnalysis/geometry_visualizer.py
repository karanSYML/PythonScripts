#!/usr/bin/env python3
"""
GeometryEngine Verification Visualizer
=======================================
Interactive 3D viewer for verifying the thruster arm / plume geometry.

Usage
-----
    python geometry_visualizer.py            # interactive window
    python geometry_visualizer.py --save     # save PNG instead of showing

Components shown
----------------
  ■ Client bus       – grey wireframe + translucent faces
  ■ Servicer bus     – steel-blue wireframe, correctly docked below client (Z−)
  ■ LAR interface    – teal band between the two buses
  ■ Solar panels     – gold translucent rectangles (±X side, sun-tracking aware)
  ■ Panel flux       – per-point flux intensity overlaid on panels (colour map)
  ■ Robotic arm      – two links, colour-coded:
                         GREEN  = IK feasible + no collision
                         ORANGE = IK infeasible (out of reach / joint limits)
                         RED    = arm segment intersects client bus
  ■ Joint markers    – pivot (■), elbow (●), thruster (★)
  ■ Stack COG        – red diamond (◆)
  ■ Thrust vector    – black arrow from thruster toward COG
  ■ Plume cone       – semi-transparent red cone (beam divergence half-angle)

Interactive controls
--------------------
  Sliders:
    Shoulder Yaw   q0    [−180°, 180°]
    Shoulder Elev  φ     [−80°,   80°]   (desired end-effector elevation)
    Arm Reach      L     [ 1 m,    8 m]  (L1 + L2)
    Link Ratio     L1/L  [ 0.2,   0.8]
    Panel Track    α     [−45°,   45°]
  Radio:
    Elbow Up / Elbow Down
"""

import sys
import numpy as np

# ── Backend must be set before pyplot is imported ──────────────────────────────
import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colorbar as mcbar
import matplotlib.cm as mcm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plume_impingement_pipeline import (
    RoboticArmGeometry,
    StackConfig,
    ThrusterParams,
    _segment_intersects_aabb,
)


# ─── Default stack & thruster ──────────────────────────────────────────────────

STACK = StackConfig(
    servicer_mass=750.0,
    servicer_bus_x=0.9, servicer_bus_y=1.5, servicer_bus_z=0.8,
    client_mass=2500.0,
    client_bus_x=2.3,  client_bus_y=3.0,  client_bus_z=5.0,
    panel_span_one_side=16.0, panel_width=2.5,
    lar_offset_z=0.05,
)
THRUSTER = ThrusterParams()
ARM = RoboticArmGeometry()

# ─── Pure geometry helpers (independent of GeometryEngine) ────────────────────

def pivot_position(stack: StackConfig, arm: RoboticArmGeometry) -> np.ndarray:
    """Pivot point in LAR frame. Placed on the servicer +Z face (toward LAR)."""
    origin = stack.servicer_origin_in_lar_frame()
    print(origin, origin+np.array([arm.pivot_offset_x, arm.pivot_offset_y, arm.pivot_offset_z]))
    return origin + np.array([arm.pivot_offset_x, arm.pivot_offset_y, arm.pivot_offset_z]) #np.array([0.0, 0.0, stack.servicer_bus_z / 2.0])


def panel_grid(stack: StackConfig, tracking_deg: float = 0.0,
               n_span: int = 20, n_chord: int = 6):
    """Panel grid points + normals in client frame.

    Returns
    -------
    pts    : (N, 3) array of panel surface points
    normal : (3,) unit surface normal (sun-facing side)
    """
    track = np.radians(tracking_deg)
    zi_base = -stack.client_bus_z / 3.0 
    hw = stack.panel_width / 2.0
    pts = []

    for side in [+1, -1]:
        base_x = side * stack.client_bus_x / 2.0
        for i in range(n_span):
            xi = base_x + side * ((i + 0.5) / n_span) * stack.panel_span_one_side
            for j in range(n_chord):
                yi = ((j + 0.5) / n_chord - 0.5) * stack.panel_width
                yi_rot = yi * np.cos(track)
                zi = zi_base + yi * np.sin(track)
                pts.append([xi, yi_rot, zi])

    pts = np.array(pts)
    normal = np.array([0.0, -np.sin(track), np.cos(track)])
    normal /= np.linalg.norm(normal)
    return pts, normal


def relative_flux(thruster_pos: np.ndarray, plume_dir: np.ndarray,
                  pts: np.ndarray, n: float = 10.0) -> np.ndarray:
    """Relative ion flux at each panel point (unnormalised, for colouring only).

    flux ~ cos^n(off-axis) / r²
    """
    dv = pts - thruster_pos[np.newaxis, :]
    dist = np.linalg.norm(dv, axis=1)
    dist = np.where(dist < 0.01, 0.01, dist)
    cos_off = np.clip(np.dot(dv / dist[:, np.newaxis], plume_dir), 0.0, 1.0)
    flux = (cos_off ** n) / dist ** 2
    return flux


def arm_collision(pivot: np.ndarray, p_elbow: np.ndarray,
                  p_thruster: np.ndarray, stack: StackConfig) -> bool:
    """True if either arm link intersects the client bus AABB."""
    bx, by, bz = stack.client_bus_x / 2, stack.client_bus_y / 2, stack.client_bus_z / 2
    box_min = np.array([-bx, -by, -bz])
    box_max = np.array([ bx,  by,  bz])
    return (_segment_intersects_aabb(pivot,   p_elbow,   box_min, box_max) or
            _segment_intersects_aabb(p_elbow, p_thruster, box_min, box_max))


# ─── Drawing helpers ───────────────────────────────────────────────────────────

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
             face_alpha=0.13, edge_alpha=0.88, lw=1.8):
    v = _box_verts(center, dims)
    if face_color is not None:
        faces = [[v[i] for i in idx] for idx in _BOX_FACE_IDX]
        ax.add_collection3d(Poly3DCollection(
            faces, alpha=face_alpha, facecolor=face_color, edgecolor="none"))
    for i, j in _BOX_EDGES:
        ax.plot3D(*zip(v[i], v[j]), color=edge_color,
                  alpha=edge_alpha, linewidth=lw)


def draw_panel_faces(ax, stack, tracking_deg=0.0):
    """Draw both solar panels as translucent gold quads."""
    track = np.radians(tracking_deg)
    zi = stack.client_bus_z / 2.0 
    hw = stack.panel_width / 2.0

    def corner(x, y):
        return [x, y * np.cos(track), zi + y * np.sin(track)]

    for side in [+1, -1]:
        x0 = side * (stack.client_bus_x / 2.0 + stack.panel_hinge_offset_y)
        x1 = x0 + side * stack.panel_span_one_side
        quad = [corner(x0, -hw), corner(x1, -hw),
                corner(x1,  hw), corner(x0,  hw)]
        ax.add_collection3d(Poly3DCollection(
            [quad], alpha=0.22,
            facecolor="#F39C12", edgecolor="#935116", linewidth=0.8))


def draw_flux_overlay(ax, pts, flux_vals, cmap="plasma"):
    """Scatter panel grid points coloured by relative flux."""
    if len(pts) == 0:
        return None
    norm = plt.Normalize(vmin=0.0, vmax=np.percentile(flux_vals, 98) or 1e-30)
    colors = plt.get_cmap(cmap)(norm(flux_vals))
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=colors, s=6, zorder=4, depthshade=False)
    return norm


def draw_plume_cone(ax, tip, direction, half_angle_deg=20.0, length=2.5):
    """Wireframe cone representing the 1/e² beam divergence envelope."""
    ha = np.radians(half_angle_deg)
    ref = np.array([0., 0., 1.]) if abs(direction[2]) < 0.9 else np.array([1., 0., 0.])
    p1 = np.cross(direction, ref);  p1 /= np.linalg.norm(p1)
    p2 = np.cross(direction, p1)

    u = np.linspace(0, 2 * np.pi, 36)
    for t in np.linspace(0, length, 7)[1:]:
        r = t * np.tan(ha)
        ring = (tip[:, None]
                + t * direction[:, None]
                + r * (np.outer(p1, np.cos(u)) + np.outer(p2, np.sin(u))))
        ax.plot3D(ring[0], ring[1], ring[2],
                  color="#E74C3C", alpha=0.10, linewidth=0.7)
    for k in range(8):
        a = 2 * np.pi * k / 8
        edge = tip + length * direction + length * np.tan(ha) * (np.cos(a)*p1 + np.sin(a)*p2)
        ax.plot3D(*zip(tip, edge), color="#E74C3C", alpha=0.22, linewidth=0.8)


def set_equal_aspect(ax, pts):
    """Force equal-scale axes in 3D given a point cloud."""
    pts = np.array(pts)
    mid   = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    half  = max(pts.max(axis=0) - pts.min(axis=0)) / 2.0 * 1.15 + 0.3
    # ax.set_xlim(mid[0] - half, mid[0] + half)
    # ax.set_ylim(mid[1] - half, mid[1] + half)
    # ax.set_zlim(mid[2] - half, mid[2] + half)
    
    # Set manually to visualize the movement of the Arm
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-3.0, 5.0)

    try:
        ax.set_box_aspect([1, 1, 1])   # matplotlib ≥ 3.3
    except AttributeError:
        pass


# ─── Core redraw ──────────────────────────────────────────────────────────────

def redraw(ax, ax_info, stack: StackConfig, arm:RoboticArmGeometry,
           yaw_deg: float, elev_deg: float,
           reach_m: float, link_ratio: float,
           tracking_deg: float, elbow_up: bool,
           show_flux: bool):
    """Clear and rebuild the complete 3D scene."""
    # Preserve current view angle
    try:
        az_view, el_view = ax.azim, ax.elev
    except AttributeError:
        az_view, el_view = -55.0, 22.0

    ax.cla()
    ax_info.cla()
    ax_info.axis("off")

    # ── Compute geometry ───────────────────────────────────────────────────
    arm = RoboticArmGeometry(
        link1_length=reach_m * link_ratio,
        link2_length=reach_m * (1.0 - link_ratio),
        shoulder_yaw_deg=yaw_deg,
        elbow_up=elbow_up,
    )

    pivot = pivot_position(stack, arm)
    serv_origin = stack.servicer_origin_in_lar_frame()

    # IK target: full reach in (yaw, elev) direction from pivot
    az = np.radians(yaw_deg)
    el = np.radians(elev_deg)
    ik_target = pivot + reach_m * np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])

    ik_result = arm.inverse_kinematics(pivot, ik_target, elbow_up=elbow_up)

    if ik_result is not None:
        q0, q1, q2 = ik_result
        feasible = arm.within_joint_limits(q0, q1, q2)
    else:
        q0, q1, q2 = np.radians(yaw_deg), 0.0, 0.0
        feasible = False

    p_elbow, p_wrist, p_thruster = arm.forward_kinematics(pivot, q0, q1, q2)

    # Full stack CoG including arm link masses at current joint configuration
    base_cog   = stack.stack_cog()
    stack_mass = stack.client_mass + stack.servicer_mass
    u_rad   = np.array([np.cos(q0), np.sin(q0), 0.0])
    u_z     = np.array([0.0, 0.0, 1.0])
    d_upper = np.cos(q1) * u_rad + np.sin(q1) * u_z
    d_lower = np.cos(q1 + q2) * u_rad + np.sin(q1 + q2) * u_z
    cog_L1      = pivot   + (arm.link1_length   / 2.0) * d_upper
    cog_L2      = p_elbow + (arm.link2_length   / 2.0) * d_lower
    cog_bracket = p_wrist + (arm.bracket_length / 2.0) * d_lower
    arm_cog  = (arm.link1_mass * cog_L1 + arm.link2_mass * cog_L2 +
                arm.bracket_mass * cog_bracket) / arm.arm_mass()
    cog = (stack_mass * base_cog + arm.arm_mass() * arm_cog) / (stack_mass + arm.arm_mass())

    # Thrust/plume direction
    to_cog = cog - p_thruster
    dist_to_cog = np.linalg.norm(to_cog)
    if dist_to_cog > 1e-6:
        thrust_dir = to_cog / dist_to_cog
    else:
        thrust_dir = np.array([0.0, 0.0, 1.0])
    plume_dir = -thrust_dir

    # Collision check
    collision = arm_collision(pivot, p_elbow, p_thruster, stack)

    # Panel grid + flux
    panel_pts, panel_normal = panel_grid(stack, tracking_deg)
    flux = relative_flux(p_thruster, plume_dir, panel_pts,
                         n=THRUSTER.plume_cosine_exponent)

    # ── Arm colour ─────────────────────────────────────────────────────────
    if collision:
        arm_clr, arm_tag = "#E74C3C", "COLLISION"
    elif not feasible:
        arm_clr, arm_tag = "#E67E22", "INFEASIBLE"
    else:
        arm_clr, arm_tag = "#27AE60", "OK"

    # ── Draw: client bus ───────────────────────────────────────────────────
    draw_box(ax, [0, 0, stack.client_bus_z/2.0],
             [stack.client_bus_x, stack.client_bus_y, stack.client_bus_z],
             edge_color="#5D6D7E", face_color="#BDC3C7", face_alpha=0.13)

    # ── Draw: servicer bus (Z−) ────────────────────────────────────────────
    draw_box(ax, serv_origin,
             [stack.servicer_bus_x, stack.servicer_bus_y, stack.servicer_bus_z],
             edge_color="#2471A3", face_color="#5DADE2", face_alpha=0.20)

    # ── Draw: LAR interface ────────────────────────────────────────────────
    # lar_z   = -(stack.client_bus_z / 2.0 + stack.lar_offset_z / 2.0)
    lar_z   = -(stack.lar_offset_z / 2.0)
    lar_w   = min(stack.client_bus_x, stack.servicer_bus_x) * 0.65
    draw_box(ax, [0, 0, lar_z],
             [lar_w, lar_w * 0.85, stack.lar_offset_z],
             edge_color="#17A589", face_color="#76D7C4",
             face_alpha=0.45, lw=1.2)

    # ── Draw: solar panels ────────────────────────────────────────────────
    draw_panel_faces(ax, stack, tracking_deg)

    # ── Draw: panel flux overlay ───────────────────────────────────────────
    flux_norm = None
    if show_flux:
        flux_norm = draw_flux_overlay(ax, panel_pts, flux)

    # ── Draw: plume cone ──────────────────────────────────────────────────
    draw_plume_cone(ax, p_thruster, plume_dir,
                    half_angle_deg=THRUSTER.beam_divergence_half_angle,
                    length=min(reach_m * 0.7, 3.0))

    # ── Draw: thrust vector arrow ─────────────────────────────────────────
    arrow_len = min(reach_m * 0.30, 1.4)
    ax.quiver(*p_thruster, *(thrust_dir * arrow_len),
              color="#1A252F", linewidth=2.5, arrow_length_ratio=0.30)

    # ── Draw: arm links ───────────────────────────────────────────────────
    ax.plot3D(*zip(pivot,   p_elbow),   color=arm_clr, lw=5.5, solid_capstyle="round")
    ax.plot3D(*zip(p_elbow, p_thruster), color=arm_clr, lw=5.5, solid_capstyle="round")

    # Joint markers
    ax.scatter(*pivot,      s=140, c="#1A252F",  marker="s",
               edgecolors="w", linewidths=1.5, zorder=9)
    ax.scatter(*p_elbow,    s=95,  c=arm_clr,   marker="o",
               edgecolors="w", linewidths=1.5, zorder=9)
    ax.scatter(*p_thruster, s=230, c="#8E44AD",  marker="*",
               edgecolors="w", linewidths=1.0, zorder=9)

    # ── Draw: stack COG ───────────────────────────────────────────────────
    ax.scatter(*cog, s=210, c="#C0392B", marker="D",
               edgecolors="w", linewidths=1.5, zorder=10)

    # ── Axes formatting ────────────────────────────────────────────────────
    boundary = [
        [0, 0, 0],
        [ stack.client_bus_x / 2 + stack.panel_span_one_side, 0, stack.client_bus_z / 2],
        [-(stack.client_bus_x / 2 + stack.panel_span_one_side), 0, stack.client_bus_z / 2],
        list(serv_origin - np.array([0, 0, stack.servicer_bus_z / 2])),
        list(p_thruster),
    ]
    set_equal_aspect(ax, boundary)

    ax.set_xlabel("X  [m]", fontsize=9, labelpad=3)
    ax.set_ylabel("Y  [m]", fontsize=9, labelpad=3)
    ax.set_zlabel("Z  [m]  (+Z = anti-earth)", fontsize=9, labelpad=3)
    ax.set_title("GeometryEngine 3D Verification", fontsize=11,
                 fontweight="bold", pad=7)
    ax.view_init(elev=el_view, azim=az_view)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_els = [
        Patch(fc="#BDC3C7", ec="#5D6D7E", label="Client bus"),
        Patch(fc="#5DADE2", ec="#2471A3", label="Servicer bus  (Z−, LAR)"),
        Patch(fc="#76D7C4", ec="#17A589", label="LAR interface"),
        Patch(fc="#F39C12", ec="#935116", label="Solar panels  (±X)"),
        Line2D([0], [0], color="#E74C3C", lw=2, alpha=0.5, label="Plume cone"),
        Line2D([0], [0], color="#1A252F", lw=2, label="Thrust vector"),
        Line2D([0], [0], color="#27AE60", lw=4, label="Arm – OK"),
        Line2D([0], [0], color="#E67E22", lw=4, label="Arm – Infeasible"),
        Line2D([0], [0], color="#E74C3C", lw=4, label="Arm – Collision"),
    ]
    ax.legend(handles=legend_els, loc="upper right",
              fontsize=7.5, framealpha=0.90)

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
    if ik_result is not None:
        row(f"q0 yaw    = {np.degrees(q0):+6.1f}°")
        row(f"q1 shldr  = {np.degrees(q1):+6.1f}°")
        row(f"q2 elbow  = {np.degrees(q2):+6.1f}°")
    else:
        row("Target unreachable", color=bad_c)
    blank()

    section("FEASIBILITY")
    row(f"IK feasible :  {'YES' if feasible   else 'NO '}",
        color=ok_c if feasible else bad_c)
    row(f"Bus collision: {'YES' if collision   else 'NO '}",
        color=bad_c if collision else ok_c)
    blank()

    section("DISTANCES")
    row(f"|Thruster–COG|  = {dist_to_cog:.2f} m")
    if len(panel_pts) > 0:
        d_tp = np.linalg.norm(panel_pts - p_thruster[np.newaxis, :], axis=1)
        row(f"Min panel dist = {d_tp.min():.2f} m")
        row(f"Peak flux pt   = {flux.max():.3e}")

    y = 0.985
    dy_map = {True: 0.050, False: 0.045}
    ax_info.set_title("Status", fontsize=10, fontweight="bold", pad=4)
    for text, bold, color, size in lines:
        if text == "":
            y -= 0.018
            continue
        ax_info.text(0.04, y, text, transform=ax_info.transAxes,
                     fontsize=size, color=color, va="top",
                     fontweight="bold" if bold else "normal",
                     fontfamily="monospace")
        y -= dy_map[bold]

    # ── Inline flux colourbar ──────────────────────────────────────────────
    if show_flux and flux_norm is not None:
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1),
                                   cmap=plt.get_cmap("plasma"))
        sm.set_array([])
        try:
            ax_info.figure.colorbar(sm, ax=ax_info, orientation="horizontal",
                                    fraction=0.06, pad=0.04,
                                    label="Rel. flux (norm.)")
        except Exception:
            pass


# ─── Figure setup + main ──────────────────────────────────────────────────────

def main():
    save_mode = "--save" in sys.argv

    fig = plt.figure(figsize=(16, 9), facecolor="#F8F9FA")
    fig.suptitle(
        "Thruster Arm Geometry Verifier  ·  Servicer below client (Z−)  ·  "
        "3-DOF IK arm  ·  Drag to rotate",
        fontsize=11, fontweight="bold", y=0.998, color="#1A252F",
    )

    ax3d    = fig.add_axes([0.01, 0.23, 0.60, 0.73], projection="3d")
    ax3d.set_facecolor("#EBF5FB")
    ax_info = fig.add_axes([0.63, 0.23, 0.19, 0.73])

    # ── Initial view ─────────────────────────────────────────────────────
    # azim=-110 looks down the +Y axis, showing the arm clearly
    ax3d.view_init(elev=20, azim=-110)

    # ── Mutable shared state ───────────────────────────────────────────────
    # Default: arm horizontal at yaw=90° (+Y side), elev=0°.
    # Pivot is at Z≈−1.56 m (just below client bottom face at −1.5 m).
    # Horizontal arm stays at that Z, clearing the client bus → no collision.
    state = {
        "yaw_deg":       90.0,   # arm extends in +Y (orbit-normal)
        "elev_deg":       0.0,   # horizontal → arm sits just below client
        "reach_m":        2.62,
        "link_ratio":     0.75,
        "tracking_deg":   0.0,
        "elbow_up":      True,
        "show_flux":     False,
    }

    def _redraw():
        redraw(ax3d, ax_info, STACK, ARM, **state)
        fig.canvas.draw_idle()

    if not save_mode:
        # ── Sliders ────────────────────────────────────────────────────────
        ax_yaw   = fig.add_axes([0.08, 0.175, 0.52, 0.022])
        ax_elev  = fig.add_axes([0.08, 0.140, 0.52, 0.022])
        ax_reach = fig.add_axes([0.08, 0.105, 0.52, 0.022])
        ax_ratio = fig.add_axes([0.08, 0.070, 0.52, 0.022])
        ax_track = fig.add_axes([0.08, 0.035, 0.52, 0.022])

        sl_yaw   = Slider(ax_yaw,   "Shoulder Yaw  q₀ [°]",    0, 270,  valinit=90,   valstep=5)
        sl_elev  = Slider(ax_elev,  "Shoulder Elev  φ [°]",      0,  90,  valinit=0,   valstep=1)
        sl_reach = Slider(ax_reach, "Arm Reach  L₁+L₂ [m]",      1.0,  3.0, valinit=2.6, valstep=0.2)
        sl_ratio = Slider(ax_ratio, "Link Ratio  L₁/(L₁+L₂)",    0.5,  0.8, valinit=0.5, valstep=0.05)
        sl_track = Slider(ax_track, "Panel Track  α [°]",         -90,  90,  valinit=0.0, valstep=5)

        def _on_slider(_val):
            state["yaw_deg"]      = sl_yaw.val
            state["elev_deg"]     = sl_elev.val
            state["reach_m"]      = sl_reach.val
            state["link_ratio"]   = sl_ratio.val
            state["tracking_deg"] = sl_track.val
            _redraw()

        for sl in (sl_yaw, sl_elev, sl_reach, sl_ratio, sl_track):
            sl.on_changed(_on_slider)

        # ── Radio: elbow up/down ────────────────────────────────────────────
        ax_radio = fig.add_axes([0.840, 0.025, 0.075, 0.105])
        radio = RadioButtons(ax_radio, ("Elbow Up", "Elbow Down"), active=0)

        def _on_radio(label):
            state["elbow_up"] = (label == "Elbow Up")
            _redraw()

        radio.on_clicked(_on_radio)

        # ── Checkbox: show flux ────────────────────────────────────────────
        ax_check = fig.add_axes([0.920, 0.025, 0.068, 0.065])
        check = CheckButtons(ax_check, ["Flux"], [False])

        def _on_check(_label):
            state["show_flux"] = check.get_status()[0]
            _redraw()

        check.on_clicked(_on_check)

    # ── Initial draw ───────────────────────────────────────────────────────
    redraw(ax3d, ax_info, STACK, ARM, **state)

    if save_mode:
        out = "geometry_verification.png"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="#F8F9FA")
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
