#!/usr/bin/env python3
"""
Plasma Plume Impingement Parametric Study Pipeline
===================================================
Integrated framework for life-extension servicer missions.

Modules:
  1. Case Matrix Generator   – builds all parameter combinations + geometry
  2. Analytical Pre-Screener  – fast erosion estimates (inverse-square + sputter model)
  3. Heatmap Visualizer       – multi-dimensional interactive dashboard

Coordinate Frame Convention (LAR frame)
----------------------------------------
  Origin : LAR interface — the mechanical docking interface between servicer and client.
  +X     : North  (direction of North solar panel)
  +Y     : East   (direction of East antenna face)
  +Z     : Nadir  (toward Earth)
  −Z     : Anti-Earth (servicer side)

  The client bus extends in the +Z (nadir) direction from the LAR.
  The servicer hangs on the −Z (anti-earth) side of the LAR.
  All positions in this file are expressed in the LAR frame unless noted otherwise.

Author: karan.anand@infiniteorbits.io
"""

import numpy as np
import itertools
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Union
import warnings

# ---------------------------------------------------------------------------
# Rodrigues rotation helper (used by RoboticArmGeometry.forward_kinematics
# and arm_kinematics.arm_fk_transforms for the general serial chain)
# ---------------------------------------------------------------------------

def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """3×3 rotation matrix for rotating `angle` radians about unit `axis`."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    kx, ky, kz = k
    c, s = np.cos(angle), np.sin(angle)
    mc = 1.0 - c
    return np.array([
        [c + mc*kx*kx,     mc*kx*ky - s*kz,  mc*kx*kz + s*ky],
        [mc*ky*kx + s*kz,  c + mc*ky*ky,      mc*ky*kz - s*kx],
        [mc*kz*kx - s*ky,  mc*kz*ky + s*kx,   c + mc*kz*kz  ],
    ])


# ---------------------------------------------------------------------------
# 1.  DATA CLASSES – Parameter Definitions
# ---------------------------------------------------------------------------

@dataclass
class ThrusterParams:
    """Plasma thruster operating parameters."""
    name: str = "SPT-100-like"
    isp: float = 1500.0              # s
    discharge_voltage: float = 300.0  # V  (sets ion energy ~ e*V)
    mass_flow_rate: float = 5e-6     # kg/s
    beam_divergence_half_angle: float = 20.0   # deg (1/e² half-angle)
    plume_cosine_exponent: float = 10.0        # n in cos^n(θ) model
    propellant: str = "Xenon"
    thrust_N: float = 0.08            # N  (nominal)


@dataclass
class ArmGeometry:
    """Thruster arm configuration."""
    arm_length: float = 2.0           # m  (pivot to thruster exit)
    pivot_offset_x: float = 0.0       # m  from servicer geometric centre
    pivot_offset_y: float = 0.0       # m
    pivot_offset_z: float = 0.5       # m  (typically on anti-earth face)
    azimuth_deg: float = 0.0          # deg  arm azimuth in body frame
    elevation_deg: float = 0.0        # deg  arm elevation in body frame
    cant_angle_deg: float = 0.0       # deg  thrust vector cant at tip


@dataclass
class RoboticArmGeometry:
    """3-DOF thruster arm: J1 root yaw (q0) + J2 elbow (q1) + J3 wrist (q2).

    Arm reference frame (TA frame)
    --------------------------------
    Origin at the centre of the root-hinge bracket. Axes align with the
    servicer body frame (same orientation, different origin).
    TA-origin offset from servicer geometric centre: see ta_origin_{x,y,z}.

    Hinge / link data (stowed positions in TA frame, mm → m)
    ----------------------------------------------------------
    Hinge 1 (Root) : (0, 0, 251.75) mm        axis (0, 0, −1)
    Hinge 2 (Elbow): (−120, −1102.88, 151.75) mm  axis (1, 0, 0)
    Hinge 3 (Wrist): (−220, 416.3, 177.7) mm    axis (0, −0.3746, 0.9272)
    Nozzle         : (259.77, 349.53, 183.70) mm
    Nozzle direction in TA frame: (0.1455, 0.9189, 0.3666)

    Forward kinematics (general serial chain, Rodrigues formula)
    -------------------------------------------------------------
    R1 = Rodrigues(axis1, q0)
    R2 = R1 @ Rodrigues(axis2, q1)   (axis2 in link-1 body frame = TA at q=0)
    R3 = R2 @ Rodrigues(axis3, q2)   (axis3 in link-2 body frame = TA at q=0)

    In LAR frame (accounting for servicer yaw Rz_s):
    p_h2     = p_h1 + (Rz_s @ R1) @ d_h1h2
    p_h3     = p_h2 + (Rz_s @ R2) @ d_h2h3
    p_nozzle = p_h3 + (Rz_s @ R3) @ d_h3n
    t_hat    = −(Rz_s @ R3) @ n_hat_body
    """
    # ── Arm body geometry ──────────────────────────────────────────────────

    # TA-frame origin in servicer body frame [m]
    ta_origin_x: float = -0.12891
    ta_origin_y: float =  0.31619
    ta_origin_z: float =  0.35600

    # Hinge 1 (root joint) position in TA frame [m]
    h1_ta_x: float = 0.0
    h1_ta_y: float = 0.0
    h1_ta_z: float = 0.25175

    # Link body-frame vectors at stowed config (q=0) [m]
    # Derived from stowed positions: d_hXhY = p_hY_stowed − p_hX_stowed
    d_h1h2: np.ndarray = field(
        default_factory=lambda: np.array([-0.12000, -1.10288, -0.10000]))
    d_h2h3: np.ndarray = field(
        default_factory=lambda: np.array([-0.10000,  1.51918,  0.02595]))
    d_h3n:  np.ndarray = field(
        default_factory=lambda: np.array([ 0.47977, -0.06677,  0.00600]))

    # Joint rotation axes in TA body frame at stowed config
    axis1: np.ndarray = field(
        default_factory=lambda: np.array([ 0.0,     0.0,    -1.0   ]))
    axis2: np.ndarray = field(
        default_factory=lambda: np.array([ 1.0,     0.0,     0.0   ]))
    axis3: np.ndarray = field(
        default_factory=lambda: np.array([ 0.0,    -0.3746,  0.9272]))

    # Nozzle exit direction in link-3 body frame (= TA frame at q=0)
    n_hat_body: np.ndarray = field(
        default_factory=lambda: np.array([0.1455, 0.9189, 0.3666]))

    # ── Link scalar properties ─────────────────────────────────────────────
    # Lengths derived from |d_hXhY|: L1≈1.1139 m, L2≈1.5227 m, L3≈0.4844 m
    link1_length:   float = 1.1139
    link1_mass:     float = 10.0   # kg
    link2_length:   float = 1.5227
    link2_mass:     float = 10.0   # kg
    bracket_length: float = 0.4844
    bracket_mass:   float = 15.0   # kg

    # Per-link CoM offset along the link axis from the proximal joint [m]
    # None → geometric midpoint (0.5 * link_length)
    link1_com_offset:   Optional[float] = None
    link2_com_offset:   Optional[float] = None
    bracket_com_offset: Optional[float] = None

    # Per-link 3×3 inertia tensor in the link frame [kg·m²]
    # None → thin-rod approximation: I_yy = I_zz = m*L²/12, I_xx ≈ 0
    link1_inertia:   Optional[np.ndarray] = field(default=None, repr=False)
    link2_inertia:   Optional[np.ndarray] = field(default=None, repr=False)
    bracket_inertia: Optional[np.ndarray] = field(default=None, repr=False)

    # ── Pivot offset (Hinge 1 from servicer origin, used by legacy formulas) ──
    # Set to full offset so pivot_position() in geometry_visualizer works
    # without adding servicer_bus_z/2. Use arm_pivot_in_servicer_body() instead.
    pivot_offset_x: float = -0.12891   # m  from servicer geometric centre
    pivot_offset_y: float =  0.31619   # m
    pivot_offset_z: float =  0.60775   # m  (= ta_origin_z + h1_ta_z)

    # ── Joint angle limits [deg] ───────────────────────────────────────────
    q0_min_deg: float =   0.0    # J1 root yaw
    q0_max_deg: float = 270.0
    q1_min_deg: float =   0.0    # J2 elbow
    q1_max_deg: float = 235.0
    q2_min_deg: float = -36.0    # J3 wrist
    q2_max_deg: float =  99.0

    # Desired shoulder yaw (primary sweep parameter, set before IK solve)
    shoulder_yaw_deg: float = 0.0
    # Elbow configuration preference
    elbow_up: bool = True

    # ------------------------------------------------------------------
    # Pivot helper
    # ------------------------------------------------------------------

    def arm_pivot_in_servicer_body(self) -> np.ndarray:
        """Hinge 1 (arm root joint) position in servicer body frame [m].

        = TA-origin offset + Hinge-1 offset in TA frame.
        Use this instead of the pivot_offset_* scalars to avoid the
        servicer_bus_z/2 ambiguity in legacy formulas.
        """
        return np.array([
            self.ta_origin_x + self.h1_ta_x,
            self.ta_origin_y + self.h1_ta_y,
            self.ta_origin_z + self.h1_ta_z,
        ])

    # ------------------------------------------------------------------
    # Arm summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def stowed_joint_angles_deg() -> Tuple[float, float, float]:
        """Return the stowed (home) joint angles: (q0, q1, q2) = (0°, 0°, 0°)."""
        return (0.0, 0.0, 0.0)

    def arm_reach(self) -> float:
        """Total arm reach fully extended (L1 + L2 + bracket) [m]."""
        return self.link1_length + self.link2_length + self.bracket_length

    def arm_mass(self) -> float:
        """Total arm mass [kg]."""
        return self.link1_mass + self.link2_mass + self.bracket_mass

    def effective_link1_com(self) -> float:
        """CoM offset along link 1 axis from Hinge 1 [m]."""
        return self.link1_com_offset if self.link1_com_offset is not None \
               else 0.5 * self.link1_length

    def effective_link2_com(self) -> float:
        """CoM offset along link 2 axis from Hinge 2 [m]."""
        return self.link2_com_offset if self.link2_com_offset is not None \
               else 0.5 * self.link2_length

    def effective_bracket_com(self) -> float:
        """CoM offset along bracket axis from Hinge 3 [m]."""
        return self.bracket_com_offset if self.bracket_com_offset is not None \
               else 0.5 * self.bracket_length

    def effective_link1_inertia(self) -> np.ndarray:
        """3×3 inertia tensor for link 1 in its link frame [kg·m²]."""
        if self.link1_inertia is not None:
            return self.link1_inertia
        I_t = self.link1_mass * self.link1_length ** 2 / 12.0
        return np.diag([0.0, I_t, I_t])

    def effective_link2_inertia(self) -> np.ndarray:
        """3×3 inertia tensor for link 2 in its link frame [kg·m²]."""
        if self.link2_inertia is not None:
            return self.link2_inertia
        I_t = self.link2_mass * self.link2_length ** 2 / 12.0
        return np.diag([0.0, I_t, I_t])

    def effective_bracket_inertia(self) -> np.ndarray:
        """3×3 inertia tensor for bracket in its link frame [kg·m²]."""
        if self.bracket_inertia is not None:
            return self.bracket_inertia
        I_t = self.bracket_mass * self.bracket_length ** 2 / 12.0
        return np.diag([0.0, I_t, I_t])

    def forward_kinematics(self, pivot: np.ndarray,
                           q0: float, q1: float, q2: float,
                           servicer_yaw_deg: float = 0.0,
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (p_h2/elbow, p_h3/wrist, p_nozzle) given joint angles [rad].

        General serial-chain FK using the Rodrigues formula. `pivot` is Hinge 1
        in the LAR frame (output of compute_pivot / arm_pivot_in_servicer_body).
        `servicer_yaw_deg` rotates TA-frame body vectors into LAR.

        Returns positions in LAR frame.
        """
        c_s, s_s = np.cos(np.radians(servicer_yaw_deg)), np.sin(np.radians(servicer_yaw_deg))
        Rz_s = np.array([[c_s, -s_s, 0.], [s_s, c_s, 0.], [0., 0., 1.]])

        R1 = _rodrigues(self.axis1, q0)
        R2 = R1 @ _rodrigues(self.axis2, q1)
        R3 = R2 @ _rodrigues(self.axis3, q2)

        # Rotate cumulative FK matrices into LAR frame
        CR1 = Rz_s @ R1
        CR2 = Rz_s @ R2
        CR3 = Rz_s @ R3

        p_h2     = pivot + CR1 @ self.d_h1h2
        p_h3     = p_h2  + CR2 @ self.d_h2h3
        p_nozzle = p_h3  + CR3 @ self.d_h3n
        return p_h2, p_h3, p_nozzle

    def inverse_kinematics(self, pivot: np.ndarray, target: np.ndarray,
                           elbow_up: bool = True
                           ) -> Optional[Tuple[float, float, float]]:
        """Closed-form IK for yaw-pitch-pitch arm targeting the thruster exit point.

        Because link 1 is always horizontal, the elbow is fixed at
        (pivot + L1*u_rad). The problem reduces to a planar 2R IK for
        link 2 (L2) and bracket (L3) from the elbow to the target.

        Returns (q0, q1, q2) in radians, or None if target is unreachable.
        """
        L1 = self.link1_length
        L2 = self.link2_length
        L3 = self.bracket_length
        dv = target - pivot

        # J1: shoulder yaw from horizontal projection of target
        q0 = np.arctan2(dv[1], dv[0])

        # Horizontal reach and vertical offset from pivot to target
        r = np.sqrt(dv[0] ** 2 + dv[1] ** 2)
        z = dv[2]

        # Elbow is always L1 horizontally from pivot; subtract to get
        # the reach from elbow to target in the pitched plane.
        r2 = r - L1

        # Cosine-rule for wrist (J3) angle between link 2 and bracket
        dist_sq = r2 ** 2 + z ** 2
        cos_q2 = (dist_sq - L2 ** 2 - L3 ** 2) / (2.0 * L2 * L3)

        if abs(cos_q2) > 1.0:
            return None  # target out of reach

        cos_q2 = np.clip(cos_q2, -1.0, 1.0)
        q2 = np.arccos(cos_q2) if elbow_up else -np.arccos(cos_q2)

        # J2: elbow pitch — angle of link 2 relative to horizontal
        alpha = np.arctan2(z, r2)
        beta  = np.arctan2(L3 * np.sin(q2), L2 + L3 * np.cos(q2))
        q1 = alpha - beta

        return q0, q1, q2

    def within_joint_limits(self, q0_rad: float, q1_rad: float,
                            q2_rad: float) -> bool:
        """Return True if all joint angles are within configured limits."""
        q0d = np.degrees(q0_rad)
        q1d = np.degrees(q1_rad)
        q2d = np.degrees(q2_rad)
        return (self.q0_min_deg <= q0d <= self.q0_max_deg and
                self.q1_min_deg <= q1d <= self.q1_max_deg and
                self.q2_min_deg <= q2d <= self.q2_max_deg)


@dataclass
class StackConfig:
    """Combined servicer + client satellite stack."""
    # Servicer
    servicer_mass: float = 735.0        # kg
    servicer_bus_x: float = 0.90        # m  (along velocity)
    servicer_bus_y: float = 1.52        # m  (along orbit-normal / N-S)
    servicer_bus_z: float = 0.80         # m  (along nadir)

    # Client (telecom satellite)
    client_mass: float = 2500.0         # kg
    client_bus_x: float = 2.5           # m
    client_bus_y: float = 3.0           # m
    client_bus_z: float = 5.0           # m

    # Solar panels (client) – modelled as flat rectangles
    panel_span_one_side: float = 16.0   # m  from bus edge to panel tip
    panel_width: float = 2.2            # m  (along orbit-normal)
    panel_hinge_offset_y: float = 1.5   # m  offset of hinge line from bus centre-y
    panel_cant_angle_deg: float = 0.0   # deg  cant about hinge axis

    # Servicer solar panels
    servicer_panel_span: float = 5.0    # m  panel span per side from inner edge
    servicer_panel_gap:  float = 1.4    # m  gap between servicer bus face and panel inner edge

    # Docking interface
    lar_offset_z: float = 0.05          # m  LAR hardware standoff height
    dock_offset_z: float =-0.8          # m  servicer offset along Z from client centre
    dock_offset_x: float = 0.0          # m

    # Antenna reflectors – 4-dish model (spec §4)
    # East face (+Y): E1 (−X side) and E2 (+X side), diameter 2.2 m
    # West face (−Y): W1 (−X side) and W2 (+X side), diameter 2.5 m
    # All dishes have their aperture normal pointing +Z (nadir-facing).
    antenna_diameter_east: float = 2.5  # m  E1, E2
    antenna_diameter_west: float = 2.3  # m  W1, W2
    antenna_x_separation:  float = 6.0  # m  centre-to-centre along X
    antenna_z_offset:      float = -1.0  # m  offset from bus centre; negative = toward servicer
    antenna_mass:          float = 20.0 # kg per dish (all four identical)

    def antenna_centers_in_lar_frame(self) -> Dict[str, np.ndarray]:
        """Return the four dish centres in LAR frame.

        Naming convention  →  face  |  X offset
        ─────────────────────────────────────────
        E1                →  +Y    |  −x_sep/2
        E2                →  +Y    |  +x_sep/2
        W1                →  −Y    |  −x_sep/2
        W2                →  −Y    |  +x_sep/2

        Z coordinate: client bus centre + antenna_z_offset
          = client_bus_z/2 + antenna_z_offset
          Negative offset places dishes toward the servicer/LAR side.
        Y coordinate: ±client_bus_y/2 (flush with bus face)
        """
        z = self.client_bus_z / 2.0 + self.antenna_z_offset
        half_sep = self.antenna_x_separation / 2.0
        y_east =  self.client_bus_y / 2.0
        y_west = -self.client_bus_y / 2.0
        return {
            "E1": np.array([-half_sep,  y_east, z]),
            "E2": np.array([ half_sep,  y_east, z]),
            "W1": np.array([-half_sep,  y_west, z]),
            "W2": np.array([ half_sep,  y_west, z]),
        }

    def servicer_origin_in_lar_frame(self) -> np.ndarray:
        """Servicer geometric centre in LAR frame.
        The servicer sits on the anti-earth side (−Z) of the LAR interface.
        Distance from LAR to servicer centre = lar_offset_z + servicer_bus_z/2.
        """
        return np.array([
            self.dock_offset_x,
            0.0,
            -(self.lar_offset_z + self.servicer_bus_z / 2.0) + self.dock_offset_z
        ])

    def client_origin_in_lar_frame(self) -> np.ndarray:
        """Client bus centre in LAR frame.
        Client extends in +Z (nadir) from LAR; centre is half a bus-height below LAR.
        """
        return np.array([0.0, 0.0, self.client_bus_z / 2.0])

    def stack_cog(self) -> np.ndarray:
        """Combined centre of gravity in LAR frame."""
        servicer_cg = self.servicer_origin_in_lar_frame()
        client_cg = self.client_origin_in_lar_frame()
        total_mass = self.servicer_mass + self.client_mass
        return (self.client_mass * client_cg + self.servicer_mass * servicer_cg) / total_mass


@dataclass
class OperationalParams:
    """Mission operational parameters."""
    firing_duration_s: float = 25000.0     # s per manoeuvre
    firings_per_day: float = 1.0
    mission_duration_years: float = 5.0
    manoeuvre_type: str = "NSSK"          # NSSK or EWSK
    panel_sun_tracking_angle_deg: float = 0.0  # panel rotation about hinge during firing


@dataclass
class MaterialParams:
    """Target surface material properties."""
    name: str = "Silver_interconnect"
    thickness_um: float = 15.0            # µm
    density_kg_m3: float = 10490.0        # kg/m³  (Ag)
    atomic_mass_amu: float = 107.87       # amu
    # Sputter yield coefficients (Yamamura-type fit for Xe+ on Ag)
    # Y(E, θ) = Y_normal(E) * f(θ)
    # Y_normal ~ a*(E - E_th)^b  for E > E_th
    sputter_yield_a: float = 0.059        # atoms/ion coefficient
    sputter_yield_b: float = 0.7          # energy exponent
    sputter_threshold_eV: float = 15.0    # threshold energy
    # Angular dependence: f(θ) = cos(θ)^(-c) * exp(-d*(1/cos(θ) - 1))
    angular_coeff_c: float = -1.3
    angular_coeff_d: float = 0.4


# ---------------------------------------------------------------------------
# 2.  GEOMETRY ENGINE
# ---------------------------------------------------------------------------

def _segment_intersects_aabb(P0: np.ndarray, P1: np.ndarray,
                              box_min: np.ndarray, box_max: np.ndarray) -> bool:
    """Return True if line segment P0→P1 intersects the axis-aligned box.

    Uses the slab method: for each axis find the t-interval where the ray
    is inside that slab, then intersect all three intervals.
    """
    d = P1 - P0
    t_min, t_max = 0.0, 1.0

    for i in range(3):
        if abs(d[i]) < 1e-12:
            # Segment is parallel to slab; check if inside
            if P0[i] < box_min[i] or P0[i] > box_max[i]:
                return False
        else:
            t1 = (box_min[i] - P0[i]) / d[i]
            t2 = (box_max[i] - P0[i]) / d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False

    return True


def _segment_intersects_obb(P0: np.ndarray, P1: np.ndarray,
                              center: np.ndarray, half_extents: np.ndarray,
                              R_world_to_body: np.ndarray) -> bool:
    """Return True if segment P0→P1 intersects an oriented bounding box (OBB).

    Transforms the segment endpoints into the box body frame, then delegates
    to the axis-aligned slab test.

    Parameters
    ----------
    center          : OBB centre in world frame
    half_extents    : (3,) half-widths [hx, hy, hz] in body frame
    R_world_to_body : 3×3 rotation matrix  (world → body)
    """
    P0b = R_world_to_body @ (P0 - center)
    P1b = R_world_to_body @ (P1 - center)
    return _segment_intersects_aabb(P0b, P1b, -half_extents, half_extents)


def _segment_intersects_disc(P0: np.ndarray, P1: np.ndarray,
                               center: np.ndarray, radius: float,
                               thickness: float = 0.15) -> bool:
    """Return True if segment P0→P1 intersects a flat disc (thin cylinder, +Z axis).

    The disc is modelled as a slab of *thickness* centred at *center[2]* in Z,
    with radius *radius* in XY.

    Algorithm: find the t-range where the segment is within the Z slab, then
    find the t in that range that minimises distance to the disc centre in XY.
    """
    z_lo = center[2] - thickness / 2.0
    z_hi = center[2] + thickness / 2.0
    dz   = P1[2] - P0[2]

    if abs(dz) < 1e-12:
        if P0[2] < z_lo or P0[2] > z_hi:
            return False
        t_lo, t_hi = 0.0, 1.0
    else:
        t1 = (z_lo - P0[2]) / dz
        t2 = (z_hi - P0[2]) / dz
        t_lo = max(0.0, min(t1, t2))
        t_hi = min(1.0, max(t1, t2))
        if t_lo > t_hi:
            return False

    # t that minimises XY distance to disc centre within the slab interval
    dx = P1[0] - P0[0]
    dy = P1[1] - P0[1]
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        t_best = t_lo
    else:
        t_best = ((center[0] - P0[0]) * dx + (center[1] - P0[1]) * dy) / denom
        t_best = max(t_lo, min(t_hi, t_best))

    for t in (t_best, t_lo, t_hi):
        pt = P0 + t * (P1 - P0)
        if (pt[0] - center[0]) ** 2 + (pt[1] - center[1]) ** 2 <= radius ** 2:
            return True
    return False


def arm_has_collision(pivot: np.ndarray, p_elbow: np.ndarray,
                       p_wrist: np.ndarray, p_thruster: np.ndarray,
                       stack: "StackConfig",
                       servicer_yaw_deg: float = 0.0) -> bool:
    """Return True if any arm segment collides with any obstacle in the stack.

    Obstacles checked
    -----------------
    1. Client bus           – axis-aligned box (LAR frame)
    2. Servicer bus         – oriented box (yawed by servicer_yaw_deg)
    3. Servicer solar panels – two oriented rectangular slabs (same yaw)
    4. Antenna reflectors   – four flat discs (axis-aligned, +Z normal)

    Parameters
    ----------
    pivot, p_elbow, p_wrist, p_thruster : arm joint positions in LAR frame
    stack            : StackConfig geometry
    servicer_yaw_deg : servicer body yaw relative to LAR frame [deg]
    """
    segments = [(pivot, p_elbow), (p_elbow, p_wrist), (p_wrist, p_thruster)]

    # ── 1. Client bus (AABB) ─────────────────────────────────────────────────
    bx = stack.client_bus_x / 2.0
    by = stack.client_bus_y / 2.0
    client_min = np.array([-bx, -by, 0.0])
    client_max = np.array([ bx,  by, stack.client_bus_z])
    for P0, P1 in segments:
        if _segment_intersects_aabb(P0, P1, client_min, client_max):
            return True

    # Precompute servicer yaw rotation matrices once
    yaw_rad = np.radians(servicer_yaw_deg)
    c, s    = np.cos(yaw_rad), np.sin(yaw_rad)
    Rz_fwd  = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # body → LAR
    Rz_inv  = Rz_fwd.T                                                   # LAR  → body

    serv_origin = stack.servicer_origin_in_lar_frame()

    # ── 2. Servicer bus (OBB) ─────────────────────────────────────────────────
    serv_half = np.array([stack.servicer_bus_x / 2.0,
                          stack.servicer_bus_y / 2.0,
                          stack.servicer_bus_z / 2.0])
    for P0, P1 in segments:
        if _segment_intersects_obb(P0, P1, serv_origin, serv_half, Rz_inv):
            return True

    # ── 3. Servicer solar panels (two OBBs, same yaw) ─────────────────────────
    half_x  = stack.servicer_bus_x / 2.0
    half_y  = stack.servicer_bus_y / 2.0
    gap     = stack.servicer_panel_gap
    span    = stack.servicer_panel_span
    p_thick = 0.10   # slab thickness [m]

    for sign in (+1, -1):
        # Panel centre in servicer body frame (z = 0 → servicer mid-plane)
        y_ctr_body  = sign * (half_y + gap + span / 2.0)
        pc_body     = np.array([0.0, y_ctr_body, 0.0])
        # Panel centre in LAR frame
        pc_lar      = serv_origin + Rz_fwd @ pc_body
        panel_half  = np.array([half_x, span / 2.0, p_thick / 2.0])
        for P0, P1 in segments:
            if _segment_intersects_obb(P0, P1, pc_lar, panel_half, Rz_inv):
                return True

    # ── 4. Antenna dishes (disc collision) ────────────────────────────────────
    ant_centers = stack.antenna_centers_in_lar_frame()
    ant_radii   = {
        "E1": stack.antenna_diameter_east / 2.0,
        "E2": stack.antenna_diameter_east / 2.0,
        "W1": stack.antenna_diameter_west / 2.0,
        "W2": stack.antenna_diameter_west / 2.0,
    }
    for name, center in ant_centers.items():
        r = ant_radii[name]
        for P0, P1 in segments:
            if _segment_intersects_disc(P0, P1, center, r):
                return True

    return False


class GeometryEngine:
    """Computes plume origin, direction, distances and angles to surfaces."""

    def __init__(self, arm: "Union[ArmGeometry, RoboticArmGeometry]", stack: StackConfig):
        self.arm = arm
        self.stack = stack

    def thruster_position(self) -> np.ndarray:
        """Thruster exit plane centre in client body frame."""
        if isinstance(self.arm, RoboticArmGeometry):
            return self._thruster_position_ik()
        else:
            return self._thruster_position_single_link()

    def _thruster_position_single_link(self) -> np.ndarray:
        """Legacy single-link arm (ArmGeometry)."""
        servicer_origin = self.stack.servicer_origin_in_lar_frame()
        pivot = np.array([
            servicer_origin[0] + self.arm.pivot_offset_x,
            servicer_origin[1] + self.arm.pivot_offset_y,
            servicer_origin[2] + self.stack.servicer_bus_z / 2.0 + self.arm.pivot_offset_z
        ])
        az = np.radians(self.arm.azimuth_deg)
        el = np.radians(self.arm.elevation_deg)
        arm_dir = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ])
        return pivot + self.arm.arm_length * arm_dir

    def _pivot_position(self) -> np.ndarray:
        """Pivot (Hinge 1) position in LAR frame."""
        servicer_origin = self.stack.servicer_origin_in_lar_frame()
        return servicer_origin + self.arm.arm_pivot_in_servicer_body()

    def _thruster_position_ik(self) -> np.ndarray:
        """Robotic arm: IK to find joint angles, then FK to get thruster position.

        Target placement strategy:
          The arm extends horizontally in the shoulder_yaw_deg azimuth direction at total reach (L1+L2).  This keeps the arm clear of the client bus (which sits above the pivot) and makes shoulder_yaw_deg the primary control for where the thruster is placed around the spacecraft.  The thrust direction (computed separately in thrust_direction()) then aims the plume toward the stack COG.

          IK still resolves the elbow angle consistent with the elbow_up flag, which is relevant when the arm geometry has unequal link lengths.  For a horizontal target at reach R the IK degenerates to q1=0, q2=0 (fully extended), but link_ratio still affects the elbow height during collision checking (arm_link_positions).

        If IK fails (out of reach or joint limits exceeded), the straight horizontal fallback is returned and ik_feasible is False.
        """
        arm = self.arm
        pivot = self._pivot_position()

        # Desired azimuth from shoulder_yaw_deg
        q0_desired = np.radians(arm.shoulder_yaw_deg)
        R = arm.link1_length + arm.link2_length

        # Target: R from pivot horizontally in the q0 azimuth direction.
        # Staying at the pivot Z keeps the arm below the client bus (pivot is on
        # the servicer Z+ face, which is just below the client bottom face).
        target = pivot + np.array([
            np.cos(q0_desired) * R,
            np.sin(q0_desired) * R,
            0.0
        ])

        # Solve IK
        ik_result = arm.inverse_kinematics(pivot, target, elbow_up=arm.elbow_up)

        self._ik_feasible = False
        self._ik_joint_angles = None
        self._pivot_pos = pivot

        if ik_result is not None:
            q0, q1, q2 = ik_result
            if arm.within_joint_limits(q0, q1, q2):
                self._ik_feasible = True
                self._ik_joint_angles = (q0, q1, q2)
                _, _, p_thruster = arm.forward_kinematics(pivot, q0, q1, q2)
                return p_thruster

        # Fallback: place thruster at target directly (joint-limit violation flagged)
        self._ik_joint_angles = ik_result  # may be None
        return target

    def arm_link_positions(self) -> Tuple[np.ndarray, Optional[np.ndarray],
                                          Optional[np.ndarray], np.ndarray]:
        """Return (pivot, p_elbow, p_wrist, p_thruster) in LAR frame.

        For ArmGeometry (single link), p_elbow and p_wrist are None.
        For RoboticArmGeometry, all four points are returned.
        """
        if not isinstance(self.arm, RoboticArmGeometry):
            p_thruster = self.thruster_position()
            return self._pivot_position(), None, None, p_thruster

        # Ensure _thruster_position_ik has been called
        p_thruster = self.thruster_position()
        pivot = self._pivot_pos
        if self._ik_joint_angles is not None:
            q0, q1, q2 = self._ik_joint_angles
            p_elbow, p_wrist, _ = self.arm.forward_kinematics(pivot, q0, q1, q2)
        else:
            p_elbow = pivot   # degenerate: IK failed
            p_wrist = pivot
        return pivot, p_elbow, p_wrist, p_thruster

    def check_arm_collision_with_client_bus(self) -> bool:
        """Return True if any arm link segment passes through the client bus box.

        Uses AABB slab intersection test for each link segment.
        In LAR frame (+Z = nadir) the client bus occupies:
          X in [-bx, +bx],  Y in [-by, +by],  Z in [0, client_bus_z]
        """
        stack = self.stack
        bx = stack.client_bus_x / 2.0
        by = stack.client_bus_y / 2.0
        box_min = np.array([-bx, -by, 0.0])
        box_max = np.array([ bx,  by, stack.client_bus_z])

        pivot, p_elbow, p_wrist, p_thruster = self.arm_link_positions()
        # Build segment chain: pivot→elbow→wrist→thruster (skip None waypoints)
        waypoints = [p for p in [pivot, p_elbow, p_wrist, p_thruster] if p is not None]
        segments = list(zip(waypoints[:-1], waypoints[1:]))

        for (P0, P1) in segments:
            if _segment_intersects_aabb(P0, P1, box_min, box_max):
                return True
        return False

    def ik_feasible(self) -> bool:
        """Return True if the last IK solve was within joint limits."""
        if not isinstance(self.arm, RoboticArmGeometry):
            return True  # single-link arm is always "feasible"
        # Trigger thruster_position if not yet computed
        if not hasattr(self, '_ik_feasible'):
            self.thruster_position()
        return getattr(self, '_ik_feasible', False)

    def cog_ik(self, target_cog_lar: np.ndarray,
                pos_mask: np.ndarray = None,
                error_weights: np.ndarray = None,
                damping_gain: float = 1e-3,
                position_tol: float = 1e-3,
                max_iters: int = 100,
                enforce_limits: bool = True
                ) -> Tuple[bool, np.ndarray]:
        """Find joint angles that place the arm CoG at target_cog_lar (LAR frame).

        Uses damped least-squares iteration (Chan's method) over the centroidal
        Jacobian.  thruster_position() must have been called first so that
        _ik_joint_angles and _pivot_pos are populated (used as the initial guess).

        pos_mask lets you constrain only a subset of CoG axes.  For example,
        when aligning with a thrust line parallel to +X (NSSK), set
        pos_mask=[0, 1, 1] to constrain only Y and Z.

        On success, _ik_joint_angles is updated so that subsequent calls to
        stack_cog_with_arm(), thrust_direction(), and thrust_metrics() all
        reflect the new configuration.

        Returns
        -------
        success : bool
        q_sol   : (3,) solution joint angles [rad]
        """
        from arm_kinematics import arm_cog_ik

        if not isinstance(self.arm, RoboticArmGeometry):
            raise TypeError("cog_ik requires a RoboticArmGeometry arm.")

        # Ensure pivot and initial angles are available
        if not hasattr(self, '_pivot_pos'):
            self.thruster_position()

        q_init = np.array(self._ik_joint_angles) \
                 if self._ik_joint_angles is not None \
                 else np.zeros(3)

        success, q_sol, _ = arm_cog_ik(
            self.arm, self._pivot_pos, q_init,
            target_cog_lar,
            pos_mask=pos_mask,
            error_weights=error_weights,
            damping_gain=damping_gain,
            position_tol=position_tol,
            max_iters=max_iters,
            enforce_limits=enforce_limits,
        )

        if success:
            self._ik_joint_angles = tuple(q_sol)

        return success, q_sol

    def stack_cog_with_arm(self) -> np.ndarray:
        """Stack CoG including arm link masses in LAR frame.

        Uses inertia-weighted CoM positions from arm_kinematics.arm_cog_position(),
        which honours per-link CoM offsets and inertia tensors when set on
        RoboticArmGeometry; otherwise falls back to thin-rod (midpoint) defaults.

        Must be called after thruster_position() so that _ik_joint_angles
        and _pivot_pos are cached.  Falls back to stack.stack_cog() (client +
        servicer only) if the arm is not a RoboticArmGeometry or IK has not
        yet been solved.
        """
        from arm_kinematics import arm_cog_position

        base_cog   = self.stack.stack_cog()
        stack_mass = self.stack.client_mass + self.stack.servicer_mass

        if not isinstance(self.arm, RoboticArmGeometry):
            return base_cog

        angles = getattr(self, '_ik_joint_angles', None)
        if angles is None:
            return base_cog

        q       = np.array(angles)
        arm_cog = arm_cog_position(self.arm, self._pivot_pos, q)
        arm_mass = self.arm.arm_mass()

        return (stack_mass * base_cog + arm_mass * arm_cog) / (stack_mass + arm_mass)

    def thrust_direction(self) -> np.ndarray:
        """Unit thrust vector in LAR frame.

        The thrust direction is constrained to point *through* the full stack
        CoG (including arm mass) to minimise disturbance torques (ideal case).
        If cant_angle != 0, a small offset from ideal is applied.
        """
        t_pos = self.thruster_position()
        cog = self.stack_cog_with_arm()
        ideal_dir = cog - t_pos
        ideal_dir = ideal_dir / np.linalg.norm(ideal_dir)

        # Apply cant angle as a rotation about an axis perpendicular to
        # both the ideal direction and the arm axis
        if abs(getattr(self.arm, 'cant_angle_deg', 0.0)) > 0.01:
            cant = np.radians(getattr(self.arm, 'cant_angle_deg', 0.0))
            # Simple rotation: tilt thrust vector away from ideal by cant angle
            # in the plane containing ideal_dir and Z-hat
            perp = np.cross(ideal_dir, np.array([0, 0, 1]))
            if np.linalg.norm(perp) < 1e-9:
                perp = np.cross(ideal_dir, np.array([1, 0, 0]))
            perp = perp / np.linalg.norm(perp)
            thrust_dir = ideal_dir * np.cos(cant) + perp * np.sin(cant)
            thrust_dir = thrust_dir / np.linalg.norm(thrust_dir)
        else:
            thrust_dir = ideal_dir

        # Plume goes opposite to thrust direction (exhaust)
        plume_dir = -thrust_dir
        return plume_dir

    def panel_grid_points(self, n_spanwise: int = 50, n_chordwise: int = 10,
                          sun_tracking_angle_deg: float = 0.0) -> np.ndarray:
        """Generate grid points on both solar panels (client).

        Returns array of shape (2 * n_spanwise * n_chordwise, 3).
        Panel normals point along +Z (sun-facing) when tracking angle = 0.
        """
        points = []
        hw = self.stack.panel_width / 2.0

        for side in [+1, -1]:  # +X and -X panels
            base_x = side * self.stack.client_bus_x / 2.0
            for i in range(n_spanwise):
                span_frac = (i + 0.5) / n_spanwise
                xi = base_x + side * span_frac * self.stack.panel_span_one_side
                for j in range(n_chordwise):
                    chord_frac = (j + 0.5) / n_chordwise - 0.5
                    yi = self.stack.panel_hinge_offset_y + chord_frac * self.stack.panel_width

                    # Panel Z position: midplane of client bus in LAR frame
                    # (+Z = nadir; client spans Z = 0 to Z = client_bus_z)
                    zi_base = self.stack.client_bus_z / 2.0

                    # Apply sun-tracking rotation about hinge (X-axis rotation)
                    track = np.radians(sun_tracking_angle_deg)
                    cant = np.radians(self.stack.panel_cant_angle_deg)
                    total_angle = track + cant
                    yi_rot = yi * np.cos(total_angle)
                    zi = zi_base + yi * np.sin(total_angle)

                    points.append([xi, yi_rot, zi])

        return np.array(points)

    def panel_normal(self, sun_tracking_angle_deg: float = 0.0) -> np.ndarray:
        """Approximate panel surface normal (sun-facing / anti-earth side).
        In LAR frame, anti-earth = −Z, so the untracked normal is [0, 0, −1].
        """
        track = np.radians(sun_tracking_angle_deg)
        cant = np.radians(self.stack.panel_cant_angle_deg)
        total = track + cant
        # −cos component: points toward −Z (anti-earth) when total = 0
        normal = np.array([0.0, -np.sin(total), -np.cos(total)])
        return normal / np.linalg.norm(normal)

    def compute_flux_geometry(self, sun_tracking_angle_deg: float = 0.0,
                              n_spanwise: int = 50, n_chordwise: int = 10
                              ) -> Dict[str, np.ndarray]:
        """For every panel grid point, compute:
           - distance from thruster
           - off-axis angle from plume centreline
           - incidence angle on panel surface
        """
        t_pos = self.thruster_position()
        plume_dir = self.thrust_direction()
        panel_pts = self.panel_grid_points(n_spanwise, n_chordwise,
                                           sun_tracking_angle_deg)
        panel_norm = self.panel_normal(sun_tracking_angle_deg)

        # Vector from thruster to each panel point
        dvec = panel_pts - t_pos[np.newaxis, :]
        distances = np.linalg.norm(dvec, axis=1)
        dvec_unit = dvec / distances[:, np.newaxis]

        # Off-axis angle from plume centreline
        cos_offaxis = np.clip(np.dot(dvec_unit, plume_dir), -1, 1)
        offaxis_angles_deg = np.degrees(np.arccos(cos_offaxis))

        # Incidence angle on panel surface
        cos_incidence = np.abs(np.dot(dvec_unit, panel_norm))
        incidence_angles_deg = np.degrees(np.arccos(np.clip(cos_incidence, 0, 1)))

        return {
            "panel_points": panel_pts,
            "distances_m": distances,
            "offaxis_angles_deg": offaxis_angles_deg,
            "incidence_angles_deg": incidence_angles_deg,
            "thruster_pos": t_pos,
            "plume_dir": plume_dir,
            "min_distance_m": np.min(distances),
            "min_distance_point": panel_pts[np.argmin(distances)],
        }

    def compute_antenna_flux_geometry(self) -> Dict[str, Dict]:
        """Compute plume geometry for all four antenna dishes.

        Each dish is treated as a point target at its centre.  The dish aperture
        normal is +Z (nadir-facing), so the incidence angle is computed against
        that fixed normal.

        Returns a dict keyed by dish name ("E1", "E2", "W1", "W2"), each value
        being a sub-dict with:
          distance_m        – thruster-to-dish-centre distance
          offaxis_deg       – off-axis angle from plume centreline
          incidence_deg     – angle between plume ray and dish normal (+Z)
          diameter_m        – dish diameter
        """
        t_pos    = self.thruster_position()
        plume_dir = self.thrust_direction()
        dish_normal = np.array([0.0, 0.0, 1.0])  # nadir-facing aperture

        centers = self.stack.antenna_centers_in_lar_frame()
        diameters = {
            "E1": self.stack.antenna_diameter_east,
            "E2": self.stack.antenna_diameter_east,
            "W1": self.stack.antenna_diameter_west,
            "W2": self.stack.antenna_diameter_west,
        }

        result = {}
        for name, center in centers.items():
            dvec = center - t_pos
            dist = float(np.linalg.norm(dvec))
            if dist < 1e-6:
                result[name] = {"distance_m": 0.0, "offaxis_deg": 0.0,
                                "incidence_deg": 0.0, "diameter_m": diameters[name]}
                continue
            dvec_unit = dvec / dist
            cos_offaxis = float(np.clip(np.dot(dvec_unit, plume_dir), -1.0, 1.0))
            offaxis_deg = float(np.degrees(np.arccos(cos_offaxis)))
            # Incidence on dish surface: angle between incoming ray and aperture normal
            cos_inc = float(np.abs(np.dot(dvec_unit, dish_normal)))
            incidence_deg = float(np.degrees(np.arccos(np.clip(cos_inc, 0.0, 1.0))))
            result[name] = {
                "distance_m":    dist,
                "offaxis_deg":   offaxis_deg,
                "incidence_deg": incidence_deg,
                "diameter_m":    diameters[name],
            }
        return result

    def thrust_metrics(self, thrust_N: float = 1.0) -> Dict[str, float]:
        """Compute thrust alignment and disturbance torque metrics.

        Parameters
        ----------
        thrust_N : float
            Thruster force in Newtons.

        LAR-frame ideal manoeuvre directions
        ─────────────────────────────────────
        NSSK  – push North (+X) or South (−X):  ideal axis = X̂
        EWSK  – push East  (+Y) or West  (−Y):  ideal axis = Ŷ

        Background
        ──────────
        The actual thrust direction is aimed through the stack CoG, so the
        moment arm relative to the *actual* thrust line is zero by construction.
        The physically meaningful attitude disturbance is the torque you would
        create if the arm geometry is used for an *ideal* NSSK or EWSK burn —
        i.e. a force applied along ±X (NSSK) or ±Y (EWSK) at the thruster
        position.  This represents the residual attitude torque that the AOCS
        must compensate during the burn.

        Geometry
        ────────
        r              = thruster_pos − CoG   (lever arm, LAR frame)
        τ_nssk         = r × (thrust_N · nssk_hat)
        τ_ewsk         = r × (thrust_N · ewsk_hat)

        nssk_hat / ewsk_hat are chosen as the sign of X̂/Ŷ that is most aligned
        with the actual thrust direction (minimises deviation).

        Returned keys
        ─────────────
        nssk_deviation_deg       angle between actual thrust and ±X [0–90°]
        ewsk_deviation_deg       angle between actual thrust and ±Y [0–90°]
        thruster_cog_distance_m  |r|: distance from CoG to thruster [m]
        nssk_moment_arm_m        √(r_y² + r_z²): ⊥ distance to X-axis  [m]
        ewsk_moment_arm_m        √(r_x² + r_z²): ⊥ distance to Y-axis  [m]
        nssk_torque_Nm           |τ_nssk| [N·m]
        ewsk_torque_Nm           |τ_ewsk| [N·m]
        torque_x_Nm              X component of τ_nssk [N·m]
        torque_y_Nm              Y component of τ_nssk [N·m]
        torque_z_Nm              Z component of τ_nssk [N·m]
        """
        t_pos      = self.thruster_position()
        cog        = self.stack_cog_with_arm()
        plume_dir  = self.thrust_direction()
        thrust_dir = -plume_dir                     # unit actual thrust direction

        # Lever arm: CoG → thruster exit
        r = t_pos - cog
        cog_dist = float(np.linalg.norm(r))

        # Choose sign of ideal NSSK/EWSK direction aligned with actual thrust
        nssk_sign = 1.0 if float(np.dot(thrust_dir, np.array([1.0, 0.0, 0.0]))) >= 0 else -1.0
        ewsk_sign = 1.0 if float(np.dot(thrust_dir, np.array([0.0, 1.0, 0.0]))) >= 0 else -1.0
        nssk_hat  = np.array([nssk_sign, 0.0, 0.0])
        ewsk_hat  = np.array([0.0, ewsk_sign, 0.0])

        # Thrust deviation angles (unsigned)
        dot_x = float(abs(np.dot(thrust_dir, np.array([1.0, 0.0, 0.0]))))
        nssk_deviation_deg = float(np.degrees(np.arccos(np.clip(dot_x, 0.0, 1.0))))
        dot_y = float(abs(np.dot(thrust_dir, np.array([0.0, 1.0, 0.0]))))
        ewsk_deviation_deg = float(np.degrees(np.arccos(np.clip(dot_y, 0.0, 1.0))))

        # Moment arms: perpendicular distance from CoG to the ideal thrust line
        # (i.e. the component of r perpendicular to the ideal direction)
        # nssk: ideal along X → perp components are Y and Z
        nssk_moment_arm = float(np.sqrt(r[1]**2 + r[2]**2))
        # ewsk: ideal along Y → perp components are X and Z
        ewsk_moment_arm = float(np.sqrt(r[0]**2 + r[2]**2))

        # Disturbance torque vectors for each ideal burn direction
        tau_nssk = thrust_N * np.cross(r, nssk_hat)  # r × F_nssk
        tau_ewsk = thrust_N * np.cross(r, ewsk_hat)  # r × F_ewsk

        return {
            "nssk_deviation_deg":      nssk_deviation_deg,
            "ewsk_deviation_deg":      ewsk_deviation_deg,
            "thruster_cog_distance_m": cog_dist,
            "nssk_moment_arm_m":       nssk_moment_arm,
            "ewsk_moment_arm_m":       ewsk_moment_arm,
            "nssk_torque_Nm":          float(np.linalg.norm(tau_nssk)),
            "ewsk_torque_Nm":          float(np.linalg.norm(tau_ewsk)),
            # NSSK torque components (for attitude control analysis)
            "torque_x_Nm":             float(tau_nssk[0]),
            "torque_y_Nm":             float(tau_nssk[1]),
            "torque_z_Nm":             float(tau_nssk[2]),
        }


# ---------------------------------------------------------------------------
# 3.  ANALYTICAL PRE-SCREENER (Sputter Erosion Estimator)
# ---------------------------------------------------------------------------

class ErosionEstimator:
    """Quick analytical erosion depth estimate using inverse-square flux
    model with cosine-power plume profile and Yamamura sputter yield."""

    def __init__(self, thruster: ThrusterParams, material: MaterialParams):
        self.thruster = thruster
        self.material = material

    def ion_energy_eV(self) -> float:
        """Approximate ion energy from discharge voltage.
        Ions are accelerated through ~60-80% of discharge voltage typically."""
        return 0.7 * self.thruster.discharge_voltage

    def sputter_yield_normal(self) -> float:
        """Normal-incidence sputter yield (atoms/ion) using Yamamura-like fit."""
        E = self.ion_energy_eV()
        m = self.material
        if E <= m.sputter_threshold_eV:
            return 0.0
        Y = m.sputter_yield_a * (E - m.sputter_threshold_eV) ** m.sputter_yield_b
        return Y

    def sputter_yield_angular(self, incidence_deg: float) -> float:
        """Angular-dependent sputter yield."""
        Y0 = self.sputter_yield_normal()
        if Y0 == 0 or incidence_deg >= 89.0:
            return 0.0
        theta = np.radians(incidence_deg)
        cos_th = np.cos(theta)
        m = self.material
        f_theta = (cos_th ** m.angular_coeff_c) * np.exp(
            -m.angular_coeff_d * (1.0 / cos_th - 1.0)
        )
        return Y0 * f_theta

    def beam_current_A(self) -> float:
        """Approximate beam ion current from mass flow rate."""
        # I_beam ≈ (mass_flow * e) / (propellant_mass_per_ion)
        # For Xe: m_ion = 131.293 amu
        e = 1.602e-19
        if self.thruster.propellant == "Xenon":
            m_ion_kg = 131.293 * 1.66054e-27
        else:
            m_ion_kg = 131.293 * 1.66054e-27  # default to Xe
        # Assume ~80% propellant utilisation
        I_beam = 0.8 * self.thruster.mass_flow_rate * e / m_ion_kg
        return I_beam

    def local_flux(self, distance_m: float, offaxis_deg: float) -> float:
        """Ion flux at a point (ions/m²/s) using cosine-power model.

        j(r, θ) = (I_beam / e) * (n+1)/(2π r²) * cos^n(θ)
        """
        if distance_m < 0.01:
            return 0.0
        e = 1.602e-19
        I = self.beam_current_A()
        n = self.thruster.plume_cosine_exponent
        theta = np.radians(offaxis_deg)

        if offaxis_deg >= 90.0:
            return 0.0

        cos_n = np.cos(theta) ** n
        flux = (I / e) * (n + 1) / (2.0 * np.pi * distance_m ** 2) * cos_n
        return flux  # ions/m²/s

    def erosion_rate_um_per_s(self, distance_m: float, offaxis_deg: float,
                               incidence_deg: float) -> float:
        """Erosion rate in µm/s at a given point."""
        flux = self.local_flux(distance_m, offaxis_deg)
        Y = self.sputter_yield_angular(incidence_deg)
        if flux == 0 or Y == 0:
            return 0.0

        # Erosion rate = flux * Y * (atomic_volume)
        # atomic_volume = M / (rho * N_A)
        N_A = 6.022e23
        m = self.material
        atom_vol_m3 = (m.atomic_mass_amu * 1e-3) / (m.density_kg_m3 * N_A)

        rate_m_per_s = flux * Y * atom_vol_m3
        return rate_m_per_s * 1e6  # convert to µm/s

    def cumulative_erosion_um(self, distance_m: float, offaxis_deg: float,
                               incidence_deg: float, ops: OperationalParams
                               ) -> float:
        """Total erosion depth over mission lifetime in µm."""
        rate = self.erosion_rate_um_per_s(distance_m, offaxis_deg, incidence_deg)
        total_firing_s = (ops.firing_duration_s * ops.firings_per_day
                          * 365.25 * ops.mission_duration_years)
        return rate * total_firing_s


# ---------------------------------------------------------------------------
# 4.  CASE MATRIX GENERATOR
# ---------------------------------------------------------------------------

class CaseMatrixGenerator:
    """Generates parameter combinations for the sweep."""

    def __init__(self):
        self.param_ranges: Dict[str, np.ndarray] = {}
        self._set_defaults()

    def _set_defaults(self):
        """Set default parameter sweep ranges."""
        self.param_ranges = {
            # Robotic arm geometry
            "arm_reach_m":         np.array([2.0, 2.5, 3.0]),
            "shoulder_yaw_deg":    np.array([-30, -15, 0, 15, 30, 45, 60, 90]),
            "link_ratio":          np.array([0.5, 0.6, 0.7]),

            # Stack
            "client_mass":         np.array([1500, 2000, 2500, 3000, 3500]),
            "servicer_mass":       np.array([700, 750, 800]),
            "panel_span_one_side": np.array([12, 14, 16, 18]),

            # Operations
            "firing_duration_s":   np.array([15000, 17500, 20000, 22000, 25000]),
            "mission_duration_yr": np.array([3, 5, 7, 10]),
            "panel_tracking_deg":  np.array([-30, -15, 0, 15, 30]),
        }

    def set_param_range(self, name: str, values: np.ndarray):
        self.param_ranges[name] = values

    def generate_full_matrix(self) -> List[Dict]:
        """Full combinatorial sweep – WARNING: can be very large."""
        keys = list(self.param_ranges.keys())
        vals = [self.param_ranges[k] for k in keys]
        cases = []
        for combo in itertools.product(*vals):
            case = dict(zip(keys, [float(v) for v in combo]))
            cases.append(case)
        return cases

    def generate_reduced_matrix(self, fixed_params: Dict[str, float],
                                 sweep_params: List[str]) -> List[Dict]:
        """Sweep only selected parameters, fix others at given values.

        This is the recommended mode – sweep 2-3 params at a time.
        """
        # Start from fixed values
        base = dict(fixed_params)

        keys = sweep_params
        vals = [self.param_ranges[k] for k in keys]
        cases = []
        for combo in itertools.product(*vals):
            case = dict(base)
            for k, v in zip(keys, combo):
                case[k] = float(v)
            cases.append(case)
        return cases

    def count_cases(self, sweep_params: Optional[List[str]] = None) -> int:
        if sweep_params is None:
            sweep_params = list(self.param_ranges.keys())
        count = 1
        for k in sweep_params:
            count *= len(self.param_ranges[k])
        return count

    @staticmethod
    def case_to_objects(case: Dict) -> Tuple[RoboticArmGeometry, StackConfig, OperationalParams]:
        """Convert a flat case dict to the structured dataclass objects."""
        reach = case.get("arm_reach_m", case.get("arm_length", 3.0))
        ratio = case.get("link_ratio", 0.5)
        arm = RoboticArmGeometry(
            link1_length=reach * ratio,
            link2_length=reach * (1.0 - ratio),
            pivot_offset_x=case.get("pivot_offset_x", 0.0),
            pivot_offset_y=case.get("pivot_offset_y", 0.0),
            pivot_offset_z=case.get("pivot_offset_z", 0.0),
            q0_min_deg=case.get("q0_min_deg", -180.0),
            q0_max_deg=case.get("q0_max_deg",  180.0),
            q1_min_deg=case.get("q1_min_deg", -90.0),
            q1_max_deg=case.get("q1_max_deg",  90.0),
            q2_min_deg=case.get("q2_min_deg", -150.0),
            q2_max_deg=case.get("q2_max_deg",  150.0),
            shoulder_yaw_deg=case.get("shoulder_yaw_deg", case.get("arm_azimuth_deg", 0.0)),
            elbow_up=bool(case.get("elbow_up", True)),
        )
        stack = StackConfig(
            servicer_mass=case.get("servicer_mass", 700.0),
            client_mass=case.get("client_mass", 2500.0),
            panel_span_one_side=case.get("panel_span_one_side", 16.0),
            panel_width=case.get("panel_width", 2.5),
            client_bus_x=case.get("client_bus_x", 2.5),
            client_bus_y=case.get("client_bus_y", 3.0),
            client_bus_z=case.get("client_bus_z", 5.0),
            panel_cant_angle_deg=case.get("panel_cant_deg", 0.0),
        )
        ops = OperationalParams(
            firing_duration_s=case.get("firing_duration_s", 15000.0),
            mission_duration_years=case.get("mission_duration_yr", 5.0),
            firings_per_day=case.get("firings_per_day", 1.0),
            panel_sun_tracking_angle_deg=case.get("panel_tracking_deg", 0.0),
        )
        return arm, stack, ops


# ---------------------------------------------------------------------------
# 5.  PIPELINE RUNNER
# ---------------------------------------------------------------------------

class PlumePipeline:
    """Orchestrates the full parametric sweep."""

    def __init__(self, thruster: ThrusterParams = None,
                 material: MaterialParams = None):
        self.thruster = thruster or ThrusterParams()
        self.material = material or MaterialParams()
        self.estimator = ErosionEstimator(self.thruster, self.material)
        self.generator = CaseMatrixGenerator()
        self.results: List[Dict] = []

    def run_sweep(self, cases: List[Dict], verbose: bool = True) -> List[Dict]:
        """Run analytical pre-screening for all cases."""
        results = []
        n_total = len(cases)

        for idx, case in enumerate(cases):
            arm, stack, ops = CaseMatrixGenerator.case_to_objects(case)
            geo = GeometryEngine(arm, stack)
            geo_data = geo.compute_flux_geometry(
                sun_tracking_angle_deg=ops.panel_sun_tracking_angle_deg,
                n_spanwise=30, n_chordwise=6
            )

            # Compute erosion at every panel grid point
            n_pts = len(geo_data["distances_m"])
            erosions = np.zeros(n_pts)
            for i in range(n_pts):
                erosions[i] = self.estimator.cumulative_erosion_um(
                    geo_data["distances_m"][i],
                    geo_data["offaxis_angles_deg"][i],
                    geo_data["incidence_angles_deg"][i],
                    ops
                )

            max_erosion = np.max(erosions)
            max_idx = np.argmax(erosions)
            mean_erosion = np.mean(erosions[erosions > 0]) if np.any(erosions > 0) else 0.0

            # Per-antenna erosion
            ant_geo = geo.compute_antenna_flux_geometry()
            ant_erosion: Dict[str, float] = {}
            for ant_name, ant_data in ant_geo.items():
                ant_erosion[ant_name] = self.estimator.cumulative_erosion_um(
                    ant_data["distance_m"],
                    ant_data["offaxis_deg"],
                    ant_data["incidence_deg"],
                    ops
                )

            # Classify (panel erosion drives status; antenna flagged separately)
            thickness = self.material.thickness_um
            if max_erosion >= thickness:
                status = "FAIL"
            elif max_erosion >= 0.5 * thickness:
                status = "MARGINAL"
            elif max_erosion >= 0.1 * thickness:
                status = "CAUTION"
            else:
                status = "SAFE"

            cog = geo.stack_cog_with_arm()
            thrust_m = geo.thrust_metrics(thrust_N=self.thruster.thrust_N)
            result = {
                **case,
                "max_erosion_um": float(max_erosion),
                "mean_erosion_um": float(mean_erosion),
                "worst_point_distance_m": float(geo_data["distances_m"][max_idx]),
                "worst_point_offaxis_deg": float(geo_data["offaxis_angles_deg"][max_idx]),
                "worst_point_incidence_deg": float(geo_data["incidence_angles_deg"][max_idx]),
                "min_panel_distance_m": float(geo_data["min_distance_m"]),
                "thruster_pos_x": float(geo_data["thruster_pos"][0]),
                "thruster_pos_y": float(geo_data["thruster_pos"][1]),
                "thruster_pos_z": float(geo_data["thruster_pos"][2]),
                "cog_x": float(cog[0]),
                "cog_y": float(cog[1]),
                "cog_z": float(cog[2]),
                "status": status,
                "erosion_fraction": float(max_erosion / thickness),
                # Per-antenna erosion (µm, lifetime)
                "ant_E1_erosion_um": float(ant_erosion.get("E1", 0.0)),
                "ant_E2_erosion_um": float(ant_erosion.get("E2", 0.0)),
                "ant_W1_erosion_um": float(ant_erosion.get("W1", 0.0)),
                "ant_W2_erosion_um": float(ant_erosion.get("W2", 0.0)),
                "ant_E1_distance_m": float(ant_geo["E1"]["distance_m"]),
                "ant_W1_distance_m": float(ant_geo["W1"]["distance_m"]),
                "ant_max_erosion_um": float(max(ant_erosion.values())),
                # Thrust alignment and disturbance torque (Step 5)
                "nssk_deviation_deg":      thrust_m["nssk_deviation_deg"],
                "ewsk_deviation_deg":      thrust_m["ewsk_deviation_deg"],
                "thruster_cog_distance_m": thrust_m["thruster_cog_distance_m"],
                "nssk_moment_arm_m":       thrust_m["nssk_moment_arm_m"],
                "ewsk_moment_arm_m":       thrust_m["ewsk_moment_arm_m"],
                "nssk_torque_Nm":          thrust_m["nssk_torque_Nm"],
                "ewsk_torque_Nm":          thrust_m["ewsk_torque_Nm"],
                "torque_x_Nm":             thrust_m["torque_x_Nm"],
                "torque_y_Nm":             thrust_m["torque_y_Nm"],
                "torque_z_Nm":             thrust_m["torque_z_Nm"],
            }
            results.append(result)

            if verbose and (idx + 1) % max(1, n_total // 10) == 0:
                print(f"  [{idx+1}/{n_total}] processed – "
                      f"last case: {status}, max erosion = {max_erosion:.3f} µm")

        self.results = results
        return results

    def export_results_csv(self, filepath: str):
        """Export results to CSV."""
        import csv
        if not self.results:
            print("No results to export.")
            return
        keys = self.results[0].keys()
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)
        print(f"Results exported to {filepath} ({len(self.results)} cases)")

    def export_results_json(self, filepath: str):
        """Export results to JSON."""
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results exported to {filepath}")

    def get_openplume_cases(self) -> List[Dict]:
        """Return only MARGINAL and CAUTION cases that need full
        OpenPlume simulation (filters out SAFE and FAIL)."""
        return [r for r in self.results if r["status"] in ("MARGINAL", "CAUTION")]

    def summary(self) -> Dict:
        """Print summary statistics."""
        if not self.results:
            return {}
        statuses = [r["status"] for r in self.results]
        summary = {
            "total_cases": len(self.results),
            "SAFE": statuses.count("SAFE"),
            "CAUTION": statuses.count("CAUTION"),
            "MARGINAL": statuses.count("MARGINAL"),
            "FAIL": statuses.count("FAIL"),
            "max_erosion_overall_um": max(r["max_erosion_um"] for r in self.results),
            "min_erosion_overall_um": min(r["max_erosion_um"] for r in self.results),
            "openplume_cases_needed": len(self.get_openplume_cases()),
        }
        return summary


# ---------------------------------------------------------------------------
# 6.  HEATMAP VISUALIZER
# ---------------------------------------------------------------------------

def generate_heatmaps(results: List[Dict],
                      param_x: str,
                      param_y: str,
                      metric: str = "max_erosion_um",
                      fixed_params: Optional[Dict[str, float]] = None,
                      output_dir: str = ".",
                      thickness_um: float = 25.0,
                      show_plot: bool = False):
    """
    Generate 2D heatmap for any pair of sweep parameters.

    Parameters
    ----------
    results      : list of result dicts from pipeline
    param_x      : parameter name for X-axis
    param_y      : parameter name for Y-axis
    metric       : which result column to plot (default: max_erosion_um)
    fixed_params : dict of {param: value} to filter results when more than 2
                   params were swept.  Selects results closest to these values.
    output_dir   : where to save figures
    thickness_um : silver interconnect thickness for fail-line overlay
    show_plot    : whether to call plt.show()
    """
    import matplotlib
    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    import matplotlib.ticker as mticker

    # Filter results if fixed_params given
    filtered = results
    if fixed_params:
        for pk, pv in fixed_params.items():
            if pk in (param_x, param_y):
                continue
            vals = np.array([r[pk] for r in filtered])
            closest = vals[np.argmin(np.abs(vals - pv))]
            filtered = [r for r in filtered if abs(r[pk] - closest) < 1e-9]

    if len(filtered) == 0:
        print(f"WARNING: No results match the fixed parameter filter.")
        return

    # Extract unique axis values
    x_vals = sorted(set(r[param_x] for r in filtered))
    y_vals = sorted(set(r[param_y] for r in filtered))

    # Build 2D grid
    Z = np.full((len(y_vals), len(x_vals)), np.nan)
    for r in filtered:
        ix = x_vals.index(r[param_x])
        iy = y_vals.index(r[param_y])
        # If multiple results map to same cell, take the worst case
        current = Z[iy, ix]
        val = r[metric]
        if np.isnan(current) or val > current:
            Z[iy, ix] = val

    # ---- Create figure ----
    fig, ax = plt.subplots(figsize=(12, 8))

    # Custom colourmap: green → yellow → orange → red
    colors_safe = [(0.18, 0.65, 0.35),   # green  – SAFE
                   (0.95, 0.85, 0.20),    # yellow – CAUTION
                   (0.95, 0.55, 0.15),    # orange – MARGINAL
                   (0.85, 0.15, 0.15)]    # red    – FAIL
    cmap = LinearSegmentedColormap.from_list("erosion", colors_safe, N=256)

    # Normalise: 0 to max(thickness, max_val)
    vmax = max(thickness_um * 1.2, np.nanmax(Z) * 1.05) if np.any(~np.isnan(Z)) else thickness_um
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
                   vmin=0, vmax=vmax,
                   extent=[min(x_vals) - 0.5 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else min(x_vals) - 0.5,
                           max(x_vals) + 0.5 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else max(x_vals) + 0.5,
                           min(y_vals) - 0.5 * (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else min(y_vals) - 0.5,
                           max(y_vals) + 0.5 * (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else max(y_vals) + 0.5])

    # Contour line at threshold
    if len(x_vals) > 1 and len(y_vals) > 1:
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        try:
            cs = ax.contour(X_grid, Y_grid, Z, levels=[thickness_um],
                            colors="white", linewidths=2.5, linestyles="--")
            ax.clabel(cs, fmt=f"%.0f µm (FAIL)", fontsize=9, colors="white")
        except Exception:
            pass  # contour may fail if all values on one side

        # Also mark 50% threshold
        try:
            cs2 = ax.contour(X_grid, Y_grid, Z, levels=[0.5 * thickness_um],
                             colors="white", linewidths=1.5, linestyles=":")
            ax.clabel(cs2, fmt=f"%.0f µm (MARGINAL)", fontsize=8, colors="white")
        except Exception:
            pass

    # Annotate cells with values
    if len(x_vals) <= 15 and len(y_vals) <= 15:
        for iy, yv in enumerate(y_vals):
            for ix, xv in enumerate(x_vals):
                val = Z[iy, ix]
                if not np.isnan(val):
                    color = "white" if val > 0.5 * vmax else "black"
                    ax.text(xv, yv, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")

    # Labels
    param_labels = {
        "arm_reach_m": "Arm Reach L1+L2 [m]",
        "shoulder_yaw_deg": "Shoulder Yaw [°]",
        "link_ratio": "Link Ratio L1/(L1+L2)",
        "arm_length": "Arm Length [m]",          # legacy
        "arm_azimuth_deg": "Arm Azimuth [°]",    # legacy
        "arm_elevation_deg": "Arm Elevation [°]",  # legacy
        "client_mass": "Client Mass [kg]",
        "servicer_mass": "Servicer Mass [kg]",
        "panel_span_one_side": "Panel Span (one side) [m]",
        "firing_duration_s": "Firing Duration [s]",
        "mission_duration_yr": "Mission Duration [yr]",
        "panel_tracking_deg": "Panel Sun-Tracking Angle [°]",
        "max_erosion_um": "Max Erosion Depth [µm]",
        "mean_erosion_um": "Mean Erosion Depth [µm]",
        "erosion_fraction": "Erosion Fraction of Thickness",
        "min_panel_distance_m": "Min Panel Distance [m]",
        "ik_feasible": "IK Feasible",
        "arm_collision": "Arm Collision with Bus",
    }

    ax.set_xlabel(param_labels.get(param_x, param_x), fontsize=12, fontweight="bold")
    ax.set_ylabel(param_labels.get(param_y, param_y), fontsize=12, fontweight="bold")

    metric_label = param_labels.get(metric, metric)
    ax.set_title(f"Plume Impingement Erosion: {metric_label}\n"
                 f"Ag interconnect threshold = {thickness_um} µm",
                 fontsize=14, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(metric_label, fontsize=11)
    # Add threshold markers on colorbar
    cbar.ax.axhline(y=thickness_um, color="white", linewidth=2, linestyle="--")
    cbar.ax.axhline(y=0.5 * thickness_um, color="white", linewidth=1, linestyle=":")

    # Add fixed-param annotation
    if fixed_params:
        fixed_text = "Fixed: " + ", ".join(
            f"{param_labels.get(k, k).split('[')[0].strip()}={v}"
            for k, v in fixed_params.items() if k not in (param_x, param_y)
        )
        ax.annotate(fixed_text, xy=(0.02, 0.98), xycoords="axes fraction",
                    fontsize=8, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()

    # Save
    fname = f"heatmap_{param_x}_vs_{param_y}_{metric}.png"
    fpath = os.path.join(output_dir, fname)
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    print(f"Heatmap saved: {fpath}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fpath


def generate_status_map(results: List[Dict],
                        param_x: str,
                        param_y: str,
                        fixed_params: Optional[Dict[str, float]] = None,
                        output_dir: str = ".",
                        show_plot: bool = False):
    """Generate a categorical status map (SAFE/CAUTION/MARGINAL/FAIL)."""
    import matplotlib
    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Filter
    filtered = results
    if fixed_params:
        for pk, pv in fixed_params.items():
            if pk in (param_x, param_y):
                continue
            vals = np.array([r[pk] for r in filtered])
            closest = vals[np.argmin(np.abs(vals - pv))]
            filtered = [r for r in filtered if abs(r[pk] - closest) < 1e-9]

    status_map = {"SAFE": 0, "CAUTION": 1, "MARGINAL": 2, "FAIL": 3}
    x_vals = sorted(set(r[param_x] for r in filtered))
    y_vals = sorted(set(r[param_y] for r in filtered))

    Z = np.full((len(y_vals), len(x_vals)), np.nan)
    for r in filtered:
        ix = x_vals.index(r[param_x])
        iy = y_vals.index(r[param_y])
        val = status_map[r["status"]]
        current = Z[iy, ix]
        if np.isnan(current) or val > current:  # worst case
            Z[iy, ix] = val

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = ListedColormap(["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                   extent=[min(x_vals) - 0.5 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else min(x_vals) - 0.5,
                           max(x_vals) + 0.5 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else max(x_vals) + 0.5,
                           min(y_vals) - 0.5 * (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else min(y_vals) - 0.5,
                           max(y_vals) + 0.5 * (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else max(y_vals) + 0.5])

    # Cell labels
    status_labels = {0: "SAFE", 1: "CAUT", 2: "MARG", 3: "FAIL"}
    if len(x_vals) <= 15 and len(y_vals) <= 15:
        for iy, yv in enumerate(y_vals):
            for ix, xv in enumerate(x_vals):
                val = Z[iy, ix]
                if not np.isnan(val):
                    ax.text(xv, yv, status_labels[int(val)],
                            ha="center", va="center", fontsize=8,
                            fontweight="bold", color="white")

    param_labels = {
        "arm_reach_m": "Arm Reach L1+L2 [m]",
        "shoulder_yaw_deg": "Shoulder Yaw [°]",
        "link_ratio": "Link Ratio L1/(L1+L2)",
        "arm_length": "Arm Length [m]",          # legacy
        "arm_azimuth_deg": "Arm Azimuth [°]",    # legacy
        "arm_elevation_deg": "Arm Elevation [°]",  # legacy
        "client_mass": "Client Mass [kg]",
        "servicer_mass": "Servicer Mass [kg]",
        "panel_span_one_side": "Panel Span (one side) [m]",
        "firing_duration_s": "Firing Duration [s]",
        "mission_duration_yr": "Mission Duration [yr]",
        "panel_tracking_deg": "Panel Sun-Tracking Angle [°]",
        "ik_feasible": "IK Feasible",
        "arm_collision": "Arm Collision with Bus",
    }

    ax.set_xlabel(param_labels.get(param_x, param_x), fontsize=12, fontweight="bold")
    ax.set_ylabel(param_labels.get(param_y, param_y), fontsize=12, fontweight="bold")
    ax.set_title(f"Plume Impingement Status Map\n{param_labels.get(param_x, param_x)} vs "
                 f"{param_labels.get(param_y, param_y)}", fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ECC71", label="SAFE (<10% thickness)"),
        Patch(facecolor="#F1C40F", label="CAUTION (10-50%)"),
        Patch(facecolor="#E67E22", label="MARGINAL (50-100%)"),
        Patch(facecolor="#E74C3C", label="FAIL (>100% thickness)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    if fixed_params:
        fixed_text = "Fixed: " + ", ".join(
            f"{param_labels.get(k, k).split('[')[0].strip()}={v}"
            for k, v in fixed_params.items() if k not in (param_x, param_y)
        )
        ax.annotate(fixed_text, xy=(0.02, 0.98), xycoords="axes fraction",
                    fontsize=8, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    fname = f"statusmap_{param_x}_vs_{param_y}.png"
    fpath = os.path.join(output_dir, fname)
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    print(f"Status map saved: {fpath}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fpath


# ---------------------------------------------------------------------------
# 7.  DEMO / MAIN
# ---------------------------------------------------------------------------

def run_demo():
    """Run a demonstration sweep and generate example heatmaps."""

    print("=" * 70)
    print("  PLASMA PLUME IMPINGEMENT – PARAMETRIC SWEEP PIPELINE")
    print("=" * 70)

    # --- Setup ---
    thruster = ThrusterParams(
        name="SPT-100-like",
        discharge_voltage=300.0,
        mass_flow_rate=5e-6,
        beam_divergence_half_angle=20.0,
        plume_cosine_exponent=10.0,
    )

    material = MaterialParams(
        name="Silver_interconnect",
        thickness_um=25.0,
    )

    pipeline = PlumePipeline(thruster, material)

    # --- Define sweep ---
    gen = pipeline.generator
    # Reduce ranges for demo
    gen.set_param_range("arm_length", np.arange(1.0, 4.5, 0.5))
    gen.set_param_range("arm_azimuth_deg", np.arange(-30, 95, 15))
    gen.set_param_range("arm_elevation_deg", np.array([-10, 0, 10, 20]))
    gen.set_param_range("panel_span_one_side", np.array([8, 10, 12, 15, 18]))
    gen.set_param_range("firing_duration_s", np.array([300, 600, 900, 1200, 1800]))
    gen.set_param_range("mission_duration_yr", np.array([3, 5, 7, 10]))
    gen.set_param_range("client_mass", np.array([2000, 3500, 5000]))
    gen.set_param_range("servicer_mass", np.array([400]))
    gen.set_param_range("panel_tracking_deg", np.array([-15, 0, 15]))

    # --- Sweep 1: Arm length vs Arm azimuth ---
    print("\n[SWEEP 1] Arm Length vs Arm Azimuth")
    fixed1 = {
        "arm_elevation_deg": 0.0,
        "client_mass": 3500.0,
        "servicer_mass": 400.0,
        "panel_span_one_side": 12.0,
        "firing_duration_s": 600.0,
        "mission_duration_yr": 5.0,
        "panel_tracking_deg": 0.0,
    }
    cases1 = gen.generate_reduced_matrix(fixed1, ["arm_length", "arm_azimuth_deg"])
    print(f"  Cases to evaluate: {len(cases1)}")
    results1 = pipeline.run_sweep(cases1, verbose=True)

    # --- Sweep 2: Panel span vs Mission duration ---
    print("\n[SWEEP 2] Panel Span vs Mission Duration")
    fixed2 = {
        "arm_length": 2.5,
        "arm_azimuth_deg": 0.0,
        "arm_elevation_deg": 0.0,
        "client_mass": 3500.0,
        "servicer_mass": 400.0,
        "firing_duration_s": 600.0,
        "panel_tracking_deg": 0.0,
    }
    cases2 = gen.generate_reduced_matrix(fixed2, ["panel_span_one_side", "mission_duration_yr"])
    print(f"  Cases to evaluate: {len(cases2)}")
    results2 = pipeline.run_sweep(cases2, verbose=True)

    # --- Sweep 3: Firing duration vs Arm elevation ---
    print("\n[SWEEP 3] Firing Duration vs Arm Elevation")
    fixed3 = {
        "arm_length": 2.5,
        "arm_azimuth_deg": 0.0,
        "client_mass": 3500.0,
        "servicer_mass": 400.0,
        "panel_span_one_side": 12.0,
        "mission_duration_yr": 5.0,
        "panel_tracking_deg": 0.0,
    }
    cases3 = gen.generate_reduced_matrix(fixed3, ["firing_duration_s", "arm_elevation_deg"])
    print(f"  Cases to evaluate: {len(cases3)}")
    results3 = pipeline.run_sweep(cases3, verbose=True)

    # --- Generate heatmaps ---
    output_dir = "/home/karan.anand/Documents/PythonScripts/ThrusterArmWorkspaceAnalysis/pipeline_output"
    os.makedirs(output_dir, exist_ok=True)

    print("\n[GENERATING HEATMAPS]")

    # Erosion depth heatmaps
    f1 = generate_heatmaps(results1, "arm_length", "arm_azimuth_deg",
                           metric="max_erosion_um", output_dir=output_dir)

    f2 = generate_heatmaps(results2, "panel_span_one_side", "mission_duration_yr",
                           metric="max_erosion_um", output_dir=output_dir)

    f3 = generate_heatmaps(results3, "firing_duration_s", "arm_elevation_deg",
                           metric="max_erosion_um", output_dir=output_dir)

    # Status maps
    f4 = generate_status_map(results1, "arm_length", "arm_azimuth_deg",
                             output_dir=output_dir)

    f5 = generate_status_map(results2, "panel_span_one_side", "mission_duration_yr",
                             output_dir=output_dir)

    f6 = generate_status_map(results3, "firing_duration_s", "arm_elevation_deg",
                             output_dir=output_dir)

    # --- Export ---
    all_results = results1 + results2 + results3
    pipeline.results = all_results
    pipeline.export_results_csv(os.path.join(output_dir, "erosion_results.csv"))

    # Summary
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    for sweep_name, res_list in [("Sweep 1 (Arm L vs Az)", results1),
                                  ("Sweep 2 (Panel vs Duration)", results2),
                                  ("Sweep 3 (Firing vs Elev)", results3)]:
        pipeline.results = res_list
        s = pipeline.summary()
        print(f"\n  {sweep_name}:")
        print(f"    Total cases:       {s['total_cases']}")
        print(f"    SAFE:              {s['SAFE']}")
        print(f"    CAUTION:           {s['CAUTION']}")
        print(f"    MARGINAL:          {s['MARGINAL']}")
        print(f"    FAIL:              {s['FAIL']}")
        print(f"    Max erosion:       {s['max_erosion_overall_um']:.3f} µm")
        print(f"    OpenPlume needed:  {s['openplume_cases_needed']}")

    # List of OpenPlume candidates
    pipeline.results = all_results
    openplume = pipeline.get_openplume_cases()
    print(f"\n  Total cases needing OpenPlume simulation: {len(openplume)}")

    print("\n  Generated files:")
    for f in [f1, f2, f3, f4, f5, f6]:
        if f:
            print(f"    → {f}")
    print(f"    → {os.path.join(output_dir, 'erosion_results.csv')}")

    return output_dir


if __name__ == "__main__":
    run_demo()
