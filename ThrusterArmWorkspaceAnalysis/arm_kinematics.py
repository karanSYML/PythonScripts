"""
arm_kinematics.py
=================
Spatial kinematics for the 3-DOF yaw-pitch-pitch thruster arm.

Joint convention (matches RoboticArmGeometry.forward_kinematics):
  J1  shoulder yaw   – revolute about +Z at the arm pivot
  J2  elbow pitch    – revolute about local Y (perpendicular to yaw direction)
  J3  wrist pitch    – revolute about the same local Y axis

All positions are in the LAR frame unless stated otherwise.

Algorithms ported from the MATLAB thruster-arm-robotics repository:
  hmg_to_spatial_tform.m      → spatial_transform_from_homogeneous
  dtma_get_cog_and_jacobian.m → arm_cog_and_jacobian
  dtma_cog_pos_to_joint_pos.m → arm_cog_ik
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from plume_impingement_pipeline import RoboticArmGeometry

from plume_impingement_pipeline import _rodrigues


# ---------------------------------------------------------------------------
# Spatial (Plücker) transform utility
# ---------------------------------------------------------------------------

def spatial_transform_from_homogeneous(T: np.ndarray) -> np.ndarray:
    """Convert a 4×4 homogeneous transform to a 6×6 Plücker spatial transform.

    X = [ R,          0 ]
        [ skew(p)·R,  R ]

    where R = T[:3,:3] (rotation) and p = T[:3,3] (translation).

    Reference: Featherstone, Rigid Body Dynamics Algorithms, eq. 2.28.
    Port of hmg_to_spatial_tform.m.
    """
    R = T[:3, :3]
    px, py, pz = T[0, 3], T[1, 3], T[2, 3]
    skew_p = np.array([[ 0.0, -pz,  py],
                        [ pz,  0.0, -px],
                        [-py,  px,  0.0]])
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[3:, :3] = skew_p @ R
    X[3:, 3:] = R
    return X


# ---------------------------------------------------------------------------
# 4×4 homogeneous transform chain
# ---------------------------------------------------------------------------

def arm_fk_transforms(arm: RoboticArmGeometry,
                       pivot: np.ndarray,
                       q: np.ndarray,
                       servicer_yaw_deg: float = 0.0) -> dict[str, np.ndarray]:
    """Full 4×4 homogeneous transform chain in the LAR frame.

    Convention: T_world_X maps points from frame X to the LAR/world frame,
    i.e. p_world = T_world_X @ np.array([*p_in_X, 1]).

    General serial-chain FK using Rodrigues rotations. `pivot` is Hinge 1
    in LAR frame. Joint axes and link vectors are in TA body frame and are
    rotated to LAR by Rz_serv (servicer yaw).

    Returns
    -------
    dict with keys:
      'T_world_j1'     4×4  LAR frame ← j1 frame  (after Hinge-1 rotation)
      'T_world_elbow'  4×4  LAR frame ← Hinge-2 (elbow) frame
      'T_world_wrist'  4×4  LAR frame ← Hinge-3 (wrist) frame
      'T_world_ee'     4×4  LAR frame ← nozzle (EE) frame
      'T_j1_elbow'     4×4  j1 frame  ← elbow frame  (local)
      'T_elbow_wrist'  4×4  elbow     ← wrist frame  (local)
      'T_wrist_ee'     4×4  wrist     ← EE frame     (fixed, no joint at nozzle)
    """
    q0, q1, q2 = float(q[0]), float(q[1]), float(q[2])

    # Servicer yaw: rotates TA body frame into LAR
    c_s, s_s = np.cos(np.radians(servicer_yaw_deg)), np.sin(np.radians(servicer_yaw_deg))
    Rz_s = np.array([[c_s, -s_s, 0.], [s_s, c_s, 0.], [0., 0., 1.]])

    # Joint rotations in TA body frame
    R1_ta = _rodrigues(arm.axis1, q0)
    R2_ta = _rodrigues(arm.axis2, q1)
    R3_ta = _rodrigues(arm.axis3, q2)

    # Cumulative rotations in LAR frame (Rz_s @ R_ta_cumulative)
    CR1  = Rz_s @ R1_ta
    CR12 = Rz_s @ (R1_ta @ R2_ta)
    CR3_local = R3_ta               # wrist → EE rotation in wrist body frame

    # Link vectors in LAR frame
    d12_lar = CR1  @ arm.d_h1h2
    d23_lar = CR12 @ arm.d_h2h3
    CR123 = CR12 @ R3_ta
    d3n_lar = CR123 @ arm.d_h3n

    # Absolute joint positions in LAR
    p_h2     = pivot + d12_lar
    p_h3     = p_h2  + d23_lar
    p_nozzle = p_h3  + d3n_lar

    # --- Absolute transforms ---
    T_world_j1    = np.eye(4); T_world_j1[:3, :3]    = CR1;  T_world_j1[:3, 3]    = pivot
    T_world_elbow = np.eye(4); T_world_elbow[:3, :3]  = CR12; T_world_elbow[:3, 3]  = p_h2
    T_world_wrist = np.eye(4); T_world_wrist[:3, :3]  = CR123; T_world_wrist[:3, 3] = p_h3
    T_world_ee    = np.eye(4); T_world_ee[:3, :3]     = CR123; T_world_ee[:3, 3]    = p_nozzle

    # --- Local transforms (expressed in parent body frame) ---
    T_j1_elbow    = np.eye(4); T_j1_elbow[:3, :3]    = R2_ta;    T_j1_elbow[:3, 3]    = arm.d_h1h2
    T_elbow_wrist = np.eye(4); T_elbow_wrist[:3, :3]  = R3_ta;    T_elbow_wrist[:3, 3]  = arm.d_h2h3
    T_wrist_ee    = np.eye(4); T_wrist_ee[:3, :3]     = np.eye(3); T_wrist_ee[:3, 3]    = arm.d_h3n

    return {
        'T_world_j1':    T_world_j1,
        'T_world_elbow': T_world_elbow,
        'T_world_wrist': T_world_wrist,
        'T_world_ee':    T_world_ee,
        'T_j1_elbow':    T_j1_elbow,
        'T_elbow_wrist': T_elbow_wrist,
        'T_wrist_ee':    T_wrist_ee,
    }


# ---------------------------------------------------------------------------
# 6D spatial inertia tensor
# ---------------------------------------------------------------------------

def build_6d_inertia(mass: float,
                      com: np.ndarray,
                      I_3x3: np.ndarray) -> np.ndarray:
    """Build the 6×6 spatial inertia tensor from mass, CoM, and 3×3 inertia.

    I6 = [ I_3x3,        m·skew(c) ]
         [ m·skew(c)ᵀ,  m·I₃      ]

    Port of the tensor6D construction in inertiaProperties.m.
    """
    cx, cy, cz = float(com[0]), float(com[1]), float(com[2])
    skew_c_m = np.array([[ 0.0,  -cz,   cy],
                          [ cz,   0.0,  -cx],
                          [-cy,   cx,   0.0]]) * mass
    I6 = np.zeros((6, 6))
    I6[:3, :3] = I_3x3
    I6[:3, 3:] = skew_c_m
    I6[3:, :3] = skew_c_m.T
    I6[3:, 3:] = mass * np.eye(3)
    return I6


# ---------------------------------------------------------------------------
# Arm CoG and centroidal Jacobian
# ---------------------------------------------------------------------------

def arm_cog_and_jacobian(arm: RoboticArmGeometry,
                          pivot: np.ndarray,
                          q: np.ndarray,
                          servicer_yaw_deg: float = 0.0,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the arm CoG position and centroidal velocity Jacobian.

    General serial-chain implementation using cross-product Jacobian columns.
    Port of dtma_get_cog_and_jacobian.m, updated for the real hardware geometry.

    Parameters
    ----------
    arm              : RoboticArmGeometry
    pivot            : (3,) Hinge-1 position in LAR frame
    q                : (3,) joint angles [q0, q1, q2] in radians
    servicer_yaw_deg : servicer body yaw relative to LAR [deg]

    Returns
    -------
    cog   : (3,) arm CoG in LAR frame [m]
    J_cog : (3, 3)  ṗ_cog = J_cog @ q̇
    """
    q0, q1, q2 = float(q[0]), float(q[1]), float(q[2])

    c_s, s_s = np.cos(np.radians(servicer_yaw_deg)), np.sin(np.radians(servicer_yaw_deg))
    Rz_s = np.array([[c_s, -s_s, 0.], [s_s, c_s, 0.], [0., 0., 1.]])

    # Cumulative FK rotations in LAR frame
    R1_ta  = _rodrigues(arm.axis1, q0)
    R12_ta = R1_ta @ _rodrigues(arm.axis2, q1)
    R123_ta = R12_ta @ _rodrigues(arm.axis3, q2)

    CR1   = Rz_s @ R1_ta
    CR12  = Rz_s @ R12_ta
    CR123 = Rz_s @ R123_ta

    # Hinge positions in LAR
    p_h1 = pivot
    p_h2 = p_h1 + CR1  @ arm.d_h1h2
    p_h3 = p_h2 + CR12 @ arm.d_h2h3

    # Joint rotation axes in LAR frame
    omega1 = Rz_s @ arm.axis1             # axis 1 is fixed in servicer body
    omega2 = CR1  @ arm.axis2             # axis 2 rotates with joint 1
    omega3 = CR12 @ arm.axis3             # axis 3 rotates with joints 1&2

    # CoM fractions along each link (distance from proximal joint / link length)
    f1 = arm.effective_link1_com()   / arm.link1_length
    f2 = arm.effective_link2_com()   / arm.link2_length
    f3 = arm.effective_bracket_com() / arm.bracket_length

    m1, m2, m3 = arm.link1_mass, arm.link2_mass, arm.bracket_mass
    M = m1 + m2 + m3

    # CoM positions
    p_com1 = p_h1 + CR1   @ (f1 * arm.d_h1h2)
    p_com2 = p_h2 + CR12  @ (f2 * arm.d_h2h3)
    p_com3 = p_h3 + CR123 @ (f3 * arm.d_h3n)

    cog = (m1 * p_com1 + m2 * p_com2 + m3 * p_com3) / M

    # Velocity Jacobian: J[:, i] = cross(omega_i, p_com - p_joint_i)
    # Joint i affects all link CoMs that are downstream (after) joint i.
    # col 0 (omega1 at p_h1): affects com1, com2, com3
    J0 = (m1 * np.cross(omega1, p_com1 - p_h1)
        + m2 * np.cross(omega1, p_com2 - p_h1)
        + m3 * np.cross(omega1, p_com3 - p_h1)) / M

    # col 1 (omega2 at p_h2): affects com2, com3 only
    J1 = (m2 * np.cross(omega2, p_com2 - p_h2)
        + m3 * np.cross(omega2, p_com3 - p_h2)) / M

    # col 2 (omega3 at p_h3): affects com3 only
    J2 = m3 * np.cross(omega3, p_com3 - p_h3) / M

    J_cog = np.column_stack([J0, J1, J2])   # (3, 3)

    return cog, J_cog


def arm_cog_position(arm: RoboticArmGeometry,
                      pivot: np.ndarray,
                      q: np.ndarray) -> np.ndarray:
    """Convenience wrapper — return only the arm CoG (no Jacobian)."""
    cog, _ = arm_cog_and_jacobian(arm, pivot, q)
    return cog


# ---------------------------------------------------------------------------
# CoG inverse kinematics — damped least-squares (Chan's method)
# ---------------------------------------------------------------------------

def arm_cog_ik(arm: RoboticArmGeometry,
                pivot: np.ndarray,
                q_init: np.ndarray,
                target_cog: np.ndarray,
                pos_mask: np.ndarray = None,
                error_weights: np.ndarray = None,
                damping_gain: float = 1e-3,
                position_tol: float = 1e-3,
                max_iters: int = 100,
                enforce_limits: bool = True
                ) -> Tuple[bool, np.ndarray, int]:
    """Damped least-squares CoG IK (Chan's adaptive damping).

    Finds joint angles q such that arm_cog_position(arm, pivot, q) ≈ target_cog.

    Port of dtma_cog_pos_to_joint_pos.m.

    Update rule per iteration
    -------------------------
      e        = pos_mask * (target_cog − cog(q))
      sq_err   = 0.5 · eᵀ W e
      λ²       = damping_gain · sq_err         ← Chan's adaptive damping
      Δq       = (JᵀWJ + λ²I)⁻¹ Jᵀ W e
      q       += Δq

    Parameters
    ----------
    arm           : RoboticArmGeometry
    pivot         : (3,) arm base position in LAR frame
    q_init        : (3,) initial joint angles [rad]
    target_cog    : (3,) desired CoG position in LAR frame
    pos_mask      : (3,) boolean/float mask — set axis to 0 to leave it free.
                    Default: all axes constrained ([1, 1, 1]).
    error_weights : (3, 3) diagonal weight matrix.  Default: identity.
    damping_gain  : scalar for Chan's adaptive damping (λ²).
    position_tol  : convergence criterion on 0.5·eᵀWe.
    max_iters     : maximum Newton iterations.
    enforce_limits: if True, returns success=False when solution violates limits.

    Returns
    -------
    success   : bool
    q_sol     : (3,) solution joint angles [rad]
    num_iters : int  iterations taken
    """
    if pos_mask is None:
        pos_mask = np.ones(3)
    if error_weights is None:
        error_weights = np.eye(3)

    pos_mask      = np.asarray(pos_mask, dtype=float)
    error_weights = np.asarray(error_weights, dtype=float)
    target_cog    = np.asarray(target_cog, dtype=float)

    q         = np.array(q_init, dtype=float)
    eye3      = np.eye(3)
    num_iters = 0

    while num_iters < max_iters:
        cog, J = arm_cog_and_jacobian(arm, pivot, q)

        error     = pos_mask * (target_cog - cog)
        sq_err    = 0.5 * float(error @ error_weights @ error)

        if sq_err < position_tol:
            if enforce_limits and not arm.within_joint_limits(q[0], q[1], q[2]):
                return False, q, num_iters
            return True, q, num_iters

        # Chan's adaptive damping: λ² proportional to current error magnitude
        lam2           = damping_gain * sq_err
        damping_matrix = lam2 * eye3

        g  = J.T @ error_weights @ error
        H  = J.T @ error_weights @ J + damping_matrix
        dq = np.linalg.solve(H, g)

        q         = q + dq
        num_iters += 1

    return False, q, num_iters
