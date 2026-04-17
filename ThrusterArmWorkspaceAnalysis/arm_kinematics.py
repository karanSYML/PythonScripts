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
                       q: np.ndarray) -> dict[str, np.ndarray]:
    """Full 4×4 homogeneous transform chain in the LAR frame.

    Convention: T_world_X maps points from frame X to the LAR/world frame,
    i.e. p_world = T_world_X @ np.array([*p_in_X, 1]).

    Local relative transforms T_A_B map points from frame B to frame A,
    i.e. p_A = T_A_B @ np.array([*p_in_B, 1]).

    Joint rotation convention
    -------------------------
    Link 1 is always horizontal (parallel to the servicer +Z face); only J1
    (q0) sweeps it. J2 (q1) pitches link 2 at the elbow: in the elbow frame
    the link-2 direction is cos(q1)·X̂ + sin(q1)·Ẑ, corresponding to Ry(−q1)
    acting on X̂. J3 (q2) pitches the bracket at the wrist by the same
    convention with Ry(−q2).

    Returns
    -------
    dict with keys:
      'T_world_j1'     4×4  LAR frame ← j1 frame  (after shoulder yaw)
      'T_world_elbow'  4×4  LAR frame ← elbow frame
      'T_world_wrist'  4×4  LAR frame ← wrist frame
      'T_world_ee'     4×4  LAR frame ← thruster-exit frame
      'T_j1_elbow'     4×4  j1 frame  ← elbow frame  (local, joint-angle dependent)
      'T_elbow_wrist'  4×4  elbow     ← wrist  (local)
      'T_wrist_ee'     4×4  wrist     ← EE     (fixed, no joint here)
    """
    q0, q1, q2 = float(q[0]), float(q[1]), float(q[2])
    c0, s0 = np.cos(q0), np.sin(q0)
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)

    L1 = arm.link1_length
    L2 = arm.link2_length
    L3 = arm.bracket_length

    # Rotation about Z by q0  (shoulder yaw)
    Rz = np.array([[c0, -s0, 0.0],
                   [s0,  c0, 0.0],
                   [0.0, 0.0, 1.0]])

    # Rotation about Y by −q1  (elbow pitch upward)
    # Ry(−θ) = [cθ, 0, −sθ; 0, 1, 0; sθ, 0, cθ] — maps X̂ → cos θ·X̂ + sin θ·Ẑ
    Ry_neg1 = np.array([[ c1, 0.0, -s1],
                         [0.0, 1.0,  0.0],
                         [ s1, 0.0,  c1]])

    # Rotation about Y by −q2  (wrist pitch)
    Ry_neg2 = np.array([[ c2, 0.0, -s2],
                         [0.0, 1.0,  0.0],
                         [ s2, 0.0,  c2]])

    # --- Local transforms (child frame origin expressed in parent frame) ---

    # j1 frame: apply yaw rotation at the pivot
    T_world_j1 = np.eye(4)
    T_world_j1[:3, :3] = Rz
    T_world_j1[:3,  3] = pivot

    # Elbow frame: link 1 is always horizontal (no pitch at shoulder).
    # Elbow origin in j1 frame is L1 along local X̂; no rotation change.
    T_j1_elbow = np.eye(4)
    T_j1_elbow[:3, :3] = np.eye(3)
    T_j1_elbow[:3,  3] = np.array([L1, 0.0, 0.0])

    # Wrist frame: q1 pitches link 2 at the elbow about local Y.
    # Ry(−q1) maps local X̂ → cos(q1)·X̂ + sin(q1)·Ẑ (link 2 direction).
    # Wrist origin in elbow frame: Ry_neg1 @ [L2, 0, 0]ᵀ = [L2·c1, 0, L2·s1]ᵀ
    T_elbow_wrist = np.eye(4)
    T_elbow_wrist[:3, :3] = Ry_neg1
    T_elbow_wrist[:3,  3] = np.array([L2 * c1, 0.0, L2 * s1])

    # EE frame: q2 pitches the bracket at the wrist about local Y.
    # Ry(−q2) maps local X̂ → cos(q2)·X̂ + sin(q2)·Ẑ (bracket direction).
    # EE origin in wrist frame: Ry_neg2 @ [L3, 0, 0]ᵀ = [L3·c2, 0, L3·s2]ᵀ
    T_wrist_ee = np.eye(4)
    T_wrist_ee[:3, :3] = Ry_neg2
    T_wrist_ee[:3,  3] = np.array([L3 * c2, 0.0, L3 * s2])

    # --- Absolute transforms (compose for full chain) ---
    T_world_elbow = T_world_j1 @ T_j1_elbow
    T_world_wrist = T_world_elbow @ T_elbow_wrist
    T_world_ee    = T_world_wrist @ T_wrist_ee

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
                          q: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the arm CoG position and centroidal Jacobian.

    Uses the direct analytical approach exploiting the yaw-pitch-pitch
    structure: after the shoulder yaw (q0), links 1, 2, and the bracket
    all move in the same vertical plane, giving clean closed-form partials.

    Port of dtma_get_cog_and_jacobian.m.

    Parameters
    ----------
    arm    : RoboticArmGeometry  (inertia fields used if set, else thin-rod defaults)
    pivot  : (3,) LAR-frame position of the arm base (shoulder joint)
    q      : (3,) joint angles [q0, q1, q2] in radians

    Returns
    -------
    cog    : (3,) arm centre-of-gravity in LAR frame [m]
    J_cog  : (3, 3) centroidal Jacobian  J  such that  ṗ_cog = J @ q̇
    """
    q0, q1, q2 = float(q[0]), float(q[1]), float(q[2])

    c0, s0   = np.cos(q0), np.sin(q0)
    c1, s1   = np.cos(q1), np.sin(q1)
    c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)

    # Unit vectors in LAR frame
    u_rad  = np.array([c0,  s0,  0.0])   # horizontal, in yaw direction
    u_perp = np.array([-s0, c0,  0.0])   # horizontal, perpendicular to yaw
    u_z    = np.array([0.0, 0.0, 1.0])   # vertical

    # Link directions in LAR frame
    # Link 1 is always horizontal; only q0 (yaw) moves it.
    # q1 pitches link 2 at the elbow; q1+q2 pitches the bracket at the wrist.
    d_link2   = c1  * u_rad + s1  * u_z    # link 2 direction
    d_bracket = c12 * u_rad + s12 * u_z    # bracket direction

    # CoM offsets along each link axis
    c_L1  = arm.effective_link1_com()
    c_L2  = arm.effective_link2_com()
    c_br  = arm.effective_bracket_com()

    m1 = arm.link1_mass
    m2 = arm.link2_mass
    m3 = arm.bracket_mass
    M  = m1 + m2 + m3

    L1 = arm.link1_length
    L2 = arm.link2_length

    # CoM positions in LAR frame (link 1 always along u_rad)
    p1 = pivot + c_L1 * u_rad                          # link 1 CoM: always horizontal
    p2 = pivot + L1 * u_rad + c_L2 * d_link2           # link 2 CoM
    p3 = pivot + L1 * u_rad + L2 * d_link2 + c_br * d_bracket  # bracket CoM

    cog = (m1 * p1 + m2 * p2 + m3 * p3) / M

    # ---- Centroidal Jacobian ----
    # Partial derivatives:
    #   d(u_rad)/dq0    = u_perp   (horizontal rotation)
    #   d(d_link2)/dq1  = −s1·u_rad + c1·u_z
    #   d(d_bracket)/dq1 = d(d_bracket)/dq2 = −s12·u_rad + c12·u_z
    #
    # dp1/dq0 = c_L1·u_perp        (link 1 CoM sweeps with yaw)
    # dp1/dq1 = 0                   (link 1 is always horizontal)
    # dp1/dq2 = 0
    #
    # dp2/dq0 = (L1 + c_L2·c1)·u_perp
    # dp2/dq1 = c_L2·d(d_link2)/dq1
    # dp2/dq2 = 0
    #
    # dp3/dq0 = (L1 + L2·c1 + c_br·c12)·u_perp
    # dp3/dq1 = L2·d(d_link2)/dq1 + c_br·d(d_bracket)/dq12
    # dp3/dq2 = c_br·d(d_bracket)/dq12

    dd_link2_dq1   = -s1  * u_rad + c1  * u_z
    dd_bracket_dq12 = -s12 * u_rad + c12 * u_z    # same partial for dq1 and dq2

    # Column 0 — d(cog)/dq0
    scale_q0 = (m1 * c_L1
                + m2 * (L1 + c_L2 * c1)
                + m3 * (L1 + L2 * c1 + c_br * c12)) / M
    J_col0 = scale_q0 * u_perp

    # Column 1 — d(cog)/dq1
    J_col1 = ((m2 * c_L2 + m3 * L2) * dd_link2_dq1
              + m3 * c_br * dd_bracket_dq12) / M

    # Column 2 — d(cog)/dq2
    J_col2 = m3 * c_br * dd_bracket_dq12 / M

    J_cog = np.column_stack([J_col0, J_col1, J_col2])   # (3, 3)

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
