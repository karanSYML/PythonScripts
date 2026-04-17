"""
arm_trajectory.py
=================
Joint-space trajectory generation for the 3-DOF thruster arm.

Provides:
  lspb_trajectory         – single-joint LSPB (trapezoidal velocity profile)
  cog_to_line_trajectory  – move arm CoG onto a target line, min joint travel

Algorithms ported from the MATLAB thruster-arm-robotics repository:
  get_lspb_traj.m              → lspb_trajectory
  dtma_get_cog_traj_to_line.m  → cog_to_line_trajectory
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Tuple

from arm_kinematics import arm_cog_ik

if TYPE_CHECKING:
    from plume_impingement_pipeline import RoboticArmGeometry

# Maximum pre-allocated trajectory length (matches DTMAConstants.MAX_NUM_TRAJ_TIMESTEPS)
MAX_TRAJ_TIMESTEPS: int = 10_000


# ---------------------------------------------------------------------------
# Single-joint LSPB trajectory
# ---------------------------------------------------------------------------

def lspb_trajectory(q0: float,
                     qf: float,
                     dt: float,
                     max_vel: float,
                     max_acc: float,
                     max_timesteps: int = MAX_TRAJ_TIMESTEPS
                     ) -> Tuple[bool, np.ndarray, int]:
    """LSPB (Linear Segment with Parabolic Blend) trajectory for one joint.

    Generates a trapezoidal velocity profile respecting max_vel and max_acc.
    Degrades to a bang-bang (triangular) profile when the displacement is
    too small to need a constant-velocity segment.

    Velocity and acceleration limits are respected in the continuous-time
    sense; discretisation by dt may cause finite-difference derivatives to
    slightly exceed the limits at blend boundaries.

    Port of get_lspb_traj.m.
    Reference: Spong, Hutchinson, Vidyasagar — Robot Modeling and Control, §9.2.

    Parameters
    ----------
    q0            : initial joint position [rad]
    qf            : final   joint position [rad]
    dt            : sample period [s]  (must be > 0)
    max_vel       : peak velocity magnitude [rad/s]  (must be > 0)
    max_acc       : acceleration magnitude  [rad/s²] (must be > 0)
    max_timesteps : pre-allocated output array length

    Returns
    -------
    success           : False if trajectory exceeds max_timesteps
    trajectory        : (max_timesteps,) array; valid over [0, num_steps),
                        remainder filled with qf
    num_valid_steps   : number of time steps that span the motion
    """
    motion_sign = 1.0 if qf >= q0 else -1.0
    displacement = abs(qf - q0)

    # Degenerate case: already at target
    if displacement < 1e-12:
        traj = np.full(max_timesteps, qf)
        return True, traj, 1

    # Maximum distance covered during a single blend (ramp-up or ramp-down)
    max_blend_time = max_vel / max_acc
    max_blend_dist = max_vel * max_blend_time   # = max_vel² / max_acc

    # If displacement can be done purely by ramp-up + ramp-down (bang-bang),
    # reduce peak velocity accordingly
    need_linear = True
    if displacement <= max_blend_dist:
        max_blend_time = np.sqrt(displacement / max_acc)
        max_vel        = displacement / max_blend_time   # reduced peak vel
        max_blend_dist = max_vel * max_blend_time
        need_linear    = False

    const_vel_time   = (displacement - max_blend_dist) / max_vel if need_linear else 0.0
    traj_duration    = 2.0 * max_blend_time + const_vel_time
    traj_num_steps   = int(np.ceil(traj_duration / dt))

    if traj_num_steps > max_timesteps:
        traj = np.full(max_timesteps, qf)
        return False, traj, traj_num_steps

    # Pre-fill with qf (tail of array = hold at target)
    traj = np.full(max_timesteps, qf)

    # Time samples for the active portion
    t = np.linspace(0.0, traj_duration, traj_num_steps)

    if need_linear:
        blend_time = (motion_sign * (q0 - qf) + max_vel * traj_duration) / max_vel
    else:
        blend_time = max_blend_time

    blend_steps      = int(np.floor(blend_time / dt))
    const_vel_steps  = int(np.floor(const_vel_time / dt)) if need_linear else 0
    second_blend_idx = blend_steps + const_vel_steps  # 0-based start of second blend

    # First blend: parabolic ramp-up from rest
    traj[:blend_steps] = (q0
                           + motion_sign * (max_acc / 2.0)
                           * t[:blend_steps] ** 2)

    # Linear (constant-velocity) segment
    if need_linear and const_vel_steps > 0:
        t_lin = t[blend_steps: second_blend_idx]
        traj[blend_steps: second_blend_idx] = (
            (qf + q0 - motion_sign * max_vel * traj_duration) / 2.0
            + motion_sign * max_vel * t_lin
        )

    # Second blend: parabolic ramp-down to rest
    t2 = t[second_blend_idx: traj_num_steps]
    traj[second_blend_idx: traj_num_steps] = (
        qf + motion_sign * (
            -max_acc * traj_duration ** 2 / 2.0
            + max_acc * traj_duration * t2
            - max_acc / 2.0 * t2 ** 2
        )
    )

    return True, traj, traj_num_steps


# ---------------------------------------------------------------------------
# CoG-to-line trajectory planner
# ---------------------------------------------------------------------------

def cog_to_line_trajectory(arm: RoboticArmGeometry,
                             pivot: np.ndarray,
                             q_init: np.ndarray,
                             line_point: np.ndarray,
                             line_dir: np.ndarray,
                             dt: float,
                             joint_vel_limits: np.ndarray,
                             joint_acc_limits: np.ndarray,
                             min_joint_limits_rad: np.ndarray = None,
                             max_joint_limits_rad: np.ndarray = None,
                             cog_workspace_lims: np.ndarray = None,
                             num_line_samples: int = 1000,
                             sample_spacing_m: float = 0.01,
                             ik_tol: float = 1e-3,
                             ik_max_iters: int = 100,
                             ) -> Tuple[bool, np.ndarray, int]:
    """Move the arm CoG onto a line while minimising joint-space travel.

    Finds the joint-space configuration nearest to q_init whose CoG lies on
    the infinite line defined by (line_point, line_dir), then generates an
    LSPB trajectory for each joint independently.

    Port of dtma_get_cog_traj_to_line.m.

    Algorithm
    ---------
    1. Sample points along the line bidirectionally from line_point,
       stopping when out of cog_workspace_lims.
    2. At each sample run arm_cog_ik() with pos_mask that zeroes the axis
       parallel to line_dir (leaving that axis unconstrained).
    3. Collect all IK solutions within joint limits.
    4. Select the solution nearest q_init in joint space (Euclidean distance).
    5. Generate an LSPB trajectory per joint; total duration = longest joint.

    Parameters
    ----------
    arm                 : RoboticArmGeometry
    pivot               : (3,) arm base position in LAR frame
    q_init              : (3,) initial joint angles [rad]
    line_point          : (3,) a point on the target line, LAR frame [m]
    line_dir            : (3,) unit direction of the target line, LAR frame
    dt                  : control period [s]
    joint_vel_limits    : (3,) max joint velocities [rad/s]
    joint_acc_limits    : (3,) max joint accelerations [rad/s²]
    min_joint_limits_rad: (3,) lower joint limits [rad].  None → use arm dataclass limits.
    max_joint_limits_rad: (3,) upper joint limits [rad].  None → use arm dataclass limits.
    cog_workspace_lims  : (3, 2) workspace box [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
                          in LAR frame [m].  None → ±5 m in XY, ±3 m in Z.
    num_line_samples    : maximum number of points to sample along the line.
    sample_spacing_m    : distance between consecutive line samples [m].
    ik_tol              : CoG IK convergence tolerance (passed to arm_cog_ik).
    ik_max_iters        : maximum IK iterations per sample.

    Returns
    -------
    success             : True if at least one valid IK solution was found
                          and all per-joint LSPB trajectories were generated.
    joint_trajectory    : (3, MAX_TRAJ_TIMESTEPS) — row i = joint i over time.
                          Valid over [:, :traj_num_steps].
    traj_num_steps      : number of valid time steps (longest per-joint trajectory).
    """
    # --- Default arguments ---
    line_dir = np.asarray(line_dir, dtype=float)
    norm = np.linalg.norm(line_dir)
    if norm < 1e-12:
        raise ValueError("line_dir must be a non-zero vector.")
    line_dir = line_dir / norm

    if cog_workspace_lims is None:
        cog_workspace_lims = np.array([[-5.0, 5.0],
                                        [-5.0, 5.0],
                                        [-3.0, 3.0]])

    if min_joint_limits_rad is None:
        min_joint_limits_rad = np.radians([arm.q0_min_deg,
                                            arm.q1_min_deg,
                                            arm.q2_min_deg])
    if max_joint_limits_rad is None:
        max_joint_limits_rad = np.radians([arm.q0_max_deg,
                                            arm.q1_max_deg,
                                            arm.q2_max_deg])

    # pos_mask: constrain all axes except those parallel to line_dir
    # An axis i is "parallel" when |line_dir[i]| ≈ 1; we zero the dominant axis.
    dominant_axis = int(np.argmax(np.abs(line_dir)))
    pos_mask = np.ones(3)
    pos_mask[dominant_axis] = 0.0

    # --- Sample points along the line, bidirectionally from line_point ---
    # Each IK call is warm-started from the previous successful solution so that
    # the solver tracks smoothly along the line without wrap-around issues.
    valid_q      = []
    direction    = 1.0       # start going in +line_dir direction
    magnitude    = 0.0
    sample_count = 0
    q_warm       = np.array(q_init, dtype=float)   # warm-start seed

    while sample_count < num_line_samples:
        sample_count += 1
        pt = np.asarray(line_point, dtype=float) + magnitude * line_dir

        # Check workspace bounds
        out_of_bounds = any(
            pt[i] < cog_workspace_lims[i, 0] or pt[i] > cog_workspace_lims[i, 1]
            for i in range(3)
        )
        if out_of_bounds:
            if direction > 0:
                # Switch to negative direction; reset warm-start to q_init
                direction = -1.0
                magnitude = sample_spacing_m
                q_warm = np.array(q_init, dtype=float)
                continue
            else:
                break   # both directions exhausted

        # Solve CoG IK at this point (warm-start from previous solution)
        success, q_sol, _ = arm_cog_ik(
            arm, pivot, q_warm, pt,
            pos_mask=pos_mask,
            damping_gain=1e-3,
            position_tol=ik_tol,
            max_iters=ik_max_iters,
            enforce_limits=False,   # check limits manually below
        )

        if success and np.all(q_sol >= min_joint_limits_rad) \
                   and np.all(q_sol <= max_joint_limits_rad):
            valid_q.append(q_sol.copy())
            q_warm = q_sol   # update warm-start only on success

        magnitude += sample_spacing_m

    if not valid_q:
        traj = np.zeros((3, MAX_TRAJ_TIMESTEPS))
        return False, traj, 0

    # --- Find the joint-space nearest solution to q_init ---
    valid_q_arr = np.array(valid_q)          # (N, 3)
    diffs       = valid_q_arr - q_init       # (N, 3)
    distances   = np.linalg.norm(diffs, axis=1)
    best_idx    = int(np.argmin(distances))
    q_target    = valid_q_arr[best_idx]

    # --- Generate per-joint LSPB trajectories ---
    joint_traj  = np.zeros((3, MAX_TRAJ_TIMESTEPS))
    traj_steps  = 0

    for j in range(3):
        ok, traj_j, n_steps = lspb_trajectory(
            float(q_init[j]), float(q_target[j]),
            dt,
            float(joint_vel_limits[j]),
            float(joint_acc_limits[j]),
            max_timesteps=MAX_TRAJ_TIMESTEPS,
        )
        if not ok:
            return False, joint_traj, 0

        joint_traj[j, :] = traj_j
        traj_steps = max(traj_steps, n_steps)

    return True, joint_traj, traj_steps
