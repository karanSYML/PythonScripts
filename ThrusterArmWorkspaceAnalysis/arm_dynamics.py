"""
arm_dynamics.py
===============
Pinocchio-based joint-torque and reaction-wrench computation for the
3-DOF yaw-pitch-pitch thruster arm mounted on a free-floating servicer bus.

Requires pinocchio 4.x:
    conda install -c conda-forge pinocchio

Two Pinocchio model variants
-----------------------------
Fixed-base model (3 DOF)
    RNEA yields the 3 joint torques needed to produce a given motion.
    Gravity is set to zero (orbital free-fall).

Free-flyer model (6 + 3 = 9 velocity DOF)
    The servicer bus is a free-floating root joint.  RNEA with zero base
    velocity/acceleration returns the 6-DOF reaction wrench applied by the
    arm on the bus when the joints accelerate.

Orbital fictitious accelerations
---------------------------------
In the LVLH (Local Vertical Local Horizontal) orbital frame the spacecraft
is in a non-inertial rotating frame.  Coriolis (2 ω × v) and centrifugal
(ω × (ω × r)) terms can be added as body-force corrections to the RNEA
gravity vector.  For LEO (ω ≈ 1.2 × 10⁻³ rad/s) these are < 0.2 % of
typical arm-dynamic loads and can usually be neglected.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict

import numpy as np
import pinocchio as pin

from generate_arm_urdf import generate_urdf
from plume_impingement_pipeline import RoboticArmGeometry, StackConfig


# ---------------------------------------------------------------------------
# Model builder (internal)
# ---------------------------------------------------------------------------

def _build_pinocchio_model(
    arm: RoboticArmGeometry,
    stack: StackConfig,
    free_flyer: bool = False,
) -> tuple[pin.Model, pin.Data]:
    """Build a Pinocchio model from arm + stack geometry via URDF.

    Parameters
    ----------
    free_flyer : if True, add a 6-DOF free-flyer joint at the root.
    """
    urdf_str = generate_urdf(arm, stack)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    try:
        tmp.write(urdf_str)
        tmp.flush()
        tmp.close()
        if free_flyer:
            model = pin.buildModelFromUrdf(tmp.name, pin.JointModelFreeFlyer())
        else:
            model = pin.buildModelFromUrdf(tmp.name)
    finally:
        os.unlink(tmp.name)

    model.gravity.linear = np.zeros(3)   # orbital free-fall
    data = model.createData()
    return model, data


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ArmDynamics:
    """Joint-torque and reaction-wrench engine for the 3-DOF thruster arm.

    Parameters
    ----------
    arm   : RoboticArmGeometry   defaults to hardware-confirmed values
    stack : StackConfig          defaults to hardware-confirmed values

    Examples
    --------
    >>> dyn = ArmDynamics()
    >>> q   = np.radians([90, 60, 30])
    >>> dq  = np.array([0.05, 0.05, 0.02])   # rad/s
    >>> ddq = np.array([0.01, 0.01, 0.005])  # rad/s²
    >>> budget = dyn.torque_budget(q, dq, ddq)
    >>> print(f"τ_max = {budget['tau_max']:.2f} N·m")
    >>> print(f"M_reaction_max = {budget['reaction_moment_max']:.2f} N·m")
    """

    def __init__(
        self,
        arm:   RoboticArmGeometry | None = None,
        stack: StackConfig         | None = None,
    ):
        self.arm   = arm   or RoboticArmGeometry()
        self.stack = stack or StackConfig()

        self._model_fb, self._data_fb = _build_pinocchio_model(
            self.arm, self.stack, free_flyer=False
        )
        self._model_ff, self._data_ff = _build_pinocchio_model(
            self.arm, self.stack, free_flyer=True
        )

    # ── Read-only model access ────────────────────────────────────────────────

    @property
    def model(self) -> pin.Model:
        """Fixed-base Pinocchio model (3 revolute joints)."""
        return self._model_fb

    @property
    def model_ff(self) -> pin.Model:
        """Free-flyer Pinocchio model (free-flyer + 3 revolute joints)."""
        return self._model_ff

    # ── Fixed-base RNEA: joint torques ────────────────────────────────────────

    def joint_torques(
        self,
        q:   np.ndarray,
        dq:  np.ndarray,
        ddq: np.ndarray,
    ) -> np.ndarray:
        """Joint torques via RNEA (fixed base, zero gravity).

        τ = M(q) q̈ + C(q, q̇) q̇      (g = 0 in orbital free-fall)

        Parameters
        ----------
        q   : (3,) joint angles       [rad]
        dq  : (3,) joint velocities   [rad/s]
        ddq : (3,) joint accelerations [rad/s²]

        Returns
        -------
        tau : (3,) joint torques [N·m]
              tau[0] = J1 shoulder yaw
              tau[1] = J2 elbow pitch
              tau[2] = J3 wrist
        """
        tau = pin.rnea(
            self._model_fb, self._data_fb,
            np.asarray(q,   dtype=float),
            np.asarray(dq,  dtype=float),
            np.asarray(ddq, dtype=float),
        )
        return np.array(tau)

    # ── Free-floating RNEA: reaction wrench at servicer bus ───────────────────

    def reaction_wrench(
        self,
        q:              np.ndarray,
        dq:             np.ndarray,
        ddq:            np.ndarray,
        base_velocity:  np.ndarray | None = None,
        base_acc:       np.ndarray | None = None,
    ) -> np.ndarray:
        """6-DOF reaction wrench applied by the arm on the servicer bus.

        Uses the free-flyer Pinocchio model.  With base_velocity=0 and
        base_acc=0 (servicer not translating/rotating during this timestep),
        the first 6 components of RNEA output equal the generalized force the
        arm dynamics impose on the base joint.

        Parameters
        ----------
        q             : (3,) joint angles       [rad]
        dq            : (3,) joint velocities   [rad/s]
        ddq           : (3,) joint accelerations [rad/s²]
        base_velocity : (6,) [v_lin, v_ang] servicer velocity [m/s, rad/s]
                        default: zeros
        base_acc      : (6,) [a_lin, a_ang] servicer acceleration [m/s², rad/s²]
                        default: zeros

        Returns
        -------
        wrench : (6,) [Fx, Fy, Fz, Mx, My, Mz] in servicer body frame [N, N·m]
        """
        model = self._model_ff
        data  = self._data_ff

        bv = np.zeros(6) if base_velocity is None else np.asarray(base_velocity, float)
        ba = np.zeros(6) if base_acc      is None else np.asarray(base_acc,      float)

        q_full     = pin.neutral(model)           # identity pose at origin
        q_full[7:] = np.asarray(q, dtype=float)

        v_full     = np.zeros(model.nv)
        v_full[:6] = bv
        v_full[6:] = np.asarray(dq, dtype=float)

        a_full     = np.zeros(model.nv)
        a_full[:6] = ba
        a_full[6:] = np.asarray(ddq, dtype=float)

        tau = pin.rnea(model, data, q_full, v_full, a_full)
        # tau[:6] = generalised force at the free-flyer root joint
        # (reaction wrench the arm exerts on the servicer bus)
        return np.array(tau[:6])

    # ── Mass matrix and Coriolis/centrifugal ──────────────────────────────────

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Composite rigid-body (joint-space) mass matrix M(q), shape (3, 3)."""
        M = pin.crba(self._model_fb, self._data_fb,
                     np.asarray(q, dtype=float))
        return np.array(M)

    def coriolis_centrifugal(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Coriolis + centrifugal bias vector C(q, q̇) q̇, shape (3,)."""
        h = pin.rnea(
            self._model_fb, self._data_fb,
            np.asarray(q,  dtype=float),
            np.asarray(dq, dtype=float),
            np.zeros(3),
        )
        return np.array(h)

    # ── Orbital fictitious accelerations ──────────────────────────────────────

    @staticmethod
    def orbital_fictitious_acc(
        r_lvlh: np.ndarray,
        v_lvlh: np.ndarray,
        omega_orb: np.ndarray,
    ) -> np.ndarray:
        """Coriolis + centrifugal acceleration in the LVLH orbital frame.

        In the Local Vertical / Local Horizontal frame the equations of motion
        have fictitious acceleration terms:

            a_fict = 2 ω × v  +  ω × (ω × r)
                     Coriolis     centrifugal

        For a circular LEO orbit (altitude ~500 km):
            |ω| ≈ 1.13 × 10⁻³ rad/s  →  |a_fict| < 1 × 10⁻³ m/s²

        These corrections can be included by adding them to the model gravity:
            model.gravity.linear = -orbital_fictitious_acc(...)

        Parameters
        ----------
        r_lvlh    : (3,) arm CoM position in LVLH frame [m]
        v_lvlh    : (3,) arm CoM velocity in LVLH frame [m/s]
        omega_orb : (3,) orbital angular velocity in LVLH frame [rad/s]
                    For circular orbit nadir-pointing: [0, -ω_orb, 0]

        Returns
        -------
        a_fict : (3,) fictitious acceleration [m/s²]
        """
        r = np.asarray(r_lvlh,    dtype=float)
        v = np.asarray(v_lvlh,    dtype=float)
        w = np.asarray(omega_orb, dtype=float)
        return 2.0 * np.cross(w, v) + np.cross(w, np.cross(w, r))

    # ── Convenience: full torque budget ──────────────────────────────────────

    def torque_budget(
        self,
        q:   np.ndarray,
        dq:  np.ndarray,
        ddq: np.ndarray,
    ) -> Dict[str, object]:
        """Joint torques and base reaction wrench in a single call.

        Parameters
        ----------
        q, dq, ddq : joint angles [rad], velocities [rad/s], accelerations [rad/s²]

        Returns
        -------
        dict with keys:
          'tau'                 : (3,) joint torques [N·m]
          'reaction_force'      : (3,) net force on servicer bus [N]
          'reaction_moment'     : (3,) net moment on servicer bus [N·m]
          'tau_max'             : max |tau_i|  [N·m]
          'reaction_moment_max' : max |M_i|    [N·m]
          'reaction_force_max'  : max |F_i|    [N]
        """
        tau    = self.joint_torques(q, dq, ddq)
        wrench = self.reaction_wrench(q, dq, ddq)
        return {
            'tau':                  tau,
            'reaction_force':       wrench[:3],
            'reaction_moment':      wrench[3:],
            'tau_max':              float(np.max(np.abs(tau))),
            'reaction_moment_max':  float(np.max(np.abs(wrench[3:]))),
            'reaction_force_max':   float(np.max(np.abs(wrench[:3]))),
        }
