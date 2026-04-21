"""
composite_mass_model.py
=======================
Mission-epoch composite centre-of-gravity (CoG) model for the servicer+client stack.

Frame convention (LAR frame)
-----------------------------
  Origin : LAR docking interface between servicer and client.
  +X     : North  (orbit-normal / North solar panel direction)
  +Y     : East   (along-track / East antenna face direction)
  +Z     : Nadir  (toward Earth)

  This is the same frame used throughout plume_impingement_pipeline.py and
  arm_kinematics.py. The spec's "composite body frame B" maps to LAR as:
    spec x_B (along-track / East) = LAR +Y
    spec y_B (orbit-normal / North) = LAR +X
    spec z_B (nadir)               = LAR +Z
  No coordinate transformation is required since the origin (LAR interface) is
  the same. The axis labelling in this module follows LAR convention.

  Station-keeping thrust directions in LAR frame:
    +N = +X_LAR,  -N = -X_LAR,  +E = +Y_LAR,  -W = -Y_LAR

Relationship to existing StackConfig.stack_cog()
-------------------------------------------------
  StackConfig.stack_cog() is a static model (total servicer_mass, no
  propellant depletion). This module adds a mission-epoch model that tracks
  propellant consumption across the mission lifetime without modifying
  any existing code.

  At BOL with matching mass parameters:
    CompositeMassModel.p_CoG_LAR(0.0) ≈ StackConfig.stack_cog()
  (Exact match requires p_C_LAR = client_origin_in_lar_frame() and
   (m_S_dry + m_prop_0) * servicer_origin = servicer_mass * servicer_origin.)

Single-tank assumption
-----------------------
  Propellant CoG is assumed fixed at the tank centroid regardless of fill
  level. This is valid for:
    - A single spherical/cylindrical tank with diaphragm/bladder
    - Symmetric multi-tank arrangements (CoG invariant with fill level)
  For slosh or asymmetric tanks, replace p_tank_LAR with a fill-level-
  dependent position vector.

Non-uniform burn cadence
------------------------
  The default model assumes a constant burn_cadence_hr_per_day throughout
  the mission. To model non-uniform cadences (orbit raising, disposal,
  off-nominal phases): compute cumulative_burn_time externally as a
  mission-time integral, then pass it directly to m_prop() by subclassing
  or by calling m_prop_from_burn_time(burn_time_s) directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

G_0: float = 9.80665  # m/s²


@dataclass
class CompositeMassModel:
    """Mission-epoch composite CoG model for the servicer+client stack.

    All position vectors are in the LAR frame [m].

    Parameters
    ----------
    m_C : float
        Client mass [kg]. Assumed constant throughout mission.
    m_S_dry : float
        Servicer dry-structure mass [kg], excluding propellant.
    m_prop_0 : float
        Initial propellant mass at BOL [kg].
    thrust_N : float
        Nominal thruster force [N].
    isp_s : float
        Specific impulse [s].
    p_C_LAR : ndarray (3,)
        Client CoG in LAR frame [m].
    p_S_dry_LAR : ndarray (3,)
        Servicer dry-structure CoG in LAR frame [m].
    p_tank_LAR : ndarray (3,)
        Propellant tank centroid in LAR frame [m]. Fixed regardless of fill
        level (single-tank / symmetric-tank assumption; see module docstring).
    burn_cadence_hr_per_day : float
        Total station-keeping burn time per day [hr/day]. Default 10 (5h N + 5h S).
    """

    m_C: float
    m_S_dry: float
    m_prop_0: float
    thrust_N: float
    isp_s: float
    p_C_LAR: np.ndarray
    p_S_dry_LAR: np.ndarray
    p_tank_LAR: np.ndarray
    burn_cadence_hr_per_day: float = 10.0

    # ------------------------------------------------------------------
    # Core mass model
    # ------------------------------------------------------------------

    def mdot(self) -> float:
        """Mass flow rate [kg/s]."""
        return self.thrust_N / (self.isp_s * G_0)

    def cumulative_burn_time(self, epoch_days: float) -> float:
        """Total thruster-on time from BOL to epoch τ [s].

        Assumes constant daily cadence. For non-uniform schedules, compute
        the integral externally and use m_prop_from_burn_time() instead.
        """
        return self.burn_cadence_hr_per_day * 3600.0 * epoch_days

    def m_prop_from_burn_time(self, burn_time_s: float) -> float:
        """Remaining propellant mass given accumulated burn time [kg]. Clamped at 0."""
        return max(0.0, self.m_prop_0 - self.mdot() * burn_time_s)

    def m_prop(self, epoch_days: float) -> float:
        """Remaining propellant mass at mission epoch τ [kg]. Clamped at 0."""
        return self.m_prop_from_burn_time(self.cumulative_burn_time(epoch_days))

    def M(self, epoch_days: float) -> float:
        """Total composite mass at mission epoch τ [kg]."""
        return self.m_C + self.m_S_dry + self.m_prop(epoch_days)

    # ------------------------------------------------------------------
    # CoG model
    # ------------------------------------------------------------------

    def p_CoG_LAR(self, epoch_days: float) -> np.ndarray:
        """Composite CoG in LAR frame at mission epoch τ [m].

        Uses cross-product-stable numerics: divides by M(τ) once.
        """
        mp = self.m_prop(epoch_days)
        M_tau = self.m_C + self.m_S_dry + mp
        return (
            self.m_C * self.p_C_LAR
            + self.m_S_dry * self.p_S_dry_LAR
            + mp * self.p_tank_LAR
        ) / M_tau

    def p_CoG_LAR_rate(self, epoch_days: float) -> np.ndarray:
        """Analytical CoG migration rate d(p_CoG_LAR)/d(epoch_days) [m/day].

        Returns zero once propellant is exhausted.
        Matches the analytical formula in spec §3.3, translated to LAR frame.
        """
        if self.m_prop(epoch_days) <= 0.0:
            return np.zeros(3)
        # dm_prop/d(epoch_days) [kg/day]
        dm_dt = -self.mdot() * self.burn_cadence_hr_per_day * 3600.0
        return (dm_dt / self.M(epoch_days)) * (self.p_tank_LAR - self.p_CoG_LAR(epoch_days))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def propellant_exhausted_day(self) -> float:
        """Mission day on which propellant is fully depleted [days from BOL]."""
        return self.m_prop_0 / (self.mdot() * self.burn_cadence_hr_per_day * 3600.0)

    def cog_migration_magnitude(self, epoch_days: float) -> float:
        """‖p_CoG(τ) − p_CoG(BOL)‖ [m]."""
        return float(np.linalg.norm(self.p_CoG_LAR(epoch_days) - self.p_CoG_LAR(0.0)))

    def cog_trajectory(self, epoch_schedule_days: list[float]) -> np.ndarray:
        """Return CoG position at each epoch as array of shape (K, 3) [m]."""
        return np.array([self.p_CoG_LAR(t) for t in epoch_schedule_days])

    def suggested_epoch_spacing(self, eps_CoG_m: float) -> float:
        """Maximum epoch spacing [days] such that CoG migration between consecutive
        epochs stays below eps_CoG_m. Based on max migration rate at BOL (fastest).

        Use this to sanity-check your epoch_schedule_days spacing.
        """
        rate_at_bol = np.linalg.norm(self.p_CoG_LAR_rate(0.0))  # m/day
        if rate_at_bol == 0.0:
            return float("inf")
        return eps_CoG_m / rate_at_bol

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        path: str = "feasibility_inputs.json",
        stack=None,
    ) -> "CompositeMassModel":
        """Load a CompositeMassModel from feasibility_inputs.json.

        Parameters
        ----------
        path : str
            Absolute path or path relative to this module's directory.
        stack : StackConfig, optional
            If provided, used to auto-compute default CoG positions for any
            position field left as null in the JSON. Mirrors the fallback
            logic used by geometry_visualizer.py / pivot_position().

        Raises
        ------
        ValueError
            If servicer_dry_mass_kg or propellant_mass_0_kg are null (they
            cannot be inferred from geometry and must be set explicitly).
        FileNotFoundError
            If the JSON file does not exist at the resolved path.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        fpath = path if os.path.isabs(path) else os.path.join(here, path)
        with open(fpath) as f:
            cfg = json.load(f)

        # --- scalar parameters ---
        thrust_N     = float(cfg.get("thrust_N", 0.054))
        isp_s        = float(cfg.get("isp_s", 1485.0))
        burn_cadence = float(cfg.get("burn_cadence_hr_per_day", 10.0))
        m_C          = float(cfg.get("client_mass_kg",
                                      stack.client_mass if stack is not None else 2500.0))

        m_S_dry  = cfg.get("servicer_dry_mass_kg")
        m_prop_0 = cfg.get("propellant_mass_0_kg")
        if m_S_dry is None or m_prop_0 is None:
            raise ValueError(
                "feasibility_inputs.json: 'servicer_dry_mass_kg' and "
                "'propellant_mass_0_kg' must both be set (not null). "
                "Their sum should equal the initial total servicer mass at BOL."
            )
        m_S_dry  = float(m_S_dry)
        m_prop_0 = float(m_prop_0)

        # --- position vectors: null → fallback to StackConfig geometry ---
        def _resolve(key: str, fallback_fn) -> np.ndarray:
            raw = cfg.get(key, [None, None, None])
            if raw is not None and all(v is not None for v in raw):
                return np.array(raw, dtype=float)
            if stack is None:
                raise ValueError(
                    f"feasibility_inputs.json: '{key}' is null and no "
                    "StackConfig was supplied to compute a geometric default."
                )
            result = fallback_fn()
            return np.asarray(result, dtype=float)

        p_C_LAR     = _resolve("client_cog_lar_m",      stack.client_origin_in_lar_frame)
        p_S_dry_LAR = _resolve("servicer_dry_cog_lar_m", stack.servicer_origin_in_lar_frame)
        # Tank centroid defaults to servicer geometric centre — conservative single-tank proxy.
        p_tank_LAR  = _resolve("tank_centroid_lar_m",   stack.servicer_origin_in_lar_frame)

        return cls(
            m_C=m_C,
            m_S_dry=m_S_dry,
            m_prop_0=m_prop_0,
            thrust_N=thrust_N,
            isp_s=isp_s,
            p_C_LAR=p_C_LAR,
            p_S_dry_LAR=p_S_dry_LAR,
            p_tank_LAR=p_tank_LAR,
            burn_cadence_hr_per_day=burn_cadence,
        )
