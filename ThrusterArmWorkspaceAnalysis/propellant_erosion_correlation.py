#!/usr/bin/env python3
"""
Propellant Budget ↔ Plume Erosion Correlation Script
=====================================================
Couples the stationkeeping propellant budget with time-resolved plume
erosion analysis, accounting for COG migration as propellant is consumed
from the servicer.

Integrates with plume_impingement_pipeline.py for geometry and erosion
calculations.

Key features:
  - GEO stationkeeping delta-V model (NSSK + EWSK) with custom override
  - Rocket equation for electric propulsion (Tsiolkovsky with mass ratio)
  - Time-resolved COG tracking as propellant depletes
  - Manoeuvre-by-manoeuvre erosion accumulation
  - Propellant-limited vs erosion-limited mission duration
  - Correlation heatmaps: propellant budget vs erosion margin

Author: Plume Impingement Analysis Framework
"""

import numpy as np
import os
import json
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings

# Import the base pipeline
from plume_impingement_pipeline import (
    ThrusterParams, ArmGeometry, StackConfig, OperationalParams,
    MaterialParams, GeometryEngine, ErosionEstimator
)


# ---------------------------------------------------------------------------
# 1.  STATIONKEEPING BUDGET MODEL
# ---------------------------------------------------------------------------

@dataclass
class StationkeepingBudget:
    """GEO stationkeeping delta-V requirements.

    Default values are for standard GEO telecom operations.
    All delta-V values in m/s per year.
    """
    # North-South stationkeeping (orbit plane correction)
    nssk_dv_per_year: float = 55.0       # m/s/yr  (dominant, ~46-53 for GEO)

    # East-West stationkeeping (longitude maintenance)
    ewsk_dv_per_year: float = 1.5        # m/s/yr  (small, ~1-5 depending on slot)

    # Momentum management (wheel desaturation via thrusters)
    momentum_dv_per_year: float = 0.5    # m/s/yr

    # Relocation / repositioning budget (one-time or periodic)
    relocation_dv: float = 0.0           # m/s total over mission

    # Margin
    margin_fraction: float = 0.10        # 10% margin on total

    # Manoeuvre scheduling
    nssk_manoeuvres_per_day: int = 2     # typically 2 burns per orbit for NSSK
    ewsk_manoeuvres_per_week: int = 2    # EWSK less frequent
    nssk_firing_per_manoeuvre_s: float = 0.0  # computed if 0
    ewsk_firing_per_manoeuvre_s: float = 0.0  # computed if 0

    def total_dv_per_year(self) -> float:
        """Total delta-V per year including margin."""
        base = self.nssk_dv_per_year + self.ewsk_dv_per_year + self.momentum_dv_per_year
        return base * (1.0 + self.margin_fraction)

    def total_dv_mission(self, years: float) -> float:
        """Total delta-V for the full mission."""
        return self.total_dv_per_year() * years + self.relocation_dv

    def nssk_fraction(self) -> float:
        """Fraction of delta-V allocated to NSSK."""
        total = self.nssk_dv_per_year + self.ewsk_dv_per_year + self.momentum_dv_per_year
        return self.nssk_dv_per_year / total if total > 0 else 0.0

    def ewsk_fraction(self) -> float:
        """Fraction of delta-V allocated to EWSK."""
        total = self.nssk_dv_per_year + self.ewsk_dv_per_year + self.momentum_dv_per_year
        return self.ewsk_dv_per_year / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# 2.  PROPELLANT BUDGET CALCULATOR
# ---------------------------------------------------------------------------

@dataclass
class PropellantConfig:
    """Propellant system configuration (all on servicer)."""
    # Tank
    tank_capacity_kg: float = 150.0       # max Xe capacity
    propellant_loaded_kg: float = 120.0   # actual loaded propellant
    tank_position_x: float = 0.0          # tank CG relative to servicer geometric centre
    tank_position_y: float = 0.0
    tank_position_z: float = 0.0          # typically near servicer CG

    # Residual
    residual_fraction: float = 0.03       # 3% unusable residual

    def usable_propellant_kg(self) -> float:
        return self.propellant_loaded_kg * (1.0 - self.residual_fraction)


class PropellantBudgetCalculator:
    """Computes propellant consumption and manoeuvre scheduling."""

    def __init__(self, thruster: ThrusterParams, sk_budget: StationkeepingBudget,
                 prop_config: PropellantConfig, stack: StackConfig):
        self.thruster = thruster
        self.sk = sk_budget
        self.prop = prop_config
        self.stack = stack

    def exhaust_velocity(self) -> float:
        """Effective exhaust velocity [m/s]."""
        g0 = 9.80665
        return self.thruster.isp * g0

    def propellant_for_dv(self, dv: float, initial_mass: float) -> float:
        """Propellant mass required for a given delta-V [kg].
        Tsiolkovsky: dv = ve * ln(m0/mf)  =>  mp = m0 * (1 - exp(-dv/ve))
        """
        ve = self.exhaust_velocity()
        mp = initial_mass * (1.0 - np.exp(-dv / ve))
        return mp

    def dv_from_propellant(self, mp: float, initial_mass: float) -> float:
        """Delta-V achievable from a given propellant mass [m/s]."""
        ve = self.exhaust_velocity()
        if mp >= initial_mass:
            return 0.0
        return ve * np.log(initial_mass / (initial_mass - mp))

    def initial_stack_mass(self) -> float:
        """Total initial stack mass [kg]."""
        return self.stack.servicer_mass + self.stack.client_mass

    def firing_duration_for_dv(self, dv: float, current_mass: float) -> float:
        """Firing time needed for a given delta-V at current mass [s].
        F = m_dot * ve  =>  a = F/m  =>  t ≈ dv / a  (for small dv)
        More precisely: t = mp / m_dot
        """
        mp = self.propellant_for_dv(dv, current_mass)
        return mp / self.thruster.mass_flow_rate

    def mission_propellant_budget(self, mission_years: float) -> Dict:
        """Complete propellant budget breakdown."""
        m0 = self.initial_stack_mass()
        total_dv = self.sk.total_dv_mission(mission_years)

        # Iterative computation accounting for mass decrease
        # Split into yearly chunks for better accuracy
        remaining_prop = self.prop.usable_propellant_kg()
        current_mass = m0
        yearly_consumption = []

        for yr in range(int(np.ceil(mission_years))):
            frac = min(1.0, mission_years - yr)  # handle partial last year
            dv_year = self.sk.total_dv_per_year() * frac
            mp_year = self.propellant_for_dv(dv_year, current_mass)

            if mp_year > remaining_prop:
                # Propellant exhaustion — compute how much dv we can still do
                dv_achievable = self.dv_from_propellant(remaining_prop, current_mass)
                partial_year_frac = dv_achievable / self.sk.total_dv_per_year()
                yearly_consumption.append({
                    "year": yr + 1,
                    "dv_planned": dv_year,
                    "dv_achieved": dv_achievable,
                    "propellant_kg": remaining_prop,
                    "mass_start": current_mass,
                    "mass_end": current_mass - remaining_prop,
                    "partial": True,
                    "year_fraction": partial_year_frac,
                })
                remaining_prop = 0.0
                current_mass -= yearly_consumption[-1]["propellant_kg"]
                break

            yearly_consumption.append({
                "year": yr + 1,
                "dv_planned": dv_year,
                "dv_achieved": dv_year,
                "propellant_kg": mp_year,
                "mass_start": current_mass,
                "mass_end": current_mass - mp_year,
                "partial": False,
                "year_fraction": frac,
            })
            current_mass -= mp_year
            remaining_prop -= mp_year

        total_prop_used = sum(y["propellant_kg"] for y in yearly_consumption)
        total_dv_achieved = sum(y["dv_achieved"] for y in yearly_consumption)
        effective_years = sum(y["year_fraction"] if y["partial"] else
                             min(1.0, mission_years - y["year"] + 1)
                             for y in yearly_consumption)

        return {
            "initial_mass_kg": m0,
            "propellant_loaded_kg": self.prop.propellant_loaded_kg,
            "propellant_usable_kg": self.prop.usable_propellant_kg(),
            "total_dv_required_ms": total_dv,
            "total_dv_achieved_ms": total_dv_achieved,
            "total_propellant_used_kg": total_prop_used,
            "propellant_remaining_kg": self.prop.usable_propellant_kg() - total_prop_used,
            "mission_years_requested": mission_years,
            "mission_years_achievable": effective_years,
            "propellant_limited": remaining_prop <= 0,
            "yearly_breakdown": yearly_consumption,
        }


# ---------------------------------------------------------------------------
# 3.  COG MIGRATION TRACKER
# ---------------------------------------------------------------------------

class COGTracker:
    """Tracks centre-of-gravity migration as propellant is consumed."""

    def __init__(self, stack: StackConfig, prop_config: PropellantConfig):
        self.stack = stack
        self.prop = prop_config

    def servicer_origin_in_client_frame(self) -> np.ndarray:
        """Servicer geometric centre position in client body frame."""
        return np.array([
            self.stack.dock_offset_x,
            0.0,
            self.stack.client_bus_z / 2.0 + self.stack.servicer_bus_z / 2.0
            + self.stack.dock_offset_z
        ])

    def tank_position_in_client_frame(self) -> np.ndarray:
        """Propellant tank CG in client body frame."""
        serv_origin = self.servicer_origin_in_client_frame()
        return serv_origin + np.array([
            self.prop.tank_position_x,
            self.prop.tank_position_y,
            self.prop.tank_position_z
        ])

    def cog_at_propellant_level(self, propellant_remaining_kg: float) -> np.ndarray:
        """Stack COG for a given propellant fill level.

        The servicer dry mass + client mass have fixed CG positions.
        The propellant mass is at the tank position and decreases over time.
        """
        # Client CG at origin
        client_cg = np.array([0.0, 0.0, 0.0])
        client_mass = self.stack.client_mass

        # Servicer dry mass (total servicer mass minus initial propellant)
        servicer_dry = self.stack.servicer_mass - self.prop.propellant_loaded_kg
        servicer_dry_cg = self.servicer_origin_in_client_frame()

        # Propellant mass at tank position
        tank_cg = self.tank_position_in_client_frame()
        prop_mass = propellant_remaining_kg

        total_mass = client_mass + servicer_dry + prop_mass
        if total_mass <= 0:
            return np.array([0.0, 0.0, 0.0])

        cog = (client_mass * client_cg
               + servicer_dry * servicer_dry_cg
               + prop_mass * tank_cg) / total_mass

        return cog

    def cog_trajectory(self, yearly_breakdown: List[Dict],
                       steps_per_year: int = 12) -> List[Dict]:
        """Compute COG position at regular intervals over the mission.

        Returns list of {time_years, propellant_kg, cog_xyz, stack_mass_kg}.
        """
        trajectory = []
        prop_remaining = self.prop.usable_propellant_kg()

        for year_data in yearly_breakdown:
            yr = year_data["year"]
            prop_used_this_year = year_data["propellant_kg"]
            frac = year_data.get("year_fraction", 1.0)
            n_steps = max(1, int(steps_per_year * frac))

            for step in range(n_steps):
                t = (yr - 1) + (step / n_steps) * frac
                # Linear propellant consumption within the year
                prop_at_step = prop_remaining - prop_used_this_year * (step / n_steps)
                cog = self.cog_at_propellant_level(prop_at_step)
                mass = (self.stack.client_mass
                        + (self.stack.servicer_mass - self.prop.propellant_loaded_kg)
                        + prop_at_step)

                trajectory.append({
                    "time_years": round(t, 4),
                    "propellant_remaining_kg": round(prop_at_step, 3),
                    "cog_x": round(cog[0], 6),
                    "cog_y": round(cog[1], 6),
                    "cog_z": round(cog[2], 6),
                    "stack_mass_kg": round(mass, 3),
                })

            prop_remaining -= prop_used_this_year

        return trajectory


# ---------------------------------------------------------------------------
# 4.  TIME-RESOLVED EROSION INTEGRATOR
# ---------------------------------------------------------------------------

class TimeResolvedErosion:
    """Integrates erosion manoeuvre-by-manoeuvre with evolving COG."""

    def __init__(self, thruster: ThrusterParams, material: MaterialParams,
                 arm: ArmGeometry, stack: StackConfig,
                 prop_config: PropellantConfig, sk_budget: StationkeepingBudget):
        self.thruster = thruster
        self.material = material
        self.arm = arm
        self.stack = stack
        self.prop = prop_config
        self.sk = sk_budget
        self.estimator = ErosionEstimator(thruster, material)
        self.cog_tracker = COGTracker(stack, prop_config)
        self.budget_calc = PropellantBudgetCalculator(
            thruster, sk_budget, prop_config, stack
        )

    def _make_stack_at_propellant_level(self, prop_remaining_kg: float) -> StackConfig:
        """Create a modified StackConfig with adjusted servicer mass
        to reflect current propellant level (for COG calculation)."""
        consumed = self.prop.propellant_loaded_kg - prop_remaining_kg
        adjusted_stack = StackConfig(
            servicer_mass=self.stack.servicer_mass - consumed,
            servicer_bus_x=self.stack.servicer_bus_x,
            servicer_bus_y=self.stack.servicer_bus_y,
            servicer_bus_z=self.stack.servicer_bus_z,
            client_mass=self.stack.client_mass,
            client_bus_x=self.stack.client_bus_x,
            client_bus_y=self.stack.client_bus_y,
            client_bus_z=self.stack.client_bus_z,
            panel_span_one_side=self.stack.panel_span_one_side,
            panel_width=self.stack.panel_width,
            panel_hinge_offset_y=self.stack.panel_hinge_offset_y,
            panel_cant_angle_deg=self.stack.panel_cant_angle_deg,
            dock_offset_z=self.stack.dock_offset_z,
            dock_offset_x=self.stack.dock_offset_x,
            antenna_diameter=self.stack.antenna_diameter,
            antenna_offset_x=self.stack.antenna_offset_x,
            antenna_offset_z=self.stack.antenna_offset_z,
        )
        return adjusted_stack

    def integrate_mission(self, mission_years: float,
                          time_step_days: float = 30.0,
                          panel_tracking_schedule: Optional[Dict] = None,
                          verbose: bool = True) -> Dict:
        """Integrate erosion over the full mission with COG migration.

        Parameters
        ----------
        mission_years : float
            Requested mission duration.
        time_step_days : float
            Time resolution for the integration (default 30 days).
        panel_tracking_schedule : dict, optional
            Maps manoeuvre type to panel tracking angle:
            {"NSSK": 0.0, "EWSK": 15.0}  (degrees)
        verbose : bool
            Print progress.

        Returns
        -------
        dict with time history of erosion, COG, propellant, and summary.
        """
        if panel_tracking_schedule is None:
            panel_tracking_schedule = {"NSSK": 0.0, "EWSK": 0.0}

        # Compute propellant budget
        budget = self.budget_calc.mission_propellant_budget(mission_years)
        achievable_years = budget["mission_years_achievable"]

        # Time stepping
        dt_years = time_step_days / 365.25
        n_steps = int(np.ceil(achievable_years / dt_years))
        prop_total = self.prop.usable_propellant_kg()
        prop_rate_per_year = budget["total_propellant_used_kg"] / max(achievable_years, 0.01)

        # Manoeuvre counts per time step
        nssk_frac = self.sk.nssk_fraction()
        ewsk_frac = self.sk.ewsk_fraction()
        nssk_firings_per_step = self.sk.nssk_manoeuvres_per_day * time_step_days
        ewsk_firings_per_step = self.sk.ewsk_manoeuvres_per_week * (time_step_days / 7.0)

        # Delta-V per manoeuvre
        nssk_dv_per_day = self.sk.nssk_dv_per_year / 365.25
        ewsk_dv_per_day = self.sk.ewsk_dv_per_year / 365.25

        # Results accumulators
        time_history = []
        cumulative_erosion_um = 0.0
        cumulative_erosion_map = None  # will hold panel grid erosion
        prop_remaining = prop_total
        mission_failed = False
        failure_time_years = None

        for step in range(n_steps):
            t_years = step * dt_years
            if prop_remaining <= 0:
                break

            # Current stack state
            current_stack = self._make_stack_at_propellant_level(prop_remaining)
            current_mass = current_stack.servicer_mass + current_stack.client_mass
            cog = current_stack.stack_cog()

            # Geometry at current COG
            geo = GeometryEngine(self.arm, current_stack)

            # --- NSSK erosion for this time step ---
            nssk_tracking = panel_tracking_schedule.get("NSSK", 0.0)
            nssk_geo = geo.compute_flux_geometry(
                sun_tracking_angle_deg=nssk_tracking,
                n_spanwise=30, n_chordwise=6
            )

            # Firing duration per NSSK manoeuvre at current mass
            nssk_dv_per_manoeuvre = nssk_dv_per_day / max(self.sk.nssk_manoeuvres_per_day, 1)
            nssk_firing_s = self.budget_calc.firing_duration_for_dv(
                nssk_dv_per_manoeuvre, current_mass
            )

            # Total NSSK firing time this step
            total_nssk_firing_s = nssk_firing_s * nssk_firings_per_step

            # Compute erosion at worst panel point for NSSK
            n_pts = len(nssk_geo["distances_m"])
            nssk_erosions = np.zeros(n_pts)
            for i in range(n_pts):
                rate = self.estimator.erosion_rate_um_per_s(
                    nssk_geo["distances_m"][i],
                    nssk_geo["offaxis_angles_deg"][i],
                    nssk_geo["incidence_angles_deg"][i]
                )
                nssk_erosions[i] = rate * total_nssk_firing_s

            # --- EWSK erosion for this time step ---
            ewsk_tracking = panel_tracking_schedule.get("EWSK", 0.0)
            ewsk_geo = geo.compute_flux_geometry(
                sun_tracking_angle_deg=ewsk_tracking,
                n_spanwise=30, n_chordwise=6
            )

            ewsk_dv_per_manoeuvre = (ewsk_dv_per_day * 7.0) / max(self.sk.ewsk_manoeuvres_per_week, 1)
            ewsk_firing_s = self.budget_calc.firing_duration_for_dv(
                ewsk_dv_per_manoeuvre, current_mass
            )
            total_ewsk_firing_s = ewsk_firing_s * ewsk_firings_per_step

            ewsk_erosions = np.zeros(n_pts)
            for i in range(n_pts):
                rate = self.estimator.erosion_rate_um_per_s(
                    ewsk_geo["distances_m"][i],
                    ewsk_geo["offaxis_angles_deg"][i],
                    ewsk_geo["incidence_angles_deg"][i]
                )
                ewsk_erosions[i] = rate * total_ewsk_firing_s

            # Combined erosion this step
            step_erosions = nssk_erosions + ewsk_erosions
            step_max_erosion = np.max(step_erosions)

            if cumulative_erosion_map is None:
                cumulative_erosion_map = step_erosions.copy()
            else:
                cumulative_erosion_map += step_erosions

            cumulative_max = np.max(cumulative_erosion_map)

            # Propellant consumed this step
            total_firing_s = total_nssk_firing_s + total_ewsk_firing_s
            prop_consumed_step = self.thruster.mass_flow_rate * total_firing_s
            prop_remaining -= prop_consumed_step

            # Check failure
            if cumulative_max >= self.material.thickness_um and not mission_failed:
                mission_failed = True
                failure_time_years = t_years

            # Record
            time_history.append({
                "time_years": round(t_years, 4),
                "time_days": round(t_years * 365.25, 1),
                "propellant_remaining_kg": round(max(prop_remaining, 0), 3),
                "stack_mass_kg": round(current_mass, 1),
                "cog_x": round(cog[0], 4),
                "cog_y": round(cog[1], 4),
                "cog_z": round(cog[2], 4),
                "nssk_firing_per_manoeuvre_s": round(nssk_firing_s, 1),
                "ewsk_firing_per_manoeuvre_s": round(ewsk_firing_s, 1),
                "step_max_erosion_um": round(step_max_erosion, 4),
                "cumulative_max_erosion_um": round(cumulative_max, 4),
                "erosion_fraction": round(cumulative_max / self.material.thickness_um, 4),
                "status": ("FAIL" if cumulative_max >= self.material.thickness_um else
                          "MARGINAL" if cumulative_max >= 0.5 * self.material.thickness_um else
                          "CAUTION" if cumulative_max >= 0.1 * self.material.thickness_um else
                          "SAFE"),
            })

            if verbose and (step + 1) % max(1, n_steps // 10) == 0:
                print(f"  t={t_years:.1f}yr  prop={prop_remaining:.1f}kg  "
                      f"erosion={cumulative_max:.3f}µm  COG_z={cog[2]:.4f}m")

        # --- Summary ---
        prop_used = prop_total - max(prop_remaining, 0)
        summary = {
            "mission_years_requested": mission_years,
            "mission_years_propellant_limited": budget["mission_years_achievable"],
            "mission_years_erosion_limited": failure_time_years if mission_failed else achievable_years,
            "mission_years_actual": min(
                failure_time_years if mission_failed else float("inf"),
                budget["mission_years_achievable"]
            ),
            "limiting_factor": ("EROSION" if mission_failed and
                                (failure_time_years or float("inf")) <= budget["mission_years_achievable"]
                                else "PROPELLANT"),
            "propellant_used_kg": round(prop_used, 3),
            "propellant_remaining_kg": round(max(prop_remaining, 0), 3),
            "total_dv_achieved_ms": round(budget["total_dv_achieved_ms"], 2),
            "max_erosion_um": round(np.max(cumulative_erosion_map) if cumulative_erosion_map is not None else 0, 4),
            "erosion_fraction": round(
                (np.max(cumulative_erosion_map) / self.material.thickness_um)
                if cumulative_erosion_map is not None else 0, 4
            ),
            "erosion_failed": mission_failed,
            "cog_shift_z_mm": round(
                (time_history[-1]["cog_z"] - time_history[0]["cog_z"]) * 1000, 1
            ) if time_history else 0,
            "initial_cog_z": round(time_history[0]["cog_z"], 4) if time_history else 0,
            "final_cog_z": round(time_history[-1]["cog_z"], 4) if time_history else 0,
        }

        return {
            "summary": summary,
            "budget": budget,
            "time_history": time_history,
        }


# ---------------------------------------------------------------------------
# 5.  PARAMETRIC SWEEP WITH PROPELLANT CORRELATION
# ---------------------------------------------------------------------------

class PropellantErosionSweep:
    """Sweeps propellant budget vs geometric/operational parameters."""

    def __init__(self, thruster: ThrusterParams = None,
                 material: MaterialParams = None):
        self.thruster = thruster or ThrusterParams()
        self.material = material or MaterialParams()
        self.results: List[Dict] = []

    def sweep_propellant_vs_mission(
        self,
        arm: ArmGeometry,
        stack: StackConfig,
        sk_budget: StationkeepingBudget,
        propellant_range_kg: np.ndarray = np.arange(40, 200, 20),
        mission_range_yr: np.ndarray = np.arange(2, 12, 1),
        verbose: bool = True
    ) -> List[Dict]:
        """Sweep propellant load vs mission duration.

        For each combination, determine:
        - Is mission propellant-limited or erosion-limited?
        - What is the actual achievable mission duration?
        - What is the final erosion state?
        """
        results = []
        total = len(propellant_range_kg) * len(mission_range_yr)
        count = 0

        for prop_kg in propellant_range_kg:
            for mission_yr in mission_range_yr:
                count += 1
                prop_config = PropellantConfig(
                    propellant_loaded_kg=float(prop_kg),
                    tank_capacity_kg=float(prop_kg) * 1.1,
                )

                # Ensure servicer mass accounts for propellant
                adjusted_stack = StackConfig(
                    servicer_mass=stack.servicer_mass - stack.servicer_mass * 0.0
                                  + prop_kg * 0.0,  # keep servicer dry mass constant
                    **{k: getattr(stack, k) for k in [
                        'servicer_bus_x', 'servicer_bus_y', 'servicer_bus_z',
                        'client_mass', 'client_bus_x', 'client_bus_y', 'client_bus_z',
                        'panel_span_one_side', 'panel_width', 'panel_hinge_offset_y',
                        'panel_cant_angle_deg', 'dock_offset_z', 'dock_offset_x',
                        'antenna_diameter', 'antenna_offset_x', 'antenna_offset_z'
                    ]}
                )
                # Servicer mass = dry mass + propellant
                servicer_dry = 280.0  # fixed dry mass
                adjusted_stack.servicer_mass = servicer_dry + prop_kg

                integrator = TimeResolvedErosion(
                    self.thruster, self.material, arm, adjusted_stack,
                    prop_config, sk_budget
                )

                result = integrator.integrate_mission(
                    mission_years=float(mission_yr),
                    time_step_days=60.0,  # coarser for sweep speed
                    verbose=False
                )

                s = result["summary"]
                results.append({
                    "propellant_loaded_kg": float(prop_kg),
                    "mission_requested_yr": float(mission_yr),
                    "mission_achievable_yr": s["mission_years_actual"],
                    "limiting_factor": s["limiting_factor"],
                    "max_erosion_um": s["max_erosion_um"],
                    "erosion_fraction": s["erosion_fraction"],
                    "propellant_used_kg": s["propellant_used_kg"],
                    "propellant_remaining_kg": s["propellant_remaining_kg"],
                    "total_dv_ms": s["total_dv_achieved_ms"],
                    "cog_shift_z_mm": s["cog_shift_z_mm"],
                    "erosion_failed": s["erosion_failed"],
                    "status": ("FAIL" if s["erosion_failed"] else
                              "MARGINAL" if s["erosion_fraction"] >= 0.5 else
                              "CAUTION" if s["erosion_fraction"] >= 0.1 else
                              "SAFE"),
                })

                if verbose and count % max(1, total // 10) == 0:
                    print(f"  [{count}/{total}] prop={prop_kg:.0f}kg, "
                          f"mission={mission_yr:.0f}yr → {s['limiting_factor']}, "
                          f"erosion={s['max_erosion_um']:.2f}µm")

        self.results = results
        return results

    def sweep_propellant_vs_arm_length(
        self,
        stack: StackConfig,
        sk_budget: StationkeepingBudget,
        propellant_range_kg: np.ndarray = np.arange(40, 200, 20),
        arm_length_range_m: np.ndarray = np.arange(1.0, 4.5, 0.5),
        mission_years: float = 5.0,
        arm_azimuth_deg: float = 0.0,
        arm_elevation_deg: float = 0.0,
        verbose: bool = True
    ) -> List[Dict]:
        """Sweep propellant load vs arm length at fixed mission duration."""
        results = []
        total = len(propellant_range_kg) * len(arm_length_range_m)
        count = 0

        for prop_kg in propellant_range_kg:
            for arm_len in arm_length_range_m:
                count += 1
                arm = ArmGeometry(
                    arm_length=float(arm_len),
                    azimuth_deg=arm_azimuth_deg,
                    elevation_deg=arm_elevation_deg,
                )
                prop_config = PropellantConfig(
                    propellant_loaded_kg=float(prop_kg),
                    tank_capacity_kg=float(prop_kg) * 1.1,
                )
                servicer_dry = 280.0
                adjusted_stack = StackConfig(
                    servicer_mass=servicer_dry + prop_kg,
                    **{k: getattr(stack, k) for k in [
                        'servicer_bus_x', 'servicer_bus_y', 'servicer_bus_z',
                        'client_mass', 'client_bus_x', 'client_bus_y', 'client_bus_z',
                        'panel_span_one_side', 'panel_width', 'panel_hinge_offset_y',
                        'panel_cant_angle_deg', 'dock_offset_z', 'dock_offset_x',
                        'antenna_diameter', 'antenna_offset_x', 'antenna_offset_z'
                    ]}
                )

                integrator = TimeResolvedErosion(
                    self.thruster, self.material, arm, adjusted_stack,
                    prop_config, sk_budget
                )
                result = integrator.integrate_mission(
                    mission_years=mission_years,
                    time_step_days=60.0,
                    verbose=False
                )
                s = result["summary"]
                results.append({
                    "propellant_loaded_kg": float(prop_kg),
                    "arm_length_m": float(arm_len),
                    "mission_achievable_yr": s["mission_years_actual"],
                    "limiting_factor": s["limiting_factor"],
                    "max_erosion_um": s["max_erosion_um"],
                    "erosion_fraction": s["erosion_fraction"],
                    "cog_shift_z_mm": s["cog_shift_z_mm"],
                    "status": ("FAIL" if s["erosion_failed"] else
                              "MARGINAL" if s["erosion_fraction"] >= 0.5 else
                              "CAUTION" if s["erosion_fraction"] >= 0.1 else
                              "SAFE"),
                })

                if verbose and count % max(1, total // 10) == 0:
                    print(f"  [{count}/{total}] prop={prop_kg:.0f}kg, "
                          f"arm={arm_len:.1f}m → {s['limiting_factor']}")

        self.results = results
        return results


# ---------------------------------------------------------------------------
# 6.  VISUALISATION
# ---------------------------------------------------------------------------

def plot_time_resolved_erosion(mission_result: Dict, output_dir: str = ".",
                                show_plot: bool = False) -> List[str]:
    """Generate time-resolved plots from a single mission integration."""
    import matplotlib
    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = mission_result["time_history"]
    if not th:
        return []

    t = [r["time_years"] for r in th]
    erosion = [r["cumulative_max_erosion_um"] for r in th]
    prop = [r["propellant_remaining_kg"] for r in th]
    cog_z = [r["cog_z"] for r in th]
    nssk_fire = [r["nssk_firing_per_manoeuvre_s"] for r in th]
    mass = [r["stack_mass_kg"] for r in th]
    thickness = 25.0  # default, could parameterize

    files = []

    # --- Plot 1: Erosion + Propellant vs Time (dual axis) ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    color1 = "#E74C3C"
    color2 = "#3498DB"

    ax1.plot(t, erosion, color=color1, linewidth=2.5, label="Cumulative Erosion")
    ax1.axhline(y=thickness, color=color1, linestyle="--", alpha=0.6,
                label=f"Ag Thickness ({thickness} µm)")
    ax1.axhline(y=0.5 * thickness, color="#E67E22", linestyle=":", alpha=0.5,
                label="50% Threshold (MARGINAL)")
    ax1.fill_between(t, thickness, max(max(erosion) * 1.1, thickness * 1.3),
                     alpha=0.1, color="red", label="FAIL Zone")

    ax2.plot(t, prop, color=color2, linewidth=2.5, linestyle="-.", label="Propellant")

    ax1.set_xlabel("Mission Time [years]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cumulative Max Erosion [µm]", fontsize=12, color=color1)
    ax2.set_ylabel("Propellant Remaining [kg]", fontsize=12, color=color2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    s = mission_result["summary"]
    title = (f"Erosion & Propellant vs Time — "
             f"Limited by: {s['limiting_factor']}")
    ax1.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    fpath = os.path.join(output_dir, "time_erosion_propellant.png")
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    files.append(fpath)
    if not show_plot:
        plt.close(fig)

    # --- Plot 2: COG Migration + Firing Duration ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t, [z * 1000 for z in cog_z], color="#2ECC71", linewidth=2.5)
    ax1.set_ylabel("Stack COG Z-position [mm]", fontsize=11, fontweight="bold")
    ax1.set_title("COG Migration & Manoeuvre Duration Over Mission", fontsize=13,
                  fontweight="bold")
    cog_shift = (cog_z[-1] - cog_z[0]) * 1000
    ax1.annotate(f"Total COG shift: {cog_shift:.1f} mm",
                 xy=(0.98, 0.05), xycoords="axes fraction",
                 fontsize=10, ha="right",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, nssk_fire, color="#9B59B6", linewidth=2.5, label="NSSK firing/manoeuvre")
    ax2.set_xlabel("Mission Time [years]", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Firing Duration per Manoeuvre [s]", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(output_dir, "time_cog_firing.png")
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    files.append(fpath)
    if not show_plot:
        plt.close(fig)

    # --- Plot 3: Erosion rate (derivative) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    step_erosions = [r["step_max_erosion_um"] for r in th]
    dt_yr = t[1] - t[0] if len(t) > 1 else 1.0
    rates = [e / (dt_yr * 365.25 * 24 * 3600) * 1e6 for e in step_erosions]  # nm/s equivalent

    ax.plot(t, step_erosions, color="#E67E22", linewidth=2)
    ax.set_xlabel("Mission Time [years]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Erosion per Time Step [µm]", fontsize=11, fontweight="bold")
    ax.set_title("Erosion Rate Evolution (shows COG drift effect)", fontsize=13,
                 fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(output_dir, "time_erosion_rate.png")
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    files.append(fpath)
    if not show_plot:
        plt.close(fig)

    print(f"Time-resolved plots saved to {output_dir}")
    return files


def plot_propellant_erosion_correlation(results: List[Dict],
                                        param_x: str,
                                        param_y: str,
                                        output_dir: str = ".",
                                        show_plot: bool = False) -> List[str]:
    """Generate correlation heatmaps from sweep results."""
    import matplotlib
    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

    files = []
    param_labels = {
        "propellant_loaded_kg": "Propellant Loaded [kg]",
        "mission_requested_yr": "Mission Duration (Requested) [yr]",
        "mission_achievable_yr": "Mission Duration (Achievable) [yr]",
        "arm_length_m": "Arm Length [m]",
        "max_erosion_um": "Max Erosion [µm]",
        "erosion_fraction": "Erosion Fraction",
        "cog_shift_z_mm": "COG Shift Z [mm]",
    }

    x_vals = sorted(set(r[param_x] for r in results))
    y_vals = sorted(set(r[param_y] for r in results))

    # --- Heatmap 1: Achievable mission duration ---
    Z_mission = np.full((len(y_vals), len(x_vals)), np.nan)
    Z_erosion = np.full((len(y_vals), len(x_vals)), np.nan)
    Z_status = np.full((len(y_vals), len(x_vals)), np.nan)
    Z_limiting = np.full((len(y_vals), len(x_vals)), np.nan)
    Z_cog = np.full((len(y_vals), len(x_vals)), np.nan)

    status_map = {"SAFE": 0, "CAUTION": 1, "MARGINAL": 2, "FAIL": 3}
    limit_map = {"PROPELLANT": 0, "EROSION": 1}

    for r in results:
        ix = x_vals.index(r[param_x])
        iy = y_vals.index(r[param_y])
        Z_mission[iy, ix] = r.get("mission_achievable_yr", np.nan)
        Z_erosion[iy, ix] = r.get("max_erosion_um", np.nan)
        Z_status[iy, ix] = status_map.get(r.get("status", "SAFE"), 0)
        Z_limiting[iy, ix] = limit_map.get(r.get("limiting_factor", "PROPELLANT"), 0)
        Z_cog[iy, ix] = r.get("cog_shift_z_mm", np.nan)

    # --- Plot: Achievable mission duration ---
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap_dur = LinearSegmentedColormap.from_list("duration",
        [(0.85, 0.15, 0.15), (0.95, 0.85, 0.20), (0.18, 0.65, 0.35)], N=256)

    vmax = np.nanmax(Z_mission) if np.any(~np.isnan(Z_mission)) else 10
    im = ax.imshow(Z_mission, origin="lower", aspect="auto", cmap=cmap_dur,
                   vmin=0, vmax=vmax,
                   extent=[min(x_vals) - 0.5 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else min(x_vals) - 0.5,
                           max(x_vals) + 0.5 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else max(x_vals) + 0.5,
                           min(y_vals) - 0.5 * (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else min(y_vals) - 0.5,
                           max(y_vals) + 0.5 * (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else max(y_vals) + 0.5])

    # Annotate cells
    if len(x_vals) <= 15 and len(y_vals) <= 15:
        for iy, yv in enumerate(y_vals):
            for ix, xv in enumerate(x_vals):
                val = Z_mission[iy, ix]
                lim = Z_limiting[iy, ix]
                if not np.isnan(val):
                    lim_str = "P" if lim == 0 else "E"
                    color = "white" if val < vmax * 0.4 else "black"
                    ax.text(xv, yv, f"{val:.1f}\n({lim_str})", ha="center",
                            va="center", fontsize=7, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Achievable Mission Duration [yr]", fontsize=11)

    ax.set_xlabel(param_labels.get(param_x, param_x), fontsize=12, fontweight="bold")
    ax.set_ylabel(param_labels.get(param_y, param_y), fontsize=12, fontweight="bold")
    ax.set_title("Achievable Mission Duration\n(P = propellant-limited, E = erosion-limited)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    fpath = os.path.join(output_dir, f"corr_mission_{param_x}_vs_{param_y}.png")
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    files.append(fpath)
    if not show_plot:
        plt.close(fig)

    # --- Plot: Erosion depth ---
    fig, ax = plt.subplots(figsize=(12, 8))
    colors_ero = [(0.18, 0.65, 0.35), (0.95, 0.85, 0.20),
                  (0.95, 0.55, 0.15), (0.85, 0.15, 0.15)]
    cmap_ero = LinearSegmentedColormap.from_list("erosion", colors_ero, N=256)
    vmax_e = max(25.0 * 1.2, np.nanmax(Z_erosion) * 1.05) if np.any(~np.isnan(Z_erosion)) else 30

    im = ax.imshow(Z_erosion, origin="lower", aspect="auto", cmap=cmap_ero,
                   vmin=0, vmax=vmax_e,
                   extent=[min(x_vals) - 0.5 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else min(x_vals) - 0.5,
                           max(x_vals) + 0.5 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else max(x_vals) + 0.5,
                           min(y_vals) - 0.5 * (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else min(y_vals) - 0.5,
                           max(y_vals) + 0.5 * (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else max(y_vals) + 0.5])

    if len(x_vals) > 1 and len(y_vals) > 1:
        X_g, Y_g = np.meshgrid(x_vals, y_vals)
        try:
            cs = ax.contour(X_g, Y_g, Z_erosion, levels=[25.0],
                            colors="white", linewidths=2.5, linestyles="--")
            ax.clabel(cs, fmt="25 µm FAIL", fontsize=9, colors="white")
        except Exception:
            pass

    if len(x_vals) <= 15 and len(y_vals) <= 15:
        for iy, yv in enumerate(y_vals):
            for ix, xv in enumerate(x_vals):
                val = Z_erosion[iy, ix]
                if not np.isnan(val):
                    color = "white" if val > 0.5 * vmax_e else "black"
                    ax.text(xv, yv, f"{val:.1f}", ha="center",
                            va="center", fontsize=7, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Max Erosion Depth [µm]", fontsize=11)
    cbar.ax.axhline(y=25.0, color="white", linewidth=2, linestyle="--")

    ax.set_xlabel(param_labels.get(param_x, param_x), fontsize=12, fontweight="bold")
    ax.set_ylabel(param_labels.get(param_y, param_y), fontsize=12, fontweight="bold")
    ax.set_title("Max Erosion vs Propellant Budget\n(Ag threshold = 25 µm)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    fpath = os.path.join(output_dir, f"corr_erosion_{param_x}_vs_{param_y}.png")
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    files.append(fpath)
    if not show_plot:
        plt.close(fig)

    # --- Plot: Limiting factor map ---
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap_lim = ListedColormap(["#3498DB", "#E74C3C"])
    norm_lim = BoundaryNorm([-0.5, 0.5, 1.5], cmap_lim.N)

    im = ax.imshow(Z_limiting, origin="lower", aspect="auto", cmap=cmap_lim,
                   norm=norm_lim,
                   extent=[min(x_vals) - 0.5 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else min(x_vals) - 0.5,
                           max(x_vals) + 0.5 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else max(x_vals) + 0.5,
                           min(y_vals) - 0.5 * (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else min(y_vals) - 0.5,
                           max(y_vals) + 0.5 * (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else max(y_vals) + 0.5])

    if len(x_vals) <= 15 and len(y_vals) <= 15:
        for iy, yv in enumerate(y_vals):
            for ix, xv in enumerate(x_vals):
                val = Z_limiting[iy, ix]
                dur = Z_mission[iy, ix]
                if not np.isnan(val):
                    lbl = "PROP" if val == 0 else "EROSION"
                    ax.text(xv, yv, f"{lbl}\n{dur:.1f}yr", ha="center",
                            va="center", fontsize=7, color="white", fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498DB", label="Propellant-Limited"),
        Patch(facecolor="#E74C3C", label="Erosion-Limited"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_xlabel(param_labels.get(param_x, param_x), fontsize=12, fontweight="bold")
    ax.set_ylabel(param_labels.get(param_y, param_y), fontsize=12, fontweight="bold")
    ax.set_title("Mission Limiting Factor Map", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fpath = os.path.join(output_dir, f"corr_limiting_{param_x}_vs_{param_y}.png")
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    files.append(fpath)
    if not show_plot:
        plt.close(fig)

    # --- Plot: COG shift ---
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap_cog = "viridis"
    im = ax.imshow(Z_cog, origin="lower", aspect="auto", cmap=cmap_cog,
                   extent=[min(x_vals) - 0.5 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else min(x_vals) - 0.5,
                           max(x_vals) + 0.5 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else max(x_vals) + 0.5,
                           min(y_vals) - 0.5 * (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else min(y_vals) - 0.5,
                           max(y_vals) + 0.5 * (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else max(y_vals) + 0.5])

    if len(x_vals) <= 15 and len(y_vals) <= 15:
        for iy, yv in enumerate(y_vals):
            for ix, xv in enumerate(x_vals):
                val = Z_cog[iy, ix]
                if not np.isnan(val):
                    vmax_c = np.nanmax(Z_cog) if np.any(~np.isnan(Z_cog)) else 1
                    color = "white" if val > 0.5 * vmax_c else "black"
                    ax.text(xv, yv, f"{val:.1f}", ha="center",
                            va="center", fontsize=7, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("COG Z-Shift Over Mission [mm]", fontsize=11)

    ax.set_xlabel(param_labels.get(param_x, param_x), fontsize=12, fontweight="bold")
    ax.set_ylabel(param_labels.get(param_y, param_y), fontsize=12, fontweight="bold")
    ax.set_title("COG Migration Over Mission Lifetime", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fpath = os.path.join(output_dir, f"corr_cogshift_{param_x}_vs_{param_y}.png")
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    files.append(fpath)
    if not show_plot:
        plt.close(fig)

    return files


# ---------------------------------------------------------------------------
# 7.  DEMO / MAIN
# ---------------------------------------------------------------------------

def run_demo():
    """Demonstration of the propellant-erosion correlation pipeline."""

    print("=" * 70)
    print("  PROPELLANT BUDGET ↔ EROSION CORRELATION PIPELINE")
    print("=" * 70)

    output_dir = "/home/karan.anand/Documents/PythonScripts/ThrusterArmWorkspaceAnalysis/output_prop_correlation"
    os.makedirs(output_dir, exist_ok=True)

    # --- Configuration ---
    thruster = ThrusterParams(
        name="SPT-100-like",
        isp=1500.0,
        discharge_voltage=300.0,
        mass_flow_rate=5e-6,
        beam_divergence_half_angle=20.0,
        plume_cosine_exponent=10.0,
        thrust_N=0.08,
    )

    material = MaterialParams(name="Silver_interconnect", thickness_um=25.0)

    sk_budget = StationkeepingBudget(
        nssk_dv_per_year=50.0,
        ewsk_dv_per_year=2.0,
        momentum_dv_per_year=0.5,
        margin_fraction=0.10,
        nssk_manoeuvres_per_day=2,
        ewsk_manoeuvres_per_week=2,
    )

    stack = StackConfig(
        servicer_mass=400.0,  # will be overridden per case
        client_mass=3500.0,
        panel_span_one_side=12.0,
    )

    arm = ArmGeometry(arm_length=2.5, azimuth_deg=0.0, elevation_deg=0.0)

    prop_config = PropellantConfig(propellant_loaded_kg=120.0)

    # ============================================================
    # PART 1: Single mission time-resolved analysis
    # ============================================================
    print("\n[1] TIME-RESOLVED SINGLE MISSION ANALYSIS")
    print("-" * 50)

    integrator = TimeResolvedErosion(
        thruster, material, arm, stack, prop_config, sk_budget
    )
    result = integrator.integrate_mission(
        mission_years=7.0,
        time_step_days=30.0,
        panel_tracking_schedule={"NSSK": 0.0, "EWSK": 15.0},
        verbose=True,
    )

    s = result["summary"]
    print(f"\n  Summary:")
    print(f"    Requested mission:    {s['mission_years_requested']} years")
    print(f"    Propellant-limited:   {s['mission_years_propellant_limited']:.1f} years")
    print(f"    Erosion-limited:      {s['mission_years_erosion_limited']:.1f} years")
    print(f"    Actual achievable:    {s['mission_years_actual']:.1f} years")
    print(f"    Limiting factor:      {s['limiting_factor']}")
    print(f"    Max erosion:          {s['max_erosion_um']:.2f} µm "
          f"({s['erosion_fraction']*100:.1f}% of thickness)")
    print(f"    Propellant used:      {s['propellant_used_kg']:.1f} kg")
    print(f"    Propellant remaining: {s['propellant_remaining_kg']:.1f} kg")
    print(f"    COG Z-shift:          {s['cog_shift_z_mm']:.1f} mm")

    print("\n  Generating time-resolved plots...")
    time_files = plot_time_resolved_erosion(result, output_dir=output_dir)

    # ============================================================
    # PART 2: Propellant vs Mission Duration sweep
    # ============================================================
    print("\n\n[2] SWEEP: PROPELLANT LOAD vs MISSION DURATION")
    print("-" * 50)

    sweep = PropellantErosionSweep(thruster, material)
    results_pm = sweep.sweep_propellant_vs_mission(
        arm=arm,
        stack=stack,
        sk_budget=sk_budget,
        propellant_range_kg=np.arange(40, 181, 20),
        mission_range_yr=np.arange(2, 13, 1),
        verbose=True,
    )

    print(f"  Total cases: {len(results_pm)}")
    prop_limited = sum(1 for r in results_pm if r["limiting_factor"] == "PROPELLANT")
    ero_limited = sum(1 for r in results_pm if r["limiting_factor"] == "EROSION")
    print(f"  Propellant-limited: {prop_limited}")
    print(f"  Erosion-limited:    {ero_limited}")

    print("\n  Generating correlation heatmaps...")
    pm_files = plot_propellant_erosion_correlation(
        results_pm, "propellant_loaded_kg", "mission_requested_yr",
        output_dir=output_dir
    )

    # ============================================================
    # PART 3: Propellant vs Arm Length sweep
    # ============================================================
    print("\n\n[3] SWEEP: PROPELLANT LOAD vs ARM LENGTH")
    print("-" * 50)

    results_pa = sweep.sweep_propellant_vs_arm_length(
        stack=stack,
        sk_budget=sk_budget,
        propellant_range_kg=np.arange(40, 181, 20),
        arm_length_range_m=np.arange(1.0, 4.5, 0.5),
        mission_years=5.0,
        verbose=True,
    )

    print(f"  Total cases: {len(results_pa)}")

    print("\n  Generating correlation heatmaps...")
    pa_files = plot_propellant_erosion_correlation(
        results_pa, "propellant_loaded_kg", "arm_length_m",
        output_dir=output_dir
    )

    # ============================================================
    # Export
    # ============================================================
    # CSV exports
    for name, data in [("sweep_prop_vs_mission.csv", results_pm),
                       ("sweep_prop_vs_arm.csv", results_pa)]:
        fpath = os.path.join(output_dir, name)
        if data:
            keys = data[0].keys()
            with open(fpath, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(data)
            print(f"\n  Exported: {fpath}")

    # Time history export
    th_path = os.path.join(output_dir, "time_history.csv")
    if result["time_history"]:
        keys = result["time_history"][0].keys()
        with open(th_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(result["time_history"])
        print(f"  Exported: {th_path}")

    print("\n\n" + "=" * 70)
    print("  ALL GENERATED FILES")
    print("=" * 70)
    all_files = time_files + pm_files + pa_files
    for fp in all_files:
        print(f"    → {fp}")

    return output_dir, all_files


if __name__ == "__main__":
    run_demo()
