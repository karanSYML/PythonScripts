"""
mission.py
==========

Mission-level wrappers: duty cycle, accumulated firing time over the mission,
combined erosion + thermal-fatigue life estimate, and sensitivity helpers.

Sits on top of `erosion.ErosionIntegrator` and reuses any of the underlying
geometry, plume, and yield models. Designed to slot into the existing
`geo_rpo` mission-analysis workflow.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from .geometry import SatelliteGeometry
from .erosion import ErosionIntegrator
from .environment import ThermalCycling


@dataclass
class FiringPhase:
    """
    A single firing phase: which thrusters are on, for how long, with what
    duty cycle. Geometry can change between phases (e.g. different thruster
    selections) by passing a different SatelliteGeometry.
    """
    name: str
    geometry: SatelliteGeometry
    duration_s: float
    duty_cycle: float = 1.0   # fraction of duration_s actually firing


@dataclass
class MissionProfile:
    """
    A sequence of firing phases that together describe the mission's
    propulsive activity.
    """
    phases: List[FiringPhase]

    def total_firing_time(self) -> float:
        return sum(p.duration_s * p.duty_cycle for p in self.phases)


@dataclass
class LifetimeAnalysis:
    """
    Combine erosion across a multi-phase mission with thermal-cycling fatigue
    to produce an end-of-life thinning estimate and a coupled life prediction.
    """
    integrator: ErosionIntegrator
    thermal: ThermalCycling = field(default_factory=ThermalCycling)

    def cumulative_thinning(
        self, profile: MissionProfile,
    ) -> Dict[tuple, float]:
        """
        Sum thinning [m] per (array_idx, ic_idx) across all firing phases.
        """
        total: Dict[tuple, float] = {}
        for ph in profile.phases:
            t_eff = ph.duration_s * ph.duty_cycle
            ph_results = self.integrator.total_thinning_per_interconnect(
                ph.geometry, t_eff
            )
            for k, v in ph_results.items():
                total[k] = total.get(k, 0.0) + v
        return total

    def life_prediction(
        self,
        profile: MissionProfile,
        initial_thickness: float = 25e-6,
        thermal_n_cycles_to_failure: float = 5e4,
    ) -> Dict:
        """
        Returns a dict with end-of-life thinning, fraction remaining, and a
        coupled life factor that combines erosion-thinning with the
        Coffin-Manson scaling of thermal fatigue cycles to failure.
        """
        thinning = self.cumulative_thinning(profile)
        out = {}
        for k, v in thinning.items():
            frac_thinned = v / initial_thickness
            life_thermal = self.thermal.coupled_failure_factor(frac_thinned)
            out[k] = {
                "thinning_m": v,
                "fraction_remaining": max(0.0, 1.0 - frac_thinned),
                "thermal_life_factor": life_thermal,
                "coupled_life_factor": (
                    max(0.0, 1.0 - frac_thinned) * life_thermal
                ),
            }
        return out

    # -- sensitivity helpers ----------------------------------------------

    def sensitivity_to_cant_angle(
        self,
        base_profile: MissionProfile,
        thruster_index: int,
        angles_deg: np.ndarray,
        rotation_axis: str = "y",
    ) -> Dict[float, Dict[tuple, float]]:
        """
        Re-run the mission for several thruster cant angles, returning
        thinning per interconnect for each angle. Geometry must be modifiable;
        this method does NOT mutate the input profile.
        """
        from copy import deepcopy
        out: Dict[float, Dict[tuple, float]] = {}
        for ang in angles_deg:
            new_profile = deepcopy(base_profile)
            for ph in new_profile.phases:
                thr = ph.geometry.thrusters[thruster_index]
                # rotate fire_direction_body about the chosen axis
                fd = thr.fire_direction_body.to_array()
                a = np.deg2rad(ang)
                if rotation_axis == "y":
                    R = np.array([
                        [ np.cos(a), 0, np.sin(a)],
                        [ 0,         1, 0       ],
                        [-np.sin(a), 0, np.cos(a)],
                    ])
                elif rotation_axis == "x":
                    R = np.array([
                        [1, 0, 0],
                        [0,  np.cos(a), -np.sin(a)],
                        [0,  np.sin(a),  np.cos(a)],
                    ])
                else:  # z
                    R = np.array([
                        [np.cos(a), -np.sin(a), 0],
                        [np.sin(a),  np.cos(a), 0],
                        [0,          0,         1],
                    ])
                from .geometry import Vector3
                fd2 = Vector3.from_array(R @ fd)
                thr.fire_direction_body = fd2
            out[float(ang)] = self.cumulative_thinning(new_profile)
        return out
