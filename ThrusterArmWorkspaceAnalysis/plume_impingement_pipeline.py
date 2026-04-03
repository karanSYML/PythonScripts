#!/usr/bin/env python3
"""
Plasma Plume Impingement Parametric Study Pipeline
===================================================
Integrated framework for life-extension servicer missions.

Modules:
  1. Case Matrix Generator   : builds all parameter combinations + geometry
  2. Analytical Pre-Screener : fast erosion estimates (inverse-square + sputter model)
  3. Heatmap Visualizer      : multi-dimensional interactive dashboard

Author: Karan Anand (karan.anand@infiniteorbits.io)
"""

import numpy as np
import itertools
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
import warnings

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
    thrust_N: float = 0.053            # N  (nominal) # 59-53 mN 

# TBC with Sushir
@dataclass
class ArmGeometry:
    """Thruster arm configuration."""
    arm_length: float = 2.0           # m  (pivot to thruster exit) TBC
    pivot_offset_x: float = 0.167      # m  from servicer geometric centre
    pivot_offset_y: float = 0.32       # m
    pivot_offset_z: float = 1.06       # m  (typically on anti-earth face)
    azimuth_deg: float = 0.0          # deg  arm azimuth in body frame
    elevation_deg: float = 0.0        # deg  arm elevation in body frame
    cant_angle_deg: float = 0.0       # deg  thrust vector cant at tip

# TBC with Sushir
@dataclass
class StackConfig:
    """Combined servicer + client satellite stack."""
    # Servicer
    servicer_mass: float = 523.0        # kg
    servicer_bus_x: float = 0.826         # m  (along velocity)
    servicer_bus_y: float = 1.5         # m  (along orbit-normal / N-S)
    servicer_bus_z: float = 0.9         # m  (along nadir)

    # Client (telecom satellite)
    client_mass: float = 2804.0         # kg
    client_bus_x: float = 2.3           # m
    client_bus_y: float = 3.0           # m
    client_bus_z: float = 5.0           # m

    # Solar panels (client) – modelled as flat rectangles
    panel_span_one_side: float = 16.60   # m  from bus edge to panel tip
    panel_width: float = 2.3            # m  (along orbit-normal)
    panel_hinge_offset_y: float = 0.0   # m  offset of hinge line from bus centre-y
    panel_cant_angle_deg: float = 0.0   # deg  cant about hinge axis

    # Docking interface
    dock_offset_z: float = 1.0          # m  servicer offset along Z from client centre
    dock_offset_x: float = 0.0          # m

    # Antenna reflectors (simplified as discs)
    antenna_diameter: float = 2.25       # m
    antenna_offset_x: float = 0.0       # m  from client centre
    antenna_offset_z: float = 1.8       # m  (earth-facing typically)

    def stack_cog(self) -> np.ndarray:
        """Combined centre of gravity in client body frame."""
        # Servicer CG assumed at its geometric centre, offset by docking
        servicer_cg = np.array([
            self.dock_offset_x,
            0.0,
            self.client_bus_z / 2.0 + self.servicer_bus_z / 2.0 + self.dock_offset_z
        ])
        client_cg = np.array([0.0, 0.0, 0.0])
        total_mass = self.servicer_mass + self.client_mass
        cog = (self.client_mass * client_cg + self.servicer_mass * servicer_cg) / total_mass
        return cog

 
@dataclass
class OperationalParams:
    """Mission operational parameters."""
    firing_duration_s: float = 9468     # s per manoeuvre // 306 minutes
    firings_per_day: float = 2.0
    mission_duration_years: float = 5.0
    manoeuvre_type: str = "NSSK"          # NSSK or EWSK
    panel_sun_tracking_angle_deg: float = 0.0  # panel rotation about hinge during firing


@dataclass
class MaterialParams:
    """Target surface material properties."""
    name: str = "Silver_interconnect"
    thickness_um: float = 25.0            # µm
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

class GeometryEngine:
    """Computes plume origin, direction, distances and angles to surfaces."""

    def __init__(self, arm: ArmGeometry, stack: StackConfig):
        self.arm = arm
        self.stack = stack

    def thruster_position(self) -> np.ndarray:
        """Thruster exit plane centre in client body frame."""
        # Pivot point in client frame
        # Servicer is docked on top (+Z), pivot is on servicer
        servicer_base_z = self.stack.client_bus_z / 2.0 + self.stack.dock_offset_z
        pivot = np.array([
            self.stack.dock_offset_x + self.arm.pivot_offset_x,
            self.arm.pivot_offset_y,
            servicer_base_z + self.arm.pivot_offset_z
        ])

        # Arm direction (azimuth from +X towards +Y, elevation from XY plane)
        az = np.radians(self.arm.azimuth_deg)
        el = np.radians(self.arm.elevation_deg)
        arm_dir = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ])

        thruster_pos = pivot + self.arm.arm_length * arm_dir
        return thruster_pos

    def thrust_direction(self) -> np.ndarray:
        """Unit thrust vector in client body frame.

        The thrust direction is constrained to point *through* the stack COG
        to minimise disturbance torques (ideal case).
        If cant_angle != 0, a small offset from ideal is applied.
        """
        t_pos = self.thruster_position()
        cog = self.stack.stack_cog()
        ideal_dir = cog - t_pos
        ideal_dir = ideal_dir / np.linalg.norm(ideal_dir)

        # Apply cant angle as a rotation about an axis perpendicular to
        # both the ideal direction and the arm axis
        if abs(self.arm.cant_angle_deg) > 0.01:
            cant = np.radians(self.arm.cant_angle_deg)
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

                    # Panel Z position: at top of client bus
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
        """Approximate panel surface normal (sun-facing side)."""
        track = np.radians(sun_tracking_angle_deg)
        cant = np.radians(self.stack.panel_cant_angle_deg)
        total = track + cant
        normal = np.array([0.0, -np.sin(total), np.cos(total)])
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


# ---------------------------------------------------------------------------
# 3.  ANALYTICAL PRE-SCREENING (Sputter Erosion Estimator)
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
            # Arm geometry
            "arm_length":          np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
            "arm_azimuth_deg":     np.array([-30, -15, 0, 15, 30, 45, 60, 90]),
            "arm_elevation_deg":   np.array([-20, -10, 0, 10, 20, 30]),

            # Stack
            "client_mass":         np.array([2000, 3000, 4000, 5000, 6000]),
            "servicer_mass":       np.array([300, 400, 500]),
            "panel_span_one_side": np.array([8, 10, 12, 15, 18]),

            # Operations
            "firing_duration_s":   np.array([300, 600, 900, 1200, 1800]),
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
    def case_to_objects(case: Dict) -> Tuple[ArmGeometry, StackConfig, OperationalParams]:
        """Convert a flat case dict to the structured dataclass objects."""
        arm = ArmGeometry(
            arm_length=case.get("arm_length", 2.0),
            azimuth_deg=case.get("arm_azimuth_deg", 0.0),
            elevation_deg=case.get("arm_elevation_deg", 0.0),
            cant_angle_deg=case.get("cant_angle_deg", 0.0),
            pivot_offset_x=case.get("pivot_offset_x", 0.0),
            pivot_offset_y=case.get("pivot_offset_y", 0.0),
            pivot_offset_z=case.get("pivot_offset_z", 0.5),
        )
        stack = StackConfig(
            servicer_mass=case.get("servicer_mass", 400.0),
            client_mass=case.get("client_mass", 3000.0),
            panel_span_one_side=case.get("panel_span_one_side", 12.0),
            panel_width=case.get("panel_width", 2.5),
            client_bus_x=case.get("client_bus_x", 2.5),
            client_bus_y=case.get("client_bus_y", 2.2),
            client_bus_z=case.get("client_bus_z", 3.0),
            panel_cant_angle_deg=case.get("panel_cant_deg", 0.0),
        )
        ops = OperationalParams(
            firing_duration_s=case.get("firing_duration_s", 600.0),
            mission_duration_years=case.get("mission_duration_yr", 5.0),
            firings_per_day=case.get("firings_per_day", 2.0),
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

            # Classify
            thickness = self.material.thickness_um
            if max_erosion >= thickness:
                status = "FAIL"
            elif max_erosion >= 0.5 * thickness:
                status = "MARGINAL"
            elif max_erosion >= 0.1 * thickness:
                status = "CAUTION"
            else:
                status = "SAFE"

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
                "cog_x": float(stack.stack_cog()[0]),
                "cog_y": float(stack.stack_cog()[1]),
                "cog_z": float(stack.stack_cog()[2]),
                "status": status,
                "erosion_fraction": float(max_erosion / thickness),
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
            ax.clabel(cs, fmt=f"{{:.0f}} µm (FAIL)", fontsize=9, colors="white")
        except Exception:
            pass  # contour may fail if all values on one side

        # Also mark 50% threshold
        try:
            cs2 = ax.contour(X_grid, Y_grid, Z, levels=[0.5 * thickness_um],
                             colors="white", linewidths=1.5, linestyles=":")
            ax.clabel(cs2, fmt=f"{{:.0f}} µm (MARGINAL)", fontsize=8, colors="white")
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
        "arm_length": "Arm Length [m]",
        "arm_azimuth_deg": "Arm Azimuth [°]",
        "arm_elevation_deg": "Arm Elevation [°]",
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
        "arm_length": "Arm Length [m]",
        "arm_azimuth_deg": "Arm Azimuth [°]",
        "arm_elevation_deg": "Arm Elevation [°]",
        "client_mass": "Client Mass [kg]",
        "servicer_mass": "Servicer Mass [kg]",
        "panel_span_one_side": "Panel Span (one side) [m]",
        "firing_duration_s": "Firing Duration [s]",
        "mission_duration_yr": "Mission Duration [yr]",
        "panel_tracking_deg": "Panel Sun-Tracking Angle [°]",
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
    gen.set_param_range("arm_length", np.arange(2.0, 3.5, 0.5))
    gen.set_param_range("arm_azimuth_deg", np.arange(-30, 95, 15))
    gen.set_param_range("arm_elevation_deg", np.array([-10, 0, 10, 20]))
    gen.set_param_range("panel_span_one_side", np.array([10, 12, 15, 18, 20]))
    gen.set_param_range("firing_duration_s", np.array([3000, 6000, 9000, 12000, 18000]))
    gen.set_param_range("mission_duration_yr", np.array([3, 5, 7, 10]))
    gen.set_param_range("client_mass", np.array([2000, 3500, 5000]))
    gen.set_param_range("servicer_mass", np.array([530]))
    gen.set_param_range("panel_tracking_deg", np.array([-15, 0, 15]))

    # --- Sweep 1: Arm length vs Arm azimuth ---
    print("\n[SWEEP 1] Arm Length vs Arm Azimuth")
    fixed1 = {
        "arm_elevation_deg": 0.0,
        "client_mass": 3000.0,
        "servicer_mass": 530.0,
        "panel_span_one_side": 16.0,
        "firing_duration_s": 9000.0,
        "mission_duration_yr": 5.0,
        "panel_tracking_deg": 0.0,
    }
    cases1 = gen.generate_reduced_matrix(fixed1, ["arm_length", "arm_azimuth_deg"])
    print(f"  Cases to evaluate: {len(cases1)}")
    results1 = pipeline.run_sweep(cases1, verbose=True)

    # --- Sweep 2: Panel span vs Mission duration ---
    print("\n[SWEEP 2] Panel Span vs Mission Duration")
    fixed2 = {
        "arm_length": 2.0,
        "arm_azimuth_deg": 0.0,
        "arm_elevation_deg": 0.0,
        "client_mass": 3000.0,
        "servicer_mass": 530.0,
        "firing_duration_s": 9000.0,
        "panel_tracking_deg": 0.0,
    }
    cases2 = gen.generate_reduced_matrix(fixed2, ["panel_span_one_side", "mission_duration_yr"])
    print(f"  Cases to evaluate: {len(cases2)}")
    results2 = pipeline.run_sweep(cases2, verbose=True)

    # --- Sweep 3: Firing duration vs Arm elevation ---
    print("\n[SWEEP 3] Firing Duration vs Arm Elevation")
    fixed3 = {
        "arm_length": 2.0,
        "arm_azimuth_deg": 0.0,
        "client_mass": 3000.0,
        "servicer_mass": 530.0,
        "panel_span_one_side": 16.0,
        "mission_duration_yr": 5.0,
        "panel_tracking_deg": 0.0,
    }
    cases3 = gen.generate_reduced_matrix(fixed3, ["firing_duration_s", "arm_elevation_deg"])
    print(f"  Cases to evaluate: {len(cases3)}")
    results3 = pipeline.run_sweep(cases3, verbose=True)

    # --- Generate heatmaps ---
    output_dir = "/home/karan.anand/Documents/PythonScripts/ThrusterArmWorkspaceAnalysis/pipeline_output/"
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
