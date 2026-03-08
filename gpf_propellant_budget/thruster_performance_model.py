"""
================================================================================
THRUSTER PERFORMANCE MODEL — Tool 3
Propulsion Engineering | GNC/AOCS Interface Tool
================================================================================
Platforms : Platform A | Platform B
EP        : Xenon Hall thruster / ion thruster
            Throttle table supplied manually in JSON config
RCS       : C3H6 / N2O bipropellant — spark/torch ignition (REACH compliant)
            Feed system: blowdown (same-tank He) OR regulated (separate He bottle)
            Both modes supported — selected per platform in config

Computes:
  EP  — Isp and thrust vs. input power from throttle table
        Duty-cycle-weighted average Isp for a given solar array power profile
        Lifetime check: fired hours and start cycles vs. qualification limits
  RCS — Tank pressure evolution during blowdown or regulated operation
        Thruster inlet pressure accounting for line pressure drop
        Isp and thrust vs. inlet pressure from supplier curve
        Mixture ratio sensitivity: Isp vs. O/F ratio
        BOL / EOL performance bounds

Outputs:
  - EP throttle table with performance at each point
  - EP power-to-Isp and power-to-thrust interpolated curves
  - RCS tank pressure vs. propellant remaining
  - RCS Isp vs. inlet pressure curve
  - RCS line pressure drop estimate
  - Recommended BOL / EOL Isp values for Tools 1 and 2
  - Auto-patched Tool 1 / Tool 2 config JSON files with updated Isp values

Usage:
    python thruster_performance_model.py --config thruster_config.json
    python thruster_performance_model.py --example
    python thruster_performance_model.py --example --patch-tool1 platform_a_config.json
    python thruster_performance_model.py --example --patch-tool2 sk_config.json

================================================================================
"""

import json
import csv
import math
import copy
import argparse
import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
G0          = 9.80665    # m/s²
R_GAS_HE    = 2077.0     # J/(kg·K) — specific gas constant for helium
T_REF_K     = 293.15     # Reference temperature 20°C in Kelvin


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES — EP
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EPThrottlePoint:
    """Single throttle point from supplier qualification table."""
    power_w: float        # Input power (W)
    thrust_mn: float      # Thrust (mN)
    isp_s: float          # Specific impulse (s)
    flow_mg_s: float      # Xenon mass flow rate (mg/s)
    notes: str = ""

    @property
    def efficiency(self) -> float:
        """Thruster efficiency = thrust power / input power."""
        thrust_w = (self.thrust_mn * 1e-3) * (self.isp_s * G0 / 2.0)
        return thrust_w / self.power_w if self.power_w > 0 else 0.0


@dataclass
class EPLifetimeLimits:
    """Qualification lifetime limits for the EP thruster."""
    max_fired_hours: float       # Total fired hours at qualification
    max_start_cycles: int        # Maximum number of start/stop cycles
    max_total_impulse_kns: float # Maximum total impulse (kN·s)


@dataclass
class EPConfig:
    """Full EP system configuration."""
    thruster_name: str
    throttle_table: list             # List of EPThrottlePoint
    lifetime: EPLifetimeLimits
    # Solar array power available at different mission points
    sa_power_bol_w: float            # Solar array power at BOL (W)
    sa_power_eol_w: float            # Solar array power at EOL (W)
    sa_power_eclipse_w: float        # Power available during eclipse (W, 0 if batteries only)
    duty_cycle_fraction: float       # Fraction of time EP can fire
    warmup_time_min: float           # Warm-up / conditioning time per session (minutes)
    warmup_flow_mg_s: float          # Xe flow during warm-up (mg/s, sub-nominal)
    # Mission usage
    mission_fired_hours: float       # Expected total fired hours this mission
    mission_start_cycles: int        # Expected total start cycles this mission


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES — RCS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RCSPerformancePoint:
    """Single performance point from supplier Isp vs. pressure curve."""
    inlet_pressure_bar: float   # Thruster inlet pressure (bar)
    isp_s: float                # Specific impulse (s)
    thrust_n: float             # Thrust (N)
    chamber_pressure_bar: float = 0.0  # Optional chamber pressure

@dataclass
class RCSBlowdownConfig:
    """
    Blowdown feed system — helium and propellant share the same tank.
    Pressure drops continuously as propellant depletes.
    """
    tank_volume_l: float          # Total tank volume (litres)
    p_initial_bar: float          # Initial tank pressure = MOP (bar)
    p_min_bar: float              # Minimum expected pressure = MEP (bar)
    he_mass_kg: float             # Initial helium mass loaded (kg)
    temperature_k: float          # Tank operating temperature (K)
    # Line pressure drop model (linear approximation)
    line_dp_nominal_bar: float    # Pressure drop at nominal flow rate (bar)


@dataclass
class RCSRegulatedConfig:
    """
    Regulated feed system — separate high-pressure He bottle + regulator.
    Propellant tank pressure held approximately constant.
    """
    p_setpoint_bar: float         # Regulator output set-point (bar)
    p_setpoint_tolerance_bar: float  # ±tolerance on set-point (bar)
    p_lockup_bar: float           # Lock-up pressure (no-flow condition) (bar)
    p_relief_bar: float           # Relief valve cracking pressure (bar)
    he_bottle_volume_l: float     # High-pressure He bottle volume (litres)
    he_bottle_p_initial_bar: float  # Initial He bottle pressure (bar)
    he_bottle_p_min_bar: float    # Minimum He bottle pressure (regulated mode ends)
    temperature_k: float          # System operating temperature (K)
    line_dp_nominal_bar: float    # Line pressure drop at nominal flow (bar)


@dataclass
class RCSTankConfig:
    """Propellant tank loading."""
    oxidiser_loaded_kg: float     # N2O loaded mass (kg)
    fuel_loaded_kg: float         # C3H6 loaded mass (kg)
    mixture_ratio_nominal: float  # N2O/C3H6 nominal mixture ratio
    oxidiser_density_kg_l: float  # N2O liquid density (kg/L) at operating temp
    fuel_density_kg_l: float      # C3H6 liquid density (kg/L) at operating temp


@dataclass
class RCSConfig:
    """Full RCS system configuration."""
    system_name: str
    feed_system_type: str          # "blowdown" or "regulated"
    tank: RCSTankConfig
    performance_curve: list        # List of RCSPerformancePoint (from supplier data)
    blowdown: Optional[RCSBlowdownConfig] = None
    regulated: Optional[RCSRegulatedConfig] = None
    # Mixture ratio sensitivity: list of {mr: float, isp_factor: float}
    # isp_factor is multiplier on nominal Isp at that MR (1.0 = nominal)
    mr_sensitivity: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES — TOP LEVEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThrusterConfig:
    platform_name: str
    ep: EPConfig
    rcs: RCSConfig


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def interp1d(x_vals: list, y_vals: list, x_query: float,
             extrapolate: bool = False) -> float:
    """
    Linear interpolation. x_vals must be monotonically increasing.
    Clamps to boundary values unless extrapolate=True.
    """
    if len(x_vals) != len(y_vals) or len(x_vals) < 2:
        raise ValueError("interp1d requires at least 2 matched data points.")

    if x_query <= x_vals[0]:
        if extrapolate:
            slope = (y_vals[1] - y_vals[0]) / (x_vals[1] - x_vals[0])
            return y_vals[0] + slope * (x_query - x_vals[0])
        return y_vals[0]

    if x_query >= x_vals[-1]:
        if extrapolate:
            slope = (y_vals[-1] - y_vals[-2]) / (x_vals[-1] - x_vals[-2])
            return y_vals[-1] + slope * (x_query - x_vals[-1])
        return y_vals[-1]

    for i in range(len(x_vals) - 1):
        if x_vals[i] <= x_query <= x_vals[i + 1]:
            frac = (x_query - x_vals[i]) / (x_vals[i + 1] - x_vals[i])
            return y_vals[i] + frac * (y_vals[i + 1] - y_vals[i])

    return y_vals[-1]


# ─────────────────────────────────────────────────────────────────────────────
# EP PERFORMANCE CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def ep_at_power(ep: EPConfig, power_w: float) -> dict:
    """
    Interpolate EP performance at a given input power from the throttle table.
    Returns thrust, Isp, flow rate, and efficiency.
    """
    pts = sorted(ep.throttle_table, key=lambda p: p.power_w)
    powers  = [p.power_w    for p in pts]
    thrusts = [p.thrust_mn  for p in pts]
    isps    = [p.isp_s      for p in pts]
    flows   = [p.flow_mg_s  for p in pts]

    thrust = interp1d(powers, thrusts, power_w)
    isp    = interp1d(powers, isps,    power_w)
    flow   = interp1d(powers, flows,   power_w)
    eff    = (thrust * 1e-3 * isp * G0 / 2.0) / power_w if power_w > 0 else 0.0

    return {
        "power_w": power_w,
        "thrust_mn": thrust,
        "isp_s": isp,
        "flow_mg_s": flow,
        "efficiency": eff
    }


def ep_weighted_isp(ep: EPConfig, power_profile: list) -> dict:
    """
    Compute duty-cycle-weighted average Isp over a power profile.
    power_profile: list of (power_w, fraction_of_time) tuples, fractions sum to 1.0.
    Returns weighted average Isp, thrust, and total impulse per second.
    """
    total_weight  = sum(f for _, f in power_profile)
    weighted_isp  = 0.0
    weighted_thr  = 0.0
    weighted_flow = 0.0

    for power_w, fraction in power_profile:
        perf = ep_at_power(ep, power_w)
        w = fraction / total_weight
        weighted_isp  += perf["isp_s"]    * w
        weighted_thr  += perf["thrust_mn"] * w
        weighted_flow += perf["flow_mg_s"] * w

    return {
        "weighted_isp_s": weighted_isp,
        "weighted_thrust_mn": weighted_thr,
        "weighted_flow_mg_s": weighted_flow
    }


def ep_lifetime_check(ep: EPConfig) -> dict:
    """
    Check mission usage against qualification lifetime limits.
    Returns fractions consumed and pass/fail per parameter.
    """
    lim = ep.lifetime
    # Warm-up hours
    warmup_hours = (ep.warmup_time_min / 60.0) * ep.mission_start_cycles
    total_hours  = ep.mission_fired_hours + warmup_hours

    # Total impulse estimate from throttle table midpoint
    pts = ep.throttle_table
    avg_thrust_mn = sum(p.thrust_mn for p in pts) / len(pts)
    total_impulse_kns = (avg_thrust_mn * 1e-3) * (total_hours * 3600.0) / 1000.0

    hours_frac   = total_hours        / lim.max_fired_hours
    cycles_frac  = ep.mission_start_cycles / lim.max_start_cycles
    impulse_frac = total_impulse_kns  / lim.max_total_impulse_kns

    return {
        "mission_fired_hours": ep.mission_fired_hours,
        "warmup_hours": warmup_hours,
        "total_hours": total_hours,
        "qualification_hours": lim.max_fired_hours,
        "hours_fraction": hours_frac,
        "hours_ok": hours_frac <= 1.0,
        "mission_start_cycles": ep.mission_start_cycles,
        "qualification_cycles": lim.max_start_cycles,
        "cycles_fraction": cycles_frac,
        "cycles_ok": cycles_frac <= 1.0,
        "total_impulse_kns": total_impulse_kns,
        "qualification_impulse_kns": lim.max_total_impulse_kns,
        "impulse_fraction": impulse_frac,
        "impulse_ok": impulse_frac <= 1.0,
        "overall_ok": hours_frac <= 1.0 and cycles_frac <= 1.0 and impulse_frac <= 1.0
    }


def ep_warmup_xenon(ep: EPConfig) -> dict:
    """Compute total Xenon consumed during non-propulsive warm-up sessions."""
    warmup_s = ep.warmup_time_min * 60.0
    xe_per_session_kg = ep.warmup_flow_mg_s * 1e-6 * warmup_s
    xe_total_kg = xe_per_session_kg * ep.mission_start_cycles
    return {
        "warmup_time_s": warmup_s,
        "xe_per_session_kg": xe_per_session_kg,
        "total_sessions": ep.mission_start_cycles,
        "xe_warmup_total_kg": xe_total_kg
    }


def ep_recommended_isp(ep: EPConfig) -> dict:
    """
    Determine recommended BOL and EOL Isp values for Tools 1 and 2.
    BOL: Isp at full solar array power (sa_power_bol_w)
    EOL: Isp at degraded solar array power (sa_power_eol_w)
    """
    bol = ep_at_power(ep, ep.sa_power_bol_w)
    eol = ep_at_power(ep, ep.sa_power_eol_w)
    return {
        "isp_bol": bol["isp_s"],
        "thrust_bol_mn": bol["thrust_mn"],
        "power_bol_w": ep.sa_power_bol_w,
        "isp_eol": eol["isp_s"],
        "thrust_eol_mn": eol["thrust_mn"],
        "power_eol_w": ep.sa_power_eol_w,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RCS BLOWDOWN MODEL
# ─────────────────────────────────────────────────────────────────────────────

def blowdown_pressure_vs_remaining(rcs: RCSConfig, n_points: int = 50) -> list:
    """
    Model tank pressure as a function of remaining propellant mass
    for a blowdown system.

    Physical model:
      The helium pressurant occupies the ullage volume above the liquid
      propellant. As propellant is expelled, ullage volume increases and
      helium pressure drops according to the ideal gas law (isothermal):
        P * V_ullage = const = P_initial * V_ullage_initial

      V_ullage_initial = V_tank - V_propellant_initial
      V_propellant(m)  = m_ox / rho_ox + m_fu / rho_fu
                       = (m_total * MR/(1+MR)) / rho_ox
                         + (m_total / (1+MR)) / rho_fu
    """
    bd  = rcs.blowdown
    tk  = rcs.tank
    mr  = tk.mixture_ratio_nominal

    # Initial propellant volume
    m_prop_initial = tk.oxidiser_loaded_kg + tk.fuel_loaded_kg
    v_prop_initial = (tk.oxidiser_loaded_kg / tk.oxidiser_density_kg_l +
                      tk.fuel_loaded_kg     / tk.fuel_density_kg_l)   # litres

    # Initial ullage = tank volume - initial propellant volume
    v_ullage_initial = bd.tank_volume_l - v_prop_initial

    if v_ullage_initial <= 0:
        raise ValueError(
            f"Initial propellant volume ({v_prop_initial:.2f} L) exceeds "
            f"tank volume ({bd.tank_volume_l:.2f} L). Check density and loading."
        )

    results = []
    step = m_prop_initial / (n_points - 1)

    for i in range(n_points):
        m_remaining = m_prop_initial - i * step
        m_remaining = max(m_remaining, 0.0)

        # Propellant volume remaining (assuming MR maintained throughout)
        m_ox = m_remaining * mr / (1.0 + mr)
        m_fu = m_remaining / (1.0 + mr)
        v_prop = m_ox / tk.oxidiser_density_kg_l + m_fu / tk.fuel_density_kg_l

        # Ullage volume now
        v_ullage = bd.tank_volume_l - v_prop

        # Ideal gas law: P_tank = P_initial * V_ullage_initial / V_ullage
        p_tank = bd.p_initial_bar * v_ullage_initial / v_ullage

        # Inlet pressure = tank pressure - line drop
        p_inlet = max(p_tank - bd.line_dp_nominal_bar, 0.0)

        results.append({
            "m_remaining_kg": m_remaining,
            "fraction_remaining": m_remaining / m_prop_initial,
            "v_ullage_l": v_ullage,
            "p_tank_bar": p_tank,
            "p_inlet_bar": p_inlet,
            "below_mep": p_tank < bd.p_min_bar
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RCS REGULATED MODEL
# ─────────────────────────────────────────────────────────────────────────────

def regulated_he_bottle_depletion(rcs: RCSConfig, n_points: int = 50) -> list:
    """
    Model He bottle pressure depletion as propellant is consumed
    for a regulated system.

    As propellant is expelled, the He bottle must supply gas to maintain
    the propellant tank at the regulator set-point.
    He consumed per unit propellant volume = set-point pressure / (R_HE * T)

    The He bottle pressure drops as its mass depletes:
      P_bottle = (m_he_remaining * R_HE * T) / V_bottle

    Regulated mode ends when P_bottle drops to p_min — after that,
    the system transitions to blowdown.
    """
    reg = rcs.regulated
    tk  = rcs.tank
    mr  = tk.mixture_ratio_nominal

    # Initial He mass in bottle (from ideal gas)
    he_mass_initial = (reg.he_bottle_p_initial_bar * 1e5 *
                       reg.he_bottle_volume_l * 1e-3) / (R_GAS_HE * reg.temperature_k)

    m_prop_initial = tk.oxidiser_loaded_kg + tk.fuel_loaded_kg
    step = m_prop_initial / (n_points - 1)

    results = []

    for i in range(n_points):
        m_remaining = m_prop_initial - i * step
        m_remaining = max(m_remaining, 0.0)
        m_consumed  = m_prop_initial - m_remaining

        # Volume of propellant consumed (He must fill this at set-point pressure)
        m_ox_consumed = m_consumed * mr / (1.0 + mr)
        m_fu_consumed = m_consumed / (1.0 + mr)
        v_prop_consumed = (m_ox_consumed / tk.oxidiser_density_kg_l +
                           m_fu_consumed / tk.fuel_density_kg_l)   # litres

        # He mass consumed to pressurise that volume at set-point
        he_consumed = (reg.p_setpoint_bar * 1e5 *
                       v_prop_consumed * 1e-3) / (R_GAS_HE * reg.temperature_k)

        he_remaining = max(he_mass_initial - he_consumed, 0.0)

        # He bottle pressure
        p_bottle = (he_remaining * R_GAS_HE * reg.temperature_k) / \
                   (reg.he_bottle_volume_l * 1e-3) / 1e5  # bar

        # Regulated mode still active?
        regulated_active = p_bottle >= reg.he_bottle_p_min_bar

        # Effective inlet pressure
        if regulated_active:
            # Pessimistic: set-point minus tolerance minus line drop
            p_inlet_nom = reg.p_setpoint_bar - reg.line_dp_nominal_bar
            p_inlet_pess = (reg.p_setpoint_bar - reg.p_setpoint_tolerance_bar
                            - reg.line_dp_nominal_bar)
        else:
            # Blowdown regime after He bottle depletion
            p_inlet_nom  = max(p_bottle - reg.line_dp_nominal_bar, 0.0)
            p_inlet_pess = p_inlet_nom

        results.append({
            "m_remaining_kg": m_remaining,
            "fraction_remaining": m_remaining / m_prop_initial,
            "he_remaining_kg": he_remaining,
            "p_bottle_bar": p_bottle,
            "regulated_active": regulated_active,
            "p_inlet_nominal_bar": p_inlet_nom,
            "p_inlet_pessimistic_bar": p_inlet_pess,
            "lockup_pressure_bar": reg.p_lockup_bar
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RCS PERFORMANCE FROM PRESSURE
# ─────────────────────────────────────────────────────────────────────────────

def rcs_isp_at_pressure(rcs: RCSConfig, inlet_pressure_bar: float) -> dict:
    """
    Interpolate RCS Isp and thrust at a given inlet pressure
    from the supplier performance curve.
    """
    pts = sorted(rcs.performance_curve, key=lambda p: p.inlet_pressure_bar)
    pressures = [p.inlet_pressure_bar for p in pts]
    isps      = [p.isp_s              for p in pts]
    thrusts   = [p.thrust_n           for p in pts]

    isp    = interp1d(pressures, isps,    inlet_pressure_bar)
    thrust = interp1d(pressures, thrusts, inlet_pressure_bar)

    return {
        "inlet_pressure_bar": inlet_pressure_bar,
        "isp_s": isp,
        "thrust_n": thrust
    }


def rcs_bol_eol_isp(rcs: RCSConfig, pressure_curve: list) -> dict:
    """
    Determine BOL and EOL Isp from the pressure-vs-remaining curve.
    BOL: inlet pressure at full propellant loading (first point)
    EOL: inlet pressure at MEP (last usable point before below_mep or
         when regulated mode ends)
    """
    # BOL — first point (full tanks)
    p_bol = pressure_curve[0].get("p_inlet_bar") or \
            pressure_curve[0].get("p_inlet_nominal_bar")

    # EOL — last point where system is still usable
    if rcs.feed_system_type == "blowdown":
        usable = [pt for pt in pressure_curve if not pt["below_mep"]]
        eol_pt = usable[-1] if usable else pressure_curve[-1]
        p_eol  = eol_pt["p_inlet_bar"]
    else:
        usable = [pt for pt in pressure_curve]
        eol_pt = usable[-1]
        p_eol  = eol_pt["p_inlet_pessimistic_bar"]

    perf_bol = rcs_isp_at_pressure(rcs, p_bol)
    perf_eol = rcs_isp_at_pressure(rcs, p_eol)

    return {
        "p_inlet_bol_bar": p_bol,
        "isp_bol_s": perf_bol["isp_s"],
        "thrust_bol_n": perf_bol["thrust_n"],
        "p_inlet_eol_bar": p_eol,
        "isp_eol_s": perf_eol["isp_s"],
        "thrust_eol_n": perf_eol["thrust_n"],
        "isp_degradation_pct": (perf_bol["isp_s"] - perf_eol["isp_s"]) /
                                 perf_bol["isp_s"] * 100.0
    }


def rcs_mr_sensitivity(rcs: RCSConfig) -> list:
    """
    Compute Isp at each mixture ratio sensitivity point.
    Returns list of {mr, isp_factor, isp_at_bol_pressure}.
    """
    if not rcs.mr_sensitivity:
        return []

    # BOL inlet pressure
    if rcs.feed_system_type == "blowdown":
        bd = rcs.blowdown
        tk = rcs.tank
        mr = tk.mixture_ratio_nominal
        m_total = tk.oxidiser_loaded_kg + tk.fuel_loaded_kg
        m_ox = m_total * mr / (1 + mr)
        m_fu = m_total / (1 + mr)
        v_prop = m_ox / tk.oxidiser_density_kg_l + m_fu / tk.fuel_density_kg_l
        v_ullage = bd.tank_volume_l - v_prop
        p_tank = bd.p_initial_bar
        p_inlet_bol = p_tank - bd.line_dp_nominal_bar
    else:
        reg = rcs.regulated
        p_inlet_bol = reg.p_setpoint_bar - reg.line_dp_nominal_bar

    perf_nominal = rcs_isp_at_pressure(rcs, p_inlet_bol)

    results = []
    for pt in rcs.mr_sensitivity:
        isp_actual = perf_nominal["isp_s"] * pt["isp_factor"]
        results.append({
            "mixture_ratio": pt["mr"],
            "isp_factor": pt["isp_factor"],
            "isp_s": isp_actual,
            "delta_isp_s": isp_actual - perf_nominal["isp_s"]
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PATCH — Tools 1 and 2
# ─────────────────────────────────────────────────────────────────────────────

def patch_tool1_config(tool1_path: str, ep_isp: dict, rcs_isp: dict,
                       output_dir: str = ".") -> str:
    """
    Read a Tool 1 (propellant_budget_calculator) JSON config and update
    Isp values in each phase based on Tool 3 recommendations.

    EP phases: update ep.isp_bol and ep.isp_eol
    RCS phases: update rcs.isp_bol and rcs.isp_eol
    BOTH phases: update both

    Writes to a new file: <original_name>_t3patched.json
    Returns path to patched file.
    """
    with open(tool1_path) as f:
        cfg = json.load(f)

    patched = copy.deepcopy(cfg)
    patch_log = []

    for phase in patched.get("phases", []):
        ptype = phase.get("primary_propulsion", "")

        if "ep" in phase and ptype in ("EP", "BOTH"):
            old_bol = phase["ep"].get("isp_bol")
            old_eol = phase["ep"].get("isp_eol")
            phase["ep"]["isp_bol"] = round(ep_isp["isp_bol"], 1)
            phase["ep"]["isp_eol"] = round(ep_isp["isp_eol"], 1)
            patch_log.append(
                f"  Phase '{phase['name']}' EP: isp_bol {old_bol} → {ep_isp['isp_bol']:.1f}  "
                f"isp_eol {old_eol} → {ep_isp['isp_eol']:.1f}"
            )

        if "rcs" in phase and ptype in ("RCS", "BOTH"):
            old_bol = phase["rcs"].get("isp_bol")
            old_eol = phase["rcs"].get("isp_eol")
            phase["rcs"]["isp_bol"] = round(rcs_isp["isp_bol_s"], 1)
            phase["rcs"]["isp_eol"] = round(rcs_isp["isp_eol_s"], 1)
            patch_log.append(
                f"  Phase '{phase['name']}' RCS: isp_bol {old_bol} → {rcs_isp['isp_bol_s']:.1f}  "
                f"isp_eol {old_eol} → {rcs_isp['isp_eol_s']:.1f}"
            )

    # Also update ACS phases (RCS only for ACS, has rcs block)
    base = os.path.splitext(os.path.basename(tool1_path))[0]
    out_path = os.path.join(output_dir, f"{base}_t3patched.json")
    with open(out_path, "w") as f:
        json.dump(patched, f, indent=2)

    return out_path, patch_log


def patch_tool2_config(tool2_path: str, ep_isp: dict, rcs_isp: dict,
                       output_dir: str = ".") -> str:
    """
    Read a Tool 2 (station_keeping_budget) JSON config and update
    EP and RCS Isp values from Tool 3 recommendations.

    Writes to a new file: <original_name>_t3patched.json
    Returns path to patched file.
    """
    with open(tool2_path) as f:
        cfg = json.load(f)

    patched = copy.deepcopy(cfg)
    patch_log = []

    if "ep" in patched:
        old_bol = patched["ep"].get("isp_bol")
        old_eol = patched["ep"].get("isp_eol")
        patched["ep"]["isp_bol"] = round(ep_isp["isp_bol"], 1)
        patched["ep"]["isp_eol"] = round(ep_isp["isp_eol"], 1)
        patch_log.append(
            f"  EP: isp_bol {old_bol} → {ep_isp['isp_bol']:.1f}  "
            f"isp_eol {old_eol} → {ep_isp['isp_eol']:.1f}"
        )

    if "rcs_acs" in patched:
        old_bol = patched["rcs_acs"].get("isp_bol")
        old_eol = patched["rcs_acs"].get("isp_eol")
        patched["rcs_acs"]["isp_bol"] = round(rcs_isp["isp_bol_s"], 1)
        patched["rcs_acs"]["isp_eol"] = round(rcs_isp["isp_eol_s"], 1)
        patch_log.append(
            f"  RCS ACS: isp_bol {old_bol} → {rcs_isp['isp_bol_s']:.1f}  "
            f"isp_eol {old_eol} → {rcs_isp['isp_eol_s']:.1f}"
        )

    base = os.path.splitext(os.path.basename(tool2_path))[0]
    out_path = os.path.join(output_dir, f"{base}_t3patched.json")
    with open(out_path, "w") as f:
        json.dump(patched, f, indent=2)

    return out_path, patch_log


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_ep_throttle_csv(ep: EPConfig, platform_name: str,
                            output_dir: str = ".") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"ep_throttle_{platform_name}_{timestamp}.csv")
    pts = sorted(ep.throttle_table, key=lambda p: p.power_w)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["power_w","thrust_mn","isp_s",
                                           "flow_mg_s","efficiency","notes"])
        w.writeheader()
        for p in pts:
            w.writerow({"power_w": p.power_w, "thrust_mn": p.thrust_mn,
                        "isp_s": p.isp_s, "flow_mg_s": p.flow_mg_s,
                        "efficiency": round(p.efficiency, 4), "notes": p.notes})
    return path


def export_rcs_pressure_csv(pressure_curve: list, platform_name: str,
                             feed_type: str, output_dir: str = ".") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir,
                        f"rcs_pressure_{platform_name}_{feed_type}_{timestamp}.csv")
    if not pressure_curve:
        return ""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pressure_curve[0].keys()))
        w.writeheader()
        w.writerows(pressure_curve)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_header(title: str):
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


def print_section(title: str):
    print(f"\n{'─' * 90}")
    print(f"  {title}")
    print(f"{'─' * 90}")


def print_ep_throttle_table(ep: EPConfig):
    print_section("EP THROTTLE TABLE — from supplier qualification data")
    col = "{:>10} {:>12} {:>10} {:>14} {:>12}  {}"
    print("\n  " + col.format("Power(W)", "Thrust(mN)", "Isp(s)",
                               "Flow(mg/s)", "Efficiency", "Notes"))
    print(f"  {'─' * 80}")
    pts = sorted(ep.throttle_table, key=lambda p: p.power_w)
    for p in pts:
        print("  " + col.format(
            f"{p.power_w:.0f}", f"{p.thrust_mn:.1f}", f"{p.isp_s:.0f}",
            f"{p.flow_mg_s:.3f}", f"{p.efficiency:.4f}", p.notes
        ))

    print(f"\n  Solar array power:")
    print(f"    BOL : {ep.sa_power_bol_w:.0f} W")
    print(f"    EOL : {ep.sa_power_eol_w:.0f} W")
    bol = ep_at_power(ep, ep.sa_power_bol_w)
    eol = ep_at_power(ep, ep.sa_power_eol_w)
    print(f"\n  Performance at SA power:")
    print(f"    BOL → Isp {bol['isp_s']:.1f} s   Thrust {bol['thrust_mn']:.1f} mN   "
          f"Flow {bol['flow_mg_s']:.3f} mg/s")
    print(f"    EOL → Isp {eol['isp_s']:.1f} s   Thrust {eol['thrust_mn']:.1f} mN   "
          f"Flow {eol['flow_mg_s']:.3f} mg/s   "
          f"(Isp drop: {bol['isp_s']-eol['isp_s']:.1f} s)")


def print_ep_lifetime(lt: dict):
    print_section("EP LIFETIME CHECK")

    def bar(frac):
        filled = int(frac * 20)
        filled = min(filled, 20)
        return "[" + "█" * filled + "░" * (20 - filled) + "]"

    def status(ok): return "✓ OK" if ok else "✗ EXCEEDS QUALIFICATION"

    col = "{:<28} {:>10} {:>14} {:>8}  {}  {}"
    print("\n  " + col.format("Parameter", "Mission", "Qualification",
                               "Fraction", "Usage", "Status"))
    print(f"  {'─' * 88}")
    rows = [
        ("Fired hours (incl. warm-up)",
         f"{lt['total_hours']:.1f} h",
         f"{lt['qualification_hours']:.0f} h",
         lt["hours_fraction"], lt["hours_ok"]),
        ("Start cycles",
         f"{lt['mission_start_cycles']}",
         f"{lt['qualification_cycles']}",
         lt["cycles_fraction"], lt["cycles_ok"]),
        ("Total impulse",
         f"{lt['total_impulse_kns']:.1f} kN·s",
         f"{lt['qualification_impulse_kns']:.1f} kN·s",
         lt["impulse_fraction"], lt["impulse_ok"]),
    ]
    for label, mission_val, qual_val, frac, ok in rows:
        print("  " + col.format(
            label, mission_val, qual_val,
            f"{frac*100:.1f}%", bar(frac), status(ok)
        ))

    print(f"\n  Warm-up time per session : {lt.get('warmup_hours',0)*60/max(1,1):.1f} min  "
          f"(total warm-up hours: {lt.get('warmup_hours',0):.2f} h)")
    verdict = "✓  Thruster lifetime ADEQUATE for mission." if lt["overall_ok"] \
              else "✗  LIFETIME MARGIN INSUFFICIENT — review mission timeline or thruster selection."
    print(f"\n  {verdict}")


def print_rcs_blowdown(pressure_curve: list, rcs: RCSConfig):
    print_section(f"RCS BLOWDOWN PRESSURE MODEL — {rcs.system_name}")
    print(f"\n  Feed system     : Blowdown (He + propellant same tank)")
    print(f"  Initial pressure: {rcs.blowdown.p_initial_bar:.1f} bar  (MOP)")
    print(f"  Min pressure    : {rcs.blowdown.p_min_bar:.1f} bar  (MEP)")
    print(f"  Line dP nominal : {rcs.blowdown.line_dp_nominal_bar:.2f} bar")

    col = "{:>16} {:>14} {:>12} {:>14} {:>14} {:>10}"
    print("\n  " + col.format("Remaining(kg)", "Fraction(%)",
                               "Ullage(L)", "P_tank(bar)",
                               "P_inlet(bar)", "Below MEP"))
    print(f"  {'─' * 82}")

    # Print every 5th point for readability
    step = max(1, len(pressure_curve) // 10)
    for i, pt in enumerate(pressure_curve):
        if i % step == 0 or i == len(pressure_curve) - 1:
            flag = " ⚠" if pt["below_mep"] else ""
            print("  " + col.format(
                f"{pt['m_remaining_kg']:.2f}",
                f"{pt['fraction_remaining']*100:.1f}",
                f"{pt['v_ullage_l']:.2f}",
                f"{pt['p_tank_bar']:.2f}",
                f"{pt['p_inlet_bar']:.2f}",
                f"{'YES'+flag if pt['below_mep'] else 'no'}"
            ))


def print_rcs_regulated(pressure_curve: list, rcs: RCSConfig):
    print_section(f"RCS REGULATED PRESSURE MODEL — {rcs.system_name}")
    reg = rcs.regulated
    print(f"\n  Feed system      : Regulated (separate He bottle)")
    print(f"  Set-point        : {reg.p_setpoint_bar:.1f} ± {reg.p_setpoint_tolerance_bar:.1f} bar")
    print(f"  Lock-up pressure : {reg.p_lockup_bar:.1f} bar")
    print(f"  He bottle        : {reg.he_bottle_volume_l:.1f} L @ "
          f"{reg.he_bottle_p_initial_bar:.0f} bar initial")
    print(f"  Regulated → blowdown transition at He bottle: "
          f"{reg.he_bottle_p_min_bar:.0f} bar")
    print(f"  Line dP nominal  : {reg.line_dp_nominal_bar:.2f} bar")

    col = "{:>16} {:>14} {:>14} {:>16} {:>18} {:>14}"
    print("\n  " + col.format("Remaining(kg)", "Fraction(%)",
                               "He left(kg)", "P_bottle(bar)",
                               "P_inlet_nom(bar)", "Regulated?"))
    print(f"  {'─' * 96}")

    step = max(1, len(pressure_curve) // 10)
    for i, pt in enumerate(pressure_curve):
        if i % step == 0 or i == len(pressure_curve) - 1:
            flag = "" if pt["regulated_active"] else " → BLOWDOWN"
            print("  " + col.format(
                f"{pt['m_remaining_kg']:.2f}",
                f"{pt['fraction_remaining']*100:.1f}",
                f"{pt['he_remaining_kg']:.4f}",
                f"{pt['p_bottle_bar']:.2f}",
                f"{pt['p_inlet_nominal_bar']:.2f}",
                f"{'YES' if pt['regulated_active'] else 'NO'+flag}"
            ))


def print_rcs_isp_curve(rcs: RCSConfig):
    print_section(f"RCS Isp vs INLET PRESSURE — {rcs.system_name} (supplier curve)")
    col = "{:>20} {:>12} {:>12}"
    print("\n  " + col.format("Inlet Pressure(bar)", "Isp(s)", "Thrust(N)"))
    print(f"  {'─' * 46}")
    pts = sorted(rcs.performance_curve, key=lambda p: p.inlet_pressure_bar)
    for p in pts:
        print("  " + col.format(
            f"{p.inlet_pressure_bar:.2f}",
            f"{p.isp_s:.1f}",
            f"{p.thrust_n:.2f}"
        ))


def print_rcs_bol_eol(result: dict, rcs: RCSConfig):
    print_section(f"RCS BOL / EOL PERFORMANCE — {rcs.system_name}")
    print(f"\n  {'':30} {'BOL':>12} {'EOL':>12}")
    print(f"  {'─' * 56}")
    rows = [
        ("Inlet pressure (bar)", "p_inlet_bol_bar",  "p_inlet_eol_bar"),
        ("Isp (s)",              "isp_bol_s",         "isp_eol_s"),
        ("Thrust (N)",           "thrust_bol_n",      "thrust_eol_n"),
    ]
    for label, bol_key, eol_key in rows:
        print(f"  {label:<30} {result[bol_key]:>12.2f} {result[eol_key]:>12.2f}")
    print(f"\n  Isp degradation BOL→EOL : {result['isp_degradation_pct']:.2f}%")


def print_mr_sensitivity(sens: list, rcs: RCSConfig):
    if not sens:
        return
    print_section(f"RCS MIXTURE RATIO SENSITIVITY — {rcs.system_name}")
    col = "{:>16} {:>14} {:>10} {:>14}"
    print("\n  " + col.format("O/F ratio", "Isp factor", "Isp(s)", "ΔIsp(s)"))
    print(f"  {'─' * 56}")
    for pt in sens:
        print("  " + col.format(
            f"{pt['mixture_ratio']:.2f}",
            f"{pt['isp_factor']:.4f}",
            f"{pt['isp_s']:.1f}",
            f"{pt['delta_isp_s']:+.1f}"
        ))


def print_recommendations(ep_isp: dict, rcs_isp: dict, platform: str):
    print_header(f"RECOMMENDED ISP VALUES FOR TOOLS 1 & 2 — {platform}")
    print(f"\n  These values are derived from hardware qualification data.")
    print(f"  They replace the placeholder values in Tool 1 and Tool 2 configs.\n")
    col = "{:<30} {:>12} {:>12}"
    print(f"  " + col.format("Parameter", "BOL", "EOL"))
    print(f"  {'─' * 56}")
    print(f"  " + col.format("EP Isp (s)",
          f"{ep_isp['isp_bol']:.1f}", f"{ep_isp['isp_eol']:.1f}"))
    print(f"  " + col.format("EP Thrust (mN)",
          f"{ep_isp['thrust_bol_mn']:.1f}", f"{ep_isp['thrust_eol_mn']:.1f}"))
    print(f"  " + col.format("EP power used (W)",
          f"{ep_isp['power_bol_w']:.0f}", f"{ep_isp['power_eol_w']:.0f}"))
    print(f"  {'─' * 56}")
    print(f"  " + col.format("RCS Isp (s)",
          f"{rcs_isp['isp_bol_s']:.1f}", f"{rcs_isp['isp_eol_s']:.1f}"))
    print(f"  " + col.format("RCS inlet pressure (bar)",
          f"{rcs_isp['p_inlet_bol_bar']:.2f}", f"{rcs_isp['p_inlet_eol_bar']:.2f}"))
    print(f"  " + col.format("RCS thrust (N)",
          f"{rcs_isp['thrust_bol_n']:.2f}", f"{rcs_isp['thrust_eol_n']:.2f}"))


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> ThrusterConfig:
    with open(path) as f:
        c = json.load(f)

    throttle = [EPThrottlePoint(**p) for p in c["ep"]["throttle_table"]]
    lifetime = EPLifetimeLimits(**c["ep"]["lifetime"])
    ep_raw = {k: v for k, v in c["ep"].items()
              if k not in ("throttle_table", "lifetime")}
    ep = EPConfig(throttle_table=throttle, lifetime=lifetime, **ep_raw)

    perf_curve = [RCSPerformancePoint(**p) for p in c["rcs"]["performance_curve"]]
    tank = RCSTankConfig(**c["rcs"]["tank"])
    mr_sens = c["rcs"].get("mr_sensitivity", [])
    feed_type = c["rcs"]["feed_system_type"]

    blowdown = RCSBlowdownConfig(**c["rcs"]["blowdown"]) \
               if "blowdown" in c["rcs"] else None
    regulated = RCSRegulatedConfig(**c["rcs"]["regulated"]) \
                if "regulated" in c["rcs"] else None

    rcs_raw = {k: v for k, v in c["rcs"].items()
               if k not in ("performance_curve", "tank", "blowdown",
                             "regulated", "mr_sensitivity")}
    rcs = RCSConfig(tank=tank, performance_curve=perf_curve,
                    blowdown=blowdown, regulated=regulated,
                    mr_sensitivity=mr_sens, **rcs_raw)

    return ThrusterConfig(platform_name=c["platform_name"], ep=ep, rcs=rcs)


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN EXAMPLE — Platform A (blowdown RCS)
# ─────────────────────────────────────────────────────────────────────────────

def build_example() -> ThrusterConfig:
    """
    Illustrative Platform A thruster configuration.
    All values are illustrative — replace with supplier qualification data.
    C3H6/N2O spark ignition — no catalyst bed preheat model.
    """
    ep = EPConfig(
        thruster_name="HET-200 (illustrative)",
        throttle_table=[
            EPThrottlePoint(1200, 100.0, 1550, 6.57,  "Min throttle"),
            EPThrottlePoint(1600, 140.0, 1650, 8.65,  ""),
            EPThrottlePoint(2000, 180.0, 1720, 10.65, "Nominal"),
            EPThrottlePoint(2400, 215.0, 1770, 12.37, ""),
            EPThrottlePoint(2800, 245.0, 1800, 13.88, "Max throttle"),
        ],
        lifetime=EPLifetimeLimits(
            max_fired_hours=20000.0,
            max_start_cycles=50000,
            max_total_impulse_kns=15000.0
        ),
        sa_power_bol_w=2400.0,
        sa_power_eol_w=1900.0,   # ~20% SA degradation over mission life
        sa_power_eclipse_w=0.0,
        duty_cycle_fraction=0.85,
        warmup_time_min=15.0,
        warmup_flow_mg_s=1.5,    # Sub-nominal flow during conditioning
        mission_fired_hours=12000.0,
        mission_start_cycles=8000
    )

    # RCS — blowdown (Platform A)
    rcs = RCSConfig(
        system_name="RCS-BD-A (blowdown, C3H6/N2O)",
        feed_system_type="blowdown",
        tank=RCSTankConfig(
            oxidiser_loaded_kg=280.0,   # N2O
            fuel_loaded_kg=40.0,        # C3H6
            mixture_ratio_nominal=7.0,
            oxidiser_density_kg_l=1.226, # N2O liquid @ 20°C
            fuel_density_kg_l=0.613      # C3H6 liquid @ 20°C
        ),
        blowdown=RCSBlowdownConfig(
            tank_volume_l=320.0,
            p_initial_bar=24.0,          # MOP
            p_min_bar=6.0,               # MEP
            he_mass_kg=0.85,
            temperature_k=293.15,
            line_dp_nominal_bar=0.3
        ),
        performance_curve=[
            # Supplier Isp vs inlet pressure — illustrative C3H6/N2O data
            RCSPerformancePoint(5.0,  278.0, 8.2),
            RCSPerformancePoint(8.0,  288.0, 11.5),
            RCSPerformancePoint(12.0, 298.0, 15.8),
            RCSPerformancePoint(16.0, 305.0, 19.2),
            RCSPerformancePoint(20.0, 310.0, 22.0),
            RCSPerformancePoint(23.0, 313.0, 23.8),
        ],
        mr_sensitivity=[
            # Isp factor at off-nominal mixture ratios (from supplier data)
            {"mr": 5.5, "isp_factor": 0.972},
            {"mr": 6.0, "isp_factor": 0.988},
            {"mr": 6.5, "isp_factor": 0.996},
            {"mr": 7.0, "isp_factor": 1.000},  # Nominal
            {"mr": 7.5, "isp_factor": 0.997},
            {"mr": 8.0, "isp_factor": 0.989},
            {"mr": 8.5, "isp_factor": 0.975},
        ]
    )

    return ThrusterConfig(platform_name="Platform_A", ep=ep, rcs=rcs)


def build_example_regulated() -> ThrusterConfig:
    """
    Illustrative Platform B thruster configuration — regulated RCS feed system.
    """
    cfg = build_example()
    cfg.platform_name = "Platform_B"
    cfg.rcs = RCSConfig(
        system_name="RCS-REG-B (regulated, C3H6/N2O)",
        feed_system_type="regulated",
        tank=RCSTankConfig(
            oxidiser_loaded_kg=180.0,
            fuel_loaded_kg=26.0,
            mixture_ratio_nominal=7.0,
            oxidiser_density_kg_l=1.226,
            fuel_density_kg_l=0.613
        ),
        regulated=RCSRegulatedConfig(
            p_setpoint_bar=18.0,
            p_setpoint_tolerance_bar=0.5,
            p_lockup_bar=19.2,
            p_relief_bar=21.0,
            he_bottle_volume_l=4.0,
            he_bottle_p_initial_bar=350.0,
            he_bottle_p_min_bar=30.0,    # Below this → blowdown transition
            temperature_k=293.15,
            line_dp_nominal_bar=0.3
        ),
        performance_curve=[
            RCSPerformancePoint(5.0,  278.0, 8.2),
            RCSPerformancePoint(8.0,  288.0, 11.5),
            RCSPerformancePoint(12.0, 298.0, 15.8),
            RCSPerformancePoint(16.0, 305.0, 19.2),
            RCSPerformancePoint(20.0, 310.0, 22.0),
            RCSPerformancePoint(23.0, 313.0, 23.8),
        ],
        mr_sensitivity=[
            {"mr": 5.5, "isp_factor": 0.972},
            {"mr": 6.0, "isp_factor": 0.988},
            {"mr": 7.0, "isp_factor": 1.000},
            {"mr": 7.5, "isp_factor": 0.997},
            {"mr": 8.5, "isp_factor": 0.975},
        ]
    )
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_tool3(cfg: ThrusterConfig, output_dir: str = ".",
              patch_t1: str = None, patch_t2: str = None):

    print_header(f"THRUSTER PERFORMANCE MODEL — {cfg.platform_name}")
    print(f"\n  EP  : {cfg.ep.thruster_name}")
    print(f"  RCS : {cfg.rcs.system_name}  [{cfg.rcs.feed_system_type}]")
    print(f"  Propellants: Xe (EP) | N2O / C3H6 (RCS) — REACH compliant")
    print(f"  Ignition   : Spark / torch — no catalyst bed preheat model")

    # ── EP ────────────────────────────────────────────────────────────────────
    print_ep_throttle_table(cfg.ep)

    wu = ep_warmup_xenon(cfg.ep)
    print_section("EP WARM-UP XENON (non-propulsive)")
    print(f"\n  Warm-up time per session : {cfg.ep.warmup_time_min:.1f} min")
    print(f"  Xe per session           : {wu['xe_per_session_kg']*1000:.3f} g")
    print(f"  Total sessions (mission) : {wu['total_sessions']}")
    print(f"  Total warm-up Xe         : {wu['xe_warmup_total_kg']*1000:.1f} g  "
          f"({wu['xe_warmup_total_kg']:.4f} kg)")

    lt = ep_lifetime_check(cfg.ep)
    print_ep_lifetime(lt)

    ep_isp = ep_recommended_isp(cfg.ep)

    # Power sweep curve
    print_section("EP PERFORMANCE ACROSS POWER RANGE")
    pts = sorted(cfg.ep.throttle_table, key=lambda p: p.power_w)
    p_min = pts[0].power_w
    p_max = pts[-1].power_w
    col = "{:>10} {:>12} {:>10} {:>14}"
    print("\n  " + col.format("Power(W)", "Thrust(mN)", "Isp(s)", "Flow(mg/s)"))
    print(f"  {'─' * 50}")
    n_sweep = 10
    for i in range(n_sweep + 1):
        pw = p_min + i * (p_max - p_min) / n_sweep
        perf = ep_at_power(cfg.ep, pw)
        marker = " ← BOL" if abs(pw - cfg.ep.sa_power_bol_w) < 50 else \
                 " ← EOL" if abs(pw - cfg.ep.sa_power_eol_w) < 50 else ""
        print("  " + col.format(
            f"{pw:.0f}", f"{perf['thrust_mn']:.1f}",
            f"{perf['isp_s']:.1f}", f"{perf['flow_mg_s']:.3f}"
        ) + marker)

    # ── RCS ───────────────────────────────────────────────────────────────────
    print_rcs_isp_curve(cfg.rcs)

    if cfg.rcs.feed_system_type == "blowdown":
        pressure_curve = blowdown_pressure_vs_remaining(cfg.rcs)
        print_rcs_blowdown(pressure_curve, cfg.rcs)
    else:
        pressure_curve = regulated_he_bottle_depletion(cfg.rcs)
        print_rcs_regulated(pressure_curve, cfg.rcs)

    rcs_isp = rcs_bol_eol_isp(cfg.rcs, pressure_curve)
    print_rcs_bol_eol(rcs_isp, cfg.rcs)

    mr_sens = rcs_mr_sensitivity(cfg.rcs)
    print_mr_sensitivity(mr_sens, cfg.rcs)

    # ── Recommendations ───────────────────────────────────────────────────────
    print_recommendations(ep_isp, rcs_isp, cfg.platform_name)

    # ── CSV exports ───────────────────────────────────────────────────────────
    p1 = export_ep_throttle_csv(cfg.ep, cfg.platform_name, output_dir)
    p2 = export_rcs_pressure_csv(pressure_curve, cfg.platform_name,
                                  cfg.rcs.feed_system_type, output_dir)
    print(f"\n  ✓  EP throttle CSV   : {p1}")
    print(f"  ✓  RCS pressure CSV  : {p2}")

    # ── Patch Tool 1 config ───────────────────────────────────────────────────
    if patch_t1:
        print_section("PATCHING TOOL 1 CONFIG")
        out, log = patch_tool1_config(patch_t1, ep_isp, rcs_isp, output_dir)
        for line in log:
            print(line)
        print(f"\n  ✓  Patched Tool 1 config written to: {out}")

    # ── Patch Tool 2 config ───────────────────────────────────────────────────
    if patch_t2:
        print_section("PATCHING TOOL 2 CONFIG")
        out, log = patch_tool2_config(patch_t2, ep_isp, rcs_isp, output_dir)
        for line in log:
            print(line)
        print(f"\n  ✓  Patched Tool 2 config written to: {out}")

    print_header("TOOL 3 COMPLETE")
    print(f"\n  EP  BOL Isp : {ep_isp['isp_bol']:.1f} s  |  "
          f"EOL Isp : {ep_isp['isp_eol']:.1f} s")
    print(f"  RCS BOL Isp : {rcs_isp['isp_bol_s']:.1f} s  |  "
          f"EOL Isp : {rcs_isp['isp_eol_s']:.1f} s  |  "
          f"Degradation : {rcs_isp['isp_degradation_pct']:.2f}%\n")


def main():
    parser = argparse.ArgumentParser(
        description="Thruster Performance Model — EP throttle table + RCS pressure model"
    )
    parser.add_argument("--config",      type=str,  help="Path to thruster JSON config")
    parser.add_argument("--example",     action="store_true",
                        help="Run built-in Platform A (blowdown) example")
    parser.add_argument("--example-reg", action="store_true",
                        help="Run built-in Platform B (regulated) example")
    parser.add_argument("--patch-tool1", type=str,  metavar="TOOL1_JSON",
                        help="Path to Tool 1 config JSON to patch with recommended Isp")
    parser.add_argument("--patch-tool2", type=str,  metavar="TOOL2_JSON",
                        help="Path to Tool 2 config JSON to patch with recommended Isp")
    parser.add_argument("--output",      type=str,  default=".",
                        help="Output directory for CSV and patched configs")
    args = parser.parse_args()

    if args.example:
        cfg = build_example()
        run_tool3(cfg, output_dir=args.output,
                  patch_t1=args.patch_tool1, patch_t2=args.patch_tool2)
    elif args.example_reg:
        cfg = build_example_regulated()
        run_tool3(cfg, output_dir=args.output,
                  patch_t1=args.patch_tool1, patch_t2=args.patch_tool2)
    elif args.config:
        cfg = load_config(args.config)
        run_tool3(cfg, output_dir=args.output,
                  patch_t1=args.patch_tool1, patch_t2=args.patch_tool2)
    else:
        parser.print_help()
        print("\n  Tips:")
        print("    --example              Platform A blowdown RCS")
        print("    --example-reg          Platform B regulated RCS")
        print("    --patch-tool1 <file>   Auto-update Tool 1 Isp values")
        print("    --patch-tool2 <file>   Auto-update Tool 2 Isp values")


if __name__ == "__main__":
    main()
