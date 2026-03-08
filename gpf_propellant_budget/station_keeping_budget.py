"""
================================================================================
STATION KEEPING BUDGET TOOL — Tool 2
Propulsion Engineering | GNC/AOCS Interface Tool
================================================================================
Platforms : Platform A (post-docking stack SK via thruster arm)
            Platform B (solo SK)
RCS        : C3H6 / N2O bipropellant (REACH compliant)
EP         : Xenon — primary SK propulsion
Manoeuvres : N/S and E/W executed SEPARATELY

N/S delta-V: Analytical estimate from luni-solar perturbation theory
             OR user-supplied GNC simulation value (overrides analytical)
E/W delta-V: Analytical estimate from tesseral harmonics + SRP eccentricity drift
             OR user-supplied GNC simulation value

Thruster arm geometry:
  - Separate arm angles defined for N/S and E/W axes
  - Cosine losses computed per axis independently
  - Arm angle uncertainty sensitivity analysis included

Stack handling:
  - Client satellite mass and CoM treated as parametric inputs
  - Budget computed across a range of client masses
  - CoM offset flagged as disturbance torque warning

Usage:
    python station_keeping_budget.py --config sk_config.json
    python station_keeping_budget.py --example
    python station_keeping_budget.py --example --sensitivity

Outputs:
    - Annual and total mission SK delta-V per axis
    - Xenon and RCS propellant consumed
    - Arm angle cosine loss breakdown
    - Sensitivity table: propellant vs arm angle uncertainty
    - Parametric table: propellant vs client satellite mass
    - CSV export
================================================================================
"""

import json
import csv
import math
import argparse
import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
G0           = 9.80665      # Standard gravity (m/s²)
MU_EARTH     = 398600.4418  # Earth gravitational parameter (km³/s²)
R_GEO        = 42164.2      # GEO radius (km)
V_GEO        = math.sqrt(MU_EARTH / R_GEO)  # GEO orbital velocity (km/s)

# Luni-solar N/S drift coefficient
# Standard industry approximation: dV_NS ≈ 50.0 m/s/year at GEO
# (varies ±15% with epoch, inclination history, and longitude)
# Reference: Soop, Handbook of Geostationary Orbits, ESA 1994
NS_BASE_DV_PER_YEAR = 50.0  # m/s/year — analytical baseline

# GEO tesseral harmonic longitude-dependent E/W drift coefficients
# Stable longitudes near 75°E and 255°E, unstable near 165°E and 345°E
# Acceleration (deg/day²) as a function of distance from nearest stable point
# Simplified model: a_EW ≈ K * sin(2*(lon - lon_stable))
# K ≈ 0.00168 deg/day² (from JGM-3 gravity model, dominant J22 term)
K_TESSERAL   = 0.00168      # deg/day² — tesseral harmonic acceleration coefficient
STABLE_LONS  = [75.0, 255.0]  # Stable equilibrium longitudes (degrees East)

# Solar radiation pressure constant
P_SRP        = 4.56e-6      # N/m² at 1 AU


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThrusterArmConfig:
    """
    Thruster arm geometry for a single SK axis.
    The arm positions the EP thruster at an angle relative to the ideal
    thrust direction for that axis. N/S and E/W arms are configured separately.
    """
    axis: str                        # "NS" or "EW"
    arm_angle_nominal_deg: float     # Nominal arm angle offset from thrust axis (deg)
    arm_angle_uncertainty_deg: float # ±1-sigma positioning uncertainty (deg)
    notes: str = ""


@dataclass
class EPSKConfig:
    """EP system parameters for SK burns."""
    isp_bol: float           # BOL Isp (s)
    isp_eol: float           # EOL Isp (s)
    thrust_mn: float         # Thrust level (mN)
    duty_cycle: float        # Fraction of SK period EP can fire (0–1)
    warmup_xenon_kg: float   # Xenon per warm-up session (kg)
    warmup_sessions_per_year: float  # Warm-up sessions per year


@dataclass
class RCSACSConfig:
    """RCS parameters for attitude control during SK burns."""
    isp_bol: float
    isp_eol: float
    mixture_ratio: float     # N2O/C3H6
    acs_dv_per_year: float   # m/s/year — ACS delta-V during SK (from GNC simulation)
    margin_performance: float
    margin_execution: float
    margin_operational: float


@dataclass
class DisturbanceConfig:
    """
    Disturbance environment inputs for SK delta-V computation.
    If gnc_override values are provided, the analytical estimate is bypassed
    for that axis and the GNC-supplied value is used instead.
    """
    # Station parameters
    station_longitude_deg: float     # GEO station longitude (degrees East, 0–360)
    mission_epoch_year: float        # Mission start year (for luni-solar amplitude)
    sk_deadband_deg: float           # Station keeping box half-width (degrees)

    # N/S disturbance
    ns_inclination_correction: float = 1.0   # Correction factor on base 50 m/s/year (1.0 = nominal)
    ns_gnc_override_dv_per_year: Optional[float] = None  # If set, overrides analytical N/S

    # E/W disturbance — SRP eccentricity drift
    spacecraft_area_m2: float = 30.0         # Solar pressure area (m²) — platform solo
    spacecraft_area_stack_m2: float = 50.0   # Solar pressure area with client docked (m²)
    srp_cr: float = 1.3                      # Radiation pressure coefficient (1.0–2.0)

    # E/W GNC override
    ew_gnc_override_dv_per_year: Optional[float] = None  # If set, overrides analytical E/W


@dataclass
class StackConfig:
    """
    Post-docking stack configuration.
    Client satellite mass and CoM are parametric — a range is analysed.
    """
    platform_mass_kg: float          # Platform dry + propellant mass at SK start
    client_mass_min_kg: float        # Minimum client satellite mass (kg)
    client_mass_nominal_kg: float    # Nominal client satellite mass (kg)
    client_mass_max_kg: float        # Maximum client satellite mass (kg)
    client_com_offset_m: float = 0.0 # CoM offset of client from nominal docking axis (m)
    notes: str = ""


@dataclass
class SKConfig:
    """Top-level station keeping configuration."""
    platform_name: str
    mission_duration_years: float    # Total SK duration
    stack_config: StackConfig
    disturbance: DisturbanceConfig
    arm_ns: ThrusterArmConfig        # Thruster arm config for N/S axis
    arm_ew: ThrusterArmConfig        # Thruster arm config for E/W axis
    ep: EPSKConfig
    rcs_acs: RCSACSConfig


# ─────────────────────────────────────────────────────────────────────────────
# DISTURBANCE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def compute_ns_dv_per_year(dist: DisturbanceConfig) -> tuple:
    """
    Compute N/S SK delta-V per year.

    Analytical method:
      dV_NS ≈ 50.0 * correction_factor  [m/s/year]
      The 50 m/s/year baseline captures the dominant luni-solar secular
      inclination drift at GEO averaged over the 18.6-year lunar nodal cycle.
      The correction factor allows the user to adjust for epoch-specific
      amplitude (higher near lunar nodal crossing, lower at mid-cycle).

    If gnc_override is provided, that value is used directly and the
    analytical estimate is reported alongside for comparison only.

    Returns: (dv_used, dv_analytical, source)
    """
    dv_analytical = NS_BASE_DV_PER_YEAR * dist.ns_inclination_correction

    if dist.ns_gnc_override_dv_per_year is not None:
        return dist.ns_gnc_override_dv_per_year, dv_analytical, "GNC simulation override"
    else:
        return dv_analytical, dv_analytical, "Analytical (luni-solar, 50 m/s/yr baseline)"


def compute_ew_dv_per_year(dist: DisturbanceConfig, spacecraft_area_m2: float,
                            spacecraft_mass_kg: float) -> tuple:
    """
    Compute E/W SK delta-V per year.

    Two contributions:
    1. Tesseral harmonic drift — gravitational acceleration toward nearest
       unstable equilibrium point. The satellite must fire to stay within
       the station box. Delta-V depends on distance from stable longitude.

       Simplified model:
         a_EW = K * sin(2 * delta_lon)   [deg/day²]
         where delta_lon = distance from nearest stable longitude

       Converting to delta-V per year:
         dV_EW_tesseral ≈ a_EW * (deadband / a_EW)^0.5 * pi  [m/s/year]
         (parabolic orbit within deadband, firing at edges)

    2. SRP eccentricity drift — solar radiation pressure pumps the
       eccentricity vector, causing apparent E/W longitude oscillation.
       Must be corrected periodically.

         dV_EW_SRP ≈ 2 * P_SRP * Cr * (A/m) * V_GEO * T_year  [m/s/year]

    If gnc_override is provided, that value is used directly.

    Returns: (dv_used, dv_tesseral, dv_srp, dv_analytical_total, source)
    """
    # Find nearest stable longitude distance
    delta_lons = [abs(dist.station_longitude_deg - s) for s in STABLE_LONS]
    delta_lons += [abs(dist.station_longitude_deg - s - 360) for s in STABLE_LONS]
    delta_lons += [abs(dist.station_longitude_deg - s + 360) for s in STABLE_LONS]
    delta_lon = min(delta_lons)

    # Tesseral acceleration (deg/day²)
    a_ew_deg_day2 = K_TESSERAL * abs(math.sin(math.radians(2 * delta_lon)))

    # Convert to m/s² :
    # 1 deg longitude at GEO = (pi/180) * R_GEO * 1000 m
    # 1 day = 86400 s
    deg_to_m = (math.pi / 180.0) * R_GEO * 1000.0  # m per degree at GEO
    a_ew_ms2 = a_ew_deg_day2 * deg_to_m / (86400.0 ** 2)

    # Delta-V for parabolic SK within deadband (standard result)
    # dV = 2 * sqrt(a * deadband_m)  per manoeuvre cycle
    # Cycles per year = sqrt(a / deadband_m) * T_year / (2*pi) ... simplified:
    # Annual dV ≈ pi * sqrt(a_ew * deadband_m) * (T_year_seconds / T_cycle)
    # Practical formula: dV_year = 2 * a_ew_ms2 * 365.25*86400 (upper bound, constant fire)
    # Standard industry formula for box-keeping:
    deadband_m = dist.sk_deadband_deg * deg_to_m
    if a_ew_ms2 > 0:
        dv_tesseral = math.pi * math.sqrt(a_ew_ms2 * deadband_m) * \
                      (365.25 * 86400) / (2 * math.pi * math.sqrt(deadband_m / a_ew_ms2))
        # Simplifies to: dV_tesseral = a_ew_ms2 * 365.25 * 86400 / 2
        dv_tesseral = 0.5 * a_ew_ms2 * 365.25 * 86400.0
    else:
        dv_tesseral = 0.0  # At stable longitude — no tesseral drift

    # SRP eccentricity drift delta-V per year
    # dV_SRP = 2 * P_SRP * Cr * (A/m) * V_GEO [m/s per orbit] * orbits_per_year
    # GEO: 1 orbit per sidereal day
    am_ratio = spacecraft_area_m2 / spacecraft_mass_kg  # m²/kg
    dv_srp = 2.0 * P_SRP * dist.srp_cr * am_ratio * (V_GEO * 1000.0) * 365.25

    dv_analytical = dv_tesseral + dv_srp

    if dist.ew_gnc_override_dv_per_year is not None:
        return (dist.ew_gnc_override_dv_per_year, dv_tesseral,
                dv_srp, dv_analytical, "GNC simulation override")
    else:
        return (dv_analytical, dv_tesseral,
                dv_srp, dv_analytical, "Analytical (tesseral + SRP)")


# ─────────────────────────────────────────────────────────────────────────────
# THRUSTER ARM GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def effective_dv(dv_required: float, arm_angle_deg: float) -> float:
    """
    Compute the delta-V that must be commanded to deliver dv_required
    along the manoeuvre axis, accounting for the cosine loss from the
    thruster arm angle.
    """
    if arm_angle_deg == 0.0:
        return dv_required
    cos_angle = math.cos(math.radians(arm_angle_deg))
    if cos_angle <= 0:
        raise ValueError(f"Arm angle {arm_angle_deg}° gives zero/negative thrust component.")
    return dv_required / cos_angle


def cosine_loss_pct(arm_angle_deg: float) -> float:
    """Return the cosine loss as a percentage."""
    return (1.0 - math.cos(math.radians(arm_angle_deg))) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# TSIOLKOVSKY
# ─────────────────────────────────────────────────────────────────────────────

def propellant_from_dv(dv: float, isp: float, wet_mass: float) -> float:
    """Propellant mass (kg) for a given delta-V, Isp and wet mass."""
    if dv <= 0:
        return 0.0
    return wet_mass * (1.0 - math.exp(-dv / (G0 * isp)))


def split_biprop(total_kg: float, mr: float) -> tuple:
    """Split total RCS mass into (N2O_kg, C3H6_kg) given mixture ratio mr = N2O/C3H6."""
    c3h6 = total_kg / (1.0 + mr)
    n2o  = total_kg - c3h6
    return n2o, c3h6


def margin_multiplier(rcs: RCSACSConfig) -> float:
    return (1 + rcs.margin_performance) * (1 + rcs.margin_execution) * (1 + rcs.margin_operational)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE CONFIGURATION BUDGET
# ─────────────────────────────────────────────────────────────────────────────

def compute_sk_budget(cfg: SKConfig, client_mass_kg: float, case: str = "BOL") -> dict:
    """
    Compute annual and total SK budget for a given client mass and BOL/EOL case.

    Returns a comprehensive result dictionary.
    """
    stack_mass = cfg.stack_config.platform_mass_kg + client_mass_kg
    dist       = cfg.disturbance
    ep         = cfg.ep
    rcs        = cfg.rcs_acs
    arm_ns     = cfg.arm_ns
    arm_ew     = cfg.arm_ew
    years      = cfg.mission_duration_years

    isp_ep  = ep.isp_bol  if case == "BOL" else ep.isp_eol
    isp_rcs = rcs.isp_bol if case == "BOL" else rcs.isp_eol

    # ── Determine spacecraft area for E/W (stack has larger area) ────────────
    area = (dist.spacecraft_area_stack_m2 if client_mass_kg > 0
            else dist.spacecraft_area_m2)

    # ── N/S delta-V ──────────────────────────────────────────────────────────
    ns_dv_year, ns_dv_analytical, ns_source = compute_ns_dv_per_year(dist)

    # Apply thruster arm cosine loss for N/S axis
    ns_dv_commanded_year = effective_dv(ns_dv_year, arm_ns.arm_angle_nominal_deg)
    ns_cosine_loss_pct   = cosine_loss_pct(arm_ns.arm_angle_nominal_deg)

    # ── E/W delta-V ──────────────────────────────────────────────────────────
    ew_dv_year, ew_dv_tesseral, ew_dv_srp, ew_dv_analytical, ew_source = \
        compute_ew_dv_per_year(dist, area, stack_mass)

    ew_dv_commanded_year = effective_dv(ew_dv_year, arm_ew.arm_angle_nominal_deg)
    ew_cosine_loss_pct   = cosine_loss_pct(arm_ew.arm_angle_nominal_deg)

    # ── Total commanded SK delta-V per year ──────────────────────────────────
    total_dv_commanded_year = ns_dv_commanded_year + ew_dv_commanded_year

    # ── Xenon consumption — EP provides SK delta-V ───────────────────────────
    # Annual xenon from delta-V (Tsiolkovsky on annual basis)
    # We approximate by computing over full mission then dividing,
    # since mass changes slowly during SK
    xe_ns_total  = propellant_from_dv(ns_dv_commanded_year  * years, isp_ep, stack_mass)
    xe_ew_total  = propellant_from_dv(ew_dv_commanded_year  * years, isp_ep,
                                       stack_mass - xe_ns_total)
    xe_dv_total  = xe_ns_total + xe_ew_total
    xe_dv_year   = xe_dv_total / years

    # Warm-up xenon
    xe_warmup_year  = ep.warmup_xenon_kg * ep.warmup_sessions_per_year
    xe_warmup_total = xe_warmup_year * years

    xe_total_year   = xe_dv_year  + xe_warmup_year
    xe_total_mission = xe_dv_total + xe_warmup_total

    # ── RCS ACS consumption during SK burns ──────────────────────────────────
    acs_dv_total = rcs.acs_dv_per_year * years
    rcs_prop_nominal = propellant_from_dv(acs_dv_total, isp_rcs, stack_mass - xe_dv_total)
    rcs_prop_total   = rcs_prop_nominal * margin_multiplier(rcs)
    rcs_margin_total = rcs_prop_total - rcs_prop_nominal

    rcs_n2o_total, rcs_c3h6_total = split_biprop(rcs_prop_total, rcs.mixture_ratio)

    rcs_prop_year    = rcs_prop_total  / years
    rcs_n2o_year     = rcs_n2o_total   / years
    rcs_c3h6_year    = rcs_c3h6_total  / years

    # ── CoM offset warning ────────────────────────────────────────────────────
    com_warning = None
    if cfg.stack_config.client_com_offset_m > 0.1:
        com_warning = (
            f"Client CoM offset {cfg.stack_config.client_com_offset_m:.2f} m from docking axis. "
            f"Disturbance torque during SK burns will increase ACS propellant. "
            f"Re-evaluate acs_dv_per_year with GNC simulation at this CoM offset."
        )

    return {
        "case": case,
        "client_mass_kg": client_mass_kg,
        "stack_mass_kg": stack_mass,
        "isp_ep_used": isp_ep,
        "isp_rcs_used": isp_rcs,
        # N/S
        "ns_dv_required_year": ns_dv_year,
        "ns_dv_analytical_year": ns_dv_analytical,
        "ns_dv_source": ns_source,
        "ns_arm_angle_deg": arm_ns.arm_angle_nominal_deg,
        "ns_cosine_loss_pct": ns_cosine_loss_pct,
        "ns_dv_commanded_year": ns_dv_commanded_year,
        "ns_dv_commanded_total": ns_dv_commanded_year * years,
        "ns_xe_year": xe_ns_total / years,
        "ns_xe_total": xe_ns_total,
        # E/W
        "ew_dv_tesseral_year": ew_dv_tesseral,
        "ew_dv_srp_year": ew_dv_srp,
        "ew_dv_required_year": ew_dv_year,
        "ew_dv_analytical_year": ew_dv_analytical,
        "ew_dv_source": ew_source,
        "ew_arm_angle_deg": arm_ew.arm_angle_nominal_deg,
        "ew_cosine_loss_pct": ew_cosine_loss_pct,
        "ew_dv_commanded_year": ew_dv_commanded_year,
        "ew_dv_commanded_total": ew_dv_commanded_year * years,
        "ew_xe_year": xe_ew_total / years,
        "ew_xe_total": xe_ew_total,
        # Totals
        "total_dv_commanded_year": total_dv_commanded_year,
        "total_dv_commanded_mission": total_dv_commanded_year * years,
        "xe_dv_year": xe_dv_year,
        "xe_dv_total": xe_dv_total,
        "xe_warmup_year": xe_warmup_year,
        "xe_warmup_total": xe_warmup_total,
        "xe_total_year": xe_total_year,
        "xe_total_mission": xe_total_mission,
        # RCS ACS
        "rcs_acs_dv_year": rcs.acs_dv_per_year,
        "rcs_prop_nominal_total": rcs_prop_nominal,
        "rcs_margin_total": rcs_margin_total,
        "rcs_prop_total": rcs_prop_total,
        "rcs_n2o_total": rcs_n2o_total,
        "rcs_c3h6_total": rcs_c3h6_total,
        "rcs_prop_year": rcs_prop_year,
        "rcs_n2o_year": rcs_n2o_year,
        "rcs_c3h6_year": rcs_c3h6_year,
        "com_warning": com_warning,
        "mission_duration_years": years,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSES
# ─────────────────────────────────────────────────────────────────────────────

def arm_angle_sensitivity(cfg: SKConfig, client_mass_kg: float,
                           case: str = "EOL", n_steps: int = 9) -> list:
    """
    Sensitivity of total Xenon consumption to arm angle uncertainty.
    Sweeps both NS and EW arm angles simultaneously over ±uncertainty range.
    Returns list of result dicts with varied arm angles.
    """
    results = []
    uncertainty_ns = cfg.arm_ns.arm_angle_uncertainty_deg
    uncertainty_ew = cfg.arm_ew.arm_angle_uncertainty_deg
    nominal_ns     = cfg.arm_ns.arm_angle_nominal_deg
    nominal_ew     = cfg.arm_ew.arm_angle_nominal_deg

    steps = [i / (n_steps - 1) for i in range(n_steps)]  # 0.0 to 1.0
    offsets = [-1.0 + 2.0 * s for s in steps]             # -1.0 to +1.0

    for offset in offsets:
        cfg_mod = SKConfig(
            platform_name=cfg.platform_name,
            mission_duration_years=cfg.mission_duration_years,
            stack_config=cfg.stack_config,
            disturbance=cfg.disturbance,
            arm_ns=ThrusterArmConfig(
                axis="NS",
                arm_angle_nominal_deg=nominal_ns + offset * uncertainty_ns,
                arm_angle_uncertainty_deg=uncertainty_ns
            ),
            arm_ew=ThrusterArmConfig(
                axis="EW",
                arm_angle_nominal_deg=nominal_ew + offset * uncertainty_ew,
                arm_angle_uncertainty_deg=uncertainty_ew
            ),
            ep=cfg.ep,
            rcs_acs=cfg.rcs_acs
        )
        r = compute_sk_budget(cfg_mod, client_mass_kg, case)
        r["arm_offset_sigma"] = offset
        r["ns_arm_angle_varied"] = nominal_ns + offset * uncertainty_ns
        r["ew_arm_angle_varied"] = nominal_ew + offset * uncertainty_ew
        results.append(r)

    return results


def client_mass_parametric(cfg: SKConfig, case: str = "EOL", n_steps: int = 7) -> list:
    """
    Parametric sweep of SK budget vs client satellite mass.
    Sweeps from client_mass_min to client_mass_max.
    Returns list of result dicts.
    """
    sc   = cfg.stack_config
    step = (sc.client_mass_max_kg - sc.client_mass_min_kg) / (n_steps - 1)
    masses = [sc.client_mass_min_kg + i * step for i in range(n_steps)]

    results = []
    for m in masses:
        r = compute_sk_budget(cfg, m, case)
        results.append(r)
    return results


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


def print_sk_summary(r: dict, cfg: SKConfig):
    years = cfg.mission_duration_years
    print(f"\n  Platform          : {cfg.platform_name}")
    print(f"  Case              : {r['case']}")
    print(f"  SK duration       : {years:.1f} years")
    print(f"  Stack mass        : {r['stack_mass_kg']:.1f} kg  "
          f"(platform {cfg.stack_config.platform_mass_kg:.1f} kg  "
          f"+ client {r['client_mass_kg']:.1f} kg)")
    print(f"  EP Isp used       : {r['isp_ep_used']:.0f} s   "
          f"RCS Isp used : {r['isp_rcs_used']:.0f} s")

    print_section("N/S STATION KEEPING")
    print(f"  Source            : {r['ns_dv_source']}")
    if r['ns_dv_source'].startswith("GNC"):
        print(f"  Analytical est.   : {r['ns_dv_analytical_year']:.2f} m/s/year  "
              f"(shown for reference)")
    print(f"  Required dV       : {r['ns_dv_required_year']:.2f} m/s/year  "
          f"→  {r['ns_dv_required_year'] * years:.2f} m/s total")
    print(f"  Arm angle (N/S)   : {r['ns_arm_angle_deg']:.2f}°  "
          f"→  cosine loss {r['ns_cosine_loss_pct']:.3f}%")
    print(f"  Commanded dV      : {r['ns_dv_commanded_year']:.2f} m/s/year  "
          f"→  {r['ns_dv_commanded_total']:.2f} m/s total")
    print(f"  Xenon (N/S)       : {r['ns_xe_year']:.3f} kg/year  "
          f"→  {r['ns_xe_total']:.3f} kg total")

    print_section("E/W STATION KEEPING")
    print(f"  Source            : {r['ew_dv_source']}")
    if r['ew_dv_source'].startswith("GNC"):
        print(f"  Analytical est.   : {r['ew_dv_analytical_year']:.2f} m/s/year  "
              f"(shown for reference)")
    else:
        print(f"  Tesseral drift    : {r['ew_dv_tesseral_year']:.2f} m/s/year")
        print(f"  SRP eccentricity  : {r['ew_dv_srp_year']:.4f} m/s/year")
    print(f"  Required dV       : {r['ew_dv_required_year']:.2f} m/s/year  "
          f"→  {r['ew_dv_required_year'] * years:.2f} m/s total")
    print(f"  Arm angle (E/W)   : {r['ew_arm_angle_deg']:.2f}°  "
          f"→  cosine loss {r['ew_cosine_loss_pct']:.3f}%")
    print(f"  Commanded dV      : {r['ew_dv_commanded_year']:.2f} m/s/year  "
          f"→  {r['ew_dv_commanded_total']:.2f} m/s total")
    print(f"  Xenon (E/W)       : {r['ew_xe_year']:.3f} kg/year  "
          f"→  {r['ew_xe_total']:.3f} kg total")

    print_section("TOTAL SK PROPELLANT BUDGET")
    col = "{:<35} {:>12} {:>12}"
    print(f"\n  " + col.format("Item", "Per Year", "Mission Total"))
    print(f"  {'─' * 60}")
    print(f"  " + col.format("Total commanded dV (m/s)",
          f"{r['total_dv_commanded_year']:.2f}",
          f"{r['total_dv_commanded_mission']:.2f}"))
    print(f"  " + col.format("Xenon — delta-V (kg)",
          f"{r['xe_dv_year']:.3f}",
          f"{r['xe_dv_total']:.3f}"))
    print(f"  " + col.format("Xenon — warm-up (kg)",
          f"{r['xe_warmup_year']:.3f}",
          f"{r['xe_warmup_total']:.3f}"))
    print(f"  " + col.format("Xenon — TOTAL (kg)",
          f"{r['xe_total_year']:.3f}",
          f"{r['xe_total_mission']:.3f}"))
    print(f"  {'─' * 60}")
    print(f"  " + col.format("RCS ACS dV (m/s)",
          f"{r['rcs_acs_dv_year']:.2f}",
          f"{r['rcs_acs_dv_year'] * years:.2f}"))
    print(f"  " + col.format("RCS propellant nominal (kg)",
          f"{r['rcs_prop_nominal_total']/years:.3f}",
          f"{r['rcs_prop_nominal_total']:.3f}"))
    print(f"  " + col.format("RCS margin (kg)",
          f"{r['rcs_margin_total']/years:.3f}",
          f"{r['rcs_margin_total']:.3f}"))
    print(f"  " + col.format("RCS total (kg)",
          f"{r['rcs_prop_year']:.3f}",
          f"{r['rcs_prop_total']:.3f}"))
    print(f"  " + col.format("  of which N2O (kg)",
          f"{r['rcs_n2o_year']:.3f}",
          f"{r['rcs_n2o_total']:.3f}"))
    print(f"  " + col.format("  of which C3H6 (kg)",
          f"{r['rcs_c3h6_year']:.3f}",
          f"{r['rcs_c3h6_total']:.3f}"))

    if r.get("com_warning"):
        print(f"\n  ⚠  CoM WARNING: {r['com_warning']}")


def print_arm_sensitivity(sens_results: list, cfg: SKConfig):
    print_section(
        f"ARM ANGLE SENSITIVITY — EOL, client mass "
        f"{cfg.stack_config.client_mass_nominal_kg:.0f} kg nominal"
    )
    print(f"\n  NS arm nominal: {cfg.arm_ns.arm_angle_nominal_deg:.1f}°  "
          f"±{cfg.arm_ns.arm_angle_uncertainty_deg:.1f}°    "
          f"EW arm nominal: {cfg.arm_ew.arm_angle_nominal_deg:.1f}°  "
          f"±{cfg.arm_ew.arm_angle_uncertainty_deg:.1f}°\n")

    col = "{:>8} {:>10} {:>10} {:>12} {:>12} {:>14} {:>14}"
    print("  " + col.format(
        "Offset", "NS arm°", "EW arm°",
        "NS dV cmd", "EW dV cmd",
        "Xe total(kg)", "ΔXe vs nom(kg)"
    ))
    print(f"  {'─' * 82}")

    nom_xe = next(r["xe_total_mission"] for r in sens_results
                  if abs(r["arm_offset_sigma"]) < 0.01)

    for r in sens_results:
        delta_xe = r["xe_total_mission"] - nom_xe
        sigma_label = f"{r['arm_offset_sigma']:+.2f}σ"
        print("  " + col.format(
            sigma_label,
            f"{r['ns_arm_angle_varied']:.2f}°",
            f"{r['ew_arm_angle_varied']:.2f}°",
            f"{r['ns_dv_commanded_year']:.2f} m/s",
            f"{r['ew_dv_commanded_year']:.2f} m/s",
            f"{r['xe_total_mission']:.3f}",
            f"{delta_xe:+.3f}"
        ))

    worst = max(sens_results, key=lambda r: r["xe_total_mission"])
    best  = min(sens_results, key=lambda r: r["xe_total_mission"])
    print(f"\n  Worst case Xe : {worst['xe_total_mission']:.3f} kg  "
          f"(NS {worst['ns_arm_angle_varied']:.2f}°, "
          f"EW {worst['ew_arm_angle_varied']:.2f}°)")
    print(f"  Best case Xe  : {best['xe_total_mission']:.3f} kg  "
          f"(NS {best['ns_arm_angle_varied']:.2f}°, "
          f"EW {best['ew_arm_angle_varied']:.2f}°)")
    print(f"  Xe uncertainty band : "
          f"{worst['xe_total_mission'] - best['xe_total_mission']:.3f} kg")


def print_mass_parametric(para_results: list):
    print_section("CLIENT MASS PARAMETRIC — EOL CASE")
    col = "{:>14} {:>14} {:>14} {:>16} {:>14} {:>14}"
    print("\n  " + col.format(
        "Client(kg)", "Stack(kg)",
        "NS dV(m/s/yr)", "EW dV(m/s/yr)",
        "Xe total(kg)", "RCS total(kg)"
    ))
    print(f"  {'─' * 88}")
    for r in para_results:
        tag = ""
        print("  " + col.format(
            f"{r['client_mass_kg']:.0f}{tag}",
            f"{r['stack_mass_kg']:.0f}",
            f"{r['ns_dv_commanded_year']:.2f}",
            f"{r['ew_dv_commanded_year']:.2f}",
            f"{r['xe_total_mission']:.3f}",
            f"{r['rcs_prop_total']:.3f}"
        ))


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(nominal_results: list, sens_results: list, para_results: list,
               platform_name: str, output_dir: str = ".") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"sk_budget_{platform_name}_{timestamp}.csv")

    all_results = []
    for r in nominal_results:
        r["result_type"] = "nominal"
        all_results.append(r)
    for r in sens_results:
        r["result_type"] = "arm_sensitivity"
        all_results.append(r)
    for r in para_results:
        r["result_type"] = "mass_parametric"
        all_results.append(r)

    if not all_results:
        return ""

    fieldnames = [k for k in all_results[0].keys()
                  if k not in ("ns_dv_source", "ew_dv_source", "com_warning")]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n  ✓  CSV exported: {filename}")
    return filename


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> SKConfig:
    with open(path) as f:
        c = json.load(f)

    return SKConfig(
        platform_name=c["platform_name"],
        mission_duration_years=c["mission_duration_years"],
        stack_config=StackConfig(**c["stack_config"]),
        disturbance=DisturbanceConfig(**c["disturbance"]),
        arm_ns=ThrusterArmConfig(**c["arm_ns"]),
        arm_ew=ThrusterArmConfig(**c["arm_ew"]),
        ep=EPSKConfig(**c["ep"]),
        rcs_acs=RCSACSConfig(**c["rcs_acs"])
    )


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN EXAMPLE — Platform A post-docking SK
# ─────────────────────────────────────────────────────────────────────────────

def build_example_platform_a() -> SKConfig:
    """
    Illustrative Platform A post-docking SK configuration.
    Replace all values with actual mission and hardware data.
    N/S and E/W manoeuvres are executed separately.
    GNC override values are commented out — remove comments to activate.
    """
    return SKConfig(
        platform_name="Platform_A_PostDocking_SK",
        mission_duration_years=5.0,

        stack_config=StackConfig(
            platform_mass_kg=1800.0,    # Platform at SK start (after EOR + rendezvous)
            client_mass_min_kg=500.0,   # Lightest expected client satellite
            client_mass_nominal_kg=1200.0,
            client_mass_max_kg=2500.0,  # Heaviest expected client satellite
            client_com_offset_m=0.3,    # 0.3 m CoM offset from docking axis
            notes="Client mass range covers full serviceable satellite class"
        ),

        disturbance=DisturbanceConfig(
            station_longitude_deg=13.0,     # Example: 13°E (between stable points)
            mission_epoch_year=2026.0,
            sk_deadband_deg=0.05,           # ±0.05° station box
            ns_inclination_correction=1.05, # Slightly above baseline (near lunar cycle peak)
            # ns_gnc_override_dv_per_year=52.5,  # Uncomment to use GNC value
            spacecraft_area_m2=35.0,        # Platform solo solar array area
            spacecraft_area_stack_m2=60.0,  # Stack area (platform + client appendages)
            srp_cr=1.3,
            # ew_gnc_override_dv_per_year=2.1,   # Uncomment to use GNC value
        ),

        # Thruster arm: N/S and E/W are separate manoeuvres, separate arm angles
        arm_ns=ThrusterArmConfig(
            axis="NS",
            arm_angle_nominal_deg=8.0,      # 8° offset from N/S thrust axis
            arm_angle_uncertainty_deg=1.5,  # ±1.5° arm positioning uncertainty
            notes="EP thruster arm angle measured from N/S manoeuvre axis"
        ),
        arm_ew=ThrusterArmConfig(
            axis="EW",
            arm_angle_nominal_deg=5.0,      # 5° offset from E/W thrust axis
            arm_angle_uncertainty_deg=1.5,
            notes="EP thruster arm angle measured from E/W manoeuvre axis"
        ),

        ep=EPSKConfig(
            isp_bol=1750.0,     # s — Hall thruster BOL (illustrative)
            isp_eol=1600.0,     # s — EOL degradation at low Xe tank pressure
            thrust_mn=180.0,    # mN
            duty_cycle=0.85,    # 85% — GEO eclipses short, high availability
            warmup_xenon_kg=0.002,
            warmup_sessions_per_year=73.0  # ~every 5 days
        ),

        rcs_acs=RCSACSConfig(
            isp_bol=310.0,
            isp_eol=295.0,
            mixture_ratio=7.0,          # N2O/C3H6
            acs_dv_per_year=3.0,        # m/s/year — ACS during SK (from GNC sim)
            margin_performance=0.03,
            margin_execution=0.05,
            margin_operational=0.05
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_sk_tool(cfg: SKConfig, run_sensitivity: bool = True, output_dir: str = "."):
    print_header(f"STATION KEEPING BUDGET TOOL — {cfg.platform_name}")
    print(f"\n  Propellant config : EP = Xenon | RCS = C3H6/N2O (REACH compliant)")
    print(f"  Ignition system   : Spark / torch (no catalyst bed preheat model)")
    print(f"  SK manoeuvres     : N/S and E/W executed SEPARATELY")
    print(f"  Station longitude : {cfg.disturbance.station_longitude_deg}°E")
    print(f"  SK duration       : {cfg.mission_duration_years} years")

    sc = cfg.stack_config
    nominal_results = []

    for case in ("BOL", "EOL"):
        for label, mass in [
            ("min",     sc.client_mass_min_kg),
            ("nominal", sc.client_mass_nominal_kg),
            ("max",     sc.client_mass_max_kg),
        ]:
            r = compute_sk_budget(cfg, mass, case)
            nominal_results.append(r)
            if case == "EOL" and label == "nominal":
                print_header(f"NOMINAL BUDGET — EOL, client mass {mass:.0f} kg")
                print_sk_summary(r, cfg)

    # BOL vs EOL comparison at nominal mass
    bol_nom = next(r for r in nominal_results
                   if r["case"] == "BOL" and r["client_mass_kg"] == sc.client_mass_nominal_kg)
    eol_nom = next(r for r in nominal_results
                   if r["case"] == "EOL" and r["client_mass_kg"] == sc.client_mass_nominal_kg)

    print_section("BOL vs EOL — NOMINAL CLIENT MASS")
    col = "{:<35} {:>12} {:>12} {:>12}"
    print("\n  " + col.format("Item", "BOL", "EOL", "Delta"))
    print(f"  {'─' * 72}")
    for label, key in [
        ("NS commanded dV (m/s/yr)",  "ns_dv_commanded_year"),
        ("EW commanded dV (m/s/yr)",  "ew_dv_commanded_year"),
        ("Xe total mission (kg)",     "xe_total_mission"),
        ("RCS ACS total (kg)",        "rcs_prop_total"),
    ]:
        bv = bol_nom[key]
        ev = eol_nom[key]
        print("  " + col.format(label, f"{bv:.3f}", f"{ev:.3f}", f"{ev-bv:+.3f}"))

    # Sensitivity and parametric
    sens_results = []
    para_results = []

    if run_sensitivity:
        sens_results = arm_angle_sensitivity(cfg, sc.client_mass_nominal_kg, case="EOL")
        print_arm_sensitivity(sens_results, cfg)

        para_results = client_mass_parametric(cfg, case="EOL")
        print_mass_parametric(para_results)

    # Export
    export_csv(nominal_results, sens_results, para_results,
               cfg.platform_name, output_dir)

    print_header("TOOL 2 COMPLETE — Feed xe_total_mission and rcs_prop_total into Tool 1 SK phase")
    print(f"\n  EOL nominal Xe   SK budget : {eol_nom['xe_total_mission']:.3f} kg  "
          f"→ use as EP xenon delta-V input in Tool 1 Post_Docking_SK phase")
    print(f"  EOL nominal RCS  SK budget : {eol_nom['rcs_prop_total']:.3f} kg  "
          f"→ split N2O/C3H6 for Tool 1 ACS budget")
    print(f"  EOL worst-case Xe (max client mass) : "
          f"{next(r for r in nominal_results if r['case']=='EOL' and r['client_mass_kg']==sc.client_mass_max_kg)['xe_total_mission']:.3f} kg\n")


def main():
    parser = argparse.ArgumentParser(
        description="Station Keeping Budget Tool — EP (Xe) + RCS (C3H6/N2O)"
    )
    parser.add_argument("--config",      type=str,  help="Path to JSON SK config file")
    parser.add_argument("--example",     action="store_true", help="Run built-in Platform A example")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run arm angle sensitivity and mass parametric analyses")
    parser.add_argument("--output",      type=str,  default=".", help="Output directory for CSV")
    args = parser.parse_args()

    if args.example:
        cfg = build_example_platform_a()
        run_sk_tool(cfg, run_sensitivity=True, output_dir=args.output)
    elif args.config:
        cfg = load_config(args.config)
        run_sk_tool(cfg, run_sensitivity=args.sensitivity, output_dir=args.output)
    else:
        parser.print_help()
        print("\n  Tip: run with --example --sensitivity to see the full Platform A SK budget.")


if __name__ == "__main__":
    main()
