"""
================================================================================
COLLISION AVOIDANCE MANOEUVRE (CAM) RESERVE SIZING TOOL — Tool 4
Propulsion Engineering | GNC/AOCS Interface Tool
================================================================================
Platforms : Platform A (solo, pre-docking phases)
            Platform B (flyby, SSA, station keeping phases)
RCS       : C3H6 / N2O bipropellant — spark/torch ignition (REACH compliant)
            CAM is always an RCS manoeuvre — EP not used for CAMs (thrust too low)

Scope:
  - Solo platform configurations ONLY
  - Post-docking stack CAM is handled as a dedicated phase in Tool 1

CAM delta-V:
  - Analytical estimate from linear conjunction theory (warning time + geometry)
  - User-supplied override from conjunction analysis (replaces analytical)
  - Both in-plane and out-of-plane geometries computed separately
  - Worst case of the two drives the reserve

Degraded platform state:
  - Nominal thruster configuration
  - Single thruster failure case (reduced effective thrust)
  Both computed side by side

Execution timeline:
  - Warm-up time (spark ignition — no catalyst bed)
  - Attitude slew time to CAM thrust direction
  - Burn duration at nominal and degraded thrust
  - Total time vs. conjunction warning time — PASS/FAIL

Reserve protection:
  - CAM reserve sized at worst case (EOL Isp, max platform mass, degraded state)
  - Verified against remaining propellant at each mission phase gate
  - Flags erosion risk if nominal mission consumption approaches reserve

Outputs:
  - CAM delta-V per geometry and phase
  - Propellant reserve (N2O + C3H6) per platform configuration
  - Execution timeline breakdown — nominal and degraded
  - Reserve protection status at each phase gate
  - CSV export
  - Patch into Tool 1 CAM_Reserve phase config

Usage:
    python cam_reserve_sizing.py --config cam_config.json
    python cam_reserve_sizing.py --example
    python cam_reserve_sizing.py --example --patch-tool1 platform_a_config.json
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
G0          = 9.80665       # m/s²
MU_EARTH    = 3.986004e14   # m³/s² Earth gravitational parameter
R_EARTH     = 6371000.0     # m Earth mean radius
R_GEO       = 42164200.0    # m GEO radius

# Minimum CAM delta-V floor — below this, no manoeuvre is operationally meaningful
CAM_DV_MIN_MS = 0.5         # m/s

# Probability of collision threshold driving CAM go/no-go (industry standard)
PC_THRESHOLD = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrbitalConfig:
    """Orbital parameters for CAM delta-V analytical computation."""
    orbit_type: str             # "GEO", "LEO", "MEO"
    altitude_km: float          # Orbit altitude (km) above Earth surface
    inclination_deg: float      # Orbit inclination (degrees)


@dataclass
class ConjunctionConfig:
    """
    Conjunction scenario parameters for CAM delta-V sizing.
    Analytical model uses warning time and combined covariance to estimate
    required delta-V to achieve a safe miss distance.
    Override replaces the analytical result entirely.
    """
    warning_time_hours: float        # Time from CAM decision to TCA (hours)
    required_miss_distance_km: float # Target miss distance to be achieved (km)
    combined_covariance_1sigma_km: float  # Combined position uncertainty 1-sigma (km)

    # In-plane geometry (radial + along-track)
    inplane_approach_angle_deg: float    # Angle of approach in orbital plane (deg)

    # Out-of-plane geometry (cross-track)
    outofplane_approach_angle_deg: float  # Cross-track angle of approach (deg)

    # GNC override — if set, bypasses analytical for that geometry
    inplane_dv_override_ms: Optional[float] = None
    outofplane_dv_override_ms: Optional[float] = None


@dataclass
class ThrusterConfig:
    """RCS thruster configuration for CAM execution."""
    num_thrusters_nominal: int       # Number of thrusters firing in nominal CAM
    thrust_per_thruster_n: float     # Thrust per thruster (N)
    isp_bol_s: float                 # BOL Isp (s) — from Tool 3
    isp_eol_s: float                 # EOL Isp (s) — from Tool 3
    mixture_ratio: float             # N2O/C3H6
    min_on_time_ms: float            # Minimum thruster on-time (ms)
    # Degraded state: one thruster from firing group has failed
    num_thrusters_degraded: int      # Thrusters available after one failure


@dataclass
class ExecutionTimeline:
    """
    Time budget from CAM go-command to end of burn.
    Spark/torch ignition — no catalyst bed preheat.
    """
    uplink_processing_s: float       # Onboard processing of uplink command (s)
    attitude_slew_rate_deg_s: float  # Spacecraft slew rate (deg/s)
    max_slew_angle_deg: float        # Worst-case slew angle to CAM thrust direction (deg)
    ignition_delay_s: float          # Spark ignition delay (s) — no preheat needed
    margin_s: float                  # Timeline margin (s)


@dataclass
class PlatformConfig:
    """Single platform configuration for CAM reserve sizing."""
    name: str
    phase_name: str                  # Mission phase this config applies to
    mass_kg: float                   # Platform mass at this phase (kg)
    orbit: OrbitalConfig
    conjunction: ConjunctionConfig
    thrusters: ThrusterConfig
    timeline: ExecutionTimeline
    # Remaining RCS propellant at start of this phase (from Tool 1 output)
    remaining_n2o_kg: float
    remaining_c3h6_kg: float
    notes: str = ""


@dataclass
class CAMConfig:
    """Top-level CAM reserve sizing configuration."""
    platform_configurations: list    # List of PlatformConfig
    # Margin policy for CAM reserve
    margin_performance: float        # Isp uncertainty margin
    margin_execution: float          # Delta-V execution error margin (CAMs higher than nominal)
    margin_contingency: float        # Additional contingency for emergency scenario


# ─────────────────────────────────────────────────────────────────────────────
# ORBITAL MECHANICS UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def orbital_velocity(altitude_km: float) -> float:
    """Circular orbit velocity at given altitude (m/s)."""
    r = R_EARTH + altitude_km * 1000.0
    return math.sqrt(MU_EARTH / r)


def orbital_period(altitude_km: float) -> float:
    """Orbital period at given altitude (seconds)."""
    r = R_EARTH + altitude_km * 1000.0
    return 2.0 * math.pi * math.sqrt(r**3 / MU_EARTH)


# ─────────────────────────────────────────────────────────────────────────────
# CAM DELTA-V ANALYTICAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def cam_dv_analytical(orbit: OrbitalConfig, conj: ConjunctionConfig,
                       geometry: str) -> dict:
    """
    Analytical CAM delta-V estimation using linear conjunction theory.

    Physical model:
    ---------------
    For a conjunction with time-to-closest-approach (TCA) = t_warn,
    a delta-V manoeuvre applied now displaces the spacecraft from its
    nominal position at TCA. The displacement grows approximately linearly
    with warning time for small manoeuvres:

      For in-plane (radial manoeuvre):
        Δr_radial   ≈ dV * t_warn          (radial displacement at TCA)
        Δr_alongtrack ≈ 2 * dV * t_warn    (along-track drift, Clohessy-Wiltshire)

      For out-of-plane (normal manoeuvre):
        Δr_normal   ≈ dV * t_warn          (cross-track displacement)

    The required dV to achieve miss_distance D along the approach direction:

      In-plane (conservative — along-track component dominates at long warning):
        dV_inplane ≈ D / (2 * t_warn * cos(approach_angle))

      Out-of-plane:
        dV_outofplane ≈ D / (t_warn * cos(approach_angle))

    Minimum floor: max(dV_computed, CAM_DV_MIN_MS)

    Note: This is a simplified linear model suitable for early mission planning
    and budget sizing. High-fidelity conjunction screening uses numerical
    propagation with full covariance. The GNC override replaces this entirely
    when conjunction analysis results are available.

    geometry: "inplane" or "outofplane"
    """
    t_warn_s = conj.warning_time_hours * 3600.0
    D = conj.required_miss_distance_km * 1000.0  # m

    if geometry == "inplane":
        approach_angle = conj.inplane_approach_angle_deg
        cos_a = abs(math.cos(math.radians(approach_angle)))
        cos_a = max(cos_a, 0.1)  # floor to avoid divide-by-zero at 90 deg
        # Along-track dominates at warning times > ~1 orbit period
        T = orbital_period(orbit.altitude_km)
        if t_warn_s > T / 4:
            # Along-track: factor of 2 from C-W equations
            dv = D / (2.0 * t_warn_s * cos_a)
        else:
            # Short warning: radial displacement dominates
            dv = D / (t_warn_s * cos_a)
        geometry_label = "In-plane (radial/along-track)"

    else:  # outofplane
        approach_angle = conj.outofplane_approach_angle_deg
        cos_a = abs(math.cos(math.radians(approach_angle)))
        cos_a = max(cos_a, 0.1)
        dv = D / (t_warn_s * cos_a)
        geometry_label = "Out-of-plane (cross-track)"

    dv = max(dv, CAM_DV_MIN_MS)

    # Uncertainty band: ±1 covariance sigma scales required miss distance
    sigma = conj.combined_covariance_1sigma_km * 1000.0
    dv_3sigma = max((D + 3.0 * sigma) / (t_warn_s * max(cos_a, 0.1)), CAM_DV_MIN_MS)

    return {
        "geometry": geometry_label,
        "warning_time_h": conj.warning_time_hours,
        "required_miss_distance_km": conj.required_miss_distance_km,
        "approach_angle_deg": approach_angle,
        "dv_nominal_ms": dv,
        "dv_3sigma_ms": dv_3sigma,
        "source": "Analytical (linear conjunction theory)"
    }


def cam_dv_for_platform(platform: PlatformConfig) -> dict:
    """
    Compute CAM delta-V for both geometries, applying GNC overrides where set.
    Returns worst-case delta-V driving the reserve.
    """
    conj  = platform.conjunction
    orbit = platform.orbit

    # In-plane
    if conj.inplane_dv_override_ms is not None:
        ip = {
            "geometry": "In-plane (radial/along-track)",
            "warning_time_h": conj.warning_time_hours,
            "required_miss_distance_km": conj.required_miss_distance_km,
            "approach_angle_deg": conj.inplane_approach_angle_deg,
            "dv_nominal_ms": conj.inplane_dv_override_ms,
            "dv_3sigma_ms": conj.inplane_dv_override_ms,
            "source": "GNC conjunction analysis override"
        }
    else:
        ip = cam_dv_analytical(orbit, conj, "inplane")

    # Out-of-plane
    if conj.outofplane_dv_override_ms is not None:
        oop = {
            "geometry": "Out-of-plane (cross-track)",
            "warning_time_h": conj.warning_time_hours,
            "required_miss_distance_km": conj.required_miss_distance_km,
            "approach_angle_deg": conj.outofplane_approach_angle_deg,
            "dv_nominal_ms": conj.outofplane_dv_override_ms,
            "dv_3sigma_ms": conj.outofplane_dv_override_ms,
            "source": "GNC conjunction analysis override"
        }
    else:
        oop = cam_dv_analytical(orbit, conj, "outofplane")

    # Worst case drives the reserve — use 3-sigma value for conservative sizing
    worst_dv = max(ip["dv_3sigma_ms"], oop["dv_3sigma_ms"])
    worst_geometry = "In-plane" if ip["dv_3sigma_ms"] >= oop["dv_3sigma_ms"] \
                     else "Out-of-plane"

    return {
        "inplane": ip,
        "outofplane": oop,
        "worst_case_dv_ms": worst_dv,
        "worst_case_geometry": worst_geometry
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION TIMELINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_execution_timeline(platform: PlatformConfig,
                                 cam_dv_ms: float, case: str) -> dict:
    """
    Compute full execution timeline from CAM go-command to end of burn.
    case: "nominal" or "degraded"

    Spark/torch ignition — no catalyst bed preheat required.
    Ignition delay is short (< 1s typically).
    """
    thr  = platform.thrusters
    tl   = platform.timeline

    n_thrusters = (thr.num_thrusters_nominal if case == "nominal"
                   else thr.num_thrusters_degraded)
    total_thrust = n_thrusters * thr.thrust_per_thruster_n
    isp = thr.isp_eol_s  # Always use EOL for conservative timeline

    # Burn duration (constant thrust approximation for short CAM burns)
    if total_thrust > 0 and cam_dv_ms > 0:
        # dm/dt = F / (Isp * g0)
        mdot = total_thrust / (isp * G0)
        # Exact: t_burn = (m/mdot) * (1 - exp(-dv/(isp*g0)))
        t_burn_s = (platform.mass_kg / mdot) * \
                   (1.0 - math.exp(-cam_dv_ms / (isp * G0)))
    else:
        t_burn_s = float('inf')

    # Attitude slew time
    t_slew_s = tl.max_slew_angle_deg / tl.attitude_slew_rate_deg_s

    # Total timeline
    t_total_s = (tl.uplink_processing_s +
                 t_slew_s +
                 tl.ignition_delay_s +
                 t_burn_s +
                 tl.margin_s)

    t_available_s = platform.conjunction.warning_time_hours * 3600.0

    # Check minimum impulse bit feasibility
    min_impulse_dv = (thr.min_on_time_ms * 1e-3 * total_thrust) / \
                     (platform.mass_kg) if total_thrust > 0 else 0.0

    timeline_ok = t_total_s <= t_available_s
    mib_ok = cam_dv_ms >= min_impulse_dv

    return {
        "case": case,
        "n_thrusters": n_thrusters,
        "total_thrust_n": total_thrust,
        "isp_used_s": isp,
        "t_uplink_s": tl.uplink_processing_s,
        "t_slew_s": t_slew_s,
        "t_ignition_s": tl.ignition_delay_s,
        "t_burn_s": t_burn_s,
        "t_margin_s": tl.margin_s,
        "t_total_s": t_total_s,
        "t_available_s": t_available_s,
        "t_remaining_after_cam_s": t_available_s - t_total_s,
        "timeline_ok": timeline_ok,
        "min_impulse_dv_ms": min_impulse_dv,
        "mib_ok": mib_ok
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROPELLANT RESERVE SIZING
# ─────────────────────────────────────────────────────────────────────────────

def margin_factor(cfg: CAMConfig) -> float:
    """Combined margin multiplier for CAM reserve."""
    return ((1 + cfg.margin_performance) *
            (1 + cfg.margin_execution) *
            (1 + cfg.margin_contingency))


def tsiolkovsky_prop(dv_ms: float, isp_s: float, mass_kg: float) -> float:
    """Propellant mass from Tsiolkovsky equation."""
    if dv_ms <= 0:
        return 0.0
    return mass_kg * (1.0 - math.exp(-dv_ms / (isp_s * G0)))


def split_biprop(total_kg: float, mr: float) -> tuple:
    """Split total into (N2O_kg, C3H6_kg) at mixture ratio mr = N2O/C3H6."""
    c3h6 = total_kg / (1.0 + mr)
    n2o  = total_kg - c3h6
    return n2o, c3h6


def size_cam_reserve(platform: PlatformConfig, cam_dv: dict,
                      cfg: CAMConfig) -> dict:
    """
    Size the CAM propellant reserve for a platform configuration.
    Uses EOL Isp (conservative) and worst-case CAM delta-V.
    Computes nominal and degraded cases side by side.
    """
    thr     = platform.thrusters
    worst_dv = cam_dv["worst_case_dv_ms"]
    mf      = margin_factor(cfg)
    mass    = platform.mass_kg

    results = {}

    for case, isp in [("nominal",  thr.isp_eol_s),
                       ("degraded", thr.isp_eol_s)]:
        # Degraded case: same Isp (same propellant), but reduced thrust
        # means longer burn — propellant mass is the same for same dv+mass
        # The thrust reduction matters for timeline, not propellant mass
        prop_nominal = tsiolkovsky_prop(worst_dv, isp, mass)
        prop_with_margin = prop_nominal * mf
        prop_margin = prop_with_margin - prop_nominal
        n2o, c3h6 = split_biprop(prop_with_margin, thr.mixture_ratio)

        # Also compute for in-plane and out-of-plane individually
        ip_dv  = cam_dv["inplane"]["dv_3sigma_ms"]
        oop_dv = cam_dv["outofplane"]["dv_3sigma_ms"]
        prop_ip  = tsiolkovsky_prop(ip_dv,  isp, mass) * mf
        prop_oop = tsiolkovsky_prop(oop_dv, isp, mass) * mf

        results[case] = {
            "isp_s": isp,
            "worst_dv_ms": worst_dv,
            "prop_nominal_kg": prop_nominal,
            "prop_margin_kg": prop_margin,
            "prop_total_kg": prop_with_margin,
            "n2o_kg": n2o,
            "c3h6_kg": c3h6,
            "prop_inplane_kg": prop_ip,
            "prop_outofplane_kg": prop_oop,
            "margin_performance": cfg.margin_performance,
            "margin_execution": cfg.margin_execution,
            "margin_contingency": cfg.margin_contingency,
            "margin_total_pct": (mf - 1.0) * 100.0
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RESERVE PROTECTION CHECK
# ─────────────────────────────────────────────────────────────────────────────

def reserve_protection_check(platform: PlatformConfig,
                               reserve: dict) -> dict:
    """
    Verify remaining RCS propellant at this phase gate exceeds the CAM reserve.
    Uses degraded case reserve (worst case) for the check.
    """
    required_n2o  = reserve["degraded"]["n2o_kg"]
    required_c3h6 = reserve["degraded"]["c3h6_kg"]
    required_total = reserve["degraded"]["prop_total_kg"]

    available_n2o  = platform.remaining_n2o_kg
    available_c3h6 = platform.remaining_c3h6_kg
    available_total = available_n2o + available_c3h6

    margin_n2o   = available_n2o   - required_n2o
    margin_c3h6  = available_c3h6  - required_c3h6
    margin_total = available_total - required_total

    n2o_ok   = margin_n2o   >= 0
    c3h6_ok  = margin_c3h6  >= 0
    total_ok = margin_total >= 0

    # Erosion warning: if remaining propellant is within 20% of reserve
    erosion_threshold = 0.20
    n2o_erosion   = available_n2o   < required_n2o   * (1 + erosion_threshold)
    c3h6_erosion  = available_c3h6  < required_c3h6  * (1 + erosion_threshold)

    return {
        "phase": platform.phase_name,
        "required_n2o_kg":    required_n2o,
        "required_c3h6_kg":   required_c3h6,
        "required_total_kg":  required_total,
        "available_n2o_kg":   available_n2o,
        "available_c3h6_kg":  available_c3h6,
        "available_total_kg": available_total,
        "margin_n2o_kg":      margin_n2o,
        "margin_c3h6_kg":     margin_c3h6,
        "margin_total_kg":    margin_total,
        "n2o_protected":      n2o_ok,
        "c3h6_protected":     c3h6_ok,
        "reserve_protected":  total_ok,
        "n2o_erosion_warning":   n2o_erosion  and n2o_ok,
        "c3h6_erosion_warning":  c3h6_erosion and c3h6_ok,
    }


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


def print_cam_dv(cam_dv: dict, platform: PlatformConfig):
    print_section(f"CAM DELTA-V SIZING — {platform.name} | {platform.phase_name}")
    print(f"\n  Warning time    : {platform.conjunction.warning_time_hours:.1f} h")
    print(f"  Required miss   : {platform.conjunction.required_miss_distance_km:.1f} km")
    print(f"  Covariance 1σ   : {platform.conjunction.combined_covariance_1sigma_km:.2f} km")

    for key, label in [("inplane", "IN-PLANE"), ("outofplane", "OUT-OF-PLANE")]:
        r = cam_dv[key]
        src_tag = "  [GNC OVERRIDE]" if "override" in r["source"].lower() else \
                  "  [analytical]"
        print(f"\n  {label} {src_tag}")
        print(f"    Approach angle  : {r['approach_angle_deg']:.1f}°")
        print(f"    dV nominal      : {r['dv_nominal_ms']:.3f} m/s")
        print(f"    dV 3-sigma      : {r['dv_3sigma_ms']:.3f} m/s  ← used for reserve sizing")

    print(f"\n  ► Worst case     : {cam_dv['worst_case_geometry']}  "
          f"→  {cam_dv['worst_case_dv_ms']:.3f} m/s  (3-sigma, drives reserve)")


def print_timeline(tl_nom: dict, tl_deg: dict, platform: PlatformConfig):
    print_section(f"EXECUTION TIMELINE — {platform.name} | {platform.phase_name}")
    print(f"\n  Ignition type   : Spark / torch  (no catalyst bed — no preheat delay)")
    print(f"  Warning time    : {platform.conjunction.warning_time_hours:.1f} h  "
          f"= {platform.conjunction.warning_time_hours*3600:.0f} s available\n")

    col = "{:<32} {:>14} {:>14}"
    print("  " + col.format("Timeline Element", "Nominal", "Degraded"))
    print(f"  {'─' * 62}")

    elements = [
        ("Thrusters firing",          "n_thrusters",     ""),
        ("Total thrust (N)",           "total_thrust_n",  ".1f"),
        ("Uplink processing (s)",      "t_uplink_s",      ".1f"),
        ("Attitude slew (s)",          "t_slew_s",        ".1f"),
        ("Ignition delay (s)",         "t_ignition_s",    ".2f"),
        ("Burn duration (s)",          "t_burn_s",        ".2f"),
        ("Timeline margin (s)",        "t_margin_s",      ".1f"),
        ("TOTAL time to CAM end (s)",  "t_total_s",       ".2f"),
        ("Time available (s)",         "t_available_s",   ".0f"),
        ("Remaining after CAM (s)",    "t_remaining_after_cam_s", ".2f"),
    ]

    for label, key, fmt in elements:
        vn = tl_nom[key]
        vd = tl_deg[key]
        fn = f"{vn:{fmt}}" if fmt else str(vn)
        fd = f"{vd:{fmt}}" if fmt else str(vd)
        print("  " + col.format(label, fn, fd))

    print(f"\n  Timeline check (nominal)  : "
          f"{'✓ PASS' if tl_nom['timeline_ok'] else '✗ FAIL — insufficient warning time'}")
    print(f"  Timeline check (degraded) : "
          f"{'✓ PASS' if tl_deg['timeline_ok'] else '✗ FAIL — insufficient warning time'}")
    print(f"  MIB check (nominal)       : "
          f"{'✓ PASS' if tl_nom['mib_ok'] else '✗ FAIL — dV below minimum impulse bit'}")
    print(f"  MIB check (degraded)      : "
          f"{'✓ PASS' if tl_deg['mib_ok'] else '✗ FAIL — dV below minimum impulse bit'}")

    if not tl_deg["timeline_ok"]:
        shortfall = tl_deg["t_total_s"] - tl_deg["t_available_s"]
        print(f"\n  ⚠  DEGRADED TIMELINE SHORTFALL: {shortfall:.1f} s")
        print(f"     Options: increase warning time threshold, reduce slew angle, "
              f"or accept single-thruster CAM infeasibility.")


def print_reserve(reserve: dict, platform: PlatformConfig):
    print_section(f"CAM PROPELLANT RESERVE — {platform.name} | {platform.phase_name}")
    print(f"\n  Platform mass   : {platform.mass_kg:.1f} kg")
    print(f"  Worst-case dV   : {reserve['nominal']['worst_dv_ms']:.3f} m/s")
    print(f"  Margins         : perf {reserve['nominal']['margin_performance']*100:.0f}%  "
          f"exec {reserve['nominal']['margin_execution']*100:.0f}%  "
          f"contingency {reserve['nominal']['margin_contingency']*100:.0f}%  "
          f"(combined +{reserve['nominal']['margin_total_pct']:.1f}%)\n")

    col = "{:<30} {:>14} {:>14}"
    print("  " + col.format("Reserve Item", "Nominal config", "Degraded config"))
    print(f"  {'─' * 60}")

    rows = [
        ("Isp used (s)",              "isp_s",           ".1f"),
        ("Prop nominal (kg)",         "prop_nominal_kg",  ".4f"),
        ("Prop margin (kg)",          "prop_margin_kg",   ".4f"),
        ("RESERVE TOTAL (kg)",        "prop_total_kg",    ".4f"),
        ("  of which N2O (kg)",       "n2o_kg",           ".4f"),
        ("  of which C3H6 (kg)",      "c3h6_kg",          ".4f"),
        ("In-plane only (kg)",        "prop_inplane_kg",  ".4f"),
        ("Out-of-plane only (kg)",    "prop_outofplane_kg", ".4f"),
    ]

    for label, key, fmt in rows:
        vn = reserve["nominal"][key]
        vd = reserve["degraded"][key]
        print("  " + col.format(label,
              f"{vn:{fmt}}", f"{vd:{fmt}}"))

    print(f"\n  ► Sizing reserve : degraded case  →  "
          f"N2O {reserve['degraded']['n2o_kg']:.4f} kg  |  "
          f"C3H6 {reserve['degraded']['c3h6_kg']:.4f} kg")


def print_protection_check(check: dict):
    def ok(v):  return "✓" if v else "✗"
    def warn(v): return "  ⚠  EROSION WARNING" if v else ""

    print_section(f"RESERVE PROTECTION CHECK — {check['phase']}")
    col = "{:<25} {:>12} {:>12} {:>12}  {}"
    print("\n  " + col.format("Propellant", "Required", "Available",
                               "Margin", "Status"))
    print(f"  {'─' * 72}")
    for key, label in [("n2o", "N2O (kg)"), ("c3h6", "C3H6 (kg)"),
                        ("total", "Total RCS (kg)")]:
        req  = check[f"required_{key}_kg"]
        avail = check[f"available_{key}_kg"]
        marg  = check[f"margin_{key}_kg"]
        prot  = check[f"{key}_protected"] if key != "total" else check["reserve_protected"]
        ew    = check.get(f"{key}_erosion_warning", False)
        print("  " + col.format(label,
              f"{req:.4f}", f"{avail:.4f}", f"{marg:+.4f}",
              f"{ok(prot)} {'PROTECTED' if prot else 'NOT PROTECTED'}{warn(ew)}"))


def print_summary_table(all_results: list):
    print_header("SUMMARY — CAM RESERVE ACROSS ALL PLATFORM CONFIGURATIONS")
    col = "{:<30} {:>10} {:>10} {:>12} {:>10} {:>10}  {}"
    print("\n  " + col.format(
        "Platform / Phase", "dV(m/s)", "Xe total", "N2O(kg)",
        "C3H6(kg)", "TL nom", "TL deg"
    ))
    print(f"  {'─' * 88}")
    for r in all_results:
        tl_n = "✓" if r["timeline_nom_ok"] else "✗"
        tl_d = "✓" if r["timeline_deg_ok"] else "✗"
        label = f"{r['platform_name']} / {r['phase_name']}"
        print("  " + col.format(
            label[:29],
            f"{r['worst_dv_ms']:.3f}",
            "N/A",
            f"{r['reserve_degraded_n2o_kg']:.4f}",
            f"{r['reserve_degraded_c3h6_kg']:.4f}",
            tl_n, tl_d
        ))

    # Governing reserve = max across all configurations
    max_n2o  = max(r["reserve_degraded_n2o_kg"]  for r in all_results)
    max_c3h6 = max(r["reserve_degraded_c3h6_kg"] for r in all_results)
    gov = max(all_results, key=lambda r: r["reserve_degraded_n2o_kg"] +
                                          r["reserve_degraded_c3h6_kg"])
    print(f"\n  ► Governing reserve (worst case across all phases):")
    print(f"    Configuration : {gov['platform_name']} / {gov['phase_name']}")
    print(f"    N2O           : {max_n2o:.4f} kg")
    print(f"    C3H6          : {max_c3h6:.4f} kg")
    print(f"    Total RCS     : {max_n2o + max_c3h6:.4f} kg")
    print(f"\n  This is the reserve to protect in Tool 1 CAM_Reserve phase.")

    return {"n2o_kg": max_n2o, "c3h6_kg": max_c3h6,
            "total_kg": max_n2o + max_c3h6,
            "governing_phase": gov["phase_name"],
            "governing_dv_ms": gov["worst_dv_ms"]}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 CONFIG PATCH
# ─────────────────────────────────────────────────────────────────────────────

def patch_tool1_cam(tool1_path: str, governing: dict,
                     output_dir: str = ".") -> str:
    """
    Patch the CAM_Reserve phase in a Tool 1 config JSON with the
    governing CAM delta-V and propellant reserve from Tool 4.

    Updates the CAM_Reserve phase delta_v field with the governing dV.
    Writes to <original>_cam_patched.json.
    """
    with open(tool1_path) as f:
        cfg = json.load(f)

    patched  = copy.deepcopy(cfg)
    patch_log = []

    for phase in patched.get("phases", []):
        if "CAM" in phase.get("name", "").upper():
            old_dv = phase.get("delta_v")
            phase["delta_v"] = round(governing["governing_dv_ms"], 3)
            patch_log.append(
                f"  Phase '{phase['name']}': delta_v {old_dv} → "
                f"{governing['governing_dv_ms']:.3f} m/s  "
                f"(governing: {governing['governing_phase']})"
            )

    base = os.path.splitext(os.path.basename(tool1_path))[0]
    out_path = os.path.join(output_dir, f"{base}_cam_patched.json")
    with open(out_path, "w") as f:
        json.dump(patched, f, indent=2)

    return out_path, patch_log


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(all_results: list, governing: dict,
               output_dir: str = ".") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"cam_reserve_{timestamp}.csv")

    if not all_results:
        return ""

    fieldnames = list(all_results[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    print(f"\n  ✓  CSV exported: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> CAMConfig:
    with open(path) as f:
        c = json.load(f)

    platforms = []
    for p in c["platform_configurations"]:
        platforms.append(PlatformConfig(
            name=p["name"],
            phase_name=p["phase_name"],
            mass_kg=p["mass_kg"],
            orbit=OrbitalConfig(**p["orbit"]),
            conjunction=ConjunctionConfig(**p["conjunction"]),
            thrusters=ThrusterConfig(**p["thrusters"]),
            timeline=ExecutionTimeline(**p["timeline"]),
            remaining_n2o_kg=p["remaining_n2o_kg"],
            remaining_c3h6_kg=p["remaining_c3h6_kg"],
            notes=p.get("notes", "")
        ))

    return CAMConfig(
        platform_configurations=platforms,
        margin_performance=c["margin_performance"],
        margin_execution=c["margin_execution"],
        margin_contingency=c["margin_contingency"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def build_example() -> CAMConfig:
    """
    Illustrative CAM configurations for Platform A (solo, pre-docking)
    and Platform B (flyby + SSA + SK).
    All values illustrative — replace with actual mission data.
    """

    # Shared RCS thruster spec (C3H6/N2O, spark ignition)
    # Isp values from Tool 3 output
    rcs_thr = dict(
        num_thrusters_nominal=4,
        thrust_per_thruster_n=22.0,      # N per thruster (illustrative)
        isp_bol_s=313.0,                 # From Tool 3
        isp_eol_s=280.6,                 # From Tool 3
        mixture_ratio=7.0,
        min_on_time_ms=20.0,             # 20 ms minimum pulse
        num_thrusters_degraded=3         # One thruster failed
    )

    # Shared timeline (spark ignition — no preheat)
    tl = dict(
        uplink_processing_s=10.0,
        attitude_slew_rate_deg_s=0.5,    # deg/s spacecraft slew capability
        max_slew_angle_deg=45.0,         # Worst-case slew to CAM thrust direction
        ignition_delay_s=0.5,            # Spark ignition delay — very short
        margin_s=60.0                    # 60 s operational margin
    )

    platforms = [
        # ── Platform A — EOR phase (heaviest, early mission) ─────────────────
        PlatformConfig(
            name="Platform_A",
            phase_name="EOR",
            mass_kg=3500.0,              # Initial wet mass
            orbit=OrbitalConfig(
                orbit_type="GEO_transfer",
                altitude_km=3000.0,      # Mid-transfer orbit (conservative)
                inclination_deg=7.0
            ),
            conjunction=ConjunctionConfig(
                warning_time_hours=4.0,
                required_miss_distance_km=5.0,
                combined_covariance_1sigma_km=0.5,
                inplane_approach_angle_deg=30.0,
                outofplane_approach_angle_deg=60.0,
                # inplane_dv_override_ms=2.5,   # Uncomment to use GNC value
                # outofplane_dv_override_ms=1.8
            ),
            thrusters=ThrusterConfig(**rcs_thr),
            timeline=ExecutionTimeline(**tl),
            remaining_n2o_kg=280.0,      # Full tanks at EOR start
            remaining_c3h6_kg=40.0,
            notes="EOR phase — heaviest platform mass, GTO/transfer orbit."
        ),

        # ── Platform A — Far Range Rendezvous ─────────────────────────────────
        PlatformConfig(
            name="Platform_A",
            phase_name="Far_Range_Rendezvous",
            mass_kg=3150.0,              # After EOR propellant consumed
            orbit=OrbitalConfig(
                orbit_type="GEO",
                altitude_km=35786.0,
                inclination_deg=0.1
            ),
            conjunction=ConjunctionConfig(
                warning_time_hours=8.0,   # Longer warning in GEO (slower relative motion)
                required_miss_distance_km=5.0,
                combined_covariance_1sigma_km=0.3,
                inplane_approach_angle_deg=45.0,
                outofplane_approach_angle_deg=45.0
            ),
            thrusters=ThrusterConfig(**rcs_thr),
            timeline=ExecutionTimeline(**tl),
            remaining_n2o_kg=245.0,
            remaining_c3h6_kg=35.0,
            notes="GEO phase — longer warning time. Target approach ongoing."
        ),

        # ── Platform A — Near Range Rendezvous ───────────────────────────────
        PlatformConfig(
            name="Platform_A",
            phase_name="Near_Range_Rendezvous",
            mass_kg=3130.0,
            orbit=OrbitalConfig(
                orbit_type="GEO",
                altitude_km=35786.0,
                inclination_deg=0.1
            ),
            conjunction=ConjunctionConfig(
                warning_time_hours=8.0,
                required_miss_distance_km=5.0,
                combined_covariance_1sigma_km=0.3,
                inplane_approach_angle_deg=45.0,
                outofplane_approach_angle_deg=45.0
            ),
            thrusters=ThrusterConfig(**rcs_thr),
            timeline=ExecutionTimeline(**tl),
            remaining_n2o_kg=237.0,
            remaining_c3h6_kg=33.9,
            notes="Near range — target satellite also in vicinity."
        ),

        # ── Platform B — Flyby ────────────────────────────────────────────────
        PlatformConfig(
            name="Platform_B",
            phase_name="Flyby",
            mass_kg=2800.0,
            orbit=OrbitalConfig(
                orbit_type="GEO",
                altitude_km=35786.0,
                inclination_deg=0.05
            ),
            conjunction=ConjunctionConfig(
                warning_time_hours=6.0,
                required_miss_distance_km=5.0,
                combined_covariance_1sigma_km=0.4,
                inplane_approach_angle_deg=30.0,
                outofplane_approach_angle_deg=60.0
            ),
            thrusters=ThrusterConfig(**rcs_thr),
            timeline=ExecutionTimeline(**tl),
            remaining_n2o_kg=180.0,
            remaining_c3h6_kg=25.7,
            notes="Flyby proximity — both Platform B and target in close vicinity."
        ),

        # ── Platform B — SSA ─────────────────────────────────────────────────
        PlatformConfig(
            name="Platform_B",
            phase_name="SSA",
            mass_kg=2750.0,
            orbit=OrbitalConfig(
                orbit_type="GEO",
                altitude_km=35786.0,
                inclination_deg=0.05
            ),
            conjunction=ConjunctionConfig(
                warning_time_hours=8.0,
                required_miss_distance_km=5.0,
                combined_covariance_1sigma_km=0.3,
                inplane_approach_angle_deg=45.0,
                outofplane_approach_angle_deg=45.0
            ),
            thrusters=ThrusterConfig(**rcs_thr),
            timeline=ExecutionTimeline(**tl),
            remaining_n2o_kg=172.0,
            remaining_c3h6_kg=24.6,
            notes="SSA operations — fixed relative position maintenance."
        ),

        # ── Platform B — Station Keeping ─────────────────────────────────────
        PlatformConfig(
            name="Platform_B",
            phase_name="Station_Keeping",
            mass_kg=2500.0,             # After years of SK propellant consumption
            orbit=OrbitalConfig(
                orbit_type="GEO",
                altitude_km=35786.0,
                inclination_deg=0.05
            ),
            conjunction=ConjunctionConfig(
                warning_time_hours=8.0,
                required_miss_distance_km=5.0,
                combined_covariance_1sigma_km=0.3,
                inplane_approach_angle_deg=45.0,
                outofplane_approach_angle_deg=45.0
            ),
            thrusters=ThrusterConfig(**rcs_thr),
            timeline=ExecutionTimeline(**tl),
            remaining_n2o_kg=85.0,      # Late mission — propellant depleted
            remaining_c3h6_kg=12.1,
            notes="EOL SK — lowest remaining propellant. Most critical protection check."
        ),
    ]

    return CAMConfig(
        platform_configurations=platforms,
        margin_performance=0.05,    # 5%
        margin_execution=0.10,      # 10% — higher for emergency manoeuvre
        margin_contingency=0.10     # 10% — additional contingency
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_cam_tool(cfg: CAMConfig, output_dir: str = ".",
                  patch_t1: str = None):

    print_header("COLLISION AVOIDANCE MANOEUVRE (CAM) RESERVE SIZING TOOL")
    print(f"\n  RCS propellant  : N2O / C3H6 (REACH compliant)")
    print(f"  Ignition        : Spark / torch — no catalyst bed — no preheat delay")
    print(f"  EP not used for CAMs — thrust level too low for time-critical manoeuvres")
    print(f"  Scope           : Solo platforms only (post-docking stack in Tool 1)")
    print(f"\n  Margin policy   : performance {cfg.margin_performance*100:.0f}%  "
          f"execution {cfg.margin_execution*100:.0f}%  "
          f"contingency {cfg.margin_contingency*100:.0f}%")

    all_results = []

    for platform in cfg.platform_configurations:
        print_header(f"{platform.name} | {platform.phase_name}")
        if platform.notes:
            print(f"  Note: {platform.notes}")

        # Delta-V sizing
        cam_dv = cam_dv_for_platform(platform)
        print_cam_dv(cam_dv, platform)

        # Execution timeline
        tl_nom = compute_execution_timeline(
            platform, cam_dv["worst_case_dv_ms"], "nominal")
        tl_deg = compute_execution_timeline(
            platform, cam_dv["worst_case_dv_ms"], "degraded")
        print_timeline(tl_nom, tl_deg, platform)

        # Reserve sizing
        reserve = size_cam_reserve(platform, cam_dv, cfg)
        print_reserve(reserve, platform)

        # Protection check
        check = reserve_protection_check(platform, reserve)
        print_protection_check(check)

        # Accumulate for summary
        all_results.append({
            "platform_name": platform.name,
            "phase_name": platform.phase_name,
            "platform_mass_kg": platform.mass_kg,
            "warning_time_h": platform.conjunction.warning_time_hours,
            "inplane_dv_3sigma_ms": cam_dv["inplane"]["dv_3sigma_ms"],
            "outofplane_dv_3sigma_ms": cam_dv["outofplane"]["dv_3sigma_ms"],
            "worst_dv_ms": cam_dv["worst_case_dv_ms"],
            "worst_geometry": cam_dv["worst_case_geometry"],
            "reserve_nominal_kg": reserve["nominal"]["prop_total_kg"],
            "reserve_nominal_n2o_kg": reserve["nominal"]["n2o_kg"],
            "reserve_nominal_c3h6_kg": reserve["nominal"]["c3h6_kg"],
            "reserve_degraded_kg": reserve["degraded"]["prop_total_kg"],
            "reserve_degraded_n2o_kg": reserve["degraded"]["n2o_kg"],
            "reserve_degraded_c3h6_kg": reserve["degraded"]["c3h6_kg"],
            "timeline_nom_ok": tl_nom["timeline_ok"],
            "timeline_deg_ok": tl_deg["timeline_ok"],
            "timeline_nom_total_s": tl_nom["t_total_s"],
            "timeline_deg_total_s": tl_deg["t_total_s"],
            "timeline_available_s": tl_nom["t_available_s"],
            "reserve_protected": check["reserve_protected"],
            "erosion_warning": check.get("n2o_erosion_warning", False) or
                               check.get("c3h6_erosion_warning", False),
            "remaining_n2o_kg": platform.remaining_n2o_kg,
            "remaining_c3h6_kg": platform.remaining_c3h6_kg,
        })

    # Summary across all platforms
    governing = print_summary_table(all_results)

    # CSV
    export_csv(all_results, governing, output_dir)

    # Patch Tool 1
    if patch_t1:
        print_section("PATCHING TOOL 1 CAM PHASE")
        out, log = patch_tool1_cam(patch_t1, governing, output_dir)
        for line in log:
            print(line)
        print(f"\n  ✓  Patched Tool 1 config written to: {out}")

    # Final verdict
    print_header("TOOL 4 COMPLETE — CAM RESERVE SIZING SUMMARY")
    all_protected = all(r["reserve_protected"] for r in all_results)
    all_tl_nom    = all(r["timeline_nom_ok"] for r in all_results)
    all_tl_deg    = all(r["timeline_deg_ok"] for r in all_results)
    any_erosion   = any(r["erosion_warning"] for r in all_results)

    print(f"\n  Governing CAM reserve : "
          f"N2O {governing['n2o_kg']:.4f} kg  |  "
          f"C3H6 {governing['c3h6_kg']:.4f} kg  |  "
          f"Total {governing['total_kg']:.4f} kg")
    print(f"  Governing phase       : {governing['governing_phase']}  "
          f"({governing['governing_dv_ms']:.3f} m/s)")
    print(f"\n  Reserve protected     : {'✓ ALL PHASES' if all_protected else '✗ CHECK FAILED PHASES'}")
    print(f"  Timeline (nominal)    : {'✓ ALL PHASES' if all_tl_nom else '✗ CHECK FAILED PHASES'}")
    print(f"  Timeline (degraded)   : {'✓ ALL PHASES' if all_tl_deg else '✗ CHECK FAILED PHASES'}")
    if any_erosion:
        print(f"  ⚠  Erosion warnings present — review late-mission propellant margins.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="CAM Reserve Sizing Tool — RCS C3H6/N2O, solo platforms"
    )
    parser.add_argument("--config",      type=str,  help="Path to CAM config JSON")
    parser.add_argument("--example",     action="store_true",
                        help="Run built-in example (Platform A + B)")
    parser.add_argument("--patch-tool1", type=str,  metavar="TOOL1_JSON",
                        help="Tool 1 config JSON to patch with governing CAM dV")
    parser.add_argument("--output",      type=str,  default=".",
                        help="Output directory for CSV and patched configs")
    args = parser.parse_args()

    if args.example:
        cfg = build_example()
        run_cam_tool(cfg, output_dir=args.output, patch_t1=args.patch_tool1)
    elif args.config:
        cfg = load_config(args.config)
        run_cam_tool(cfg, output_dir=args.output, patch_t1=args.patch_tool1)
    else:
        parser.print_help()
        print("\n  Tip: run with --example to see the full Platform A + B CAM budget.")
        print("       Add --patch-tool1 <file> to auto-update the Tool 1 CAM phase.")


if __name__ == "__main__":
    main()
