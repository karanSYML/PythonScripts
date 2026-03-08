"""
================================================================================
PROPELLANT BUDGET CALCULATOR — Tool 1
Propulsion Engineering | GNC/AOCS Interface Tool
================================================================================
Platforms : Platform A (EOR + Docking + SK) | Platform B (Flyby + SSA + SK)
RCS        : C3H6 / N2O bipropellant (REACH compliant — no hydrazine)
EP         : Xenon (Hall effect / ion thruster)
Ignition   : Spark / torch ignition (no catalyst bed preheat model)

Usage:
    python propellant_budget_calculator.py --config mission_config.json
    python propellant_budget_calculator.py --example   (runs built-in example)

Outputs:
    - Phase-by-phase propellant budget printed to console
    - CSV export of full budget table
    - Mixture ratio balance check
    - BOL / EOL bounding cases
    - Margin closure check per propellant type
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
G0 = 9.80665  # Standard gravity (m/s²)

PROPELLANT_REGISTRY = {
    "N2O":  {"full_name": "Nitrous Oxide",   "reach_compliant": True,  "role": "RCS oxidiser"},
    "C3H6": {"full_name": "Propylene",        "reach_compliant": True,  "role": "RCS fuel"},
    "Xe":   {"full_name": "Xenon",            "reach_compliant": True,  "role": "EP propellant"},
    "N2H4": {"full_name": "Hydrazine",        "reach_compliant": False, "role": "RCS monoprop — SVHC under REACH"},
}

MISSION_PHASES_PLATFORM_A = [
    "EOR", "Far_Range_Rendezvous", "Near_Range_Rendezvous",
    "Docking", "Post_Docking_SK", "CAM"
]

MISSION_PHASES_PLATFORM_B = [
    "EOR", "Flyby", "SSA", "Station_Keeping", "CAM"
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EPConfig:
    """Electric propulsion parameters for a mission phase."""
    isp_bol: float               # BOL specific impulse (s)
    isp_eol: float               # EOL specific impulse (s)
    thrust_mn: float             # Thrust level (mN)
    duty_cycle: float            # Fraction of phase where EP can fire (0–1)
    warmup_xenon_kg: float = 0.0 # Xenon consumed during non-propulsive warm-up per session
    warmup_sessions: int = 0     # Number of warm-up sessions in this phase


@dataclass
class RCSConfig:
    """RCS bipropellant parameters for a mission phase."""
    isp_bol: float               # BOL specific impulse (s)
    isp_eol: float               # EOL specific impulse (s)
    mixture_ratio: float         # Oxidiser/fuel mass ratio (N2O/C3H6)
    margin_performance: float    # Isp/thrust uncertainty margin (fraction, e.g. 0.05)
    margin_execution: float      # Delta-V execution error margin (fraction)
    margin_operational: float    # Unplanned manoeuvre margin (fraction)


@dataclass
class MissionPhase:
    """Full definition of one mission phase."""
    name: str
    primary_propulsion: str      # "EP", "RCS", or "BOTH"
    delta_v: float               # Translational delta-V (m/s)
    acs_delta_v: float           # ACS delta-V charged to RCS (m/s)
    duration_days: float         # Phase duration (days)
    ep: Optional[EPConfig] = None
    rcs: Optional[RCSConfig] = None
    notes: str = ""

    # Thruster arm cosine loss (for EP SK through arm, Platform A post-docking)
    thruster_arm_angle_deg: float = 0.0   # Angle between arm thrust vector and manoeuvre axis (deg)


@dataclass
class SpacecraftConfig:
    """Top-level spacecraft and propellant loading configuration."""
    platform_name: str
    wet_mass_kg: float
    dry_mass_kg: float

    # EP propellant
    xenon_loaded_kg: float
    xenon_residual_fraction: float   # Fraction trapped/unusable at EOL

    # RCS propellant
    n2o_loaded_kg: float             # Oxidiser
    c3h6_loaded_kg: float            # Fuel
    rcs_residual_fraction: float     # Applied to total RCS propellant

    # Mission phases in execution order
    phases: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# CORE CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def tsiolkovsky(delta_v: float, isp: float, wet_mass: float) -> float:
    """
    Tsiolkovsky rocket equation.
    Returns propellant mass (kg) required to deliver delta_v (m/s)
    given specific impulse isp (s) and initial wet mass wet_mass (kg).
    """
    if isp <= 0 or wet_mass <= 0:
        raise ValueError(f"Invalid Isp ({isp}s) or wet mass ({wet_mass}kg).")
    mass_ratio = math.exp(delta_v / (G0 * isp))
    propellant_mass = wet_mass * (1.0 - 1.0 / mass_ratio)
    return propellant_mass


def split_bipropellant(total_mass: float, mixture_ratio: float):
    """
    Split total RCS propellant mass into oxidiser (N2O) and fuel (C3H6).
    mixture_ratio = m_oxidiser / m_fuel
    Returns (oxidiser_mass, fuel_mass).
    """
    fuel = total_mass / (1.0 + mixture_ratio)
    oxidiser = total_mass - fuel
    return oxidiser, fuel


def apply_cosine_loss(delta_v: float, angle_deg: float) -> float:
    """
    Adjust delta-V for cosine loss due to thrust vector misalignment or
    thruster arm angle offset. Returns the effective delta-V that must be
    commanded to deliver the required delta_v along the intended axis.
    """
    if angle_deg == 0.0:
        return delta_v
    cos_loss = math.cos(math.radians(angle_deg))
    if cos_loss <= 0:
        raise ValueError(f"Thruster arm angle {angle_deg}° produces zero or negative thrust component.")
    return delta_v / cos_loss


def compute_total_margin_fraction(rcs: RCSConfig) -> float:
    """
    Compute combined margin multiplier from individual margin fractions.
    Margins are applied multiplicatively (conservative approach).
    Total = (1 + perf) * (1 + exec) * (1 + ops) - 1
    """
    return (
        (1 + rcs.margin_performance) *
        (1 + rcs.margin_execution) *
        (1 + rcs.margin_operational)
    ) - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# PHASE BUDGET COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_phase_budget(phase: MissionPhase, wet_mass: float, case: str = "BOL"):
    """
    Compute propellant consumption for one mission phase.

    Returns a dict with all budget line items and updated wet mass.
    case: "BOL" or "EOL"
    """
    results = {
        "phase": phase.name,
        "case": case,
        "primary_propulsion": phase.primary_propulsion,
        "duration_days": phase.duration_days,
        "delta_v_commanded": phase.delta_v,
        "acs_delta_v": phase.acs_delta_v,
        "thruster_arm_angle_deg": phase.thruster_arm_angle_deg,
        "wet_mass_start": wet_mass,
        # EP line items
        "ep_delta_v_effective": 0.0,
        "ep_xenon_dv_kg": 0.0,
        "ep_xenon_warmup_kg": 0.0,
        "ep_xenon_total_kg": 0.0,
        "ep_isp_used": 0.0,
        "ep_effective_duty_cycle": 0.0,
        # RCS translational line items
        "rcs_delta_v_total": 0.0,
        "rcs_prop_nominal_kg": 0.0,
        "rcs_margin_kg": 0.0,
        "rcs_prop_with_margin_kg": 0.0,
        "rcs_n2o_kg": 0.0,
        "rcs_c3h6_kg": 0.0,
        "rcs_isp_used": 0.0,
        "rcs_mixture_ratio": 0.0,
        # ACS RCS line items
        "acs_rcs_prop_kg": 0.0,
        "acs_rcs_n2o_kg": 0.0,
        "acs_rcs_c3h6_kg": 0.0,
        # Totals
        "total_xenon_kg": 0.0,
        "total_n2o_kg": 0.0,
        "total_c3h6_kg": 0.0,
        "total_rcs_kg": 0.0,
        "wet_mass_end": 0.0,
        "notes": phase.notes,
        "warnings": []
    }

    # ── EP computation ────────────────────────────────────────────────────────
    if phase.primary_propulsion in ("EP", "BOTH"):
        ep = phase.ep
        if ep is None:
            raise ValueError(f"Phase '{phase.name}' uses EP but no EPConfig provided.")

        isp = ep.isp_bol if case == "BOL" else ep.isp_eol
        results["ep_isp_used"] = isp
        results["ep_effective_duty_cycle"] = ep.duty_cycle

        # Apply cosine loss for thruster arm angle (EP SK through arm)
        dv_effective = apply_cosine_loss(phase.delta_v, phase.thruster_arm_angle_deg)
        results["ep_delta_v_effective"] = dv_effective

        if dv_effective > 0:
            xenon_dv = tsiolkovsky(dv_effective, isp, wet_mass)
        else:
            xenon_dv = 0.0

        # Non-propulsive warm-up Xenon
        xenon_warmup = ep.warmup_xenon_kg * ep.warmup_sessions

        results["ep_xenon_dv_kg"] = xenon_dv
        results["ep_xenon_warmup_kg"] = xenon_warmup
        results["ep_xenon_total_kg"] = xenon_dv + xenon_warmup

        # Warn if cosine loss is significant
        if phase.thruster_arm_angle_deg > 0:
            cosine_loss_pct = (1 - math.cos(math.radians(phase.thruster_arm_angle_deg))) * 100
            if cosine_loss_pct > 1.0:
                results["warnings"].append(
                    f"Thruster arm cosine loss: {cosine_loss_pct:.2f}% "
                    f"(arm angle = {phase.thruster_arm_angle_deg}°)"
                )

        wet_mass -= results["ep_xenon_total_kg"]

    # ── RCS translational computation ────────────────────────────────────────
    if phase.primary_propulsion in ("RCS", "BOTH"):
        rcs = phase.rcs
        if rcs is None:
            raise ValueError(f"Phase '{phase.name}' uses RCS but no RCSConfig provided.")

        isp = rcs.isp_bol if case == "BOL" else rcs.isp_eol
        results["rcs_isp_used"] = isp
        results["rcs_mixture_ratio"] = rcs.mixture_ratio

        if phase.delta_v > 0:
            rcs_prop_nominal = tsiolkovsky(phase.delta_v, isp, wet_mass)
        else:
            rcs_prop_nominal = 0.0

        margin_fraction = compute_total_margin_fraction(rcs)
        rcs_prop_with_margin = rcs_prop_nominal * (1.0 + margin_fraction)
        rcs_margin = rcs_prop_with_margin - rcs_prop_nominal

        n2o, c3h6 = split_bipropellant(rcs_prop_with_margin, rcs.mixture_ratio)

        results["rcs_delta_v_total"] = phase.delta_v
        results["rcs_prop_nominal_kg"] = rcs_prop_nominal
        results["rcs_margin_kg"] = rcs_margin
        results["rcs_prop_with_margin_kg"] = rcs_prop_with_margin
        results["rcs_n2o_kg"] = n2o
        results["rcs_c3h6_kg"] = c3h6

        wet_mass -= rcs_prop_with_margin

    # ── ACS RCS computation (always charged to RCS) ───────────────────────────
    if phase.acs_delta_v > 0:
        # Use RCS config if available, otherwise require one
        rcs = phase.rcs
        if rcs is None:
            raise ValueError(
                f"Phase '{phase.name}' has ACS delta-V but no RCSConfig for ACS computation."
            )
        isp = rcs.isp_bol if case == "BOL" else rcs.isp_eol

        acs_prop = tsiolkovsky(phase.acs_delta_v, isp, wet_mass)
        acs_n2o, acs_c3h6 = split_bipropellant(acs_prop, rcs.mixture_ratio)

        results["acs_rcs_prop_kg"] = acs_prop
        results["acs_rcs_n2o_kg"] = acs_n2o
        results["acs_rcs_c3h6_kg"] = acs_c3h6

        wet_mass -= acs_prop

    # ── Totals ────────────────────────────────────────────────────────────────
    results["total_xenon_kg"] = results["ep_xenon_total_kg"]
    results["total_n2o_kg"] = results["rcs_n2o_kg"] + results["acs_rcs_n2o_kg"]
    results["total_c3h6_kg"] = results["rcs_c3h6_kg"] + results["acs_rcs_c3h6_kg"]
    results["total_rcs_kg"] = results["total_n2o_kg"] + results["total_c3h6_kg"]
    results["wet_mass_end"] = wet_mass

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FULL MISSION BUDGET
# ─────────────────────────────────────────────────────────────────────────────

def compute_mission_budget(sc: SpacecraftConfig, case: str = "BOL"):
    """
    Compute the full mission propellant budget for all phases sequentially.
    Mass depletes correctly from phase to phase.
    Returns list of per-phase result dicts and a summary dict.
    """
    wet_mass = sc.wet_mass_kg
    phase_results = []

    cumulative_xenon = 0.0
    cumulative_n2o = 0.0
    cumulative_c3h6 = 0.0

    for phase in sc.phases:
        result = compute_phase_budget(phase, wet_mass, case)
        wet_mass = result["wet_mass_end"]

        cumulative_xenon += result["total_xenon_kg"]
        cumulative_n2o   += result["total_n2o_kg"]
        cumulative_c3h6  += result["total_c3h6_kg"]

        result["cumulative_xenon_kg"] = cumulative_xenon
        result["cumulative_n2o_kg"]   = cumulative_n2o
        result["cumulative_c3h6_kg"]  = cumulative_c3h6

        phase_results.append(result)

    # ── Residual / trapped propellant ─────────────────────────────────────────
    xenon_residual = sc.xenon_loaded_kg * sc.xenon_residual_fraction
    rcs_loaded_total = sc.n2o_loaded_kg + sc.c3h6_loaded_kg
    rcs_residual = rcs_loaded_total * sc.rcs_residual_fraction
    # Split residual proportionally by mixture ratio (use last phase MR if available)
    last_rcs = next(
        (p.rcs for p in reversed(sc.phases) if p.rcs is not None), None
    )
    if last_rcs:
        n2o_res, c3h6_res = split_bipropellant(rcs_residual, last_rcs.mixture_ratio)
    else:
        n2o_res = rcs_residual * 0.5
        c3h6_res = rcs_residual * 0.5

    # ── Budget closure check ──────────────────────────────────────────────────
    xenon_required  = cumulative_xenon + xenon_residual
    n2o_required    = cumulative_n2o + n2o_res
    c3h6_required   = cumulative_c3h6 + c3h6_res

    xenon_margin_kg  = sc.xenon_loaded_kg - xenon_required
    n2o_margin_kg    = sc.n2o_loaded_kg - n2o_required
    c3h6_margin_kg   = sc.c3h6_loaded_kg - c3h6_required

    xenon_margin_pct  = (xenon_margin_kg / sc.xenon_loaded_kg * 100) if sc.xenon_loaded_kg > 0 else 0
    n2o_margin_pct    = (n2o_margin_kg / sc.n2o_loaded_kg * 100) if sc.n2o_loaded_kg > 0 else 0
    c3h6_margin_pct   = (c3h6_margin_kg / sc.c3h6_loaded_kg * 100) if sc.c3h6_loaded_kg > 0 else 0

    # ── Mixture ratio balance check ───────────────────────────────────────────
    total_n2o_consumed  = cumulative_n2o + n2o_res
    total_c3h6_consumed = cumulative_c3h6 + c3h6_res
    actual_mr = total_n2o_consumed / total_c3h6_consumed if total_c3h6_consumed > 0 else 0

    n2o_remaining  = sc.n2o_loaded_kg - total_n2o_consumed
    c3h6_remaining = sc.c3h6_loaded_kg - total_c3h6_consumed

    summary = {
        "case": case,
        "platform": sc.platform_name,
        "wet_mass_initial_kg": sc.wet_mass_initial_kg if hasattr(sc, 'wet_mass_initial_kg') else sc.wet_mass_kg,
        "wet_mass_final_kg": wet_mass,
        # Xenon
        "xenon_loaded_kg": sc.xenon_loaded_kg,
        "xenon_consumed_dv_kg": cumulative_xenon - sum(r["ep_xenon_warmup_kg"] for r in phase_results),
        "xenon_consumed_warmup_kg": sum(r["ep_xenon_warmup_kg"] for r in phase_results),
        "xenon_consumed_total_kg": cumulative_xenon,
        "xenon_residual_kg": xenon_residual,
        "xenon_required_kg": xenon_required,
        "xenon_margin_kg": xenon_margin_kg,
        "xenon_margin_pct": xenon_margin_pct,
        "xenon_closes": xenon_margin_kg >= 0,
        # N2O
        "n2o_loaded_kg": sc.n2o_loaded_kg,
        "n2o_consumed_kg": cumulative_n2o,
        "n2o_residual_kg": n2o_res,
        "n2o_required_kg": n2o_required,
        "n2o_margin_kg": n2o_margin_kg,
        "n2o_margin_pct": n2o_margin_pct,
        "n2o_remaining_at_eol_kg": n2o_remaining,
        "n2o_closes": n2o_margin_kg >= 0,
        # C3H6
        "c3h6_loaded_kg": sc.c3h6_loaded_kg,
        "c3h6_consumed_kg": cumulative_c3h6,
        "c3h6_residual_kg": c3h6_res,
        "c3h6_required_kg": c3h6_required,
        "c3h6_margin_kg": c3h6_margin_kg,
        "c3h6_margin_pct": c3h6_margin_pct,
        "c3h6_remaining_at_eol_kg": c3h6_remaining,
        "c3h6_closes": c3h6_margin_kg >= 0,
        # Mixture ratio balance
        "actual_mission_mr": actual_mr,
        "n2o_remaining_kg": n2o_remaining,
        "c3h6_remaining_kg": c3h6_remaining,
    }

    return phase_results, summary


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_header(title: str):
    width = 90
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title: str):
    print(f"\n{'─' * 90}")
    print(f"  {title}")
    print(f"{'─' * 90}")


def print_reach_compliance(sc: SpacecraftConfig):
    print_header("REACH COMPLIANCE CHECK")
    propellants = [("N2O", "RCS Oxidiser"), ("C3H6", "RCS Fuel"), ("Xe", "EP Propellant")]
    for key, role in propellants:
        info = PROPELLANT_REGISTRY.get(key, {})
        status = "✓ COMPLIANT" if info.get("reach_compliant") else "✗ NON-COMPLIANT (SVHC)"
        print(f"  {role:25s}  {info.get('full_name',''):20s}  {status}")
    print(f"\n  Ignition system : Spark / torch ignition (no catalyst bed — no preheat model)")
    print(f"  Platform        : {sc.platform_name}")


def print_phase_table(phase_results: list, case: str):
    print_section(f"PHASE-BY-PHASE BUDGET — {case} CASE")
    col = "{:<28} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}"
    header = col.format(
        "Phase", "dV(m/s)", "ACS(m/s)",
        "Xe_dV(kg)", "Xe_wu(kg)", "N2O(kg)", "C3H6(kg)", "Margin(kg)", "Mass_end(kg)"
    )
    print(f"\n  {header}")
    print(f"  {'─'*88}")
    for r in phase_results:
        rcs_margin = r.get("rcs_margin_kg", 0.0)
        print(f"  " + col.format(
            r["phase"][:27],
            f"{r['delta_v_commanded']:.1f}",
            f"{r['acs_delta_v']:.1f}",
            f"{r['ep_xenon_dv_kg']:.3f}",
            f"{r['ep_xenon_warmup_kg']:.3f}",
            f"{r['total_n2o_kg']:.3f}",
            f"{r['total_c3h6_kg']:.3f}",
            f"{rcs_margin:.3f}",
            f"{r['wet_mass_end']:.2f}"
        ))
        for w in r.get("warnings", []):
            print(f"    ⚠  {w}")


def print_cumulative_table(phase_results: list, case: str):
    print_section(f"CUMULATIVE PROPELLANT CONSUMED — {case} CASE")
    col = "{:<28} {:>14} {:>14} {:>14}"
    print(f"\n  " + col.format("After Phase", "Xe_cum(kg)", "N2O_cum(kg)", "C3H6_cum(kg)"))
    print(f"  {'─'*72}")
    for r in phase_results:
        print(f"  " + col.format(
            r["phase"][:27],
            f"{r['cumulative_xenon_kg']:.3f}",
            f"{r['cumulative_n2o_kg']:.3f}",
            f"{r['cumulative_c3h6_kg']:.3f}"
        ))


def print_summary(summary: dict):
    print_section(f"BUDGET SUMMARY — {summary['case']} CASE")

    def closure(closes): return "✓ CLOSES" if closes else "✗ DOES NOT CLOSE"

    print(f"\n  Platform : {summary['platform']}")
    print(f"  Wet mass : {summary['wet_mass_initial_kg']:.2f} kg  →  {summary['wet_mass_final_kg']:.2f} kg")

    print(f"\n  {'Propellant':<12} {'Loaded':>10} {'Consumed':>10} {'Residual':>10} {'Required':>10} {'Margin kg':>10} {'Margin %':>10}  {'Status'}")
    print(f"  {'─'*88}")

    rows = [
        ("Xenon (Xe)", "xenon"),
        ("N2O",        "n2o"),
        ("C3H6",       "c3h6"),
    ]
    for label, key in rows:
        print(f"  {label:<12}"
              f"  {summary[f'{key}_loaded_kg']:>9.3f}"
              f"  {summary[f'{key}_consumed_kg' if key != 'xenon' else 'xenon_consumed_total_kg']:>9.3f}"
              f"  {summary[f'{key}_residual_kg']:>9.3f}"
              f"  {summary[f'{key}_required_kg']:>9.3f}"
              f"  {summary[f'{key}_margin_kg']:>9.3f}"
              f"  {summary[f'{key}_margin_pct']:>8.1f}%"
              f"  {closure(summary[f'{key}_closes'])}")

    print(f"\n  ── Mixture Ratio Balance Check ──────────────────────────────────────────")
    print(f"  Mission-average N2O/C3H6 ratio : {summary['actual_mission_mr']:.3f}")
    print(f"  N2O  remaining at EOL          : {summary['n2o_remaining_kg']:.3f} kg")
    print(f"  C3H6 remaining at EOL          : {summary['c3h6_remaining_kg']:.3f} kg")
    if summary['n2o_remaining_kg'] > 0.5 or summary['c3h6_remaining_kg'] > 0.5:
        print(f"  ⚠  Significant residual imbalance detected — review mixture ratio strategy.")
    else:
        print(f"  ✓  Mixture ratio balance acceptable.")


def print_bol_eol_comparison(summary_bol: dict, summary_eol: dict):
    print_header("BOL vs EOL COMPARISON")
    labels = [
        ("Xenon consumed (kg)",   "xenon_consumed_total_kg"),
        ("Xenon margin (kg)",     "xenon_margin_kg"),
        ("N2O consumed (kg)",     "n2o_consumed_kg"),
        ("N2O margin (kg)",       "n2o_margin_kg"),
        ("C3H6 consumed (kg)",    "c3h6_consumed_kg"),
        ("C3H6 margin (kg)",      "c3h6_margin_kg"),
        ("Final wet mass (kg)",   "wet_mass_final_kg"),
    ]
    col = "{:<30} {:>14} {:>14} {:>14}"
    print(f"\n  " + col.format("Item", "BOL", "EOL", "Delta (EOL-BOL)"))
    print(f"  {'─'*74}")
    for label, key in labels:
        bol_val = summary_bol.get(key, 0.0)
        eol_val = summary_eol.get(key, 0.0)
        delta = eol_val - bol_val
        print(f"  " + col.format(label, f"{bol_val:.3f}", f"{eol_val:.3f}", f"{delta:+.3f}"))


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(phase_results_bol: list, phase_results_eol: list,
               summary_bol: dict, summary_eol: dict,
               platform_name: str, output_dir: str = "."):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"propellant_budget_{platform_name}_{timestamp}.csv")

    fieldnames = [
        "case", "phase", "primary_propulsion", "duration_days",
        "delta_v_commanded", "acs_delta_v", "thruster_arm_angle_deg",
        "wet_mass_start", "wet_mass_end",
        "ep_isp_used", "ep_delta_v_effective",
        "ep_xenon_dv_kg", "ep_xenon_warmup_kg", "ep_xenon_total_kg",
        "rcs_isp_used", "rcs_mixture_ratio",
        "rcs_prop_nominal_kg", "rcs_margin_kg", "rcs_prop_with_margin_kg",
        "rcs_n2o_kg", "rcs_c3h6_kg",
        "acs_rcs_prop_kg", "acs_rcs_n2o_kg", "acs_rcs_c3h6_kg",
        "total_xenon_kg", "total_n2o_kg", "total_c3h6_kg", "total_rcs_kg",
        "cumulative_xenon_kg", "cumulative_n2o_kg", "cumulative_c3h6_kg",
        "notes", "warnings"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in phase_results_bol:
            r["warnings"] = "; ".join(r.get("warnings", []))
            writer.writerow(r)
        for r in phase_results_eol:
            r["warnings"] = "; ".join(r.get("warnings", []))
            writer.writerow(r)

    print(f"\n  ✓  CSV exported: {filename}")
    return filename


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> SpacecraftConfig:
    """Load spacecraft and mission configuration from a JSON file."""
    with open(config_path) as f:
        cfg = json.load(f)

    phases = []
    for p in cfg["phases"]:
        ep_cfg = None
        rcs_cfg = None

        if "ep" in p:
            ep_cfg = EPConfig(**p["ep"])
        if "rcs" in p:
            rcs_cfg = RCSConfig(**p["rcs"])

        phases.append(MissionPhase(
            name=p["name"],
            primary_propulsion=p["primary_propulsion"],
            delta_v=p["delta_v"],
            acs_delta_v=p.get("acs_delta_v", 0.0),
            duration_days=p.get("duration_days", 0.0),
            ep=ep_cfg,
            rcs=rcs_cfg,
            notes=p.get("notes", ""),
            thruster_arm_angle_deg=p.get("thruster_arm_angle_deg", 0.0)
        ))

    sc = SpacecraftConfig(
        platform_name=cfg["platform_name"],
        wet_mass_kg=cfg["wet_mass_kg"],
        dry_mass_kg=cfg["dry_mass_kg"],
        xenon_loaded_kg=cfg["xenon_loaded_kg"],
        xenon_residual_fraction=cfg["xenon_residual_fraction"],
        n2o_loaded_kg=cfg["n2o_loaded_kg"],
        c3h6_loaded_kg=cfg["c3h6_loaded_kg"],
        rcs_residual_fraction=cfg["rcs_residual_fraction"],
        phases=phases
    )
    return sc


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN EXAMPLE — Platform A
# ─────────────────────────────────────────────────────────────────────────────

def build_example_platform_a() -> SpacecraftConfig:
    """
    Example configuration for Platform A.
    All values are illustrative — replace with actual mission and hardware data.
    Isp values for C3H6/N2O and Xe shall be taken from supplier qualification data.
    """

    rcs_default = RCSConfig(
        isp_bol=310.0,       # s — C3H6/N2O bipropellant, BOL (illustrative)
        isp_eol=295.0,       # s — EOL (blowdown pressure degradation)
        mixture_ratio=7.0,   # N2O/C3H6 mass ratio (illustrative)
        margin_performance=0.03,   # 3% Isp/thrust uncertainty
        margin_execution=0.05,     # 5% delta-V execution error (3-sigma)
        margin_operational=0.05    # 5% unplanned manoeuvres
    )

    phases = [
        # ── EOR: EP primary, RCS for ACS ────────────────────────────────────
        MissionPhase(
            name="EOR",
            primary_propulsion="BOTH",
            delta_v=1500.0,         # m/s — illustrative GTO to GEO transfer
            acs_delta_v=8.0,        # m/s — RCS attitude control during EP arcs
            duration_days=90.0,
            ep=EPConfig(
                isp_bol=1800.0,     # s — Hall thruster BOL (illustrative)
                isp_eol=1700.0,     # s — EOL (Xe tank pressure drop, slight Isp degradation)
                thrust_mn=200.0,    # mN
                duty_cycle=0.55,    # 55% — eclipse and power-limited
                warmup_xenon_kg=0.002,  # kg per warm-up session
                warmup_sessions=90      # ~1 per day
            ),
            rcs=rcs_default,
            notes="EP primary delta-V. RCS ACS only. Duty cycle limited by eclipses and solar power.",
        ),

        # ── Far Range Rendezvous: RCS only ───────────────────────────────────
        MissionPhase(
            name="Far_Range_Rendezvous",
            primary_propulsion="RCS",
            delta_v=25.0,
            acs_delta_v=3.0,
            duration_days=5.0,
            rcs=rcs_default,
            notes="RCS only. Includes hold point station-keeping delta-V.",
        ),

        # ── Near Range Rendezvous: RCS only ──────────────────────────────────
        MissionPhase(
            name="Near_Range_Rendezvous",
            primary_propulsion="RCS",
            delta_v=10.0,
            acs_delta_v=2.0,
            duration_days=1.0,
            rcs=rcs_default,
            notes="Fine proximity manoeuvres. MIB-limited pulsing.",
        ),

        # ── Docking: RCS only ─────────────────────────────────────────────────
        MissionPhase(
            name="Docking",
            primary_propulsion="RCS",
            delta_v=5.0,
            acs_delta_v=1.5,
            duration_days=0.5,
            rcs=rcs_default,
            notes="Final approach and capture. Includes abort reserve allocation.",
        ),

        # ── Post-Docking SK: EP through thruster arm, RCS for ACS ────────────
        # Thruster arm has 8° offset from N/S axis (illustrative)
        MissionPhase(
            name="Post_Docking_SK",
            primary_propulsion="BOTH",
            delta_v=500.0,          # m/s — multi-year N/S + E/W SK (illustrative)
            acs_delta_v=15.0,       # m/s — attitude control of stack
            duration_days=1825.0,   # 5 years
            thruster_arm_angle_deg=8.0,  # Arm angle offset from thrust axis
            ep=EPConfig(
                isp_bol=1750.0,
                isp_eol=1600.0,     # Larger EOL degradation — lower Xe tank pressure
                thrust_mn=180.0,
                duty_cycle=0.85,
                warmup_xenon_kg=0.002,
                warmup_sessions=365
            ),
            rcs=rcs_default,
            notes="Stack SK. EP through thruster arm — cosine loss applied. Stack mass drives ACS budget.",
        ),

        # ── CAM reserve: RCS only ─────────────────────────────────────────────
        MissionPhase(
            name="CAM_Reserve",
            primary_propulsion="RCS",
            delta_v=30.0,           # m/s — worst-case CAM at stack configuration
            acs_delta_v=2.0,
            duration_days=0.0,
            rcs=RCSConfig(
                isp_bol=310.0,
                isp_eol=295.0,
                mixture_ratio=7.0,
                margin_performance=0.03,
                margin_execution=0.10,  # Higher execution margin for emergency manoeuvre
                margin_operational=0.10
            ),
            notes="Protected CAM reserve. Computed at stack mass. Not consumed by nominal ops.",
        ),
    ]

    return SpacecraftConfig(
        platform_name="Platform_A",
        wet_mass_kg=3500.0,
        dry_mass_kg=1200.0,
        xenon_loaded_kg=900.0,
        xenon_residual_fraction=0.01,   # 1% trapped
        n2o_loaded_kg=280.0,
        c3h6_loaded_kg=40.0,
        rcs_residual_fraction=0.02,     # 2% trapped
        phases=phases
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_budget(sc: SpacecraftConfig, output_dir: str = "."):
    # Store initial wet mass for summary
    initial_wet = sc.wet_mass_kg

    print_reach_compliance(sc)

    # BOL case
    phase_results_bol, summary_bol = compute_mission_budget(sc, case="BOL")
    summary_bol["wet_mass_initial_kg"] = initial_wet

    # EOL case — reset wet mass
    sc.wet_mass_kg = initial_wet
    phase_results_eol, summary_eol = compute_mission_budget(sc, case="EOL")
    summary_eol["wet_mass_initial_kg"] = initial_wet

    # Print BOL
    print_phase_table(phase_results_bol, "BOL")
    print_cumulative_table(phase_results_bol, "BOL")
    print_summary(summary_bol)

    # Print EOL
    print_phase_table(phase_results_eol, "EOL")
    print_cumulative_table(phase_results_eol, "EOL")
    print_summary(summary_eol)

    # Comparison
    print_bol_eol_comparison(summary_bol, summary_eol)

    # Export
    export_csv(phase_results_bol, phase_results_eol,
               summary_bol, summary_eol,
               sc.platform_name, output_dir)

    # Final closure banner
    print_header("BUDGET CLOSURE — FINAL VERDICT")
    all_close = (
        summary_eol["xenon_closes"] and
        summary_eol["n2o_closes"] and
        summary_eol["c3h6_closes"]
    )
    if all_close:
        print(f"\n  ✓  EOL budget CLOSES for all propellants on {sc.platform_name}.")
    else:
        print(f"\n  ✗  EOL budget DOES NOT CLOSE on {sc.platform_name}. Review flagged items above.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Propellant Budget Calculator — C3H6/N2O + Xe dual propulsion"
    )
    parser.add_argument("--config", type=str, help="Path to JSON mission config file")
    parser.add_argument("--example", action="store_true", help="Run built-in Platform A example")
    parser.add_argument("--output", type=str, default=".", help="Output directory for CSV")
    args = parser.parse_args()

    if args.example:
        sc = build_example_platform_a()
        run_budget(sc, output_dir=args.output)
    elif args.config:
        sc = load_config(args.config)
        run_budget(sc, output_dir=args.output)
    else:
        parser.print_help()
        print("\n  Tip: run with --example to see a full Platform A budget.")


if __name__ == "__main__":
    main()
