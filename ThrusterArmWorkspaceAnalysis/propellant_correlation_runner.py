#!/usr/bin/env python3
"""
Propellant-Erosion Correlation Runner
======================================
Runs the plume impingement pipeline sweep first, then feeds each candidate
case into TimeResolvedErosion for a full time-resolved analysis.

Servicer dry mass : 600 kg  (hardware-confirmed)
Xenon propellant  : 144 kg  (hardware-confirmed)
Total servicer    : 744 kg
"""

import os
import csv
import numpy as np

from plume_impingement_pipeline import (
    ThrusterParams, MaterialParams, RoboticArmGeometry, StackConfig, OperationalParams,
    PlumePipeline, CaseMatrixGenerator,
)
from propellant_erosion_correlation import (
    StationkeepingBudget, PropellantConfig,
    TimeResolvedErosion, plot_time_resolved_erosion,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVICER_DRY_MASS_KG  = 600.0   # hardware-confirmed
XENON_LOADED_KG       = 144.0   # hardware-confirmed
SERVICER_TOTAL_MASS   = SERVICER_DRY_MASS_KG + XENON_LOADED_KG   # 744 kg

# Hardware-fixed arm geometry (supplier constraint)
_L1          = 1.1139
_L2          = 1.5227
ARM_REACH_M  = _L1 + _L2             # 2.6366 m  (L1+L2; bracket 0.4844 m separate)
LINK_RATIO   = _L1 / ARM_REACH_M     # ≈ 0.4222

OUTPUT_DIR = "./propellant_correlation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thruster: SPT-100-like
thruster = ThrusterParams(
    name="SPT-100-like",
    isp=1500.0,
    discharge_voltage=300.0,
    mass_flow_rate=5e-6,
    beam_divergence_half_angle=20.0,
    plume_cosine_exponent=10.0,
    thrust_N=0.08,
)

# Solar panel silver interconnect
material = MaterialParams(
    name="Silver_interconnect",
    thickness_um=25.0,
)

# GEO stationkeeping budget (NSSK dominant)
sk_budget = StationkeepingBudget(
    nssk_dv_per_year=55.0,
    ewsk_dv_per_year=1.5,
    momentum_dv_per_year=0.5,
    margin_fraction=0.10,
    nssk_manoeuvres_per_day=2,
    ewsk_manoeuvres_per_week=2,
)

# Propellant config — 140 kg Xe, tank at servicer geometric centre
prop_config = PropellantConfig(
    tank_capacity_kg=160.0,
    propellant_loaded_kg=XENON_LOADED_KG,
    residual_fraction=0.03,
    tank_position_x=0.0,
    tank_position_y=0.0,
    tank_position_z=0.0,
)

# ---------------------------------------------------------------------------
# Step 1 — Pipeline parametric sweep
# ---------------------------------------------------------------------------

print("=" * 70)
print("  STEP 1: PIPELINE PARAMETRIC SWEEP")
print("=" * 70)

pipeline = PlumePipeline(thruster, material)
gen = pipeline.generator

gen.set_param_range("shoulder_yaw_deg",
                    np.arange(0.0, 271.0, 15.0))   # full J1 range: 0–270°

fixed_params = {
    "arm_reach_m":         ARM_REACH_M,   # hardware-fixed: L1+L2 = 2.52 m
    "link_ratio":          LINK_RATIO,    # hardware-fixed: L1/(L1+L2) ≈ 0.444
    "client_mass":         2800.0,
    "servicer_mass":       SERVICER_TOTAL_MASS,
    "panel_span_one_side": 16.0,
    "firing_duration_s":   25000.0,
    "mission_duration_yr": 5.0,
    "panel_tracking_deg":  0.0,
}

cases = gen.generate_reduced_matrix(
    fixed_params,
    sweep_params=["shoulder_yaw_deg"],
)
print(f"\nTotal cases to screen: {len(cases)}")

sweep_results = pipeline.run_sweep(cases, verbose=True)

summary = pipeline.summary()
print(f"\nPipeline summary: {summary}")

# ---------------------------------------------------------------------------
# Step 2 — Select candidates for time-resolved analysis
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  STEP 2: SELECTING CANDIDATES")
print("=" * 70)

# Take MARGINAL and CAUTION cases; fall back to all if none found
candidates = [r for r in sweep_results if r["status"] in ("MARGINAL", "CAUTION")]
if not candidates:
    print("  No MARGINAL/CAUTION cases found — running all cases.")
    candidates = sweep_results

print(f"  Candidates selected: {len(candidates)}")
for r in candidates:
    print(f"    reach={r['arm_reach_m']}m  yaw={r['shoulder_yaw_deg']}°  "
          f"ratio={r['link_ratio']}  status={r['status']}  "
          f"max_erosion={r['max_erosion_um']:.2f} µm")

# ---------------------------------------------------------------------------
# Step 3 — Time-resolved analysis for each candidate
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  STEP 3: TIME-RESOLVED EROSION ANALYSIS")
print("=" * 70)

time_resolved_results = []

for i, case in enumerate(candidates):
    # Build arm from hardware-confirmed geometry; only shoulder yaw varies per case
    arm = RoboticArmGeometry(shoulder_yaw_deg=case['shoulder_yaw_deg'])
    # stack and ops still come from the case dict
    _, stack, ops = CaseMatrixGenerator.case_to_objects(case)

    # Ensure servicer mass is consistent: dry + Xe
    stack.servicer_mass = SERVICER_TOTAL_MASS

    label = (f"reach{case['arm_reach_m']:.1f}m_"
             f"yaw{int(case['shoulder_yaw_deg'])}deg_"
             f"ratio{case['link_ratio']:.1f}")

    print(f"\n[{i+1}/{len(candidates)}] {label}")
    print(f"  Pipeline: {case['status']}, max_erosion={case['max_erosion_um']:.2f} µm")

    integrator = TimeResolvedErosion(
        thruster, material, arm, stack, prop_config, sk_budget
    )

    result = integrator.integrate_mission(
        mission_years=ops.mission_duration_years,
        time_step_days=30.0,
        panel_tracking_schedule={
            "NSSK": ops.panel_sun_tracking_angle_deg,
            "EWSK": 0.0,
        },
        verbose=True,
    )

    s = result["summary"]
    print(f"  Limiting factor  : {s['limiting_factor']}")
    print(f"  Achievable       : {s['mission_years_actual']:.1f} yr "
          f"(prop-limited: {s['mission_years_propellant_limited']:.1f} yr, "
          f"erosion-limited: {s['mission_years_erosion_limited']:.1f} yr)")
    print(f"  Max erosion      : {s['max_erosion_um']:.2f} µm "
          f"({s['erosion_fraction']*100:.1f}% of {material.thickness_um} µm)")
    print(f"  Propellant used  : {s['propellant_used_kg']:.1f} kg  "
          f"(remaining: {s['propellant_remaining_kg']:.1f} kg)")
    print(f"  COG Z-shift      : {s['cog_shift_z_mm']:.1f} mm")

    # Save plots per case
    case_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(case_dir, exist_ok=True)
    plot_time_resolved_erosion(result, output_dir=case_dir)

    time_resolved_results.append({
        # Case identity
        "label":                  label,
        "arm_reach_m":            case["arm_reach_m"],
        "shoulder_yaw_deg":       case["shoulder_yaw_deg"],
        "link_ratio":             case["link_ratio"],
        # Pipeline screening result
        "pipeline_status":        case["status"],
        "pipeline_max_erosion_um": case["max_erosion_um"],
        # Time-resolved result
        "limiting_factor":        s["limiting_factor"],
        "mission_years_actual":   s["mission_years_actual"],
        "mission_years_prop_lim": s["mission_years_propellant_limited"],
        "mission_years_ero_lim":  s["mission_years_erosion_limited"],
        "max_erosion_um":         s["max_erosion_um"],
        "erosion_fraction":       s["erosion_fraction"],
        "propellant_used_kg":     s["propellant_used_kg"],
        "propellant_remaining_kg": s["propellant_remaining_kg"],
        "cog_shift_z_mm":         s["cog_shift_z_mm"],
        "erosion_failed":         s["erosion_failed"],
    })

# ---------------------------------------------------------------------------
# Step 4 — Export combined results
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  STEP 4: EXPORT")
print("=" * 70)

csv_path = os.path.join(OUTPUT_DIR, "combined_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=time_resolved_results[0].keys())
    writer.writeheader()
    writer.writerows(time_resolved_results)
print(f"  Results saved to: {csv_path}")

# Final summary table
print("\n  Final Summary:")
print(f"  {'Label':<45} {'Pipeline':>10} {'Limit':>10} {'Achievable':>11} {'Erosion':>10}")
print("  " + "-" * 90)
for r in time_resolved_results:
    print(f"  {r['label']:<45} {r['pipeline_status']:>10} "
          f"{r['limiting_factor']:>10} {r['mission_years_actual']:>10.1f}yr "
          f"{r['max_erosion_um']:>9.2f}µm")
