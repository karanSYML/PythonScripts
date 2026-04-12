#!/usr/bin/env python3
"""
pipeline_runner.py — Operational parametric sweep for the thruster arm assembly.

Arm geometry is hardware-fixed (supplier constraint):
  Link 1  : 1.12 m   (shoulder → elbow)
  Link 2  : 1.40 m   (elbow → wrist)
  Bracket : 0.35 m   (wrist → thruster exit)
  Reach   : 2.87 m   total

The only operational degree of freedom swept here is shoulder_yaw_deg (J1).
J2 and J3 are resolved by IK to place the thruster horizontally at the
CoG-aimed angle — they are not independent sweep parameters.

Secondary sweeps:
  client_mass        — uncertainty in client satellite dry mass
  panel_tracking_deg — solar panel sun-tracking angle during burns

Outputs (pipeline_runner_output/):
  results.csv              full sweep results (all cases)
  pareto_front.csv         Pareto-optimal configurations
  heatmap_*.png            erosion heatmaps
  pareto_2d.png            Pareto scatter: fuel cost vs erosion risk
  heatmap_score.png        composite score over yaw × client_mass
  openplume_cases/         OpenPlume input files for MARGINAL/CAUTION cases
"""

import os
import numpy as np

from plume_impingement_pipeline import (
    ThrusterParams, MaterialParams,
    PlumePipeline, CaseMatrixGenerator,
    generate_heatmaps,
)
from openplume_exporter import export_openplume_cases
from pareto_scoring import ParetoScorer

# ---------------------------------------------------------------------------
# Hardware-fixed arm geometry (supplier constraint)
# ---------------------------------------------------------------------------
L1           = 1.12   # m
L2           = 1.40   # m
ARM_REACH_M  = L1 + L2          # 2.52 m  (L1+L2; bracket is separate)
LINK_RATIO   = L1 / ARM_REACH_M # 0.4444…

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = "pipeline_runner_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

thruster = ThrusterParams(
    name="SPT-100-like",
    discharge_voltage=300.0,
    mass_flow_rate=5e-6,
    thrust_N=0.08,
)
material = MaterialParams(
    name="Silver_interconnect",
    thickness_um=15.0,
)

# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------
pipeline = PlumePipeline(thruster, material)
gen      = pipeline.generator

# Operational sweep: shoulder yaw covers the full J1 range (spec: 0–270°)
gen.set_param_range("shoulder_yaw_deg",
                    np.arange(0.0, 271.0, 15.0))   # 0, 15, 30 … 270

# Secondary sensitivity sweeps
gen.set_param_range("client_mass",
                    np.array([1500.0, 2000.0, 2500.0, 3000.0]))
gen.set_param_range("panel_tracking_deg",
                    np.array([-15.0, 0.0, 15.0]))

# Everything else fixed — arm geometry locked to hardware spec
fixed = {
    "arm_reach_m":         ARM_REACH_M,
    "link_ratio":          LINK_RATIO,
    "servicer_mass":       670.0,       # 530 kg dry + 140 kg Xe
    "panel_span_one_side": 16.0,
    "firing_duration_s":   25000.0,
    "mission_duration_yr": 5.0,
}

cases = gen.generate_reduced_matrix(
    fixed,
    sweep_params=["shoulder_yaw_deg", "client_mass", "panel_tracking_deg"],
)
print(f"Total cases: {len(cases)}")

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
results = pipeline.run_sweep(cases, verbose=True)

print("\nPipeline summary:", pipeline.summary())

# ---------------------------------------------------------------------------
# Pareto scoring (NSSK — dominant manoeuvre)
# ---------------------------------------------------------------------------
print("\n[PARETO SCORING — NSSK]")
scorer = ParetoScorer(manoeuvre_type="NSSK", angle_budget_deg=50.0)
scored = scorer.score(results)
scorer.summary(scored)

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
pipeline.export_results_csv(os.path.join(OUTPUT_DIR, "results.csv"))
scorer.export_csv(scored, os.path.join(OUTPUT_DIR, "pareto_front.csv"),
                  pareto_only=True)

print("\n[GENERATING PLOTS]")
generate_heatmaps(
    results, "shoulder_yaw_deg", "client_mass",
    metric="max_erosion_um",
    output_dir=OUTPUT_DIR,
    thickness_um=material.thickness_um,
)
scorer.plot_pareto(
    scored,
    x="fuel_cost_norm", y="erosion_risk_norm", color="disturbance_norm",
    save_path=os.path.join(OUTPUT_DIR, "pareto_2d.png"),
)
scorer.plot_heatmap(
    scored,
    x_param="shoulder_yaw_deg", y_param="client_mass",
    metric="pareto_score",
    save_path=os.path.join(OUTPUT_DIR, "heatmap_score.png"),
)

# ---------------------------------------------------------------------------
# OpenPlume export (MARGINAL / CAUTION cases only)
# ---------------------------------------------------------------------------
op_cases = pipeline.get_openplume_cases()
print(f"\nCases for OpenPlume simulation: {len(op_cases)}")
if op_cases:
    export_openplume_cases(
        op_cases,
        output_dir=os.path.join(OUTPUT_DIR, "openplume_cases"),
        thruster=thruster,
        material=material,
    )
