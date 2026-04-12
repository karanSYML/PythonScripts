# Thruster Arm Workspace Analysis

Plasma plume impingement analysis pipeline for a GEO satellite life-extension servicing mission. A servicer spacecraft docks to an end-of-life client via the Launch Adapter Ring (LAR). A 3-DOF robotic arm positions an ion thruster for North-South and East-West station-keeping (NSSK/EWSK) while managing plume erosion risk on the client's solar panels and antenna dishes.

---

## Coordinate Frame

All geometry is expressed in the **LAR frame**:

| Axis | Direction |
|------|-----------|
| +Z | Nadir (toward Earth) — client bus extends this way |
| −Z | Anti-Earth — servicer sits on this side |
| +X | North |
| +Y | East |
| Origin | LAR mechanical interface (Z = 0) |

---

## Repository Structure

```
ThrusterArmWorkspaceAnalysis/
│
├── plume_impingement_pipeline.py      # Core library — all data classes + engines
├── propellant_erosion_correlation.py  # Time-resolved erosion + CoG migration
├── geometry_visualizer.py             # 3-D matplotlib assembly visualiser
├── openplume_exporter.py              # Export cases for full OpenPlume simulation
├── pareto_scoring.py                  # Multi-objective Pareto scoring + plots
├── urdf_generator.py                  # Parametric URDF + parabolic dish STLs
│
├── pipeline_runner.py                 # Example: quick parametric sweep
├── propellant_correlation_runner.py   # Example: sweep + time-resolved analysis
├── daily_on_summary_stats.py          # Station-keeping schedule statistics
│
├── pdsk_6month_maneuverPlan.xlsx      # 6-month NSSK/EWSK manoeuvre plan
├── thruster_arm_urdf_spec.md          # Hardware + URDF specification document
├── ThrusterArmAnalysis.md             # Background analysis notes
│
├── pipeline_runner_output/            # Outputs from pipeline_runner.py
├── propellant_correlation_results/    # Outputs from propellant_correlation_runner.py
├── pareto_output/                     # Outputs from pareto_scoring.py
└── urdf_output/                       # Generated URDF + STL mesh files
```

---

## Dependencies

```bash
pip install numpy matplotlib openpyxl yourdfpy "pyglet<2" trimesh
```

| Package | Used for |
|---------|----------|
| `numpy` | All numerical computation |
| `matplotlib` | Geometry visualisation, heatmaps, Pareto plots |
| `openpyxl` | Reading the manoeuvre plan Excel file |
| `yourdfpy` | Loading and viewing the generated URDF |
| `pyglet<2` | 3-D window backend for yourdfpy/trimesh |
| `trimesh` | Mesh handling for URDF viewer |

---

## Module Overview

### `plume_impingement_pipeline.py` — Core library

Contains all data classes and the main analysis engine. Import everything from here.

**Key data classes:**

```python
ThrusterParams   # thruster physics: Isp, discharge voltage, mass flow, thrust
MaterialParams   # target surface: coating thickness, sputter yield coefficients
StackConfig      # combined geometry: client + servicer buses, panels, 4 antennas
RoboticArmGeometry  # 3-DOF arm: link lengths/masses, joint limits, IK/FK
OperationalParams   # mission ops: firing duration, firings/day, mission years
```

**Key engines:**

```python
GeometryEngine(arm, stack)
    .thruster_position()          # thruster exit in LAR frame (runs IK)
    .thrust_direction()           # plume unit vector aimed through stack CoG
    .compute_flux_geometry()      # distance/angle arrays for panel grid points
    .compute_antenna_flux_geometry()  # per-dish geometry (E1, E2, W1, W2)
    .stack_cog_with_arm()         # arm-inclusive CoG in LAR frame
    .thrust_metrics(thrust_N)     # deviation angles, moment arms, torques
    .check_arm_collision_with_client_bus()

ErosionEstimator(thruster, material)
    .cumulative_erosion_um(distance, offaxis_deg, incidence_deg, ops)

PlumePipeline(thruster, material)
    .run_sweep(cases)             # main entry point — returns list of result dicts
    .export_results_csv(path)
    .get_openplume_cases()        # filter to MARGINAL/CAUTION for full simulation

CaseMatrixGenerator()
    .set_param_range(name, values)
    .generate_reduced_matrix(fixed_params, sweep_params)  # recommended
    .generate_full_matrix()       # full Cartesian product — can be very large
    .case_to_objects(case)        # dict → (arm, stack, ops)
```

**`run_sweep()` result dict keys** (per case):

| Key | Description |
|-----|-------------|
| `status` | `SAFE` / `CAUTION` / `MARGINAL` / `FAIL` |
| `max_erosion_um` | Worst-point panel erosion depth [µm] |
| `erosion_fraction` | `max_erosion_um / coating_thickness` |
| `ant_{E1,E2,W1,W2}_erosion_um` | Per-dish lifetime erosion [µm] |
| `nssk_deviation_deg` | Angle from ideal North/South thrust direction |
| `ewsk_deviation_deg` | Angle from ideal East/West thrust direction |
| `nssk_moment_arm_m` | ⊥ distance from CoG to X-axis through thruster [m] |
| `ewsk_moment_arm_m` | ⊥ distance from CoG to Y-axis through thruster [m] |
| `nssk_torque_Nm` | Disturbance torque for an ideal NSSK burn [N·m] |
| `ewsk_torque_Nm` | Disturbance torque for an ideal EWSK burn [N·m] |
| `cog_x/y/z` | Stack + arm CoG position in LAR frame [m] |
| `thruster_pos_x/y/z` | Thruster exit position in LAR frame [m] |

---

### `propellant_erosion_correlation.py` — Time-resolved analysis

Integrates erosion over the full mission with propellant depletion and CoG migration at configurable time steps.

```python
from propellant_erosion_correlation import (
    PropellantConfig, StationkeepingBudget,
    TimeResolvedErosion, plot_time_resolved_erosion
)

prop_config = PropellantConfig(
    tank_capacity_kg=160.0,
    propellant_loaded_kg=140.0,
    residual_fraction=0.03,
)
sk_budget = StationkeepingBudget(
    nssk_dv_per_year=50.0,
    ewsk_dv_per_year=2.0,
    nssk_manoeuvres_per_day=2,
    ewsk_manoeuvres_per_week=2,
)

integrator = TimeResolvedErosion(thruster, material, arm, stack,
                                  prop_config, sk_budget)
result = integrator.integrate_mission(mission_years=5.0, time_step_days=30.0)

# result["summary"] keys:
#   mission_years_actual, mission_years_propellant_limited,
#   mission_years_erosion_limited, limiting_factor,
#   max_erosion_um, erosion_fraction, propellant_used_kg,
#   propellant_remaining_kg, cog_shift_z_mm, erosion_failed

plot_time_resolved_erosion(result, output_dir="results/")
```

---

### `pareto_scoring.py` — Multi-objective Pareto scoring

Scores sweep results on three objectives (all minimised):

| Objective | Formula |
|-----------|---------|
| **Fuel cost** | `1 − cos(deviation_deg)` — ΔV cosine penalty |
| **Disturbance** | `nssk_torque_Nm` or `ewsk_torque_Nm` |
| **Erosion risk** | `erosion_fraction + 0.3 × ant_max / thickness` |

```python
from pareto_scoring import ParetoScorer

scorer = ParetoScorer(manoeuvre_type="NSSK", angle_budget_deg=50.0)
scored = scorer.score(sweep_results)    # adds fuel_cost, disturbance, erosion_risk,
                                         # *_norm, pareto_score, feasible columns
front  = scorer.pareto_front(scored)    # non-dominated feasible subset

scorer.summary(scored)                  # print ranked table
scorer.plot_pareto(scored,
    x="fuel_cost_norm", y="erosion_risk_norm", color="disturbance_norm",
    save_path="pareto_2d.png")
scorer.plot_heatmap(scored,
    x_param="shoulder_yaw_deg", y_param="arm_reach_m",
    metric="pareto_score", save_path="heatmap.png")
scorer.plot_objectives_3d(scored, save_path="pareto_3d.png")
scorer.export_csv(scored, "sweep_scored.csv")
scorer.export_csv(scored, "pareto_front.csv", pareto_only=True)
```

**Feasibility gates** (hard filters applied before Pareto):
- Pipeline status must not be `FAIL`
- Thrust angle deviation ≤ `angle_budget_deg` (default 50°, per spec §5.1)

One-call convenience wrapper:

```python
from pareto_scoring import run_pareto_analysis

result = run_pareto_analysis(
    cases,
    manoeuvre_type="NSSK",
    angle_budget_deg=50.0,
    output_dir="pareto_output/"   # saves CSVs + 4 plots automatically
)
# result["all"]    → full scored list
# result["front"]  → Pareto-optimal subset
```

---

### `urdf_generator.py` — URDF generation

Generates a parametric URDF for the full assembly (client bus, panels, 4 antenna dishes, servicer, 3-DOF arm) with parabolic dish STL meshes.

```python
from urdf_generator import URDFGenerator
from plume_impingement_pipeline import StackConfig, RoboticArmGeometry

gen = URDFGenerator(StackConfig(), RoboticArmGeometry(shoulder_yaw_deg=45))
gen.save("urdf_output/")   # writes thruster_arm.urdf + meshes/*.stl
```

**Kinematic tree:** `world → LAR_interface → {client_bus, servicer_bus} → arm_base → J1(yaw) → J2(pitch) → J3(pitch) → thruster_frame`

**View in yourdfpy:**

```python
import yourdfpy, numpy as np

robot = yourdfpy.URDF.load("urdf_output/thruster_arm.urdf")
robot.update_cfg({
    "J1_shoulder_yaw": np.radians(45),
    "J2_elbow_pitch":  np.radians(20),
    "J3_wrist_pitch":  np.radians(-10),
})
robot.show()
```

**CLI:**

```bash
python3 urdf_generator.py urdf_output/ --yaw 45
python3 urdf_generator.py urdf_output/ --yaw 90 --no-mesh   # cylinder placeholders
```

---

### `geometry_visualizer.py` — 3-D matplotlib visualiser

Renders the arm, panel extent, thruster position, plume direction, and CoG using matplotlib. No extra dependencies.

```python
from geometry_visualizer import visualize_geometry
from plume_impingement_pipeline import StackConfig, RoboticArmGeometry

visualize_geometry(StackConfig(), RoboticArmGeometry(shoulder_yaw_deg=45))
```

---

### `openplume_exporter.py` — OpenPlume case export

Exports MARGINAL/CAUTION sweep cases into the input format expected by the OpenPlume high-fidelity plasma simulation tool.

```python
from openplume_exporter import export_openplume_cases

op_cases = pipeline.get_openplume_cases()
manifest = export_openplume_cases(op_cases, output_dir="openplume_cases/",
                                   thruster=thruster, material=material)
```

---

## Runner Scripts

### `pipeline_runner.py` — Quick parametric sweep

Runs a reduced sweep over `client_mass` × `mission_duration_yr`, generates heatmaps, exports results CSV, and prepares OpenPlume cases.

```bash
python3 pipeline_runner.py
# Outputs → pipeline_runner_output/
```

### `propellant_correlation_runner.py` — Sweep + time-resolved pipeline

Full two-stage analysis for the servicer configuration (530 kg dry + 140 kg Xe):
1. Screen 144 arm geometry cases (reach × yaw × link ratio)
2. Select MARGINAL/CAUTION candidates
3. Run `TimeResolvedErosion` per candidate at 30-day time steps
4. Export combined CSV + per-case erosion plots

```bash
python3 propellant_correlation_runner.py
# Outputs → propellant_correlation_results/
```

### `daily_on_summary_stats.py` — Manoeuvre plan statistics

Reads `pdsk_6month_maneuverPlan.xlsx` (Sheet: `Daily_ON_Summary`) and computes average firing events and firing time per day for each thruster direction, normalised over the 180-day plan.

```bash
python3 daily_on_summary_stats.py
```

---

## Typical Workflow

```
1. Configure thruster + material + stack parameters
        ↓
2. Run parametric sweep  (pipeline_runner.py or inline)
   PlumePipeline.run_sweep(cases)
        ↓
3a. Score and filter    (pareto_scoring.py)
    ParetoScorer.score() → .pareto_front() → .plot_*() → .export_csv()
        ↓
3b. Time-resolved analysis on candidates  (propellant_correlation_runner.py)
    TimeResolvedErosion.integrate_mission()
        ↓
4. Generate URDF for selected configurations  (urdf_generator.py)
   URDFGenerator.save() → yourdfpy / RViz visualisation
        ↓
5. Export borderline cases to OpenPlume  (openplume_exporter.py)
   export_openplume_cases()
```

---

## Arm Geometry Quick Reference

| Parameter | Value |
|-----------|-------|
| Link 1 (shoulder → elbow) | 1.12 m, 10 kg |
| Link 2 (elbow → wrist) | 1.40 m, 10 kg |
| Bracket (wrist → thruster) | 0.35 m, 3 kg |
| Total reach (extended) | 2.87 m |
| Total arm mass | 23 kg |
| J1 shoulder yaw range | 0° – 270° |
| J2 elbow pitch range | 0° – 235° |
| J3 wrist pitch range | −36° – +99° |
| Thrust angle budget | ≤ 50° deviation from ideal (soft limit) |

**Antenna dish parameters:**

| Dish | Face | Diameter | X offset | Z offset from bus centre |
|------|------|----------|----------|--------------------------|
| E1, E2 | +Y (East) | 2.2 m | ±0.75 m | +0.8 m |
| W1, W2 | −Y (West) | 2.5 m | ±0.75 m | +0.8 m |

All dishes are nadir-facing (aperture normal = +Z), 20 kg each, f/D = 0.5.
