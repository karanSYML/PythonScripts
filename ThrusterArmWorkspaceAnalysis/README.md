# Thruster Arm Workspace Analysis

Plasma plume impingement analysis pipeline for a GEO satellite life-extension servicing mission. A servicer spacecraft docks to an end-of-life client via the Launch Adapter Ring (LAR). A 3-DOF robotic arm positions an ion thruster for North-South and East-West station-keeping (NSSK/EWSK) while managing plume erosion risk on the client's solar panels and antenna dishes.

---

## Coordinate Frame

All geometry is expressed in the **LAR frame**:

| Axis | Direction |
|------|-----------|
| +X | North |
| +Y | East |
| +Z | Nadir (toward Earth) — client bus extends this way |
| −Z | Anti-Earth — servicer sits on this side |
| Origin | LAR mechanical docking interface (Z = 0) |

---

## Repository Structure

```
ThrusterArmWorkspaceAnalysis/
│
├── plume_impingement_pipeline.py      # Core library — all data classes + engines
├── arm_kinematics.py                  # FK, CoG + Jacobian, CoG IK (spatial kinematics)
├── arm_trajectory.py                  # LSPB joint trajectories + CoG-to-line planner
├── composite_mass_model.py            # Mission-epoch CoG model with propellant depletion
├── feasibility_cells.py               # Joint-space grid cells, F_kin / F_align / F_CoG filters
├── feasibility_map.py                 # Multi-epoch feasibility map generator
├── workspace_erosion_viz.py           # 3D workspace coloured by erosion proxy (matplotlib + Plotly)
├── geometry_visualizer.py             # Interactive 3D assembly visualiser (joint angle sliders)
├── propellant_erosion_correlation.py  # Time-resolved erosion + CoG migration integrator
├── pareto_scoring.py                  # Multi-objective Pareto scoring + plots
├── openplume_exporter.py              # Export cases for full OpenPlume simulation
├── urdf_generator.py                  # Parametric URDF + parabolic dish STLs
├── make_video.py                      # Render animation video (arm sweep + Pareto walk)
│
├── pipeline_runner.py                 # Runner: quick parametric sweep
├── propellant_correlation_runner.py   # Runner: sweep + time-resolved analysis
├── daily_on_summary_stats.py          # Manoeuvre plan statistics from Excel
│
├── feasibility_inputs.json            # All feasibility simulation parameters
├── stowed_config.json                 # Stowed geometry for visual verification
├── pdsk_6month_maneuverPlan.xlsx      # 6-month NSSK/EWSK manoeuvre plan
│
├── thruster_arm_urdf_spec.md          # Hardware + URDF specification document
├── feasibility_map_spec.md            # Feasibility map algorithm specification
├── ThrusterArmAnalysis.md             # Background analysis notes
├── CHANGELOG.md                       # Version history and change log
│
├── pipeline_runner_output/            # Outputs from pipeline_runner.py
├── propellant_correlation_results/    # Outputs from propellant_correlation_runner.py
├── pareto_output/                     # Outputs from pareto_scoring.py
└── urdf_output/                       # Generated URDF + STL mesh files
```

---

## Dependencies

```bash
pip install numpy matplotlib openpyxl plotly yourdfpy "pyglet<2" trimesh
```

| Package | Used for |
|---------|----------|
| `numpy` | All numerical computation |
| `matplotlib` | 3D geometry visualisation, heatmaps, Pareto plots, animation |
| `openpyxl` | Reading the manoeuvre plan Excel file |
| `plotly` | Interactive 3D workspace erosion browser |
| `yourdfpy` | Loading and viewing the generated URDF |
| `pyglet<2` | 3-D window backend for yourdfpy / trimesh |
| `trimesh` | Mesh handling for URDF viewer |

---

## Hardware Geometry

### Arm Reference Frame (TA Frame)
Origin at the root-hinge bracket centre. Axes align with the servicer body frame. TA-origin offset from servicer geometric centre: **(−0.12891, 0.31619, 0.356) m**.

### Hinge Positions in TA Frame

| Joint | Position (m) | Rotation Axis | Limit |
|-------|-------------|--------------|-------|
| Hinge 1 — Root (q0) | (0, 0, 0.25175) | (0, 0, −1) | 0° – 270° |
| Hinge 2 — Elbow (q1) | (−0.12, −1.10288, 0.15175) | (1, 0, 0) | 0° – 235° |
| Hinge 3 — Wrist (q2) | (−0.22, 0.4163, 0.1777) | (0, −0.3746, 0.9272) | −36° – +99° |
| Nozzle exit | (0.25977, 0.34953, 0.18370) | direction (0.1455, 0.9189, 0.3666) | — |

### Link Scalar Properties

| Segment | Length | Mass |
|---------|--------|------|
| Link 1 (Hinge 1 → Hinge 2) | 1.1139 m | 10 kg |
| Link 2 (Hinge 2 → Hinge 3) | 1.5227 m | 10 kg |
| Bracket (Hinge 3 → Nozzle) | 0.4844 m | 15 kg |
| **Total reach** | **3.1210 m** | **35 kg** |

### Hinge 1 in Servicer Body Frame
`arm_pivot_in_servicer_body()` → **(−0.12891, 0.31619, 0.60775) m**

### Antenna Dish Parameters

| Dish | Face | Diameter | X offset | Z offset |
|------|------|----------|----------|----------|
| E1, E2 | +Y (East) | 2.2 m | ±0.75 m | +0.8 m from bus centre |
| W1, W2 | −Y (West) | 2.5 m | ±0.75 m | +0.8 m from bus centre |

All dishes are nadir-facing (aperture normal ≈ +Z), 20 kg each, f/D = 0.5.

---

## Module Reference

### `plume_impingement_pipeline.py` — Core library

The central module. All other modules import from here.

**Data classes:**

```python
ThrusterParams      # thruster physics: Isp, discharge voltage, mass flow, thrust_N
MaterialParams      # target surface: coating thickness, sputter yield coefficients
StackConfig         # combined geometry: client + servicer buses, panels, 4 antennas
RoboticArmGeometry  # 3-DOF arm: hinge positions, axes, link vectors, joint limits, FK
ArmGeometry         # legacy simplified arm (single-parameter reach model)
OperationalParams   # mission ops: firing duration, firings/day, mission years
```

**`RoboticArmGeometry` — key methods:**

```python
arm = RoboticArmGeometry()

arm.arm_pivot_in_servicer_body()       # → (3,) Hinge-1 in servicer body frame [m]
arm.arm_reach()                        # → total reach L1+L2+bracket [m]
arm.arm_mass()                         # → total arm mass [kg]
arm.within_joint_limits(q0, q1, q2)   # → bool (radians)
arm.stowed_joint_angles_deg()          # → (0.0, 0.0, 0.0)

# General serial-chain FK using Rodrigues rotations
p_elbow, p_wrist, p_nozzle = arm.forward_kinematics(
    pivot, q0, q1, q2, servicer_yaw_deg=0.0)   # all in LAR frame [m]
```

**`StackConfig` — key methods:**

```python
stack = StackConfig(servicer_mass=750.0, client_mass=2800.0, ...)

stack.servicer_origin_in_lar_frame()   # → (3,) servicer geometric centre [m]
stack.stack_cog()                      # → (3,) static mass-weighted CoG [m]
stack.antenna_centers_in_lar_frame()   # → dict {E1, E2, W1, W2} → (3,) [m]
```

**`GeometryEngine` — analysis engine:**

```python
from plume_impingement_pipeline import GeometryEngine

ge = GeometryEngine(arm, stack, servicer_yaw_deg=-25.0)
ge.thruster_position()                  # thruster exit in LAR frame [m]
ge.thrust_direction()                   # unit vector toward stack CoG
ge.compute_flux_geometry()              # (distance, cos_theta) arrays for panel grid
ge.compute_antenna_flux_geometry()      # per-dish geometry {E1, E2, W1, W2}
ge.stack_cog_with_arm()                 # arm-inclusive CoG in LAR frame [m]
ge.thrust_metrics(thrust_N)             # deviation angles, moment arms, torques
ge.check_arm_collision_with_client_bus()
```

**`PlumePipeline` — sweep engine:**

```python
from plume_impingement_pipeline import PlumePipeline, CaseMatrixGenerator

pipeline = PlumePipeline(thruster, material)
results  = pipeline.run_sweep(cases, verbose=True)   # list of result dicts
pipeline.export_results_csv("output/results.csv")
op_cases = pipeline.get_openplume_cases()            # MARGINAL/CAUTION subset
```

**`run_sweep()` result dict keys (per case):**

| Key | Description |
|-----|-------------|
| `status` | `SAFE` / `CAUTION` / `MARGINAL` / `FAIL` |
| `max_erosion_um` | Worst panel erosion depth [µm] |
| `erosion_fraction` | `max_erosion_um / coating_thickness_um` |
| `ant_{E1,E2,W1,W2}_erosion_um` | Per-dish lifetime erosion [µm] |
| `nssk_deviation_deg` | Angle from ideal North/South thrust [°] |
| `ewsk_deviation_deg` | Angle from ideal East/West thrust [°] |
| `nssk_torque_Nm` | Disturbance torque for an ideal NSSK burn [N·m] |
| `ewsk_torque_Nm` | Disturbance torque for an ideal EWSK burn [N·m] |
| `cog_x/y/z` | Stack + arm CoG in LAR frame [m] |
| `thruster_pos_x/y/z` | Thruster exit position in LAR frame [m] |

---

### `arm_kinematics.py` — Spatial kinematics

Full 4×4 homogeneous transform chain, 6D spatial (Plücker) transforms, composite CoG + Jacobian, and CoG IK.

```python
from arm_kinematics import arm_fk_transforms, arm_cog_and_jacobian, arm_cog_ik

# Full transform chain
T = arm_fk_transforms(arm, pivot, q, servicer_yaw_deg=-25.0)
# Keys: T_world_j1, T_world_elbow, T_world_wrist, T_world_ee,
#       T_j1_elbow, T_elbow_wrist, T_wrist_ee

# Arm CoG + Jacobian (J maps joint velocities → CoG velocity)
cog, J = arm_cog_and_jacobian(arm, pivot, q, servicer_yaw_deg=-25.0)
# cog : (3,) arm centre-of-mass in LAR frame [m]
# J   : (3, 3) Jacobian  d(cog)/d(q) [m/rad]

# CoG IK — find joint angles that place arm CoG at a target
success, q_sol = arm_cog_ik(arm, pivot, q_init, p_cog_target,
                              pos_mask=[True, True, True],
                              tol=1e-3, max_iters=100)
```

---

### `arm_trajectory.py` — Joint trajectory planning

LSPB (trapezoidal velocity profile) and CoG-to-line trajectory planner.

```python
from arm_trajectory import lspb_trajectory, cog_to_line_trajectory

# Single-joint LSPB trajectory
success, traj, n_steps = lspb_trajectory(
    q0=0.0, qf=np.radians(90.0),
    dt=0.01, max_vel=0.1, max_acc=0.05)
# traj : (MAX_TRAJ_TIMESTEPS,) joint angle array [rad]
# n_steps : number of valid time steps

# Move arm CoG onto a target line, minimising joint travel
success, q_traj, n_steps = cog_to_line_trajectory(
    arm, pivot, q_init=np.zeros(3),
    line_point=np.array([0., 0., 1.]),
    line_dir=np.array([1., 0., 0.]),
    dt=0.01,
    joint_vel_limits=np.array([0.1, 0.1, 0.1]),
    joint_acc_limits=np.array([0.05, 0.05, 0.05]))
# q_traj : (3, MAX_TRAJ_TIMESTEPS) joint angle trajectories [rad]
```

---

### `composite_mass_model.py` — Mission-epoch CoG model

Tracks composite CoG from BOL to EOL as propellant depletes. Loaded from `feasibility_inputs.json`.

```python
from composite_mass_model import CompositeMassModel

mass = CompositeMassModel.from_json(stack=stack)  # reads feasibility_inputs.json

cog_bol = mass.p_CoG_LAR(0.0)           # CoG at BOL [m]
cog_eol = mass.p_CoG_LAR(1825.0)        # CoG at EOL (5 yr) [m]
migration = mass.cog_migration_magnitude(1825.0)  # total migration [m]
rate      = mass.p_CoG_LAR_rate(0.0)    # migration rate at BOL [m/day]
depletion = mass.propellant_exhausted_day()        # day propellant runs out

# Epoch sampling helper
spacing = mass.suggested_epoch_spacing(eps_CoG_m=0.05)  # max safe spacing [days]
```

---

### `feasibility_cells.py` — Joint-space feasibility filters

Builds the joint-space grid and evaluates per-cell geometric feasibility on a vectorised 50³ grid. Implements F_kin, F_align, F_CoG (F_plume deferred to v2).

```python
from feasibility_cells import (
    FeasibilityConfig, build_joint_grid,
    compute_static_cell_quantities, compute_F_kin,
    compute_alpha, compute_r_miss, compute_F_align, compute_F_CoG,
    binding_constraint_breakdown,
)

config  = FeasibilityConfig.from_json()         # reads feasibility_inputs.json
q0g, q1g, q2g = build_joint_grid(arm, resolution=config.grid_resolution)

# One-shot vectorised FK over entire grid
cq = compute_static_cell_quantities(arm, pivot, n_hat_ee, q0g, q1g, q2g,
                                     servicer_yaw_deg=-25.0)
# cq['p_nozzle'] : (N0,N1,N2,3) nozzle positions
# cq['t_hat']    : (N0,N1,N2,3) thrust unit vectors

F_kin   = compute_F_kin(arm, pivot, stack, servicer_yaw_deg, cq, q0g, q1g, q2g)
alpha   = compute_alpha(cq['t_hat'], d_hat)    # alignment angle (N0,N1,N2)
r_miss  = compute_r_miss(cq['p_nozzle'], cq['t_hat'], p_CoG)
F_align = compute_F_align(alpha, config.alpha_max_rad)
F_CoG   = compute_F_CoG(r_miss, config.eps_CoG_m)

breakdown = binding_constraint_breakdown(F_kin, F_align, F_CoG)
# → {'frac_fail_kin': ..., 'frac_fail_align': ..., 'frac_fail_CoG': ...}
```

**Station-keeping directions:**

| Key | LAR direction |
|-----|--------------|
| `'N'` | +X (North) |
| `'-N'` | −X (South) |
| `'E'` | +Y (East) |
| `'-W'` | −Y (West) |

---

### `feasibility_map.py` — Multi-epoch feasibility map

Orchestrates the full feasibility calculation across all epochs and SK directions.

```python
from feasibility_map import build_feasibility_maps, compute_pivot, print_summary

pivot = compute_pivot(arm, stack, servicer_yaw_deg=-25.0)

results = build_feasibility_maps(
    arm, mass_model, stack, pivot,
    n_hat_ee=np.array([0.1455, 0.9189, 0.3666]),
    servicer_yaw_deg=-25.0,
    config=config,
    directions=['N', '-N', 'E', '-W'],
    verbose=True,
)

print_summary(results)

# Access per-direction results
r = results['N']
r.F_persistent   # (N0,N1,N2) bool — cells feasible at all epochs
r.F_per_epoch    # (K,N0,N1,N2) bool — per-epoch feasibility
r.alpha_map      # (N0,N1,N2) float — alignment angle [rad], NaN outside F_kin
r.r_miss_per_epoch   # (K,N0,N1,N2) float — CoG miss distance [m]
r.diagnostics    # dict: epoch schedule, cell counts, binding constraints, N/S asymmetry
```

---

### `workspace_erosion_viz.py` — 3D workspace erosion map

Plots all collision-free nozzle positions in 3D space, coloured by the integrated relative ion flux (erosion proxy) on the solar panels. Also produces an interactive Plotly browser view.

```bash
python workspace_erosion_viz.py           # interactive: matplotlib + Plotly browser tab
python workspace_erosion_viz.py --save    # saves workspace_erosion.png + workspace_erosion.html
```

Colour scale (log-scaled plasma map): dark purple = low erosion risk → bright yellow = high.

---

### `geometry_visualizer.py` — Interactive 3D assembly visualiser

Renders the full docked assembly with live joint-angle sliders. No IK is performed — joint angles drive FK directly.

```bash
python geometry_visualizer.py            # interactive window
python geometry_visualizer.py --save     # saves geometry_verification.png
```

**Sliders:**

| Slider | Range | Initial |
|--------|-------|---------|
| Hinge 1 q0 | 0° – 270° | 0° |
| Hinge 2 q1 | 0° – 235° | 0° |
| Hinge 3 q2 | −36° – 99° | 0° |
| Panel Track α | −90° – 90° | 0° |
| Client mass | 1500 – 6000 kg | 2500 kg |
| Servicer mass | 700 – 800 kg | 750 kg |

Arm colour: **green** = OK · **orange** = joint limit exceeded · **red** = collision.

Configure via `stowed_config.json`:
- `servicer_yaw_deg` — servicer docking yaw [deg]
- `stowed_joint_angles_deg` — initial slider values [q0, q1, q2]
- `stowed_ee_unit_vector` — spec nozzle direction for stowed verification panel
- `pivot_position_lar_m` — override computed pivot (null = auto)

---

### `propellant_erosion_correlation.py` — Time-resolved analysis

Integrates erosion over the full mission with propellant depletion and CoG migration at configurable time steps.

```python
from propellant_erosion_correlation import (
    PropellantConfig, StationkeepingBudget,
    TimeResolvedErosion, plot_time_resolved_erosion
)

prop_config = PropellantConfig(tank_capacity_kg=160.0,
                                propellant_loaded_kg=140.0,
                                residual_fraction=0.03)
sk_budget = StationkeepingBudget(nssk_dv_per_year=50.0, ewsk_dv_per_year=2.0,
                                  nssk_manoeuvres_per_day=2, ewsk_manoeuvres_per_week=2)

integrator = TimeResolvedErosion(thruster, material, arm, stack, prop_config, sk_budget)
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
from pareto_scoring import ParetoScorer, run_pareto_analysis

scorer = ParetoScorer(manoeuvre_type="NSSK", angle_budget_deg=50.0)
scored = scorer.score(sweep_results)     # adds normalised scores + feasible flag
front  = scorer.pareto_front(scored)     # non-dominated feasible subset

scorer.summary(scored)
scorer.plot_pareto(scored, x="fuel_cost_norm", y="erosion_risk_norm",
                   color="disturbance_norm", save_path="pareto_2d.png")
scorer.plot_heatmap(scored, x_param="shoulder_yaw_deg", y_param="client_mass",
                    metric="pareto_score", save_path="heatmap.png")
scorer.plot_objectives_3d(scored, save_path="pareto_3d.png")
scorer.export_csv(scored, "pareto_front.csv", pareto_only=True)

# One-call convenience wrapper — runs scoring + saves CSVs + 4 plots
result = run_pareto_analysis(cases, manoeuvre_type="NSSK",
                              angle_budget_deg=50.0, output_dir="pareto_output/")
```

**Feasibility gates:** pipeline status must not be `FAIL`; thrust angle deviation ≤ `angle_budget_deg`.

---

### `urdf_generator.py` — URDF generation

Generates a parametric URDF for the full assembly with parabolic dish STL meshes.

```python
from urdf_generator import URDFGenerator
from plume_impingement_pipeline import StackConfig, RoboticArmGeometry

gen = URDFGenerator(StackConfig(), RoboticArmGeometry(shoulder_yaw_deg=45))
gen.save("urdf_output/")   # writes thruster_arm.urdf + meshes/*.stl
```

```bash
python urdf_generator.py urdf_output/ --yaw 45
python urdf_generator.py urdf_output/ --yaw 0 --no-mesh   # cylinder placeholders only
```

**Kinematic tree:** `world → LAR_interface → {client_bus, servicer_bus} → arm_base → J1(yaw) → J2(pitch) → J3(pitch) → thruster_frame`

View with yourdfpy:
```python
import yourdfpy, numpy as np
robot = yourdfpy.URDF.load("urdf_output/thruster_arm.urdf")
robot.update_cfg({"J1_shoulder_yaw": np.radians(45),
                  "J2_elbow_pitch":  np.radians(20),
                  "J3_wrist_pitch":  np.radians(-10)})
robot.show()
```

---

### `openplume_exporter.py` — OpenPlume case export

Exports MARGINAL/CAUTION pipeline cases into the OpenPlume high-fidelity plasma simulation input format.

```python
from openplume_exporter import export_openplume_cases

op_cases = pipeline.get_openplume_cases()
manifest = export_openplume_cases(op_cases, output_dir="openplume_cases/",
                                   thruster=thruster, material=material)
```

---

### `make_video.py` — Animation renderer

Renders a two-scene MP4 of the arm assembly.

- **Scene 1 — Arm Sweep (~15 s):** shoulder yaw sweeps 0° → 270°, camera orbits, arm colour/CoG/plume update each frame.
- **Scene 2 — Pareto Walk (~10 s):** steps through Pareto-optimal configurations with score annotations.

```bash
python make_video.py                          # both scenes → assembly_animation.mp4
python make_video.py --scene sweep            # scene 1 only
python make_video.py --scene pareto           # scene 2 only
python make_video.py --fps 24 --dpi 120       # quality tuning
python make_video.py --output my_video.mp4    # custom output path
```

---

## Runner Scripts

### `pipeline_runner.py` — Quick parametric sweep

Sweeps `shoulder_yaw_deg` × `client_mass` × `panel_tracking_deg`, scores with Pareto, generates heatmaps, exports OpenPlume cases.

```bash
python pipeline_runner.py
# → pipeline_runner_output/
#     results.csv, pareto_front.csv, heatmap_*.png, pareto_2d.png, openplume_cases/
```

### `propellant_correlation_runner.py` — Sweep + time-resolved pipeline

1. Screens 144 arm geometry cases (reach × yaw × link ratio)
2. Selects MARGINAL/CAUTION candidates
3. Runs `TimeResolvedErosion` per candidate at 30-day time steps
4. Exports combined CSV + per-case erosion plots

```bash
python propellant_correlation_runner.py
# → propellant_correlation_results/
```

### `daily_on_summary_stats.py` — Manoeuvre plan statistics

Reads `pdsk_6month_maneuverPlan.xlsx` (Sheet: `Daily_ON_Summary`) and prints average firing events and total firing time per day for each thruster direction, normalised over 180 days.

```bash
python daily_on_summary_stats.py
```

---

## Configuration Files

### `feasibility_inputs.json`

Master parameter file for the feasibility simulation pipeline. Edit this file before running `workspace_erosion_viz.py`, `feasibility_map.py`, or any feasibility analysis.

| Key | Description | Default |
|-----|-------------|---------|
| `thrust_N` | Nominal thruster thrust [N] | 0.054 |
| `isp_s` | Specific impulse [s] | 1485 |
| `nozzle_exit_direction_ee` | Nozzle exit unit vector in link-3 body frame | [0.1455, 0.9189, 0.3666] |
| `client_mass_kg` | Client satellite mass [kg] | 2800 |
| `servicer_dry_mass_kg` | Servicer dry mass [kg] | 600 |
| `propellant_mass_0_kg` | Initial propellant mass at BOL [kg] | 144 |
| `burn_cadence_hr_per_day` | Daily station-keeping burn time [hr/day] | 10 |
| `mission_duration_days` | Mission duration [days] | 1825 |
| `client_cog_lar_m` | Client CoG in LAR frame [m] | [0.03, 0.0, 1.72] |
| `tank_centroid_lar_m` | Propellant tank centroid in LAR frame [m] | [0.2, 0.0, −1.15] |
| `eps_CoG_m` | CoG miss distance threshold [m] | 0.05 |
| `alpha_max_deg` | Max thrust alignment angle [°] | 5.0 |
| `grid_resolution` | Joint-space grid [N_q0, N_q1, N_q2] | [50, 50, 50] |
| `epoch_schedule_days` | Mission epochs to evaluate [days from BOL] | [0, 456, 913, 1369, 1825] |

### `stowed_config.json`

Configuration for `geometry_visualizer.py`.

| Key | Description |
|-----|-------------|
| `servicer_yaw_deg` | Servicer docking yaw relative to LAR frame [°] |
| `stowed_joint_angles_deg` | Initial joint angles for sliders [q0, q1, q2] |
| `stowed_ee_unit_vector` | Spec nozzle direction for stowed verification |
| `pivot_position_lar_m` | Override pivot position (null = auto-computed) |

---

## Typical Workflow

```
1. Configure parameters
   feasibility_inputs.json  ←  thrust, mass, CoG, grid resolution, epochs
   stowed_config.json       ←  servicer yaw, initial joint angles
         ↓
2. Validate geometry visually
   python geometry_visualizer.py
   Drag Hinge 1/2/3 sliders, verify arm traces correct positions
         ↓
3. Inspect workspace + erosion risk
   python workspace_erosion_viz.py
   3D scatter: all feasible nozzle positions coloured by erosion proxy
         ↓
4. Run parametric sweep
   python pipeline_runner.py
   → results.csv, Pareto plots, OpenPlume cases
         ↓
5. Time-resolved analysis on candidates
   python propellant_correlation_runner.py
   → per-case erosion curves + combined CSV
         ↓
6. Multi-epoch feasibility map (Python API)
   from feasibility_map import build_feasibility_maps, compute_pivot, print_summary
   results = build_feasibility_maps(arm, mass_model, stack, pivot, n_hat_ee,
                                    servicer_yaw_deg=-25.0, config=config)
   print_summary(results)
         ↓
7. Generate URDF for selected configurations
   python urdf_generator.py urdf_output/ --yaw 45
         ↓
8. Render animation
   python make_video.py
```
