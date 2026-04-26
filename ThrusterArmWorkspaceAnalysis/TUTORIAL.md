# Tutorial: Running the Thruster Arm Simulation Pipeline

This guide walks through every module in the repository in dependency order. Start at Step 1 and work forward; each step builds on the outputs of the previous one.

---

## Prerequisites

Install all dependencies before starting:

```bash
pip install numpy matplotlib openpyxl plotly yourdfpy "pyglet<2" trimesh
```

Confirm the environment works:

```bash
python -c "import numpy, matplotlib, plotly; print('Dependencies OK')"
```

---

## Step 0 — Edit the configuration files

All simulation parameters live in two JSON files. Review and edit them before running anything.

### `feasibility_inputs.json`

This file drives the feasibility simulation, workspace erosion map, and composite mass model.

```json
{
  "thrust_N": 0.054,
  "isp_s": 1485.0,
  "nozzle_exit_direction_ee": [0.1455, 0.9189, 0.3666],

  "client_mass_kg": 2800.0,
  "servicer_dry_mass_kg": 600.0,
  "propellant_mass_0_kg": 144.0,
  "burn_cadence_hr_per_day": 10.0,
  "mission_duration_days": 1825,

  "client_cog_lar_m": [0.03, 0.0, 1.72],
  "tank_centroid_lar_m":  [0.2, 0.0, -1.15],

  "eps_CoG_m": 0.05,
  "alpha_max_deg": 5.0,
  "grid_resolution": [50, 50, 50],
  "epoch_schedule_days": [0, 456, 913, 1369, 1825]
}
```

**Key parameters to adjust:**

- `grid_resolution` — set to `[20, 20, 20]` for fast exploratory runs, `[50, 50, 50]` for production (~125 k cells, ~30 s on modern hardware).
- `epoch_schedule_days` — must include `0` (BOL). Add intermediate epochs if the CoG migration rate is high early in the mission.
- `eps_CoG_m` — CoG miss distance threshold. 5 cm is a reasonable starting point; tighten to 2 cm for final analysis.
- `alpha_max_deg` — maximum allowed thrust-to-SK-direction misalignment. 5° is the spec limit.

### `stowed_config.json`

This file controls `geometry_visualizer.py` — the interactive 3D tool.

```json
{
  "servicer_yaw_deg": -25.0,
  "stowed_joint_angles_deg": [0.0, 0.0, 0.0],
  "stowed_ee_unit_vector": [0.1455, 0.9189, 0.3666],
  "pivot_position_lar_m": [null, null, null]
}
```

- `servicer_yaw_deg` — the servicer's docking yaw relative to the LAR frame. `-25.0` is the nominal configuration.
- `stowed_joint_angles_deg` — initial position of the three joint sliders when the visualiser opens.
- `pivot_position_lar_m` — leave as `[null, null, null]` to let the code compute the pivot from geometry; fill in if you have a direct measurement.

---

## Step 1 — Validate forward kinematics visually

Run the interactive 3D visualiser. This validates that the FK implementation produces correct arm geometry before running any analysis.

```bash
python geometry_visualizer.py
```

**What you see:**

- Grey wireframe box: client bus (LAR interface at Z = 0, nadir end at +Z).
- Steel-blue box: servicer (docked below, rotated by `servicer_yaw_deg`).
- Gold rectangles: client solar panels (±X wings).
- Silver discs: four antenna reflectors (E1, E2, W1, W2).
- Coloured arm: three links from the root pivot.
- Purple star (★): nozzle/thruster exit point.
- Red diamond: composite stack CoG.
- Black arrow: thrust vector (thruster → CoG direction).
- Translucent red cone: plume divergence cone.

**Using the sliders:**

| Slider | What it controls |
|--------|-----------------|
| Hinge 1 q0 | Root joint yaw about −Z axis (0° – 270°) |
| Hinge 2 q1 | Elbow joint rotation about X axis (0° – 235°) |
| Hinge 3 q2 | Wrist joint rotation about tilted axis (−36° – 99°) |
| Panel Track α | Solar panel sun-tracking rotation angle |
| Client mass | Client satellite mass (affects CoG position) |
| Servicer mass | Servicer spacecraft mass |

**Things to check:**

1. At q0=q1=q2=0 (stowed), the "STOWED VERIFY" section in the status panel should show ≈0° angular difference between the actual bracket direction and `stowed_ee_unit_vector`.
2. Sweep q0 from 0° to 270° — the arm should rotate around the servicer in a roughly horizontal arc.
3. The arm turns **orange** when any joint exceeds its limit; **red** when it intersects the client bus.
4. The CoG marker moves as mass slider values change.

**Save a static snapshot:**

```bash
python geometry_visualizer.py --save
# → geometry_verification.png
```

---

## Step 2 — Inspect the 3D workspace and erosion proxy

The workspace erosion visualiser computes FK over the full joint-space grid, filters out collisions, and colours each reachable nozzle position by the integrated ion flux on the solar panels.

```bash
python workspace_erosion_viz.py
```

This opens two windows simultaneously:

1. **Matplotlib** — 3D scatter (nozzle positions, plasma colour = erosion proxy) plus an XY top-down view showing azimuthal workspace coverage.
2. **Plotly** — interactive browser tab with the same 3D scatter; rotate, zoom, and hover over individual cells to read (q0, q1, q2) and erosion proxy value.

**Save outputs instead of showing interactively:**

```bash
python workspace_erosion_viz.py --save
# → workspace_erosion.png   (matplotlib)
# → workspace_erosion.html  (Plotly — open in any browser)
```

**What the colours mean:**

| Colour | Meaning |
|--------|---------|
| Dark purple | Low erosion proxy — safe arm pose |
| Yellow/green | High erosion proxy — high ion flux on panels |
| Not shown | Collision-flagged poses (removed from plot) |

**What to look for:**

- Overall workspace envelope: does the reachable volume cover the required SK directions?
- High-erosion zones: which q0 ranges point the plume toward the panels?
- Azimuthal coverage gaps (XY top-down view): are there SK directions with no feasible poses?

The grid resolution is set in `feasibility_inputs.json` → `grid_resolution`. Use `[20, 20, 20]` for a quick overview, `[50, 50, 50]` for the full workspace.

---

## Step 3 — Run the composite mass model

Before running the multi-epoch feasibility map, verify the CoG trajectory is physically sensible.

```python
import numpy as np
from plume_impingement_pipeline import StackConfig
from composite_mass_model import CompositeMassModel

stack = StackConfig(servicer_mass=744.0, client_mass=2800.0,
                    servicer_bus_x=0.9, servicer_bus_y=1.5, servicer_bus_z=0.8,
                    client_bus_x=2.3, client_bus_y=3.0, client_bus_z=5.0,
                    panel_span_one_side=16.0, panel_width=2.5, lar_offset_z=0.05)

mass = CompositeMassModel.from_json(stack=stack)

epochs = [0, 456, 913, 1369, 1825]
print("Epoch [day]  |  CoG (X, Y, Z) [m]  |  Migration [cm]")
for tau in epochs:
    cog = mass.p_CoG_LAR(tau)
    mig = mass.cog_migration_magnitude(tau) * 100
    print(f"  {tau:5.0f}       |  ({cog[0]:+.3f}, {cog[1]:+.3f}, {cog[2]:+.3f})  |  {mig:.2f}")

print(f"\nPropellant exhausted: day {mass.propellant_exhausted_day():.0f}")
print(f"Suggested epoch spacing for eps=5 cm: {mass.suggested_epoch_spacing(0.05):.0f} days")
```

**Expected output (with default inputs):** CoG drifts in +Z (nadir) as propellant depletes from the anti-nadir tank. EOL migration should be a few centimetres.

---

## Step 4 — Build the multi-epoch feasibility map

The feasibility map evaluates every cell in the joint-space grid at each mission epoch. This is the core analysis for arm pose selection.

```python
import numpy as np
from plume_impingement_pipeline import RoboticArmGeometry, StackConfig
from composite_mass_model import CompositeMassModel
from feasibility_cells import FeasibilityConfig
from feasibility_map import build_feasibility_maps, compute_pivot, print_summary

# --- Geometry setup ---
arm   = RoboticArmGeometry()
stack = StackConfig(
    servicer_mass=744.0, client_mass=2800.0,
    servicer_bus_x=0.9, servicer_bus_y=1.5, servicer_bus_z=0.8,
    client_bus_x=2.3, client_bus_y=3.0, client_bus_z=5.0,
    panel_span_one_side=16.0, panel_width=2.5, lar_offset_z=0.05,
)
SERVICER_YAW_DEG = -25.0

# --- Pivot and mass model ---
pivot = compute_pivot(arm, stack, servicer_yaw_deg=SERVICER_YAW_DEG)
mass  = CompositeMassModel.from_json(stack=stack)

# --- Feasibility configuration ---
config  = FeasibilityConfig.from_json()
n_hat_ee = np.array([0.1455, 0.9189, 0.3666])  # from feasibility_inputs.json

# --- Run ---
results = build_feasibility_maps(
    arm, mass, stack, pivot, n_hat_ee,
    servicer_yaw_deg=SERVICER_YAW_DEG,
    config=config,
    directions=['N', '-N', 'E', '-W'],
    verbose=True,
)

print_summary(results)
```

**Console output** (verbose=True) shows per-epoch cell counts, constraint breakdown (what fraction of cells fail each filter), and global CoG trajectory statistics.

**Using the results:**

```python
# Cells feasible at every epoch for North SK
r_N = results['N']
print(f"Persistent feasible cells (North): {r_N.F_persistent.sum()}")

# Which cells drop out earliest?
first_out = r_N.diagnostics['first_dropout_epoch']
# first_out[i,j,k] = first epoch (days) at which cell (i,j,k) becomes infeasible
#                  = inf  if the cell is in F_persistent
#                  = nan  if the cell was already infeasible at BOL

# Binding constraint at the first epoch
bd = r_N.diagnostics['binding_constraint'][0]
print(f"BOL — kin fails {bd['frac_fail_kin']*100:.1f}%  "
      f"align fails {bd['frac_fail_align']*100:.1f}%  "
      f"CoG fails {bd['frac_fail_CoG']*100:.1f}%")

# North/South asymmetry
ns = r_N.diagnostics['global']['ns_asymmetry']
print(f"N÷−N count ratio: BOL={ns['ratio_N_over_mN'][0]:.2f}  EOL={ns['ratio_N_over_mN'][-1]:.2f}")
```

**Tuning tips:**

- If `F_persistent` is empty for a direction, try relaxing `alpha_max_deg` (e.g. 10°) or `eps_CoG_m` (e.g. 10 cm) in `feasibility_inputs.json`.
- Increase `grid_resolution` to `[80, 80, 80]` for a finer feasibility boundary; run time scales as N³.
- Add more `epoch_schedule_days` near BOL if the CoG migration rate is high (check `suggested_epoch_spacing`).

---

## Step 5 — Run the parametric sweep

The pipeline runner sweeps shoulder yaw, client mass, and panel tracking angle, then scores and plots results.

```bash
python pipeline_runner.py
```

**Outputs in `pipeline_runner_output/`:**

| File | Description |
|------|-------------|
| `results.csv` | Full sweep results for all cases |
| `pareto_front.csv` | Non-dominated (Pareto-optimal) cases |
| `heatmap_*.png` | Erosion heatmaps (yaw × client mass) |
| `pareto_2d.png` | Pareto scatter: fuel cost vs erosion risk |
| `heatmap_score.png` | Composite Pareto score over yaw × client mass |
| `openplume_cases/` | Input files for MARGINAL/CAUTION cases |

**Running inline (library API):**

```python
from plume_impingement_pipeline import (
    ThrusterParams, MaterialParams, PlumePipeline, CaseMatrixGenerator
)
import numpy as np

thruster = ThrusterParams(name="SPT-100-like", thrust_N=0.054, isp=1485.0)
material = MaterialParams(name="Silver_interconnect", thickness_um=15.0)

pipeline = PlumePipeline(thruster, material)
gen      = pipeline.generator

gen.set_param_range("shoulder_yaw_deg", np.arange(0.0, 271.0, 15.0))
gen.set_param_range("client_mass", [2500.0, 2800.0, 3200.0])

fixed = {
    "servicer_mass": 744.0,
    "arm_reach_m": 2.62,
    "link_ratio": 0.444,
    "panel_span_one_side": 16.0,
    "firing_duration_s": 25000.0,
    "mission_duration_yr": 5.0,
}

cases   = gen.generate_reduced_matrix(fixed, sweep_params=["shoulder_yaw_deg", "client_mass"])
results = pipeline.run_sweep(cases, verbose=True)

# Filter to best cases
ok   = [r for r in results if r["status"] in ("SAFE", "CAUTION")]
best = min(ok, key=lambda r: r["max_erosion_um"])
print(f"Best case: yaw={best['shoulder_yaw_deg']:.0f}°  "
      f"erosion={best['max_erosion_um']:.2f} µm  status={best['status']}")
```

---

## Step 6 — Pareto scoring

Score any list of sweep results on fuel cost, disturbance torque, and erosion risk.

```python
from pareto_scoring import ParetoScorer

scorer = ParetoScorer(manoeuvre_type="NSSK", angle_budget_deg=50.0)
scored = scorer.score(results)
front  = scorer.pareto_front(scored)

scorer.summary(scored)     # prints ranked table to console

scorer.plot_pareto(scored,
    x="fuel_cost_norm", y="erosion_risk_norm", color="disturbance_norm",
    save_path="pareto_2d.png")

scorer.plot_objectives_3d(scored, save_path="pareto_3d.png")

scorer.export_csv(scored, "all_scored.csv")
scorer.export_csv(scored, "pareto_front.csv", pareto_only=True)
```

The `pareto_score` column in the scored DataFrame is the sum of the three normalised objectives. Lower = better.

---

## Step 7 — Time-resolved erosion analysis

Run this on the Pareto-optimal subset to see how erosion and CoG evolve across the mission.

```bash
python propellant_correlation_runner.py
# → propellant_correlation_results/
```

Or run on a specific arm configuration inline:

```python
from plume_impingement_pipeline import (
    ThrusterParams, MaterialParams, RoboticArmGeometry, StackConfig
)
from propellant_erosion_correlation import (
    PropellantConfig, StationkeepingBudget, TimeResolvedErosion, plot_time_resolved_erosion
)

thruster = ThrusterParams(name="SPT-100-like", thrust_N=0.054, isp=1485.0)
material = MaterialParams(name="Silver_interconnect", thickness_um=15.0)

# RoboticArmGeometry — hardware-confirmed geometry; shoulder_yaw_deg sets the burn azimuth
arm   = RoboticArmGeometry(shoulder_yaw_deg=45.0)
stack = StackConfig(servicer_mass=744.0, client_mass=2800.0,
                    servicer_bus_x=0.9, servicer_bus_y=1.5, servicer_bus_z=0.8,
                    client_bus_x=2.3, client_bus_y=3.0, client_bus_z=5.0,
                    panel_span_one_side=16.0, panel_width=2.5, lar_offset_z=0.05)

prop_config = PropellantConfig(tank_capacity_kg=160.0,
                                propellant_loaded_kg=144.0,
                                residual_fraction=0.03)
sk_budget = StationkeepingBudget(
    nssk_dv_per_year=50.0, ewsk_dv_per_year=2.0,
    nssk_manoeuvres_per_day=2, ewsk_manoeuvres_per_week=2)

integrator = TimeResolvedErosion(thruster, material, arm, stack, prop_config, sk_budget)
result = integrator.integrate_mission(mission_years=5.0, time_step_days=30.0)

summary = result["summary"]
print(f"Limiting factor: {summary['limiting_factor']}")
print(f"Max erosion at EOL: {summary['max_erosion_um']:.2f} µm "
      f"({summary['erosion_fraction']*100:.1f}% of coating)")
print(f"CoG Z shift: {summary['cog_shift_z_mm']:.1f} mm")

plot_time_resolved_erosion(result, output_dir="my_output/")
```

---

## Step 8 — Generate URDF

Generate the URDF for the selected arm configuration and load it for visualisation or RViz.

```bash
python urdf_generator.py urdf_output/ --yaw 45
```

For a quick check without generating full parabolic dish meshes:

```bash
python urdf_generator.py urdf_output/ --yaw 45 --no-mesh
```

View interactively:

```python
import yourdfpy, numpy as np

robot = yourdfpy.URDF.load("urdf_output/thruster_arm.urdf")
robot.update_cfg({
    "J1_shoulder_yaw": np.radians(45.0),
    "J2_elbow_pitch":  np.radians(20.0),
    "J3_wrist_pitch":  np.radians(-10.0),
})
robot.show()
```

---

## Step 9 — Render animation

> **Known issue — update required before running.**
> `make_video.py` imports `redraw` from `geometry_visualizer`, but that function was
> refactored to `update_dynamic_scene()` in the joint-angle slider update (see CHANGELOG.md).
> The script will fail on import until it is updated to call `update_dynamic_scene()` with
> a `state` dict containing `q0_deg / q1_deg / q2_deg` instead of `yaw_deg / reach_m / link_ratio`.
> The CLI flags and scene structure described below are correct.

Render a video showing the arm sweeping through its full range and stepping through the Pareto-optimal configurations.

```bash
# Both scenes (arm sweep + Pareto walk) → assembly_animation.mp4
python make_video.py

# Arm sweep only
python make_video.py --scene sweep

# Pareto walk only
python make_video.py --scene pareto

# Higher quality
python make_video.py --fps 24 --dpi 150 --output high_quality.mp4
```

**Scene 1 — Arm Sweep (~10 s at 24 fps):** sweeps Hinge-1 (q0) from 0° to 270°, camera orbits the assembly.

**Scene 2 — Pareto Walk:** steps through the Pareto-optimal configurations from a compact internal sweep, dwelling ~1.5 s per configuration with score annotations overlaid.

---

## Step 10 — Analyse the manoeuvre plan

Read firing statistics from the 6-month NSSK/EWSK manoeuvre plan spreadsheet.

```bash
python daily_on_summary_stats.py
```

Reads `pdsk_6month_maneuverPlan.xlsx` (sheet: `Daily_ON_Summary`) and prints average events per day and average firing time per day for each thruster direction.

---

## Step 11 — Export to OpenPlume

Export MARGINAL and CAUTION cases from any sweep run to the OpenPlume high-fidelity plasma simulation format.

```python
from openplume_exporter import export_openplume_cases

op_cases = pipeline.get_openplume_cases()   # filters results to MARGINAL/CAUTION
print(f"{len(op_cases)} cases for OpenPlume")

manifest = export_openplume_cases(
    op_cases,
    output_dir="openplume_cases/",
    thruster=thruster,
    material=material,
)
```

---

## Common Issues

### `ModuleNotFoundError: No module named 'plotly'`

```bash
pip install plotly
```

### `workspace_erosion_viz.py` hangs or shows a blank Plotly tab

The Plotly figure opens in the default browser. If the browser tab is blank, try:

```python
import plotly.io as pio
pio.renderers.default = "browser"   # add to the top of the script
```

### Feasibility map takes too long

Reduce the grid resolution in `feasibility_inputs.json`:

```json
"grid_resolution": [20, 20, 20]
```

A 20³ grid (8 000 cells) runs in a few seconds. A 50³ grid (125 000 cells) takes ~30 seconds. The `compute_F_kin` step is the bottleneck — it loops over cells sequentially.

### `F_persistent` is empty for all directions

Constraints are too tight for the current geometry. Try:

1. Increase `alpha_max_deg` from 5 to 10 degrees.
2. Increase `eps_CoG_m` from 0.05 to 0.10 metres.
3. Check that `servicer_yaw_deg` in `stowed_config.json` matches the yaw used in `SERVICER_YAW_DEG` inside `workspace_erosion_viz.py`.

### Pivot position looks wrong in the visualiser

1. Verify `servicer_yaw_deg` in `stowed_config.json` is correct (nominal: −25°).
2. Set `pivot_position_lar_m` to a confirmed value if available, otherwise leave as `[null, null, null]` for auto-computation.
3. At q=0 the pivot in LAR frame should be approximately: `servicer_origin + Rz(-25°) @ [-0.12891, 0.31619, 0.60775]`.

### URDF viewer shows wrong link orientations

The URDF uses the stowed-config geometry (q=0). Joint axes in the URDF definition may not match the simulation axes directly because URDF uses a different convention. Use the geometry visualiser (Step 1) as the ground truth for FK validation.

---

## Quick-Start Checklist

```
[ ] pip install numpy matplotlib openpyxl plotly yourdfpy "pyglet<2" trimesh
[ ] Edit feasibility_inputs.json — confirm masses, CoG positions, burn cadence
[ ] Edit stowed_config.json — confirm servicer yaw
[ ] python geometry_visualizer.py — check stowed pose matches spec
[ ] python workspace_erosion_viz.py --save — confirm workspace envelope
[ ] python pipeline_runner.py — run full parametric sweep
[ ] python propellant_correlation_runner.py — time-resolved analysis on best cases
[ ] python make_video.py — render animation
```
