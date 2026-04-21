# Changelog — Feasibility Simulator

All changes to the feasibility simulator and workspace analysis pipeline.
Dates are in ISO 8601 format (YYYY-MM-DD).

---

## [Unreleased] — 2026-04-21

### Added — `feasibility_inputs.json`

New configuration file centralising all parameters that are not yet confirmed
or are expected to change between analysis runs. Eliminates hardcoded values
scattered across modules.

| Field | Purpose |
|---|---|
| `thrust_N`, `isp_s` | Hall-effect thruster parameters (54 mN / 1485 s, confirmed) |
| `nozzle_exit_direction_ee` | Fixed nozzle exit direction in EE frame — **placeholder** `[0,0,1]`, requires hardware confirmation |
| `client_mass_kg` | Client satellite mass |
| `servicer_dry_mass_kg` | Servicer dry-structure mass (propellant excluded) |
| `propellant_mass_0_kg` | Initial propellant mass at BOL |
| `burn_cadence_hr_per_day` | Nominal daily station-keeping burn time (10 hr/day = 5h N + 5h S) |
| `mission_duration_days` | Mission lifetime in days from BOL |
| `client_cog_lar_m` | Client CoG in LAR frame — null = geometric bus centre proxy |
| `servicer_dry_cog_lar_m` | Servicer dry CoG — null = servicer geometric centre |
| `tank_centroid_lar_m` | Propellant tank centroid — null = servicer geometric centre |
| `eps_CoG_m` | CoG miss-distance feasibility threshold [m] |
| `alpha_max_deg` | Thrust-alignment feasibility threshold [deg] |
| `grid_resolution` | Joint-space grid dimensions [N0, N1, N2] |
| `epoch_schedule_days` | Mission epochs at which feasibility maps are evaluated |

**Null-handling contract**: any position field set to `[null, null, null]` falls
back to the geometric centre of the relevant body (computed from `StackConfig`).
`servicer_dry_mass_kg` and `propellant_mass_0_kg` are required and raise
`ValueError` if null.

---

### Added — `composite_mass_model.py`

Mission-epoch composite centre-of-gravity (CoG) model for the servicer+client
stack. Implements spec §3 (mass and CoG evolution across mission lifetime).

**Frame convention**: LAR frame throughout (`+X` = North, `+Y` = East,
`+Z` = Nadir, origin = LAR docking interface). The spec's "composite body frame
B" uses the same origin with axes labelled differently (`x_B` = East = LAR `+Y`,
`y_B` = North = LAR `+X`); no coordinate transformation is required.

**Relationship to existing code**: `StackConfig.stack_cog()` is static (no
propellant depletion). `CompositeMassModel` is additive — it does not modify or
replace any existing function.

#### `CompositeMassModel` dataclass

| Method | Returns | Notes |
|---|---|---|
| `mdot()` | kg/s | Mass flow rate = `thrust_N / (isp_s · g₀)` |
| `cumulative_burn_time(epoch_days)` | s | Assumes constant daily cadence |
| `m_prop_from_burn_time(burn_time_s)` | kg | Clamped at 0 — extension point for non-uniform cadences |
| `m_prop(epoch_days)` | kg | Remaining propellant; clamped at 0 after exhaustion |
| `M(epoch_days)` | kg | Total composite mass |
| `p_CoG_LAR(epoch_days)` | (3,) m | Composite CoG in LAR frame |
| `p_CoG_LAR_rate(epoch_days)` | (3,) m/day | Analytical migration rate (spec §3.3); zero after exhaustion |
| `propellant_exhausted_day()` | days | Day on which propellant reaches 0 |
| `cog_migration_magnitude(epoch_days)` | m | `‖p_CoG(τ) − p_CoG(0)‖` |
| `cog_trajectory(epoch_schedule_days)` | (K,3) m | CoG positions at each epoch |
| `suggested_epoch_spacing(eps_CoG_m)` | days | Max spacing such that inter-epoch CoG migration < `eps_CoG_m` |
| `from_json(path, stack)` | `CompositeMassModel` | Factory; loads from `feasibility_inputs.json` |

**Single-tank assumption**: propellant CoG is fixed at `p_tank_LAR` regardless
of fill level. Valid for bladder/diaphragm tanks and symmetric arrangements.
For slosh or asymmetric multi-tank layouts, replace with a fill-level-dependent
position vector.

**Non-uniform cadence extension point**: compute cumulative burn time externally
and call `m_prop_from_burn_time(burn_time_s)` directly, bypassing
`cumulative_burn_time()`.

#### Spec test results (all pass)

1. Mass conservation across mission
2. CoG at BOL matches weighted formula
3. CoG migration at EOL order-of-magnitude (~22 cm for 300 kg propellant, 5-yr mission)
4. Analytical rate matches finite-difference to < 1e-15 m/day
5. r_miss = 0 when CoG lies exactly on thrust line
6. Cross-product and projection forms of r_miss agree to < 1e-12 m
7. alpha = 0° (aligned), 180° (antiparallel)
8. m_prop monotonically decreasing
9. Suggested epoch spacing diagnostic
10. Propellant exhaustion: m_prop = 0, rate = 0 after exhaustion day

---

### Added — `feasibility_cells.py`

Per-cell geometric quantities and feasibility filters for the joint-space grid.
Implements spec §4, §5.1–5.2 (F_kin, F_align, F_CoG). F_plume deferred (CEX
model not a v1 priority).

#### Frame and EE rotation convention

EE frame axes in LAR frame (derived analytically from `arm_fk_transforms` chain
`Rz(q0) @ Ry(-q1) @ Ry(-q2)`):

```
X_EE_LAR = d_bracket = cos(q1+q2)·u_rad + sin(q1+q2)·Ẑ   (along bracket)
Y_EE_LAR = [−sin(q0), cos(q0), 0]                          (yaw-perpendicular)
Z_EE_LAR = −sin(q1+q2)·u_rad + cos(q1+q2)·Ẑ              (perpendicular to bracket)
```

Thrust axis: `t̂_LAR = R_EE_LAR @ (−n̂^E) = −nx·X_EE − ny·Y_EE − nz·Z_EE`

Station-keeping directions in LAR frame:
`+N = +X_LAR`, `−N = −X_LAR`, `+E = +Y_LAR`, `−W = −Y_LAR`

#### `FeasibilityConfig` dataclass

Loaded from `feasibility_inputs.json` via `FeasibilityConfig.from_json()`.
Holds `eps_CoG_m`, `alpha_max_deg`, `grid_resolution`, `epoch_schedule_days`.

#### Functions

| Function | Description | Vectorized |
|---|---|---|
| `build_joint_grid(arm, resolution)` | Build `(q0, q1, q2)` meshgrids in radians over full joint limits | ✓ |
| `compute_static_cell_quantities(arm, pivot, n_hat_ee, q0g, q1g, q2g)` | FK for every cell: `p_elbow`, `p_wrist`, `p_nozzle`, `t_hat`, `d_bracket` | ✓ (no loops) |
| `compute_F_kin(arm, pivot, stack, yaw_deg, cq, q0g, q1g, q2g)` | Collision + joint-limit mask; wraps `arm_has_collision()` per cell | Iterative |
| `compute_alpha(t_hat_grid, d_hat)` | Alignment angle `arccos(clip(t̂·d̂, −1, 1))` [rad] | ✓ |
| `compute_F_align(alpha, alpha_max_rad)` | Boolean mask: `alpha ≤ alpha_max` | ✓ |
| `compute_r_miss(p_nozzle_grid, t_hat_grid, p_CoG_LAR)` | Miss-distance `‖(p_CoG − p_nozzle) × t̂‖` [m]; cross-product form | ✓ |
| `compute_F_CoG(r_miss, eps_CoG_m)` | Boolean mask: `r_miss ≤ eps_CoG` | ✓ |
| `compute_feasibility_epoch(F_kin, t_hat, p_nozzle, p_CoG, d_hat, config)` | `F_total = F_kin ∩ F_align ∩ F_CoG` for one direction and epoch | ✓ |
| `binding_constraint_breakdown(F_kin, F_align, F_CoG)` | Fraction of cells eliminated by each filter | ✓ |

**Performance note**: `compute_F_kin` calls `arm_has_collision()` once per cell.
At 50³ = 125,000 cells, expect 30–120 s (one-time cost; F_kin is epoch- and
direction-independent and is cached by the caller).

---

### Added — `feasibility_map.py`

Multi-epoch feasibility map generator. Implements spec §7 steps 5–7 and §8–9.
Orchestrates the full pipeline: grid construction → static cell quantities →
F_kin → per-epoch/per-direction filters → persistent feasibility →
annotations → diagnostics.

#### `FeasibilityMapResult` dataclass

| Field | Shape | Description |
|---|---|---|
| `direction` | str | `'N'`, `'-N'`, `'E'`, `'-W'` |
| `F_per_epoch` | `(K, N0, N1, N2)` bool | Feasible cells at each epoch |
| `F_persistent` | `(N0, N1, N2)` bool | Intersection across all epochs |
| `alpha_map` | `(N0, N1, N2)` float | Alignment angle [rad]; NaN outside F_kin |
| `r_miss_per_epoch` | `(K, N0, N1, N2)` float | Miss-distance [m]; NaN where infeasible |
| `diagnostics` | dict | Per-direction + global diagnostics (see below) |

#### `compute_pivot(arm, stack, servicer_yaw_deg)`

Helper computing arm pivot position in LAR frame. Pivot offset is expressed in
servicer body frame and rotated by `servicer_yaw_deg` into LAR. Mirrors the
logic in `geometry_visualizer.py::pivot_position()`.

#### `build_feasibility_maps(arm, mass_model, stack, pivot, n_hat_ee, servicer_yaw_deg, config, directions)`

Main entry point. Accepts all four SK directions by default. Returns
`dict[str, FeasibilityMapResult]`.

**Invariant verified**: `F_persistent ⊆ F_per_epoch[k]` for all `k`
(persistent set is a subset of every epoch's feasible set).

#### Diagnostics emitted (spec §9)

| Diagnostic | Location | Description |
|---|---|---|
| CoG migration trajectory | `global.cog_trajectory` | `‖p_CoG(τ) − p_CoG(0)‖` vs τ across all epochs |
| Per-direction cell counts | `per_direction.cell_counts_per_epoch` | Feasibility erosion across mission life |
| First-dropout epoch | `per_direction.first_dropout_epoch` | Per-cell epoch at which cell first fails; `inf` = persistent, `NaN` = infeasible at BOL |
| Binding-constraint breakdown | `per_direction.binding_constraint` | Fraction eliminated by F_kin / F_align / F_CoG at each epoch |
| N/S asymmetry | `global.ns_asymmetry` | `|F_N(τ)| / |F_-N(τ)|` ratio per epoch |
| Suggested epoch spacing | `global.suggested_epoch_spacing_days` | From `CompositeMassModel.suggested_epoch_spacing(eps_CoG_m)` |

#### Known limitation — empty feasible sets with current default parameters

With `client_cog_lar_m = null` (geometric centre proxy at Z = 1.64 m) and
`eps_CoG_m = 0.05 m`, all four directions return empty feasible sets. Root cause:
the arm's maximum nozzle Z is 1.55 m (pivot Z −0.35 m + max vertical reach
L2 + L3 = 1.90 m), leaving a 9 cm gap to the proxy CoG — larger than the 5 cm
threshold. This is a **configuration issue, not a code defect**. Resolution:
set `client_cog_lar_m` to the actual client CoG in `feasibility_inputs.json`,
or relax `eps_CoG_m`.

---

### Added — `workspace_erosion_viz.py`

3D workspace visualisation for the thruster arm, coloured by plume-erosion
proxy at each reachable end-effector (nozzle) position.

**Erosion proxy**: integrated relative ion flux over both solar-panel wings,
`proxy = Σ cos^n(θᵢ) / rᵢ²`, using the same cosine-power plume model as
`geometry_visualizer.py::relative_flux()`.

**Thrust direction convention**: nozzle → composite CoG (same as
`geometry_visualizer.py`), not the fixed `n_hat_ee` from `feasibility_inputs.json`.
This matches the operational aim-point used during burns.

#### Outputs

| Flag | PNG | HTML |
|---|---|---|
| `--save` | `workspace_erosion.png` (180 dpi, matplotlib) | `workspace_erosion.html` (3.8 MB, CDN-linked Plotly) |
| *(none)* | matplotlib interactive window | Plotly in browser tab |

#### Matplotlib figure (`workspace_erosion.png`)

- Left panel: 3D scatter of 20,141 collision-free EE positions coloured by
  `log₁₀(proxy)` on the `plasma` colormap. Client bus, servicer bus, panel
  outlines, LAR ring, pivot, and CoG drawn as context.
- Right panel: XY top-down projection (azimuthal footprint, clipped to ±4 m).
- Cyan arm = min-erosion pose; gold arm = max-erosion pose.
- Info box: grid stats, min/max joint angles and proxy values.

#### Plotly figure (`workspace_erosion.html`)

- Full 3D interactive scatter. Hover on any point to see EE (x, y, z),
  joint angles (q0, q1, q2), raw proxy, and log₁₀ value.
- Geometry traces (client bus, servicer, panel outlines, LAR ring, panel
  sample points) toggleable via legend clicks.
- Min/max-erosion arm configurations drawn as 3D line traces with joint
  markers.
- Colorbar on right; scene camera at `eye=(−1.4, −1.6, 0.8)`.

#### Key findings from 40³ grid run

| Metric | Value |
|---|---|
| Grid cells | 64,000 |
| Collision-free (F_kin pass) | 20,141 (31.5%) |
| Min-erosion EE | (−0.33, +0.24, −0.39) m — arm folded toward servicer, thruster pointing anti-nadir |
| Max-erosion EE | (+1.59, +0.24, +1.54) m — arm extended North at mid-pitch, plume sweeping panel face |
| Erosion range | 0 → 2.49 (arbitrary flux units) |

The azimuthal pattern confirms that poses in the ±Y (East/West) sector carry
lower erosion risk, while poses in the ±X (North/South) sector — where the
solar panels extend — carry the highest risk. This is consistent with N/S
station-keeping being the dominant burn regime.

---

## Open items / next steps

- [ ] Set `client_cog_lar_m` to actual client CoG → enables non-empty F_CoG feasible sets
- [ ] Confirm `nozzle_exit_direction_ee` with hardware drawings
- [ ] Confirm `servicer_dry_mass_kg` / `propellant_mass_0_kg` split
- [ ] Confirm `tank_centroid_lar_m` from structural model
- [ ] Implement F_plume (primary beam cone check against no-impingement zones)
- [ ] Build `feasibility_map_runner.py` — CLI wrapper that runs the full pipeline, saves results to disk, and calls `print_summary()`
- [ ] Add CEX halo model (deferred from v1)
- [ ] Vectorise `compute_F_kin` for performance at 50³+ grids
