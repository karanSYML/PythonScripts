# Changelog — Feasibility Simulator

All changes to the feasibility simulator and workspace analysis pipeline.
Dates are in ISO 8601 format (YYYY-MM-DD).

---

## [2026-05-01] — GEO space environment: substorm charging + thermal-cycling lifetime

### Added — `plume_impingement_pipeline.py`: GEO environment wiring

Imported `GEOEnvironment` and `ThermalCycling` from `sputter_erosion`.

**`_sheath_from_geo_env(geo_env, string_voltage=100.0, Te_local=2.0) → SheathModel`**
Helper that translates a `GEOEnvironment` instance into the near-thruster `SheathModel`:

| Scenario | `floating_potential` |
|---|---|
| Quiescent GEO | `−3.5 × Te_local` = −7 V  (local near-thruster sheath, Te_local = 2 eV) |
| Substorm worst-case | `geo_env.floating_potential()` = −8 000 V  (whole spacecraft charges to −8 kV) |

The quiescent ambient GEO potential (−3 500 V at Te = 1 000 eV) is *not* applied
during quiescent mode because it represents the spacecraft body potential, which is
physically separate from the local near-thruster sheath used by `SheathModel`.

**`_build_sputter_geometry`** gains an optional `geo_env` parameter (default `None`).
When supplied, the sheath is built via `_sheath_from_geo_env`; otherwise the
pre-existing −15 V hardcoded sheath is used (backward compatible).

**`_hifi_erosion_metrics`** gains three optional parameters — `sat_geo`, `thermal`,
`integrator` — and now returns a `coupled_life_factor` key:
- When all three are provided, calls `LifetimeAnalysis(integrator, thermal).life_prediction(
  MissionProfile([FiringPhase("mission", sat_geo, total_s)]))`
  and stores `min(coupled_life_factor)` across all interconnects.
- `coupled_life_factor = fraction_remaining × thermal_life_factor` (Coffin-Manson).
- Default `None` when not computed (backward compatible).

`PlumePipeline.run_sweep` high-fidelity path now passes all three to
`_hifi_erosion_metrics`, so every HF result dict includes `coupled_life_factor`.

---

### Added — `workspace_erosion_viz.py`: substorm HF comparison + lifetime output

Imported `GEOEnvironment`, `ThermalCycling`, `LifetimeAnalysis`, `MissionProfile`,
`FiringPhase` from `sputter_erosion`.

New module-level constants:

```python
_GEO_QUIESCENT      = GEOEnvironment()                    # Te=1 keV, quiescent
_GEO_SUBSTORM       = GEOEnvironment(in_substorm=True)    # −8 kV worst-case
_HF_THERMAL         = ThermalCycling()                    # 90 cycles/yr, ΔT=120 K
_HF_MISSION_YEARS   = 7.0
_HF_FIRINGS_PER_DAY = 1.0
```

**`_sheath_from_geo_env(geo_env, ...)`** — same helper as in the pipeline.

**`_build_hifi_sat_geo(p_nozzle, plume_dir, panel_pts, hall_plume, stack, geo_env=None)`** —
extracted from `_hifi_for_pose` to avoid code duplication; builds `SatelliteGeometry`
for any environment scenario.

**`_hifi_for_pose`** gains `geo_env=None` parameter; delegates sat_geo construction to
`_build_hifi_sat_geo`.  Legacy call sites (no `geo_env`) continue to use −15 V.

**`_hifi_coupled_life_factor(p_nozzle, plume_dir, panel_pts, hall_plume, stack)`** —
runs `LifetimeAnalysis(_HF_INTEGRATOR, _HF_THERMAL)` over a full-mission firing
schedule (`HF_FIRING_S × _HF_FIRINGS_PER_DAY × 365.25 × _HF_MISSION_YEARS`); returns
worst-case `coupled_life_factor` across all interconnects.

**HF top-K loop** now evaluates three quantities per pose:

| Column | Description |
|---|---|
| `quiescent` [nm] | HF thinning at local near-thruster sheath (−7 V) |
| `substorm`  [nm] | HF thinning at spacecraft-level −8 kV charging |
| `CLF`            | Coupled life factor over 7-yr NSSK mission (1 = no degradation) |

Console output example:
```
[1/20]  proxy=4.21e-02  quiescent=0.312 nm  substorm=1.847 nm  CLF=0.9991
  Quiescent HF range : 0.148 – 0.312 nm
  Substorm  HF range : 0.873 – 1.847 nm
  CLF range          : 0.9988 – 0.9994
```

---

## [2026-04-30] — Robotics upgrade: capsule collision, IK obstacle avoidance, Pinocchio RNEA, F_torque

### Added — `plume_impingement_pipeline.py`: capsule collision geometry (Phase 1)

`arm_has_collision` now models each arm link as a capsule (segment swept by a
sphere of radius `link_radius=0.08 m`) by inflating every obstacle bound before
the segment-vs-primitive test — equivalent to a full Minkowski-sum at zero extra
computational cost.

| Obstacle | Inflation applied |
|---|---|
| Client bus AABB | `min -= r`, `max += r` on all three axes |
| Servicer bus OBB | all three half-extents `+= r` |
| Servicer panels OBB (×2) | all three half-extents `+= r` |
| Antenna disc (×4) | radius `+= r`, slab thickness `+= 2r` |

`compute_F_kin` in `feasibility_cells.py` gains a matching `link_radius: float = 0.08`
parameter which is forwarded to `arm_has_collision`.  Default is backward-compatible.

---

### Added — `plume_impingement_pipeline.py`: signed-distance clearance helpers (Phase 2 support)

Three new module-level functions support gradient-based obstacle avoidance:

- `_point_to_aabb_sdf(pt, box_min, box_max)` — AABB signed distance function.
  Positive = outside, negative = penetrating (depth below nearest face), zero =
  on boundary.  Gives a non-zero gradient even when the arm centreline is inside
  an obstacle.
- `_segment_to_aabb_sdf(P0, P1, box_min, box_max)` — minimum SDF along a
  segment (20 uniform samples).
- `arm_min_clearance(pivot, p_elbow, p_wrist, p_thruster, stack, servicer_yaw_deg,
  link_radius)` — minimum signed clearance across all four obstacle types using
  `_segment_to_aabb_sdf`; OBBs are converted to body-frame AABB before the call;
  antenna discs use their bounding box cylinder.  Returns `raw_sdf − link_radius`;
  negative means the capsule surface is penetrating.

---

### Updated — `arm_kinematics.py`: null-space obstacle avoidance in CoG IK (Phase 2)

`arm_cog_ik` gains six new optional keyword arguments:

| Parameter | Default | Description |
|---|---|---|
| `obstacle_avoidance` | `False` | enable repulsion |
| `stack` | `None` | StackConfig for clearance queries |
| `servicer_yaw_deg` | `0.0` | forwarded to FK and `arm_min_clearance` |
| `link_radius` | `0.08` | capsule radius [m] |
| `obstacle_threshold` | `0.15` | repulsion activates when `d_min < threshold` [m] |
| `repulsion_gain` | `0.5` | scales repulsion as `gain / |d_min|` |

Because the 3×3 Jacobian is square (fully constrained), there is no null-space.
The repulsion gradient `∂d_min/∂q` (finite-differenced at `fd_step = 0.005 rad`)
is instead injected into the DLS right-hand side:

```
g = J^T W e  +  k_rep · ∇_q d_min
dq = (J^T W J + λ²I)⁻¹ g
```

The primary task objective still dominates when far from obstacles; near obstacles
the solution is steered toward clearance.  All existing callers are unaffected
(`obstacle_avoidance=False` by default).

---

### Added — `generate_arm_urdf.py` + `arm_dynamics.py`: Pinocchio RNEA (Phase 3)

**`generate_arm_urdf.py`**

Standalone script and importable `generate_urdf(arm, stack, output_path)` function.
Produces a 72-line URDF with:
- `servicer_bus` floating-base root link (solid-box inertia, 744 kg)
- Three revolute joints with hardware-confirmed axes and joint limits
- Per-link thin-rod inertia tensors expressed in the actual link direction:
  `I = m L² / 12 · (I₃ − û û^T)` — not a coordinate-aligned approximation
- Fixed `nozzle_frame` child link

CLI: `python generate_arm_urdf.py -o thruster_arm.urdf`

**`arm_dynamics.py`** — `ArmDynamics` class (requires `pinocchio ≥ 4.0`)

| Method | Returns | Description |
|---|---|---|
| `joint_torques(q, dq, ddq)` | (3,) N·m | Fixed-base RNEA; gravity = 0 |
| `reaction_wrench(q, dq, ddq, ...)` | (6,) N/N·m | Free-flyer RNEA; `τ[:6]` = arm→bus wrench |
| `mass_matrix(q)` | (3,3) kg·m² | Composite rigid-body mass matrix via CRBA |
| `coriolis_centrifugal(q, dq)` | (3,) N·m | Bias vector `C(q,q̇)q̇` |
| `orbital_fictitious_acc(r, v, ω)` | (3,) m/s² | Coriolis + centrifugal in LVLH frame |
| `torque_budget(q, dq, ddq)` | dict | τ, wrench, and scalar peak values |

Verified: RNEA = M·q̈ + C·q̇ to 2.8 × 10⁻¹⁷ N·m.

At a 90°/60°/30° pose, 0.05 rad/s deployment velocity, 0.01 rad/s² acceleration:
- `τ_max` = 0.44 N·m,  reaction moment on bus = 0.45 N·m

---

### Added — `feasibility_cells.py`: F_torque actuator-torque feasibility filter

**`FeasibilityConfig`** gains `tau_max_Nm: float = 50.0` (actuator torque limit).

**New functions:**

- `compute_torque_grid(dyn, q0g, q1g, q2g, F_kin, dq_sweep, ddq_sweep)` — iterates
  over kinematically feasible cells and calls `dyn.joint_torques()` via Pinocchio
  RNEA.  Returns `tau_peak` grid (NaN for skipped cells).  Default sweep state:
  `dq = [0.02, 0.02, 0.01]` rad/s, `ddq = [0.01, 0.01, 0.005]` rad/s².
- `compute_F_torque(tau_peak_grid, tau_max_Nm)` — boolean mask; NaN → True.

**Updated functions:**

- `compute_feasibility_epoch` accepts optional `tau_peak_grid`; when supplied,
  `F_torque` is included in `F_total` and returned in the result dict.
- `binding_constraint_breakdown` accepts optional `F_torque`; adds
  `frac_fail_torque` to its output dict.

Benchmark on 10³ grid: 550 F_kin-feasible cells evaluated; τ_peak mean = 1.12 N·m,
max = 1.68 N·m — all pass the 50 N·m actuator limit.

---

## [2026-04-30] — workspace_erosion_viz.py: six fidelity improvements + HF overlay

### Fixed — `workspace_erosion_viz.py`

**Bug: missing `servicer_yaw_deg` in arm-drawing FK calls.**  
Three `ARM.forward_kinematics()` calls used for drawing arm poses (Plotly trace,
print info, and matplotlib 3D pose) defaulted to `servicer_yaw_deg=0.0` instead
of `SERVICER_YAW_DEG=-25.0`.  The joint-space grid was computed *with* −25°
(via `compute_static_cell_quantities(..., servicer_yaw_deg=SERVICER_YAW_DEG)`),
so the arm was drawn 25° rotated away from its actual position, making the
min-erosion pose appear to pass through the client satellite body.
All three calls now explicitly pass `servicer_yaw_deg=SERVICER_YAW_DEG`.

---

### Updated — `workspace_erosion_viz.py`: six proxy/visualisation fidelity improvements

#### 1 — Line-of-sight occlusion (correctness fix)

Added `_client_aabb(stack)` helper returning `(aabb_min, aabb_max)` for the
client bus in LAR frame.  `_erosion_proxy` now accepts `los_aabb=` and, for
every (nozzle pose, panel point) pair, runs the slab-method ray-AABB test on
the segment.  Blocked segments contribute zero flux.

- **Effect**: 31.1 % of the 16.5 M nozzle→panel segments are occluded by the
  client bus body.  Without this, poses where the satellite body lies between
  the nozzle and the panels are incorrectly credited with high flux.

Processing is now chunked (default 2 048 poses/pass) so peak scratch-array
memory stays ≲50 MB regardless of grid size.

#### 2 — Panel surface incidence angle

`_panel_grid` now returns `(panel_pts, n_inc)` where `n_inc` is the inward
surface normal of the irradiated (servicer-facing) panel face:

```
n_inc = Rx(track) @ [0, 0, 1] = [0, −sin(φ), cos(φ)]
```

Each panel-point contribution is weighted by
`cos(α_inc) = clip(dot(unit_ion_dir, n_inc), 0, 1)`.
This penalises grazing incidence and correctly goes to zero when the plume
arrives parallel to the panel surface.

- **Effect**: max proxy 4.08 → 2.62; median halved.

#### 3 — Multi-species beam exponent

Module constant `_SPECIES_N_EXPS = (10.0, 8.0, 6.0)` for Xe⁺/Xe²⁺/Xe³⁺.
The directed-beam term is now a species-weighted sum:

```
beam = Σ_k  f_k · cos^{n_k}(θ)      k ∈ {Xe⁺, Xe²⁺, Xe³⁺}
```

using `THRUSTER.xe1_fraction` / `xe2_fraction` / `xe3_fraction` from
`ThrusterParams`.  Higher charge-state ions have slightly broader
distributions (lower `n_k`), spreading more flux to off-axis panel regions.

- **Effect**: max +5 %; median +2× vs. single-species model.

#### 4 — Solar panel tracking angle worst-case envelope

`_TRACKING_DEGS = np.linspace(-30, 30, 7)`.  For each tracking angle the proxy
is computed independently with the correspondingly rotated `panel_pts` and
`n_inc`, and the element-wise maximum is accumulated across the sweep.  Gives
a conservative (worst-case over-orbit) erosion estimate per arm pose.

- **Effect**: max +1 %; median envelope raised by ~20 %.

#### 5 — CEX isotropic wing

Module constant `_CEX_FRACTION = 0.10`.  An additive `cex_coeff / r²` term
(no angular dependence) is added to the directed beam flux before the LOS
mask is applied, representing the near-isotropic low-energy charge-exchange
population (~10 % of beam current in SPT-100-class thrusters).

- **Effect**: dominates for poses far off the plume axis; median proxy 1.5e-3
  → 0.57.  Confirms that CEX background flux sets a non-zero erosion floor
  across the entire accessible workspace.

#### 6 — High-fidelity IEDF-integrated overlay on top-K poses

Imports `ErosionIntegrator`, `HallThrusterPlume`, `SatelliteGeometry`, and
related classes from `sputter_erosion/`.  After the proxy sweep:

1. Top `HF_TOP_K = 20` worst-proxy poses are selected.
2. `_hifi_for_pose(p_nozzle, plume_dir, panel_pts, hall_plume, stack)` builds
   a full `SatelliteGeometry` (each panel sample point as one Ag interconnect,
   25 µm exposed edge) and runs `ErosionIntegrator.evaluate` for
   `HF_FIRING_S = 3600 s`.
3. Results printed to console and overlaid as diamond markers on both the
   matplotlib 3D plot (hot colormap, scaled by nm thinning) and the Plotly
   figure (hover tooltip: proxy + HF nm).

**Result from 40³ grid run, top-20 HF evaluation:**

| Metric | Value |
|---|---|
| HF thinning range (top-20) | 147–253 nm / hr |
| Worst pose | EE=(−1.34, −0.12, +0.91) m, q=(55.4°, 96.4°, 29.8°) |
| Proxy for worst pose | 3.936 |

---

### Updated — Open items

- [x] ~~Add CEX halo model~~ — CEX isotropic proxy term added (`_CEX_FRACTION`);
  full HallThrusterPlume CEX wing used in HF path.

---

## [2026-04-29] — High-fidelity sputter-erosion integration

### Added — `sputter_erosion/` package integration into `plume_impingement_pipeline.py`

Integrated the `sputter_erosion` library (in `sputter_erosion/`) as an optional
high-fidelity erosion path in `PlumePipeline`.  The analytical Yamamura power-law
path is preserved and remains the default.

#### `plume_impingement_pipeline.py`

**New `ThrusterParams` fields:**

| Field | Default | Description |
|---|---|---|
| `xe1_fraction` | 0.78 | Xe⁺ current fraction |
| `xe2_fraction` | 0.18 | Xe²⁺ current fraction |
| `xe3_fraction` | 0.04 | Xe³⁺ current fraction |
| `sheath_potential_V` | 20.0 | Local sheath potential for CEX ion acceleration [V] |

**New module-level helpers:**

- `_make_hall_plume(thruster, estimator)` — maps `ThrusterParams` + `ErosionEstimator`
  to a `HallThrusterPlume` (composite primary Gaussian IEDF + CEX wing).
- `_build_sputter_geometry(geo, hall_plume, material_name, ops, stack)` — translates
  LAR-frame pipeline geometry into a `SatelliteGeometry`.  Uses
  `geo.plume_direction_fk()` (FK nozzle axis) for the plume direction, and models
  each panel grid point as one `Interconnect` with its `exposed_face_normal` oriented
  toward the thruster projected into the panel surface plane — the correct sidewall
  face for Ag interconnect erosion.
- `_hifi_erosion_metrics(hifi_results, thickness_um, ops)` — maps
  `List[ErosionResult]` to the pipeline result-dict scalar format, scaling
  per-firing thinning to full mission lifetime.

**Updated `PlumePipeline.__init__`:**

```python
PlumePipeline(thruster=None, material=None, erosion_mode="analytical")
```

- `erosion_mode="analytical"` (default) — unchanged behaviour.
- `erosion_mode="high_fidelity"` — Eckstein-Preuss yield model, full IEDF
  integration, multi-species (Xe⁺/Xe²⁺/Xe³⁺), sheath bias.
  Material name aliases (`"Silver_interconnect"` → `"Ag"`, etc.) are resolved
  automatically; unknown materials raise `ValueError` immediately at construction.

**Updated `run_sweep()` result dict** — in `"high_fidelity"` mode the following
extra keys are populated:

| Key | Description |
|---|---|
| `hifi_mean_E_eV` | Mean ion energy at worst interconnect [eV] |
| `hifi_sheath_boost_eV` | Sheath-bias energy added to CEX population [eV] |
| `hifi_max_fluence_ions_m2` | Cumulative ion fluence at worst interconnect [ions/m²] |
| `hifi_worst_incidence_deg` | Ion incidence angle at worst interconnect [°] |
| `hifi_worst_j_i` | Local ion current density at worst interconnect [A/m²] |

**New `PlumePipeline.run_monte_carlo(case_indices=None, n_samples=200, seed=42)`:**
Requires `erosion_mode="high_fidelity"`.  Runs Bayesian MC over the
Zameshin & Sturm (2022) Ag yield-parameter posterior built into
`sputter_erosion.MATERIALS["Ag"].bayesian["Xe"]`.  Returns
`{case_idx: {p5, p50, p95, mean}}` in µm.

#### Key physics differences vs. analytical path

| Aspect | Analytical | High-fidelity |
|---|---|---|
| Yield model | Yamamura power-law, single energy | Eckstein-Preuss, IEDF-integrated |
| Angular dependence | Cosine Lambert on panel face | Garcia-Rosales/Eckstein on interconnect sidewall |
| Target surface | Full panel face | Exposed Ag interconnect edge (25 µm) |
| Ion species | Single Xe⁺ | Xe⁺/Xe²⁺/Xe³⁺ with correct per-charge energy |
| Sheath bias | Not modelled | CEX population accelerated by string bias (66–113 eV) |
| Uncertainty | None | Bayesian MC (p5/p50/p95) |

Empirical result (8-case yaw sweep, SPT-100-like, 5 yr, Ag 25 µm): HF model
predicts ~10–40× less erosion than the analytical model.  Both models agree on
the worst-case yaw angle (yaw=0°, Spearman ρ=1.0 over the sweep).  The analytical
model's conservatism is driven by treating the whole panel face as the erosion
surface and using a simplified single-energy yield.

### Added — `test_sputter_integration.py`

Four-section integration test suite:

1. **Smoke tests** — library import, pipeline construction, material alias resolution,
   bad-mode error.
2. **Key parity** — verifies all shared keys are present in both modes and all
   `hifi_*` keys appear only in the HF result.
3. **Yaw sweep comparison** — 8-case shoulder-yaw sweep (−30° to 90°), analytical
   vs. HF side-by-side with HF/analytic ratio, mean ion energy, sheath boost,
   and Spearman rank correlation.
4. **Monte Carlo** — 200-sample Bayesian MC on the 3 worst HF cases; reports
   p5/p50/p95/mean in µm and the uncertainty spread (p95/p5 ≈ 3.9×).

Run with `python test_sputter_integration.py`.

### Added — `test_workspace_hifi.py`

Validates whether the fast erosion proxy (`Σ cos^n(θ)/r²` from
`workspace_erosion_viz`) correctly rank-orders arm poses by erosion risk relative
to the high-fidelity integrator.

Pipeline: 12×12×12 joint-space grid → F_kin (collision + joint limits) →
F_align (NSSK-N, α_max=15°) → 19-pose sample across proxy quintiles →
HF integrator (1-hr snapshot per pose) → Spearman/Pearson rank correlation.

Result: Spearman ρ=0.99, Pearson r=0.99 (log-linear).  The proxy
correctly ranks erosion risk across the accessible NSSK-N workspace.
Mean HF erosion per proxy quintile rises monotonically (Q1: 0.18 nm →
Q5: 6.6 nm over 1 hr), validating the proxy as a reliable first-pass
filter before invoking the HF integrator.

Run with `python test_workspace_hifi.py`.

---

## [Unreleased] — 2026-04-23

### Updated — Hardware geometry inputs (all six files)

Replaced the placeholder simplified-arm geometry with confirmed hardware data.

#### `feasibility_inputs.json` — confirmed values

| Field | Old | New |
|---|---|---|
| `nozzle_exit_direction_ee` | `[0,0,1]` placeholder | `[0.1455, 0.9189, 0.3666]` — hardware-confirmed nozzle direction in link-3 body frame (= TA frame at q=0) |
| `client_mass_kg` | 2500 | 2800 |
| `client_cog_lar_m` | `[null,null,null]` | `[0.03, 0, 1.72]` — hardware-confirmed |
| `tank_centroid_lar_m` | `[null,null,null]` | `[0.2, 0, −1.15]` — servicer CoG + body-frame offset (0.2, 0, 0.1) m at yaw=0 |

#### `plume_impingement_pipeline.py` — `RoboticArmGeometry`

Replaced the simplified yaw-pitch-pitch chain with the general serial-chain hardware geometry.

**New fields**:

| Field | Value | Description |
|---|---|---|
| `ta_origin_{x,y,z}` | (−0.12891, 0.31619, 0.356) m | TA-frame origin in servicer body |
| `h1_ta_{x,y,z}` | (0, 0, 0.25175) m | Hinge 1 in TA frame |
| `d_h1h2` | (−0.12, −1.10288, −0.1) m | Link vector H1→H2 in body frame |
| `d_h2h3` | (−0.1, 1.51918, 0.02595) m | Link vector H2→H3 in body frame |
| `d_h3n` | (0.47977, −0.06677, 0.006) m | Link vector H3→Nozzle in body frame |
| `axis1` | (0, 0, −1) | Hinge 1 rotation axis in TA frame |
| `axis2` | (1, 0, 0) | Hinge 2 rotation axis in link-1 body frame |
| `axis3` | (0, −0.3746, 0.9272) | Hinge 3 rotation axis in link-2 body frame |
| `n_hat_body` | (0.1455, 0.9189, 0.3666) | Nozzle exit direction in link-3 body frame |

Updated link lengths derived from `|d_hXhY|`: L1=1.1139 m, L2=1.5227 m, L3=0.4844 m.

**Updated `pivot_offset_*`**: now full offset from servicer origin (no `servicer_bus_z/2` dependency): (−0.12891, 0.31619, 0.60775).

**New method `arm_pivot_in_servicer_body()`**: returns Hinge-1 position in servicer body frame = TA-origin + H1-in-TA = (−0.12891, 0.31619, 0.60775) m.

**New `_rodrigues(axis, angle)` module-level helper**: 3×3 rotation matrix via Rodrigues formula.

**Updated `forward_kinematics(pivot, q0, q1, q2, servicer_yaw_deg=0)`**: uses `_rodrigues` chain; returns (p_H2, p_H3, p_nozzle) in LAR.

**Updated `GeometryEngine._pivot_position()`**: now uses `arm.arm_pivot_in_servicer_body()` (removes `servicer_bus_z/2` addition).

#### `arm_kinematics.py`

**Updated `arm_fk_transforms(arm, pivot, q, servicer_yaw_deg=0)`**: general serial-chain via Rodrigues; 4×4 homogeneous transforms identical in interface but correct for new geometry.

**Updated `arm_cog_and_jacobian(arm, pivot, q, servicer_yaw_deg=0)`**: cross-product Jacobian (`J[:, i] = ωᵢ × (p_com − pᵢ) / M`) replaces old yaw-pitch-pitch closed-form partials. Correct for arbitrary rotation axes.

#### `feasibility_cells.py`

**New `_vrodrigues(axis, angles)` helper**: vectorized Rodrigues over a grid of angles, returns `(*angles.shape, 3, 3)`.

**Updated `compute_static_cell_quantities(..., servicer_yaw_deg=0.0)`**: replaced closed-form EE frame computation with batched Rodrigues chain:
```
R1_ta, R12_ta, R123_ta  →  CR1, CR12, CR123  (Rz_serv prepended)
p_h2     = pivot + einsum(CR1,   d_h1h2)
p_h3     = p_h2  + einsum(CR12,  d_h2h3)
p_nozzle = p_h3  + einsum(CR123, d_h3n)
t_hat    = −einsum(CR123, n_hat_body)
```

#### `feasibility_map.py`

**Updated `compute_pivot()`**: calls `arm.arm_pivot_in_servicer_body()` directly; removes `servicer_bus_z/2` hack.

**Updated `build_feasibility_maps()`**: passes `servicer_yaw_deg` to `compute_static_cell_quantities`.

#### `geometry_visualizer.py`

**Updated `pivot_position()`**: calls `arm.arm_pivot_in_servicer_body()` (consistent with the other pivot formulas).

#### `workspace_erosion_viz.py`

**Updated `compute_static_cell_quantities` call**: passes `servicer_yaw_deg=SERVICER_YAW_DEG`.

#### Verification

Stowed-config FK error vs. hardware-spec positions: < 1e-15 m (machine precision) for all four joints.

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
- [x] ~~Add CEX halo model~~ — CEX isotropic proxy term + full HF path added (2026-04-30)
- [ ] Vectorise `compute_F_kin` for performance at 50³+ grids
