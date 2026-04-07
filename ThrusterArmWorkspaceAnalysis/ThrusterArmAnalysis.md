# Plasma Plume Impingement Parametric Study — Project Summary

## 1. What We Have Built

### Script 1: `plume_impingement_pipeline.py` — Parametric Geometry & Erosion Screening

This is the foundational pipeline that generates the geometric case matrix and performs analytical pre-screening of plume erosion before committing to full OpenPlume simulations. It contains:

**Case Matrix Generator** — Defines sweep ranges for 9 parameters (arm length, arm azimuth, arm elevation, client mass, servicer mass, panel span, firing duration, mission duration, panel sun-tracking angle) and produces either full combinatorial or reduced (2–3 parameter) sweep matrices.

**Geometry Engine** — For each parameter combination, computes the thruster exit position in the client body frame, the plume direction (constrained to point through the stack centre of gravity), and a grid of points across both solar panels. For every panel grid point it calculates the distance from the thruster, the off-axis angle from the plume centreline, and the incidence angle on the panel surface.

**Analytical Erosion Estimator** — Uses a cosine-power plume density model with inverse-square falloff, combined with a Yamamura-type sputter yield model for Xe⁺ on silver (with angular dependence), to estimate cumulative erosion depth at each panel point over the mission lifetime. Each case is classified as SAFE (<10% of 25 µm), CAUTION (10–50%), MARGINAL (50–100%), or FAIL (>100%).

**Heatmap Visualiser** — Takes any two sweep parameters as axes and generates both continuous erosion-depth heatmaps (with FAIL/MARGINAL contour overlays) and categorical status maps. The MARGINAL and CAUTION cases are flagged as the subset that should be fed into OpenPlume for high-fidelity simulation.

**Demo output**: 3 sweeps (arm length vs azimuth, panel span vs mission duration, firing duration vs arm elevation), 103 total cases, 6 heatmaps, CSV export. Identified 25 cases needing full OpenPlume simulation.


### Script 2: `propellant_erosion_correlation.py` — Propellant Budget Coupling

This script correlates the stationkeeping propellant budget with erosion, including time-resolved COG migration. It contains:

**Stationkeeping Budget Model** — GEO-default delta-V requirements (NSSK ~50 m/s/yr, EWSK ~2 m/s/yr, momentum management ~0.5 m/s/yr) with 10% margin, plus a custom-input option. Computes manoeuvre scheduling (NSSK 2×/day, EWSK 2×/week).

**Propellant Budget Calculator** — Tsiolkovsky rocket equation for electric propulsion, iterative yearly computation accounting for decreasing stack mass, propellant-limited mission duration calculation, and firing duration per manoeuvre at current mass.

**COG Migration Tracker** — Tracks the stack centre of gravity as xenon is consumed from the servicer tank. Separates the mass model into client (fixed CG), servicer dry mass (fixed CG), and propellant mass (at tank position, decreasing over time). Outputs a time trajectory of COG position.

**Time-Resolved Erosion Integrator** — Steps through the mission in configurable time increments (default 30 days). At each step: recomputes the stack COG for the current propellant level, recalculates the thrust vector direction (which must pass through the new COG), evaluates NSSK and EWSK erosion separately (different panel tracking angles), accumulates erosion, and tracks both the propellant-limited and erosion-limited mission endpoints.

**Parametric Sweeps** — Propellant load vs mission duration, and propellant load vs arm length, each producing correlation heatmaps for achievable mission duration, erosion depth, limiting factor (propellant vs erosion), and COG shift.

**Demo output**: Single 7-year mission time-resolved analysis, two parametric sweeps (88 + 56 cases), 11 heatmaps, 3 CSV exports. Key finding: for 0° arm azimuth at 2.5 m, the mission is erosion-limited at ~1.3 years despite having sufficient propellant for 7+ years.


---

## 2. Current Assumptions

### Geometry Assumptions

- **Docking configuration**: The servicer is placed on top of the client satellite in the +Z direction (anti-earth face). The current code places the servicer geometric centre at `client_bus_z/2 + servicer_bus_z/2 + dock_offset_z` above the client origin. **This does not correctly model the actual configuration** where the servicer docks on the client's Z− panel (earth-facing) at the Launch Adapter Ring. The coordinate system needs to be flipped so the servicer is below the client, and the LAR docking interface geometry needs to be properly defined.

- **Thruster arm model**: The arm is modelled as a single rigid link from a pivot point to the thruster exit plane. The arm direction is defined by two angles (azimuth and elevation) from the pivot. **This is a significant simplification** — the actual arm is a multi-link robotic arm with joints, and its configuration is determined by inverse kinematics to achieve the desired thrust vector through the COG while respecting joint limits and collision avoidance constraints.

- **Thrust vector constraint**: The plume direction is assumed to be the exact opposite of the ideal thrust direction, which always passes through the instantaneous stack COG. No pointing offset or deadband is modelled. In reality, the arm controller may accept some offset (absorbed by reaction wheels) to improve the erosion situation.

- **Solar panels as flat rectangles**: Panels are modelled as planar surfaces at the top of the client bus (Z+), with sun-tracking rotation about the hinge line. No panel yoke geometry, deployment mechanism standoffs, or curved/segmented panel shapes are modelled.

- **Panel positioning**: Both panels are assumed symmetric about the bus centre, extending in the ±X direction. No asymmetric panel configurations or articulated panel wings are considered.

- **Antenna reflectors**: Defined as parameters but not currently included in the erosion calculation. Only solar panel erosion is computed.

- **Bus geometry**: Both servicer and client buses are rectangular parallelepipeds. No appendages, thermal radiators, or other protruding structures that could shadow or redirect the plume are modelled.


### Plume Physics Assumptions

- **Cosine-power model**: The plume density is modelled as `j(r,θ) = (I_beam/e) × (n+1)/(2πr²) × cos^n(θ)` where n is a single exponent. This is a far-field approximation that does not capture near-field structure, and uses a single population. Real Hall thruster plumes often require a two-population model (main beam + scattered/CEX population) to correctly represent the large-angle flux.

- **No charge-exchange (CEX) ions**: The model only accounts for direct beam ions. CEX ions — created when fast beam ions undergo charge exchange with slow neutrals in the plume — can scatter to very large off-axis angles (>60°) and deposit on surfaces that appear geometrically shadowed from the direct beam. This can be a significant contributor to erosion on surfaces not in the direct line of sight.

- **No plume-surface interactions**: Sputtered material is not tracked. Silver atoms eroded from one surface could redeposit on other surfaces (cross-contamination), potentially degrading optical surfaces, solar cell cover glasses, or thermal coatings.

- **No ambient plasma interaction**: The plume propagates into vacuum. In GEO, the ambient plasma density is low enough that this is a reasonable assumption, but near eclipses or during geomagnetic storms the ambient conditions could affect CEX production.

- **Steady-state thruster operation**: The plume profile is assumed constant during each firing. Thruster startup and shutdown transients (which can have very different divergence profiles) are not modelled.

- **Ion energy from discharge voltage**: Ion energy is approximated as 70% of the discharge voltage. Actual ion energy distributions can be broad and depend on the specific thruster design and operating point.


### Sputter Yield Assumptions

- **Yamamura-type fit**: The sputter yield model uses `Y(E) = a × (E − E_th)^b` for normal incidence with an angular correction `f(θ) = cos(θ)^c × exp(−d × (1/cos(θ) − 1))`. The coefficients (a=0.059, b=0.7, E_th=15 eV, c=−1.3, d=0.4) are generic fits for Xe⁺ on Ag. They may not accurately represent the specific silver interconnect composition (which may contain other elements or have surface oxides).

- **Single species sputtering**: Only silver erosion is computed. In reality, the interconnect sits on a substrate and has cover layers. The effective erosion rate changes as different materials are exposed.

- **No redeposition**: Sputtered silver does not return to the surface. In practice, some fraction of sputtered atoms may redeposit nearby, effectively reducing the net erosion rate.

- **No enhanced sputtering effects**: Chemical sputtering, sputtering by multiply-charged ions, and sputtering at elevated temperatures are not considered.

- **Uniform thickness**: The 25 µm silver interconnect thickness is assumed uniform across all panel locations.


### Propellant & Operations Assumptions

- **Propellant on servicer only**: All xenon propellant is stored on the servicer. No residual propellant on the client is considered (the client is at end-of-life, so its propellant is presumably depleted).

- **Linear propellant consumption within each time step**: Within each 30-day integration step, propellant is consumed at a constant rate. In reality, consumption is concentrated during the actual firing windows.

- **Fixed stationkeeping budget per year**: The NSSK and EWSK delta-V requirements are constant each year. In reality, NSSK requirements vary slightly with solar cycle (solar radiation pressure perturbations), and EWSK varies with the satellite's longitude slot.

- **Servicer dry mass fixed at 280 kg**: In the propellant sweep, the servicer dry mass is held constant and propellant is added on top. The actual dry mass depends on the servicer design.

- **Tank COG at servicer geometric centre**: The propellant tank centre of gravity is assumed to be at the servicer's geometric centre. In practice, the tank may be offset, and as propellant is consumed the CG within the tank shifts (liquid settling in microgravity is complex, though for xenon stored as a supercritical fluid this is less of an issue).

- **No orbit mechanics**: The manoeuvre timing, orbit position, and solar geometry are not modelled. The panel sun-tracking angle is a fixed parameter per manoeuvre type, rather than being computed from the actual orbit position and solar vector.

- **Two manoeuvre types only**: NSSK and EWSK. Momentum management firings, orbit relocation burns, and collision avoidance manoeuvres are accounted for in the delta-V budget but not geometrically modelled (they use the same plume geometry as NSSK).


### COG Migration Assumptions

- **Three-body mass model**: The stack is decomposed into client (fixed), servicer dry (fixed), and propellant (decreasing). This does not account for thermal fuel migration, structural deformation, or any mass changes on the client side (e.g. antenna deployment mechanism shifts).

- **Instantaneous COG recalculation**: At each time step the COG is recalculated and the thrust vector immediately adjusts. No controller dynamics, arm repositioning time, or transient pointing errors are modelled.

- **No moment of inertia tracking**: Only COG position is tracked, not the inertia tensor. Changes in MOI affect attitude control authority and wheel desaturation requirements.


---

## 3. Details We Have Missed / Areas for Improvement

### Critical — Docking Configuration (Your Point)

The current model has the servicer on the client's Z+ (anti-earth) face. In reality, the servicer docks on the client's Z− panel where the Launch Adapter Ring (LAR) sits. This means:

- The servicer is below the client bus, not above it.
- The thruster arm pivot is on the servicer's Z+ face (top of the servicer, facing toward the client).
- The arm must navigate around the client bus to position the thruster where it can fire without impinging on the client's structures.
- The geometric relationship between the thruster and the solar panels is fundamentally different — the panels are now above and to the sides of the servicer, rather than below.
- The LAR diameter and the clamping mechanism geometry constrain where the arm pivot can be located on the servicer.

**Fix required**: Flip the servicer placement to Z−, define the LAR interface geometry, and recompute all thruster-to-panel geometry. This will change the erosion results significantly.


### Critical — Multi-Link Robotic Arm with Inverse Kinematics (Your Point)

The current single-link arm model doesn't capture the real arm's behaviour. A 3-link arm with joints has:

- **Joint angle limits**: Each joint has mechanical limits that constrain the workspace envelope.
- **Workspace reachability**: Not all thruster positions and orientations are achievable — the arm has a finite workspace that depends on link lengths and joint limits.
- **Inverse kinematics for COG tracking**: As the COG migrates, the arm must reconfigure to point the thrust vector through the new COG. The IK solution determines the actual joint angles and therefore the actual thruster position and orientation. There may be multiple IK solutions (elbow-up vs elbow-down configurations), and the chosen solution affects the thruster's proximity to the panels.
- **Collision avoidance**: The arm links must not collide with the client bus, solar panels, antennas, or the servicer body itself. This further constrains the feasible configurations.
- **Singularity avoidance**: Near kinematic singularities the arm loses dexterity, which may force less-optimal configurations.
- **Configuration switching**: As the COG shifts over the mission, the arm may need to switch between IK solution branches, causing discontinuous jumps in thruster position.

**Fix required**: Replace the single-link model with a 3-DOF (or more) robotic arm model. Define link lengths, joint types (revolute/prismatic), joint limits, and implement an IK solver that finds the arm configuration achieving the desired thrust direction through the current COG, subject to collision and joint-limit constraints. The parametric sweep should then be over link lengths, joint limits, and arm mounting position rather than abstract azimuth/elevation angles.


### Important — CEX Backflow Population

Charge-exchange ions can contribute 10–30% of the total erosion on surfaces outside the main beam cone. A two-population plume model (beam + CEX) with different divergence profiles would significantly improve accuracy. The CEX population typically has a much broader angular distribution (nearly isotropic at large angles) and lower energy (thermal, ~1 eV), which means lower sputter yield per ion but potentially high flux on shadowed surfaces.


### Important — Plume Interaction with Structures

The client satellite bus, antenna reflectors, and thermal radiators can block, reflect, or scatter plume ions. A direct line-of-sight check between the thruster and each panel point, with the client bus as an occluder, would identify truly shadowed regions. Ions scattering off the bus surface also contribute a diffuse secondary flux.


### Important — Thermal Effects on Surfaces

Plume heating of panel surfaces during thruster firing can:
- Temporarily increase surface temperature, potentially altering sputter yield coefficients.
- Cause thermal cycling fatigue on interconnects.
- Affect solar cell performance during firing windows.
- Exacerbate erosion if the surface reaches temperatures where diffusion-enhanced sputtering becomes significant.


### Important — Multiple Surface Materials

The solar panels have multiple materials exposed to the plume:
- Silver interconnects (25 µm) — the critical failure mode.
- Cover glass (CMG or CMX borosilicate, ~100–150 µm) — if eroded, degrades cell output.
- Solar cell semiconductor (GaInP/GaAs/Ge triple-junction) — not directly eroded but performance degrades.
- Kapton or CFRP substrate — structural.
- Panel adhesives.

Each material has a different sputter yield profile and a different failure threshold. A complete analysis would track erosion of all layers, not just silver.


### Moderate — NSSK vs EWSK Geometry Differentiation

NSSK burns are typically performed near the orbit nodes (ascending/descending node), while EWSK burns happen at different orbit positions. This means the solar panel sun-tracking angle is different for each type, and the stack orientation relative to the velocity vector may differ. The current model allows different tracking angles per manoeuvre type but doesn't compute them from orbit mechanics.


### Moderate — Disturbance Torque Budget

If the thrust vector doesn't pass exactly through the COG (due to pointing error, arm flexibility, or intentional offset), the resulting disturbance torque must be absorbed by reaction wheels. The torque magnitude `τ = F × d` where d is the moment arm (perpendicular distance from thrust vector to COG) determines the wheel momentum loading. If wheels saturate, additional desaturation firings are needed, creating more plume exposure. This is a feedback loop between pointing accuracy, wheel capacity, and total erosion.


### Moderate — Firing Arc Effects

For electric propulsion, NSSK manoeuvres are long burns (potentially hours) during which the satellite moves along its orbit. The solar panel orientation relative to the plume changes continuously during a single burn as the sun vector rotates. The current model treats the panel angle as constant during each time step.


### Minor — Plume Divergence Evolution with Thruster Wear

Hall thruster plume divergence typically increases over the thruster's lifetime as the channel erodes. A thruster operated for 5+ years may have a measurably broader plume than at beginning of life, increasing the flux at off-axis panel locations.


### Minor — Electrostatic Effects

Charged plume ions impacting on dielectric surfaces (cover glass, Kapton) can cause differential charging. In GEO's high-energy electron environment, this can lead to electrostatic discharge events that damage solar cells independently of erosion.


### Minor — Contamination from Thruster

Hall thrusters erode their own ceramic channel, producing boron nitride or alumina particles that deposit on nearby surfaces. This contamination can degrade solar cell transmittance and thermal coating properties independently of the sputter erosion of the panel materials themselves.


---

## 4. Files Produced

### Scripts
- `plume_impingement_pipeline.py` — Core parametric sweep and erosion screening pipeline
- `propellant_erosion_correlation.py` — Propellant budget coupling with time-resolved COG migration

### Data
- `erosion_results.csv` — All cases from the geometry pipeline (103 cases)
- `sweep_prop_vs_mission.csv` — Propellant vs mission duration sweep (88 cases)
- `sweep_prop_vs_arm.csv` — Propellant vs arm length sweep (56 cases)
- `time_history.csv` — Time-resolved data for single 7-year mission (monthly resolution)

### Visualisations
- 6 heatmaps/status maps from the geometry pipeline
- 3 time-resolved plots (erosion+propellant, COG+firing, erosion rate)
- 8 correlation heatmaps from the propellant pipeline
