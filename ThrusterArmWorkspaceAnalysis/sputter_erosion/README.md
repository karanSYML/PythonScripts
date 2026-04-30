# sputter_erosion

A modular Python framework for computing erosion of spacecraft surfaces
(particularly solar-array silver interconnects) due to plasma-thruster plume
impingement in the GEO environment. Designed as a companion to `geo_rpo` and
intended to slot into a Mission Analysis Report workflow.

## Architecture

The package is organised in layers that mirror the physical-parameter groups
identified in scoping:

| Layer | Module | Physical-parameter group |
|---|---|---|
| Sputter-yield models | `yields.py` | Group 3 — target response |
| Material database (Eckstein, Y-T, Bayesian posteriors) | `materials.py` | Group 3 — target response |
| Plume / IEDF / species / CEX | `plume.py` | Group 1 — local plume state |
| Geometry, sheath, line-of-sight | `geometry.py` | Group 2 — impact conditions |
| GEO plasma + thermal cycling | `environment.py` | Group 4 — environmental modulators |
| Master integrator | `erosion.py` | Couples all four groups |
| Bayesian uncertainty propagation | `monte_carlo.py` | V&V layer |
| Mission-level wrapper | `mission.py` | Lifetime, duty cycle, sensitivity |

## Yield-model choices

Three energy-dependence models are implemented:

- **`YamamuraTawara`** (1996) — the EP-community workhorse; weak near threshold.
- **`EcksteinPreuss`** (2003) — the recommended default. Better near-threshold
  behaviour, parameters anchored against Tartz et al. (2011) low-energy Xe→Ag
  and Eckstein 2007 IPP-Report 9/132 fits.
- **`Seah2005`** — analytic Q(target) form, useful when the target material is
  not in the Y-T or Eckstein parameter sets.

Two angular-dependence models:

- **`YamamuraAngular`** (1981) — original f / theta_opt form.
- **`EcksteinAngular`** (Garcia-Rosales/Eckstein 1994) — better at large angles
  and for heavy projectiles.

A `FullYield` object composes one of each plus an optional sub-threshold floor
to represent the Mantenieks-style finite yield observed below the nominal
threshold.

## Bayesian uncertainty (Zameshin & Sturm 2022 style)

`monte_carlo.py` propagates posteriors of (Q, s, Eth) through the full
geometric+plume+yield pipeline. Each material in `materials.MATERIALS` carries
a `BayesianPosterior` entry; the MC sampler perturbs both the Y-T and the
mapped-to Eckstein parameters consistently so the V&V is honest across
yield-model choices.

## Geometric coupling

`geometry.py` exposes the satellite/thruster/array layout in a body-fixed
frame. Each `Interconnect` carries:

- its panel-local position,
- its exposed-face normal (the actual sidewall hit by ions),
- its electrical position along the string (for sheath bias),
- its initial exposed thickness (life-limiting dimension).

`ThrusterPlacement` evaluates the plume model at the right (theta, range) for
each interconnect and computes the local incidence angle automatically. The
`SatelliteGeometry.iter_targets` generator walks every (thruster, array,
interconnect) triple, skipping back-faces.

## Mission-level workflow

```python
from sputter_erosion import (
    HallThrusterPlume, ThrusterPlacement, Vector3, SolarArray, Interconnect,
    SatelliteGeometry, ErosionIntegrator, MissionProfile, FiringPhase,
    LifetimeAnalysis,
)
from sputter_erosion.yields import FullYield
from sputter_erosion import EcksteinPreuss, EcksteinAngular

# ... build plume, thrusters, arrays ...
geometry = SatelliteGeometry(thrusters=[...], solar_arrays=[...])
integrator = ErosionIntegrator(
    yield_model=FullYield(EcksteinPreuss(), EcksteinAngular()),
    apply_sheath=True,
)
mission = MissionProfile(phases=[
    FiringPhase("NSSK_15yr", geometry, duration_s=15*3500*3600.0),
])
life = LifetimeAnalysis(integrator)
prediction = life.life_prediction(mission, initial_thickness=25e-6)
```

See `examples/example_geo_nssk.py` for a worked end-to-end case with NSSK Hall
thrusters firing past a deployed solar wing, plus Bayesian MC and cant-angle
sensitivity sweep.

## Caveats

- Material parameter values in `materials.py` are anchored to the literature
  but should be replaced with project-specific tank data where available.
- The plume model is a semi-empirical cosine^n + Gaussian-CEX-wing form.
  Production work should ingest measured (or PIC-simulated) j_i(theta) and
  IEDF tables instead — `IEDF.composite` accepts arbitrary energy grids.
- Sub-threshold sputtering is included as an optional `subthreshold_floor`
  parameter; default 0. Carry it as a sensitivity case for CEX-dominated
  surfaces.
- Thermal-fatigue coupling in `LifetimeAnalysis` is a Coffin-Manson heuristic;
  for actual qualification, couple to a proper interconnect FE / fatigue model.
