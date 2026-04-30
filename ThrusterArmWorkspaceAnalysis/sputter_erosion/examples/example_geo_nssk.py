"""
example_geo_nssk.py
===================

End-to-end demonstration of the sputter_erosion package on a representative
GEO N-S station-keeping (NSSK) firing scenario.

Scenario
--------
  * Satellite body-fixed frame: +X ram, +Z anti-nadir (sun-pointing array
    normal), +Y completes RH frame.
  * Two NSSK Hall thrusters mounted on the +Y and -Y faces, canted 45 deg
    aft (toward +X) so the plume points roughly along the +Y / -Y
    directions but not exactly on the array plane.
  * Single deployed solar wing along the +Y axis with a string of Ag
    interconnects running the panel length.
  * 3500 hours of total firing per year (typical for NSSK on a 6 kW HET).
  * 15-year mission lifetime.

Outputs
-------
  * Per-interconnect erosion rate, fluence, and total thinning.
  * Bayesian MC (200 samples) of the Ag yield-parameter posterior, giving
    5/50/95% percentiles of end-of-mission thinning.
  * Sensitivity to thruster cant angle.
"""

from __future__ import annotations
import numpy as np

from sputter_erosion import (
    Vector3, ThrusterPlacement, SolarArray, Interconnect, SatelliteGeometry,
    SheathModel, HallThrusterPlume, SpeciesFractions,
    YamamuraTawara, EcksteinPreuss, EcksteinAngular, YamamuraAngular,
    ErosionIntegrator, MonteCarloErosion, ParameterPosterior,
    MissionProfile, FiringPhase, LifetimeAnalysis, ThermalCycling,
    MATERIALS,
)
from sputter_erosion.yields import FullYield


# ---------------------------------------------------------------------------
# 1. Build the plume model
# ---------------------------------------------------------------------------

plume_north = HallThrusterPlume(
    V_d=300.0,                   # 300 V discharge
    I_beam=15.0,                 # 15 A beam current
    mdot_neutral=20e-6,          # 20 mg/s xenon
    half_angle_90=np.deg2rad(35),
    cex_wing_amp=0.025,          # 2.5% of axis density in the wing
    cex_wing_width=np.deg2rad(45),
    species=SpeciesFractions(0.78, 0.18, 0.04),
    sheath_potential=20.0,
)
plume_south = HallThrusterPlume(  # identical S thruster
    V_d=300.0, I_beam=15.0, mdot_neutral=20e-6,
    half_angle_90=np.deg2rad(35), cex_wing_amp=0.025,
    cex_wing_width=np.deg2rad(45),
    species=SpeciesFractions(0.78, 0.18, 0.04),
    sheath_potential=20.0,
)

# ---------------------------------------------------------------------------
# 2. Geometry: thrusters + solar wing
# ---------------------------------------------------------------------------

# Thrusters mounted on the aft (-X) face, near the +/-Z corners, firing
# predominantly in -X (aft) but canted +/- 45 deg toward N/S so they push
# against the orbit-normal disturbance. This is a typical NSSK geometry:
# the primary beam goes aft, the deployed solar wings (along +/-Y) see only
# the broad CEX wing.
ca = np.deg2rad(45.0)
thr_north = ThrusterPlacement(
    position_body=Vector3(-1.5, 0.0,  0.8),
    fire_direction_body=Vector3(-np.cos(ca), np.sin(ca), 0.0),
    plume=plume_north,
    cant_angle_deg=45.0,
)
thr_south = ThrusterPlacement(
    position_body=Vector3(-1.5, 0.0, -0.8),
    fire_direction_body=Vector3(-np.cos(ca), -np.sin(ca), 0.0),
    plume=plume_south,
    cant_angle_deg=45.0,
)

# Deployed solar wings extending along +/-Y. Panel normal points +Z (sun).
# Origin starts at y = +/-2 m (yoke length) and extends 8 m outboard.
wing_north = SolarArray(
    origin_body=Vector3(0.0, 2.0, 0.0),
    panel_normal_body=Vector3(0.0, 0.0, 1.0),
    panel_x_body=Vector3(0.0, 1.0, 0.0),
    width=8.0, height=2.5,
    interconnects=[
        Interconnect(
            position_local=(y, 1.25),
            # Exposed sidewall faces -X (back toward thruster pod) and +Z
            # (skyward); a moderate fraction of the CEX wing arrives this way.
            exposed_face_normal=Vector3(-1.0, 0.0, 0.3),
            material_name="Ag",
            coverglass_overhang=50e-6,
            string_position=y / 8.0,
            exposed_thickness=25e-6,
        )
        for y in np.linspace(0.5, 7.5, 15)
    ],
)

wing_south = SolarArray(
    origin_body=Vector3(0.0, -2.0, 0.0),
    panel_normal_body=Vector3(0.0, 0.0, 1.0),
    panel_x_body=Vector3(0.0, -1.0, 0.0),
    width=8.0, height=2.5,
    interconnects=[
        Interconnect(
            position_local=(y, 1.25),
            exposed_face_normal=Vector3(-1.0, 0.0, 0.3),
            material_name="Ag",
            coverglass_overhang=50e-6,
            string_position=y / 8.0,
            exposed_thickness=25e-6,
        )
        for y in np.linspace(0.5, 7.5, 15)
    ],
)

geometry = SatelliteGeometry(
    thrusters=[thr_north, thr_south],
    solar_arrays=[wing_north, wing_south],
    sheath=SheathModel(
        string_voltage=100.0,
        floating_potential=-12.0,
        Te_local=2.0,
    ),
)

# ---------------------------------------------------------------------------
# 3. Yield model: Eckstein-Preuss energy + Eckstein angular + sub-threshold
# ---------------------------------------------------------------------------

yield_model = FullYield(
    energy_model=EcksteinPreuss(),
    angular_model=EcksteinAngular(b=1.8, c=1.5, theta_opt_deg=65.0),
    subthreshold_floor=0.005,        # small Mantenieks-style sub-threshold tail
    Eth_floor_frac=0.6,
)

integrator = ErosionIntegrator(
    yield_model=yield_model,
    include_xe2=True,
    include_xe3=True,
    apply_sheath=True,
)

# ---------------------------------------------------------------------------
# 4. Single-firing diagnostic snapshot (1 hour of continuous N+S firing)
# ---------------------------------------------------------------------------

print("=" * 78)
print("Snapshot: 1 hour of continuous N+S firing")
print("=" * 78)
results_snapshot = integrator.evaluate(geometry, firing_duration_s=3600.0)

# Pick the worst (most-thinned) interconnect for reporting
worst = max(results_snapshot, key=lambda r: r.total_thinning_m)
print(f"  Worst-case interconnect:")
print(f"    array={worst.array_index} ic_idx={worst.interconnect_index} mat={worst.material}")
print(f"    j_i             = {worst.j_i:.3e} A/m^2")
print(f"    incidence       = {worst.incidence_angle_deg:.1f} deg")
print(f"    mean ion E      = {worst.mean_E_eV:.1f} eV")
print(f"    sheath boost    = {worst.sheath_boost_eV:.1f} eV")
print(f"    erosion rate    = {worst.erosion_rate_m_s:.3e} m/s "
      f"= {worst.erosion_rate_m_s*1e9*3600:.3f} nm/hour")
print(f"    fluence (1h)    = {worst.fluence_ions_m2:.3e} ions/m^2")
print(f"    thinning (1h)   = {worst.total_thinning_m*1e9:.3f} nm")

# ---------------------------------------------------------------------------
# 5. Mission profile: 3500 hr/yr * 15 yr = 52500 hr total firing
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("Mission lifetime analysis (15 yr, 3500 hr/yr NSSK)")
print("=" * 78)

mission = MissionProfile(phases=[
    FiringPhase(
        name="NSSK_15yr",
        geometry=geometry,
        duration_s=15 * 3500 * 3600.0,
        duty_cycle=1.0,
    ),
])

life = LifetimeAnalysis(
    integrator=integrator,
    thermal=ThermalCycling(
        n_cycles_per_year=90.0,    # GEO eclipse season
        delta_T=120.0,
        fatigue_exponent=2.0,
    ),
)

life_pred = life.life_prediction(mission, initial_thickness=25e-6)
worst_key = max(life_pred, key=lambda k: life_pred[k]["thinning_m"])
worst_pred = life_pred[worst_key]
print(f"  Worst interconnect (array {worst_key[0]}, ic {worst_key[1]}):")
print(f"    EOL thinning         = {worst_pred['thinning_m']*1e6:.2f} um")
print(f"    fraction remaining   = {worst_pred['fraction_remaining']*100:.1f} %")
print(f"    thermal life factor  = {worst_pred['thermal_life_factor']:.3f}")
print(f"    coupled life factor  = {worst_pred['coupled_life_factor']:.3f}")

# ---------------------------------------------------------------------------
# 6. Bayesian Monte Carlo over Ag yield parameters (Zameshin & Sturm style)
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("Bayesian MC: 200 samples of (Q, s, Eth) posterior for Xe -> Ag")
print("=" * 78)

posteriors = {
    "Ag": ParameterPosterior(
        posterior=MATERIALS["Ag"].bayesian["Xe"],
        rho_Q_Eth=-0.4,
    ),
}
mc = MonteCarloErosion(
    integrator=integrator,
    posteriors=posteriors,
    n_samples=200,
    seed=42,
)

mc_results = mc.run(
    geometry=geometry,
    firing_duration_s=15 * 3500 * 3600.0,  # full mission
    projectile="Xe",
)

# Find worst column
mean = mc_results["mean_thinning"]
p5 = mc_results["p5"]
p50 = mc_results["p50"]
p95 = mc_results["p95"]
keys = mc_results["keys"]
j_worst = int(np.argmax(mean))
print(f"  Worst column key: array={keys[j_worst][0]}, ic={keys[j_worst][1]}")
print(f"    mean thinning   = {mean[j_worst]*1e6:.2f} um")
print(f"    5/50/95 pctile  = "
      f"{p5[j_worst]*1e6:.2f} / {p50[j_worst]*1e6:.2f} / {p95[j_worst]*1e6:.2f} um")

# ---------------------------------------------------------------------------
# 7. Cant-angle sensitivity
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("Sensitivity to N-thruster cant angle (rotation about body Z axis)")
print("=" * 78)

cant_sweep = np.array([-30, -20, -10, 0, 10, 20, 30])
sens = life.sensitivity_to_cant_angle(
    base_profile=mission,
    thruster_index=0,            # the +Y/+Z thruster
    angles_deg=cant_sweep,
    rotation_axis="x",           # rotate about body X to vary the N/S cant
)
for ang, thinning in sens.items():
    worst_t = max(thinning.values())
    print(f"  delta_cant = {ang:+.1f} deg -> worst EOL thinning = "
          f"{worst_t*1e6:.2f} um")

print("\nDone.")
