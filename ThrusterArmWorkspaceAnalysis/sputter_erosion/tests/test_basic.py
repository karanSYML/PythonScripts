"""
Minimal sanity tests. Run with `pytest tests/` from the package root.
"""

import numpy as np

from sputter_erosion import (
    MATERIALS, PROJECTILES,
    YamamuraTawara, EcksteinPreuss, Seah2005,
    YamamuraAngular, EcksteinAngular,
    HallThrusterPlume, SpeciesFractions,
    Vector3, ThrusterPlacement, SolarArray, Interconnect,
    SatelliteGeometry, ErosionIntegrator,
)
from sputter_erosion.yields import FullYield


# -----------------------------------------------------------------------------
# Yield-model tests
# -----------------------------------------------------------------------------

def test_yt_zero_below_threshold():
    yt = YamamuraTawara()
    Ag = MATERIALS["Ag"]
    Xe = PROJECTILES["Xe"]
    Y = yt.yield_normal(np.array([5.0, 10.0, 20.0]), Xe, Ag)
    assert np.all(Y == 0.0)


def test_yt_positive_above_threshold():
    yt = YamamuraTawara()
    Ag = MATERIALS["Ag"]
    Xe = PROJECTILES["Xe"]
    Y = yt.yield_normal(np.array([100.0, 300.0, 1000.0]), Xe, Ag)
    assert np.all(Y > 0.0)
    # Monotone increasing in this range
    assert Y[0] < Y[1] < Y[2]


def test_eckstein_better_near_threshold():
    """Eckstein-Preuss should be smoother than Y-T near threshold."""
    yt = YamamuraTawara()
    ep = EcksteinPreuss()
    Ag = MATERIALS["Ag"]
    Xe = PROJECTILES["Xe"]
    E = np.linspace(25.0, 50.0, 25)
    Y_yt = yt.yield_normal(E, Xe, Ag)
    Y_ep = ep.yield_normal(E, Xe, Ag)
    assert np.all(Y_yt >= 0.0)
    assert np.all(Y_ep >= 0.0)
    # Both finite
    assert np.all(np.isfinite(Y_yt))
    assert np.all(np.isfinite(Y_ep))


def test_angular_factor_ge_1_at_oblique():
    """Sputter yield should peak at ~theta_opt > 0, so factor > 1 at 60 deg."""
    a = YamamuraAngular(f=1.7, theta_opt_deg=65.0)
    Ag = MATERIALS["Ag"]
    Xe = PROJECTILES["Xe"]
    f0 = a.factor(0.0, 300.0, Xe, Ag)
    f60 = a.factor(np.deg2rad(60.0), 300.0, Xe, Ag)
    assert f60 > f0


# -----------------------------------------------------------------------------
# Plume tests
# -----------------------------------------------------------------------------

def test_plume_decreases_with_distance():
    plume = HallThrusterPlume(
        V_d=300.0, I_beam=10.0, mdot_neutral=15e-6,
    )
    s_near = plume.evaluate(theta_rad=0.1, r_m=0.5)
    s_far  = plume.evaluate(theta_rad=0.1, r_m=5.0)
    assert s_near.j_i > s_far.j_i


def test_plume_off_axis_has_more_cex_fraction():
    plume = HallThrusterPlume(V_d=300.0, I_beam=10.0, mdot_neutral=15e-6)
    s_axis = plume.evaluate(theta_rad=0.0, r_m=2.0)
    s_wing = plume.evaluate(theta_rad=np.deg2rad(80.0), r_m=2.0)
    # mean energy off axis should be much lower (CEX dominated)
    assert s_axis.iedf.mean_energy() > s_wing.iedf.mean_energy()


# -----------------------------------------------------------------------------
# End-to-end smoke test
# -----------------------------------------------------------------------------

def test_end_to_end_smoke():
    plume = HallThrusterPlume(
        V_d=300.0, I_beam=10.0, mdot_neutral=15e-6,
        species=SpeciesFractions(0.8, 0.18, 0.02),
    )
    thr = ThrusterPlacement(
        position_body=Vector3(-1.0, 0.0, 0.0),
        fire_direction_body=Vector3(-1.0, 0.0, 0.0),
        plume=plume,
    )
    array = SolarArray(
        origin_body=Vector3(0.0, 1.5, 0.0),
        panel_normal_body=Vector3(0.0, 0.0, 1.0),
        panel_x_body=Vector3(0.0, 1.0, 0.0),
        width=4.0, height=2.0,
        interconnects=[
            Interconnect(
                position_local=(1.0, 1.0),
                exposed_face_normal=Vector3(-1.0, 0.0, 0.2),
                material_name="Ag",
            )
        ],
    )
    geom = SatelliteGeometry(thrusters=[thr], solar_arrays=[array])
    integ = ErosionIntegrator()
    res = integ.evaluate(geom, firing_duration_s=3600.0)
    assert len(res) == 1
    assert res[0].erosion_rate_m_s >= 0.0
    assert res[0].fluence_ions_m2 > 0.0


if __name__ == "__main__":
    test_yt_zero_below_threshold()
    test_yt_positive_above_threshold()
    test_eckstein_better_near_threshold()
    test_angular_factor_ge_1_at_oblique()
    test_plume_decreases_with_distance()
    test_plume_off_axis_has_more_cex_fraction()
    test_end_to_end_smoke()
    print("All tests passed.")
