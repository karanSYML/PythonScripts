"""
Example: drive geo_pvt_generator using a ground-track specification instead
of raw Keplerian elements.  The propagator script itself is unchanged.
"""

from pathlib import Path

# Re-use the existing pipeline as-is
from geo_pvt_generator import (
    SpacecraftConfig, PropagationConfig,
    _parse_utc, build_initial_orbit, build_propagator, propagate_and_export,
)
from groundtrack_to_elements import (
    GroundTrackShape, shape_to_elements, fit_groundtrack,
)


def run_with_groundtrack_mode1():
    """Mode 1: define the figure-8 shape, get PVT."""
    shape = GroundTrackShape(
        sub_satellite_lon_deg=1.5,    # parking longitude of IO sat
        inclination_deg=0.05,         # north-south amplitude
        eccentricity=2.0e-4,          # east-west amplitude
        inc_node_dir_deg=0.0,         # figure-8 tilt
        ecc_perigee_dir_deg=90.0,     # symmetric (perigee at desc. pass)
    )
    orbit_cfg = shape_to_elements(shape)

    sc_cfg = SpacecraftConfig(mass_kg=3000.0, srp_area_m2=40.0, srp_cr=1.5)
    prop_cfg = PropagationConfig(
        start_utc="2028-09-01T00:00:00.000",
        end_utc="2028-09-02T00:00:00.000",
        step_s=60.0,
    )

    epoch = _parse_utc(prop_cfg.start_utc)
    end   = _parse_utc(prop_cfg.end_utc)
    initial_orbit = build_initial_orbit(orbit_cfg, epoch)
    propagator    = build_propagator(initial_orbit, sc_cfg, prop_cfg)

    out = Path("io_sat_geo_pvt_from_groundtrack.csv")
    n = propagate_and_export(propagator, epoch, end, prop_cfg.step_s, out)
    print(f"Mode 1: wrote {n} rows to {out.resolve()}")


def run_with_groundtrack_mode2():
    """Mode 2: provide (lat, lon) waypoints, fit elements, propagate."""
    # Example waypoints describing a small figure-8 centered on 1.5 deg E
    waypoints = [
        ( 0.05,  1.50),
        ( 0.04,  1.52),
        ( 0.00,  1.50),
        (-0.04,  1.48),
        (-0.05,  1.50),
        (-0.04,  1.52),
        ( 0.00,  1.50),
        ( 0.04,  1.48),
    ]
    orbit_cfg = fit_groundtrack(waypoints, verbose=True)

    sc_cfg = SpacecraftConfig(mass_kg=3000.0)
    prop_cfg = PropagationConfig(
        start_utc="2028-09-01T00:00:00.000",
        end_utc="2028-09-02T00:00:00.000",
        step_s=60.0,
    )

    epoch = _parse_utc(prop_cfg.start_utc)
    end   = _parse_utc(prop_cfg.end_utc)
    initial_orbit = build_initial_orbit(orbit_cfg, epoch)
    propagator    = build_propagator(initial_orbit, sc_cfg, prop_cfg)

    out = Path("io_sat_geo_pvt_fitted.csv")
    n = propagate_and_export(propagator, epoch, end, prop_cfg.step_s, out)
    print(f"Mode 2: wrote {n} rows to {out.resolve()}")


if __name__ == "__main__":
    run_with_groundtrack_mode1()
    # run_with_groundtrack_mode2()
