"""
GEO satellite PVT ephemeris generator.

Propagates a GEO satellite using a numerical propagator with:
    - Earth gravity field (Holmes-Featherstone, configurable degree/order)
    - Sun and Moon third-body perturbations (point mass)
    - Solar radiation pressure (isotropic spherical model)
    - Atmospheric drag hook (commented out — negligible at GEO)

Propagation is performed in GCRF (inertial); output is transformed to
ITRF and written as CSV with UTC timestamps.

Output columns: utc_iso, x_m, y_m, z_m, vx_mps, vy_mps, vz_mps

Requires: orekit (Python wrapper) + an orekit-data.zip in the working
directory (or set OREKIT_DATA_PATH).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import orekit
from orekit import JArray_double
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime

# Initialise the JVM before importing any org.* classes
orekit.initVM()
setup_orekit_curdir()  # expects orekit-data.zip in cwd

from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.forces.gravity import (
    HolmesFeatherstoneAttractionModel,
    ThirdBodyAttraction,
)
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.radiation import (
    IsotropicRadiationSingleCoefficient,
    SolarRadiationPressure,
)
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, OrbitType, PositionAngleType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class OrbitConfig:
    """Initial Keplerian elements (epoch matches `start_utc`)."""
    sma_m: float          # semi-major axis [m]
    ecc: float            # eccentricity
    inc_deg: float        # inclination [deg]
    raan_deg: float       # right ascension of ascending node [deg]
    aop_deg: float        # argument of perigee [deg]
    true_anom_deg: float  # true anomaly [deg]


@dataclass
class SpacecraftConfig:
    mass_kg: float = 3000.0       # GEO telecom-class default
    srp_area_m2: float = 40.0     # cross-section for SRP
    srp_cr: float = 1.5           # reflectivity coefficient


@dataclass
class PropagationConfig:
    start_utc: str          # ISO 8601, e.g. "2028-09-01T00:00:00.000"
    end_utc: str            # ISO 8601
    step_s: float = 60.0    # output step
    gravity_degree: int = 4
    gravity_order: int = 4
    # Integrator tolerances
    min_step_s: float = 0.001
    max_step_s: float = 1000.0
    pos_tolerance_m: float = 1.0


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------

def _parse_utc(iso: str) -> AbsoluteDate:
    """Parse an ISO 8601 UTC string into an Orekit AbsoluteDate."""
    utc = TimeScalesFactory.getUTC()
    return AbsoluteDate(iso, utc)


def build_initial_orbit(orbit_cfg: OrbitConfig, epoch: AbsoluteDate):
    """Build an initial KeplerianOrbit in GCRF."""
    from math import radians

    inertial_frame = FramesFactory.getGCRF()
    mu = Constants.EIGEN5C_EARTH_MU  # consistent with Holmes-Featherstone

    return KeplerianOrbit(
        float(orbit_cfg.sma_m),
        float(orbit_cfg.ecc),
        radians(orbit_cfg.inc_deg),
        radians(orbit_cfg.aop_deg),
        radians(orbit_cfg.raan_deg),
        radians(orbit_cfg.true_anom_deg),
        PositionAngleType.TRUE,
        inertial_frame,
        epoch,
        mu,
    )


def build_propagator(
    initial_orbit,
    sc_cfg: SpacecraftConfig,
    prop_cfg: PropagationConfig,
) -> NumericalPropagator:
    """Configure a NumericalPropagator with gravity, third bodies, and SRP."""

    # --- Integrator --------------------------------------------------------
    tolerances = NumericalPropagator.tolerances(
        prop_cfg.pos_tolerance_m,
        initial_orbit,
        OrbitType.CARTESIAN,
    )
    # Orekit-Python returns Java double[] objects that don't always auto-convert
    # to the constructor's expected double[] signature — wrap explicitly.
    abs_tol = JArray_double.cast_(tolerances[0])
    rel_tol = JArray_double.cast_(tolerances[1])
    integrator = DormandPrince853Integrator(
        prop_cfg.min_step_s,
        prop_cfg.max_step_s,
        abs_tol,
        rel_tol,
    )

    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)

    # --- Initial state -----------------------------------------------------
    initial_state = SpacecraftState(initial_orbit, float(sc_cfg.mass_kg))
    propagator.setInitialState(initial_state)

    # --- Earth gravity (J2-J4+ via Holmes-Featherstone) --------------------
    earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        earth_frame,
    )
    gravity_provider = GravityFieldFactory.getNormalizedProvider(
        prop_cfg.gravity_degree, prop_cfg.gravity_order
    )
    propagator.addForceModel(
        HolmesFeatherstoneAttractionModel(earth_frame, gravity_provider)
    )

    # --- Third-body perturbations -----------------------------------------
    sun = CelestialBodyFactory.getSun()
    moon = CelestialBodyFactory.getMoon()
    propagator.addForceModel(ThirdBodyAttraction(sun))
    propagator.addForceModel(ThirdBodyAttraction(moon))

    # --- Solar radiation pressure -----------------------------------------
    srp_model = IsotropicRadiationSingleCoefficient(
        float(sc_cfg.srp_area_m2), float(sc_cfg.srp_cr)
    )
    propagator.addForceModel(
        SolarRadiationPressure(sun, earth, srp_model)
    )

    # --- Atmospheric drag (disabled — negligible at GEO) -------------------
    # from org.orekit.forces.drag import DragForce, IsotropicDrag
    # from org.orekit.models.earth.atmosphere import HarrisPriester
    # atmosphere = HarrisPriester(sun, earth)
    # drag_model = IsotropicDrag(sc_cfg.drag_area_m2, sc_cfg.cd)
    # propagator.addForceModel(DragForce(atmosphere, drag_model))

    return propagator


# -----------------------------------------------------------------------------
# Propagation + export
# -----------------------------------------------------------------------------

def propagate_and_export(
    propagator: NumericalPropagator,
    start: AbsoluteDate,
    end: AbsoluteDate,
    step_s: float,
    output_csv: Path,
) -> int:
    """Step through propagation, transform PV to ITRF, write CSV. Returns row count."""

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    duration_s = end.durationFrom(start)
    if duration_s <= 0:
        raise ValueError("end_utc must be strictly after start_utc")

    n_steps = int(duration_s // step_s) + 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["utc_iso", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]
        )

        for k in range(n_steps):
            t = start.shiftedBy(float(k * step_s))
            if t.durationFrom(end) > 0:
                t = end

            state = propagator.propagate(t)
            # Transform PV from propagation frame (GCRF) to ITRF
            pv_itrf = state.getPVCoordinates(itrf)
            p = pv_itrf.getPosition()
            v = pv_itrf.getVelocity()

            ts = absolutedate_to_datetime(t).isoformat(timespec="milliseconds")
            writer.writerow(
                [ts, p.getX(), p.getY(), p.getZ(),
                 v.getX(), v.getY(), v.getZ()]
            )
            rows += 1

            if t.equals(end):
                break

    return rows


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    # ---- USER INPUTS ----------------------------------------------------
    # IO satellite initial Keplerian elements — fill in your actual values.
    # Defaults below are a generic GEO placeholder (a ≈ 42164 km, near-zero e/i).
    orbit_cfg = OrbitConfig(
        sma_m=42_164_172.0,
        ecc=1.0e-4,
        inc_deg=0.05,
        raan_deg=0.0,
        aop_deg=0.0,
        true_anom_deg=0.0,
    )

    sc_cfg = SpacecraftConfig(
        mass_kg=3000.0,
        srp_area_m2=40.0,
        srp_cr=1.5,
    )

    prop_cfg = PropagationConfig(
        start_utc="2028-09-01T00:00:00.000",
        end_utc="2028-09-02T00:00:00.000",
        step_s=60.0,
        gravity_degree=4,
        gravity_order=4,
    )

    output_csv = Path("io_sat_geo_pvt_itrf.csv")
    # ---------------------------------------------------------------------

    epoch = _parse_utc(prop_cfg.start_utc)
    end   = _parse_utc(prop_cfg.end_utc)

    initial_orbit = build_initial_orbit(orbit_cfg, epoch)
    propagator    = build_propagator(initial_orbit, sc_cfg, prop_cfg)

    n_rows = propagate_and_export(
        propagator, epoch, end, prop_cfg.step_s, output_csv
    )

    print(f"Wrote {n_rows} ephemeris points to {output_csv.resolve()}")


if __name__ == "__main__":
    main()
