"""
Ground-track -> Keplerian elements for GEO satellites.

Provides two inverse-problem modes that produce an `OrbitConfig` consumable
by `geo_pvt_generator.py` without modifying that script.

Mode 1 (analytical):  GroundTrackShape(longitude, inclination, eccentricity,
                                       inc_node_dir_deg, ecc_perigee_dir_deg)
                      -> OrbitConfig
Mode 2 (least-squares fit): list of (lat_deg, lon_deg) waypoints + epoch
                            -> OrbitConfig

Theory
------
For a GEO satellite (a = a_geo, P = sidereal day) with small e and i, the
ground track is a near-closed figure-8 controlled by:

  inclination vector  i_vec = (i*cos(RAAN),  i*sin(RAAN))
  eccentricity vector e_vec = (e*cos(varpi), e*sin(varpi)),  varpi = RAAN + AOP
  mean longitude      lambda = RAAN + AOP + M  (≈ ground-track centroid longitude)

  - |i_vec|     -> peak |latitude|         (figure-8 height,  ~i)
  - |e_vec|     -> peak longitude swing    (figure-8 width,   ~2e in radians)
  - arg(i_vec)  -> orientation of the ascending-node crossing
  - arg(e_vec)  -> location of perigee, controls E/W asymmetry of the 8

This module assumes a = a_geo (geosynchronous) and inverts the relations
above.  Lunisolar drift will perturb i_vec by ~0.85 deg/year; the produced
elements are correct AT EPOCH only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin, sqrt
from typing import Sequence

import numpy as np

# Import the OrbitConfig dataclass from the existing generator so callers get
# exactly the type the propagator expects.
from geo_pvt_generator import OrbitConfig

# Constants
A_GEO = 42_164_172.0          # geosynchronous semi-major axis [m]
MU_EARTH = 3.986004418e14     # [m^3/s^2]
SIDEREAL_DAY_S = 86_164.0905  # [s]
OMEGA_EARTH = 2.0 * np.pi / SIDEREAL_DAY_S  # [rad/s]


# =============================================================================
# Mode 1 — Analytical inversion from figure-8 shape parameters
# =============================================================================

@dataclass
class GroundTrackShape:
    """High-level description of a GEO figure-8 ground track.

    Attributes
    ----------
    sub_satellite_lon_deg : float
        Mean longitude of the figure-8 centroid [deg, -180..180].
    inclination_deg : float
        Peak latitude amplitude of the figure-8 (= orbital inclination) [deg].
    eccentricity : float
        Orbital eccentricity. Peak longitude half-swing ≈ 2*e (rad) ≈ 114.6*e (deg).
    inc_node_dir_deg : float
        Direction of the inclination vector (= RAAN) [deg]. Default 0
        gives an upright figure-8 with ascending node on the +X (vernal eq.).
    ecc_perigee_dir_deg : float
        Direction of the eccentricity vector (= longitude of perigee = RAAN+AOP)
        [deg]. Default 90 puts perigee at the descending pass, giving the
        canonical symmetric figure-8.
    """
    sub_satellite_lon_deg: float
    inclination_deg: float = 0.05
    eccentricity: float = 1.0e-4
    inc_node_dir_deg: float = 0.0
    ecc_perigee_dir_deg: float = 90.0


def shape_to_elements(shape: GroundTrackShape) -> OrbitConfig:
    """Mode 1: invert (i_vec, e_vec, lambda) -> (a, e, i, RAAN, AOP, nu)."""
    i_deg     = float(shape.inclination_deg)
    e         = float(shape.eccentricity)
    raan_deg  = float(shape.inc_node_dir_deg) % 360.0
    varpi_deg = float(shape.ecc_perigee_dir_deg) % 360.0
    lon_deg   = float(shape.sub_satellite_lon_deg)

    aop_deg = (varpi_deg - raan_deg) % 360.0

    # Mean longitude lambda = RAAN + AOP + M (mod 360).  Solve for M at epoch.
    # The "centroid" longitude of the ground track equals lambda - GMST(epoch),
    # but the propagator handles GMST internally via the ITRF transform; what
    # we need is the inertial mean longitude at epoch that *projects* to the
    # desired sub-satellite longitude at epoch.  Since the propagator's start
    # epoch sets GMST, we encode the desired sub-sat longitude directly into
    # M and let the GCRF->ITRF transform place it correctly.  For a GEO this
    # is consistent because the orbital rate equals Earth's rotation rate.
    M_deg = (lon_deg - raan_deg - aop_deg) % 360.0

    # For small e, mean anomaly ≈ true anomaly. Convert exactly via Kepler.
    nu_deg = degrees(_mean_to_true_anomaly(radians(M_deg), e))

    return OrbitConfig(
        sma_m=A_GEO,
        ecc=e,
        inc_deg=i_deg,
        raan_deg=raan_deg,
        aop_deg=aop_deg,
        true_anom_deg=nu_deg,
    )


def _mean_to_true_anomaly(M: float, e: float, tol: float = 1e-12) -> float:
    """Solve Kepler's equation M = E - e sin E, then convert E -> true anomaly."""
    # Newton-Raphson; converges in a handful of iterations for e << 1
    E = M if e < 0.8 else np.pi
    for _ in range(50):
        f  = E - e * sin(E) - M
        fp = 1.0 - e * cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < tol:
            break
    nu = 2.0 * atan2(
        sqrt(1.0 + e) * sin(E / 2.0),
        sqrt(1.0 - e) * cos(E / 2.0),
    )
    return nu % (2.0 * np.pi)


# =============================================================================
# Mode 2 — Least-squares fit from (lat, lon) waypoints
# =============================================================================

def _elements_to_groundtrack(
    elements: OrbitConfig,
    times_s: np.ndarray,
    gmst0_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward model: propagate two-body Keplerian elements -> (lat, lon) at
    each time offset (s) past epoch. Two-body is sufficient for shape fitting
    over <= 1 day; the numerical propagator refines downstream.

    Returns lat_deg, lon_deg arrays.
    """
    a    = elements.sma_m
    e    = elements.ecc
    i    = radians(elements.inc_deg)
    raan = radians(elements.raan_deg)
    aop  = radians(elements.aop_deg)
    nu0  = radians(elements.true_anom_deg)

    # Mean motion
    n = sqrt(MU_EARTH / a ** 3)

    # nu0 -> E0 -> M0
    E0 = 2.0 * atan2(
        sqrt(1.0 - e) * sin(nu0 / 2.0),
        sqrt(1.0 + e) * cos(nu0 / 2.0),
    )
    M0 = E0 - e * sin(E0)

    lats = np.empty_like(times_s)
    lons = np.empty_like(times_s)

    for k, t in enumerate(times_s):
        M = M0 + n * t
        # Solve Kepler
        E = M if e < 0.8 else np.pi
        for _ in range(40):
            dE = (E - e * sin(E) - M) / (1.0 - e * cos(E))
            E -= dE
            if abs(dE) < 1e-12:
                break
        nu = 2.0 * atan2(
            sqrt(1.0 + e) * sin(E / 2.0),
            sqrt(1.0 - e) * cos(E / 2.0),
        )
        r = a * (1.0 - e * cos(E))

        # Position in perifocal frame
        x_pf = r * cos(nu)
        y_pf = r * sin(nu)

        # Rotate to ECI: Rz(-RAAN) * Rx(-i) * Rz(-AOP)
        cO, sO = cos(raan), sin(raan)
        ci, si = cos(i),    sin(i)
        cw, sw = cos(aop),  sin(aop)

        x_eci = (cO * cw - sO * sw * ci) * x_pf + (-cO * sw - sO * cw * ci) * y_pf
        y_eci = (sO * cw + cO * sw * ci) * x_pf + (-sO * sw + cO * cw * ci) * y_pf
        z_eci =  (sw * si)               * x_pf +  (cw * si)                * y_pf

        # ECI -> ECEF: rotate by -GMST(t)
        theta = gmst0_rad + OMEGA_EARTH * t
        ct, st = cos(theta), sin(theta)
        x_ecef =  ct * x_eci + st * y_eci
        y_ecef = -st * x_eci + ct * y_eci
        z_ecef =  z_eci

        # ECEF -> geodetic (spherical approximation good enough for fitting)
        rxy = sqrt(x_ecef ** 2 + y_ecef ** 2)
        lats[k] = degrees(atan2(z_ecef, rxy))
        lons[k] = degrees(atan2(y_ecef, x_ecef))

    return lats, lons


def fit_groundtrack(
    waypoints_lat_lon_deg: Sequence[tuple[float, float]],
    waypoint_times_s: Sequence[float] | None = None,
    initial_guess: GroundTrackShape | None = None,
    verbose: bool = False,
) -> OrbitConfig:
    """Mode 2: least-squares fit of GEO elements to a list of (lat, lon) waypoints.

    Parameters
    ----------
    waypoints_lat_lon_deg : Sequence of (lat_deg, lon_deg)
        Desired ground-track samples, ordered in time.
    waypoint_times_s : Sequence of float, optional
        Time of each waypoint, seconds past epoch. If None, points are assumed
        to be uniformly spaced over one sidereal day.
    initial_guess : GroundTrackShape, optional
        Starting point for the optimizer.  Defaults to the mean longitude
        and amplitude inferred from the waypoints.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise RuntimeError("Mode 2 fitting requires scipy") from e

    pts = np.asarray(waypoints_lat_lon_deg, dtype=float)
    n_pts = len(pts)
    if n_pts < 4:
        raise ValueError("Need at least 4 waypoints to fit 4 free parameters")

    if waypoint_times_s is None:
        times_s = np.linspace(0.0, SIDEREAL_DAY_S, n_pts, endpoint=False)
    else:
        times_s = np.asarray(waypoint_times_s, dtype=float)

    # ---- Initial guess from waypoint statistics -------------------------
    if initial_guess is None:
        lat_amp = float(np.max(np.abs(pts[:, 0])))
        lon_mean = float(np.mean(pts[:, 1]))
        lon_amp_deg = float(np.max(pts[:, 1]) - np.min(pts[:, 1])) / 2.0
        e_guess = max(radians(lon_amp_deg) / 2.0, 1e-6)
        initial_guess = GroundTrackShape(
            sub_satellite_lon_deg=lon_mean,
            inclination_deg=max(lat_amp, 1e-3),
            eccentricity=e_guess,
            inc_node_dir_deg=0.0,
            ecc_perigee_dir_deg=90.0,
        )

    x0 = np.array([
        initial_guess.inclination_deg,
        initial_guess.eccentricity,
        initial_guess.inc_node_dir_deg,
        initial_guess.ecc_perigee_dir_deg,
        initial_guess.sub_satellite_lon_deg,
    ])

    def residuals(x):
        i_deg, e, raan_deg, varpi_deg, lon_deg = x
        # Guard against unphysical region
        if e < 0 or e > 0.5 or i_deg < 0 or i_deg > 30:
            return np.full(2 * n_pts, 1e6)
        shape = GroundTrackShape(
            sub_satellite_lon_deg=lon_deg,
            inclination_deg=i_deg,
            eccentricity=e,
            inc_node_dir_deg=raan_deg,
            ecc_perigee_dir_deg=varpi_deg,
        )
        elements = shape_to_elements(shape)
        lat_pred, lon_pred = _elements_to_groundtrack(elements, times_s)

        # Wrap longitude residual into [-180, 180]
        d_lon = (lon_pred - pts[:, 1] + 180.0) % 360.0 - 180.0
        d_lat = lat_pred - pts[:, 0]
        return np.concatenate([d_lat, d_lon])

    result = least_squares(
        residuals, x0,
        method="lm",
        xtol=1e-10, ftol=1e-10,
        max_nfev=2000,
    )

    if verbose:
        rms = sqrt(np.mean(result.fun ** 2))
        print(f"  fit cost     : {result.cost:.6e}")
        print(f"  fit rms (deg): {rms:.4e}")
        print(f"  iterations   : {result.nfev}")

    fitted = GroundTrackShape(
        sub_satellite_lon_deg=result.x[4],
        inclination_deg=result.x[0],
        eccentricity=result.x[1],
        inc_node_dir_deg=result.x[2],
        ecc_perigee_dir_deg=result.x[3],
    )
    return shape_to_elements(fitted)


# =============================================================================
# Demo / smoke test
# =============================================================================

if __name__ == "__main__":
    print("=== Mode 1: shape -> elements ===")
    shape = GroundTrackShape(
        sub_satellite_lon_deg=1.5,   # IO sat parking longitude
        inclination_deg=0.05,
        eccentricity=2.0e-4,
        inc_node_dir_deg=0.0,
        ecc_perigee_dir_deg=90.0,
    )
    cfg = shape_to_elements(shape)
    print(cfg)

    print("\n=== Mode 2: synthetic waypoints -> elements (round-trip check) ===")
    # Generate ground track from a known shape, sample it, refit
    truth = GroundTrackShape(
        sub_satellite_lon_deg=-15.0,
        inclination_deg=0.10,
        eccentricity=3.0e-4,
        inc_node_dir_deg=30.0,
        ecc_perigee_dir_deg=120.0,
    )
    truth_cfg = shape_to_elements(truth)
    sample_times = np.linspace(0.0, SIDEREAL_DAY_S, 24, endpoint=False)
    lats, lons = _elements_to_groundtrack(truth_cfg, sample_times)
    waypoints = list(zip(lats.tolist(), lons.tolist()))

    fitted_cfg = fit_groundtrack(waypoints, sample_times.tolist(), verbose=True)
    print("\n  truth :", truth_cfg)
    print("  fitted:", fitted_cfg)
