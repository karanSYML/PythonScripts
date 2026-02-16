# dv_models.py
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, exp, log
from typing import Optional, Dict, Tuple
import random

G0 = 9.80665  # m/s^2
MU_EARTH = 398600.4418e9  # m^3/s^2
R_EARTH = 6378.137e3      # m
R_GEO = 42164e3           # m (geocentric)

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


def hohmann_dv_circular(r1_m: float, r2_m: float, mu: float = MU_EARTH) -> float:
    """Total ΔV (m/s) for coplanar Hohmann transfer between circular orbits r1->r2."""
    v1 = sqrt(mu / r1_m)
    v2 = sqrt(mu / r2_m)
    a_t = 0.5 * (r1_m + r2_m)
    v_peri = sqrt(mu * (2.0 / r1_m - 1.0 / a_t))
    v_apo  = sqrt(mu * (2.0 / r2_m - 1.0 / a_t))
    dv1 = abs(v_peri - v1)
    dv2 = abs(v2 - v_apo)
    return dv1 + dv2


def stationkeeping_ns(years: float, dv_per_year_mps: float) -> float:
    """North/South stationkeeping ΔV budget (m/s)."""
    return max(0.0, years) * max(0.0, dv_per_year_mps)


def altitude_change_dv_per_event(delta_a_km: float, dv_ref_mps: float = 11.0, ref_km: float = 300.0) -> float:
    """
    Parametric altitude/semi-major-axis change ΔV per event.

    Calibrated to: 300 km -> 11 m/s (from your Excel derived parameters).
    This is a simple linear calibration (good enough for Phase A budgeting).
    """
    if ref_km <= 0:
        raise ValueError("ref_km must be > 0")
    return abs(delta_a_km) * (dv_ref_mps / ref_km)


def impulse_from_momentum_dump(H_dump_Nms: float, lever_arm_m: float) -> float:
    """
    Convert reaction wheel momentum dump requirement to thruster impulse.
    I [N*s] = H [N*m*s] / r [m]
    """
    if lever_arm_m <= 0:
        raise ValueError("lever_arm_m must be > 0")
    return H_dump_Nms / lever_arm_m


def prop_from_impulse(impulse_Ns: float, isp_s: float) -> float:
    """Prop mass (kg) from impulse (N*s)."""
    if isp_s <= 0:
        raise ValueError("isp_s must be > 0")
    return impulse_Ns / (isp_s * G0)


def dv_from_propellant(m0_kg: float, mprop_kg: float, isp_s: float) -> float:
    """
    Convert prop mass to equivalent ΔV (m/s) at mass m0.
    Uses rocket equation: dv = Isp*g0*ln(m0/(m0-mprop))
    """
    if mprop_kg <= 0:
        return 0.0
    if m0_kg <= 0 or mprop_kg >= m0_kg:
        raise ValueError("Invalid masses for dv_from_propellant")
    return isp_s * G0 * log(m0_kg / (m0_kg - mprop_kg))


def wheel_desat_prop_per_year(
    tau_dist_Nm: float,
    H_threshold_Nms: float,
    lever_arm_m: float,
    isp_s: float,
    seconds_per_year: float = SECONDS_PER_YEAR
) -> Dict[str, float]:
    """
    First-order wheel desaturation model:
      dH/dt = tau_dist (constant)
      dump when H reaches H_threshold

    Returns a dict with:
      dumps_per_year, impulse_per_dump_Ns, total_impulse_per_year_Ns, prop_per_year_kg
    """
    if H_threshold_Nms <= 0:
        return {
            "dumps_per_year": 0.0,
            "impulse_per_dump_Ns": 0.0,
            "total_impulse_per_year_Ns": 0.0,
            "prop_per_year_kg": 0.0,
        }

    dH_per_year = abs(tau_dist_Nm) * seconds_per_year
    dumps_per_year = dH_per_year / H_threshold_Nms

    impulse_per_dump = impulse_from_momentum_dump(H_threshold_Nms, lever_arm_m)
    total_impulse_per_year = dumps_per_year * impulse_per_dump
    prop_per_year = prop_from_impulse(total_impulse_per_year, isp_s)

    return {
        "dumps_per_year": dumps_per_year,
        "impulse_per_dump_Ns": impulse_per_dump,
        "total_impulse_per_year_Ns": total_impulse_per_year,
        "prop_per_year_kg": prop_per_year,
    }


def safe_mode_dv_events(
    years: float,
    mean_events_per_year: float,
    dv_per_event_mps: float,
    stochastic: bool = False,
    rng: Optional[random.Random] = None
) -> Tuple[float, int]:
    """
    Safe-mode ΔV as event-driven budget.

    If stochastic=False:
      N = round(mean_events_per_year * years)
    If stochastic=True:
      N ~ Poisson(mean_events_per_year * years)  (simple sampler)
    Returns: (dv_total_mps, n_events)
    """
    lam = max(0.0, mean_events_per_year) * max(0.0, years)
    if not stochastic:
        n = int(round(lam))
        return n * max(0.0, dv_per_event_mps), n

    # Poisson sampler (Knuth)
    if rng is None:
        rng = random.Random()
    L = exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    n = max(0, k - 1)
    return n * max(0.0, dv_per_event_mps), n


def detumble_prop_or_dv(
    omega0_rad_s: float,
    I_eff_kgm2: float,
    lever_arm_m: float,
    thrust_N: float,
    isp_s: float,
    duty_cycle: float = 1.0,
    m0_kg_for_dv: Optional[float] = None,
) -> Dict[str, float]:
    """
    Conservative detumble model based on angular momentum removal.

    H0 = I_eff * omega0
    Available torque tau = lever_arm * thrust * duty_cycle
    time = H0 / tau
    mdot = thrust / (Isp*g0)
    mprop = mdot * time

    If m0_kg_for_dv is provided, also returns dv_equiv_mps from rocket equation.

    NOTE: This assumes you are using thrust to create pure torque continuously
    (worst-case conservative) and ignores geometry/switching losses beyond duty_cycle.
    """
    omega0 = abs(omega0_rad_s)
    Ieff = abs(I_eff_kgm2)
    r = lever_arm_m
    F = abs(thrust_N)
    dc = max(0.0, min(1.0, duty_cycle))

    if r <= 0 or F <= 0 or isp_s <= 0:
        raise ValueError("lever_arm_m, thrust_N, isp_s must be > 0")

    H0 = Ieff * omega0                    # N*m*s
    tau = r * F * dc                      # N*m (effective)
    t = H0 / tau if tau > 0 else 0.0      # s

    mdot = F / (isp_s * G0)               # kg/s
    mprop = mdot * t                      # kg
    impulse = F * t                       # N*s (useful diagnostic)

    out = {
        "H0_Nms": H0,
        "tau_Nm": tau,
        "time_s": t,
        "impulse_Ns": impulse,
        "mprop_kg": mprop,
    }

    if m0_kg_for_dv is not None and mprop > 0:
        out["dv_equiv_mps"] = dv_from_propellant(m0_kg_for_dv, mprop, isp_s)
    else:
        out["dv_equiv_mps"] = 0.0

    return out
