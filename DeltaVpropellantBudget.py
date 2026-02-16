#!/usr/bin/env python3
"""
Delta-V + propellant budget for mixed propulsion modes.

Assumptions:
- Hohmann transfer for GEO+400 -> GEO-300 (coplanar, impulsive equivalent ΔV)
- No gravity/steering losses, no finite-burn losses
- Constant Isp per maneuver
- "dry_mass_kg" is mass after *all* propellant is depleted
"""

from dataclasses import dataclass
from math import sqrt, exp
from typing import List, Dict, Optional

G0 = 9.80655 # m/s²
MU_EARTH = 398600.4418e9 # m³/s²
R_EARTH = 6378.137e3     # m
R_GEO = 42164e3          # m (geocentric radius)

@dataclass
class Maneuver:
    name: str
    dv_mps: float
    isp_s: float
    thruster: str        # "EP", "CP-cold", "CP-hot"
    thrust_N: Optional[float] = None # if provided, we estimate burn time

def hohmann_dv_circular(r1_m: float, r2_m:float, mu: float = MU_EARTH) -> float:
    """Total DeltaV (m/s) for coplanar Hohmann transfer between circular orbits r1->r2"""
    v1 = sqrt(mu / r1_m)
    v2 = sqrt(mu / r2_m)

    a_t = 0.5 * (r1_m + r2_m)
    v_peri = sqrt(mu * (2.0 / r1_m - 1.0 / a_t))
    v_apo  = sqrt(mu * (2.0 / r2_m - 1.0 / a_t))
    dv1 = abs(v_peri - v1)
    dv2 = abs(v2 - v_apo)

    return dv1 + dv2


def rocket_mass_before(m_after: float, dv: float, isp: float, g0: float = G0) -> float:
    """Given mass after maneuver, compute mass before maneuver."""
    return m_after * exp(dv / (isp * g0))


def mdot_from_thrust(thrust_N: float, isp_s: float, g0: float = G0) -> float:
    """Mass flow rate (kg/s) from thrust and Isp."""
    return thrust_N / (isp_s * g0)


def budget_from_dry_mass(dry_mass_kg: float, maneuvers: List[Maneuver]) -> Dict:
    """
    Compute propellant per maneuver by propagating mass backward from dry mass.
    Return detailed rows + totals.
    """
    rows = []
    m_after = dry_mass_kg
    totals_by_thruster = {}

    # Working backwards from last maneuver executed 
    # is accounted first in this reverse loop 
    for mnv in reversed(maneuvers):
        m_before = rocket_mass_before(m_after, mnv.dv_mps, mnv.isp_s)
        m_prop = m_before - m_after

        burn_time_s = None
        if mnv.thrust_N is not None and mnv.thrust_N > 0:
            mdot = mdot_from_thrust(mnv.thrust_N, mnv.isp_s)
            burn_time_s = m_prop / mdot if mdot > 0 else None

        totals_by_thruster[mnv.thruster] = totals_by_thruster.get(mnv.thruster, 0.0) + m_prop

        rows.append({
            "name": mnv.name,
            "thruster": mnv.thruster,
            "dv_mps": mnv.dv_mps,
            "isp_s": mnv.isp_s,
            "m_before_kg": m_before,
            "m_after_kg": m_after,
            "m_prop_kg": m_prop,
            "burn_time_s": burn_time_s,
        })

        m_after = m_before


    # rows currently reversed (because we iterated reversed(maneuvers))
    rows = list(reversed(rows))
    wet_mass_kg = rows[0]["m_before_kg"] if rows else dry_mass_kg
    total_prop_kg = wet_mass_kg - dry_mass_kg


    return {
        "dry_mass_kg": dry_mass_kg,
        "wet_mass_kg": wet_mass_kg,
        "total_prop_kg": total_prop_kg,
        "totals_by_thruster": totals_by_thruster,
        "rows": rows   
    }


def fmt_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    
    days = seconds / 86400.0
    if days >= 1:
        return f"{days:.2f} d"
    hours = seconds / 3600.0
    if hours >= 1:
        return f"{hours:.2f} h"
    return f"{seconds:.0f} s"


def main():
    # ------------------------------
    #           USER INPUTS
    # ------------------------------
    dry_mass_kg = 150.0

    years_stationkeeping = 5.0      # mission duration in years
    dv_nssk_per_year = 55.0         # m/s per year

    # EP properties
    isp_ep = 1200.0                 # s
    thrust_ep = 7.5e-3              # N (7.5 mN)

    # CP properties
    isp_cold = 50.0                 # s
    thrust_cp_cold = 1.03           # N
    isp_hot  = 253.0                # s
    thrust_cp_hot  = 1.03           # N 

    # Detumble + AOCS/safe-mode budgets 
    # placeholder values for now
    dv_detumble = 0.42               # m/s
    dv_attitude_and_safemode = 10.0 # m/s total

    # Which Thruster does what?
    stationkeeping_thruster = "EP"  
    aocs_thruster = "CP-hot"  # could be "CP-cold"

    # ---------------------------------
    # Orbit Change GEO+400 -> GEO-300
    # ---------------------------------
    r1 = R_GEO + 400e3
    r2 = R_GEO - 300e3
    dv_eor = hohmann_dv_circular(r1, r2) # m/s (impulsive equivalent)

    # Resolve ISPs per chosen thrusters
    thruster_isp = {
        "EP": isp_ep,
        "CP-cold": isp_cold,
        "CP-hot": isp_hot
    }

    thruster_thrust = {
        "EP": thrust_ep,
        "CP-cold": thrust_cp_cold,
        "CP-hot": thrust_cp_hot
    }

    maneuvers = []

    # Electric Orbit Raising / lowering using EOR
    maneuvers.append(Maneuver(
        name="GEO+400 -> GEO-300 (Hohmann equiv.)",
        dv_mps=dv_eor,
        isp_s=thruster_isp["EP"],
        thruster="EP",
        thrust_N=thruster_thrust.get("EP")
    )) 

    # North/South station keeping
    dv_nssk_total = dv_nssk_per_year * years_stationkeeping
    maneuvers.append(Maneuver(
        name=f"North/South station keeping ({years_stationkeeping:g} yr)",
        dv_mps=dv_nssk_total,
        isp_s=thruster_isp[stationkeeping_thruster],
        thruster=stationkeeping_thruster,
        thrust_N=thruster_thrust.get(stationkeeping_thruster)
    ))

    # Detumbling (Cold gas)
    maneuvers.append(Maneuver(
        name="Detumble",
        dv_mps=dv_detumble,
        isp_s=thruster_isp["CP-cold"],
        thruster="CP-cold",
        thrust_N=thruster_thrust.get("CP-cold")
    ))

    # Attitude keeping + safe mode (CP hot)
    maneuvers.append(Maneuver(
        name="Attitude keeping + safe mode",
        dv_mps=dv_attitude_and_safemode,
        isp_s=thruster_isp[aocs_thruster],
        thruster=aocs_thruster,
        thrust_N=thruster_thrust.get(aocs_thruster)
    ))

    # ---------------------------------
    #        Compute Budget               
    # ---------------------------------
    result = budget_from_dry_mass(dry_mass_kg, maneuvers)

    # ---------------------------------
    #      Print Report               
    # ---------------------------------
    print("\n ==== Inputs ====")
    print(f"Dry mass: {result['dry_mass_kg']:.3f} kg")
    print(f"Mission Duration: {years_stationkeeping} years")
    print(f"EOR transfer: r1={r1/1e3:.1f} km, r2={r2/1e3:.1f} km, ΔV={dv_eor:.3f} m/s")

    print("\n=== Maneuver-by-maneuver budget ===")
    header = f"{'Maneuver':40s} {'Thruster':10s} {'ΔV (m/s)':>10s} {'Isp (s)':>8s} {'Prop (kg)':>10s} {'Burn time':>10s}"
    print(header)
    print("-" * len(header))
    for row in result["rows"]:
        print(f"{row['name'][:40]:40s}" 
              f"{row['thruster'][:10]:10s} "
              f"{row['dv_mps']:10.3f} "
              f"{row['isp_s']:8.1f} "
              f"{row['m_prop_kg']:10.4f} "
              f"{fmt_time(row['burn_time_s']):>10s}")
        
    
    print("\n=== Totals ===")
    print(f"Wet mass needed: {result['wet_mass_kg']:.3f} kg")
    print(f"Total prop mass: {result['total_prop_kg']:.3f} kg")
    print("\nPropellant by thruster: ")
    for k, v in sorted(result["totals_by_thruster"].items()):
        print(f" {k:10s}: {v:.3f} kg")


if __name__ == "__main__":
    main()
