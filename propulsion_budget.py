# propulsion_budget.py
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Optional, List, Dict, Any

G0 = 9.80665  # m/s^2


@dataclass
class Maneuver:
    name: str
    dv_mps: float
    margin: float
    isp_s: float
    thrust_N: Optional[float]
    thruster: str
    use_dv_with_margin: bool = True


def dv_effective(m: Maneuver) -> float:
    return m.dv_mps * (1.0 + m.margin) if m.use_dv_with_margin else m.dv_mps


def rocket_mass_before(m_after: float, dv: float, isp: float) -> float:
    return m_after * exp(dv / (isp * G0))


def mdot_from_thrust(thrust_N: float, isp_s: float) -> float:
    return thrust_N / (isp_s * G0)


def budget_from_dry_mass(dry_mass_kg: float, maneuvers: List[Maneuver]) -> Dict[str, Any]:
    """
    Exact mass chaining from dry mass backwards.
    Returns per-maneuver prop, impulse, burn time, and totals by thruster.
    """
    rows = []
    m_after = dry_mass_kg
    totals_by_thruster: Dict[str, float] = {}
    totals_impulse_by_thruster_Ns: Dict[str, float] = {}

    for mnv in reversed(maneuvers):
        dv = dv_effective(mnv)
        m_before = rocket_mass_before(m_after, dv, mnv.isp_s)
        m_prop = m_before - m_after

        impulse_Ns = m_prop * mnv.isp_s * G0  # approx total delivered impulse
        burn_time_s = None
        if mnv.thrust_N is not None and mnv.thrust_N > 0:
            mdot = mdot_from_thrust(mnv.thrust_N, mnv.isp_s)
            burn_time_s = m_prop / mdot if mdot > 0 else None

        totals_by_thruster[mnv.thruster] = totals_by_thruster.get(mnv.thruster, 0.0) + m_prop
        totals_impulse_by_thruster_Ns[mnv.thruster] = totals_impulse_by_thruster_Ns.get(mnv.thruster, 0.0) + impulse_Ns

        rows.append({
            "name": mnv.name,
            "thruster": mnv.thruster,
            "dv_base_mps": mnv.dv_mps,
            "margin": mnv.margin,
            "dv_used_mps": dv,
            "isp_s": mnv.isp_s,
            "thrust_N": mnv.thrust_N,
            "m_before_kg": m_before,
            "m_after_kg": m_after,
            "m_prop_kg": m_prop,
            "impulse_kNs": impulse_Ns / 1e3,
            "burn_time_s": burn_time_s,
        })
        m_after = m_before

    rows = list(reversed(rows))
    wet_mass_kg = rows[0]["m_before_kg"] if rows else dry_mass_kg
    total_prop_kg = wet_mass_kg - dry_mass_kg

    return {
        "dry_mass_kg": dry_mass_kg,
        "wet_mass_kg": wet_mass_kg,
        "total_prop_kg": total_prop_kg,
        "totals_by_thruster": totals_by_thruster,
        "totals_impulse_by_thruster_kNs": {k: v/1e3 for k, v in totals_impulse_by_thruster_Ns.items()},
        "rows": rows,
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


def print_report(res: Dict[str, Any], available_prop: Optional[Dict[str, float]] = None) -> None:
    print("\n=== Maneuver budget (computed) ===")
    header = f"{'Maneuver':38s} {'Thr':8s} {'DV used':>8s} {'Isp':>6s} {'Prop':>8s} {'Imp[kNs]':>9s} {'Time':>8s}"
    print(header)
    print("-" * len(header))
    for r in res["rows"]:
        print(f"{r['name'][:38]:38s} {r['thruster'][:8]:8s} "
              f"{r['dv_used_mps']:8.2f} {r['isp_s']:6.0f} "
              f"{r['m_prop_kg']:8.3f} {r['impulse_kNs']:9.2f} {fmt_time(r['burn_time_s']):>8s}")

    print("\n=== Totals ===")
    print(f"Dry mass:  {res['dry_mass_kg']:.3f} kg")
    print(f"Wet mass:  {res['wet_mass_kg']:.3f} kg")
    print(f"Total prop:{res['total_prop_kg']:.3f} kg")

    print("\nPropellant totals by thruster:")
    for thr, m in sorted(res["totals_by_thruster"].items()):
        if available_prop and thr in available_prop:
            margin = available_prop[thr] - m
            status = "OK" if margin >= 0 else "OVER"
            print(f"  {thr:8s}: {m:.3f} kg   (available {available_prop[thr]:.3f} kg) => {status} by {margin:.3f} kg")
        else:
            print(f"  {thr:8s}: {m:.3f} kg")

    print("\nImpulse totals by thruster:")
    for thr, imp in sorted(res["totals_impulse_by_thruster_kNs"].items()):
        print(f"  {thr:8s}: {imp:.2f} kN·s")
