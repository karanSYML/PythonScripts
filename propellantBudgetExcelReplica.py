#!/usr/bin/env python3
from dataclasses import dataclass
from math import exp
from typing import Optional, List, Dict, Any

G0 = 9.80665  # m/s^2

# TODO
# Add into EP budget
# Residuals : 0.805 kg
# Leakage : 0.110 kg
# Overheads : 0.615 kg


@dataclass
class Maneuver:
    name: str
    dv_mps: float                 # base DV (without margin)
    margin: float                 # e.g. 0.05 for +5%
    isp_s: float
    thrust_N: Optional[float]
    thruster: str                 # "EP" / "CP-cold" / "CP-hot"
    use_dv_with_margin: bool = True  # if True: dv*(1+margin), else dv

def dv_effective(m: Maneuver) -> float:
    return m.dv_mps * (1.0 + m.margin) if m.use_dv_with_margin else m.dv_mps

def rocket_mass_before(m_after: float, dv: float, isp: float) -> float:
    # m_before = m_after * exp(dv/(Isp*g0))
    return m_after * exp(dv / (isp * G0))

def mdot_from_thrust(thrust_N: float, isp_s: float) -> float:
    return thrust_N / (isp_s * G0)

def budget_from_dry_mass(dry_mass_kg: float, maneuvers: List[Maneuver]) -> Dict[str, Any]:
    """
    Work backwards from dry mass to compute exact propellant per maneuver,
    consistent with the rocket equation.
    """
    rows = []
    m_after = dry_mass_kg
    totals_by_thruster: Dict[str, float] = {}
    totals_impulse_by_thruster: Dict[str, float] = {}

    # Backward propagation
    for mnv in reversed(maneuvers):
        dv = dv_effective(mnv)
        m_before = rocket_mass_before(m_after, dv, mnv.isp_s)
        m_prop = m_before - m_after

        # Total impulse delivered (approx): I = m_prop * Isp * g0  (N·s)
        impulse_Ns = m_prop * mnv.isp_s * G0

        burn_time_s = None
        if mnv.thrust_N is not None and mnv.thrust_N > 0:
            mdot = mdot_from_thrust(mnv.thrust_N, mnv.isp_s)
            burn_time_s = m_prop / mdot if mdot > 0 else None

        totals_by_thruster[mnv.thruster] = totals_by_thruster.get(mnv.thruster, 0.0) + m_prop
        totals_impulse_by_thruster[mnv.thruster] = totals_impulse_by_thruster.get(mnv.thruster, 0.0) + impulse_Ns

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
        "totals_impulse_by_thruster_kNs": {k: v/1e3 for k, v in totals_impulse_by_thruster.items()},
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

def print_report(res: Dict[str, Any], available_prop: Dict[str, float]) -> None:
    print("\n=== Maneuver budget (computed) ===")
    header = f"{'Maneuver':38s} {'Thr':8s} {'DV used':>8s} {'Isp':>6s} {'Prop':>8s} {'Imp[kNs]':>9s} {'Time':>8s}"
    print(header)
    print("-" * len(header))
    for r in res["rows"]:
        print(f"{r['name'][:38]:38s} {r['thruster'][:8]:8s} "
              f"{r['dv_used_mps']:8.2f} {r['isp_s']:6.0f} "
              f"{r['m_prop_kg']:8.3f} {r['impulse_kNs']:9.2f} {fmt_time(r['burn_time_s']):>8s}")

    print("\n=== Totals ===")
    print(f"Dry mass: {res['dry_mass_kg']:.3f} kg")
    print(f"Wet mass: {res['wet_mass_kg']:.3f} kg")
    print(f"Total prop: {res['total_prop_kg']:.3f} kg")

    print("\nPropellant totals by thruster:")
    for thr, m in sorted(res["totals_by_thruster"].items()):
        avail = available_prop.get(thr)
        if avail is None:
            print(f"  {thr:8s}: {m:.3f} kg")
        else:
            margin = avail - m
            status = "OK" if margin >= 0 else "OVER"
            print(f"  {thr:8s}: {m:.3f} kg   (available {avail:.3f} kg)  => {status} by {margin:.3f} kg")

    print("\nImpulse totals by thruster:")
    for thr, imp in sorted(res["totals_impulse_by_thruster_kNs"].items()):
        print(f"  {thr:8s}: {imp:.2f} kN·s")

def main():
    # ---- From your Excel assumptions ----
    dry_mass_kg = 144.0

    # Available prop (sheet: EP=8.0, CP=2.9)
    available_prop = {
        "EP": 8.0,
        "CP": 2.9,        # we lump CP-cold and CP-hot into "CP" below for comparison
    }

    # Thruster properties (your sheet)
    EP_ISP = 1100.0
    EP_THRUST = 15.5e-3

    CP_COLD_ISP = 55.0
    CP_HOT_ISP = 253.0
    CP_THRUST = 1.03  # table uses ~1.030 N
    CP_THRUST_COLD = 14.0e-3 # table uses ~1.0N

    # ---- Paste mission-phase table here (this matches rows) ----
    # NOTE: We keep CP as one "bucket" named "CP" for availability check.
    maneuvers = [
        Maneuver("EOR Injection -> subGEO", 26.00, 0.05, EP_ISP, EP_THRUST, "EP"),
        Maneuver("Detumbling & Commissioning", 0.51, 0.05, CP_COLD_ISP, CP_THRUST, "CP"),
        Maneuver("WoL due to Perturbations", 1.01e-03, 0.10, CP_HOT_ISP, CP_THRUST, "CP"),
        Maneuver("WoL due to Maneuvers", 2.95, 0.30, CP_HOT_ISP, CP_THRUST, "CP"),
        Maneuver("Acquisition and Safe Mode", 4.61, 0.05, CP_COLD_ISP, CP_THRUST, "CP"),
        Maneuver("Far Range RdV EP (60-1 km)", 150.00, 0.30, EP_ISP, EP_THRUST, "EP"),
        Maneuver("Far Range RdV CP (60-1 km)", 11.47, 0.30, CP_HOT_ISP, CP_THRUST, "CP"),
        Maneuver("Altitude Changes (+/-300 km)", 88.00, 0.05, EP_ISP, EP_THRUST, "EP"),
        Maneuver("Station keeping NS (5 yrs)", 275.00, 0.05, EP_ISP, EP_THRUST, "EP"),
        Maneuver("Station keeping EW", 0.00, 0.05, EP_ISP, EP_THRUST, "EP"),
        Maneuver("Disposal", 26.00, 0.35, EP_ISP, EP_THRUST, "EP"),
    ]

    res = budget_from_dry_mass(dry_mass_kg, maneuvers)
    print_report(res, available_prop)

    # dv_detumble = dv_effective(maneuvers[1])
    # dv_attitude_and_safemode = dv_effective(maneuvers[4])

    # print("\n=== Key extracted budgets from earlier script ===")
    # print(f"dv_detumble (Detumbling & Commissioning, with margin): {dv_detumble:.3f} m/s")
    # print(f"dv_attitude_and_safemode (Acquisition+Safe Mode, with margin): {dv_attitude_and_safemode:.3f} m/s")

if __name__ == "__main__":
    main()
