# mission_scenario.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import random

import dv_models as dv
from propulsion_budget import Maneuver


@dataclass
class Scenario:
    # From your Excel assumptions
    dry_mass_kg: float = 161.0
    years: float = 5.0

    dv_ns_per_year_mps: float = 55.0
    dv_ew_per_year_mps: float = 0.0

    n_alt_transfers_per_year: float = 4.0
    alt_transfer_delta_a_km: float = 300.0  # ±300 km “event size”

    # Rendezvous/inspection counting (these can be wired later if you clarify clients)
    n_insp_in_geo_per_year: float = 5.0
    n_far_range_rdv_attempts_per_client: float = 2.0
    n_inspection_attempts_per_client: float = 2.0
    n_clients: float = 1.0  # default; set if you know

    # Thruster + propulsion properties (from your sheet)
    isp_ep_s: float = 1200.0
    thrust_ep_N: float = 7.5e-3

    isp_cp_cold_s: float = 50.0
    isp_cp_hot_s: float = 253.0
    thrust_cp_N: float = 1.03

    # Lever arm you provided
    lever_arm_m: float = 0.2873786408

    # Margins (match your table style)
    margin_eor: float = 0.05
    margin_detumble: float = 0.05
    margin_wol_pert: float = 0.10
    margin_wol_man: float = 0.30
    margin_safe: float = 0.05
    margin_rdv: float = 0.30
    margin_alt: float = 0.05
    margin_ns: float = 0.05
    margin_ew: float = 0.05
    margin_disposal: float = 0.35

    # --- Parameters for “computed” AOCS models (placeholders until you get more inputs)
    # Wheel desat model
    tau_dist_Nm: float = 2e-6          # disturbance torque (placeholder)
    H_threshold_Nms: float = 10.0      # wheel momentum threshold (placeholder)

    # Safe mode model
    safe_mode_events_per_year: float = 1.0   # placeholder
    safe_mode_dv_per_event_mps: float = 4.0  # placeholder

    # Detumble model
    omega0_deg_s: float = 5.0          # placeholder
    I_eff_kgm2: float = 20.0           # placeholder
    detumble_duty_cycle: float = 0.5   # placeholder

    # Injection/orbit parameters for EOR Hohmann
    r_injection_m: float = dv.R_GEO + 400e3
    r_target_m: float = dv.R_GEO - 300e3


def build_maneuvers(s: Scenario, mode: str = "computed", rng: Optional[random.Random] = None) -> Tuple[List[Maneuver], Dict]:
    """
    Returns:
      maneuvers: list of Maneuver objects for propulsion_budget
      diag: dict of diagnostics (e.g., computed detumble prop, wheel dumps/year, safe events)
    """
    if mode not in ("computed", "excel"):
        raise ValueError("mode must be 'computed' or 'excel'")

    diag: Dict = {}

    # --- EOR (computed via Hohmann unless mode='excel' uses your 26 m/s baseline)
    dv_eor = dv.hohmann_dv_circular(s.r_injection_m, s.r_target_m) if mode == "computed" else 26.0
    maneuvers: List[Maneuver] = [
        Maneuver("EOR Injection -> subGEO", dv_eor, s.margin_eor, s.isp_ep_s, s.thrust_ep_N, "EP"),
    ]

    # --- Detumble & commissioning (computed from angular momentum model OR excel 0.42 m/s)
    if mode == "computed":
        omega0 = math.radians(s.omega0_deg_s)
        det = dv.detumble_prop_or_dv(
            omega0_rad_s=omega0,
            I_eff_kgm2=s.I_eff_kgm2,
            lever_arm_m=s.lever_arm_m,
            thrust_N=s.thrust_cp_N,
            isp_s=s.isp_cp_cold_s,
            duty_cycle=s.detumble_duty_cycle,
            m0_kg_for_dv=None,  # we keep as prop diagnostic; dv budget below is conservative
        )
        diag["detumble"] = det
        # Convert prop->equivalent dv at an approximate mass (use dry+rough prop guess).
        # For now, produce a DV budget directly from prop diagnostic assuming ~ (dry + 10 kg).
        m0_approx = s.dry_mass_kg + 10.0
        dv_detumble = dv.dv_from_propellant(m0_approx, det["mprop_kg"], s.isp_cp_cold_s)
    else:
        dv_detumble = 0.42

    maneuvers.append(Maneuver("Detumbling & Commissioning", dv_detumble, s.margin_detumble, s.isp_cp_cold_s, s.thrust_cp_N, "CP"))

    # --- WoL due to perturbations (wheel desat model OR excel tiny)
    if mode == "computed":
        wd = dv.wheel_desat_prop_per_year(
            tau_dist_Nm=s.tau_dist_Nm,
            H_threshold_Nms=s.H_threshold_Nms,
            lever_arm_m=s.lever_arm_m,
            isp_s=s.isp_cp_hot_s,
        )
        diag["wheel_desat"] = wd
        # Convert yearly prop -> equivalent dv using an approximate mass
        m0_approx = s.dry_mass_kg + 10.0
        mprop_total = wd["prop_per_year_kg"] * s.years
        dv_wol_pert = dv.dv_from_propellant(m0_approx, mprop_total, s.isp_cp_hot_s)
    else:
        dv_wol_pert = 4.01e-3

    maneuvers.append(Maneuver("WoL due to Perturbations", dv_wol_pert, s.margin_wol_pert, s.isp_cp_hot_s, s.thrust_cp_N, "CP"))

    # --- WoL due to maneuvers (leave as param for now; excel has 2.86 m/s)
    dv_wol_man = 2.86 if mode == "excel" else 2.86  # keep same until you decide what to model here
    maneuvers.append(Maneuver("WoL due to Maneuvers", dv_wol_man, s.margin_wol_man, s.isp_cp_hot_s, s.thrust_cp_N, "CP"))

    # --- Acquisition & Safe Mode (event model OR excel 3.82 m/s)
    if mode == "computed":
        dv_safe, n_events = dv.safe_mode_dv_events(
            years=s.years,
            mean_events_per_year=s.safe_mode_events_per_year,
            dv_per_event_mps=s.safe_mode_dv_per_event_mps,
            stochastic=False,
            rng=rng,
        )
        diag["safe_mode"] = {"dv_total_mps": dv_safe, "n_events": n_events}
        dv_acq_safe = dv_safe
    else:
        dv_acq_safe = 3.82

    # your sheet uses cold gas for acquisition/safe-mode; keep that here
    maneuvers.append(Maneuver("Acquisition and Safe Mode", dv_acq_safe, s.margin_safe, s.isp_cp_cold_s, s.thrust_cp_N, "CP"))

    # --- Far-range rendezvous (counts → totals)
    # Excel is inconsistent: it says 15 m/s in derived params but table has 150 m/s.
    # Here we model: dv_per_attempt * (attempts/client) * clients * (some factor like phases per year).
    dv_rdv_ep_per_attempt = 15.0
    dv_rdv_cp_per_attempt = 1.147  # from your derived line (CP Δv for Far Range RdV)

    n_attempts_total = s.n_far_range_rdv_attempts_per_client * s.n_clients
    dv_rdv_ep_total = dv_rdv_ep_per_attempt * n_attempts_total
    dv_rdv_cp_total = dv_rdv_cp_per_attempt * n_attempts_total

    if mode == "excel":
        # Use the mission phase table values
        dv_rdv_ep_total = 150.0
        dv_rdv_cp_total = 11.47

    maneuvers.append(Maneuver("Far Range RdV EP (60–1 km)", dv_rdv_ep_total, s.margin_rdv, s.isp_ep_s, s.thrust_ep_N, "EP"))
    maneuvers.append(Maneuver("Far Range RdV CP (60–1 km)", dv_rdv_cp_total, s.margin_rdv, s.isp_cp_hot_s, s.thrust_cp_N, "CP"))

    # --- Altitude changes ±300 km (counts → totals, calibrated)
    dv_alt_per_event = dv.altitude_change_dv_per_event(s.alt_transfer_delta_a_km)
    n_alt_events_total = s.n_alt_transfers_per_year * s.years
    dv_alt_total = dv_alt_per_event * n_alt_events_total
    if mode == "excel":
        dv_alt_total = 88.0

    maneuvers.append(Maneuver("Altitude Changes (+/-300 km)", dv_alt_total, s.margin_alt, s.isp_ep_s, s.thrust_ep_N, "EP"))

    # --- Station keeping
    dv_ns = dv.stationkeeping_ns(s.years, s.dv_ns_per_year_mps) if mode == "computed" else 275.0
    dv_ew = s.years * s.dv_ew_per_year_mps if mode == "computed" else 0.0

    maneuvers.append(Maneuver("Station keeping NS", dv_ns, s.margin_ns, s.isp_ep_s, s.thrust_ep_N, "EP"))
    maneuvers.append(Maneuver("Station keeping EW", dv_ew, s.margin_ew, s.isp_ep_s, s.thrust_ep_N, "EP"))

    # --- Disposal (keep as 26 m/s unless you define an end-of-life orbit change)
    dv_disposal = 26.0
    maneuvers.append(Maneuver("Disposal", dv_disposal, s.margin_disposal, s.isp_ep_s, s.thrust_ep_N, "EP"))

    return maneuvers, diag
