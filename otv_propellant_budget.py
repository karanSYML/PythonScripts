"""
=============================================================================
OTV GEO SERVICING MISSION — PROPELLANT BUDGET TOOL
=============================================================================
Mission Architecture (OTV = Orbital Transfer / Servicing Vehicle):

  PHASE A — GRAVEYARD OPS (Super-GEO, +300 km above GEO)
    0. SGEO Phasing:  Drift & maneuver to longitude of graveyard target (RCS)
    1. Far-Range RDV: Approach to ~1 km hold point (RCS)
    2. Near-Range RDV 1: 1 km → docking contact (RCS, non-cooperative target)
    3. Docking 1:     Capture graveyard satellite        [★ +2500 kg]
    4. Detumbling 1:  Dampen captured satellite tumble (RCS)

  PHASE B — TRANSIT TO GEO
    5. Descent:       Low-thrust plasma spiral, Super-GEO → GEO
    6. GEO Circ.:     Circularization & orbit correction (plasma)

  PHASE C — CLIENT SATELLITE OPS (GEO)
    7. Far-Range RDV 2: Approach to ~1 km of client satellite (RCS)
    8. Near-Range RDV 2: 1 km → docking (RCS, cooperative target)
    9. Docking 2:     Capture client GEO satellite       [★ +2500 kg]
   10. Detumbling 2:  Settle full stack (RCS)

  PHASE D — STATION KEEPING (5 years, full 3-body stack)
   11. N-S + E-W SK:  Plasma thrusters via thruster arm
   12. AOCS / Momentum Management: RCS (wheel desaturation, attitude)
   13. Collision Avoidance Maneuvers (CAM): RCS

  PHASE E — DISPOSAL
   14. Raise full stack to Super-GEO +300 km (plasma)

Propellant chaining: Tsiolkovsky applied sequentially (backward pass to
compute required initial mass, forward pass to track mass state).

Edit the CONFIG section below to run trade studies.
=============================================================================
"""

import math
import re
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# ANSI COLORS (disable if not supported)
# ─────────────────────────────────────────────────────────────────────────────
USE_COLOR = sys.stdout.isatty() or True  # set False to strip colors

class C:
    RST  = "\033[0m"   if USE_COLOR else ""
    B    = "\033[1m"   if USE_COLOR else ""
    DIM  = "\033[2m"   if USE_COLOR else ""
    CYN  = "\033[96m"  if USE_COLOR else ""
    GRN  = "\033[92m"  if USE_COLOR else ""
    YLW  = "\033[93m"  if USE_COLOR else ""
    RED  = "\033[91m"  if USE_COLOR else ""
    MAG  = "\033[95m"  if USE_COLOR else ""
    BLU  = "\033[94m"  if USE_COLOR else ""
    WHT  = "\033[97m"  if USE_COLOR else ""

def co(text, *codes):
    return "".join(codes) + str(text) + C.RST

def strip_ansi(s):
    return re.sub(r'\033\[[0-9;]*m', '', str(s))

def vlen(s):   # visible length (no ANSI)
    return len(strip_ansi(s))

def pad(s, w, align="left"):
    s = str(s)
    n = max(0, w - vlen(s))
    if align == "right":  return " " * n + s
    if align == "center": return " " * (n//2) + s + " " * (n - n//2)
    return s + " " * n

def rule(title="", width=128, col=C.BLU):
    if title:
        side = max(2, (width - vlen(title) - 4) // 2)
        return col + "─"*side + "  " + C.B + C.WHT + title + C.RST + col + "  " + "─"*side + C.RST
    return col + "─" * width + C.RST

W = 128  # line width

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
G0       = 9.80665          # m/s² standard gravity
MU       = 3.986004418e14   # m³/s² Earth gravitational parameter
R_EARTH  = 6.3781e6         # m
ALT_GEO  = 35_786_000       # m
ALT_SGEO = 36_086_000       # m  Super-GEO / graveyard (+300 km, IADC)

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════  CONFIG  ═══════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# ── Spacecraft ─────────────────────────────────────────────────────────────
M_DRY_OTV           = 550.0    # kg   OTV dry mass (structure, GNC, arm, thrusters)
PROP_TANK_CAPACITY  = 800.0    # kg   Total propellant capacity (xenon + hydrazine combined)
M_PAYLOAD_GYARD     = 2500.0   # kg   Graveyard satellite: dead, tumbling, non-functional
M_PAYLOAD_CLIENT    = 2500.0   # kg   Client GEO sat: active, cooperative

# ── Propulsion ────────────────────────────────────────────────────────────
ISP_PLASMA          = 1500.0   # s    Hall-effect plasma thruster Isp (range: 1200–2000 s)
ISP_RCS             = 220.0    # s    Chemical mono-prop RCS (hydrazine, range: 200–230 s)
BURN_EFF_PLASMA     = 0.975    # —    Plasma efficiency (gravity losses in spirals)
BURN_EFF_RCS        = 0.990    # —    RCS efficiency (short burns)
LOADING_EFF         = 0.992    # —    Propellant loading fill fraction

# ── Mission ────────────────────────────────────────────────────────────────
SK_YEARS            = 5.0      # years  Station keeping duration

# ── Margins (ECSS-E-ST-35C) ────────────────────────────────────────────────
SYSTEM_MARGIN_PCT   = 0.05     # 5%    Phase B/C standard system margin
RESIDUAL_KG         = 18.0     # kg    Unusable, trapped, lines, filters

# ── Phase ΔV values (m/s, NOMINAL — contingency applied separately) ────────
DV = dict(
    sgeo_phase  = 10.0,   # Phasing burns in graveyard belt
    far_rdv_1   = 20.0,   # Far-range approach to 1 km hold (gyard sat)
    nrr_1       = 15.0,   # Near-range RDV 1 km→contact (non-cooperative)
    dock_1      = 2.0,    # Final contact + arm grasp + structural capture
    detumble_1  = 5.0,    # Momentum transfer to stop tumble (equiv. ΔV)
    descent     = 2.5,    # Plasma spiral Super-GEO → GEO (gravity loss incl.)
    geo_circ    = 5.0,    # Orbit shape/inclination correction at GEO
    far_rdv_2   = 20.0,   # Far-range approach to client sat (1 km hold)
    nrr_2       = 12.0,   # Near-range RDV cooperative (lower ΔV)
    dock_2      = 2.0,    # Docking to client satellite
    detumble_2  = 1.5,    # Stack settle (cooperative, lower)
    sk_ns_yr    = 50.0,   # N-S station keeping per year
    sk_ew_yr    = 2.0,    # E-W station keeping per year
    aocs_yr     = 5.0,    # AOCS / momentum management per year
    cam_event   = 3.0,    # Per collision avoidance event
    cam_per_yr  = 1.0,    # CAMs per year
    disposal    = 2.5,    # Low-thrust raise to Super-GEO
)

# Derived totals
DV['sk_total']   = (DV['sk_ns_yr'] + DV['sk_ew_yr']) * SK_YEARS
DV['aocs_total'] = DV['aocs_yr'] * SK_YEARS
DV['cam_total']  = DV['cam_event'] * DV['cam_per_yr'] * SK_YEARS

# ── Contingencies (fraction) ───────────────────────────────────────────────
CONT = dict(
    sgeo_phase  = 0.15,  # Nav uncertainty in graveyard belt
    far_rdv_1   = 0.15,  # Approach nav
    nrr_1       = 0.20,  # Non-cooperative: higher dispersion
    dock_1      = 0.20,  # Non-cooperative capture
    detumble_1  = 0.30,  # Tumble rate highly uncertain for dead sat
    descent     = 0.10,  # Low-thrust, well-modeled
    geo_circ    = 0.10,  # Orbit trim
    far_rdv_2   = 0.15,  # Cooperative approach
    nrr_2       = 0.15,  # Cooperative
    dock_2      = 0.15,  # Cooperative capture
    detumble_2  = 0.15,  # Mostly cooperative
    sk          = 0.05,  # Well-characterized (mature solar pressure models)
    aocs        = 0.10,  # RW sizing uncertainty
    cam         = 0.20,  # Unpredictable event
    disposal    = 0.10,  # Low-thrust, well-modeled
)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Phase:
    pid:      int
    name:     str
    dv_key:   str        # key into DV dict
    cont_key: str        # key into CONT dict
    thruster: str        # 'plasma' | 'rcs'
    note:     str

    @property
    def dv_nom(self):    return DV[self.dv_key]
    @property
    def cont(self):      return CONT[self.cont_key]
    @property
    def dv_total(self):  return self.dv_nom * (1.0 + self.cont)
    @property
    def isp(self):       return ISP_PLASMA if self.thruster == "plasma" else ISP_RCS
    @property
    def eff(self):       return BURN_EFF_PLASMA if self.thruster == "plasma" else BURN_EFF_RCS


# Capture events: AFTER phase pid, add satellite to stack
CAPTURE_EVENTS = {
    3:  ("Graveyard Satellite", M_PAYLOAD_GYARD),
    9:  ("Client GEO Satellite", M_PAYLOAD_CLIENT),
}

SECTION_LABELS = {
    0:  "PHASE A ── Graveyard Operations (Super-GEO +300 km)",
    5:  "PHASE B ── Transit: Super-GEO → GEO",
    7:  "PHASE C ── Client Satellite Operations (GEO)",
    11: "PHASE D ── Station Keeping (5 Years, Full 3-Body Stack)",
    14: "PHASE E ── Disposal",
}

def build_phases() -> List[Phase]:
    return [
        # ── Phase A ─────────────────────────────────────────────────────────
        Phase(0,  "Super-GEO Phasing / Drift to Target",    "sgeo_phase","sgeo_phase","rcs",
              "2 correction burns. Phasing to target longitude in graveyard belt."),
        Phase(1,  "Far-Range Rendezvous 1 (→ 1 km hold)",  "far_rdv_1", "far_rdv_1", "rcs",
              "2-impulse approach. Drift to 1 km safety hold point."),
        Phase(2,  "Near-Range Rendezvous 1 (1km → contact)","nrr_1",    "nrr_1",     "rcs",
              "1km→200m→contact. Non-cooperative. V-bar/R-bar corridor control."),
        Phase(3,  "Docking 1 — Capture Graveyard Satellite","dock_1",   "dock_1",    "rcs",
              "Robotic arm grasp. Hard dock structural engagement. [+2500 kg]"),
        Phase(4,  "Detumbling — Graveyard Satellite",        "detumble_1","detumble_1","rcs",
              "Dampen tumble via thrusters. Rate 0.5–5 deg/s assumed. High uncertainty."),
        # ── Phase B ─────────────────────────────────────────────────────────
        Phase(5,  "Stack Descent: Super-GEO → GEO (Plasma)","descent",  "descent",   "plasma",
              "Low-thrust spiral. OTV + gyard sat. Gravity losses embedded in eff."),
        Phase(6,  "GEO Orbit Correction & Circularization", "geo_circ", "geo_circ",  "plasma",
              "Fine orbit trim: shape, inclination residual, longitude slot entry."),
        # ── Phase C ─────────────────────────────────────────────────────────
        Phase(7,  "Far-Range Rendezvous 2 (→ Client, 1 km)","far_rdv_2","far_rdv_2", "rcs",
              "Cooperative approach from GEO slot to 1 km hold."),
        Phase(8,  "Near-Range Rendezvous 2 (1km → contact)","nrr_2",    "nrr_2",     "rcs",
              "Cooperative target. Reduced contingency. Approach corridor control."),
        Phase(9,  "Docking 2 — Capture Client GEO Satellite","dock_2",  "dock_2",    "rcs",
              "Full 3-body stack formed. [+2500 kg client satellite]"),
        Phase(10, "Detumbling — Client GEO Satellite",       "detumble_2","detumble_2","rcs",
              "Attitude residuals post-capture. Cooperative → lower contingency."),
        # ── Phase D ─────────────────────────────────────────────────────────
        Phase(11, f"Station Keeping N-S+E-W ({SK_YEARS:.0f} yrs, Plasma)","sk_total","sk","plasma",
              f"N-S: {DV['sk_ns_yr']} m/s/yr, E-W: {DV['sk_ew_yr']} m/s/yr × {SK_YEARS:.0f} yrs. Thruster arm."),
        Phase(12, f"AOCS / Momentum Mgmt ({SK_YEARS:.0f} yrs, RCS)",    "aocs_total","aocs","rcs",
              f"RW desaturation, attitude hold, eclipse transitions. {DV['aocs_yr']} m/s/yr."),
        Phase(13, f"Collision Avoidance ({SK_YEARS:.0f} yrs, RCS)",     "cam_total", "cam", "rcs",
              f"{DV['cam_per_yr']:.0f} CAM/yr × {SK_YEARS:.0f} yrs × {DV['cam_event']} m/s. GEO congestion risk."),
        # ── Phase E ─────────────────────────────────────────────────────────
        Phase(14, "Disposal: Full Stack → Super-GEO +300 km","disposal", "disposal", "plasma",
              "IADC compliant. Low-thrust spiral. All 3 bodies remain joined."),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# ROCKET EQUATION
# ─────────────────────────────────────────────────────────────────────────────
def prop_needed(dv_cont: float, isp: float, m_final: float, eff: float = 1.0) -> float:
    """Tsiolkovsky: propellant mass to produce dv_cont starting from m_final."""
    ve = isp * G0
    return m_final * (math.exp(dv_cont / (ve * eff)) - 1.0)

def hohmann_delta_v(r1: float, r2: float) -> Tuple[float, float]:
    v1 = math.sqrt(MU / r1);  v2 = math.sqrt(MU / r2)
    at = (r1 + r2) / 2
    vt1 = math.sqrt(MU * (2/r1 - 1/at))
    vt2 = math.sqrt(MU * (2/r2 - 1/at))
    return abs(vt1-v1), abs(v2-vt2)

# ─────────────────────────────────────────────────────────────────────────────
# BUDGET COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PhaseResult:
    phase:        Phase
    prop_used:    float
    mass_before:  float
    mass_after:   float
    mass_ratio:   float
    stack_desc:   str

def compute_budget(phases: List[Phase]):
    # ── Backward pass: work from EOL dry mass to find total prop needed ─────
    m_eol = M_DRY_OTV + M_PAYLOAD_GYARD + M_PAYLOAD_CLIENT
    prop_map = {}
    m_bwd = m_eol
    for p in reversed(phases):
        prop = prop_needed(p.dv_total, p.isp, m_bwd, p.eff)
        prop_map[p.pid] = prop
        m_bwd += prop
        if p.pid in CAPTURE_EVENTS:
            _, sat_mass = CAPTURE_EVENTS[p.pid]
            m_bwd -= sat_mass   # un-capture going backwards

    total_mission_prop = sum(prop_map.values())
    sys_margin    = total_mission_prop * SYSTEM_MARGIN_PCT
    total_load    = total_mission_prop + sys_margin + RESIDUAL_KG
    total_load_eff = total_load / LOADING_EFF
    tank_margin   = PROP_TANK_CAPACITY - total_load_eff

    # ── Forward pass: track mass state ──────────────────────────────────────
    results = []
    prop_rem = total_mission_prop
    cap_g = 0.0; cap_c = 0.0

    for p in phases:
        prop = prop_map[p.pid]
        m_before = M_DRY_OTV + prop_rem + cap_g + cap_c
        prop_rem -= prop

        if p.pid in CAPTURE_EVENTS:
            name, smass = CAPTURE_EVENTS[p.pid]
            if "Graveyard" in name: cap_g = smass
            else: cap_c = smass

        m_after = M_DRY_OTV + prop_rem + cap_g + cap_c
        mr = m_before / m_after if m_after > 0 else 1.0
        
        if cap_c > 0:    desc = "OTV + Gyard-Sat + Client-Sat"
        elif cap_g > 0:  desc = "OTV + Gyard-Sat"
        else:            desc = "OTV only"

        results.append(PhaseResult(p, prop, m_before, m_after, mr, desc))

    plasma_total = sum(r.prop_used for r in results if r.phase.thruster == "plasma")
    rcs_total    = sum(r.prop_used for r in results if r.phase.thruster == "rcs")

    summary = dict(
        total_dv_nom       = sum(p.dv_nom for p in phases),
        total_dv_cont      = sum(p.dv_total for p in phases),
        total_mission_prop = total_mission_prop,
        sys_margin         = sys_margin,
        residual           = RESIDUAL_KG,
        total_load         = total_load,
        total_load_eff     = total_load_eff,
        tank_margin        = tank_margin,
        launch_wet         = M_DRY_OTV + total_load_eff,
        prop_frac          = total_load_eff / (M_DRY_OTV + total_load_eff),
        plasma_total       = plasma_total,
        rcs_total          = rcs_total,
    )
    return results, summary

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
def print_rule(title="", col=C.BLU):
    print(rule(title, W, col))

def display_header(summary):
    print()
    print_rule("OTV GEO SERVICING MISSION — PROPELLANT BUDGET TOOL", C.CYN)
    print()
    cfg = [
        ("OTV Dry Mass",          f"{M_DRY_OTV:.0f} kg",
         "Graveyard Sat (dead)",  f"{M_PAYLOAD_GYARD:.0f} kg"),
        ("Plasma Isp (Hall)",     f"{ISP_PLASMA:.0f} s",
         "Client Sat (active)",   f"{M_PAYLOAD_CLIENT:.0f} kg"),
        ("RCS Isp (N2H4 mono)",   f"{ISP_RCS:.0f} s",
         "Tank Capacity",         f"{PROP_TANK_CAPACITY:.0f} kg"),
        ("SK Duration",           f"{SK_YEARS:.0f} years",
         "System Margin",         f"{SYSTEM_MARGIN_PCT*100:.0f}%  +  {RESIDUAL_KG:.0f} kg residual"),
        ("Plasma Burn Eff.",       f"{BURN_EFF_PLASMA*100:.1f}%",
         "RCS Burn Eff.",         f"{BURN_EFF_RCS*100:.1f}%"),
    ]
    for a, b, d, e in cfg:
        print("  " + co(f"{a:<28}", C.DIM) + co(f"{b:<18}", C.B+C.CYN) +
              "  " + co(f"{d:<28}", C.DIM) + co(f"{e:<18}", C.B+C.CYN))
    print()

def display_phase_table(results: List[PhaseResult], summary: dict):
    print_rule("MISSION PHASE DETAIL", C.BLU)
    
    # Column widths
    cw = [3, 42, 7, 9, 6, 9, 6, 11, 11, 10, 8, 23]
    headers = ["#","Phase Name","Engine","ΔV nom","Cont.","ΔV+cont",
               "Isp","M.Before","M.After","Prop Used","M.Ratio","Stack Config"]
    al = ["r","l","c","r","r","r","r","r","r","r","r","l"]

    def row_str(cells, colors=None):
        out = "  "
        for i, (cell, w, a) in enumerate(zip(cells, cw, al)):
            s = pad(str(cell), w, a)
            if colors and colors[i]:
                s = co(s, colors[i])
            out += s + "  "
        return out

    def sep_line():
        print("  " + "  ".join("─"*w for w in cw))

    print()
    print(row_str(headers, [C.DIM]*len(headers)))
    sep_line()

    for r in results:
        pid = r.phase.pid

        if pid in SECTION_LABELS:
            print()
            print("  " + co(SECTION_LABELS[pid], C.DIM+C.BLU))

        thr_col = C.MAG if r.phase.thruster == "plasma" else C.GRN
        thr_str = "PLASMA" if r.phase.thruster == "plasma" else "RCS"
        name    = r.phase.name[:42]

        cells = [
            str(pid), name, thr_str,
            f"{r.phase.dv_nom:.1f}", f"{r.phase.cont*100:.0f}%",
            f"{r.phase.dv_total:.1f}", f"{r.phase.isp:.0f}",
            f"{r.mass_before:.1f}", f"{r.mass_after:.1f}",
            f"{r.prop_used:.2f}", f"{r.mass_ratio:.4f}",
            r.stack_desc,
        ]
        clrs = [C.DIM, C.WHT, thr_col, None, C.DIM,
                C.CYN, C.DIM, None, None, C.YLW, C.DIM, C.DIM]
        print(row_str(cells, clrs))

        if pid in CAPTURE_EVENTS:
            nm, sm = CAPTURE_EVENTS[pid]
            print("  " + co(f"  ★  → {nm} captured and added to stack: +{sm:.0f} kg", C.RED+C.B))

    sep_line()
    total_cells = [
        "TOT", "", "",
        f"{summary['total_dv_nom']:.1f}", "",
        f"{summary['total_dv_cont']:.1f}", "",
        "", "",
        f"{summary['total_mission_prop']:.2f}", "", ""
    ]
    tclrs = [C.DIM,None,None, C.CYN,None, C.B+C.CYN, None, None, None, C.B+C.YLW, None, None]
    print(row_str(total_cells, tclrs))
    print()

def display_dv_waterfall(results: List[PhaseResult], summary: dict):
    print_rule("ΔV WATERFALL  (with contingency)", C.BLU)
    print()
    BAR = 54
    max_dv = max(r.phase.dv_total for r in results)
    for r in results:
        col = C.MAG if r.phase.thruster == "plasma" else C.GRN
        n   = max(1, int(r.phase.dv_total / max_dv * BAR)) if r.phase.dv_total > 0 else 0
        pct = r.phase.dv_total / summary['total_dv_cont'] * 100
        nm  = r.phase.name[:42].ljust(42)
        bar = co("█"*n, col)
        print(f"  {nm}  {bar:<54}  "
              f"{co(f'{r.phase.dv_total:7.1f} m/s', col)}  "
              f"{co(f'({pct:4.1f}%  {r.phase.cont*100:.0f}% cont.)', C.DIM)}")
    print()
    tdv_c = summary['total_dv_cont']
    tdv_n = summary['total_dv_nom']
    print(f"  {'TOTAL':42}  {'':54}  "
          + co(f"{tdv_c:7.1f} m/s", C.B+C.CYN) + "  "
          + co(f"(nominal {tdv_n:.1f} m/s)", C.DIM))
    print()

def display_budget_summary(results: List[PhaseResult], summary: dict):
    print_rule("PROPELLANT BUDGET SUMMARY", C.BLU)
    print()
    
    plasma_phases = [(r.phase.name, r.prop_used) for r in results if r.phase.thruster == "plasma"]
    rcs_phases    = [(r.phase.name, r.prop_used) for r in results if r.phase.thruster == "rcs"]
    total = summary['total_mission_prop']

    WN, WV = 55, 11
    hsep = "  " + "─"*WN + "  " + "─"*WV + "  " + "─"*9

    def brow(label, val="", pct="", lc="", vc=C.YLW):
        l = co(pad(label, WN), lc) if lc else pad(label, WN)
        v = co(pad(val, WV, "right"), vc) if vc else pad(val, WV, "right")
        p = pad(pct, 9, "right")
        print(f"  {l}  {v}  {p}")

    print(co(f"  {'Budget Item':<55}  {'Mass (kg)':>11}  {'% Mission':>9}", C.DIM))
    print(hsep)

    print(f"  {co('── Plasma (Hall Thruster) Phases', C.B+C.MAG)}")
    for nm, mp in plasma_phases:
        brow(f"   {nm[:52]}", f"{mp:.2f}", f"{mp/total*100:.1f}%", "", C.MAG)
    brow(f"   PLASMA SUBTOTAL", f"{summary['plasma_total']:.2f}",
         f"{summary['plasma_total']/total*100:.1f}%", C.B+C.MAG, C.B+C.MAG)

    print()
    print(f"  {co('── RCS (Hydrazine Mono-prop) Phases', C.B+C.GRN)}")
    for nm, mr in rcs_phases:
        brow(f"   {nm[:52]}", f"{mr:.2f}", f"{mr/total*100:.1f}%", "", C.GRN)
    brow(f"   RCS SUBTOTAL", f"{summary['rcs_total']:.2f}",
         f"{summary['rcs_total']/total*100:.1f}%", C.B+C.GRN, C.B+C.GRN)

    print(); print(hsep)
    brow("MISSION PROPELLANT TOTAL",             f"{total:.2f}", "100.0%", C.B+C.WHT, C.B+C.WHT)
    brow(f"+ System Margin ({SYSTEM_MARGIN_PCT*100:.0f}%)",
                                                  f"{summary['sys_margin']:.2f}", "", C.DIM, C.YLW)
    brow(f"+ Residual / Unusable / Trapped",      f"{summary['residual']:.2f}", "", C.DIM, C.YLW)
    print(hsep)
    brow("TOTAL PROPELLANT LOAD (nominal)",       f"{summary['total_load']:.2f}", "", C.B+C.CYN, C.B+C.CYN)
    brow(f"TOTAL LOAD (÷ loading eff {LOADING_EFF*100:.1f}%)",
                                                   f"{summary['total_load_eff']:.2f}", "", C.B+C.CYN, C.B+C.CYN)
    print(hsep)
    brow("Tank Capacity",                         f"{PROP_TANK_CAPACITY:.2f}", "", "", C.WHT)

    tm  = summary['tank_margin']
    tmc = C.B+C.GRN if tm > 50 else (C.B+C.YLW if tm > 0 else C.B+C.RED)
    tag = "✔ OK" if tm > 50 else ("⚠ TIGHT" if tm > 0 else "✘ OVERRUN")
    brow(f"Tank Margin  [{tag}]", f"{tm:.2f}",
         f"{tm/PROP_TANK_CAPACITY*100:.1f}%", tmc, tmc)
    print()

def display_kpis(summary: dict):
    print_rule("KEY PERFORMANCE INDICATORS", C.BLU)
    print()
    tm  = summary['tank_margin']
    tmc = C.B+C.GRN if tm > 50 else (C.B+C.YLW if tm > 0 else C.B+C.RED)
    
    kpis = [
        ("Total ΔV (nominal)",          f"{summary['total_dv_nom']:.1f} m/s",  C.CYN+C.B),
        ("Total ΔV (with contingency)", f"{summary['total_dv_cont']:.1f} m/s", C.CYN+C.B),
        ("Mission Propellant",          f"{summary['total_mission_prop']:.1f} kg", C.YLW+C.B),
        ("Total Prop Load (w/ margins)",f"{summary['total_load_eff']:.1f} kg", C.YLW+C.B),
        ("Plasma Propellant (Xenon)",   f"{summary['plasma_total']:.1f} kg",   C.MAG+C.B),
        ("RCS Propellant (Hydrazine)",  f"{summary['rcs_total']:.1f} kg",      C.GRN+C.B),
        ("OTV Dry Mass",                f"{M_DRY_OTV:.0f} kg",                  C.WHT),
        ("OTV Launch Wet Mass",         f"{summary['launch_wet']:.1f} kg",      C.WHT+C.B),
        ("OTV Prop. Mass Fraction",     f"{summary['prop_frac']*100:.1f}%",     C.WHT),
        ("Full Stack Dry (EOL)",        f"{M_DRY_OTV+M_PAYLOAD_GYARD+M_PAYLOAD_CLIENT:.0f} kg", C.WHT),
        ("Tank Capacity",               f"{PROP_TANK_CAPACITY:.0f} kg",         C.WHT),
        ("Tank Margin",                 f"{tm:.1f} kg  ({tm/PROP_TANK_CAPACITY*100:.1f}%)", tmc),
    ]
    mid = len(kpis) // 2 + len(kpis) % 2
    for i in range(mid):
        l = kpis[i]
        r = kpis[i+mid] if i+mid < len(kpis) else None
        left  = co(pad(l[0], 32), C.DIM) + co(pad(l[1], 18), l[2])
        right = (co(pad(r[0], 32), C.DIM) + co(pad(r[1], 18), r[2])) if r else ""
        print(f"  {left}  {right}")
    print()

def display_prop_stack(results: List[PhaseResult], summary: dict):
    print_rule("PROPELLANT CONSUMPTION  (cumulative, forward through mission)", C.BLU)
    print()
    BAR = 42
    total = summary['total_mission_prop']
    cum   = 0.0

    hdr = (f"  {pad('Phase', 42)}  {pad('Used',8,'r')}  "
           f"{pad('Cumul.',10,'r')}  {pad('Remaining',12,'r')}  Bar")
    print(co(hdr, C.DIM))
    print("  " + "─"*122)

    for r in results:
        cum  += r.prop_used
        rem   = total - cum
        col   = C.MAG if r.phase.thruster == "plasma" else C.GRN
        n_use = int(cum / total * BAR)
        n_rem = BAR - n_use
        bar   = co("▓"*n_use, col) + co("░"*n_rem, C.DIM)
        nm    = r.phase.name[:41]
        print(f"  {pad(nm,42)}  {co(f'{r.prop_used:8.2f}',col)}  "
              f"{cum:10.2f}  {rem:12.2f}  {bar}")
    print()

def display_sensitivity(phases_fn):
    print_rule("SENSITIVITY  —  Total Prop Load (kg) vs. Isp Values", C.YLW)
    print()
    global ISP_PLASMA, ISP_RCS
    orig_p, orig_r = ISP_PLASMA, ISP_RCS

    plasma_isps = [900, 1200, 1500, 1800, 2000]
    rcs_isps    = [180, 200, 220, 240]

    hdr = f"  {pad('RCS Isp ↓  / Plasma →', 22)}" + \
          "".join(f"  {pad(f'{p}s', 10, 'r')}" for p in plasma_isps)
    print(co(hdr, C.DIM))
    print("  " + "─"*78)

    for ri in rcs_isps:
        ISP_RCS = ri
        row = f"  {pad(f'RCS {ri} s', 22)}"
        for pi in plasma_isps:
            ISP_PLASMA = pi
            ph = phases_fn()
            _, s = compute_budget(ph)
            val = s['total_load_eff']
            tm  = s['tank_margin']
            col = C.B+C.GRN if tm > 50 else (C.B+C.YLW if tm > 0 else C.B+C.RED)
            row += f"  {co(pad(f'{val:.1f}',10,'r'), col)}"
        print(row)

    ISP_PLASMA, ISP_RCS = orig_p, orig_r
    print()
    print(f"  {co('Tank capacity: '+str(int(PROP_TANK_CAPACITY))+' kg.  ', C.DIM)}"
          f"{co('Green', C.B+C.GRN)}{co(' = >50 kg margin  ', C.DIM)}"
          f"{co('Yellow', C.B+C.YLW)}{co(' = 0–50 kg  ', C.DIM)}"
          f"{co('Red', C.B+C.RED)}{co(' = overrun', C.DIM)}")
    print()

def display_assumptions():
    print_rule("KEY ASSUMPTIONS & ENGINEERING RATIONALE", C.DIM)
    print()
    dv1, dv2 = hohmann_delta_v(R_EARTH+ALT_GEO, R_EARTH+ALT_SGEO)
    items = [
        ("Launch injection",    "LV delivers OTV directly to Super-GEO (+300 km). OTV pays ZERO ΔV for GTO→GEO transfer."),
        ("Prop. system",        "Two propellant systems: xenon (plasma) + hydrazine (RCS mono-prop). Budget sums both."),
        ("Graveyard satellite", f"Dead, non-functional, tumbling ~0.5–5 deg/s. No propulsion. {M_PAYLOAD_GYARD:.0f} kg dry mass."),
        ("Client satellite",    f"Active GEO satellite. Cooperative target for docking. {M_PAYLOAD_CLIENT:.0f} kg dry mass."),
        ("Detumbling ΔV",       "Equivalent-ΔV method. Actual technique TBD (contact thrust / robotic arm / net capture)."),
        ("Low-thrust penalty",  f"Plasma spirals: gravity losses embedded in efficiency factor ({BURN_EFF_PLASMA*100:.1f}%)."),
        ("Hohmann ref.",        f"2-impulse ideal GEO↔+300km: {dv1:.2f}+{dv2:.2f}={dv1+dv2:.2f} m/s. Low-thrust spiral used → {DV['descent']} m/s."),
        ("N-S station keeping", f"{DV['sk_ns_yr']} m/s/yr: solar radiation pressure + luni-solar inclination drift. Mass-independent."),
        ("E-W station keeping", f"{DV['sk_ew_yr']} m/s/yr: triaxiality + eccentricity control. Value is mid-range (varies by slot)."),
        ("AOCS / momentum",     f"{DV['aocs_yr']} m/s/yr: reaction wheel desaturation, attitude hold, eclipse entry/exit."),
        ("CAM budget",          f"{DV['cam_per_yr']:.0f} CAM/yr × {SK_YEARS:.0f} yrs × {DV['cam_event']} m/s/event = {DV['cam_total']:.0f} m/s. GEO congestion risk growing."),
        ("Disposal",            "Full 3-body stack (OTV + both captured sats) raised together. IADC: +300 km above GEO."),
        ("ECSS margins",        f"ECSS-E-ST-35C: {SYSTEM_MARGIN_PCT*100:.0f}% system margin + {RESIDUAL_KG:.0f} kg residual (tank, lines, traps, filters)."),
        ("Mass chaining",       "Tsiolkovsky backward pass from EOL. Capture events add satellite mass at correct phase boundaries."),
        ("Non-coop. RDV",       "Phases 1–4: no GPS relay, no comms, potential tumble → 20–30% contingency (vs. 15% cooperative)."),
    ]
    for tag, text in items:
        print(f"  {co(pad(tag+':', 27), C.CYN)} {co(text, C.DIM)}")
    print()

def display_verdict(summary: dict):
    tm  = summary['tank_margin']
    cap = PROP_TANK_CAPACITY
    if tm < 0:
        col = C.RED; sym = "✘"
        msg = (f"TANK OVERRUN: {abs(tm):.1f} kg short. "
               f"Increase tank to ≥ {cap-tm:.0f} kg, reduce ΔV, or increase Isp.")
    elif tm < 50:
        col = C.YLW; sym = "⚠"
        msg = (f"MARGIN TIGHT: {tm:.1f} kg remaining ({tm/cap*100:.1f}%). "
               f"Review contingencies or increase tank capacity.")
    else:
        col = C.GRN; sym = "✔"
        msg = (f"BUDGET CLOSED: {tm:.1f} kg tank margin ({tm/cap*100:.1f}%). "
               f"Prop load: {summary['total_load_eff']:.1f} / {cap:.0f} kg.  "
               f"Plasma: {summary['plasma_total']:.1f} kg | RCS: {summary['rcs_total']:.1f} kg.")
    print(co("╔" + "═"*(W-2) + "╗", col))
    inner = f"  {sym}  {msg}"
    print(co("║", col) + co(inner, C.B+col) + " " * max(0, W-2-vlen(inner)) + co("║", col))
    print(co("╚" + "═"*(W-2) + "╝", col))
    print()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    phases  = build_phases()
    results, summary = compute_budget(phases)

    display_header(summary)
    display_phase_table(results, summary)
    display_dv_waterfall(results, summary)
    display_budget_summary(results, summary)
    display_kpis(summary)
    display_prop_stack(results, summary)
    display_sensitivity(build_phases)
    display_assumptions()
    display_verdict(summary)

if __name__ == "__main__":
    main()
