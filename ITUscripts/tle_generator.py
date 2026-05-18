"""
TLE Scenario Generator — Python/Tkinter replica of the HTML tool.
Identical backend math and same UI layout: left config panel, right output.
"""

import tkinter as tk
from tkinter import ttk, font
import math
import random
import datetime


# ── Colours (matching HTML CSS vars) ─────────────────────────────────────────
BG       = "#080c14"
PANEL    = "#0d1624"
PANEL2   = "#111e30"
BORDER   = "#1e3a5f"
ACCENT   = "#00d4ff"
ACCENT2  = "#ff6b35"
GREEN    = "#00ff9d"
TEXT     = "#c8ddf5"
DIM      = "#4a6a8a"
LINE_CLR = "#7ab8e0"
LINE0_CLR= "#9ab8d0"

ROLE_COLOURS = {
    "interferer": ACCENT2,
    "victim":     ACCENT,
    "third":      GREEN,
}

MU = 398600.4418   # km³/s²
RE = 6378.137      # km

PRESETS = {
    "LEO Polar Swarm":    dict(altitude=550,   inclination=97.6, eccentricity=0.0001, arg_perigee=0,   num_sats=6, raan_spread=360, bstar=0.00010),
    "MEO / GNSS-like":    dict(altitude=20200, inclination=55.0, eccentricity=0.0002, arg_perigee=0,   num_sats=4, raan_spread=120, bstar=0.00001),
    "GEO Comsat":         dict(altitude=35786, inclination=0.05, eccentricity=0.0001, arg_perigee=0,   num_sats=3, raan_spread=30,  bstar=0.000001),
    "ISS-like LEO":       dict(altitude=408,   inclination=51.6, eccentricity=0.0003, arg_perigee=150, num_sats=5, raan_spread=180, bstar=0.00020),
    "Walker Delta":       dict(altitude=1200,  inclination=53.0, eccentricity=0.0001, arg_perigee=0,   num_sats=8, raan_spread=360, bstar=0.00005),
    "Molniya HEO":        dict(altitude=1000,  inclination=63.4, eccentricity=0.7400, arg_perigee=270, num_sats=3, raan_spread=120, bstar=0.00003),
}


# ── TLE Math (exact JS port) ──────────────────────────────────────────────────

def alt_to_mean_motion(alt_km):
    a = RE + alt_km
    T = 2 * math.pi * math.sqrt(a**3 / MU)
    return 86400 / T


def epoch_to_tle_format(dt_str):
    dt = datetime.datetime.fromisoformat(dt_str)
    yr2 = str(dt.year % 100).zfill(2)
    start = datetime.datetime(dt.year, 1, 1)
    day_of_year = (dt - start).total_seconds() / 86400 + 1
    return yr2 + f"{day_of_year:012.8f}"


def tle_checksum(line):
    s = 0
    for c in line[:-1]:
        if c.isdigit():
            s += int(c)
        elif c == '-':
            s += 1
    return s % 10


def format_bstar(val):
    if val == 0:
        return " 00000-0"
    sign = "-" if val < 0 else " "
    av = abs(val)
    exp = math.floor(math.log10(av)) + 1
    mantissa = av / (10 ** exp)
    m_str = f"{mantissa:.5f}"[2:]   # 5 digits after "0."
    e_str = ("-" if exp < 0 else "+") + str(abs(exp))
    return sign + m_str + e_str


def build_tle(name, catalog_no, epoch_str, bstar, inclination, raan,
              eccentricity, arg_perigee, mean_anomaly, mean_motion, rev_at_epoch):
    cat_str   = str(catalog_no).zfill(5)
    epoch_fmt = epoch_to_tle_format(epoch_str)
    bstar_fmt = format_bstar(bstar)

    l1 = f"1 {cat_str}U 25001A   {epoch_fmt}  .00000000  00000-0 {bstar_fmt} 0  999"
    l1 = l1[:68] + str(tle_checksum(l1[:68] + "0"))

    inc_str  = f"{inclination:8.4f}"
    raan_str = f"{raan:8.4f}"
    ecc_str  = f"{eccentricity:.7f}"[2:].zfill(7)
    ap_str   = f"{arg_perigee:8.4f}"
    ma_str   = f"{mean_anomaly:8.4f}"
    mm_str   = f"{mean_motion:11.8f}"
    rev_str  = str(rev_at_epoch).zfill(5)

    l2 = f"2 {cat_str} {inc_str} {raan_str} {ecc_str} {ap_str} {ma_str} {mm_str}{rev_str}"
    l2 = l2[:68] + str(tle_checksum(l2[:68] + "0"))

    line0 = name.ljust(24)[:24]
    return line0, l1, l2


def assign_role(i, total, mode):
    if mode != "mixed":
        return mode
    if i == 0:
        return "victim"
    if i <= math.ceil(total / 2):
        return "interferer"
    return "third"


def generate_tles(altitude, inclination, eccentricity, arg_perigee,
                  num_sats, raan_spread, raan_base, manomaly_spread,
                  epoch_str, bstar, cat_start, name_prefix, sat_role):
    mm     = alt_to_mean_motion(altitude)
    period = 1440 / mm
    results = []
    for i in range(num_sats):
        raan = (raan_base + (raan_spread / num_sats) * i) % 360
        ma   = (manomaly_spread / num_sats) * i
        role = assign_role(i, num_sats, sat_role)
        name = f"{name_prefix[:8].upper()}-{str(i+1).zfill(2)}"
        rev  = random.randint(1000, 6000)
        l0, l1, l2 = build_tle(name, cat_start + i, epoch_str, bstar,
                                inclination, raan, eccentricity, arg_perigee,
                                ma, mm, rev)
        cs1_ok = tle_checksum(l1) == int(l1[-1])
        cs2_ok = tle_checksum(l2) == int(l2[-1])
        results.append(dict(
            line0=l0, line1=l1, line2=l2, role=role,
            altitude=altitude, inclination=inclination,
            raan=f"{raan:.2f}", ma=f"{ma:.2f}",
            period=f"{period:.1f}", mm=f"{mm:.6f}",
            cs1_ok=cs1_ok, cs2_ok=cs2_ok,
        ))
    return results, mm, period


# ── GUI ───────────────────────────────────────────────────────────────────────

class TLEGenerator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TLE Scenario Generator — Interference Validation")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("1280x860")

        self._last_tles = []
        self._active_tab = tk.StringVar(value="cards")

        self._build_ui()
        self._generate()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(20, 0))
        tk.Label(hdr, text="TLE SCENARIO GENERATOR", bg=BG, fg=ACCENT,
                 font=("Courier", 13, "bold")).pack(anchor="w")
        tk.Label(hdr, text="Synthetic TLE generation for ground-station interference validation · Orekit compatible",
                 bg=BG, fg=DIM, font=("Helvetica", 9)).pack(anchor="w")

        sep = tk.Frame(self, bg=BG, height=14)
        sep.pack(fill="x")

        # Main two-column layout
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left  = tk.Frame(main, bg=PANEL, bd=0, highlightthickness=1,
                         highlightbackground=BORDER)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.configure(width=340)

        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_left(left)
        self._build_right(right)

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg=PANEL, fg=ACCENT,
                 font=("Helvetica", 7, "bold")).pack(anchor="w", padx=14, pady=(12, 2))

    def _label(self, parent, text):
        tk.Label(parent, text=text, bg=PANEL, fg=DIM,
                 font=("Helvetica", 8)).pack(anchor="w", padx=14, pady=(6, 1))

    def _entry(self, parent, var, width=None):
        e = tk.Entry(parent, textvariable=var, bg=BG, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     font=("Courier", 9), bd=0,
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT)
        e.pack(fill="x", padx=14, ipady=4)
        return e

    def _row2(self, parent, items):
        """items: list of (label, var)"""
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=14, pady=0)
        for i, (lbl, var) in enumerate(items):
            col = tk.Frame(row, bg=PANEL)
            col.grid(row=0, column=i, sticky="ew", padx=(0, 6 if i == 0 else 0))
            row.columnconfigure(i, weight=1)
            tk.Label(col, text=lbl, bg=PANEL, fg=DIM,
                     font=("Helvetica", 8)).pack(anchor="w", pady=(6, 1))
            tk.Entry(col, textvariable=var, bg=BG, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     font=("Courier", 9), bd=0,
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT).pack(fill="x", ipady=4)

    def _build_left(self, parent):
        # Panel title
        title_frame = tk.Frame(parent, bg=PANEL)
        title_frame.pack(fill="x", padx=14, pady=(14, 0))
        tk.Label(title_frame, text="SCENARIO CONFIGURATION", bg=PANEL, fg=DIM,
                 font=("Helvetica", 7, "bold")).pack(anchor="w")
        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=14, pady=8)

        # Presets
        self._section(parent, "QUICK PRESETS")
        preset_outer = tk.Frame(parent, bg=PANEL)
        preset_outer.pack(fill="x", padx=14)
        preset_names = list(PRESETS.keys())
        for idx, name in enumerate(preset_names):
            col = idx % 2
            row_idx = idx // 2
            if col == 0:
                row_frame = tk.Frame(preset_outer, bg=PANEL)
                row_frame.pack(fill="x", pady=2)
                row_frame.columnconfigure(0, weight=1)
                row_frame.columnconfigure(1, weight=1)
            btn = tk.Button(row_frame, text=name, bg=PANEL2, fg=TEXT,
                            activebackground=PANEL2, activeforeground=ACCENT,
                            relief="flat", bd=0,
                            highlightthickness=1, highlightbackground=BORDER,
                            font=("Helvetica", 8), anchor="w", padx=8, pady=5,
                            cursor="hand2",
                            command=lambda n=name: self._apply_preset(n))
            btn.grid(row=0, column=col, sticky="ew", padx=(0, 4 if col == 0 else 0))

        # Orbital parameters
        self._section(parent, "ORBITAL PARAMETERS")

        self._alt_var  = tk.StringVar(value="550")
        self._inc_var  = tk.StringVar(value="97.6")
        self._ecc_var  = tk.StringVar(value="0.0001")
        self._ap_var   = tk.StringVar(value="0")

        self._row2(parent, [("Altitude (km)", self._alt_var),
                            ("Inclination (°)", self._inc_var)])
        self._row2(parent, [("Eccentricity", self._ecc_var),
                            ("Arg. of Perigee (°)", self._ap_var)])

        # Scenario layout
        self._section(parent, "SCENARIO LAYOUT")

        self._role_var = tk.StringVar(value="mixed")
        role_frame = tk.Frame(parent, bg=PANEL)
        role_frame.pack(fill="x", padx=14)
        tk.Label(role_frame, text="Role of generated satellites", bg=PANEL, fg=DIM,
                 font=("Helvetica", 8)).pack(anchor="w", pady=(6, 1))
        role_cb = ttk.Combobox(role_frame, textvariable=self._role_var,
                               values=["mixed", "interferer", "victim", "third"],
                               state="readonly", font=("Courier", 9))
        role_cb.pack(fill="x", ipady=3)
        self._style_combobox(role_cb)

        self._num_var    = tk.StringVar(value="6")
        self._raan_sp_var= tk.StringVar(value="360")
        self._raan_b_var = tk.StringVar(value="0")
        self._ma_sp_var  = tk.StringVar(value="360")

        self._row2(parent, [("Number of satellites", self._num_var),
                            ("RAAN spread (°)", self._raan_sp_var)])
        self._row2(parent, [("RAAN offset base (°)", self._raan_b_var),
                            ("Mean anomaly spread (°)", self._ma_sp_var)])

        self._epoch_var  = tk.StringVar(value="2025-01-01T00:00:00")
        self._label(parent, "Epoch (UTC)")
        self._entry(parent, self._epoch_var)

        self._bstar_var  = tk.StringVar(value="0.0001")
        self._cat_var    = tk.StringVar(value="99001")
        self._pfx_var    = tk.StringVar(value="VALSAT")

        # row3 — bstar / cat / prefix
        r3 = tk.Frame(parent, bg=PANEL)
        r3.pack(fill="x", padx=14)
        for ci, (lbl, var) in enumerate([("BSTAR (drag)", self._bstar_var),
                                         ("Catalog # start", self._cat_var),
                                         ("Name prefix", self._pfx_var)]):
            col = tk.Frame(r3, bg=PANEL)
            col.grid(row=0, column=ci, sticky="ew", padx=(0, 4 if ci < 2 else 0))
            r3.columnconfigure(ci, weight=1)
            tk.Label(col, text=lbl, bg=PANEL, fg=DIM,
                     font=("Helvetica", 8)).pack(anchor="w", pady=(6, 1))
            tk.Entry(col, textvariable=var, bg=BG, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     font=("Courier", 9), bd=0,
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT).pack(fill="x", ipady=4)

        # Tip note
        note = tk.Frame(parent, bg="#090f1c", bd=0,
                        highlightthickness=1, highlightbackground="#1a3050")
        note.pack(fill="x", padx=14, pady=(14, 4))
        tk.Label(note, text=(
            "⚡ For interference testing: generate a victim group at one\n"
            "RAAN/altitude, then add interferers with a slight altitude\n"
            "offset or different RAAN phasing to create overlapping\n"
            "pass windows with the same ground station."
        ), bg="#090f1c", fg=DIM, font=("Helvetica", 8), justify="left",
            wraplength=280).pack(padx=10, pady=8)

        # Generate button
        gen_btn = tk.Button(parent, text="▶  GENERATE TLEs",
                            bg=BG, fg=ACCENT,
                            activebackground=ACCENT, activeforeground=BG,
                            relief="flat", bd=0,
                            highlightthickness=1, highlightbackground=ACCENT,
                            font=("Helvetica", 9, "bold"),
                            cursor="hand2", pady=10,
                            command=self._generate)
        gen_btn.pack(fill="x", padx=14, pady=(10, 16))

    def _style_combobox(self, cb):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                        fieldbackground=BG, background=BG,
                        foreground=TEXT, selectbackground=BG,
                        selectforeground=ACCENT,
                        bordercolor=BORDER, lightcolor=BORDER,
                        darkcolor=BORDER, arrowcolor=DIM)

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right(self, parent):
        # Stats row
        self._stat_frame = tk.Frame(parent, bg=BG)
        self._stat_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._stat_labels = {}
        for i, (key, lbl) in enumerate([("count", "SATELLITES"),
                                         ("alt",   "ALTITUDE KM"),
                                         ("period","PERIOD (MIN)"),
                                         ("mm",    "MEAN MOTION")]):
            box = tk.Frame(self._stat_frame, bg=PANEL, bd=0,
                           highlightthickness=1, highlightbackground=BORDER)
            box.grid(row=0, column=i, sticky="ew", padx=(0, 8))
            self._stat_frame.columnconfigure(i, weight=1)
            val_lbl = tk.Label(box, text="—", bg=PANEL, fg=ACCENT,
                               font=("Courier", 13))
            val_lbl.pack(pady=(8, 2))
            tk.Label(box, text=lbl, bg=PANEL, fg=DIM,
                     font=("Helvetica", 7)).pack(pady=(0, 8))
            self._stat_labels[key] = val_lbl

        # Tabs
        tab_bar = tk.Frame(parent, bg=BG,
                           highlightthickness=1, highlightbackground=BORDER)
        tab_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self._tab_btns = {}
        for name, label in [("cards", "Cards"), ("raw", "Raw TLE")]:
            btn = tk.Button(tab_bar, text=label, bg=BG,
                            fg=ACCENT if name == "cards" else DIM,
                            activebackground=BG, activeforeground=ACCENT,
                            relief="flat", bd=0, pady=8, padx=16,
                            font=("Helvetica", 8, "bold"),
                            cursor="hand2",
                            command=lambda n=name: self._switch_tab(n))
            btn.pack(side="left")
            self._tab_btns[name] = btn

        # Output area (scrollable)
        output_wrapper = tk.Frame(parent, bg=BG)
        output_wrapper.grid(row=2, column=0, sticky="nsew")
        output_wrapper.rowconfigure(0, weight=1)
        output_wrapper.columnconfigure(0, weight=1)

        # Cards tab
        self._cards_outer = tk.Frame(output_wrapper, bg=BG)
        self._cards_outer.grid(row=0, column=0, sticky="nsew")
        self._cards_outer.rowconfigure(0, weight=1)
        self._cards_outer.columnconfigure(0, weight=1)

        cards_canvas = tk.Canvas(self._cards_outer, bg=BG, highlightthickness=0)
        cards_canvas.grid(row=0, column=0, sticky="nsew")
        vsb = tk.Scrollbar(self._cards_outer, orient="vertical",
                           command=cards_canvas.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        cards_canvas.configure(yscrollcommand=vsb.set)

        self._cards_inner = tk.Frame(cards_canvas, bg=BG)
        cards_canvas.create_window((0, 0), window=self._cards_inner, anchor="nw")
        self._cards_inner.bind("<Configure>",
            lambda e: cards_canvas.configure(scrollregion=cards_canvas.bbox("all")))
        cards_canvas.bind("<Configure>",
            lambda e: cards_canvas.itemconfig(
                cards_canvas.find_withtag("all")[0], width=e.width)
            if cards_canvas.find_withtag("all") else None)
        self._cards_canvas = cards_canvas

        # Raw tab
        self._raw_outer = tk.Frame(output_wrapper, bg=BG)
        # header with copy-all button
        raw_hdr = tk.Frame(self._raw_outer, bg=BG)
        raw_hdr.pack(fill="x", pady=(0, 6))
        tk.Label(raw_hdr, text="RAW TLE BLOCK", bg=BG, fg=DIM,
                 font=("Helvetica", 7, "bold")).pack(side="left")
        self._copy_all_btn = tk.Button(raw_hdr, text="Copy All",
                                       bg=BG, fg=DIM,
                                       activebackground=BG, activeforeground=GREEN,
                                       relief="flat", bd=0,
                                       highlightthickness=1, highlightbackground=BORDER,
                                       font=("Helvetica", 7), padx=8, pady=2,
                                       cursor="hand2",
                                       command=self._copy_all)
        self._copy_all_btn.pack(side="right")

        self._raw_text = tk.Text(self._raw_outer, bg=BG, fg=GREEN,
                                 font=("Courier", 9), relief="flat", bd=0,
                                 highlightthickness=1, highlightbackground=BORDER,
                                 insertbackground=TEXT, wrap="none",
                                 state="disabled")
        raw_sb_y = tk.Scrollbar(self._raw_outer, command=self._raw_text.yview)
        raw_sb_x = tk.Scrollbar(self._raw_outer, orient="horizontal",
                                 command=self._raw_text.xview)
        self._raw_text.configure(yscrollcommand=raw_sb_y.set,
                                 xscrollcommand=raw_sb_x.set)
        raw_sb_y.pack(side="right", fill="y")
        raw_sb_x.pack(side="bottom", fill="x")
        self._raw_text.pack(fill="both", expand=True)

        self._raw_outer.grid_remove()   # hidden by default
        self._raw_outer.grid(row=0, column=0, sticky="nsew")
        self._raw_outer.grid_remove()

        # show cards by default
        self._cards_outer.tkraise()

    # ── Tab switching ─────────────────────────────────────────────────────────

    def _switch_tab(self, name):
        self._active_tab.set(name)
        for n, btn in self._tab_btns.items():
            btn.configure(fg=ACCENT if n == name else DIM)
        if name == "cards":
            self._raw_outer.place_forget()
            self._cards_outer.place(relx=0, rely=0, relwidth=1, relheight=1)
        else:
            self._cards_outer.place_forget()
            self._raw_outer.place(relx=0, rely=0, relwidth=1, relheight=1)

    # ── Preset loading ────────────────────────────────────────────────────────

    def _apply_preset(self, name):
        p = PRESETS[name]
        self._alt_var.set(str(p["altitude"]))
        self._inc_var.set(str(p["inclination"]))
        self._ecc_var.set(str(p["eccentricity"]))
        self._ap_var.set(str(p["arg_perigee"]))
        self._num_var.set(str(p["num_sats"]))
        self._raan_sp_var.set(str(p["raan_spread"]))
        self._bstar_var.set(str(p["bstar"]))

    # ── Generate ──────────────────────────────────────────────────────────────

    def _generate(self):
        try:
            altitude       = float(self._alt_var.get())
            inclination    = float(self._inc_var.get())
            eccentricity   = float(self._ecc_var.get())
            arg_perigee    = float(self._ap_var.get())
            num_sats       = int(self._num_var.get())
            raan_spread    = float(self._raan_sp_var.get())
            raan_base      = float(self._raan_b_var.get())
            ma_spread      = float(self._ma_sp_var.get())
            epoch_str      = self._epoch_var.get()
            bstar          = float(self._bstar_var.get())
            cat_start      = int(self._cat_var.get())
            name_prefix    = self._pfx_var.get()
            sat_role       = self._role_var.get()
        except ValueError:
            return

        self._last_tles, mm, period = generate_tles(
            altitude, inclination, eccentricity, arg_perigee,
            num_sats, raan_spread, raan_base, ma_spread,
            epoch_str, bstar, cat_start, name_prefix, sat_role
        )

        # Update stats
        self._stat_labels["count"].configure(text=str(num_sats))
        self._stat_labels["alt"].configure(text=str(altitude))
        self._stat_labels["period"].configure(text=f"{period:.1f}")
        self._stat_labels["mm"].configure(text=f"{mm:.6f}")

        self._render_cards()
        self._render_raw()

    # ── Render cards ──────────────────────────────────────────────────────────

    def _render_cards(self):
        for w in self._cards_inner.winfo_children():
            w.destroy()

        if not self._last_tles:
            tk.Label(self._cards_inner, text="🛰  Configure parameters and click Generate",
                     bg=BG, fg=DIM, font=("Helvetica", 10)).pack(pady=60)
            return

        for i, t in enumerate(self._last_tles):
            card = tk.Frame(self._cards_inner, bg=PANEL, bd=0,
                            highlightthickness=1, highlightbackground=BORDER)
            card.pack(fill="x", pady=(0, 10), padx=2)

            # Header row
            hdr = tk.Frame(card, bg=PANEL)
            hdr.pack(fill="x", padx=14, pady=(12, 4))

            role_colour = ROLE_COLOURS.get(t["role"], TEXT)
            tk.Label(hdr, text=t["line0"].strip(), bg=PANEL, fg=ACCENT2,
                     font=("Courier", 9, "bold")).pack(side="left")

            badge_text = {"interferer": "INTERFERER",
                          "victim": "VICTIM",
                          "third": "THIRD-PARTY"}.get(t["role"], t["role"].upper())
            tk.Label(hdr, text=badge_text, bg=PANEL, fg=role_colour,
                     font=("Helvetica", 7, "bold"),
                     relief="flat", padx=6, pady=2).pack(side="right", padx=(4, 0))

            copy_btn = tk.Button(hdr, text="Copy", bg=PANEL, fg=DIM,
                                 activebackground=PANEL, activeforeground=GREEN,
                                 relief="flat", bd=0,
                                 highlightthickness=1, highlightbackground=BORDER,
                                 font=("Helvetica", 7), padx=6, pady=1,
                                 cursor="hand2",
                                 command=lambda idx=i: self._copy_single(idx))
            copy_btn.pack(side="right")

            # TLE lines
            lines_frame = tk.Frame(card, bg=PANEL)
            lines_frame.pack(fill="x", padx=14, pady=4)
            tk.Label(lines_frame, text=t["line0"].strip(), bg=PANEL, fg=LINE0_CLR,
                     font=("Courier", 9), anchor="w").pack(fill="x")
            tk.Label(lines_frame, text=t["line1"], bg=PANEL, fg=LINE_CLR,
                     font=("Courier", 9), anchor="w").pack(fill="x")
            tk.Label(lines_frame, text=t["line2"], bg=PANEL, fg=LINE_CLR,
                     font=("Courier", 9), anchor="w").pack(fill="x")

            # Meta row
            sep_frame = tk.Frame(card, bg="#0f2035", height=1)
            sep_frame.pack(fill="x", padx=14, pady=(6, 4))

            meta = tk.Frame(card, bg=PANEL)
            meta.pack(fill="x", padx=14, pady=(0, 12))
            cs1 = "✓" if t["cs1_ok"] else "✗"
            cs2 = "✓" if t["cs2_ok"] else "✗"
            cs1_col = GREEN if t["cs1_ok"] else "#ff4444"
            cs2_col = GREEN if t["cs2_ok"] else "#ff4444"

            items = [
                ("Alt", f"{t['altitude']} km", TEXT),
                ("Inc", f"{t['inclination']}°", TEXT),
                ("RAAN", f"{t['raan']}°", TEXT),
                ("M₀",  f"{t['ma']}°", TEXT),
                ("Period", f"{t['period']} min", TEXT),
                ("CS₁", cs1, cs1_col),
                ("CS₂", cs2, cs2_col),
            ]
            for lbl, val, vcol in items:
                pair = tk.Frame(meta, bg=PANEL)
                pair.pack(side="left", padx=(0, 14))
                tk.Label(pair, text=lbl, bg=PANEL, fg=DIM,
                         font=("Helvetica", 7)).pack(side="left")
                tk.Label(pair, text=" " + val, bg=PANEL, fg=vcol,
                         font=("Helvetica", 7)).pack(side="left")

        self._cards_canvas.update_idletasks()
        self._cards_canvas.configure(
            scrollregion=self._cards_canvas.bbox("all"))

    # ── Render raw ────────────────────────────────────────────────────────────

    def _render_raw(self):
        raw = "\n".join(
            f"{t['line0'].strip()}\n{t['line1']}\n{t['line2']}"
            for t in self._last_tles
        ) if self._last_tles else "— no TLEs generated yet —"

        self._raw_text.configure(state="normal")
        self._raw_text.delete("1.0", "end")
        self._raw_text.insert("end", raw)
        self._raw_text.configure(state="disabled")

    # ── Clipboard ─────────────────────────────────────────────────────────────

    def _copy_single(self, i):
        t = self._last_tles[i]
        text = f"{t['line0'].strip()}\n{t['line1']}\n{t['line2']}"
        self.clipboard_clear()
        self.clipboard_append(text)

    def _copy_all(self):
        raw = "\n".join(
            f"{t['line0'].strip()}\n{t['line1']}\n{t['line2']}"
            for t in self._last_tles
        )
        self.clipboard_clear()
        self.clipboard_append(raw)
        self._copy_all_btn.configure(text="Copied!", fg=GREEN)
        self.after(1500, lambda: self._copy_all_btn.configure(
            text="Copy All", fg=DIM))


if __name__ == "__main__":
    app = TLEGenerator()

    # Wire up tab switching properly after geometry is settled
    def _init_tabs():
        app._switch_tab("cards")
    app.after(100, _init_tabs)

    app.mainloop()
