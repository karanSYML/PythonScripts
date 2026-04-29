#!/usr/bin/env python3
"""
earth_target_separation.py
==========================
Computes the geometric angular separation between Earth and rendezvous
target as seen from the spacecraft (attitude-independent).

Reads .mat data packages directly via scipy.io.loadmat.

Produces two figures:
  fig1a_earth_target_far.png   — far range (−60 to −5 km)
  fig1b_earth_target_close.png — close range (−5 to +1 km) with 90°
                                  crossing analysis and periodicity

Usage:
    python earth_target_separation.py --mat-dir end1_target_sunOpt
"""

import argparse
import os
import sys
import warnings
import numpy as np
import scipy.io
from datetime import datetime, timezone

# ── Mission constants ──────────────────────────────────────────────────────────
DT_DAYS  = 300.0 / 86400.0
J2000    = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
T0_UTC   = datetime(2028, 9, 1,  3, 49, 53, tzinfo=timezone.utc)
T0_SEC   = (T0_UTC - J2000).total_seconds()

CLOSE_KM       = -5.0   # along-track boundary far ↔ close
ANTENNA_HCONE  = 30.0   # S-band half-cone angle [deg] — confirmed 60° full cone
WINDOW_HRS     = 6.0    # ConOps window interval [h]
WINDOW_DUR_MIN = 25.0   # ConOps window duration [min]


# ── Data loading ──────────────────────────────────────────────────────────────

def _safe_flat(v):
    if v.size == 0:
        return np.array([])
    flat = v.flatten()
    return flat.astype(float) if flat.dtype.kind in "fiuc" else np.array([])


def load_mat(fp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = scipy.io.loadmat(fp)
    d   = mat["data4thermal"][0, 0]
    man = d["maneuvers"][0, 0]
    n   = d["pos_Earth2satCoG_ECI_m"].shape[1]

    # ECI positions (m)
    sat_pos = d["pos_Earth2satCoG_ECI_m"].T   # (n, 3)
    tgt_pos = d["pos_Earth2tgtCoG_ECI_m"].T

    # Angular separation (compute from ECI positions)
    s2e = -sat_pos
    s2e /= np.linalg.norm(s2e, axis=1, keepdims=True)
    s2t  = tgt_pos - sat_pos
    s2t /= np.linalg.norm(s2t,  axis=1, keepdims=True)
    dot  = np.clip((s2e * s2t).sum(axis=1), -1, 1)
    ang  = np.degrees(np.arccos(dot))

    return {
        "n":       n,
        "ang":     ang,
        "dlambda": d["a_dlambda_m"].flatten() / 1e3,
        "eph_rcs": _safe_flat(man["manEphSec_rcs"]),
        "eph_pps": _safe_flat(man["manEphSec_pps"]),
    }


def load_data(mat_dir):
    rdv = load_mat(os.path.join(mat_dir, "dataPackage4Thermal_RDV.mat"))
    ins = load_mat(os.path.join(mat_dir, "dataPackage4Thermal_INS.mat"))
    nr, ni = rdv["n"], ins["n"]

    days    = np.r_[np.arange(nr) * DT_DAYS,
                    nr * DT_DAYS + np.arange(ni) * DT_DAYS]
    ang     = np.r_[rdv["ang"],     ins["ang"]]
    dlambda = np.r_[rdv["dlambda"], ins["dlambda"]]

    def _days(arr):
        return (arr - T0_SEC) / 86400.0 if arr.size > 0 else np.array([])

    eph_rcs = np.r_[rdv["eph_rcs"], ins["eph_rcs"]]
    return {
        "days":    days,
        "ang":     ang,
        "dlambda": dlambda,
        "man_rcs": _days(eph_rcs),
        "man_pps": _days(rdv["eph_pps"]),
    }


# ── Analysis helpers ──────────────────────────────────────────────────────────

def find_crossings_90(days, ang):
    """Return interpolated mission-day of each 90° crossing."""
    diff = ang - 90.0
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    crossings = []
    for i in sign_changes:
        d1, d2 = days[i], days[i + 1]
        a1, a2 = diff[i], diff[i + 1]
        t = d1 - a1 * (d2 - d1) / (a2 - a1)
        crossings.append(t)
    return np.array(crossings)


def time_to_next_90(window_starts_days, crossings_days):
    """For each window start, time in minutes until the next 90° crossing."""
    result = []
    for ws in window_starts_days:
        future = crossings_days[crossings_days >= ws]
        result.append((future[0] - ws) * 1440.0 if future.size > 0 else np.nan)
    return np.array(result)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor("#FAFBFC")
    ax.grid(True, alpha=0.45, color="#E2E8F0", linewidth=0.5)
    ax.tick_params(colors="#334155", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#CBD5E1"); sp.set_linewidth(0.5)


def _add_maneuvers(ax, man_rcs, man_pps, lo, hi):
    for d in man_rcs[(man_rcs >= lo) & (man_rcs <= hi)]:
        ax.axvline(d, color="#3B82F6", lw=0.6, alpha=0.45, ls=":", zorder=2)
    for d in man_pps[(man_pps >= lo) & (man_pps <= hi)]:
        ax.axvline(d, color="#F97316", lw=0.8, alpha=0.55, ls="--", zorder=2)


# ── Far-range figure ──────────────────────────────────────────────────────────

def plot_far(data, x_range, filepath, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    txt  = "#334155"
    lo, hi = x_range
    mask = (data["days"] >= lo) & (data["days"] <= hi)
    days    = data["days"][mask]
    ang     = data["ang"][mask]
    dlambda = data["dlambda"][mask]
    win_int = WINDOW_HRS / 24.0
    win_dur = WINDOW_DUR_MIN / 1440.0

    fig, axes = plt.subplots(2, 1, figsize=(15, 8),
                             gridspec_kw={"height_ratios": [1.4, 0.7], "hspace": 0.10})
    fig.patch.set_facecolor("white")
    for ax in axes:
        _style_ax(ax)

    # Panel 1: angular separation
    ax1 = axes[0]
    ax1.plot(days, ang, color="#7C3AED", lw=0.7, alpha=0.9)
    ax1.fill_between(days, 90, ang, where=ang > 90, alpha=0.10, color="#DC2626")
    ax1.fill_between(days, 90, ang, where=ang <= 90, alpha=0.10, color="#2563EB")
    ax1.axhline(90, color="#059669", lw=1.5, ls="-", alpha=0.8,
                label="90° (camera ⊥ antenna)")
    # Feasibility band
    ax1.fill_between(days, 90 - ANTENNA_HCONE, 90 + ANTENNA_HCONE,
                     alpha=0.12, color="#059669",
                     label=f"Feasible ±{ANTENNA_HCONE:.0f}° (S-band 3 dB half-cone)")
    ax1.set_ylim(0, 185)
    ax1.set_ylabel("Earth–Target\nangular sep. [deg]", fontsize=10,
                   color=txt, fontweight="medium")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9, edgecolor="#E2E8F0")
    _add_maneuvers(ax1, data["man_rcs"], data["man_pps"], lo, hi)
    ax1.text(0.01, 0.04,
             "Geometric angle between Earth & target (attitude-independent)",
             transform=ax1.transAxes, fontsize=8, color="#64748B",
             fontstyle="italic", va="bottom")

    # Statistics box
    in_band = np.sum(np.abs(ang - 90) <= ANTENNA_HCONE) / len(ang) * 100
    stats = (f"In ±{ANTENNA_HCONE:.0f}° band: {in_band:.0f}%\n"
             f"Min: {ang.min():.1f}°   Max: {ang.max():.1f}°\n"
             f"Mean: {ang.mean():.1f}°")
    ax1.text(0.985, 0.97, stats, transform=ax1.transAxes,
             fontsize=8, va="top", ha="right", family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CBD5E1", alpha=0.93))

    # Panel 2: along-track
    ax2 = axes[1]
    ax2.plot(days, dlambda, color="#1E293B", lw=0.8)
    ax2.fill_between(days, dlambda, alpha=0.06, color="#1E293B")
    ax2.axhline(0, color="#DC2626", lw=0.6, ls="--", alpha=0.5)
    ax2.set_ylabel("a·δλ [km]", fontsize=10, color=txt, fontweight="medium")
    ax2.set_xlabel("Mission elapsed time [days]", fontsize=10, color=txt)
    _add_maneuvers(ax2, data["man_rcs"], data["man_pps"], lo, hi)

    for ax in axes:
        ax.set_xlim(lo, hi)

    fig.suptitle(
        "Earth–Target Angular Separation — Far Range (−60 to −5 km)\n"
        "How far apart are the Earth and target pointing directions?",
        fontsize=13, color=txt, fontweight="bold", y=0.99, linespacing=1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {filepath}")


# ── Close-range figure with 90° crossing analysis ─────────────────────────────

def plot_close(data, x_range, filepath, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    txt  = "#334155"
    lo, hi = x_range
    mask = (data["days"] >= lo) & (data["days"] <= hi)
    days    = data["days"][mask]
    ang     = data["ang"][mask]
    dlambda = data["dlambda"][mask]
    win_int = WINDOW_HRS / 24.0
    win_dur = WINDOW_DUR_MIN / 1440.0

    # ── 90° crossing analysis ────────────────────────────────────────────────
    crossings   = find_crossings_90(days, ang)
    win_starts  = np.arange(lo, hi, win_int)
    ttc_min     = time_to_next_90(win_starts, crossings)

    if crossings.size > 1:
        gaps_hr     = np.diff(crossings) * 24.0
        max_gap_hr  = gaps_hr.max()
        mean_gap_hr = gaps_hr.mean()
        period_hr   = mean_gap_hr * 2   # half-period → full period estimate
    else:
        max_gap_hr = mean_gap_hr = period_hr = np.nan

    fig, axes = plt.subplots(3, 1, figsize=(15, 11),
                             gridspec_kw={"height_ratios": [1.8, 1.0, 0.6],
                                          "hspace": 0.10})
    fig.patch.set_facecolor("white")
    for ax in axes:
        _style_ax(ax)

    # ── Panel 1: angular separation (main analysis) ──────────────────────────
    ax1 = axes[0]

    # 6-hour window bands
    for d0 in np.arange(lo, hi, win_int):
        ax1.axvspan(d0, min(d0 + win_dur, hi), alpha=0.10, color="#F59E0B", zorder=0)

    # Feasibility shading: |ang - 90| ≤ ANTENNA_HCONE
    ax1.fill_between(days, 90 - ANTENNA_HCONE, 90 + ANTENNA_HCONE,
                     alpha=0.14, color="#059669", zorder=1,
                     label=f"Feasible band ±{ANTENNA_HCONE:.0f}° (S-band half-cone)")

    # Main trace
    ax1.plot(days, ang, color="#7C3AED", lw=1.0, alpha=0.95, zorder=3,
             label="Earth–Target separation")
    ax1.axhline(90, color="#059669", lw=1.4, ls="-", alpha=0.7,
                label="90° (ideal for combined pointing)")

    # 90° crossing markers
    for cx in crossings:
        ax1.axvline(cx, color="#059669", lw=0.8, ls="--", alpha=0.5, zorder=2)
    if crossings.size > 0:
        ax1.scatter(crossings, np.full_like(crossings, 90.0),
                    s=22, color="#059669", zorder=5,
                    label=f"90° crossings ({crossings.size} total)")

    # Annotate max gap
    if not np.isnan(max_gap_hr):
        ax1.text(0.985, 0.97,
                 f"Oscillation period ≈ {period_hr:.1f} h\n"
                 f"Max gap between crossings: {max_gap_hr:.1f} h\n"
                 f"Crossings every {mean_gap_hr:.1f} h (mean half-period)",
                 transform=ax1.transAxes, fontsize=8.5, va="top", ha="right",
                 family="monospace",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                           edgecolor="#CBD5E1", alpha=0.93))

    ax1.set_ylim(0, 185)
    ax1.set_ylabel("Earth–Target\nangular sep. [deg]", fontsize=10,
                   color=txt, fontweight="medium")
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.9, edgecolor="#E2E8F0")
    _add_maneuvers(ax1, data["man_rcs"], data["man_pps"], lo, hi)

    # ── Panel 2: time to next 90° crossing per 6-h window ───────────────────
    ax2 = axes[1]
    # Cap at WINDOW_HRS * 60 — crossings beyond one full cycle shown as clipped
    y_cap = WINDOW_HRS * 60.0
    ttc_capped = np.minimum(ttc_min, y_cap)
    clipped    = ttc_min > y_cap
    colors = np.where(ttc_min <= WINDOW_DUR_MIN, "#059669", "#E11D48")
    bars = ax2.bar(win_starts, ttc_capped, width=win_int * 0.7, align="edge",
                   color=colors, alpha=0.75, zorder=2)
    # Hatching for clipped bars (crossing beyond 6 h)
    for bar, is_clipped in zip(bars, clipped):
        if is_clipped:
            bar.set_hatch("///")
            bar.set_edgecolor("#94A3B8")
    ax2.axhline(WINDOW_DUR_MIN, color="#E11D48", lw=1.4, ls="--",
                label=f"Window duration ({WINDOW_DUR_MIN:.0f} min)")
    ax2.axhline(0, color="#334155", lw=0.4, alpha=0.5)
    ax2.set_ylim(0, y_cap * 1.05)
    ax2.set_ylabel(f"Time to 90° from\nwindow start [min]\n(capped at {y_cap:.0f} min)",
                   fontsize=9, color=txt, fontweight="medium")
    ax2.legend(fontsize=8, framealpha=0.9, loc="upper right", edgecolor="#E2E8F0")
    ax2.text(0.01, 0.95,
             "Green = 90° within window  |  Red = outside slot  |  Hatched = no crossing within 6 h",
             transform=ax2.transAxes, fontsize=8, color="#64748B",
             fontstyle="italic", va="top")
    # Count feasible windows
    n_win = np.sum(~np.isnan(ttc_min))
    n_ok  = np.sum(ttc_min <= WINDOW_DUR_MIN)
    ax2.text(0.985, 0.95,
             f"{n_ok}/{n_win} windows ({n_ok/n_win*100:.0f}%) have\n"
             f"90° within {WINDOW_DUR_MIN:.0f} min",
             transform=ax2.transAxes, fontsize=8.5, va="top", ha="right",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CBD5E1", alpha=0.93))
    _add_maneuvers(ax2, data["man_rcs"], data["man_pps"], lo, hi)

    # ── Panel 3: along-track separation ─────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(days, dlambda, color="#1E293B", lw=0.8)
    ax3.fill_between(days, dlambda, alpha=0.06, color="#1E293B")
    ax3.axhline(0, color="#DC2626", lw=0.6, ls="--", alpha=0.5)
    ax3.set_ylabel("a·δλ [km]", fontsize=10, color=txt, fontweight="medium")
    ax3.set_xlabel("Mission elapsed time [days]", fontsize=10, color=txt)
    _add_maneuvers(ax3, data["man_rcs"], data["man_pps"], lo, hi)

    for ax in axes:
        ax.set_xlim(lo, hi)

    fig.suptitle(
        "Earth–Target Angular Separation — Close Range (−5 to +1 km)\n"
        "90° crossing periodicity  |  Time to feasible geometry per 6-h window",
        fontsize=13, color=txt, fontweight="bold", y=0.99, linespacing=1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {filepath}")

    return crossings, max_gap_hr, ttc_min


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Earth–Target angular separation analysis")
    p.add_argument("--mat-dir", default="end1_target_sunOpt",
                   help="Mat data directory (geometry is mode-independent)")
    p.add_argument("--output-far",   default="fig1a_earth_target_far.png")
    p.add_argument("--output-close", default="fig1b_earth_target_close.png")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.mat_dir):
        print(f"ERROR: Directory not found: {args.mat_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  Earth–Target Angular Separation Analysis")
    print(f"{'='*60}")

    print(f"\nLoading from {args.mat_dir}...")
    data = load_data(args.mat_dir)
    days, ang, dlambda = data["days"], data["ang"], data["dlambda"]
    print(f"  {len(days)} points, {days[-1]:.2f} days")
    print(f"  ang sep: {ang.min():.1f}° – {ang.max():.1f}° (mean {ang.mean():.1f}°)")
    print(f"  along-track: {dlambda.min():.1f} km – {dlambda.max():.1f} km")

    # Phase split
    close_idx = np.where(dlambda > CLOSE_KM)[0]
    phase_day = days[close_idx[0]] if close_idx.size > 0 else days[-1] * 0.6
    print(f"  Phase split day: {phase_day:.2f} d  (dlambda > {CLOSE_KM} km)")

    far_range   = (0.0,       phase_day)
    close_range = (phase_day, days[-1])

    print("\nGenerating far-range figure...")
    plot_far(data, far_range, args.output_far, args.dpi)

    print("Generating close-range figure...")
    crossings, max_gap_hr, ttc = plot_close(data, close_range, args.output_close, args.dpi)

    print(f"\n  Close-range 90° crossings: {crossings.size}")
    if not np.isnan(max_gap_hr):
        print(f"  Max gap between crossings: {max_gap_hr:.1f} h "
              f"({'≤ 6 h ✓' if max_gap_hr <= 6 else '> 6 h ✗'})")

    print("\nDone.")


if __name__ == "__main__":
    main()
