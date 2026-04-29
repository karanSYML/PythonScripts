#!/usr/bin/env python3
"""
pointing_feasibility.py  (mode3_feasibility.py)
================================================
Analyzes the feasibility of simultaneous Earth + Target pointing for
Mode 1 (Target+Sun) and Mode 2 (Nadir+Sun).

Spacecraft geometry:
    Camera boresight: +Z body axis
    S-band antenna:   +Y body axis (also on −Y)

Feasibility metric per mode
  Mode 1 (Target+Sun): camera locked on target (+Z → target).
    Residual rotation about +Z positions +Y as close to Earth as possible.
    Antenna error = |90° − Earth-Target angular separation|
    (because camera and antenna are fixed 90° apart on the body)

  Mode 2 (Nadir+Sun): camera at nadir (+Z → Earth).
    To also image the target, the camera must rotate from nadir by the
    full Earth-Target angular separation (≥ 40° at GEO range).
    Target pointing error ≡ Earth-Target angular separation itself.

Reads .mat data packages via scipy.io.

Produces:
    feasibility_far.png   — far range (−60 to −5 km)
    feasibility_close.png — close range (−5 to +1 km)

Usage:
    python mode3_feasibility.py \\
        --mode1-dir end1_target_sunOpt \\
        --mode2-dir end1_Nadir_sunOpt
"""

import argparse
import os
import sys
import warnings
import numpy as np
import scipy.io
from datetime import datetime, timezone

# ── Mission constants ──────────────────────────────────────────────────────────
DT_DAYS        = 300.0 / 86400.0
J2000          = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
T0_UTC         = datetime(2028, 9, 1,  3, 49, 53, tzinfo=timezone.utc)
T0_SEC         = (T0_UTC - J2000).total_seconds()

ANTENNA_HCONE  = 30.0   # S-band 3 dB half-cone [deg] (60° full cone confirmed)
CLOSE_KM       = -5.0   # along-track boundary far ↔ close
WINDOW_HRS     = 6.0
WINDOW_DUR_MIN = 25.0


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

    sat_pos = d["pos_Earth2satCoG_ECI_m"].T
    tgt_pos = d["pos_Earth2tgtCoG_ECI_m"].T

    s2e = -sat_pos;  s2e /= np.linalg.norm(s2e, axis=1, keepdims=True)
    s2t = tgt_pos - sat_pos;  s2t /= np.linalg.norm(s2t, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((s2e * s2t).sum(axis=1), -1, 1)))

    return {
        "n":       n,
        "ang":     ang,
        "s2e":     s2e,
        "s2t":     s2t,
        "dlambda": d["a_dlambda_m"].flatten() / 1e3,
        "eph_rcs": _safe_flat(man["manEphSec_rcs"]),
        "eph_pps": _safe_flat(man["manEphSec_pps"]),
    }


def load_mode(mat_dir):
    rdv = load_mat(os.path.join(mat_dir, "dataPackage4Thermal_RDV.mat"))
    ins = load_mat(os.path.join(mat_dir, "dataPackage4Thermal_INS.mat"))
    nr, ni = rdv["n"], ins["n"]

    days = np.r_[np.arange(nr) * DT_DAYS,
                 nr * DT_DAYS + np.arange(ni) * DT_DAYS]

    def cat(k):
        return np.concatenate([rdv[k], ins[k]], axis=0)

    def _days(arr):
        return (arr - T0_SEC) / 86400.0 if arr.size > 0 else np.array([])

    eph_rcs = np.r_[rdv["eph_rcs"], ins["eph_rcs"]]
    return {
        "days":    days,
        "ang":     cat("ang"),
        "s2e":     cat("s2e"),
        "s2t":     cat("s2t"),
        "dlambda": cat("dlambda"),
        "man_rcs": _days(eph_rcs),
        "man_pps": _days(rdv["eph_pps"]),
    }


# ── Computation ───────────────────────────────────────────────────────────────

def antenna_error_mode1(s2e, s2t):
    """
    Mode 1: camera (+Z) on target, rotate about +Z to minimise antenna error.
    Optimal residual = |90° − ang_sep|  (exact — see analysis doc).
    Both +Y and −Y antennas considered; best is returned.
    """
    n = len(s2e)
    err_yp = np.zeros(n)
    for i in range(n):
        z = s2t[i]
        earth = s2e[i]
        proj = earth - np.dot(earth, z) * z
        pn = np.linalg.norm(proj)
        if pn < 1e-10:
            err_yp[i] = 90.0
            continue
        y = proj / pn
        err_yp[i] = np.degrees(np.arccos(np.clip(np.dot(y, earth), -1, 1)))
    err_ym = 180.0 - err_yp          # flipping to −Y antenna
    return np.minimum(err_yp, err_ym)


def target_error_mode2(ang):
    """
    Mode 2: camera (+Z) at nadir (Earth).
    To point at target, the camera must rotate by the full Earth-Target
    angular separation — there is no residual optimization freedom.
    """
    return ang.copy()


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


def _feasibility_table(err, threshold=ANTENNA_HCONE):
    rows = [f"{'Threshold':<9} {'Feasibility':>11}", "─" * 21]
    for bw in [10, 20, 30, 45]:
        pct = np.sum(err <= bw) / len(err) * 100
        marker = " ←" if bw == int(threshold) else ""
        rows.append(f"≤{bw:2d}°{'':<6} {pct:6.1f}%{marker}")
    return "\n".join(rows)


# ── Figure generation ─────────────────────────────────────────────────────────

def generate_figure(m1, m2, x_range, phase_label, filepath, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    txt = "#334155"
    lo, hi = x_range
    mask1 = (m1["days"] >= lo) & (m1["days"] <= hi)
    mask2 = (m2["days"] >= lo) & (m2["days"] <= hi)

    d1    = m1["days"][mask1]
    ang1  = m1["ang"][mask1]
    err1  = antenna_error_mode1(m1["s2e"][mask1], m1["s2t"][mask1])

    d2    = m2["days"][mask2]
    ang2  = m2["ang"][mask2]
    err2  = target_error_mode2(ang2)   # Mode 2: target error = ang_sep

    dlam  = m1["dlambda"][mask1]
    win_int = WINDOW_HRS / 24.0
    win_dur = WINDOW_DUR_MIN / 1440.0

    fig, axes = plt.subplots(3, 1, figsize=(15, 12),
                             gridspec_kw={"height_ratios": [1.1, 1.4, 0.65],
                                          "hspace": 0.11})
    fig.patch.set_facecolor("white")
    for ax in axes:
        _style_ax(ax)

    # ── Panel 1: Earth-Target angular separation ─────────────────────────────
    ax1 = axes[0]
    ax1.plot(d1, ang1, color="#7C3AED", lw=0.7, alpha=0.9,
             label="Earth–Target separation (same for both modes)")
    ax1.axhline(90, color="#059669", lw=1.4, ls="-", alpha=0.75,
                label="90° (camera ⊥ antenna)")
    ax1.fill_between(d1, 90 - ANTENNA_HCONE, 90 + ANTENNA_HCONE,
                     alpha=0.12, color="#059669",
                     label=f"Feasible band ±{ANTENNA_HCONE:.0f}°")
    ax1.fill_between(d1, 90, ang1, where=ang1 > 90, alpha=0.09, color="#DC2626")
    ax1.fill_between(d1, 90, ang1, where=ang1 <= 90, alpha=0.09, color="#2563EB")
    ax1.set_ylim(0, 185)
    ax1.set_ylabel("Earth–Target\nangular sep. [deg]", fontsize=10,
                   color=txt, fontweight="medium")
    ax1.legend(fontsize=8, framealpha=0.9, loc="upper right", edgecolor="#E2E8F0")
    _add_maneuvers(ax1, m1["man_rcs"], m1["man_pps"], lo, hi)

    # ── Panel 2: Pointing errors — Mode 1 (antenna) vs Mode 2 (target) ───────
    ax2 = axes[1]
    ax2.plot(d1, err1, color="#E11D48", lw=0.7, alpha=0.9,
             label="Mode 1 antenna error  |90° − sep|  (best of ±Y)")
    ax2.plot(d2, err2, color="#7C3AED", lw=0.7, alpha=0.75,
             label="Mode 2 target error  = ang_sep")
    ax2.fill_between(d1, 0, err1, alpha=0.07, color="#E11D48")
    ax2.fill_between(d2, 0, err2, alpha=0.05, color="#7C3AED")

    # S-band 3 dB half-cone
    ax2.axhline(ANTENNA_HCONE, color="#059669", lw=1.6, ls="-",
                label=f"S-band 3 dB half-cone {ANTENNA_HCONE:.0f}°")
    ax2.fill_between(d1, 0, ANTENNA_HCONE, alpha=0.07, color="#059669")

    # ConOps windows
    for d0 in np.arange(lo, hi, win_int):
        ax2.axvspan(d0, min(d0 + win_dur, hi), alpha=0.08, color="#F59E0B", zorder=0)

    ax2.set_ylim(0, 95)
    ax2.set_ylabel("Pointing error [deg]", fontsize=10,
                   color=txt, fontweight="medium")
    ax2.legend(fontsize=8.5, framealpha=0.9, loc="upper right", edgecolor="#E2E8F0")
    _add_maneuvers(ax2, m1["man_rcs"], m1["man_pps"], lo, hi)

    # Feasibility inset table for Mode 1
    ax2.text(0.015, 0.97,
             f"Mode 1 (antenna feasibility)\n{_feasibility_table(err1)}",
             transform=ax2.transAxes, fontsize=7.5, va="top",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor="#CBD5E1", alpha=0.93))
    m1_ok = np.sum(err1 <= ANTENNA_HCONE) / len(err1) * 100
    m2_ok = np.sum(err2 <= ANTENNA_HCONE) / len(err2) * 100
    ax2.text(0.985, 0.97,
             f"Within {ANTENNA_HCONE:.0f}° cone:\n"
             f"  Mode 1: {m1_ok:.0f}%\n"
             f"  Mode 2: {m2_ok:.0f}%",
             transform=ax2.transAxes, fontsize=8.5, va="top", ha="right",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor="#CBD5E1", alpha=0.93))

    # ── Panel 3: along-track ─────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(d1, dlam, color="#1E293B", lw=0.8)
    ax3.fill_between(d1, dlam, alpha=0.06, color="#1E293B")
    ax3.axhline(0, color="#DC2626", lw=0.6, ls="--", alpha=0.5)
    ax3.set_ylabel("a·δλ [km]", fontsize=10, color=txt, fontweight="medium")
    ax3.set_xlabel("Mission elapsed time [days]", fontsize=10, color=txt)
    _add_maneuvers(ax3, m1["man_rcs"], m1["man_pps"], lo, hi)

    for ax in axes:
        ax.set_xlim(lo, hi)

    fig.suptitle(
        f"Earth + Target Pointing Feasibility — {phase_label}\n"
        f"Mode 1: antenna error  |  Mode 2: target error  |  S-band {ANTENNA_HCONE:.0f}° half-cone",
        fontsize=13, color=txt, fontweight="bold", y=0.99, linespacing=1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {filepath}")


def print_summary(label, err, threshold=ANTENNA_HCONE):
    print(f"\n  {label}:")
    print(f"    Error range: {err.min():.1f}° – {err.max():.1f}°  (mean {err.mean():.1f}°)")
    for bw in [10, 20, 30, 45]:
        pct = np.sum(err <= bw) / len(err) * 100
        marker = " ← S-band 3 dB" if bw == int(threshold) else ""
        print(f"    ≤{bw:2d}°: {pct:5.1f}%{marker}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Earth+Target combined pointing feasibility")
    p.add_argument("--mode1-dir", default="end1_target_sunOpt",
                   help="Mode 1 (Target+Sun) mat directory")
    p.add_argument("--mode2-dir", default="end1_Nadir_sunOpt",
                   help="Mode 2 (Nadir+Sun) mat directory")
    p.add_argument("--output-far",   default="feasibility_far.png")
    p.add_argument("--output-close", default="feasibility_close.png")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Earth+Target Pointing Feasibility")
    print(f"  Camera +Z  |  Antenna ±Y  |  S-band {ANTENNA_HCONE:.0f}° half-cone")
    print(f"{'='*60}")

    print("\nLoading Mode 1 (Target+Sun)...")
    m1 = load_mode(args.mode1_dir)
    print(f"  {len(m1['days'])} points, {m1['days'][-1]:.2f} days")

    print("Loading Mode 2 (Nadir+Sun)...")
    m2 = load_mode(args.mode2_dir)
    print(f"  {len(m2['days'])} points, {m2['days'][-1]:.2f} days")

    # Phase split
    close_idx = np.where(m1["dlambda"] > CLOSE_KM)[0]
    phase_day = m1["days"][close_idx[0]] if close_idx.size > 0 else m1["days"][-1] * 0.6
    print(f"\n  Phase split day: {phase_day:.2f} d")

    far_range   = (0.0, phase_day)
    close_range = (phase_day, m1["days"][-1])

    # Full-timeline summaries
    err_m1_all = antenna_error_mode1(m1["s2e"], m1["s2t"])
    err_m2_all = target_error_mode2(m1["ang"])
    print_summary("Mode 1 antenna error (full timeline)", err_m1_all)
    print_summary("Mode 2 target error  (full timeline)", err_m2_all)

    print("\nGenerating far-range figure...")
    generate_figure(m1, m2, far_range,   "Far Range (−60 to −5 km)",
                    args.output_far, args.dpi)

    print("Generating close-range figure...")
    generate_figure(m1, m2, close_range, "Close Range (−5 to +1 km)",
                    args.output_close, args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
