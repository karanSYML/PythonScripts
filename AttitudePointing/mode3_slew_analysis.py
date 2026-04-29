#!/usr/bin/env python3
"""
mode3_slew_analysis.py  (slew_analysis.py)
==========================================
Compares the slew cost of transitioning into the combined Earth+Target
attitude from Mode 1 (Target+Sun) vs Mode 2 (Nadir+Sun).

Combined Earth+Target attitude definition:
    Primary:   +Z_body = target direction (camera, exact)
    Secondary: +Y_body = closest to Earth direction (antenna, best-effort)

Reads .mat data packages via scipy.io.

Produces:
    slew_far.png   — far range (−60 to −5 km)
    slew_close.png — close range (−5 to +1 km)

Usage:
    python mode3_slew_analysis.py \\
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

ANTENNA_HCONE  = 30.0
CLOSE_KM       = -5.0
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

    # Quaternion: shape (4, n) → (n, 4), scalar-first convention
    quats = d["quat_ECI2MRF"].T

    return {
        "n":       n,
        "ang":     ang,
        "s2e":     s2e,
        "s2t":     s2t,
        "quats":   quats,
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
        "quats":   cat("quats"),
        "dlambda": cat("dlambda"),
        "man_rcs": _days(eph_rcs),
        "man_pps": _days(rdv["eph_pps"]),
    }


# ── Attitude computation ──────────────────────────────────────────────────────

def quat_to_dcm(q):
    """Quaternion (scalar-first: w, x, y, z) → DCM (ECI → body)."""
    w, x, y, z = q
    return np.array([
        [w*w + x*x - y*y - z*z, 2*(x*y + w*z),         2*(x*z - w*y)],
        [2*(x*y - w*z),         w*w - x*x + y*y - z*z,  2*(y*z + w*x)],
        [2*(x*z + w*y),         2*(y*z - w*x),           w*w - x*x - y*y + z*z]
    ])


def combined_dcm(s2e_i, s2t_i):
    """
    Combined Earth+Target attitude DCM at one timestep.
    Primary:   +Z = target direction
    Secondary: +Y = closest to Earth (projection onto plane ⊥ Z)
    """
    z = s2t_i
    earth = s2e_i
    y = earth - np.dot(earth, z) * z
    yn = np.linalg.norm(y)
    if yn < 1e-10:
        y = np.cross(z, [1, 0, 0])
        if np.linalg.norm(y) < 1e-10:
            y = np.cross(z, [0, 1, 0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.array([x, y, z])   # rows = body X, Y, Z in ECI


def dcm_angle(dcm1, dcm2):
    """Eigen-angle between two DCMs [deg]."""
    R = dcm1 @ dcm2.T
    trace = np.clip(np.trace(R), -1.0, 3.0)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def compute_slews(m1, m2):
    """Compute slew angles and antenna error for all timesteps."""
    n = len(m1["days"])
    slew_m1 = np.zeros(n)
    slew_m2 = np.zeros(n)
    ant_err = np.zeros(n)

    for i in range(n):
        c_dcm     = combined_dcm(m1["s2e"][i], m1["s2t"][i])
        dcm_m1    = quat_to_dcm(m1["quats"][i])
        dcm_m2    = quat_to_dcm(m2["quats"][i])
        slew_m1[i] = dcm_angle(dcm_m1, c_dcm)
        slew_m2[i] = dcm_angle(dcm_m2, c_dcm)

        # Antenna error in combined attitude
        z = m1["s2t"][i]
        earth = m1["s2e"][i]
        proj = earth - np.dot(earth, z) * z
        pn = np.linalg.norm(proj)
        if pn < 1e-10:
            ant_err[i] = 90.0
        else:
            y = proj / pn
            err_yp = np.degrees(np.arccos(np.clip(np.dot(y, earth), -1, 1)))
            ant_err[i] = min(err_yp, 180.0 - err_yp)

    return slew_m1, slew_m2, ant_err


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


# ── Figure generation ─────────────────────────────────────────────────────────

def generate_figure(days, ang, slew_m1, slew_m2, ant_err, dlambda,
                    man_rcs, man_pps, x_range, phase_label, filepath, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    txt = "#334155"
    lo, hi = x_range
    mask = (days >= lo) & (days <= hi)
    d     = days[mask]
    a     = ang[mask]
    s1    = slew_m1[mask]
    s2    = slew_m2[mask]
    ae    = ant_err[mask]
    dlam  = dlambda[mask]
    win_int = WINDOW_HRS / 24.0
    win_dur = WINDOW_DUR_MIN / 1440.0

    fig, axes = plt.subplots(4, 1, figsize=(15, 14),
                             gridspec_kw={"height_ratios": [0.9, 1.2, 1.0, 0.6],
                                          "hspace": 0.10})
    fig.patch.set_facecolor("white")
    for ax in axes:
        _style_ax(ax)

    # ── Panel 1: Earth-Target separation ─────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(d, a, color="#7C3AED", lw=0.7, alpha=0.9,
             label="Earth–Target angular separation")
    ax1.axhline(90, color="#059669", lw=1.3, ls="-", alpha=0.7,
                label="90° (camera ⊥ antenna)")
    ax1.fill_between(d, 90 - ANTENNA_HCONE, 90 + ANTENNA_HCONE,
                     alpha=0.12, color="#059669",
                     label=f"Feasible band ±{ANTENNA_HCONE:.0f}°")
    ax1.set_ylim(0, 185)
    ax1.set_ylabel("Earth–Target\nang. sep. [deg]", fontsize=10,
                   color=txt, fontweight="medium")
    ax1.legend(fontsize=8, framealpha=0.9, loc="upper right", edgecolor="#E2E8F0")
    _add_maneuvers(ax1, man_rcs, man_pps, lo, hi)

    # ── Panel 2: Slew angles ──────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(d, s1, color="#2563EB", lw=0.7, alpha=0.9,
             label=f"Mode 1 → Combined  (mean {s1.mean():.0f}°)")
    ax2.plot(d, s2, color="#DC2626", lw=0.7, alpha=0.9,
             label=f"Mode 2 → Combined  (mean {s2.mean():.0f}°)")
    ax2.fill_between(d, s1, alpha=0.07, color="#2563EB")
    ax2.fill_between(d, s2, alpha=0.06, color="#DC2626")
    m1_cheaper = np.sum(s1 < s2) / len(s1) * 100
    ax2.text(0.015, 0.97,
             f"Mode 1 cheaper: {m1_cheaper:.0f}% of the time\n"
             f"Mode 1: min {s1.min():.0f}°  mean {s1.mean():.0f}°  max {s1.max():.0f}°\n"
             f"Mode 2: min {s2.min():.0f}°  mean {s2.mean():.0f}°  max {s2.max():.0f}°",
             transform=ax2.transAxes, fontsize=8.5, va="top",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor="#CBD5E1", alpha=0.93))
    ax2.set_ylim(0, 185)
    ax2.set_ylabel("Slew to combined\nattitude [deg]", fontsize=10,
                   color=txt, fontweight="medium")
    ax2.legend(fontsize=9, framealpha=0.9, loc="upper right", edgecolor="#E2E8F0")
    _add_maneuvers(ax2, man_rcs, man_pps, lo, hi)

    # ── Panel 3: Antenna error in combined attitude ───────────────────────────
    ax3 = axes[2]
    ax3.plot(d, ae, color="#E11D48", lw=0.7, alpha=0.9,
             label="Antenna error in combined attitude (best of ±Y)")
    ax3.fill_between(d, 0, ae, alpha=0.08, color="#E11D48")
    ax3.axhline(ANTENNA_HCONE, color="#059669", lw=1.5, ls="-",
                label=f"S-band 3 dB half-cone {ANTENNA_HCONE:.0f}°")
    ax3.fill_between(d, 0, ANTENNA_HCONE, alpha=0.08, color="#059669")
    # ConOps windows
    for d0 in np.arange(lo, hi, win_int):
        ax3.axvspan(d0, min(d0 + win_dur, hi), alpha=0.08, color="#F59E0B", zorder=0)
    pct_ok = np.sum(ae <= ANTENNA_HCONE) / len(ae) * 100
    ax3.text(0.985, 0.97,
             f"Antenna within {ANTENNA_HCONE:.0f}°: {pct_ok:.0f}%",
             transform=ax3.transAxes, fontsize=9, va="top", ha="right",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CBD5E1", alpha=0.93))
    ax3.set_ylim(0, 95)
    ax3.set_ylabel("S-band antenna\nerror [deg]", fontsize=10,
                   color=txt, fontweight="medium")
    ax3.legend(fontsize=8, framealpha=0.9, loc="upper right", edgecolor="#E2E8F0")
    _add_maneuvers(ax3, man_rcs, man_pps, lo, hi)

    # ── Panel 4: along-track ─────────────────────────────────────────────────
    ax4 = axes[3]
    ax4.plot(d, dlam, color="#1E293B", lw=0.8)
    ax4.fill_between(d, dlam, alpha=0.06, color="#1E293B")
    ax4.axhline(0, color="#DC2626", lw=0.6, ls="--", alpha=0.5)
    ax4.set_ylabel("a·δλ [km]", fontsize=10, color=txt, fontweight="medium")
    ax4.set_xlabel("Mission elapsed time [days]", fontsize=10, color=txt)
    _add_maneuvers(ax4, man_rcs, man_pps, lo, hi)

    for ax in axes:
        ax.set_xlim(lo, hi)

    fig.suptitle(
        f"Slew Analysis — {phase_label}\n"
        "Mode 1 (Target+Sun) vs Mode 2 (Nadir+Sun) → Combined Earth+Target attitude",
        fontsize=13, color=txt, fontweight="bold", y=0.99, linespacing=1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Slew analysis: Mode 1 vs Mode 2")
    p.add_argument("--mode1-dir", default="end1_target_sunOpt")
    p.add_argument("--mode2-dir", default="end1_Nadir_sunOpt")
    p.add_argument("--output-far",   default="slew_far.png")
    p.add_argument("--output-close", default="slew_close.png")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("  Slew Analysis: Mode 1 vs Mode 2 → Combined Earth+Target")
    print(f"{'='*60}")

    print("\nLoading Mode 1 (Target+Sun)...")
    m1 = load_mode(args.mode1_dir)
    print(f"  {len(m1['days'])} points, {m1['days'][-1]:.2f} days")

    print("Loading Mode 2 (Nadir+Sun)...")
    m2 = load_mode(args.mode2_dir)
    print(f"  {len(m2['days'])} points, {m2['days'][-1]:.2f} days")

    print("\nComputing slew angles (this may take ~30 s)...")
    slew_m1, slew_m2, ant_err = compute_slews(m1, m2)

    print(f"  Mode 1→Combined: mean {slew_m1.mean():.1f}°  "
          f"range {slew_m1.min():.0f}°–{slew_m1.max():.0f}°")
    print(f"  Mode 2→Combined: mean {slew_m2.mean():.1f}°  "
          f"range {slew_m2.min():.0f}°–{slew_m2.max():.0f}°")
    m1_cheaper = np.sum(slew_m1 < slew_m2) / len(slew_m1) * 100
    print(f"  Mode 1 is cheaper: {m1_cheaper:.1f}% of the time")

    close_idx = np.where(m1["dlambda"] > CLOSE_KM)[0]
    phase_day = m1["days"][close_idx[0]] if close_idx.size > 0 else m1["days"][-1] * 0.6
    print(f"\n  Phase split day: {phase_day:.2f} d")

    far_range   = (0.0, phase_day)
    close_range = (phase_day, m1["days"][-1])

    print("\nGenerating far-range figure...")
    generate_figure(m1["days"], m1["ang"], slew_m1, slew_m2, ant_err,
                    m1["dlambda"], m1["man_rcs"], m1["man_pps"],
                    far_range, "Far Range (−60 to −5 km)",
                    args.output_far, args.dpi)

    print("Generating close-range figure...")
    generate_figure(m1["days"], m1["ang"], slew_m1, slew_m2, ant_err,
                    m1["dlambda"], m1["man_rcs"], m1["man_pps"],
                    close_range, "Close Range (−5 to +1 km)",
                    args.output_close, args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
