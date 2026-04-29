#!/usr/bin/env python3
"""
thermal_constraints.py
======================
Compares thermal / operational sun-exposure constraints for
Mode 1 (Target+Sun) and Mode 2 (Nadir+Sun).

Metrics:
  Camera sun exclusion: angle between +Z boresight and Sun  (threshold 30°, TBC)
  STR blinding:         angle between +X boresight and Sun  (threshold 35°, TBC – GEO typical)

Produces separate figures for far range (−60 to −5 km) and
close range (−5 to +1 km).

Usage:
    python thermal_constraints.py \\
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

warnings.filterwarnings("ignore")

# ── Mission constants ──────────────────────────────────────────────────────────
DT_DAYS = 300.0 / 86400.0
J2000   = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
T0_UTC  = datetime(2028, 9, 1,  3, 49, 53, tzinfo=timezone.utc)
T0_SEC  = (T0_UTC - J2000).total_seconds()

# Body-axis row index in angle_sunFromMRFaxes_deg  (0=X, 1=Y, 2=Z)
CAM_AXIS = 2   # camera boresight +Z
STR_AXIS = 0   # STR boresight +X (TBC)

CAMERA_EXCL_DEG = 30.0   # half-cone exclusion (TBC)
STR_EXCL_DEG    = 35.0   # GEO typical half-cone exclusion (TBC)

CLOSE_KM = -5.0   # along-track boundary far ↔ close range


# ── Data loading ──────────────────────────────────────────────────────────────

def _safe_flat(mat_field):
    """Flatten a mat field that may be empty or variously shaped."""
    try:
        v = mat_field
        if v.size == 0:
            return np.array([])
        flat = v.flatten()
        if flat.dtype.kind in ("f", "c", "i", "u"):
            return flat.astype(float)
        return np.array([])
    except Exception:
        return np.array([])


def load_mat(filepath):
    """Load one dataPackage4Thermal .mat file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = scipy.io.loadmat(filepath)
    d   = mat["data4thermal"][0, 0]
    man = d["maneuvers"][0, 0]
    return {
        "n":        d["pos_Earth2satCoG_ECI_m"].shape[1],
        "sun_deg":  d["angle_sunFromMRFaxes_deg"].T,      # (n, 3): X Y Z
        "dlambda":  d["a_dlambda_m"].flatten() / 1e3,     # km
        "eph_rcs":  _safe_flat(man["manEphSec_rcs"]),
        "eph_pps":  _safe_flat(man["manEphSec_pps"]),
    }


def load_mode(mat_dir):
    """Load RDV + INS phases for one mode directory."""
    rdv = load_mat(os.path.join(mat_dir, "dataPackage4Thermal_RDV.mat"))
    ins = load_mat(os.path.join(mat_dir, "dataPackage4Thermal_INS.mat"))
    nr, ni = rdv["n"], ins["n"]

    days    = np.r_[np.arange(nr) * DT_DAYS,
                    nr * DT_DAYS + np.arange(ni) * DT_DAYS]
    sun_deg = np.vstack([rdv["sun_deg"], ins["sun_deg"]])
    dlambda = np.r_[rdv["dlambda"], ins["dlambda"]]

    def _to_days(arr):
        return (arr - T0_SEC) / 86400.0 if arr.size > 0 else np.array([])

    eph_rcs = np.r_[rdv["eph_rcs"], ins["eph_rcs"]]
    return {
        "days":    days,
        "sun_deg": sun_deg,
        "dlambda": dlambda,
        "man_rcs": _to_days(eph_rcs),
        "man_pps": _to_days(rdv["eph_pps"]),
    }


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _style_ax(ax, txt="#334155", grid_c="#E2E8F0"):
    ax.set_facecolor("#FAFBFC")
    ax.grid(True, alpha=0.45, color=grid_c, linewidth=0.5)
    ax.tick_params(colors=txt, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#CBD5E1"); sp.set_linewidth(0.5)


def _add_maneuvers(ax, man_rcs, man_pps, x_range):
    lo, hi = x_range
    for d in man_rcs[(man_rcs >= lo) & (man_rcs <= hi)]:
        ax.axvline(d, color="#3B82F6", lw=0.6, alpha=0.45, ls=":", zorder=2)
    for d in man_pps[(man_pps >= lo) & (man_pps <= hi)]:
        ax.axvline(d, color="#F97316", lw=0.8, alpha=0.55, ls="--", zorder=2)


def _violation_stats(ang_arr, threshold):
    """Return % of time below threshold and min angle."""
    pct = np.sum(ang_arr < threshold) / len(ang_arr) * 100
    return pct, ang_arr.min()


def generate_figure(m1, m2, x_range, phase_label, filepath, dpi):
    """
    Two-panel figure: camera sun angle (top) + STR sun angle (bottom).
    Mode 1 and Mode 2 overlaid on each panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    txt = "#334155"
    lo, hi = x_range

    mask1 = (m1["days"] >= lo) & (m1["days"] <= hi)
    mask2 = (m2["days"] >= lo) & (m2["days"] <= hi)
    d1 = m1["days"][mask1]
    d2 = m2["days"][mask2]

    cam1 = m1["sun_deg"][mask1, CAM_AXIS]
    cam2 = m2["sun_deg"][mask2, CAM_AXIS]
    str1 = m1["sun_deg"][mask1, STR_AXIS]
    str2 = m2["sun_deg"][mask2, STR_AXIS]

    fig, axes = plt.subplots(2, 1, figsize=(15, 9),
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
    fig.patch.set_facecolor("white")

    for ax in axes:
        _style_ax(ax)

    # ── Panel 1: Camera sun angle ───────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(d1, cam1, color="#7C3AED", lw=0.7, alpha=0.9, label="Mode 1 (Target+Sun)")
    ax1.plot(d2, cam2, color="#0EA5E9", lw=0.7, alpha=0.9, label="Mode 2 (Nadir+Sun)")
    ax1.axhline(CAMERA_EXCL_DEG, color="#DC2626", lw=1.4, ls="--",
                label=f"Camera exclusion {CAMERA_EXCL_DEG:.0f}° (TBC)")
    ax1.fill_between(d1, 0, CAMERA_EXCL_DEG, alpha=0.08, color="#DC2626")

    # Shade violations for each mode
    ax1.fill_between(d1, cam1, CAMERA_EXCL_DEG,
                     where=(cam1 < CAMERA_EXCL_DEG),
                     alpha=0.25, color="#7C3AED",
                     label="Mode 1 violation")
    ax1.fill_between(d2, cam2, CAMERA_EXCL_DEG,
                     where=(cam2 < CAMERA_EXCL_DEG),
                     alpha=0.25, color="#0EA5E9",
                     label="Mode 2 violation")

    ax1.set_ylim(0, 185)
    ax1.set_ylabel("Sun angle from\ncamera (+Z) [deg]", fontsize=10,
                   color=txt, fontweight="medium")
    ax1.legend(fontsize=8, framealpha=0.9, loc="upper right", edgecolor="#E2E8F0")
    _add_maneuvers(ax1, m1["man_rcs"], m1["man_pps"], x_range)

    p1_m1, min1_m1 = _violation_stats(cam1, CAMERA_EXCL_DEG)
    p1_m2, min1_m2 = _violation_stats(cam2, CAMERA_EXCL_DEG)
    note = (f"Mode 1 violation: {p1_m1:.1f}%  (min {min1_m1:.1f}°)\n"
            f"Mode 2 violation: {p1_m2:.1f}%  (min {min1_m2:.1f}°)")
    ax1.text(0.01, 0.97, note, transform=ax1.transAxes,
             fontsize=8, color=txt, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CBD5E1", alpha=0.9))

    # ── Panel 2: STR sun angle ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(d1, str1, color="#7C3AED", lw=0.7, alpha=0.9, label="Mode 1 (Target+Sun)")
    ax2.plot(d2, str2, color="#0EA5E9", lw=0.7, alpha=0.9, label="Mode 2 (Nadir+Sun)")
    ax2.axhline(STR_EXCL_DEG, color="#DC2626", lw=1.4, ls="--",
                label=f"STR exclusion {STR_EXCL_DEG:.0f}° (TBC – GEO)")
    ax2.fill_between(d1, 0, STR_EXCL_DEG, alpha=0.08, color="#DC2626")

    ax2.fill_between(d1, str1, STR_EXCL_DEG,
                     where=(str1 < STR_EXCL_DEG),
                     alpha=0.25, color="#7C3AED",
                     label="Mode 1 violation")
    ax2.fill_between(d2, str2, STR_EXCL_DEG,
                     where=(str2 < STR_EXCL_DEG),
                     alpha=0.25, color="#0EA5E9",
                     label="Mode 2 violation")

    ax2.set_ylim(0, 185)
    ax2.set_ylabel("Sun angle from\nSTR (+X, TBC) [deg]", fontsize=10,
                   color=txt, fontweight="medium")
    ax2.set_xlabel("Mission elapsed time [days]", fontsize=10, color=txt)
    ax2.legend(fontsize=8, framealpha=0.9, loc="upper right", edgecolor="#E2E8F0")
    _add_maneuvers(ax2, m1["man_rcs"], m1["man_pps"], x_range)

    p2_m1, min2_m1 = _violation_stats(str1, STR_EXCL_DEG)
    p2_m2, min2_m2 = _violation_stats(str2, STR_EXCL_DEG)
    note2 = (f"Mode 1 violation: {p2_m1:.1f}%  (min {min2_m1:.1f}°)\n"
             f"Mode 2 violation: {p2_m2:.1f}%  (min {min2_m2:.1f}°)")
    ax2.text(0.01, 0.97, note2, transform=ax2.transAxes,
             fontsize=8, color=txt, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CBD5E1", alpha=0.9))

    # shared x-range
    for ax in axes:
        ax.set_xlim(lo, hi)

    # Legend for maneuver lines
    from matplotlib.lines import Line2D
    man_legend = [
        Line2D([0], [0], color="#3B82F6", lw=0.8, ls=":", label="RCS firing"),
        Line2D([0], [0], color="#F97316", lw=0.9, ls="--", label="PPS firing"),
    ]
    axes[-1].legend(handles=man_legend, fontsize=7, loc="lower right",
                    framealpha=0.8, edgecolor="#E2E8F0")

    fig.suptitle(
        f"Thermal Constraints — {phase_label}\n"
        "Camera sun exclusion (+Z, 30° TBC)  |  STR blinding (+X TBC, 35° TBC)",
        fontsize=13, color=txt, fontweight="bold", y=0.99, linespacing=1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Thermal constraints: Mode 1 vs Mode 2")
    p.add_argument("--mode1-dir", default="end1_target_sunOpt",
                   help="Mode 1 (Target+Sun) mat directory")
    p.add_argument("--mode2-dir", default="end1_Nadir_sunOpt",
                   help="Mode 2 (Nadir+Sun) mat directory")
    p.add_argument("--output-far",   default="thermal_far.png")
    p.add_argument("--output-close", default="thermal_close.png")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("  Thermal Constraints Analysis")
    print(f"  Camera exclusion: {CAMERA_EXCL_DEG}°  |  STR exclusion: {STR_EXCL_DEG}°")
    print(f"{'='*60}")

    print("\nLoading Mode 1 (Target+Sun)...")
    m1 = load_mode(args.mode1_dir)
    print(f"  {len(m1['days'])} points, {m1['days'][-1]:.1f} days")

    print("Loading Mode 2 (Nadir+Sun)...")
    m2 = load_mode(args.mode2_dir)
    print(f"  {len(m2['days'])} points, {m2['days'][-1]:.1f} days")

    # Find phase split day from along-track data
    close_idx = np.where(m1["dlambda"] > CLOSE_KM)[0]
    phase_day = m1["days"][close_idx[0]] if close_idx.size > 0 else m1["days"][-1] * 0.6

    days_max  = m1["days"][-1]
    far_range   = (0.0, phase_day)
    close_range = (phase_day, days_max)

    print(f"\n  Phase split: far range [0, {phase_day:.1f} d]  |  "
          f"close range [{phase_day:.1f}, {days_max:.1f} d]")

    # Print summary statistics
    for label, m, x_range in [("Far range   — Mode 1", m1, far_range),
                               ("Far range   — Mode 2", m2, far_range),
                               ("Close range — Mode 1", m1, close_range),
                               ("Close range — Mode 2", m2, close_range)]:
        mask = (m["days"] >= x_range[0]) & (m["days"] <= x_range[1])
        cam = m["sun_deg"][mask, CAM_AXIS]
        st  = m["sun_deg"][mask, STR_AXIS]
        pc_cam, _ = _violation_stats(cam, CAMERA_EXCL_DEG)
        pc_str, _ = _violation_stats(st, STR_EXCL_DEG)
        print(f"  {label}: camera viol {pc_cam:5.1f}%  STR viol {pc_str:5.1f}%")

    print("\nGenerating figures...")
    generate_figure(m1, m2, far_range,   "Far Range (−60 to −5 km)",
                    args.output_far, args.dpi)
    generate_figure(m1, m2, close_range, "Close Range (−5 to +1 km)",
                    args.output_close, args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
