#!/usr/bin/env python3
"""
mode3_slew_analysis.py
======================
Compares the slew cost of transitioning into Mode 3 (Earth+Target) from
Mode 1 (Target+Sun) vs Mode 2 (Nadir+Sun).

Given the spacecraft geometry:
    - Camera boresight:  +Z body axis
    - S-band antenna:    +Y body axis (also on -Y)

Mode 3 attitude is constructed as:
    Primary:   +Z_body = target direction (camera, exact)
    Secondary: +Y_body = closest to Earth direction (antenna, best-effort)

This script computes:
    1. The Mode 3 attitude at each timestep (from ECI positions)
    2. The slew angle from Mode 1 into Mode 3
    3. The slew angle from Mode 2 into Mode 3
    4. The residual antenna pointing error in Mode 3
    5. Per-day window summary

Input:  Two sets of AOCS data package CSVs:
        --mode1-csvs: Target+Sun pointing CSVs
        --mode2-csvs: Nadir+Sun pointing CSVs

Output: 4-panel plot (PNG) + summary to stdout

Usage:
    python mode3_slew_analysis.py \\
        --mode1-csvs RDV_target.csv INS_target.csv \\
        --mode2-csvs RDV_nadir.csv INS_nadir.csv \\
        --output mode3_slew_analysis.png
"""

import argparse
import csv
import os
import sys
import numpy as np
from datetime import datetime


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mode 3 slew analysis: compare transition cost from Mode 1 vs Mode 2"
    )
    parser.add_argument("--mode1-csvs", nargs="+", required=True,
                        help="Target+Sun pointing CSV file(s) in chronological order")
    parser.add_argument("--mode2-csvs", nargs="+", required=True,
                        help="Nadir+Sun pointing CSV file(s) in chronological order")
    parser.add_argument("--output", default="mode3_slew_analysis.png",
                        help="Output plot filename (default: mode3_slew_analysis.png)")
    parser.add_argument("--dpi", type=int, default=180,
                        help="Plot resolution (default: 180)")
    parser.add_argument("--window-hours", type=float, default=6.0,
                        help="Mode 3 window interval in hours (default: 6)")
    parser.add_argument("--window-duration-min", type=float, default=25.0,
                        help="Mode 3 window duration in minutes (default: 25)")
    return parser.parse_args()


# =============================================================================
# Data reading
# =============================================================================

def parse_timestamp(raw):
    """Parse ISO timestamp, truncating sub-microsecond digits."""
    s = raw.strip()
    if "." in s:
        base, frac = s.split(".")
        s = f"{base}.{frac[:6]}"
    return datetime.fromisoformat(s)


def read_quaternions(filepaths):
    """Read timestamps and quaternions from CSV files."""
    days, quats = [], []
    t0 = None
    for fpath in filepaths:
        with open(fpath, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                t = parse_timestamp(row[0])
                if t0 is None:
                    t0 = t
                days.append((t - t0).total_seconds() / 86400.0)
                quats.append([float(row[13]), float(row[14]),
                              float(row[15]), float(row[16])])
    return np.array(days), np.array(quats), t0


def read_positions(filepaths, t0):
    """Read satellite and target ECI positions, along-track separation."""
    earth_dirs, tgt_dirs, dlambda, days = [], [], [], []
    for fpath in filepaths:
        with open(fpath, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                t = parse_timestamp(row[0])
                days.append((t - t0).total_seconds() / 86400.0)

                sat_pos = np.array([float(row[1]), float(row[2]), float(row[3])])
                tgt_pos = np.array([float(row[7]), float(row[8]), float(row[9])])

                s2e = -sat_pos
                s2e /= np.linalg.norm(s2e)
                s2t = tgt_pos - sat_pos
                s2t /= np.linalg.norm(s2t)

                earth_dirs.append(s2e)
                tgt_dirs.append(s2t)
                dlambda.append(float(row[27]) / 1000.0)

    return (np.array(days), np.array(earth_dirs),
            np.array(tgt_dirs), np.array(dlambda))


# =============================================================================
# Attitude computation
# =============================================================================

def quat_to_dcm(q):
    """Quaternion (scalar-first: a, i, j, k) to DCM (ECI -> body)."""
    a, b, c, d = q
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c + a*d),         2*(b*d - a*c)],
        [2*(b*c - a*d),         a*a - b*b + c*c - d*d,  2*(c*d + a*b)],
        [2*(b*d + a*c),         2*(c*d - a*b),           a*a - b*b - c*c + d*d]
    ])


def compute_mode3_dcms(earth_dirs, tgt_dirs):
    """
    Compute Mode 3 attitude DCMs.
    Primary:   +Z_body = target direction
    Secondary: +Y_body = closest to Earth direction
    """
    dcms = []
    for i in range(len(earth_dirs)):
        z = tgt_dirs[i]
        earth = earth_dirs[i]

        # Project Earth onto plane perpendicular to Z
        y = earth - np.dot(earth, z) * z
        yn = np.linalg.norm(y)

        if yn < 1e-10:
            # Degenerate case
            if abs(z[0]) < 0.9:
                y = np.cross(z, [1, 0, 0])
            else:
                y = np.cross(z, [0, 1, 0])
            y = y / np.linalg.norm(y)
        else:
            y = y / yn

        x = np.cross(y, z)
        dcms.append(np.array([x, y, z]))
    return dcms


def dcm_angle(dcm1, dcm2):
    """Eigen-angle between two DCMs in degrees."""
    R = dcm1 @ dcm2.T
    trace = np.clip(np.trace(R), -1.0, 3.0)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def compute_slew_angles(quats, mode3_dcms):
    """Compute slew angle from a mode's attitude to Mode 3 at each timestep."""
    n = len(quats)
    slew = np.zeros(n)
    for i in range(n):
        dcm = quat_to_dcm(quats[i])
        slew[i] = dcm_angle(dcm, mode3_dcms[i])
    return slew


def compute_antenna_error(earth_dirs, tgt_dirs):
    """Compute antenna pointing error in Mode 3 (= |90° - Earth-Target sep|)."""
    ang = np.array([
        np.degrees(np.arccos(np.clip(np.dot(earth_dirs[i], tgt_dirs[i]), -1, 1)))
        for i in range(len(earth_dirs))
    ])
    return np.abs(90.0 - ang), ang


# =============================================================================
# Summary output
# =============================================================================

def print_summary(days, slew_m1, slew_m2, ant_err, window_hours, window_dur_min):
    """Print per-day window summary."""
    window_interval = window_hours / 24.0
    window_dur_days = window_dur_min / 1440.0

    print(f"\n{'='*80}")
    print(f"  Per-day window summary ({window_dur_min:.0f} min every {window_hours:.0f} h)")
    print(f"{'='*80}")
    print(f"  {'Day':>4}  {'Mode1→3 slew [deg]':>28}  {'Mode2→3 slew [deg]':>28}  {'Ant err':>8}")
    print(f"  {'':>4}  {'(Target+Sun → Earth+Tgt)':>28}  {'(Nadir+Sun → Earth+Tgt)':>28}  {'worst':>8}")

    for day_int in range(int(days[-1]) + 1):
        m1_errs, m2_errs, ant_errs = [], [], []
        for h_idx in range(int(24 / window_hours)):
            d_start = day_int + h_idx * window_interval
            mask = (days >= d_start) & (days < d_start + window_dur_days)
            if mask.any():
                m1_errs.append(slew_m1[mask].mean())
                m2_errs.append(slew_m2[mask].mean())
                ant_errs.append(ant_err[mask].mean())
        if m1_errs:
            m1_str = ", ".join([f"{e:.0f}°" for e in m1_errs])
            m2_str = ", ".join([f"{e:.0f}°" for e in m2_errs])
            worst_ant = max(ant_errs)
            flag = " ⚠" if worst_ant > 20 else ""
            print(f"  {day_int:4d}  [{m1_str:>26s}]  [{m2_str:>26s}]  {worst_ant:6.1f}°{flag}")

    # Overall
    print(f"\n  Overall statistics:")
    print(f"    Mode 1 → Mode 3 slew: mean {slew_m1.mean():.1f}°, "
          f"range {slew_m1.min():.1f}°–{slew_m1.max():.1f}°")
    print(f"    Mode 2 → Mode 3 slew: mean {slew_m2.mean():.1f}°, "
          f"range {slew_m2.min():.1f}°–{slew_m2.max():.1f}°")
    print(f"    Mode 1 is cheaper transition {np.sum(slew_m1 < slew_m2)/len(days)*100:.1f}% "
          f"of the time")
    print(f"\n  Antenna error in Mode 3:")
    for bw in [5, 10, 15, 20, 30, 45]:
        pct = np.sum(ant_err <= bw) / len(ant_err) * 100
        print(f"    ≤{bw:2d}°: {pct:5.1f}% of timeline")


# =============================================================================
# Plotting
# =============================================================================

def generate_plot(days, ang_eci, slew_m1, slew_m2, ant_err, dlambda,
                  filepath, window_hours, window_dur_min, dpi):
    """Generate the 4-panel slew analysis plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    txt = "#334155"
    bg = "#FAFBFC"
    grid_c = "#E2E8F0"
    window_interval = window_hours / 24.0
    window_dur_days = window_dur_min / 1440.0

    fig, axes = plt.subplots(4, 1, figsize=(16, 14),
                             gridspec_kw={"height_ratios": [1, 1.2, 1, 0.7],
                                          "hspace": 0.10})
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor(bg)
        ax.grid(True, alpha=0.5, color=grid_c, linewidth=0.5)
        ax.tick_params(colors=txt, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#CBD5E1")
            sp.set_linewidth(0.5)

    # --- Panel 1: Earth-Target angular separation ---
    ax1 = axes[0]
    ax1.plot(days, ang_eci, color="#7C3AED", linewidth=0.6, alpha=0.9)
    ax1.axhline(90, color="#059669", linewidth=1.5, linestyle="-", alpha=0.7,
                label="90° (camera ⊥ antenna)")
    ax1.fill_between(days, 90, ang_eci, where=ang_eci > 90,
                     alpha=0.10, color="#DC2626")
    ax1.fill_between(days, 90, ang_eci, where=ang_eci < 90,
                     alpha=0.10, color="#2563EB")
    ax1.set_ylabel("Earth–Target\nangular sep. [deg]",
                    fontsize=10, color=txt, fontweight="medium")
    ax1.set_ylim(0, 185)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.8, edgecolor="#E2E8F0")
    ax1.text(0.01, 0.92,
             "Geometric angle between Earth and target (attitude-independent)",
             transform=ax1.transAxes, fontsize=8, color="#64748B",
             fontstyle="italic", va="top")

    # --- Panel 2: Slew angles ---
    ax2 = axes[1]
    ax2.plot(days, slew_m1, color="#2563EB", linewidth=0.6, alpha=0.9,
             label="From Mode 1 (Target+Sun)")
    ax2.plot(days, slew_m2, color="#DC2626", linewidth=0.6, alpha=0.9,
             label="From Mode 2 (Nadir+Sun)")
    ax2.fill_between(days, slew_m1, alpha=0.06, color="#2563EB")
    ax2.fill_between(days, slew_m2, alpha=0.06, color="#DC2626")
    ax2.set_ylabel("Slew angle\nto Mode 3 [deg]",
                    fontsize=10, color=txt, fontweight="medium")
    ax2.set_ylim(0, 185)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#E2E8F0")
    ax2.text(0.01, 0.92,
             "Rotation required to transition from current mode into Mode 3 (Earth+Target)",
             transform=ax2.transAxes, fontsize=8, color="#64748B",
             fontstyle="italic", va="top")

    # --- Panel 3: Antenna pointing error in Mode 3 ---
    ax3 = axes[2]
    ax3.plot(days, ant_err, color="#E11D48", linewidth=0.6, alpha=0.9)
    ax3.fill_between(days, 0, ant_err, alpha=0.08, color="#E11D48")

    for bw, c, ls in [(10, "#059669", "--"),
                       (20, "#F59E0B", "--"),
                       (45, "#94A3B8", ":")]:
        ax3.axhline(bw, color=c, linewidth=1.0, linestyle=ls, alpha=0.7,
                    label=f"{bw}° threshold")

    for d_start in np.arange(0, days[-1], window_interval):
        ax3.axvspan(d_start, d_start + window_dur_days,
                    alpha=0.08, color="#F59E0B", zorder=0)

    ax3.set_ylabel("S-band antenna\nerror in Mode 3 [deg]",
                    fontsize=10, color=txt, fontweight="medium")
    ax3.set_ylim(0, 95)
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.8, edgecolor="#E2E8F0")
    ax3.text(0.01, 0.92,
             "Residual antenna error when in Mode 3  |  "
             "Yellow = 25-min windows every 6h",
             transform=ax3.transAxes, fontsize=8, color="#64748B",
             fontstyle="italic", va="top")

    # --- Panel 4: Along-track separation ---
    ax4 = axes[3]
    ax4.plot(days, dlambda, color="#1E293B", linewidth=0.8)
    ax4.fill_between(days, dlambda, alpha=0.06, color="#1E293B")
    ax4.set_ylabel("a·δλ [km]", fontsize=10, color=txt, fontweight="medium")
    ax4.set_xlabel("Mission elapsed time [days]",
                    fontsize=10, color=txt, fontweight="medium")
    ax4.axhline(0, color="#DC2626", linewidth=0.5, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Mode 3 (Earth+Target) — Slew Analysis & Antenna Feasibility\n"
        "Which mode is cheaper to transition from?  |  "
        "Camera +Z  |  S-band antenna +Y",
        fontsize=13, color=txt, fontweight="bold", y=0.98, linespacing=1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"\n  Plot saved: {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    for fpath in args.mode1_csvs + args.mode2_csvs:
        if not os.path.isfile(fpath):
            print(f"ERROR: File not found: {fpath}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Mode 3 Slew Analysis")
    print(f"  Camera: +Z  |  Antenna: +Y/−Y (S-band)")
    print(f"{'='*60}")

    # Read Mode 1 quaternions
    print("\nReading Mode 1 (Target+Sun)...")
    days_m1, quats_m1, t0 = read_quaternions(args.mode1_csvs)
    print(f"  {len(days_m1)} points")

    # Read Mode 2 quaternions
    print("Reading Mode 2 (Nadir+Sun)...")
    days_m2, quats_m2, _ = read_quaternions(args.mode2_csvs)
    print(f"  {len(days_m2)} points")

    # Read ECI positions (same orbit for both — use Mode 2 files)
    print("Reading orbital geometry...")
    days_pos, earth_dirs, tgt_dirs, dlambda = read_positions(args.mode2_csvs, t0)

    # Compute Mode 3 attitude
    print("Computing Mode 3 attitude at each timestep...")
    mode3_dcms = compute_mode3_dcms(earth_dirs, tgt_dirs)

    # Compute slew angles
    print("Computing slew angles...")
    slew_m1 = compute_slew_angles(quats_m1, mode3_dcms)
    slew_m2 = compute_slew_angles(quats_m2, mode3_dcms)

    # Compute antenna error and Earth-Target angle
    ant_err, ang_eci = compute_antenna_error(earth_dirs, tgt_dirs)

    # Print summary
    print_summary(days_pos, slew_m1, slew_m2, ant_err,
                  args.window_hours, args.window_duration_min)

    # Generate plot
    print("\nGenerating plot...")
    generate_plot(days_pos, ang_eci, slew_m1, slew_m2, ant_err, dlambda,
                  args.output, args.window_hours, args.window_duration_min,
                  args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
