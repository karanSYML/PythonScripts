#!/usr/bin/env python3
"""
mode3_feasibility.py
====================
Analyzes Mode 3 (Earth+Target) pointing feasibility for the Endurance
FR RDV mission.

Given the spacecraft geometry:
    - Camera boresight:  +Z body axis
    - S-band antenna:    +Y body axis (also on -Y)

This script computes:
    1. The geometric Earth-Target angular separation (attitude-independent)
    2. The residual antenna pointing error when camera is locked on target
    3. Feasibility at each 6-hour ConOps window

Input:  One or more AOCS data package CSVs (same format as dataPackage4Thermal)
Output: Feasibility plot (PNG) + per-window summary to stdout

Usage:
    python mode3_feasibility.py --csvs RDV.csv INS.csv --output mode3_feasibility.png
"""

import argparse
import csv
import os
import sys
import numpy as np
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mode 3 (Earth+Target) pointing feasibility analysis"
    )
    parser.add_argument("--csvs", nargs="+", required=True,
                        help="Input CSV file(s) in chronological order")
    parser.add_argument("--output", default="mode3_feasibility.png",
                        help="Output plot filename (default: mode3_feasibility.png)")
    parser.add_argument("--dpi", type=int, default=180,
                        help="Plot resolution (default: 180)")
    parser.add_argument("--window-hours", type=float, default=6.0,
                        help="Mode 3 window interval in hours (default: 6)")
    parser.add_argument("--window-duration-min", type=float, default=25.0,
                        help="Mode 3 window duration in minutes (default: 25)")
    return parser.parse_args()


def read_geometry(filepaths):
    """Extract Earth/target direction vectors and along-track separation."""
    days, ang_eci, dlambda = [], [], []
    earth_dirs, tgt_dirs = [], []
    t0 = None

    for fpath in filepaths:
        with open(fpath, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                t_str = row[0].strip()
                if "." in t_str:
                    base, frac = t_str.split(".")
                    t_str = f"{base}.{frac[:6]}"
                t = datetime.fromisoformat(t_str)
                if t0 is None:
                    t0 = t
                days.append((t - t0).total_seconds() / 86400.0)

                # Satellite and target positions in ECI
                sat_pos = np.array([float(row[1]), float(row[2]), float(row[3])])
                tgt_pos = np.array([float(row[7]), float(row[8]), float(row[9])])

                # Unit vectors from spacecraft to Earth and target
                sat2earth = -sat_pos
                sat2earth_uv = sat2earth / np.linalg.norm(sat2earth)
                sat2tgt = tgt_pos - sat_pos
                sat2tgt_uv = sat2tgt / np.linalg.norm(sat2tgt)

                # Angular separation
                dot = np.dot(sat2earth_uv, sat2tgt_uv)
                ang_eci.append(np.degrees(np.arccos(np.clip(dot, -1, 1))))
                dlambda.append(float(row[27]) / 1000.0)  # m -> km
                earth_dirs.append(sat2earth_uv)
                tgt_dirs.append(sat2tgt_uv)

    return (np.array(days), np.array(ang_eci), np.array(dlambda),
            np.array(earth_dirs), np.array(tgt_dirs))


def compute_antenna_error(earth_dirs, tgt_dirs):
    """
    Compute antenna pointing error for Mode 3.

    Attitude strategy:
        Primary:   +Z_body = target direction (camera, exact)
        Secondary: +Y_body = as close to Earth as possible (antenna)

    The residual antenna error = |90° - Earth-Target angular separation|.
    This is because camera and antenna are fixed at 90° apart on the body.
    """
    n = len(earth_dirs)
    err_yp = np.zeros(n)
    err_ym = np.zeros(n)

    for i in range(n):
        z_body = tgt_dirs[i]  # +Z locked on target
        earth = earth_dirs[i]

        # Project Earth direction onto plane perpendicular to Z_body
        earth_proj = earth - np.dot(earth, z_body) * z_body
        proj_norm = np.linalg.norm(earth_proj)

        if proj_norm < 1e-10:
            # Degenerate: Earth exactly along camera boresight
            err_yp[i] = 90.0
            err_ym[i] = 90.0
            continue

        y_body = earth_proj / proj_norm

        # Pointing error for Y+ and Y- antennas
        err_yp[i] = np.degrees(np.arccos(np.clip(np.dot(y_body, earth), -1, 1)))
        err_ym[i] = np.degrees(np.arccos(np.clip(np.dot(-y_body, earth), -1, 1)))

    err_best = np.minimum(err_yp, err_ym)
    return err_yp, err_ym, err_best


def print_window_summary(days, antenna_err, window_hours, window_dur_min):
    """Print per-day feasibility summary for ConOps windows."""
    window_dur_days = window_dur_min / 1440.0
    window_interval = window_hours / 24.0

    print(f"\n{'='*72}")
    print(f"  Mode 3 window summary ({window_dur_min:.0f} min every {window_hours:.0f} h)")
    print(f"{'='*72}")
    print(f"  {'Day':>4}  {'Window errors (Y+ antenna)':>40}  {'Worst':>8}")
    print(f"  {'':>4}  {'':>40}  {'':>8}")

    for day_int in range(int(days[-1]) + 1):
        errors = []
        for h_idx in range(int(24 / window_hours)):
            d_start = day_int + h_idx * window_interval
            mask = (days >= d_start) & (days < d_start + window_dur_days)
            if mask.any():
                errors.append(antenna_err[mask].mean())
        if errors:
            errs_str = ", ".join([f"{e:.1f}°" for e in errors])
            worst = max(errors)
            flag = " ⚠" if worst > 20 else "  " if worst > 10 else ""
            print(f"  {day_int:4d}  [{errs_str:>40s}]  {worst:6.1f}°{flag}")

    # Overall feasibility
    print(f"\n  Feasibility thresholds:")
    for bw in [5, 10, 15, 20, 30, 45]:
        pct = np.sum(antenna_err <= bw) / len(antenna_err) * 100
        print(f"    ≤{bw:2d}° error: {pct:5.1f}% of timeline")


def generate_plot(days, ang_eci, antenna_err_yp, dlambda, filepath,
                  window_hours, window_dur_min, dpi):
    """Generate the 3-panel Mode 3 feasibility plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    txt = "#334155"
    bg = "#FAFBFC"
    grid_c = "#E2E8F0"
    window_dur_days = window_dur_min / 1440.0
    window_interval = window_hours / 24.0

    fig, axes = plt.subplots(3, 1, figsize=(16, 11),
                             gridspec_kw={"height_ratios": [1.2, 1.2, 0.7],
                                          "hspace": 0.12})
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor(bg)
        ax.grid(True, alpha=0.5, color=grid_c, linewidth=0.5)
        ax.tick_params(colors=txt, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#CBD5E1")
            sp.set_linewidth(0.5)

    # Panel 1: Earth-Target angular separation
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

    # Panel 2: Antenna pointing error
    ax2 = axes[1]
    ax2.plot(days, antenna_err_yp, color="#E11D48", linewidth=0.6, alpha=0.9)
    ax2.fill_between(days, 0, antenna_err_yp, alpha=0.08, color="#E11D48")

    for bw, c, ls in [(10, "#059669", "--"),
                       (20, "#F59E0B", "--"),
                       (45, "#94A3B8", ":")]:
        ax2.axhline(bw, color=c, linewidth=1.0, linestyle=ls, alpha=0.7,
                    label=f"{bw}° threshold")

    # Mark ConOps windows
    for d_start in np.arange(0, days[-1], window_interval):
        ax2.axvspan(d_start, d_start + window_dur_days,
                    alpha=0.08, color="#F59E0B", zorder=0)

    ax2.set_ylabel("S-band antenna\npointing error [deg]",
                    fontsize=10, color=txt, fontweight="medium")
    ax2.set_ylim(0, 95)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.8, edgecolor="#E2E8F0")
    ax2.text(0.01, 0.92,
             "Residual antenna error when camera (+Z) is pointed exactly at target\n"
             "Using Y+ antenna; error = |90° − Earth-Target separation|",
             transform=ax2.transAxes, fontsize=8, color="#64748B",
             fontstyle="italic", va="top", linespacing=1.4)

    # Panel 3: Along-track separation
    ax3 = axes[2]
    ax3.plot(days, dlambda, color="#1E293B", linewidth=0.8)
    ax3.fill_between(days, dlambda, alpha=0.06, color="#1E293B")
    ax3.set_ylabel("a·δλ [km]", fontsize=10, color=txt, fontweight="medium")
    ax3.set_xlabel("Mission elapsed time [days]", fontsize=10, color=txt,
                    fontweight="medium")
    ax3.axhline(0, color="#DC2626", linewidth=0.5, linestyle="--", alpha=0.5)

    fig.suptitle(
        "(Earth+Target) — Antenna Pointing Feasibility\n"
        "Camera (+Z) locked on target  |  S-band antenna (+Y)  |"
        "  Can we reach Earth?",
        fontsize=13, color=txt, fontweight="bold", y=0.98, linespacing=1.4
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"\n  Plot saved: {filepath}")


def main():
    args = parse_args()

    for fpath in args.csvs:
        if not os.path.isfile(fpath):
            print(f"ERROR: File not found: {fpath}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Feasibility Analysis")
    print(f"  Camera: +Z  |  Antenna: +Y/−Y (S-band)")
    print(f"{'='*60}")

    # Read geometry
    print("\nReading CSV files...")
    days, ang_eci, dlambda, earth_dirs, tgt_dirs = read_geometry(args.csvs)
    print(f"  {len(days)} points, {days[-1]:.1f} days")
    print(f"  Earth-Target separation: {ang_eci.min():.1f}° – {ang_eci.max():.1f}° "
          f"(mean {ang_eci.mean():.1f}°)")

    # Compute pointing errors
    print("\nComputing Mode 3 pointing errors...")
    err_yp, err_ym, err_best = compute_antenna_error(earth_dirs, tgt_dirs)
    print(f"  Y+ antenna error: {err_yp.min():.1f}° – {err_yp.max():.1f}° "
          f"(mean {err_yp.mean():.1f}°)")
    print(f"  Best of Y+/Y-:    {err_best.min():.1f}° – {err_best.max():.1f}° "
          f"(mean {err_best.mean():.1f}°)")

    # Window summary
    print_window_summary(days, err_yp, args.window_hours, args.window_duration_min)

    # Plot
    print("\nGenerating plot...")
    generate_plot(days, ang_eci, err_yp, dlambda, args.output,
                  args.window_hours, args.window_duration_min, args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
