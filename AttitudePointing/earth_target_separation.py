#!/usr/bin/env python3
"""
earth_target_separation.py
==========================
Computes and plots the geometric angular separation between the Earth
and rendezvous target as seen from the spacecraft, using ECI position
vectors from the AOCS data package.

This angle is attitude-independent — it depends only on the orbital
geometry (satellite position, target position, Earth at origin).

Input:  One or more AOCS data package CSVs (chronological order)
Output: 3-panel plot (PNG):
          Panel 1: Earth-Target angular separation vs time
          Panel 2: Angular separation vs along-track separation (scatter)
          Panel 3: Along-track separation vs time

Usage:
    python earth_target_separation.py \\
        --csvs RDV.csv INS.csv \\
        --output fig1_earth_target_separation.png
"""

import argparse
import csv
import os
import sys
import numpy as np
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Earth-Target angular separation analysis"
    )
    parser.add_argument("--csvs", nargs="+", required=True,
                        help="Input CSV file(s) in chronological order")
    parser.add_argument("--output", default="fig1_earth_target_separation.png",
                        help="Output plot filename (default: fig1_earth_target_separation.png)")
    parser.add_argument("--dpi", type=int, default=180,
                        help="Plot resolution (default: 180)")
    parser.add_argument("--window-hours", type=float, default=6.0,
                        help="Mode 3 window interval in hours (default: 6)")
    parser.add_argument("--window-duration-min", type=float, default=25.0,
                        help="Mode 3 window duration in minutes (default: 25)")
    return parser.parse_args()


def parse_timestamp(raw):
    """Parse ISO timestamp, truncating sub-microsecond digits."""
    s = raw.strip()
    if "." in s:
        base, frac = s.split(".")
        s = f"{base}.{frac[:6]}"
    return datetime.fromisoformat(s)


def read_geometry(filepaths):
    """
    Read satellite and target ECI positions from CSV files.
    Compute:
      - Earth-Target angular separation (attitude-independent)
      - Along-track separation (a·delta-lambda)
      - Mission elapsed time in days
    """
    days, ang_sep, dlambda = [], [], []
    t0 = None

    for fpath in filepaths:
        count = 0
        with open(fpath, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                t = parse_timestamp(row[0])
                if t0 is None:
                    t0 = t
                days.append((t - t0).total_seconds() / 86400.0)

                # Satellite and target positions in ECI (meters)
                sat_pos = np.array([float(row[1]), float(row[2]), float(row[3])])
                tgt_pos = np.array([float(row[7]), float(row[8]), float(row[9])])

                # Unit vector: spacecraft -> Earth (Earth is at ECI origin)
                sat2earth = -sat_pos
                sat2earth_uv = sat2earth / np.linalg.norm(sat2earth)

                # Unit vector: spacecraft -> target
                sat2tgt = tgt_pos - sat_pos
                sat2tgt_uv = sat2tgt / np.linalg.norm(sat2tgt)

                # Angular separation
                dot = np.dot(sat2earth_uv, sat2tgt_uv)
                ang = np.degrees(np.arccos(np.clip(dot, -1, 1)))
                ang_sep.append(ang)

                # Along-track separation (m -> km)
                dlambda.append(float(row[27]) / 1000.0)
                count += 1

        print(f"  {os.path.basename(fpath)}: {count} records")

    return np.array(days), np.array(ang_sep), np.array(dlambda)


def generate_plot(days, ang_sep, dlambda, filepath, args):
    """Generate 3-panel Earth-Target separation plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    txt = "#334155"
    bg = "#FAFBFC"
    grid_c = "#E2E8F0"
    window_interval = args.window_hours / 24.0
    window_dur_days = args.window_duration_min / 1440.0

    fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [1.2, 1, 0.7],
                                          "hspace": 0.12})
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor(bg)
        ax.grid(True, alpha=0.5, color=grid_c, linewidth=0.5)
        ax.tick_params(colors=txt, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#CBD5E1")
            sp.set_linewidth(0.5)

    # --- Panel 1: Angular separation vs time ---
    ax1 = axes[0]
    ax1.plot(days, ang_sep, color="#7C3AED", linewidth=0.6, alpha=0.9)
    ax1.fill_between(days, 0, ang_sep, alpha=0.08, color="#7C3AED")
    ax1.set_ylabel("Earth–Target\nangular sep. [deg]",
                    fontsize=10, color=txt, fontweight="medium")
    ax1.set_ylim(0, 185)
    ax1.axhline(90, color="#94A3B8", linewidth=0.5, linestyle="--", alpha=0.5)

    # Mark 6-hour Mode 3 windows
    for d_start in np.arange(0, days[-1], window_interval):
        ax1.axvspan(d_start, d_start + window_dur_days,
                    alpha=0.12, color="#F59E0B", zorder=0)

    ax1.text(0.01, 0.92,
             "Geometric angle between Earth and target as seen from spacecraft (ECI)",
             transform=ax1.transAxes, fontsize=8, color="#64748B",
             fontstyle="italic", va="top")
    ax1.text(0.99, 0.92,
             f"Yellow bands = {args.window_duration_min:.0f}-min Mode 3 windows "
             f"(every {args.window_hours:.0f}h)",
             transform=ax1.transAxes, fontsize=8, color="#B45309",
             fontstyle="italic", va="top", ha="right")

    # --- Panel 2: Angle vs along-track separation (scatter) ---
    ax2 = axes[1]
    sc = ax2.scatter(dlambda, ang_sep, c=days, cmap="viridis",
                     s=1.5, alpha=0.5, rasterized=True)
    ax2.set_ylabel("Earth–Target\nangular sep. [deg]",
                    fontsize=10, color=txt, fontweight="medium")
    ax2.set_xlabel("Along-track separation a·δλ [km]",
                    fontsize=10, color=txt, fontweight="medium")
    ax2.axhline(90, color="#94A3B8", linewidth=0.5, linestyle="--", alpha=0.5)
    cb = plt.colorbar(sc, ax=ax2, label="Mission day", shrink=0.8, aspect=30)
    cb.ax.tick_params(labelsize=8)
    ax2.text(0.01, 0.92,
             "Angular separation vs range (color = time progression)",
             transform=ax2.transAxes, fontsize=8, color="#64748B",
             fontstyle="italic", va="top")

    # --- Panel 3: Along-track separation vs time ---
    ax3 = axes[2]
    ax3.plot(days, dlambda, color="#1E293B", linewidth=0.8)
    ax3.fill_between(days, dlambda, alpha=0.06, color="#1E293B")
    ax3.set_ylabel("a·δλ [km]", fontsize=10, color=txt, fontweight="medium")
    ax3.set_xlabel("Mission elapsed time [days]",
                    fontsize=10, color=txt, fontweight="medium")
    ax3.axhline(0, color="#DC2626", linewidth=0.5, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Earth-Target Angular Separation\n"
        "How far apart are the Earth and target pointing directions?",
        fontsize=13, color=txt, fontweight="bold", y=0.98, linespacing=1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(filepath, dpi=args.dpi, bbox_inches="tight",
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
    print(f"  Earth–Target Angular Separation Analysis")
    print(f"{'='*60}")

    print("\nReading CSV files...")
    days, ang_sep, dlambda = read_geometry(args.csvs)
    print(f"  Total: {len(days)} points, {days[-1]:.1f} days")
    print(f"  Angular separation: {ang_sep.min():.1f}° – {ang_sep.max():.1f}° "
          f"(mean {ang_sep.mean():.1f}°)")
    print(f"  Along-track range:  {dlambda.min():.1f} km – {dlambda.max():.1f} km")

    print("\nGenerating plot...")
    generate_plot(days, ang_sep, dlambda, args.output, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
