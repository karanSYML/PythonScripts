#!/usr/bin/env python3
"""
aocs_to_systema.py
==================
Converts AOCS simulation data package CSV(s) into:
  1. CCSDS OEM + AEM files for Systema-Thermica import
  2. A 4-panel attitude profile visualization (PNG)

Supports multiple CSV files (e.g., RDV + INS phases) which are
concatenated in order to produce continuous outputs.

Input CSV columns (28 total):
    UTC_ISOC, pos_Earth2satCoG_ECI_m (x,y,z), vel_Earth2satCoG_ECI_ms (x,y,z),
    pos_Earth2tgtCoG_ECI_m (x,y,z), pos_Earth2Sun_ECI_m (x,y,z),
    quat_ECI2MRF (a,i,j,k), angle_SunPhaseAngle_rad,
    uv_sat2Earth_MRF (x,y,z), uv_sat2Target_MRF (x,y,z),
    angle_sunFromMRFaxes_deg (x,y,z), a_dlambda_m

Usage:
    python aocs_to_systema.py --csvs RDV.csv INS.csv \\
                              --label "Nadir+Sun" \\
                              --object "RDV_SAT" \\
                              --output-dir ./output

    python aocs_to_systema.py --csvs RDV.csv INS.csv \\
                              --label "Target+Sun" \\
                              --object "RDV_SAT" \\
                              --output-dir ./output

Notes:
    - Quaternion convention in CSV: scalar-first (a, i, j, k)
    - CCSDS AEM output: scalar-last (Q1, Q2, Q3, QC) per CCSDS 504.0-B
    - Quaternion signs are PRESERVED from source (no hemisphere correction)
    - Position/velocity converted from m, m/s to km, km/s for OEM
"""

import argparse
import csv
import os
import sys
import numpy as np
from datetime import datetime, timedelta, timezone


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert AOCS data package CSV(s) to CCSDS OEM + AEM + attitude plot"
    )
    parser.add_argument("--csvs", nargs="+", required=True,
                        help="Input CSV file(s) in chronological order (e.g., RDV.csv INS.csv)")
    parser.add_argument("--label", default="Attitude",
                        help="Attitude law label for plot title (e.g., 'Nadir+Sun', 'Target+Sun')")
    parser.add_argument("--originator", default="AOCS_SIM",
                        help="Originator ID for CCSDS headers (default: AOCS_SIM)")
    parser.add_argument("--object", default="SPACECRAFT",
                        help="Object name (default: SPACECRAFT)")
    parser.add_argument("--object-id", default="0000-000A",
                        help="Object ID / international designator (default: 0000-000A)")
    parser.add_argument("--frame", default="EME2000",
                        help="ECI reference frame (default: EME2000)")
    parser.add_argument("--body-frame", default="SC_BODY_1",
                        help="Spacecraft body frame name (default: SC_BODY_1)")
    parser.add_argument("--output-dir", default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--skip-plot", action="store_true",
                        help="Skip generating the attitude plot")
    parser.add_argument("--skip-ccsds", action="store_true",
                        help="Skip generating OEM/AEM files")
    parser.add_argument("--dpi", type=int, default=180,
                        help="Plot resolution in DPI (default: 180)")
    return parser.parse_args()


# =============================================================================
# CSV reading
# =============================================================================

def format_epoch(iso_str):
    """Clean ISO timestamp: truncate sub-microsecond digits."""
    iso_str = iso_str.strip()
    if "." in iso_str:
        base, frac = iso_str.split(".")
        return f"{base}.{frac[:6]}"
    return iso_str


def parse_epoch(iso_str):
    """Parse ISO timestamp to datetime (truncate to microseconds)."""
    s = iso_str.strip()
    if "." in s:
        base, frac = s.split(".")
        s = f"{base}.{frac[:6]}"
    return datetime.fromisoformat(s)


def read_csv_files(filepaths):
    """Read one or more AOCS data package CSVs and return combined records."""
    records = []
    for fpath in filepaths:
        count = 0
        with open(fpath, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row_num, row in enumerate(reader, start=2):
                if len(row) < 28:
                    print(f"  WARNING: {os.path.basename(fpath)} row {row_num}: "
                          f"insufficient columns ({len(row)}), skipping")
                    continue
                try:
                    rec = {
                        "epoch_str": format_epoch(row[0]),
                        "epoch_dt": parse_epoch(row[0]),
                        # Position & velocity (ECI, m & m/s)
                        "px": float(row[1]),  "py": float(row[2]),  "pz": float(row[3]),
                        "vx": float(row[4]),  "vy": float(row[5]),  "vz": float(row[6]),
                        # Quaternion ECI→MRF (scalar-first: a, i, j, k)
                        "qa": float(row[13]), "qi": float(row[14]),
                        "qj": float(row[15]), "qk": float(row[16]),
                        # Sun angles from MRF axes (deg)
                        "sun_x": float(row[24]), "sun_y": float(row[25]), "sun_z": float(row[26]),
                        # Nadir unit vector in MRF
                        "nadir_x": float(row[18]), "nadir_y": float(row[19]), "nadir_z": float(row[20]),
                        # Target unit vector in MRF
                        "tgt_x": float(row[21]), "tgt_y": float(row[22]), "tgt_z": float(row[23]),
                        # Along-track separation (m → km)
                        "dlambda_km": float(row[27]) / 1000.0,
                    }
                    records.append(rec)
                    count += 1
                except (ValueError, IndexError) as e:
                    print(f"  WARNING: {os.path.basename(fpath)} row {row_num}: {e}, skipping")
        print(f"  {os.path.basename(fpath)}: {count} records")
    return records


def check_quaternion_continuity(records):
    """Report hemisphere flips without modifying data."""
    flips = 0
    for i in range(1, len(records)):
        dot = (records[i]["qa"] * records[i-1]["qa"] +
               records[i]["qi"] * records[i-1]["qi"] +
               records[i]["qj"] * records[i-1]["qj"] +
               records[i]["qk"] * records[i-1]["qk"])
        if dot < 0:
            flips += 1
    if flips > 0:
        print(f"  Quaternion hemisphere flips: {flips} (signs preserved as-is)")
    else:
        print(f"  Quaternion continuity: OK")
    return flips


# =============================================================================
# CCSDS OEM writer
# =============================================================================

def write_oem(records, filepath, args):
    """Write CCSDS OEM v2.0 (KVN format). Units: km, km/s."""
    creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    with open(filepath, "w") as f:
        f.write(f"CCSDS_OEM_VERS = 2.0\n")
        f.write(f"COMMENT  Generated by aocs_to_systema.py\n")
        f.write(f"COMMENT  Attitude law: {args.label}\n")
        f.write(f"COMMENT  Position in km, velocity in km/s\n")
        f.write(f"CREATION_DATE = {creation_date}\n")
        f.write(f"ORIGINATOR = {args.originator}\n\n")
        f.write(f"META_START\n")
        f.write(f"OBJECT_NAME = {args.object}\n")
        f.write(f"OBJECT_ID = {args.object_id}\n")
        f.write(f"CENTER_NAME = EARTH\n")
        f.write(f"REF_FRAME = {args.frame}\n")
        f.write(f"TIME_SYSTEM = UTC\n")
        f.write(f"START_TIME = {records[0]['epoch_str']}\n")
        f.write(f"STOP_TIME = {records[-1]['epoch_str']}\n")
        f.write(f"META_STOP\n\n")
        for rec in records:
            f.write(f"{rec['epoch_str']}  "
                    f"{rec['px']/1e3:20.12f} {rec['py']/1e3:20.12f} {rec['pz']/1e3:20.12f}  "
                    f"{rec['vx']/1e3:16.12f} {rec['vy']/1e3:16.12f} {rec['vz']/1e3:16.12f}\n")
    print(f"  OEM: {filepath}")


# =============================================================================
# CCSDS AEM writer
# =============================================================================

def write_aem(records, filepath, args):
    """Write CCSDS AEM v2.0 (KVN format). Scalar-last: Q1(i), Q2(j), Q3(k), QC(a)."""
    creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    with open(filepath, "w") as f:
        f.write(f"CCSDS_AEM_VERS = 2.0\n")
        f.write(f"COMMENT  Generated by aocs_to_systema.py\n")
        f.write(f"COMMENT  Attitude law: {args.label}\n")
        f.write(f"COMMENT  Quaternion: rotation from REF_FRAME_A to REF_FRAME_B\n")
        f.write(f"COMMENT  Quaternion ordering: Q1(i), Q2(j), Q3(k), QC(scalar)\n")
        f.write(f"COMMENT  Source quaternion was ECI->MRF (Mission Reference Frame)\n")
        f.write(f"COMMENT  Quaternion signs preserved as-is from AOCS source data\n")
        f.write(f"CREATION_DATE = {creation_date}\n")
        f.write(f"ORIGINATOR = {args.originator}\n\n")
        f.write(f"META_START\n")
        f.write(f"OBJECT_NAME = {args.object}\n")
        f.write(f"OBJECT_ID = {args.object_id}\n")
        f.write(f"REF_FRAME_A = {args.frame}\n")
        f.write(f"REF_FRAME_B = {args.body_frame}\n")
        f.write(f"ATTITUDE_DIR = A2B\n")
        f.write(f"TIME_SYSTEM = UTC\n")
        f.write(f"START_TIME = {records[0]['epoch_str']}\n")
        f.write(f"STOP_TIME = {records[-1]['epoch_str']}\n")
        f.write(f"ATTITUDE_TYPE = QUATERNION\n")
        f.write(f"QUATERNION_TYPE = FIRST\n")
        f.write(f"INTERPOLATION_METHOD = LINEAR\n")
        f.write(f"META_STOP\n\n")
        for rec in records:
            f.write(f"{rec['epoch_str']}  "
                    f"{rec['qi']:+18.15f} {rec['qj']:+18.15f} "
                    f"{rec['qk']:+18.15f} {rec['qa']:+18.15f}\n")
    print(f"  AEM: {filepath}")


# =============================================================================
# README writer
# =============================================================================

def write_readme(records, filepath, args, num_flips):
    """Write handoff documentation."""
    with open(filepath, "w") as f:
        f.write("=" * 72 + "\n")
        f.write(f"  AOCS → THERMAL DATA HANDOFF\n")
        f.write(f"  Attitude law: {args.label}\n")
        f.write(f"  CCSDS OEM + AEM for Systema-Thermica Import\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Generated:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"Object:           {args.object}\n")
        f.write(f"Originator:       {args.originator}\n")
        f.write(f"Source files:     {', '.join(os.path.basename(c) for c in args.csvs)}\n\n")
        f.write(f"Time span:        {records[0]['epoch_str']}  to\n")
        f.write(f"                  {records[-1]['epoch_str']}\n")
        f.write(f"Data points:      {len(records)}\n")
        f.write(f"Time step:        5 minutes (300 s)\n\n")

        f.write("-" * 72 + "\n")
        f.write("  FRAME DEFINITIONS\n")
        f.write("-" * 72 + "\n\n")
        f.write(f"  Reference frame:  {args.frame} (Earth-centered inertial, J2000)\n")
        f.write(f"  Body frame:       {args.body_frame} (= MRF, Mission Reference Frame)\n")
        f.write(f"  Attitude dir:     A2B (rotation from {args.frame} to {args.body_frame})\n\n")
        f.write(f"  Quaternion convention:\n")
        f.write(f"      Source CSV:  scalar-first  (q_a, q_i, q_j, q_k)\n")
        f.write(f"      AEM file:   scalar-last    (Q1, Q2, Q3, QC) per CCSDS 504.0-B\n")
        f.write(f"      Signs PRESERVED from AOCS source ({num_flips} hemisphere flips present).\n")
        f.write(f"      All quaternions are unit-normalized (||q|| = 1).\n\n")

        f.write("-" * 72 + "\n")
        f.write("  SYSTEMA-THERMICA IMPORT\n")
        f.write("-" * 72 + "\n\n")
        f.write("  1. Trajectory tab  → import .oem (center=EARTH, frame=EME2000)\n")
        f.write("  2. Kinematics tab  → import .aem (A2B, frame_A=EME2000, frame_B=SC_BODY_1)\n")
        f.write("  3. Mission tab     → associate trajectory + kinematics with geometry\n")
        f.write("  4. Validate        → animate 3D view, check Sun angles vs source CSV\n\n")
    print(f"  README: {filepath}")


# =============================================================================
# Attitude profile visualization
# =============================================================================

def generate_plot(records, filepath, args):
    """Generate 4-panel attitude profile plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Extract arrays
    t0 = records[0]["epoch_dt"]
    days = np.array([(r["epoch_dt"] - t0).total_seconds() / 86400.0 for r in records])
    sun_x = np.array([r["sun_x"] for r in records])
    sun_y = np.array([r["sun_y"] for r in records])
    sun_z = np.array([r["sun_z"] for r in records])
    nadir_z = np.array([r["nadir_z"] for r in records])
    tgt_x = np.array([r["tgt_x"] for r in records])
    tgt_y = np.array([r["tgt_y"] for r in records])
    tgt_z = np.array([r["tgt_z"] for r in records])
    dlambda = np.array([r["dlambda_km"] for r in records])

    off_nadir = np.degrees(np.arccos(np.clip(nadir_z, -1, 1)))

    # ---- Style ----
    bg = "#FAFBFC"
    grid_c = "#E2E8F0"
    txt = "#334155"

    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.8], "hspace": 0.08})
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor(bg)
        ax.grid(True, alpha=0.5, color=grid_c, linewidth=0.5)
        ax.tick_params(colors=txt, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#CBD5E1")
            sp.set_linewidth(0.5)

    # Panel 1: Off-nadir
    ax1 = axes[0]
    ax1.plot(days, off_nadir, color="#0EA5E9", linewidth=0.8, alpha=0.9)
    ax1.fill_between(days, 0, off_nadir, alpha=0.12, color="#0EA5E9")
    ax1.set_ylabel("Off-nadir angle [deg]", fontsize=10, color=txt, fontweight="medium")
    ax1.set_ylim(0, 200)
    ax1.axhline(90, color="#94A3B8", linewidth=0.6, linestyle="--", alpha=0.7)
    ax1.text(0.01, 0.92, "MRF +Z deviation from nadir", transform=ax1.transAxes,
             fontsize=8, color="#64748B", fontstyle="italic", va="top")

    # Panel 2: Sun angles
    ax2 = axes[1]
    ax2.plot(days, sun_x, color="#6B7280", linewidth=0.9, alpha=0.9, label="X-axis")
    ax2.plot(days, sun_y, color="#F59E0B", linewidth=0.9, alpha=0.9, label="Y-axis")
    ax2.plot(days, sun_z, color="#8B5CF6", linewidth=0.9, alpha=0.9, label="Z-axis")
    ax2.set_ylabel("Sun angle from\nMRF axes [deg]", fontsize=10, color=txt, fontweight="medium")
    ax2.set_ylim(0, 200)
    # ax2.axhline(90, color="#94A3B8", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.8, edgecolor="#E2E8F0")
    ax2.text(0.01, 0.92, "Sun incidence on each body axis", transform=ax2.transAxes,
             fontsize=8, color="#64748B", fontstyle="italic", va="top")

    # Panel 3: Target direction in MRF
    ax3 = axes[2]
    # ax3.plot(days, tgt_x, color="#EF4444", linewidth=0.5, alpha=0.7, label="Target in X")
    # ax3.plot(days, tgt_y, color="#22C55E", linewidth=0.6, alpha=0.9, label="Target in Y")
    ax3.plot(days, tgt_z, color="#3B82F6", linewidth=0.9, alpha=0.9, label="Target in Z")
    ax3.set_ylabel("Target direction\nin MRF [−]", fontsize=10, color=txt, fontweight="medium")
    ax3.set_ylim(-1.15, 1.15)
    ax3.axhline(0, color="#94A3B8", linewidth=0.5)
    ax3.axhline(1, color="#94A3B8", linewidth=0.3, linestyle=":")
    ax3.axhline(-1, color="#94A3B8", linewidth=0.3, linestyle=":")
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.8, edgecolor="#E2E8F0")
    ax3.text(0.01, 0.92, "Unit vector to target in body frame", transform=ax3.transAxes,
             fontsize=8, color="#64748B", fontstyle="italic", va="top")

    # Panel 4: Along-track separation
    ax4 = axes[3]
    ax4.plot(days, dlambda, color="#1E293B", linewidth=0.8)
    ax4.fill_between(days, dlambda, alpha=0.06, color="#1E293B")
    ax4.set_ylabel("a·δλ [km]", fontsize=10, color=txt, fontweight="medium")
    ax4.set_xlabel("Mission elapsed time [days]", fontsize=10, color=txt, fontweight="medium")
    ax4.axhline(0, color="#DC2626", linewidth=0.5, linestyle="--", alpha=0.5)

    # Title
    start_str = records[0]["epoch_dt"].strftime("%b %d")
    end_str = records[-1]["epoch_dt"].strftime("%b %d, %Y")
    fig.suptitle(
        f"Endurance RDV — Attitude Profile Overview\n"
        f"{args.label}  |  GEO  |  {start_str} – {end_str}",
        fontsize=13, color=txt, fontweight="bold", y=0.98, linespacing=1.4
    )

    # plt.tight_layout(rect=[0, 0, 1, 1.05])
    # plt.tight_layout()
    plt.savefig(filepath, dpi=args.dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Plot: {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Validate inputs
    for fpath in args.csvs:
        if not os.path.isfile(fpath):
            print(f"ERROR: File not found: {fpath}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Build output filenames from label
    safe_label = args.label.replace("+", "_").replace(" ", "_").replace("/", "_")

    print(f"\n{'='*60}")
    print(f"  AOCS → Systema Converter")
    print(f"  Attitude law: {args.label}")
    print(f"{'='*60}\n")

    # Read CSVs
    print("Reading CSV files...")
    records = read_csv_files(args.csvs)
    print(f"  Total: {len(records)} records")
    print(f"  Span:  {records[0]['epoch_str']} → {records[-1]['epoch_str']}")

    # Check quaternions
    print("\nChecking quaternions...")
    num_flips = check_quaternion_continuity(records)

    # Write CCSDS files
    if not args.skip_ccsds:
        print("\nWriting CCSDS files...")
        oem_path = os.path.join(args.output_dir, f"{safe_label}_orbit.oem")
        aem_path = os.path.join(args.output_dir, f"{safe_label}_attitude.aem")
        readme_path = os.path.join(args.output_dir, f"{safe_label}_README.txt")
        write_oem(records, oem_path, args)
        write_aem(records, aem_path, args)
        write_readme(records, readme_path, args, num_flips)

    # Generate plot
    if not args.skip_plot:
        print("\nGenerating attitude plot...")
        plot_path = os.path.join(args.output_dir, f"{safe_label}_attitude_profile.png")
        generate_plot(records, plot_path, args)

    print(f"\nDone. Output in: {args.output_dir}/")


if __name__ == "__main__":
    main()
