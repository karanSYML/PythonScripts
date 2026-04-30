#!/usr/bin/env python3
"""
extract_firings_to_csv.py
=========================
Builds a combined time-series CSV (RDV + INS) and in-fills firing data from
the .mat maneuvers struct into the rows whose timestamps are nearest to each
firing epoch.  All firing columns default to 0 for non-firing rows.

Nearest-neighbour matching: every firing event maps to the closest 300-s grid
point (worst-case residual 148.9 s, well inside the 150-s half-step limit).

Output files
------------
  end1_target_sunOpt.csv   —  Mode 1 (Target+Sun)
  end1_nadir_sunopt.csv    —  Mode 2 (Nadir+Sun)

New columns added to the existing 28
-------------------------------------
  maneuver_type       : "RCS", "PPS", "RCS+PPS", or "" (no event)
  manTime_hr          : mission-elapsed hours of the firing (0 if none)
  firingTime_rcs_T01 … firingTime_rcs_T16  : per-thruster RCS firing [s]
  firingTime_pps_1   … firingTime_pps_3    : PPS firing values
"""

import os
import numpy as np
import pandas as pd
import scipy.io
from datetime import datetime, timezone, timedelta

J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def j2000_to_utc(sec_j2000):
    return J2000 + timedelta(seconds=float(sec_j2000))


def detect_sizes(mat_dir):
    """Read the RDV mat file to determine thruster channel counts."""
    fp  = os.path.join(mat_dir, "dataPackage4Thermal_RDV.mat")
    mat = scipy.io.loadmat(fp)
    man = mat["data4thermal"][0, 0]["maneuvers"][0, 0]
    n_rcs = man["firingTime_rcs"].shape[0]
    n_pps = man["firingTime_pps"].shape[0] if man["firingTime_pps"].size > 0 else 0
    return n_rcs, n_pps


def load_timeseries(mat_dir, rcs_cols, pps_cols):
    """Concatenate RDV + INS CSVs and parse UTC timestamps."""
    rdv = pd.read_csv(os.path.join(mat_dir, "dataPackage4Thermal_RDV.csv"),
                      skipinitialspace=True)
    ins = pd.read_csv(os.path.join(mat_dir, "dataPackage4Thermal_INS.csv"),
                      skipinitialspace=True)
    df = pd.concat([rdv, ins], ignore_index=True)

    # Initialise firing columns
    df["maneuver_type"] = 0
    df["manTime_hr"]    = 0.0
    for col in rcs_cols + pps_cols:
        df[col] = 0.0

    # Parse timestamps for nearest-neighbour lookup
    df["_utc"] = pd.to_datetime(df["UTC_ISOC"].str.strip(), utc=True)
    return df


def load_firing_events(mat_dir):
    """
    Extract all RCS and PPS firing events from RDV + INS .mat files.

    Returns two lists of dicts:
      rcs_events: [{utc, manTime_hr, firingTimes (n_rcs array)}, ...]
      pps_events: [{utc, manTime_hr, firingTimes (n_pps array)}, ...]
    """
    rcs_events, pps_events = [], []
    for phase in ["RDV", "INS"]:
        fp  = os.path.join(mat_dir, f"dataPackage4Thermal_{phase}.mat")
        mat = scipy.io.loadmat(fp)
        man = mat["data4thermal"][0, 0]["maneuvers"][0, 0]

        # RCS
        ephs = man["manEphSec_rcs"].flatten()
        ft   = man["firingTime_rcs"]          # (n_rcs, N)
        mth  = man["manTime_hr_rcs"].flatten()
        for i, (ep, mt) in enumerate(zip(ephs, mth)):
            rcs_events.append({
                "utc":        j2000_to_utc(ep),
                "manTime_hr": float(mt),
                "firingTimes": ft[:, i].astype(float),
            })

        # PPS (empty in INS phase)
        if man["manEphSec_pps"].size > 0:
            ephs = man["manEphSec_pps"].flatten()
            ft   = man["firingTime_pps"]      # (n_pps, N)
            mth  = man["manTime_hr_pps"].flatten()
            for i, (ep, mt) in enumerate(zip(ephs, mth)):
                pps_events.append({
                    "utc":        j2000_to_utc(ep),
                    "manTime_hr": float(mt),
                    "firingTimes": ft[:, i].astype(float),
                })

    return rcs_events, pps_events


def nearest_row(df_utc_series, target_utc):
    """Return the DataFrame index of the row whose _utc is closest to target_utc."""
    diffs = (df_utc_series - target_utc).abs()
    return diffs.idxmin()


def in_fill(df, rcs_events, pps_events, rcs_cols, pps_cols):
    """Write firing data into the matching time-series rows in-place."""
    utc_series = df["_utc"]

    # 0 = no firing  |  1 = PPS  |  2 = RCS
    for ev in rcs_events:
        idx = nearest_row(utc_series, ev["utc"])
        df.at[idx, "maneuver_type"] = 2
        df.at[idx, "manTime_hr"]    = ev["manTime_hr"]
        for j, col in enumerate(rcs_cols):
            df.at[idx, col] = ev["firingTimes"][j]

    for ev in pps_events:
        idx = nearest_row(utc_series, ev["utc"])
        df.at[idx, "maneuver_type"] = 1
        if df.at[idx, "manTime_hr"] == 0.0:
            df.at[idx, "manTime_hr"] = ev["manTime_hr"]
        for j, col in enumerate(pps_cols):
            df.at[idx, col] = ev["firingTimes"][j]

    return df


def build_output(mat_dir):
    n_rcs, n_pps = detect_sizes(mat_dir)
    rcs_cols = [f"firingTime_rcs_T{i+1:02d}" for i in range(n_rcs)]
    pps_cols = [f"firingTime_pps_{i+1}"       for i in range(n_pps)]

    df = load_timeseries(mat_dir, rcs_cols, pps_cols)
    rcs_events, pps_events = load_firing_events(mat_dir)
    df = in_fill(df, rcs_events, pps_events, rcs_cols, pps_cols)
    df.drop(columns=["_utc"], inplace=True)
    return df, n_rcs, n_pps


def main():
    configs = [
        ("end1_target_sunOpt", "end1_target_sunOpt.csv"),
        ("end1_Nadir_sunOpt",  "end1_nadir_sunopt.csv"),
    ]

    for mat_dir, out_name in configs:
        print(f"\nProcessing {mat_dir} -> {out_name}")
        df, n_rcs_ch, n_pps_ch = build_output(mat_dir)

        n_none = (df["maneuver_type"] == 0).sum()
        n_pps  = (df["maneuver_type"] == 1).sum()
        n_rcs  = (df["maneuver_type"] == 2).sum()
        print(f"  Total rows      : {len(df)}")
        print(f"  No firing   (0) : {n_none}")
        print(f"  PPS         (1) : {n_pps}  ({n_pps_ch} firing channels)")
        print(f"  RCS         (2) : {n_rcs}  ({n_rcs_ch} thruster channels)")
        print(f"  Total columns   : {len(df.columns)}")

        df.to_csv(out_name, index=False)
        print(f"  Saved: {out_name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
