#!/usr/bin/env python3
"""
extract_firings_to_csv.py
=========================
Builds a combined time-series CSV (RDV + INS) directly from the .mat files in
each input folder, then in-fills firing data from the maneuvers struct into
the rows whose timestamps are nearest to each firing epoch.

Time-series UTC_ISOC is reconstructed from the per-file mission anchor:
    t0_TT_J2000_s = manEphSec - manTime_hr*3600   (constant within a file)
    t0_UTC_J2000_s = t0_TT_J2000_s - 69.184       (TT->UTC, valid for 2028)
    row i timestamp = t0_UTC + i*300 s

Firing events use the same -69.184 s correction so nearest-neighbour matching
collapses to pure 300-s grid quantization (worst-case residual <150 s).

Output files
------------
  end1_target_sunOpt.csv   —  Mode 1 (Target+Sun)
  end1_nadir_sunopt.csv    —  Mode 2 (Nadir+Sun)

Columns
-------
Time-series (28):
  UTC_ISOC, pos_Earth2satCoG_ECI_m_{x,y,z}, vel_Earth2satCoG_ECI_ms_{x,y,z},
  pos_Earth2tgtCoG_ECI_m_{x,y,z}, pos_Earth2Sun_ECI_m_{x,y,z},
  quat_ECI2MRF_{a,i,j,k}, angle_SunPhaseAngle_rad, uv_sat2Earth_MRF_{x,y,z},
  uv_sat2Target_MRF_{x,y,z}, angle_sunFromMRFaxes_deg_{x,y,z}, a_dlambda_m

Firing in-fill (variable):
  maneuver_type       : 0 = none, 1 = PPS, 2 = RCS
  manTime_hr          : mission-elapsed hours of the firing (0 if none)
  firingTime_rcs_T01 … firingTime_rcs_TNN
  firingTime_pps_1   … firingTime_pps_M
"""

import os
import numpy as np
import pandas as pd
import scipy.io
from datetime import datetime, timezone, timedelta

J2000        = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
TT_MINUS_UTC = 69.184      # seconds, valid through 2028 (no leap second)
DT_S         = 300.0       # propagation step

VEC3_FIELDS = [
    "pos_Earth2satCoG_ECI_m",
    "vel_Earth2satCoG_ECI_ms",
    "pos_Earth2tgtCoG_ECI_m",
    "pos_Earth2Sun_ECI_m",
    "uv_sat2Earth_MRF",
    "uv_sat2Target_MRF",
    "angle_sunFromMRFaxes_deg",
]
VEC3_SUFFIX  = ["x", "y", "z"]
QUAT_SUFFIX  = ["a", "i", "j", "k"]
SCALAR_FIELDS = ["angle_SunPhaseAngle_rad", "a_dlambda_m"]


def j2000_to_utc(sec_j2000):
    return J2000 + timedelta(seconds=float(sec_j2000))


def load_phase(mat_path):
    """Return (df_phase, t0_utc_j2000_s, n_rcs, n_pps) for a single .mat file."""
    mat  = scipy.io.loadmat(mat_path)
    data = mat["data4thermal"][0, 0]
    man  = data["maneuvers"][0, 0]

    # Mission anchor from any maneuver event (consistent across all events)
    if man["manEphSec_pps"].size > 0:
        eph0 = float(man["manEphSec_pps"].flatten()[0])
        mt0  = float(man["manTime_hr_pps"].flatten()[0])
    else:
        eph0 = float(man["manEphSec_rcs"].flatten()[0])
        mt0  = float(man["manTime_hr_rcs"].flatten()[0])
    t0_tt  = eph0 - mt0 * 3600.0
    t0_utc = t0_tt - TT_MINUS_UTC

    n = data["pos_Earth2satCoG_ECI_m"].shape[1]

    # Build column dict
    cols = {}
    times = [j2000_to_utc(t0_utc + i * DT_S) for i in range(n)]
    cols["UTC_ISOC"] = [t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond * 1000:09d}"
                        for t in times]

    for f in VEC3_FIELDS:
        arr = data[f]
        for k, suf in enumerate(VEC3_SUFFIX):
            cols[f"{f}_{suf}"] = arr[k, :]

    quat = data["quat_ECI2MRF"]
    for k, suf in enumerate(QUAT_SUFFIX):
        cols[f"quat_ECI2MRF_{suf}"] = quat[k, :]

    for f in SCALAR_FIELDS:
        cols[f] = data[f].flatten()

    df = pd.DataFrame(cols)

    # Reorder to match the legacy CSV column order
    order = (
        ["UTC_ISOC"]
        + [f"pos_Earth2satCoG_ECI_m_{s}"  for s in VEC3_SUFFIX]
        + [f"vel_Earth2satCoG_ECI_ms_{s}" for s in VEC3_SUFFIX]
        + [f"pos_Earth2tgtCoG_ECI_m_{s}"  for s in VEC3_SUFFIX]
        + [f"pos_Earth2Sun_ECI_m_{s}"     for s in VEC3_SUFFIX]
        + [f"quat_ECI2MRF_{s}"            for s in QUAT_SUFFIX]
        + ["angle_SunPhaseAngle_rad"]
        + [f"uv_sat2Earth_MRF_{s}"        for s in VEC3_SUFFIX]
        + [f"uv_sat2Target_MRF_{s}"       for s in VEC3_SUFFIX]
        + [f"angle_sunFromMRFaxes_deg_{s}" for s in VEC3_SUFFIX]
        + ["a_dlambda_m"]
    )
    df = df[order]

    n_rcs = man["firingTime_rcs"].shape[0]
    n_pps = man["firingTime_pps"].shape[0] if man["firingTime_pps"].size > 0 else 0
    return df, t0_utc, n_rcs, n_pps


def detect_sizes(mat_dir):
    fp_rdv = os.path.join(mat_dir, "dataPackage4Thermal_RDV.mat")
    mat = scipy.io.loadmat(fp_rdv)
    man = mat["data4thermal"][0, 0]["maneuvers"][0, 0]
    n_rcs = man["firingTime_rcs"].shape[0]
    n_pps = man["firingTime_pps"].shape[0] if man["firingTime_pps"].size > 0 else 0
    return n_rcs, n_pps


def load_timeseries(mat_dir, rcs_cols, pps_cols):
    """Read time-series from RDV + INS .mat files, concatenate, init firing cols."""
    rdv_df, _, _, _ = load_phase(os.path.join(mat_dir, "dataPackage4Thermal_RDV.mat"))
    ins_df, _, _, _ = load_phase(os.path.join(mat_dir, "dataPackage4Thermal_INS.mat"))
    df = pd.concat([rdv_df, ins_df], ignore_index=True)

    df["maneuver_type"] = 0
    df["manTime_hr"]    = 0.0
    for col in rcs_cols + pps_cols:
        df[col] = 0.0

    df["_utc"] = pd.to_datetime(df["UTC_ISOC"], utc=True)
    return df


def load_firing_events(mat_dir):
    """
    Extract all RCS and PPS firing events from RDV + INS .mat files.

    Maneuver epochs are TT-J2000 seconds; subtract TT_MINUS_UTC so they sit on
    the same UTC clock as the reconstructed time-series timestamps.
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
                "utc":         j2000_to_utc(float(ep) - TT_MINUS_UTC),
                "manTime_hr":  float(mt),
                "firingTimes": ft[:, i].astype(float),
            })

        # PPS (empty in INS phase)
        if man["manEphSec_pps"].size > 0:
            ephs = man["manEphSec_pps"].flatten()
            ft   = man["firingTime_pps"]      # (n_pps, N)
            mth  = man["manTime_hr_pps"].flatten()
            for i, (ep, mt) in enumerate(zip(ephs, mth)):
                pps_events.append({
                    "utc":         j2000_to_utc(float(ep) - TT_MINUS_UTC),
                    "manTime_hr":  float(mt),
                    "firingTimes": ft[:, i].astype(float),
                })

    return rcs_events, pps_events


def nearest_row(df_utc_series, target_utc):
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
