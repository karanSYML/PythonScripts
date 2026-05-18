#!/usr/bin/env python3
"""
power_budget.py
===============
Correlates solar power generation with subsystem consumption across the
full OG3 rendezvous timeline, broken down by operating event:
  - Baseline platform
  - Camera (NAC, active during combined pointing windows)
  - X-band comms TX (active during combined pointing windows)
  - RCS firing delta
  - PPS firing delta

Solar generation:  344 W × sin(angle_sun_from_Y)  [SAD tracks around Y-axis]
Eclipse:           detected from Earth shadow geometry

Usage:
    python power_budget.py \\
        --mode1-dir og3_target_sunOpt_nominal \\
        --mode2-dir og3_Nadir_sunOpt_nominal
"""

import argparse
import os
import warnings
import numpy as np
import scipy.io
from datetime import datetime, timezone
from rdv_phases import shade_phases

warnings.filterwarnings("ignore")

# ── Mission constants ──────────────────────────────────────────────────────────
DT_DAYS        = 300.0 / 86400.0
DT_SEC         = 300.0
J2000          = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
T0_UTC         = datetime(2028, 9, 1, 3, 49, 53, tzinfo=timezone.utc)
T0_SEC         = (T0_UTC - J2000).total_seconds()
CLOSE_KM       = -5.0
R_EARTH_M      = 6.3781e6

# ── Power budget constants (W) ─────────────────────────────────────────────────
P_SOLAR_MAX       = 344.0    # EOL, SAD fully sun-pointed
P_BASELINE        = 260.0    # platform (incl. RCS idle, AOCS, OBC, TT&C rx)
P_PPS_TOTAL       = 570.0    # total bus power during PPS firing
P_RCS_ARMED       = 6.0      # RCS armed (vs 4 W idle → +2 W, inside baseline)
P_RCS_FIRING      = 14.1     # hot gas firing, 4-thruster → delta above baseline
P_CAMERA_IDLE     = 5.0      # camera standby (one unit, 5 W — NAC or WAC TBC)
P_CAMERA_ACTIVE   = 8.0      # camera active during combined pointing window (8 W)
# SPEC_CAM (Triscape100) power TBC — add here when confirmed
P_SPECCAM_ACTIVE  = 0.0      # TBC
P_XBAND_TX        = 40.0     # X-band TX during comms window (educated guess)
ANTENNA_HCONE_DEG = 4.5      # X-band 3 dB half-cone
BATTERY_CAP_WH    = 550.0    # battery capacity


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
    ex  = d["extras"][0, 0]
    n   = d["pos_Earth2satCoG_ECI_m"].shape[1]
    return {
        "n":          n,
        "sun_y_deg":  d["angle_sunFromMRFaxes_deg"][1, :],   # angle from +Y
        "sat_pos":    d["pos_Earth2satCoG_ECI_m"].T,          # (n, 3) m
        "sun_pos":    d["pos_Earth2Sun_ECI_m"].T,             # (n, 3) m
        "dlambda":    d["a_dlambda_m"].flatten() / 1e3,
        "ant_earth":  np.degrees(ex["angle_antenna2Earth_rad"].flatten()),
        "eph_rcs":    _safe_flat(man["manEphSec_rcs"]),
        "eph_pps":    _safe_flat(man["manEphSec_pps"]),
    }


def load_mode(mat_dir):
    rdv = load_mat(os.path.join(mat_dir, "dataPackage4Thermal_RDV.mat"))
    ins = load_mat(os.path.join(mat_dir, "dataPackage4Thermal_INS.mat"))
    nr  = rdv["n"]

    days = np.r_[np.arange(nr) * DT_DAYS,
                 nr * DT_DAYS + np.arange(ins["n"]) * DT_DAYS]

    def cat(k):
        a, b = rdv[k], ins[k]
        return np.concatenate([a, b], axis=0 if a.ndim == 1 else 0)

    def eph_days(arr):
        return (arr - T0_SEC) / 86400.0 if arr.size > 0 else np.array([])

    return {
        "days":      days,
        "sun_y_deg": cat("sun_y_deg"),
        "sat_pos":   cat("sat_pos"),
        "sun_pos":   cat("sun_pos"),
        "dlambda":   cat("dlambda"),
        "ant_earth": cat("ant_earth"),
        "man_rcs":   eph_days(np.r_[rdv["eph_rcs"], ins["eph_rcs"]]),
        "man_pps":   eph_days(rdv["eph_pps"]),
    }


# ── Eclipse detection ──────────────────────────────────────────────────────────

def compute_eclipse(sat_pos, sun_pos):
    """True where satellite is in Earth's umbra."""
    n = len(sat_pos)
    eclipse = np.zeros(n, dtype=bool)
    for i in range(n):
        sp = sun_pos[i]
        sv = sat_pos[i]
        sun_hat = sp / np.linalg.norm(sp)
        proj = np.dot(sv, sun_hat)
        if proj >= 0:          # sat on sunlit side
            continue
        perp = np.linalg.norm(sv - proj * sun_hat)
        if perp < R_EARTH_M:
            eclipse[i] = True
    return eclipse


# ── Power computation ──────────────────────────────────────────────────────────

def compute_power(m):
    """Return generation and consumption arrays (W) for every timestep."""
    n = len(m["days"])

    # ── Generation ────────────────────────────────────────────────────────────
    eclipse = compute_eclipse(m["sat_pos"], m["sun_pos"])
    theta_y = np.radians(m["sun_y_deg"])
    p_gen = np.where(eclipse, 0.0, P_SOLAR_MAX * np.abs(np.sin(theta_y)))

    # ── Consumption: build stacked components ─────────────────────────────────
    p_base    = np.full(n, P_BASELINE)

    # Camera + comms active during combined pointing window
    in_window = m["ant_earth"] <= ANTENNA_HCONE_DEG
    p_camera  = np.where(in_window, P_CAMERA_ACTIVE + P_SPECCAM_ACTIVE, P_CAMERA_IDLE)
    p_comms   = np.where(in_window, P_XBAND_TX, 0.0)

    # RCS firing: mark epochs within ±DT_SEC/2 of a maneuver event
    p_rcs_delta = np.zeros(n)
    for d_rcs in m["man_rcs"]:
        mask = np.abs(m["days"] - d_rcs) <= DT_DAYS / 2
        p_rcs_delta[mask] = P_RCS_FIRING - 4.0   # delta above idle

    # PPS firing: whole-step if epoch falls within any PPS event span
    p_pps_delta = np.zeros(n)
    for d_pps in m["man_pps"]:
        mask = np.abs(m["days"] - d_pps) <= DT_DAYS / 2
        # PPS total replaces baseline; delta = P_PPS_TOTAL - P_BASELINE
        p_pps_delta[mask] = P_PPS_TOTAL - P_BASELINE

    p_total = p_base + p_camera + p_comms + p_rcs_delta + p_pps_delta

    return {
        "p_gen":       p_gen,
        "p_total":     p_total,
        "p_base":      p_base,
        "p_camera":    p_camera,
        "p_comms":     p_comms,
        "p_rcs_delta": p_rcs_delta,
        "p_pps_delta": p_pps_delta,
        "eclipse":     eclipse,
    }


def compute_battery(p_gen, p_total, dt_sec=DT_SEC, capacity_wh=BATTERY_CAP_WH):
    """Integrate net power into battery state-of-charge (Wh), clamped 0–capacity."""
    soc = np.zeros(len(p_gen))
    soc[0] = capacity_wh
    for i in range(1, len(p_gen)):
        delta = (p_gen[i] - p_total[i]) * dt_sec / 3600.0
        soc[i] = np.clip(soc[i - 1] + delta, 0.0, capacity_wh)
    return soc


# ── Plotting ──────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor("#FAFBFC")
    ax.grid(True, alpha=0.4, color="#E2E8F0", linewidth=0.5)
    ax.tick_params(colors="#334155", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#CBD5E1"); sp.set_linewidth(0.5)


def generate_power_figure(m, pw, mode_label, x_range, phase_label, filepath, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    txt  = "#334155"
    lo, hi = x_range
    mask = (m["days"] >= lo) & (m["days"] <= hi)
    d    = m["days"][mask]

    p_gen       = pw["p_gen"][mask]
    p_base      = pw["p_base"][mask]
    p_camera    = pw["p_camera"][mask]
    p_comms     = pw["p_comms"][mask]
    p_rcs       = pw["p_rcs_delta"][mask]
    p_pps       = pw["p_pps_delta"][mask]
    p_total     = pw["p_total"][mask]
    eclipse     = pw["eclipse"][mask]
    soc         = compute_battery(pw["p_gen"], pw["p_total"])[mask]

    fig, axes = plt.subplots(4, 1, figsize=(15, 14),
                             gridspec_kw={"height_ratios": [1.1, 1.2, 0.9, 0.8],
                                          "hspace": 0.10})
    fig.patch.set_facecolor("white")
    for ax in axes:
        _style_ax(ax)

    # ── Panel 1: Generation ───────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(d, p_gen, alpha=0.25, color="#F59E0B")
    ax1.plot(d, p_gen, color="#D97706", lw=0.8, label="Solar generation")
    ax1.axhline(P_BASELINE, color="#64748B", lw=1.0, ls="--", alpha=0.7,
                label=f"Baseline load ({P_BASELINE:.0f} W)")
    # Eclipse shading
    for i in range(len(d)):
        if eclipse[i]:
            ax1.axvspan(d[i] - DT_DAYS/2, d[i] + DT_DAYS/2,
                        alpha=0.3, color="#1E293B", zorder=0)
    ax1.set_ylim(0, P_SOLAR_MAX * 1.15)
    ax1.set_ylabel("Solar generation [W]", fontsize=10, color=txt, fontweight="medium")
    ax1.legend(fontsize=8, framealpha=0.9, loc="lower right", edgecolor="#E2E8F0")
    shade_phases(ax1, lo, hi)
    ax1.text(0.012, 0.96, f"Mean gen: {p_gen.mean():.0f} W  |  Eclipse epochs: {eclipse.sum()}",
             transform=ax1.transAxes, fontsize=8.5, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CBD5E1", alpha=0.93))

    # ── Panel 2: Consumption stacked ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.stackplot(d,
                  p_base, p_camera, p_comms, p_rcs, p_pps,
                  labels=["Baseline", "Camera + SPEC_CAM (TBC)", "X-band TX (est.)",
                          "RCS firing Δ", "PPS firing Δ"],
                  colors=["#94A3B8", "#8B5CF6", "#06B6D4", "#3B82F6", "#EF4444"],
                  alpha=0.85)
    ax2.plot(d, p_total, color="#1E293B", lw=0.7, alpha=0.6, label="Total consumption")
    ax2.plot(d, p_gen,   color="#D97706", lw=1.0, ls="--", alpha=0.9, label="Generation")
    ax2.set_ylim(0, max(p_total.max(), p_gen.max()) * 1.15)
    ax2.set_ylabel("Power [W]", fontsize=10, color=txt, fontweight="medium")
    ax2.legend(fontsize=8, framealpha=0.9, loc="upper right",
               edgecolor="#E2E8F0", ncol=2)
    shade_phases(ax2, lo, hi)

    # ── Panel 3: Net balance ──────────────────────────────────────────────────
    ax3 = axes[2]
    net = p_gen - p_total
    ax3.fill_between(d, net, 0,
                     where=net >= 0, alpha=0.30, color="#10B981", label="Surplus")
    ax3.fill_between(d, net, 0,
                     where=net < 0,  alpha=0.30, color="#EF4444", label="Deficit")
    ax3.plot(d, net, color="#334155", lw=0.7, alpha=0.8)
    ax3.axhline(0, color="#64748B", lw=0.8, ls="--")
    surplus_pct = np.sum(net >= 0) / len(net) * 100
    ax3.text(0.012, 0.96,
             f"Surplus: {surplus_pct:.0f}% of epochs  |  "
             f"Mean net: {net.mean():.0f} W  |  "
             f"Worst deficit: {net.min():.0f} W",
             transform=ax3.transAxes, fontsize=8.5, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CBD5E1", alpha=0.93))
    ax3.set_ylabel("Net power [W]", fontsize=10, color=txt, fontweight="medium")
    ax3.legend(fontsize=8, framealpha=0.9, loc="upper right", edgecolor="#E2E8F0")
    shade_phases(ax3, lo, hi)

    # ── Panel 4: Battery SoC ──────────────────────────────────────────────────
    ax4 = axes[3]
    soc_pct = soc / BATTERY_CAP_WH * 100
    ax4.fill_between(d, soc_pct, alpha=0.25, color="#6366F1")
    ax4.plot(d, soc_pct, color="#4338CA", lw=0.8)
    ax4.axhline(20, color="#EF4444", lw=1.0, ls="--", alpha=0.8, label="20% DoD limit")
    ax4.set_ylim(0, 110)
    ax4.set_ylabel("Battery SoC [%]", fontsize=10, color=txt, fontweight="medium")
    ax4.set_xlabel("Mission elapsed time [days]", fontsize=10, color=txt)
    ax4.legend(fontsize=8, framealpha=0.9, loc="lower right", edgecolor="#E2E8F0")
    shade_phases(ax4, lo, hi, label_y=0.015, label_va="bottom")

    for ax in axes:
        ax.set_xlim(lo, hi)

    fig.suptitle(
        f"Power Budget — {mode_label}  |  {phase_label}\n"
        f"Generation: 344 W × sin(θ_Y)  |  "
        f"Baseline: {P_BASELINE:.0f} W  |  "
        f"PPS: {P_PPS_TOTAL:.0f} W  |  "
        f"X-band TX: {P_XBAND_TX:.0f} W (est.)",
        fontsize=11, color=txt, fontweight="bold", y=0.995, linespacing=1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode1-dir", default="og3_target_sunOpt_nominal")
    p.add_argument("--mode2-dir", default="og3_Nadir_sunOpt_nominal")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("  Power Budget Analysis")
    print(f"{'='*60}")

    for mode_dir, mode_label, prefix in [
        (args.mode1_dir, "Mode 1 (Target+Sun)", "power_m1"),
        (args.mode2_dir, "Mode 2 (Nadir+Sun)",  "power_m2"),
    ]:
        print(f"\nLoading {mode_label}...")
        m  = load_mode(mode_dir)
        pw = compute_power(m)
        n  = len(m["days"])
        print(f"  {n} points, {m['days'][-1]:.2f} days")
        print(f"  Mean generation:  {pw['p_gen'].mean():.1f} W")
        print(f"  Mean consumption: {pw['p_total'].mean():.1f} W")
        print(f"  Eclipse epochs:   {pw['eclipse'].sum()}")

        close_idx = np.where(m["dlambda"] > CLOSE_KM)[0]
        phase_day = m["days"][close_idx[0]] if close_idx.size > 0 else m["days"][-1] * 0.6

        for phase, rng, tag in [
            ("Far Range (−60 to −5 km)",  (0.0, phase_day),        f"{prefix}_far.png"),
            ("Close Range (−5 to +1 km)", (phase_day, m["days"][-1]), f"{prefix}_close.png"),
        ]:
            print(f"  Generating {phase}...")
            generate_power_figure(m, pw, mode_label, rng, phase, tag, args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
