#!/usr/bin/env python3
"""
plot_rtn_trajectory.py
======================
Plots the servicer trajectory relative to the target in the target's
RTN (Radial-Tangential-Normal) frame across the full OG3 RDV timeline,
annotated with mission sub-phases.

RTN frame (target-centred):
  R̂ = r_tgt / |r_tgt|
  Ñ = (r_tgt × v_tgt) / |r_tgt × v_tgt|
  T̂ = Ñ × R̂

Target velocity derived by central-difference of target position (300 s step).

Usage:
    python plot_rtn_trajectory.py \\
        --mode1-dir og3_target_sunOpt_nominal \\
        --output rtn_trajectory.png
"""

import argparse
import os
import warnings
import numpy as np
import scipy.io

warnings.filterwarnings("ignore")

# ── Timestep ──────────────────────────────────────────────────────────────────
DT_DAYS = 300.0 / 86400.0
DT_SEC  = 300.0

# ── Sub-phase definitions ─────────────────────────────────────────────────────
# (start_day, end_day, short_label, full_label, color)
PHASES = [
    ( 0,  1,  "P1", "Phase 1\nSK at −60 km",               "#CBD5E1"),
    ( 1, 11,  "P2", "Phase 2\nApproach\n−60→−30 km",        "#BFDBFE"),
    (11, 12,  "P3", "Phase 3\nSK at −30 km",                "#CBD5E1"),
    (12, 16,  "P4", "Phase 4\nAccel. approach\n−30→−1 km",  "#FDE68A"),
    (16, 17,  "P5", "Phase 5\nSK at −1 km",                 "#CBD5E1"),
    (17, 19,  "P6", "Phase 6\nCheckpoint\nresizing",        "#FBCFE8"),
    (19, 20,  "P7", "Phase 7\nSK at −1 km",                 "#CBD5E1"),
    (20, 22,  "P8", "Phase 8\nFly-by\n−1→+1 km",            "#A7F3D0"),
    (22, 24,  "P9", "Phase 9\nSK at +1 km",                 "#CBD5E1"),
]

# colours for trajectory segments
PHASE_TRAJ_COLORS = [
    "#64748B", "#3B82F6", "#64748B", "#F59E0B",
    "#64748B", "#EC4899", "#64748B", "#10B981", "#64748B",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mat(fp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = scipy.io.loadmat(fp)
    d = mat["data4thermal"][0, 0]
    n = d["pos_Earth2satCoG_ECI_m"].shape[1]
    return {
        "n":       n,
        "sat_pos": d["pos_Earth2satCoG_ECI_m"].T,   # (n, 3) m
        "tgt_pos": d["pos_Earth2tgtCoG_ECI_m"].T,   # (n, 3) m
        "dlambda": d["a_dlambda_m"].flatten() / 1e3,
    }


def load_full(mat_dir):
    rdv = load_mat(os.path.join(mat_dir, "dataPackage4Thermal_RDV.mat"))
    ins = load_mat(os.path.join(mat_dir, "dataPackage4Thermal_INS.mat"))
    nr  = rdv["n"]
    days = np.r_[np.arange(nr) * DT_DAYS,
                 nr * DT_DAYS + np.arange(ins["n"]) * DT_DAYS]
    sat  = np.vstack([rdv["sat_pos"], ins["sat_pos"]])
    tgt  = np.vstack([rdv["tgt_pos"], ins["tgt_pos"]])
    dlam = np.r_[rdv["dlambda"], ins["dlambda"]]
    return days, sat, tgt, dlam


# ── RTN computation ───────────────────────────────────────────────────────────

def compute_rtn(sat_pos, tgt_pos, dt_sec=DT_SEC):
    """Project relative position into target RTN frame."""
    n = len(tgt_pos)

    # Target velocity via central differences (forward/backward at edges)
    tgt_vel = np.empty_like(tgt_pos)
    tgt_vel[1:-1] = (tgt_pos[2:] - tgt_pos[:-2]) / (2 * dt_sec)
    tgt_vel[0]    = (tgt_pos[1]  - tgt_pos[0])   / dt_sec
    tgt_vel[-1]   = (tgt_pos[-1] - tgt_pos[-2])  / dt_sec

    dR = np.zeros(n)
    dT = np.zeros(n)
    dN = np.zeros(n)

    for i in range(n):
        r = tgt_pos[i]
        v = tgt_vel[i]
        R_hat = r / np.linalg.norm(r)
        N_hat = np.cross(r, v)
        N_hat /= np.linalg.norm(N_hat)
        T_hat = np.cross(N_hat, R_hat)

        delta = sat_pos[i] - tgt_pos[i]
        dR[i] = np.dot(delta, R_hat) / 1e3   # → km
        dT[i] = np.dot(delta, T_hat) / 1e3
        dN[i] = np.dot(delta, N_hat) / 1e3

    return dR, dT, dN


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor("#FAFBFC")
    ax.grid(True, alpha=0.4, color="#E2E8F0", linewidth=0.5)
    ax.tick_params(colors="#334155", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#CBD5E1"); sp.set_linewidth(0.5)


def _shade_phases(ax, days, vertical=True):
    """Add alternating phase band shading and labels."""
    txt = "#334155"
    for start, end, short, _, color in PHASES:
        lo = max(start, days[0])
        hi = min(end,   days[-1])
        if lo >= hi:
            continue
        if vertical:
            ax.axvspan(lo, hi, alpha=0.18, color=color, zorder=0)
            mid = (lo + hi) / 2
            ax.text(mid, ax.get_ylim()[1] * 0.97, short,
                    ha="center", va="top", fontsize=7, color=txt,
                    fontweight="bold", alpha=0.7)
        else:
            ax.axhspan(lo, hi, alpha=0.18, color=color, zorder=0)


# ── Main figure ───────────────────────────────────────────────────────────────

def generate_figure(days, dR, dT, dN, dlam, filepath, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    txt = "#334155"
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.08, wspace=0.25,
                           height_ratios=[1.6, 1, 1])

    # ── Panel A: T–R plane (main drift diagram) ───────────────────────────────
    ax_tr = fig.add_subplot(gs[0, 0])
    _style_ax(ax_tr)

    # colour trajectory by phase
    for i, (start, end, short, _, color) in enumerate(PHASES):
        mask = (days >= start) & (days < end)
        if mask.any():
            ax_tr.plot(dT[mask], dR[mask],
                       color=PHASE_TRAJ_COLORS[i], lw=1.2, alpha=0.85,
                       label=short)
            # mark start with a dot
            idx = np.where(mask)[0][0]
            ax_tr.scatter(dT[idx], dR[idx],
                          color=PHASE_TRAJ_COLORS[i], s=30, zorder=5)

    # target cross
    ax_tr.axhline(0, color="#94A3B8", lw=0.6, ls="--", alpha=0.6)
    ax_tr.axvline(0, color="#94A3B8", lw=0.6, ls="--", alpha=0.6)
    ax_tr.scatter([0], [0], marker="+", s=120, color="#EF4444",
                  linewidths=1.5, zorder=6, label="Target")

    # phase boundary vertical markers
    for start, end, short, _, _ in PHASES[1:]:
        if days[0] <= start <= days[-1]:
            ax_tr.axvline(dT[np.argmin(np.abs(days - start))],
                          color="#94A3B8", lw=0.5, ls=":", alpha=0.5)

    ax_tr.set_xlabel("Along-track T [km]", fontsize=10, color=txt)
    ax_tr.set_ylabel("Radial R [km]", fontsize=10, color=txt)
    ax_tr.set_title("T–R Plane (top view)", fontsize=11, color=txt,
                    fontweight="semibold")
    ax_tr.legend(fontsize=7.5, ncol=3, framealpha=0.9,
                 edgecolor="#E2E8F0", loc="upper right")

    # ── Panel B: T–N plane ────────────────────────────────────────────────────
    ax_tn = fig.add_subplot(gs[0, 1])
    _style_ax(ax_tn)

    for i, (start, end, short, _, color) in enumerate(PHASES):
        mask = (days >= start) & (days < end)
        if mask.any():
            ax_tn.plot(dT[mask], dN[mask],
                       color=PHASE_TRAJ_COLORS[i], lw=1.2, alpha=0.85,
                       label=short)
            idx = np.where(mask)[0][0]
            ax_tn.scatter(dT[idx], dN[idx],
                          color=PHASE_TRAJ_COLORS[i], s=30, zorder=5)

    ax_tn.axhline(0, color="#94A3B8", lw=0.6, ls="--", alpha=0.6)
    ax_tn.axvline(0, color="#94A3B8", lw=0.6, ls="--", alpha=0.6)
    ax_tn.scatter([0], [0], marker="+", s=120, color="#EF4444",
                  linewidths=1.5, zorder=6)
    ax_tn.set_xlabel("Along-track T [km]", fontsize=10, color=txt)
    ax_tn.set_ylabel("Normal N [km]", fontsize=10, color=txt)
    ax_tn.set_title("T–N Plane (side view)", fontsize=11, color=txt,
                    fontweight="semibold")

    # ── Panel C: Along-track T vs time ────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1, :])
    _style_ax(ax_t)
    ax_t.plot(days, dT, color="#2563EB", lw=0.8, alpha=0.9)
    ax_t.fill_between(days, dT, alpha=0.08, color="#2563EB")
    ax_t.axhline(0, color="#EF4444", lw=0.8, ls="--", alpha=0.6)

    # phase bands
    for start, end, short, _, color in PHASES:
        lo, hi = max(start, days[0]), min(end, days[-1])
        if lo < hi:
            ax_t.axvspan(lo, hi, alpha=0.15, color=color, zorder=0)
    # phase boundary lines + labels
    for start, end, short, full, _ in PHASES:
        if days[0] < start < days[-1]:
            ax_t.axvline(start, color="#94A3B8", lw=0.7, ls=":", alpha=0.7)
        mid = (max(start, days[0]) + min(end, days[-1])) / 2
        if days[0] <= mid <= days[-1]:
            ax_t.text(mid, ax_t.get_ylim()[0] if ax_t.get_ylim()[0] != 0 else -58,
                      short, ha="center", va="bottom", fontsize=7.5,
                      color="#475569", fontweight="bold")

    ax_t.set_ylabel("Along-track T [km]", fontsize=10, color=txt,
                    fontweight="medium")
    ax_t.set_title("Along-track separation vs time", fontsize=10,
                   color=txt, fontweight="semibold")

    # ── Panel D: R and N vs time ──────────────────────────────────────────────
    ax_rn = fig.add_subplot(gs[2, :])
    _style_ax(ax_rn)
    ax_rn.plot(days, dR, color="#7C3AED", lw=0.7, alpha=0.85, label="Radial R")
    ax_rn.plot(days, dN, color="#059669", lw=0.7, alpha=0.85, label="Normal N")
    ax_rn.axhline(0, color="#94A3B8", lw=0.5, ls="--", alpha=0.5)

    for start, end, short, _, color in PHASES:
        lo, hi = max(start, days[0]), min(end, days[-1])
        if lo < hi:
            ax_rn.axvspan(lo, hi, alpha=0.15, color=color, zorder=0)
    for start, _, _, _, _ in PHASES[1:]:
        if days[0] < start < days[-1]:
            ax_rn.axvline(start, color="#94A3B8", lw=0.7, ls=":", alpha=0.7)

    ax_rn.set_ylabel("Radial R, Normal N [km]", fontsize=10, color=txt,
                     fontweight="medium")
    ax_rn.set_xlabel("Mission elapsed time [days]", fontsize=10, color=txt)
    ax_rn.legend(fontsize=9, framealpha=0.9, edgecolor="#E2E8F0",
                 loc="upper right")

    # shared x-range for time panels
    for ax in [ax_t, ax_rn]:
        ax.set_xlim(days[0], days[-1])

    # phase label strip on T panel (redo after ylim is set)
    ylim_t = ax_t.get_ylim()
    for start, end, short, _, _ in PHASES:
        mid = (max(start, days[0]) + min(end, days[-1])) / 2
        if days[0] <= mid <= days[-1]:
            ax_t.text(mid, ylim_t[0] + (ylim_t[1] - ylim_t[0]) * 0.04,
                      short, ha="center", va="bottom", fontsize=7.5,
                      color="#475569", fontweight="bold")

    fig.suptitle(
        "OG3 Rendezvous — Servicer trajectory in target RTN frame\n"
        "P1: SK −60 km  |  P2: Approach −60→−30 km  |  P3: SK −30 km  |  "
        "P4: Accel. −30→−1 km  |  P5–P7: SK/CR at −1 km  |  "
        "P8: Fly-by −1→+1 km  |  P9: SK +1 km",
        fontsize=11, color=txt, fontweight="bold", y=1.01, linespacing=1.5)

    plt.savefig(filepath, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode1-dir", default="og3_target_sunOpt_nominal")
    p.add_argument("--output",    default="rtn_trajectory.png")
    p.add_argument("--dpi",       type=int, default=180)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print("  RTN Trajectory Plot")
    print(f"{'='*60}")

    print(f"\nLoading {args.mode1_dir}...")
    days, sat, tgt, dlam = load_full(args.mode1_dir)
    print(f"  {len(days)} points, {days[-1]:.2f} days")
    print(f"  Along-track range: {dlam.min():.1f} to {dlam.max():.1f} km")

    print("  Computing RTN components...")
    dR, dT, dN = compute_rtn(sat, tgt)
    print(f"  R: {dR.min():.2f} to {dR.max():.2f} km")
    print(f"  T: {dT.min():.2f} to {dT.max():.2f} km")
    print(f"  N: {dN.min():.2f} to {dN.max():.2f} km")

    print("\nGenerating figure...")
    generate_figure(days, dR, dT, dN, dlam, args.output, args.dpi)
    print("\nDone.")


if __name__ == "__main__":
    main()
