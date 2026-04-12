#!/usr/bin/env python3
"""
make_video.py — Render an animation video of the thruster arm assembly.

Two scenes, saved as a single MP4:

  Scene 1 – Arm Sweep (≈ 15 s)
      Shoulder yaw sweeps 0° → 270° at fixed reach.
      The camera slowly orbits the assembly.
      Arm colour = green (OK) / orange (infeasible) / red (collision).
      Plume cone and CoG marker update every frame.

  Scene 2 – Pareto Walk (≈ 10 s)
      Steps through the Pareto-optimal configurations produced by
      pareto_scoring.run_pareto_analysis(), dwelling 1 s on each.
      Score annotations are shown in the status panel.

Usage
─────
    python make_video.py                         # default output: assembly_animation.mp4
    python make_video.py --output my_video.mp4
    python make_video.py --scene sweep           # scene 1 only
    python make_video.py --scene pareto          # scene 2 only
    python make_video.py --fps 24 --dpi 120      # quality tuning
"""

import argparse
import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")                 # headless — must come before pyplot import

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Project imports ────────────────────────────────────────────────────────────
from plume_impingement_pipeline import (
    StackConfig, RoboticArmGeometry, ThrusterParams,
    PlumePipeline, CaseMatrixGenerator,
)
from geometry_visualizer import (
    redraw, STACK, THRUSTER,
    pivot_position, panel_grid, relative_flux,
    draw_box, draw_panel_faces, draw_flux_overlay,
    draw_plume_cone, set_equal_aspect,
)
from pareto_scoring import ParetoScorer, run_pareto_analysis


# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT = "assembly_animation.mp4"
DEFAULT_FPS    = 24
DEFAULT_DPI    = 120
REACH_M        = 3.5       # fixed reach for the sweep scene
LINK_RATIO     = 0.5


# ==============================================================================
# SCENE 1  –  ARM SWEEP
# ==============================================================================

def build_sweep_frames(n_frames: int = 360) -> list:
    """Generate per-frame state dicts for the arm sweep scene.

    Yaw sweeps 0° → 270°.
    Camera azimuth orbits −60° → −240° (full turn).
    """
    yaw_vals  = np.linspace(0.0,   270.0, n_frames)
    cam_az    = np.linspace(-60.0, -240.0, n_frames)
    cam_el    = np.full(n_frames, 22.0)

    frames = []
    for i in range(n_frames):
        frames.append({
            "yaw_deg":      yaw_vals[i],
            "elev_deg":     0.0,
            "reach_m":      REACH_M,
            "link_ratio":   LINK_RATIO,
            "tracking_deg": 0.0,
            "elbow_up":     True,
            "show_flux":    False,
            "_cam_az":      cam_az[i],
            "_cam_el":      cam_el[i],
            "_scene_label": f"Scene 1 — Arm Sweep   yaw = {yaw_vals[i]:.0f}°",
        })
    return frames


# ==============================================================================
# SCENE 2  –  PARETO WALK
# ==============================================================================

def build_pareto_cases() -> list:
    """Run a compact sweep and return Pareto-optimal cases."""
    yaws    = list(range(0, 271, 15))     # 0, 15, 30, … 270
    reaches = [2.5, 3.0, 3.5, 4.0]
    ratios  = [0.4, 0.5, 0.6]

    cases = []
    for yaw, reach, ratio in itertools.product(yaws, reaches, ratios):
        cases.append({
            "arm_reach_m":         reach,
            "shoulder_yaw_deg":    yaw,
            "link_ratio":          ratio,
            "client_mass":         2000.0,
            "servicer_mass":       670.0,
            "panel_span_one_side": 16.0,
            "firing_duration_s":   25000.0,
            "mission_duration_yr": 5.0,
            "panel_tracking_deg":  0.0,
        })

    print(f"  Running sweep: {len(cases)} cases for Pareto scene …")
    pipeline = PlumePipeline(THRUSTER)
    results  = pipeline.run_sweep(cases, verbose=False)

    scorer = ParetoScorer(manoeuvre_type="NSSK", angle_budget_deg=50.0)
    scored = scorer.score(results)
    front  = scorer.pareto_front(scored, feasible_only=True)
    print(f"  Pareto front : {len(front)} configurations")
    return front


def build_pareto_frames(front: list, dwell_frames: int) -> list:
    """One dwell_frames-long hold per Pareto configuration.

    Camera slowly orbits during each hold.
    """
    if not front:
        return []

    frames = []
    n_configs = len(front)

    for k, cfg in enumerate(front):
        yaw   = cfg.get("shoulder_yaw_deg", 45.0)
        reach = cfg.get("arm_reach_m",      REACH_M)
        ratio = cfg.get("link_ratio",        LINK_RATIO)

        cam_az_start = -60.0 - k * (180.0 / max(n_configs - 1, 1))
        cam_az_vals  = np.linspace(cam_az_start, cam_az_start - 30.0, dwell_frames)

        for i in range(dwell_frames):
            frames.append({
                "yaw_deg":      yaw,
                "elev_deg":     0.0,
                "reach_m":      reach,
                "link_ratio":   ratio,
                "tracking_deg": 0.0,
                "elbow_up":     True,
                "show_flux":    False,
                "_cam_az":      cam_az_vals[i],
                "_cam_el":      20.0,
                "_scene_label": (
                    f"Scene 2 — Pareto Config {k+1}/{n_configs}\n"
                    f"yaw={yaw:.0f}°  reach={reach:.1f}m  ratio={ratio:.2f}\n"
                    f"score={cfg.get('pareto_score', 0):.3f}  "
                    f"NSSK dev={cfg.get('nssk_deviation_deg', 0):.1f}°  "
                    f"status={cfg.get('status','?')}"
                ),
                "_pareto_rank": k + 1,
                "_pareto_total": n_configs,
                "_pareto_score": cfg.get("pareto_score", 0.0),
                "_nssk_dev":     cfg.get("nssk_deviation_deg", 0.0),
                "_torque_mNm":   cfg.get("nssk_torque_Nm", 0.0) * 1000,
                "_eros_pct":     cfg.get("erosion_fraction", 0.0) * 100,
                "_status":       cfg.get("status", "?"),
            })
    return frames


# ==============================================================================
# FRAME RENDERER
# ==============================================================================

def render_frame(fig, ax3d, ax_info, ax_title, frame_state: dict):
    """Render a single animation frame into ax3d + ax_info + ax_title."""
    # ── 3D scene ──────────────────────────────────────────────────────────────
    redraw(
        ax3d, ax_info, STACK,
        yaw_deg      = frame_state["yaw_deg"],
        elev_deg     = frame_state["elev_deg"],
        reach_m      = frame_state["reach_m"],
        link_ratio   = frame_state["link_ratio"],
        tracking_deg = frame_state["tracking_deg"],
        elbow_up     = frame_state["elbow_up"],
        show_flux    = frame_state["show_flux"],
    )

    # ── Camera ────────────────────────────────────────────────────────────────
    ax3d.view_init(elev=frame_state["_cam_el"],
                   azim=frame_state["_cam_az"])

    # ── Title bar ─────────────────────────────────────────────────────────────
    ax_title.cla()
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5,
        frame_state.get("_scene_label", ""),
        transform=ax_title.transAxes,
        ha="center", va="center",
        fontsize=10, fontfamily="monospace",
        color="#1A252F",
    )

    # ── Pareto annotation overlay (scene 2 only) ───────────────────────────────
    if "_pareto_rank" in frame_state:
        rank  = frame_state["_pareto_rank"]
        total = frame_state["_pareto_total"]
        lines = [
            f"Rank   : {rank} / {total}",
            f"Score  : {frame_state['_pareto_score']:.4f}",
            f"NSSK Δ : {frame_state['_nssk_dev']:.1f}°",
            f"Torque : {frame_state['_torque_mNm']:.2f} mN·m",
            f"Erosion: {frame_state['_eros_pct']:.2f}%",
            f"Status : {frame_state['_status']}",
        ]
        status_color = {
            "SAFE": "#1E8449", "CAUTION": "#D4AC0D",
            "MARGINAL": "#E67E22", "FAIL": "#C0392B",
        }.get(frame_state["_status"], "#2C3E50")

        y = 0.42
        for j, line in enumerate(lines):
            color = status_color if j == 5 else "#2C3E50"
            ax_info.text(
                0.04, y, line,
                transform=ax_info.transAxes,
                fontsize=9, color=color, va="top",
                fontfamily="monospace",
            )
            y -= 0.052


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Render thruster arm animation video.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output MP4 file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--fps",    type=int,   default=DEFAULT_FPS,
                        help=f"Frames per second (default: {DEFAULT_FPS})")
    parser.add_argument("--dpi",    type=int,   default=DEFAULT_DPI,
                        help=f"Render DPI (default: {DEFAULT_DPI})")
    parser.add_argument("--scene",  choices=["sweep", "pareto", "both"],
                        default="both",
                        help="Which scenes to render (default: both)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Thruster Arm Assembly — Video Renderer")
    print("=" * 60)

    # ── Collect frames ────────────────────────────────────────────────────────
    all_frames: list = []

    if args.scene in ("sweep", "both"):
        n_sweep = int(10 * args.fps)    # 10 seconds
        print(f"\n[Scene 1] Arm sweep — {n_sweep} frames")
        all_frames += build_sweep_frames(n_sweep)

    if args.scene in ("pareto", "both"):
        front = build_pareto_cases()
        if front:
            dwell = max(int(args.fps * 1.5), 10)   # 1.5 s per config
            print(f"[Scene 2] Pareto walk — {len(front)} configs × {dwell} frames")
            all_frames += build_pareto_frames(front, dwell_frames=dwell)
        else:
            print("[Scene 2] No Pareto-optimal cases found — skipping.")

    if not all_frames:
        print("No frames to render. Exiting.")
        sys.exit(1)

    total_frames   = len(all_frames)
    total_duration = total_frames / args.fps
    print(f"\nTotal : {total_frames} frames  ({total_duration:.1f} s @ {args.fps} fps)")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), facecolor="#F8F9FA")

    ax_title = fig.add_axes([0.00, 0.93, 1.00, 0.07])
    ax_title.axis("off")

    ax3d    = fig.add_axes([0.01, 0.05, 0.62, 0.87], projection="3d")
    ax3d.set_facecolor("#EBF5FB")
    ax_info = fig.add_axes([0.65, 0.05, 0.19, 0.87])

    # ── Progress bar axis ─────────────────────────────────────────────────────
    ax_prog = fig.add_axes([0.01, 0.01, 0.98, 0.025])
    ax_prog.set_xlim(0, total_frames)
    ax_prog.set_ylim(0, 1)
    ax_prog.axis("off")
    prog_bar, = ax_prog.fill([0, 0, 0, 0], [0, 0, 1, 1],
                              color="#2471A3", alpha=0.7)

    # ── Animation function ────────────────────────────────────────────────────
    def _animate(frame_idx: int):
        if frame_idx % max(1, total_frames // 20) == 0:
            pct = 100 * frame_idx / total_frames
            print(f"  Rendering frame {frame_idx+1}/{total_frames}  ({pct:.0f}%)",
                  end="\r", flush=True)

        state = all_frames[frame_idx]
        render_frame(fig, ax3d, ax_info, ax_title, state)

        # Update progress bar
        prog_bar.set_xy([[0, 0], [frame_idx + 1, 0],
                         [frame_idx + 1, 1], [0, 1]])
        return []

    anim = animation.FuncAnimation(
        fig, _animate,
        frames=total_frames,
        interval=1000 / args.fps,
        blit=False,
    )

    # ── Write video ────────────────────────────────────────────────────────────
    output_path = args.output
    writer = animation.FFMpegWriter(
        fps=args.fps,
        metadata={"title": "Thruster Arm Assembly Animation",
                  "artist": "ThrusterArmWorkspaceAnalysis"},
        bitrate=2500,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
    )

    print(f"\nWriting → {output_path}")
    anim.save(output_path, writer=writer, dpi=args.dpi,
              savefig_kwargs={"facecolor": "#F8F9FA"})
    print(f"\nDone. Video saved: {output_path}")
    print(f"Size : {os.path.getsize(output_path) / 1e6:.1f} MB")

    plt.close(fig)


if __name__ == "__main__":
    main()
