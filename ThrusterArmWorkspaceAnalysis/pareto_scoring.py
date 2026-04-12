"""
pareto_scoring.py — Multi-objective Pareto scoring and filtering for the
Thruster Arm plume impingement sweep results.

Three objectives (all minimised after normalisation):
  1. Fuel cost        – cosine penalty from thrust angle deviation
                        cost = 1 − cos(deviation_deg)  ∈ [0, 1]
  2. Disturbance      – NSSK or EWSK torque [N·m], normalised by fleet max
  3. Erosion risk     – panel erosion fraction + antenna penalty, normalised

Feasibility gates (hard filters applied before Pareto):
  • IK feasible (joint limits satisfied, or status != FAIL on geometry)
  • Thrust angle budget  ≤ angle_budget_deg  (soft budget from spec §5.1, default 50°)
  • Arm does not collide with client bus  (checked via GeometryEngine)

Usage
─────
    from pareto_scoring import ParetoScorer
    from plume_impingement_pipeline import PlumePipeline

    pipeline = PlumePipeline()
    results  = pipeline.run_sweep(cases)

    scorer = ParetoScorer(manoeuvre_type="NSSK")
    scored = scorer.score(results)          # adds objective + score columns
    pareto = scorer.pareto_front(scored)    # non-dominated subset

    scorer.plot_pareto(scored, x="fuel_cost", y="erosion_risk",
                       color="disturbance_score", save_path="pareto.png")
    scorer.export_csv(scored, "sweep_scored.csv")
"""

import csv
import math
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ---------------------------------------------------------------------------
# 1.  OBJECTIVE FUNCTIONS
# ---------------------------------------------------------------------------

def _fuel_cost(deviation_deg: float) -> float:
    """Fractional ΔV penalty from off-axis thrust.

    cost = 1 − cos(θ)  ∈ [0, 1]
    At 0° → 0 (no penalty).  At 45° → 0.293.  At 90° → 1.0 (total loss).
    """
    return 1.0 - math.cos(math.radians(deviation_deg))


def _erosion_risk(erosion_fraction: float,
                  ant_max_erosion_um: float,
                  mat_thickness_um: float = 15.0,
                  ant_weight: float = 0.3) -> float:
    """Combined erosion risk score ∈ [0, ∞).

    Panel component  : erosion_fraction  (erosion_um / thickness_um)
    Antenna component: ant_max_erosion_um / mat_thickness_um  (same scale)
    Combined         : panel + ant_weight * antenna
    """
    ant_frac = ant_max_erosion_um / max(mat_thickness_um, 1e-9)
    return erosion_fraction + ant_weight * ant_frac


# ---------------------------------------------------------------------------
# 2.  PARETO DOMINANCE
# ---------------------------------------------------------------------------

def _is_dominated(candidate: np.ndarray, others: np.ndarray) -> bool:
    """Return True if `candidate` is dominated by any row in `others`.

    A solution a dominates b iff a ≤ b on ALL objectives AND a < b on
    at least one objective (minimisation sense).
    """
    dominated = np.all(others <= candidate, axis=1) & np.any(others < candidate, axis=1)
    return bool(np.any(dominated))


def pareto_filter(objective_matrix: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated rows in objective_matrix (n × k).

    All objectives are assumed to be minimised.
    """
    n = len(objective_matrix)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        if _is_dominated(objective_matrix[i], objective_matrix[mask]):
            mask[i] = False
    return mask


# ---------------------------------------------------------------------------
# 3.  SCORER
# ---------------------------------------------------------------------------

class ParetoScorer:
    """Score, filter, and plot sweep results from PlumePipeline.run_sweep().

    Parameters
    ──────────
    manoeuvre_type : "NSSK" or "EWSK"
        Selects which deviation/torque columns to use as objectives.
    angle_budget_deg : float
        Hard feasibility gate on thrust angle deviation (spec §5.1: 50°).
    mat_thickness_um : float
        Solar-panel coating thickness for erosion normalisation.
    ant_erosion_weight : float
        Relative weight of antenna erosion vs panel erosion in risk score.
    """

    ANGLE_BUDGET_DEFAULT = 50.0   # deg  (spec §5.1)

    def __init__(self,
                 manoeuvre_type: str = "NSSK",
                 angle_budget_deg: float = ANGLE_BUDGET_DEFAULT,
                 mat_thickness_um: float = 15.0,
                 ant_erosion_weight: float = 0.3):
        assert manoeuvre_type in ("NSSK", "EWSK"), \
            "manoeuvre_type must be 'NSSK' or 'EWSK'"
        self.manoeuvre_type     = manoeuvre_type
        self.angle_budget_deg   = angle_budget_deg
        self.mat_thickness_um   = mat_thickness_um
        self.ant_erosion_weight = ant_erosion_weight

        self._dev_key    = f"{manoeuvre_type.lower()}_deviation_deg"
        self._torque_key = f"{manoeuvre_type.lower()}_torque_Nm"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, results: List[Dict],
              add_feasibility: bool = True) -> List[Dict]:
        """Add objective and score columns to each result dict.

        New keys added
        ──────────────
        fuel_cost           1 − cos(deviation)  ∈ [0, 1]
        disturbance_raw     torque [N·m]  (raw, not yet normalised)
        erosion_risk_raw    panel_fraction + ant_weight * ant_fraction
        fuel_cost_norm      normalised to [0, 1] over the sweep
        disturbance_norm    normalised to [0, 1]
        erosion_risk_norm   normalised to [0, 1]
        pareto_score        equal-weight sum of the three normalised objectives
        feasible            bool — passes all hard gates
        feasibility_reason  "" if feasible, else description of first failing gate
        """
        if not results:
            return results

        # ── raw objectives ─────────────────────────────────────────────
        for r in results:
            r["fuel_cost"]        = _fuel_cost(r[self._dev_key])
            r["disturbance_raw"]  = r[self._torque_key]
            r["erosion_risk_raw"] = _erosion_risk(
                r["erosion_fraction"],
                r["ant_max_erosion_um"],
                self.mat_thickness_um,
                self.ant_erosion_weight,
            )

        # ── normalise each objective to [0, 1] over the full sweep ────
        for raw_key, norm_key in [
            ("fuel_cost",        "fuel_cost_norm"),
            ("disturbance_raw",  "disturbance_norm"),
            ("erosion_risk_raw", "erosion_risk_norm"),
        ]:
            vals = np.array([r[raw_key] for r in results], dtype=float)
            lo, hi = vals.min(), vals.max()
            span = hi - lo if hi > lo else 1.0
            for r, v in zip(results, vals):
                r[norm_key] = float((v - lo) / span)

        # ── composite score ────────────────────────────────────────────
        for r in results:
            r["pareto_score"] = (r["fuel_cost_norm"]
                                 + r["disturbance_norm"]
                                 + r["erosion_risk_norm"]) / 3.0

        # ── feasibility ────────────────────────────────────────────────
        if add_feasibility:
            self._flag_feasibility(results)

        return results

    def pareto_front(self, scored_results: List[Dict],
                     feasible_only: bool = True) -> List[Dict]:
        """Return the non-dominated subset.

        Parameters
        ──────────
        scored_results : output of score()
        feasible_only  : if True, only consider feasible cases for the front

        Returns
        ───────
        List of result dicts that lie on the Pareto front, sorted by
        pareto_score ascending.
        """
        pool = scored_results
        if feasible_only:
            pool = [r for r in scored_results if r.get("feasible", True)]

        if not pool:
            return []

        obj = np.array([[r["fuel_cost_norm"],
                         r["disturbance_norm"],
                         r["erosion_risk_norm"]] for r in pool])

        mask = pareto_filter(obj)
        front = [r for r, m in zip(pool, mask) if m]
        front.sort(key=lambda r: r["pareto_score"])
        return front

    def summary(self, scored_results: List[Dict]) -> None:
        """Print a compact summary table to stdout."""
        total     = len(scored_results)
        feasible  = sum(1 for r in scored_results if r.get("feasible", True))
        front     = self.pareto_front(scored_results, feasible_only=True)

        print(f"\n{'='*72}")
        print(f"  Pareto Scoring Summary  –  Manoeuvre: {self.manoeuvre_type}")
        print(f"{'='*72}")
        print(f"  Total cases   : {total}")
        print(f"  Feasible      : {feasible}  ({100*feasible/max(total,1):.0f}%)")
        print(f"  Pareto-optimal: {len(front)}")
        print(f"  Angle budget  : {self.angle_budget_deg}°")

        if not front:
            print("  No Pareto-optimal feasible cases found.")
            return

        print(f"\n  {'#':>3}  {'yaw°':>5}  {'reach':>6}  {'dev°':>6}  "
              f"{'torque mNm':>11}  {'eros%':>6}  {'score':>7}  status")
        print(f"  {'-'*70}")
        for i, r in enumerate(front[:20]):   # cap at 20 rows
            yaw   = r.get("shoulder_yaw_deg", float("nan"))
            reach = r.get("arm_reach_m", float("nan"))
            dev   = r[self._dev_key]
            torq  = r["disturbance_raw"] * 1000   # N·m → mN·m
            eros  = r["erosion_fraction"] * 100    # fraction → %
            score = r["pareto_score"]
            stat  = r.get("status", "?")
            print(f"  {i+1:>3}  {yaw:>5.1f}  {reach:>6.2f}  {dev:>6.1f}  "
                  f"{torq:>11.3f}  {eros:>6.2f}  {score:>7.4f}  {stat}")

        if len(front) > 20:
            print(f"  ... {len(front)-20} more (export CSV for full list)")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_pareto(self,
                    scored_results: List[Dict],
                    x: str = "fuel_cost_norm",
                    y: str = "erosion_risk_norm",
                    color: str = "disturbance_norm",
                    feasible_only: bool = False,
                    save_path: Optional[str] = None,
                    title: Optional[str] = None) -> None:
        """2-D scatter plot with Pareto front overlaid.

        Parameters
        ──────────
        x, y, color : keys from scored result dicts
        feasible_only : filter to feasible cases only in the background scatter
        save_path     : if given, save figure to this path instead of showing
        """
        pool = scored_results
        if feasible_only:
            pool = [r for r in scored_results if r.get("feasible", True)]

        if not pool:
            print("No data to plot.")
            return

        x_all   = np.array([r[x]     for r in pool])
        y_all   = np.array([r[y]     for r in pool])
        c_all   = np.array([r[color] for r in pool])

        front   = self.pareto_front(scored_results, feasible_only=True)
        x_front = np.array([r[x] for r in front])
        y_front = np.array([r[y] for r in front])

        fig, ax = plt.subplots(figsize=(9, 6))

        sc = ax.scatter(x_all, y_all, c=c_all, cmap="viridis_r",
                        s=30, alpha=0.6, zorder=2, label="All cases")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(_pretty_label(color), fontsize=10)

        # Pareto front markers
        if len(x_front) > 0:
            ax.scatter(x_front, y_front, c="red", s=80, marker="*",
                       zorder=5, label=f"Pareto front ({len(front)})")
            # Connect front points (sort by x for a clean line)
            idx_sort = np.argsort(x_front)
            ax.plot(x_front[idx_sort], y_front[idx_sort],
                    "r--", linewidth=1.0, alpha=0.7, zorder=4)

        ax.set_xlabel(_pretty_label(x), fontsize=11)
        ax.set_ylabel(_pretty_label(y), fontsize=11)
        ax.set_title(title or
                     f"Pareto Analysis — {self.manoeuvre_type}  "
                     f"(colour = {_pretty_label(color)})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Plot saved: {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_objectives_3d(self,
                           scored_results: List[Dict],
                           save_path: Optional[str] = None) -> None:
        """3-D scatter of the three normalised objectives with Pareto front."""
        from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers projection)

        pool  = [r for r in scored_results if r.get("feasible", True)]
        front = self.pareto_front(scored_results, feasible_only=True)

        if not pool:
            print("No feasible data to plot.")
            return

        fig  = plt.figure(figsize=(10, 7))
        ax   = fig.add_subplot(111, projection="3d")

        xp = [r["fuel_cost_norm"]    for r in pool]
        yp = [r["disturbance_norm"]  for r in pool]
        zp = [r["erosion_risk_norm"] for r in pool]
        ax.scatter(xp, yp, zp, c="steelblue", s=20, alpha=0.4, label="Feasible")

        if front:
            xf = [r["fuel_cost_norm"]    for r in front]
            yf = [r["disturbance_norm"]  for r in front]
            zf = [r["erosion_risk_norm"] for r in front]
            ax.scatter(xf, yf, zf, c="red", s=80, marker="*",
                       zorder=5, label=f"Pareto ({len(front)})")

        ax.set_xlabel("Fuel cost (norm)", fontsize=9)
        ax.set_ylabel("Disturbance (norm)", fontsize=9)
        ax.set_zlabel("Erosion risk (norm)", fontsize=9)
        ax.set_title(f"3-D Objective Space — {self.manoeuvre_type}", fontsize=11)
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Plot saved: {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_heatmap(self,
                     scored_results: List[Dict],
                     x_param: str = "shoulder_yaw_deg",
                     y_param: str = "arm_reach_m",
                     metric: str = "pareto_score",
                     save_path: Optional[str] = None) -> None:
        """Heatmap of `metric` over a 2-D parameter grid.

        Works best when the sweep contains a regular grid of x_param × y_param.
        Cells with no data are shown as white.
        """
        x_vals = sorted(set(r[x_param] for r in scored_results if x_param in r))
        y_vals = sorted(set(r[y_param] for r in scored_results if y_param in r))

        if len(x_vals) < 2 or len(y_vals) < 2:
            print(f"Not enough grid points for heatmap "
                  f"({len(x_vals)} × {len(y_vals)}). Use plot_pareto instead.")
            return

        # Build grid
        grid = np.full((len(y_vals), len(x_vals)), np.nan)
        xi   = {v: i for i, v in enumerate(x_vals)}
        yi   = {v: i for i, v in enumerate(y_vals)}
        for r in scored_results:
            if x_param not in r or y_param not in r or metric not in r:
                continue
            i, j = yi.get(r[y_param]), xi.get(r[x_param])
            if i is not None and j is not None:
                grid[i, j] = r[metric]

        fig, ax = plt.subplots(figsize=(max(8, len(x_vals)), max(5, len(y_vals)//2)))
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       cmap="RdYlGn_r", interpolation="nearest",
                       extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]])
        fig.colorbar(im, ax=ax, label=_pretty_label(metric))

        # Overlay Pareto stars
        front = self.pareto_front(scored_results, feasible_only=True)
        for r in front:
            if x_param in r and y_param in r:
                ax.plot(r[x_param], r[y_param], "w*", markersize=12, zorder=5)

        ax.set_xlabel(_pretty_label(x_param), fontsize=11)
        ax.set_ylabel(_pretty_label(y_param), fontsize=11)
        ax.set_title(
            f"Heatmap: {_pretty_label(metric)} — {self.manoeuvre_type}\n"
            f"(★ = Pareto-optimal)", fontsize=11)
        ax.set_xticks(x_vals)
        ax.set_yticks(y_vals)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Plot saved: {save_path}")
        else:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, scored_results: List[Dict],
                   filepath: str,
                   pareto_only: bool = False) -> None:
        """Export scored results to CSV.

        Parameters
        ──────────
        pareto_only : if True, export only Pareto-optimal feasible cases
        """
        rows = scored_results
        if pareto_only:
            rows = self.pareto_front(scored_results, feasible_only=True)
        if not rows:
            print("No rows to export.")
            return

        # Ensure all rows have the same keys (fill missing with empty string)
        all_keys = list(rows[0].keys())
        with open(filepath, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        label = "Pareto-optimal" if pareto_only else "all"
        print(f"Exported {len(rows)} {label} cases → {filepath}")

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def _flag_feasibility(self, results: List[Dict]) -> None:
        """Add `feasible` and `feasibility_reason` keys in-place."""
        dev_key = self._dev_key
        budget  = self.angle_budget_deg

        for r in results:
            reason = ""

            # Gate 1: pipeline status — FAIL means geometry makes erosion certain
            if r.get("status") == "FAIL":
                reason = "panel erosion FAIL"

            # Gate 2: thrust angle budget (spec §5.1 – soft budget 45–50°)
            elif r.get(dev_key, 0.0) > budget:
                reason = (f"{self.manoeuvre_type} deviation "
                          f"{r[dev_key]:.1f}° > budget {budget:.0f}°")

            r["feasible"]           = (reason == "")
            r["feasibility_reason"] = reason


# ---------------------------------------------------------------------------
# 4.  LABEL HELPER
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    "fuel_cost":           "Fuel cost  (1−cosθ)",
    "fuel_cost_norm":      "Fuel cost (normalised)",
    "disturbance_raw":     "Disturbance torque [N·m]",
    "disturbance_norm":    "Disturbance torque (normalised)",
    "erosion_risk_raw":    "Erosion risk (raw)",
    "erosion_risk_norm":   "Erosion risk (normalised)",
    "pareto_score":        "Pareto score  (lower = better)",
    "nssk_deviation_deg":  "NSSK thrust deviation [°]",
    "ewsk_deviation_deg":  "EWSK thrust deviation [°]",
    "nssk_torque_Nm":      "NSSK disturbance torque [N·m]",
    "ewsk_torque_Nm":      "EWSK disturbance torque [N·m]",
    "erosion_fraction":    "Panel erosion fraction",
    "ant_max_erosion_um":  "Max antenna erosion [µm]",
    "shoulder_yaw_deg":    "Shoulder yaw [°]",
    "arm_reach_m":         "Arm reach [m]",
    "link_ratio":          "Link ratio (L1/reach)",
    "thruster_cog_distance_m": "Thruster–CoG distance [m]",
    "nssk_moment_arm_m":   "NSSK moment arm [m]",
    "ewsk_moment_arm_m":   "EWSK moment arm [m]",
}

def _pretty_label(key: str) -> str:
    return _LABEL_MAP.get(key, key.replace("_", " "))


# ---------------------------------------------------------------------------
# 5.  QUICK-RUN HELPER
# ---------------------------------------------------------------------------

def run_pareto_analysis(cases: List[Dict],
                        manoeuvre_type: str = "NSSK",
                        angle_budget_deg: float = 50.0,
                        output_dir: Optional[str] = None) -> Dict:
    """Full pipeline: sweep → score → Pareto → plots + CSV.

    Parameters
    ──────────
    cases          : list of case dicts (from CaseMatrixGenerator or manual)
    manoeuvre_type : "NSSK" or "EWSK"
    angle_budget_deg : feasibility gate
    output_dir     : if given, saves CSV + plots here

    Returns
    ───────
    Dict with keys:
      "all"    : full scored results list
      "front"  : Pareto-optimal subset
      "scorer" : ParetoScorer instance
    """
    import os
    from plume_impingement_pipeline import PlumePipeline

    pipeline = PlumePipeline()
    print(f"Running sweep: {len(cases)} cases …")
    results = pipeline.run_sweep(cases, verbose=True)

    scorer  = ParetoScorer(manoeuvre_type=manoeuvre_type,
                           angle_budget_deg=angle_budget_deg)
    scored  = scorer.score(results)
    front   = scorer.pareto_front(scored)
    scorer.summary(scored)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        scorer.export_csv(scored,
                          os.path.join(output_dir, "sweep_scored.csv"))
        scorer.export_csv(scored,
                          os.path.join(output_dir, "pareto_front.csv"),
                          pareto_only=True)
        scorer.plot_pareto(
            scored, x="fuel_cost_norm", y="erosion_risk_norm",
            color="disturbance_norm",
            save_path=os.path.join(output_dir, "pareto_2d.png"))
        scorer.plot_objectives_3d(
            scored,
            save_path=os.path.join(output_dir, "pareto_3d.png"))
        scorer.plot_heatmap(
            scored, x_param="shoulder_yaw_deg", y_param="arm_reach_m",
            metric="pareto_score",
            save_path=os.path.join(output_dir, "heatmap_score.png"))
        scorer.plot_heatmap(
            scored, x_param="shoulder_yaw_deg", y_param="arm_reach_m",
            metric="nssk_deviation_deg" if manoeuvre_type == "NSSK"
                   else "ewsk_deviation_deg",
            save_path=os.path.join(output_dir, "heatmap_deviation.png"))

    return {"all": scored, "front": front, "scorer": scorer}
