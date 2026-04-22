"""
feasibility_map.py
==================
Multi-epoch feasibility map generator for station-keeping arm poses.

Implements spec §7 steps 5–7 and §8–9.

Orchestrates:
  1. Joint-space grid construction
  2. Static per-cell geometry (p_nozzle, t_hat) — computed once
  3. F_kin collision mask — computed once, cached
  4. Per-epoch, per-direction feasibility (F_align ∩ F_CoG ∩ F_kin)
  5. Persistent feasibility = intersection across all epochs
  6. Annotation layers (alpha, r_miss per epoch)
  7. Diagnostics (CoG trajectory, cell counts, first-dropout, binding constraints, N/S asymmetry)

F_plume is not implemented in v1 (CEX model deferred per spec §5.2).

Usage
-----
  from plume_impingement_pipeline import RoboticArmGeometry, StackConfig
  from composite_mass_model import CompositeMassModel
  from feasibility_cells import FeasibilityConfig
  from feasibility_map import build_feasibility_maps, compute_pivot

  arm   = RoboticArmGeometry()
  stack = StackConfig()
  pivot = compute_pivot(arm, stack, servicer_yaw_deg=-25.0)
  mass  = CompositeMassModel.from_json(stack=stack)
  cfg   = FeasibilityConfig.from_json()
  n_hat = np.array([0.0, 0.0, 1.0])   # from feasibility_inputs.json

  results = build_feasibility_maps(arm, mass, stack, pivot, n_hat,
                                   servicer_yaw_deg=-25.0, config=cfg)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from plume_impingement_pipeline import RoboticArmGeometry, StackConfig
from composite_mass_model import CompositeMassModel
from feasibility_cells import (
    FeasibilityConfig,
    SK_DIRECTIONS,
    build_joint_grid,
    compute_static_cell_quantities,
    compute_F_kin,
    compute_alpha,
    compute_r_miss,
    compute_F_align,
    compute_F_CoG,
    binding_constraint_breakdown,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityMapResult:
    """Feasibility map and annotations for one SK direction.

    Shapes use (K, N0, N1, N2) where K = number of epochs and
    (N0, N1, N2) = grid_resolution from FeasibilityConfig.
    """
    direction: str

    F_per_epoch: np.ndarray       # bool  (K, N0, N1, N2)
    F_persistent: np.ndarray      # bool  (N0, N1, N2)

    alpha_map: np.ndarray         # float (N0, N1, N2), NaN where F_kin fails
    r_miss_per_epoch: np.ndarray  # float (K, N0, N1, N2), NaN where F_total fails

    # erosion_proxy_per_epoch: deferred to v2 (F_plume / CEX model not yet implemented)

    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pivot helper
# ---------------------------------------------------------------------------

def compute_pivot(arm: RoboticArmGeometry,
                  stack: StackConfig,
                  servicer_yaw_deg: float = 0.0) -> np.ndarray:
    """Compute arm pivot position in LAR frame.

    The pivot is on the servicer +Z face (the face closest to the LAR
    interface). The pivot offset is expressed in the servicer body frame
    and rotated into LAR by the servicer yaw.
    """
    origin = stack.servicer_origin_in_lar_frame()
    c, s   = np.cos(np.radians(servicer_yaw_deg)), np.sin(np.radians(servicer_yaw_deg))
    Rz     = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    offset_body = np.array([arm.pivot_offset_x,
                             arm.pivot_offset_y,
                             stack.servicer_bus_z / 2.0 + arm.pivot_offset_z])
    return origin + Rz @ offset_body


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_feasibility_maps(
    arm: RoboticArmGeometry,
    mass_model: CompositeMassModel,
    stack: StackConfig,
    pivot: np.ndarray,
    n_hat_ee: np.ndarray,
    servicer_yaw_deg: float,
    config: FeasibilityConfig,
    directions: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, FeasibilityMapResult]:
    """Generate feasibility maps for all SK directions across mission epochs.

    Parameters
    ----------
    arm              : arm geometry (link lengths, joint limits)
    mass_model       : mission-epoch CoG model
    stack            : servicer+client geometry
    pivot            : arm base in LAR frame (use compute_pivot())
    n_hat_ee         : fixed nozzle exit direction in EE frame (from feasibility_inputs.json)
    servicer_yaw_deg : servicer body yaw relative to LAR frame [deg]
    config           : grid resolution, thresholds, epoch schedule
    directions       : subset of ['N', '-N', 'E', '-W']; default = all four
    verbose          : print progress

    Returns
    -------
    dict mapping direction string → FeasibilityMapResult
    """
    if directions is None:
        directions = list(SK_DIRECTIONS.keys())

    epochs  = config.epoch_schedule_days
    K       = len(epochs)
    res     = config.grid_resolution
    t0_wall = time.time()

    # ------------------------------------------------------------------
    # Step 1: joint grid
    # ------------------------------------------------------------------
    if verbose:
        print(f"Building {res[0]}×{res[1]}×{res[2]} joint-space grid "
              f"({np.prod(res):,} cells)...")
    q0g, q1g, q2g = build_joint_grid(arm, resolution=res)

    # ------------------------------------------------------------------
    # Step 2: static per-cell quantities (vectorized, one shot)
    # ------------------------------------------------------------------
    if verbose:
        print("Computing static cell quantities (FK, thrust axis)...")
    cq = compute_static_cell_quantities(arm, pivot, n_hat_ee, q0g, q1g, q2g)
    p_nozzle = cq['p_nozzle']   # (N0,N1,N2,3)
    t_hat    = cq['t_hat']      # (N0,N1,N2,3)

    # ------------------------------------------------------------------
    # Step 3: F_kin (computed once, shared across all directions/epochs)
    # ------------------------------------------------------------------
    if verbose:
        print("Computing F_kin (collision check)...")
    F_kin = compute_F_kin(arm, pivot, stack, servicer_yaw_deg, cq, q0g, q1g, q2g, verbose=verbose)

    # ------------------------------------------------------------------
    # CoG trajectory across epochs (for diagnostics and per-epoch filters)
    # ------------------------------------------------------------------
    cog_trajectory = np.array([mass_model.p_CoG_LAR(tau) for tau in epochs])  # (K, 3)
    cog_migration  = np.array([mass_model.cog_migration_magnitude(tau) for tau in epochs])

    # ------------------------------------------------------------------
    # Steps 4–6: per-direction, per-epoch feasibility + annotations
    # ------------------------------------------------------------------
    results: dict[str, FeasibilityMapResult] = {}

    for direction in directions:
        d_hat = SK_DIRECTIONS[direction]
        if verbose:
            print(f"\n--- Direction {direction:3s} ({d_hat}) ---")

        # Alpha is epoch-independent (geometry only)
        alpha = compute_alpha(t_hat, d_hat)                  # (N0,N1,N2)
        alpha_map = np.where(F_kin, alpha, np.nan)           # NaN outside F_kin

        F_per_epoch      = np.zeros((K, *res), dtype=bool)
        r_miss_per_epoch = np.full((K, *res), np.nan)

        breakdown_per_epoch: list[dict] = []
        cell_counts: list[int] = []

        for k, tau in enumerate(epochs):
            p_CoG = cog_trajectory[k]

            r_miss  = compute_r_miss(p_nozzle, t_hat, p_CoG)
            F_align = compute_F_align(alpha, config.alpha_max_rad)
            F_CoG   = compute_F_CoG(r_miss, config.eps_CoG_m)
            F_total = F_kin & F_align & F_CoG

            F_per_epoch[k] = F_total
            # Store r_miss only where feasible (NaN elsewhere, per spec §8)
            r_miss_per_epoch[k] = np.where(F_total, r_miss, np.nan)

            n_feas = int(F_total.sum())
            cell_counts.append(n_feas)
            breakdown_per_epoch.append(binding_constraint_breakdown(F_kin, F_align, F_CoG))

            if verbose:
                bd = breakdown_per_epoch[k]
                print(f"  τ={tau:5.0f}d  feasible={n_feas:6d}  "
                      f"kin={bd['frac_fail_kin']*100:.1f}%  "
                      f"align={bd['frac_fail_align']*100:.1f}%  "
                      f"CoG={bd['frac_fail_CoG']*100:.1f}% eliminated")

        # Persistent feasibility = AND across all epochs
        F_persistent = np.all(F_per_epoch, axis=0)
        n_persistent = int(F_persistent.sum())
        if verbose:
            if n_persistent == 0:
                print(f"  *** F_persistent is EMPTY for direction {direction} ***")
            else:
                print(f"  F_persistent: {n_persistent} cells")

        # First-dropout epoch per cell: for cells in F_per_epoch[0] not in F_persistent
        first_dropout = _compute_first_dropout(F_per_epoch, epochs)

        diagnostics = {
            'epoch_schedule_days':    epochs,
            'cell_counts_per_epoch':  cell_counts,
            'binding_constraint':     breakdown_per_epoch,
            'first_dropout_epoch':    first_dropout,
            'F_persistent_empty':     n_persistent == 0,
        }

        results[direction] = FeasibilityMapResult(
            direction=direction,
            F_per_epoch=F_per_epoch,
            F_persistent=F_persistent,
            alpha_map=alpha_map,
            r_miss_per_epoch=r_miss_per_epoch,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Step 7: global diagnostics (shared across directions)
    # ------------------------------------------------------------------
    global_diag = _compute_global_diagnostics(
        results, mass_model, config, cog_trajectory, cog_migration, verbose
    )
    for res_obj in results.values():
        res_obj.diagnostics['global'] = global_diag

    elapsed = time.time() - t0_wall
    if verbose:
        print(f"\nDone in {elapsed:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def _compute_first_dropout(
    F_per_epoch: np.ndarray,    # (K, N0, N1, N2) bool
    epochs: list[float],
) -> np.ndarray:
    """For each cell feasible at BOL, return the first epoch day at which it fails.

    Returns float array (N0, N1, N2):
      - inf  : cell is in F_persistent (never drops out)
      - epoch day : first epoch where cell fails
      - NaN  : cell was already infeasible at BOL (epoch 0)
    """
    K = F_per_epoch.shape[0]
    shape = F_per_epoch.shape[1:]
    result = np.full(shape, np.inf)

    bol_feasible = F_per_epoch[0]
    result[~bol_feasible] = np.nan   # infeasible from the start

    for k in range(K - 1, -1, -1):  # iterate in reverse so earlier epochs overwrite
        dropped = bol_feasible & ~F_per_epoch[k]
        result[dropped] = float(epochs[k])

    return result


def _compute_global_diagnostics(
    results: dict[str, FeasibilityMapResult],
    mass_model: CompositeMassModel,
    config: FeasibilityConfig,
    cog_trajectory: np.ndarray,   # (K, 3)
    cog_migration: np.ndarray,    # (K,)
    verbose: bool,
) -> dict:
    """Compute diagnostics that span all directions (spec §9)."""
    epochs = config.epoch_schedule_days

    # 1. CoG migration trajectory
    cog_diag = {
        'epochs_days':          epochs,
        'cog_positions_LAR':    cog_trajectory.tolist(),
        'cog_migration_m':      cog_migration.tolist(),
        'cog_migration_eol_cm': float(cog_migration[-1] * 100),
    }

    # 2. Per-direction cell counts (already in per-direction diagnostics, summarised here)
    counts = {d: results[d].diagnostics['cell_counts_per_epoch'] for d in results}

    # 3. N/S asymmetry: compare |F_N(τ)| vs |F_-N(τ)| at each epoch
    ns_asymmetry = None
    if 'N' in results and '-N' in results:
        count_N  = results['N'].diagnostics['cell_counts_per_epoch']
        count_mN = results['-N'].diagnostics['cell_counts_per_epoch']
        ns_asymmetry = {
            'count_N':  count_N,
            'count_mN': count_mN,
            'ratio_N_over_mN': [
                (n / m if m > 0 else float('inf'))
                for n, m in zip(count_N, count_mN)
            ],
        }

    # 4. Suggested epoch resampling spacing
    suggested_spacing = mass_model.suggested_epoch_spacing(config.eps_CoG_m)

    if verbose:
        print(f"\n--- Global diagnostics ---")
        print(f"  CoG migration at EOL: {cog_migration[-1]*100:.2f} cm")
        print(f"  Suggested epoch spacing for eps={config.eps_CoG_m*100:.0f} cm: "
              f"{suggested_spacing:.1f} days")
        if ns_asymmetry:
            ratio = ns_asymmetry['ratio_N_over_mN']
            print(f"  N/S asymmetry (count N / count -N) at BOL: {ratio[0]:.2f}, "
                  f"at EOL: {ratio[-1]:.2f}")

    return {
        'cog_trajectory':          cog_diag,
        'cell_counts_per_direction': counts,
        'ns_asymmetry':            ns_asymmetry,
        'suggested_epoch_spacing_days': suggested_spacing,
    }


# ---------------------------------------------------------------------------
# Convenience: print a summary report
# ---------------------------------------------------------------------------

def print_summary(results: dict[str, FeasibilityMapResult]) -> None:
    """Print a human-readable summary of all direction results."""
    any_result = next(iter(results.values()))
    epochs = any_result.diagnostics['epoch_schedule_days']
    global_d = any_result.diagnostics.get('global', {})

    print("=" * 70)
    print("FEASIBILITY MAP SUMMARY")
    print("=" * 70)

    cog_d = global_d.get('cog_trajectory', {})
    if cog_d:
        print(f"CoG migration: BOL→EOL = {cog_d['cog_migration_eol_cm']:.2f} cm  "
              f"({len(epochs)} epochs: {[f'{e:.0f}d' for e in epochs]})")

    for direction, res in results.items():
        counts = res.diagnostics['cell_counts_per_epoch']
        n_persist = int(res.F_persistent.sum())
        N_total = res.F_per_epoch.shape[1] * res.F_per_epoch.shape[2] * res.F_per_epoch.shape[3]
        print(f"\n  {direction:4s}: persistent={n_persist:6d}/{N_total}  "
              f"({100*n_persist/N_total:.1f}%)  "
              f"cell counts: {counts}")
        if res.diagnostics.get('F_persistent_empty'):
            print(f"        *** WARNING: F_persistent is EMPTY ***")

    ns = global_d.get('ns_asymmetry')
    if ns:
        r = ns['ratio_N_over_mN']
        print(f"\n  N/S asymmetry (N÷−N): BOL={r[0]:.2f}  EOL={r[-1]:.2f}")

    sug = global_d.get('suggested_epoch_spacing_days')
    if sug:
        print(f"  Suggested epoch spacing: {sug:.1f} days")
    print("=" * 70)
