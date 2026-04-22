"""
feasibility_cells.py
====================
Per-cell geometric quantities and feasibility filters for the joint-space grid.

Implements spec §4, §5.1–5.2 (F_kin, F_align, F_CoG), and §7 steps 1–4.
F_plume is not implemented in v1 (CEX model deferred).

Frame convention: LAR frame throughout (+X=North, +Y=East, +Z=Nadir).

EE frame axes in LAR frame (derived from arm_fk_transforms chain):
  X_EE_LAR = d_bracket = cos(q1+q2)*u_rad + sin(q1+q2)*Z_hat   (along bracket)
  Y_EE_LAR = [-sin(q0), cos(q0), 0]                              (yaw-perpendicular)
  Z_EE_LAR = -sin(q1+q2)*u_rad + cos(q1+q2)*Z_hat               (perpendicular to bracket)

Thrust axis: t̂_LAR = R_EE_LAR @ (-n̂^E) = -n[0]*X_EE - n[1]*Y_EE - n[2]*Z_EE

Station-keeping directions in LAR frame (spec §2 translated to LAR):
  +N = +X_LAR,  -N = -X_LAR,  +E = +Y_LAR,  -W = -Y_LAR

Integration order (spec §11.3):
  Step 2 — compute_static_cell_quantities() : p_nozzle, t_hat (vectorized, one shot)
  Step 3 — compute_F_kin()                  : collision wrapper (iterative over grid)
  Step 4 — compute_alpha(), compute_r_miss(), compute_F_align(), compute_F_CoG()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from plume_impingement_pipeline import RoboticArmGeometry, StackConfig, arm_has_collision

_Z_HAT = np.array([0.0, 0.0, 1.0])

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

SK_DIRECTIONS: dict[str, np.ndarray] = {
    "N": np.array([ 1.0, 0.0, 0.0]),
    "-N": np.array([-1.0, 0.0, 0.0]),
    "E": np.array([ 0.0, 1.0, 0.0]),
    "-W": np.array([ 0.0,-1.0, 0.0]),
}


@dataclass
class FeasibilityConfig:
    """Thresholds and grid settings for the feasibility map.

    Load from feasibility_inputs.json via FeasibilityConfig.from_json().
    """
    eps_CoG_m: float = 0.05
    alpha_max_deg: float = 5.0
    grid_resolution: tuple[int, int, int] = (50, 50, 50)
    epoch_schedule_days: list[float] = field(default_factory=lambda: [0, 456, 913, 1369, 1825])

    @property
    def alpha_max_rad(self) -> float:
        return np.radians(self.alpha_max_deg)

    @classmethod
    def from_json(cls, path: str = "feasibility_inputs.json") -> "FeasibilityConfig":
        here = os.path.dirname(os.path.abspath(__file__))
        fpath = path if os.path.isabs(path) else os.path.join(here, path)
        with open(fpath) as f:
            cfg = json.load(f)
        return cls(
            eps_CoG_m=float(cfg.get("eps_CoG_m", 0.05)),
            alpha_max_deg=float(cfg.get("alpha_max_deg", 5.0)),
            grid_resolution=tuple(cfg.get("grid_resolution", [50, 50, 50])),
            epoch_schedule_days=cfg.get("epoch_schedule_days", [0, 456, 913, 1369, 1825]),
        )


# ---------------------------------------------------------------------------
# Step 1: Joint-space grid
# ---------------------------------------------------------------------------

def build_joint_grid(arm: RoboticArmGeometry,
                     resolution: tuple[int, int, int] = (50, 50, 50)
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (q0, q1, q2) meshgrids in radians, shape (N0, N1, N2) each.

    Grid spans the full joint limit range from arm.q*_min_deg to arm.q*_max_deg.
    Indexing='ij' so axis 0 = q0, axis 1 = q1, axis 2 = q2.
    """
    N0, N1, N2 = resolution
    q0 = np.radians(np.linspace(arm.q0_min_deg, arm.q0_max_deg, N0))
    q1 = np.radians(np.linspace(arm.q1_min_deg, arm.q1_max_deg, N1))
    q2 = np.radians(np.linspace(arm.q2_min_deg, arm.q2_max_deg, N2))
    return np.meshgrid(q0, q1, q2, indexing='ij')


# ---------------------------------------------------------------------------
# Step 2: Static per-cell quantities (vectorized)
# ---------------------------------------------------------------------------

def compute_static_cell_quantities(
    arm: RoboticArmGeometry,
    pivot: np.ndarray,
    n_hat_ee: np.ndarray,
    q0_grid: np.ndarray,
    q1_grid: np.ndarray,
    q2_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute p_nozzle_LAR and t_hat_LAR for every cell in the grid.

    Fully vectorized over the (N0, N1, N2) grid — no Python loops.

    Parameters
    ----------
    arm       : arm geometry (link lengths)
    pivot     : arm base in LAR frame, shape (3,)
    n_hat_ee  : fixed nozzle exit direction in EE frame, shape (3,)
                Plume exits along n̂^E; thrust axis = R_EE_LAR @ (-n̂^E).
    q0/q1/q2_grid : joint angle meshgrids in radians, shape (N0, N1, N2)

    Returns
    -------
    dict with:
      'p_elbow'    : (N0, N1, N2, 3) — elbow joint position in LAR
      'p_wrist'    : (N0, N1, N2, 3) — wrist joint position in LAR
      'p_nozzle'   : (N0, N1, N2, 3) — thruster exit position in LAR
      't_hat'      : (N0, N1, N2, 3) — thrust axis unit vector in LAR
      'd_bracket'  : (N0, N1, N2, 3) — bracket direction (EE X-axis) in LAR
    """
    n_hat_ee = np.asarray(n_hat_ee, dtype=float)
    nx, ny, nz = n_hat_ee[0], n_hat_ee[1], n_hat_ee[2]

    L1 = arm.link1_length
    L2 = arm.link2_length
    L3 = arm.bracket_length

    # Expand for broadcasting: (N0,1,1) and (1,N1,N2)
    c0 = np.cos(q0_grid)[..., np.newaxis]    # (N0,N1,N2,1) after below
    s0 = np.sin(q0_grid)[..., np.newaxis]
    q12 = q1_grid + q2_grid                  # (N0,N1,N2)
    c1  = np.cos(q1_grid)[..., np.newaxis]
    s1  = np.sin(q1_grid)[..., np.newaxis]
    c12 = np.cos(q12)[..., np.newaxis]       # (N0,N1,N2,1) for broadcasting with (3,)
    s12 = np.sin(q12)[..., np.newaxis]

    # Horizontal yaw direction: u_rad = [cos(q0), sin(q0), 0], shape (N0,N1,N2,3)
    zeros = np.zeros_like(c0)
    u_rad = np.concatenate([c0, s0, zeros], axis=-1)  # (N0,N1,N2,3)

    # EE frame axes in LAR (see module docstring for derivation)
    d_link2   = c1  * u_rad + s1  * _Z_HAT   # link-2 direction (N0,N1,N2,3)
    d_bracket = c12 * u_rad + s12 * _Z_HAT   # X_EE_LAR (N0,N1,N2,3)
    y_ee = np.concatenate([-s0, c0, zeros], axis=-1)   # Y_EE_LAR (N0,N1,N2,3)
    z_ee = -s12 * u_rad + c12 * _Z_HAT                 # Z_EE_LAR (N0,N1,N2,3)

    # Arm positions in LAR frame
    p_elbow  = pivot + L1 * u_rad
    p_wrist  = p_elbow + L2 * d_link2
    p_nozzle = p_wrist + L3 * d_bracket

    # Thrust axis: t̂ = R_EE_LAR @ (-n̂^E) = -nx*X_EE - ny*Y_EE - nz*Z_EE
    t_hat = -nx * d_bracket - ny * y_ee - nz * z_ee
    # Normalize (should already be unit length for unit n̂^E, but floating-point safety)
    t_norm = np.linalg.norm(t_hat, axis=-1, keepdims=True)
    t_hat = t_hat / np.where(t_norm > 0, t_norm, 1.0)

    return {
        'p_elbow':   p_elbow,
        'p_wrist':   p_wrist,
        'p_nozzle':  p_nozzle,
        't_hat':     t_hat,
        'd_bracket': d_bracket,
    }


# ---------------------------------------------------------------------------
# Step 3: F_kin — collision check wrapper
# ---------------------------------------------------------------------------

def compute_F_kin(
    arm: RoboticArmGeometry,
    pivot: np.ndarray,
    stack: StackConfig,
    servicer_yaw_deg: float,
    cell_quants: dict[str, np.ndarray],
    q0_grid: np.ndarray,
    q1_grid: np.ndarray,
    q2_grid: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Compute the kinematic feasibility mask F_kin over the full grid.

    Calls arm_has_collision() once per cell (direction- and epoch-independent).
    A cell passes F_kin if:
      - All joint angles are within arm limits, AND
      - No arm segment collides with any obstacle (client bus, servicer bus,
        servicer panels, antenna dishes).

    Returns
    -------
    F_kin : bool array, shape (N0, N1, N2). True = feasible (no collision).

    Performance note: iterates over all cells (~125k for 50^3). Expect 30–120 s
    depending on hardware. F_kin is epoch- and direction-independent, so it is
    computed once and cached.
    """
    shape = q0_grid.shape
    N_total = q0_grid.size
    F_kin = np.ones(shape, dtype=bool)

    p_elbow  = cell_quants['p_elbow']
    p_wrist  = cell_quants['p_wrist']
    p_nozzle = cell_quants['p_nozzle']

    if verbose:
        print(f"  Computing F_kin over {N_total} cells...", end="", flush=True)

    it = np.nditer([q0_grid, q1_grid, q2_grid], flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        q0v, q1v, q2v = float(it[0]), float(it[1]), float(it[2])

        # Joint limit check (early exit before geometry)
        if not arm.within_joint_limits(q0v, q1v, q2v):
            F_kin[idx] = False
            it.iternext()
            continue

        if arm_has_collision(
            pivot,
            p_elbow[idx],
            p_wrist[idx],
            p_nozzle[idx],
            stack,
            servicer_yaw_deg=servicer_yaw_deg,
        ):
            F_kin[idx] = False

        it.iternext()

    if verbose:
        n_feas = int(F_kin.sum())
        print(f" done. {n_feas}/{N_total} cells pass ({100*n_feas/N_total:.1f}%)")

    return F_kin


# ---------------------------------------------------------------------------
# Step 4: F_align and F_CoG filters
# ---------------------------------------------------------------------------

def compute_alpha(t_hat_grid: np.ndarray, d_hat: np.ndarray) -> np.ndarray:
    """Alignment angle α(q, d̂) = arccos(clip(t̂·d̂, -1, 1)) [radians].

    Parameters
    ----------
    t_hat_grid : (N0, N1, N2, 3) — thrust axis per cell
    d_hat      : (3,) — desired SK direction unit vector in LAR frame

    Returns
    -------
    alpha : (N0, N1, N2) float array [radians]
    """
    d_hat = np.asarray(d_hat, dtype=float)
    dot = np.einsum('...i,i->...', t_hat_grid, d_hat)   # (N0,N1,N2)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def compute_F_align(alpha: np.ndarray, alpha_max_rad: float) -> np.ndarray:
    """Return bool mask: True where α ≤ α_max."""
    return alpha <= alpha_max_rad


def compute_r_miss(p_nozzle_grid: np.ndarray,
                   t_hat_grid: np.ndarray,
                   p_CoG_LAR: np.ndarray) -> np.ndarray:
    """Miss-distance r_miss = ‖(p_CoG − p_nozzle) × t̂‖ [m].

    Cross-product form is numerically stable when r_miss is small.

    Parameters
    ----------
    p_nozzle_grid : (N0, N1, N2, 3)
    t_hat_grid    : (N0, N1, N2, 3)
    p_CoG_LAR     : (3,) — CoG at current epoch in LAR frame

    Returns
    -------
    r_miss : (N0, N1, N2) float array [m]
    """
    p_CoG_LAR = np.asarray(p_CoG_LAR, dtype=float)
    r_vec = p_CoG_LAR - p_nozzle_grid                       # (N0,N1,N2,3)
    cross = np.cross(r_vec, t_hat_grid)                      # (N0,N1,N2,3)
    return np.linalg.norm(cross, axis=-1)                    # (N0,N1,N2)


def compute_F_CoG(r_miss: np.ndarray, eps_CoG_m: float) -> np.ndarray:
    """Return bool mask: True where r_miss ≤ ε_CoG."""
    return r_miss <= eps_CoG_m


# ---------------------------------------------------------------------------
# Combined single-epoch feasibility (without F_plume in v1)
# ---------------------------------------------------------------------------

def compute_feasibility_epoch(
    F_kin: np.ndarray,
    t_hat_grid: np.ndarray,
    p_nozzle_grid: np.ndarray,
    p_CoG_LAR: np.ndarray,
    d_hat: np.ndarray,
    config: FeasibilityConfig,
) -> dict[str, np.ndarray]:
    """Compute F_d(τ) = F_kin ∩ F_align(d) ∩ F_CoG(τ) for one direction and epoch.

    F_plume is omitted in v1 (CEX model deferred); treat as all-True.

    Returns
    -------
    dict with boolean masks and annotation scalars (all shape (N0, N1, N2)):
      'F_total', 'F_align', 'F_CoG', 'alpha', 'r_miss'
    """
    alpha   = compute_alpha(t_hat_grid, d_hat)
    r_miss  = compute_r_miss(p_nozzle_grid, t_hat_grid, p_CoG_LAR)
    F_align = compute_F_align(alpha, config.alpha_max_rad)
    F_CoG   = compute_F_CoG(r_miss, config.eps_CoG_m)
    F_total = F_kin & F_align & F_CoG

    return {
        'F_total':  F_total,
        'F_align':  F_align,
        'F_CoG':    F_CoG,
        'alpha':    alpha,
        'r_miss':   r_miss,
    }


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def binding_constraint_breakdown(F_kin: np.ndarray,
                                  F_align: np.ndarray,
                                  F_CoG: np.ndarray) -> dict[str, float]:
    """Fraction of cells failing each constraint (among cells that fail F_total).

    A cell is counted under whichever constraint eliminates it; if multiple
    constraints fail, it is counted once per failing constraint.

    Returns dict of fractions (0–1) relative to total grid size.
    """
    N = F_kin.size
    infeasible = ~(F_kin & F_align & F_CoG)
    return {
        'frac_fail_kin':   float((~F_kin).sum()) / N,
        'frac_fail_align': float((~F_align).sum()) / N,
        'frac_fail_CoG':   float((~F_CoG).sum()) / N,
        'frac_infeasible': float(infeasible.sum()) / N,
    }
