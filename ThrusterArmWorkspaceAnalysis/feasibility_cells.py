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


def _vrodrigues(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues rotation matrices for a fixed axis and grid of angles.

    Parameters
    ----------
    axis   : (3,) unit rotation axis
    angles : (...) array of rotation angles [rad]

    Returns
    -------
    R : (*angles.shape, 3, 3)  rotation matrix for each angle
    """
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    kx, ky, kz = k[0], k[1], k[2]

    c  = np.cos(angles)
    s  = np.sin(angles)
    mc = 1.0 - c

    R = np.zeros((*angles.shape, 3, 3))
    R[..., 0, 0] = c + mc * kx * kx
    R[..., 0, 1] = mc * kx * ky - s * kz
    R[..., 0, 2] = mc * kx * kz + s * ky
    R[..., 1, 0] = mc * ky * kx + s * kz
    R[..., 1, 1] = c + mc * ky * ky
    R[..., 1, 2] = mc * ky * kz - s * kx
    R[..., 2, 0] = mc * kz * kx - s * ky
    R[..., 2, 1] = mc * kz * ky + s * kx
    R[..., 2, 2] = c + mc * kz * kz
    return R

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
    tau_max_Nm: float = 50.0         # actuator torque limit [N·m] for F_torque filter

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
    servicer_yaw_deg: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compute p_nozzle_LAR and t_hat_LAR for every cell in the grid.

    Fully vectorized over the (N0, N1, N2) grid — no Python loops.
    Uses the general serial-chain FK with Rodrigues rotations.

    Parameters
    ----------
    arm              : arm geometry (joint axes, link body-frame vectors)
    pivot            : Hinge-1 position in LAR frame, shape (3,)
    n_hat_ee         : nozzle exit direction in link-3 body frame (= TA frame at q=0)
                       Plume exits along n̂; thrust on spacecraft is along −n̂.
    q0/q1/q2_grid    : joint angle meshgrids in radians, shape (N0, N1, N2)
    servicer_yaw_deg : servicer body yaw relative to LAR [deg]; rotates TA body
                       vectors to LAR before FK composition

    Returns
    -------
    dict with:
      'p_elbow'   : (N0, N1, N2, 3) — Hinge-2 (elbow) position in LAR
      'p_wrist'   : (N0, N1, N2, 3) — Hinge-3 (wrist) position in LAR
      'p_nozzle'  : (N0, N1, N2, 3) — thruster exit position in LAR
      't_hat'     : (N0, N1, N2, 3) — thrust axis (−nozzle direction) in LAR
      'd_bracket' : (N0, N1, N2, 3) — bracket direction (H3→nozzle unit vec) in LAR
    """
    n_hat = np.asarray(n_hat_ee, dtype=float)

    # Servicer-yaw rotation: TA body frame → LAR frame
    c_s, s_s = np.cos(np.radians(servicer_yaw_deg)), np.sin(np.radians(servicer_yaw_deg))
    Rz_s = np.array([[c_s, -s_s, 0.], [s_s, c_s, 0.], [0., 0., 1.]])

    # Vectorized Rodrigues rotation matrices over the joint-angle grids
    # Shape: (N0, N1, N2, 3, 3) for each
    R1_ta  = _vrodrigues(arm.axis1, q0_grid)   # joint-1 rotation in TA frame
    R2_ta  = _vrodrigues(arm.axis2, q1_grid)   # joint-2 rotation in TA frame
    R3_ta  = _vrodrigues(arm.axis3, q2_grid)   # joint-3 rotation in TA frame

    # Cumulative rotations in LAR frame: CR_i = Rz_s @ R1 @ ... @ Ri
    # np.einsum('ij,...jk->...ik', Rz_s, R) left-multiplies Rz_s onto each cell matrix
    R12_ta  = np.einsum('...ij,...jk->...ik', R1_ta,  R2_ta)   # R1 @ R2 in TA
    R123_ta = np.einsum('...ij,...jk->...ik', R12_ta, R3_ta)   # R1 @ R2 @ R3 in TA

    CR1   = np.einsum('ij,...jk->...ik', Rz_s, R1_ta)    # (N0,N1,N2,3,3) in LAR
    CR12  = np.einsum('ij,...jk->...ik', Rz_s, R12_ta)
    CR123 = np.einsum('ij,...jk->...ik', Rz_s, R123_ta)

    # Apply to link body-frame vectors → LAR-frame displacements per cell
    # np.einsum('...ij,j->...i', R, d) applies R[cell] to fixed vector d
    d1_lar = np.einsum('...ij,j->...i', CR1,   arm.d_h1h2)   # (N0,N1,N2,3)
    d2_lar = np.einsum('...ij,j->...i', CR12,  arm.d_h2h3)
    d3_lar = np.einsum('...ij,j->...i', CR123, arm.d_h3n)

    # Joint and nozzle positions in LAR
    p_elbow  = pivot + d1_lar                # (N0,N1,N2,3)
    p_wrist  = p_elbow + d2_lar
    p_nozzle = p_wrist + d3_lar

    # Bracket direction (Hinge-3 → nozzle, unit vector in LAR)
    d3_norm = np.linalg.norm(d3_lar, axis=-1, keepdims=True)
    d_bracket = d3_lar / np.where(d3_norm > 0, d3_norm, 1.0)

    # Thrust axis: spacecraft is pushed in direction opposite to nozzle exit
    # t̂ = −(CR123 @ n̂_body)
    t_hat = -np.einsum('...ij,j->...i', CR123, n_hat)
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
    link_radius: float = 0.08,
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
            link_radius=link_radius,
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
    tau_peak_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute F_d(τ) = F_kin ∩ F_align(d) ∩ F_CoG(τ) [∩ F_torque] for one epoch.

    F_plume is omitted in v1 (CEX model deferred); treat as all-True.
    F_torque is included when *tau_peak_grid* is provided.

    Parameters
    ----------
    tau_peak_grid : (N0, N1, N2) peak joint torque per cell [N·m], or None.
                    Produced by compute_torque_grid().  NaN entries (skipped
                    cells) are treated as zero torque.

    Returns
    -------
    dict with boolean masks and annotation scalars (all shape (N0, N1, N2)):
      'F_total', 'F_align', 'F_CoG', 'alpha', 'r_miss'
      and 'F_torque', 'tau_peak' when tau_peak_grid is not None.
    """
    alpha   = compute_alpha(t_hat_grid, d_hat)
    r_miss  = compute_r_miss(p_nozzle_grid, t_hat_grid, p_CoG_LAR)
    F_align = compute_F_align(alpha, config.alpha_max_rad)
    F_CoG   = compute_F_CoG(r_miss, config.eps_CoG_m)
    F_total = F_kin & F_align & F_CoG

    result: dict[str, np.ndarray] = {
        'F_total':  F_total,
        'F_align':  F_align,
        'F_CoG':    F_CoG,
        'alpha':    alpha,
        'r_miss':   r_miss,
    }

    if tau_peak_grid is not None:
        F_torque = compute_F_torque(tau_peak_grid, config.tau_max_Nm)
        result['F_torque'] = F_torque
        result['tau_peak'] = tau_peak_grid
        result['F_total']  = F_total & F_torque

    return result


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def binding_constraint_breakdown(
    F_kin: np.ndarray,
    F_align: np.ndarray,
    F_CoG: np.ndarray,
    F_torque: np.ndarray | None = None,
) -> dict[str, float]:
    """Fraction of cells failing each constraint relative to the total grid size.

    A cell is counted once per failing constraint (multi-counting allowed).

    Parameters
    ----------
    F_torque : optional actuator-torque feasibility mask from compute_F_torque().
    """
    N = F_kin.size
    F_total = F_kin & F_align & F_CoG
    if F_torque is not None:
        F_total = F_total & F_torque

    result = {
        'frac_fail_kin':   float((~F_kin).sum()) / N,
        'frac_fail_align': float((~F_align).sum()) / N,
        'frac_fail_CoG':   float((~F_CoG).sum()) / N,
        'frac_infeasible': float((~F_total).sum()) / N,
    }
    if F_torque is not None:
        result['frac_fail_torque'] = float((~F_torque).sum()) / N
    return result


# ---------------------------------------------------------------------------
# Step 5 (optional): F_torque — actuator torque feasibility filter
# ---------------------------------------------------------------------------

_DQ_SWEEP_DEFAULT  = np.array([0.02, 0.02, 0.01])   # rad/s  — slow deployment
_DDQ_SWEEP_DEFAULT = np.array([0.01, 0.01, 0.005])  # rad/s² — gentle acceleration


def compute_torque_grid(
    dyn,                            # ArmDynamics instance (typed loosely to avoid hard import)
    q0_grid: np.ndarray,
    q1_grid: np.ndarray,
    q2_grid: np.ndarray,
    F_kin: np.ndarray,
    dq_sweep: np.ndarray | None = None,
    ddq_sweep: np.ndarray | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Peak joint torque at every cell, evaluated via Pinocchio RNEA.

    Only cells where F_kin is True are evaluated (others receive NaN).
    Torques are computed for a single representative motion state
    (dq_sweep, ddq_sweep) modelling a slow nominal deployment.

    Parameters
    ----------
    dyn       : ArmDynamics instance from arm_dynamics.py.
    F_kin     : (N0, N1, N2) bool mask — only True cells are evaluated.
    dq_sweep  : (3,) joint velocity  [rad/s]  used for RNEA.  Default: 0.02/0.02/0.01.
    ddq_sweep : (3,) joint accel     [rad/s²] used for RNEA.  Default: 0.01/0.01/0.005.
    verbose   : print progress and summary.

    Returns
    -------
    tau_peak : (N0, N1, N2) float array.  Peak |τ_i| across joints [N·m].
               NaN for cells where F_kin is False (skipped).
    """
    if dq_sweep  is None:
        dq_sweep  = _DQ_SWEEP_DEFAULT
    if ddq_sweep is None:
        ddq_sweep = _DDQ_SWEEP_DEFAULT

    dq_sweep  = np.asarray(dq_sweep,  dtype=float)
    ddq_sweep = np.asarray(ddq_sweep, dtype=float)

    shape    = q0_grid.shape
    n_kin    = int(F_kin.sum())
    tau_peak = np.full(shape, np.nan)

    if verbose:
        print(f"  Computing torque grid for {n_kin} kinematically feasible cells...",
              end="", flush=True)

    it = np.nditer([q0_grid, q1_grid, q2_grid], flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        if F_kin[idx]:
            q   = np.array([float(it[0]), float(it[1]), float(it[2])])
            tau = dyn.joint_torques(q, dq_sweep, ddq_sweep)
            tau_peak[idx] = float(np.max(np.abs(tau)))
        it.iternext()

    if verbose:
        valid = tau_peak[F_kin]
        print(f" done.  τ_peak: mean={np.nanmean(valid):.2f} N·m,"
              f" max={np.nanmax(valid):.2f} N·m")

    return tau_peak


def compute_F_torque(
    tau_peak_grid: np.ndarray,
    tau_max_Nm: float,
) -> np.ndarray:
    """Return bool mask: True where peak joint torque ≤ tau_max_Nm.

    NaN entries (cells skipped by compute_torque_grid) are treated as
    within-limit (True) so that F_kin=False cells do not count as torque
    failures.
    """
    return np.where(np.isnan(tau_peak_grid), True, tau_peak_grid <= tau_max_Nm)
