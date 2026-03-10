"""
2D SPH Satellite Propellant Tank Sloshing Simulation
=====================================================
Uses NVIDIA Warp for GPU-accelerated Smoothed Particle Hydrodynamics (SPH).
Falls back to NumPy if Warp is not installed.

OVERVIEW OF THE METHOD
-----------------------
Smoothed Particle Hydrodynamics (SPH) is a mesh-free Lagrangian method: the
fluid is discretised into a set of moving particles, each carrying properties
such as mass, density, velocity and pressure.  Interactions between particles
are computed via a smoothing kernel W(r, h), which weights contributions from
neighbours within a radius of 2h (the "compact support" of the cubic B-spline).

This simulation uses the Weakly Compressible SPH (WCSPH) formulation, meaning:
  - Density is computed by summation over neighbours (not via continuity).
  - Pressure is obtained from an artificial equation of state (Tait EOS) that
    keeps density variations small (≲ 1%) while remaining explicit in time.
  - Time integration uses a simple Euler-leapfrog scheme with an XSPH position
    correction to suppress particle interpenetration.

PHYSICAL SCENARIO
-----------------
A rectangular satellite propellant tank (e.g. MMH or hydrazine, ρ₀ ≈ 1000 kg/m³)
is partially filled to 55 % of its volume.  The tank is subjected to:
  1. A sinusoidal lateral body force that mimics a reaction-control thruster
     firing that excites the lowest sloshing mode.
  2. At t = THRUST_T seconds a step-change axial acceleration is added,
     representing the main engine being ignited (orbit-raising maneuver).
The free-surface deformation and fluid kinetic energy are tracked over time.

REFERENCES
----------
  Monaghan, J.J. (1992) "Smoothed Particle Hydrodynamics", ARAA 30, 543-574.
  Monaghan, J.J. (1994) "Simulating Free Surface Flows with SPH", JCP 110, 399.
  Becker & Teschner (2007) "Weakly Compressible SPH for Free Surface Flows".

Run:
    pip install warp-lang matplotlib numpy
    python sph_sloshing.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
import sys

# ─────────────────────────────────────────────────────────────────────────────
#  BACKEND SELECTION
#  Try to import NVIDIA Warp first.  Warp compiles Python-annotated kernels
#  to CUDA (GPU) or LLVM (CPU) at runtime, giving near-native performance
#  without hand-writing CUDA C.  If Warp is not installed we fall back to a
#  pure NumPy reference implementation which is physically identical but slower.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import warp as wp
    wp.init()           # initialise CUDA context and JIT compiler
    WARP_AVAILABLE = True
    print(f"[INFO] NVIDIA Warp {wp.__version__} detected — GPU kernels enabled.")
except ImportError:
    WARP_AVAILABLE = False
    print("[INFO] NVIDIA Warp not found — using NumPy CPU fallback.")
    print("       Install with: pip install warp-lang")


# ═════════════════════════════════════════════════════════════════════════════
#  SIMULATION PARAMETERS
#  All physical and numerical settings live here so they are easy to tune
#  without touching the solver code.
# ═════════════════════════════════════════════════════════════════════════════
class SPHConfig:
    # ── Tank geometry ────────────────────────────────────────────────────────
    # The tank is centred at the origin.  TANK_W / TANK_H are the half-extents,
    # so the full tank spans [-TANK_W, TANK_W] × [-TANK_H, TANK_H].
    # Units: metres (1 unit = 1 m in the current parametrisation).
    TANK_W   = 1.0          # half-width of tank  [m]
    TANK_H   = 1.0          # half-height of tank [m]
    FILL     = 0.7         # propellant fill fraction (volume fraction, 0–1)
                            # 0.55 means fluid fills 55 % of the tank volume

    # ── SPH discretisation ───────────────────────────────────────────────────
    # H is the smoothing length.  The compact support radius is 2H, so only
    # particles within distance 2H interact.  Choosing H ≈ 1.2–1.5 * DX
    # ensures each particle has ~20 neighbours in 2D, which gives good accuracy.
    H        = 0.12         # smoothing length h  [m]
    DX       = 0.065        # initial inter-particle spacing [m]  (≈ H / 1.85)
    RHO0     = 1000.0       # reference (rest) density [kg/m³]
                            # Hydrazine ≈ 1004, MMH ≈ 866; 1000 is a safe proxy
    GAMMA    = 7.0          # Tait EOS stiffness exponent (7 is standard for water-like fluids)
    C0       = 20.0         # numerical speed of sound [m/s]
                            # Must satisfy C0 >> V_max so density fluctuations
                            # stay below ~1 %.  Rule of thumb: C0 ≥ 10 * V_max.
    ALPHA    = 0.16         # artificial viscosity coefficient α (dimensionless)
                            # Controls damping of pressure waves; too large → overdamping,
                            # too small → numerical noise / particle penetration.
    EPSILON  = 0.01         # XSPH smoothing parameter ε
                            # Blends each particle's velocity with its neighbours'
                            # to prevent particle clumping (Monaghan 1989).

    # ── Time integration ─────────────────────────────────────────────────────
    # The CFL stability condition for WCSPH requires:
    #   dt < CFL * h / C0        (acoustic CFL,  CFL ≈ 0.25–0.4)
    #   dt < CFL * sqrt(h / |a|) (force  CFL)
    # With h=0.12, C0=20: dt < 0.4 * 0.12 / 20 = 2.4e-3 s → 5e-4 is safe.
    DT       = 5e-4         # time step [s]
    T_END    = 8.0          # total simulation duration [s]
    SUBSTEPS = 4            # SPH steps per rendered animation frame
                            # (not used directly; steps_per_frame is derived from DT & FPS)

    # ── External forcing — orbital maneuver profile ───────────────────────────
    # The lateral body force g_x(t) = SLOSH_AMP * sin(2π * SLOSH_FREQ * t)
    # excites the fundamental sloshing mode of the propellant.
    # At t = THRUST_T a constant axial deceleration THRUST_G is added,
    # representing main-engine ignition (spacecraft accelerates, propellant is
    # pressed toward the aft dome).
    G_BASE      = 9.81      # baseline downward gravity [m/s²]  (keeps fluid settled)
    SLOSH_AMP   = 4.0       # lateral excitation amplitude [m/s²]
    SLOSH_FREQ  = 0.6       # lateral excitation frequency [Hz]
                            # First sloshing mode of a rectangular tank:
                            # f₁ ≈ (1/2π) * sqrt(π g / L * tanh(π h / L))
    THRUST_G    = 2.0       # additional axial thrust acceleration [m/s²]
    THRUST_T    = 2.0       # simulation time at which thrust is applied [s]

    # ── Boundary treatment ───────────────────────────────────────────────────
    # Wall boundaries are represented by two layers of fixed "dummy" particles
    # placed outside the tank.  They contribute to the density summation of
    # nearby fluid particles (ensuring consistent pressure near walls) and
    # exert a short-range repulsive force to prevent fluid particles from
    # tunnelling through the wall.
    N_BDRY_LAYERS = 2       # number of boundary particle layers per wall

    # ── Rendering / output ───────────────────────────────────────────────────
    FPS      = 30           # animation frame rate [frames/s]
    SAVE_GIF = True         # True → save GIF file; False → display interactively
    GIF_FILE = "sloshing.gif"
    COLORMAP = "coolwarm"   # matplotlib colormap for particle speed visualisation


cfg = SPHConfig()


# ═════════════════════════════════════════════════════════════════════════════
#  CUBIC B-SPLINE SMOOTHING KERNEL  W(r, h)
#
#  The kernel is a piecewise cubic polynomial that is:
#    - Positive and symmetric:  W(r) = W(-r) ≥ 0
#    - Normalised:              ∫ W(r) dV = 1
#    - Compactly supported:     W(r) = 0  for r ≥ 2h
#    - Smooth (C² continuous)
#
#  In 2D the normalisation constant σ = 10 / (7π h²).
#
#  The kernel piece-wise form (q = r/h):
#    W = σ * { 1 - 1.5q² + 0.75q³     if 0 ≤ q < 1
#            { 0.25*(2 - q)³            if 1 ≤ q < 2
#            { 0                         if q ≥ 2
# ═════════════════════════════════════════════════════════════════════════════
SIGMA2D = 10.0 / (7.0 * np.pi * cfg.H ** 2)   # 2D normalisation constant


def W(r, h):
    """
    Cubic B-spline kernel value W(r, h).

    Parameters
    ----------
    r : ndarray  — inter-particle distances (must be ≥ 0)
    h : float    — smoothing length

    Returns
    -------
    ndarray of kernel values, same shape as r.
    """
    q = r / h                              # dimensionless distance
    s = np.zeros_like(q)

    mask1 = q < 1.0                        # inner region  (0 ≤ q < 1)
    mask2 = (q >= 1.0) & (q < 2.0)        # outer region  (1 ≤ q < 2)

    s[mask1] = 1.0 - 1.5*q[mask1]**2 + 0.75*q[mask1]**3
    s[mask2] = 0.25*(2.0 - q[mask2])**3

    return SIGMA2D * s


def dW_dr(r, h):
    """
    Radial derivative of the kernel: dW/dr.

    This is used to construct the kernel gradient:
        ∇W_ij = (dW/dr) * (r_ij / |r_ij|)

    Note the derivative is with respect to the distance r, not q = r/h,
    hence the extra 1/h factor (chain rule).

    Parameters
    ----------
    r : ndarray  — inter-particle distances
    h : float    — smoothing length

    Returns
    -------
    ndarray of dW/dr values.
    """
    q = r / h
    ds = np.zeros_like(q)

    mask1 = (q > 1e-8) & (q < 1.0)        # skip r≈0 to avoid division issues
    mask2 = (q >= 1.0) & (q < 2.0)

    # Derivative of the B-spline pieces, multiplied by 1/h (chain rule: dW/dr = dW/dq * dq/dr = dW/dq / h)
    ds[mask1] = (-3.0*q[mask1] + 2.25*q[mask1]**2)
    ds[mask2] = -0.75*(2.0 - q[mask2])**2

    return SIGMA2D / h * ds


def grad_W(rx, ry, r, h):
    """
    Vector kernel gradient ∇W(r_ij, h) = (dW/dr) * r̂_ij.

    In the SPH pressure and viscosity force formulae the kernel gradient
    appears as ∇_i W(r_ij) = (dW/dr) * (x_i - x_j) / |x_i - x_j|.

    Parameters
    ----------
    rx, ry : ndarray  — components of (x_i - x_j)
    r      : ndarray  — |x_i - x_j|
    h      : float    — smoothing length

    Returns
    -------
    (gwx, gwy) — x and y components of ∇W
    """
    dw = dW_dr(r, h)
    safe_r = np.where(r > 1e-8, r, 1.0)   # prevent divide-by-zero at r≈0
    return dw / safe_r * rx,  dw / safe_r * ry


# ═════════════════════════════════════════════════════════════════════════════
#  PARTICLE INITIALISATION
#
#  Two classes of particles are created:
#    1. FLUID particles — mobile, carry mass/velocity/density/pressure.
#       Placed on a regular lattice in the lower portion of the tank
#       (height determined by the fill fraction).
#    2. BOUNDARY (ghost) particles — fixed in space, placed in N_BDRY_LAYERS
#       layers just outside each wall.  They contribute to the density
#       summation of fluid particles near the walls, giving a more accurate
#       near-wall pressure, and exert a repulsive force to enforce no-slip.
# ═════════════════════════════════════════════════════════════════════════════
def init_particles():
    """
    Create and return fluid + boundary particle arrays.

    Returns
    -------
    fluid_x, fluid_y : position arrays [m]
    vx, vy           : velocity arrays [m/s]  (zero initially)
    rho              : density array [kg/m³]  (= RHO0 initially)
    mass             : mass array [kg]         (= RHO0 * DX² per particle)
    bdry_x, bdry_y   : boundary particle position arrays
    """
    dx = cfg.DX
    W2, H2 = cfg.TANK_W, cfg.TANK_H

    # Height of the initial liquid column.
    # The tank spans [-H2, +H2] vertically, so the fluid occupies
    # [-H2, -H2 + fill_h] (filling from the bottom).
    fill_h = 2.0 * H2 * cfg.FILL

    # ── Fluid particles ──────────────────────────────────────────────────────
    # Regular Cartesian grid with spacing DX.  A small inset from the walls
    # (one DX) prevents particles from being initialised inside the boundary
    # layer, which would cause unphysically large repulsive forces at t=0.
    xs, ys = [], []
    x = -W2 + dx                           # start one cell inset from left wall
    while x < W2 - dx * 0.5:              # stop one cell inset from right wall
        y = -H2 + dx                       # start one cell above the floor
        while y < -H2 + fill_h:           # stop at the fill height
            xs.append(x)
            ys.append(y)
            y += dx
        x += dx

    fluid_x = np.array(xs, dtype=np.float64)
    fluid_y = np.array(ys, dtype=np.float64)
    N_fluid = len(fluid_x)

    # ── Boundary particles ───────────────────────────────────────────────────
    # Two layers of particles are placed on each of the four walls.
    # Layer k (k=1,2) is placed at distance k*DX outside the wall face,
    # giving a "ghost" zone of thickness 2*DX that smoothly transitions the
    # kernel support from fluid to solid.
    bx, by = [], []
    for layer in range(cfg.N_BDRY_LAYERS):
        d = (layer + 1) * dx               # distance of this layer from the wall face

        # Bottom wall — particles span the full width plus corner overlap
        x = -W2 - cfg.N_BDRY_LAYERS*dx
        while x <= W2 + cfg.N_BDRY_LAYERS*dx:
            bx.append(x); by.append(-H2 - d)
            x += dx

        # Top wall (ceiling) — same convention
        x = -W2 - cfg.N_BDRY_LAYERS*dx
        while x <= W2 + cfg.N_BDRY_LAYERS*dx:
            bx.append(x); by.append(H2 + d)
            x += dx

        # Left wall — spans the inner fluid region only (corners already covered)
        y = -H2
        while y <= H2:
            bx.append(-W2 - d); by.append(y)
            y += dx

        # Right wall
        y = -H2
        while y <= H2:
            bx.append(W2 + d); by.append(y)
            y += dx

    bdry_x = np.array(bx, dtype=np.float64)
    bdry_y = np.array(by, dtype=np.float64)
    N_bdry = len(bdry_x)

    # ── Initial conditions ───────────────────────────────────────────────────
    vx   = np.zeros(N_fluid)               # all particles at rest
    vy   = np.zeros(N_fluid)
    rho  = np.full(N_fluid, cfg.RHO0)     # uniform rest density
    # Particle mass = rest density × volume element.
    # In 2D, each particle represents a square cell of side DX, so volume = DX².
    mass = (cfg.RHO0 * dx ** 2) * np.ones(N_fluid)

    print(f"[INFO] Fluid particles:    {N_fluid}")
    print(f"[INFO] Boundary particles: {N_bdry}")
    return (fluid_x, fluid_y, vx, vy, rho, mass,
            bdry_x, bdry_y)


# ═════════════════════════════════════════════════════════════════════════════
#  NUMPY SPH BACKEND  (reference / CPU fallback)
#
#  This implements a single time step of the WCSPH algorithm:
#
#    1. DENSITY SUMMATION
#       ρ_i = Σ_j m_j W(|x_i - x_j|, h)   (sum over fluid + boundary neighbours)
#
#    2. TAIT EQUATION OF STATE
#       p_i = B [(ρ_i/ρ₀)^γ - 1],   B = ρ₀ C₀² / γ
#
#    3. MOMENTUM EQUATION (pressure + viscosity)
#       a_i = -Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij
#            + Σ_j m_j π_ij ∇W_ij   (artificial viscosity)
#            + g(t)                   (body force)
#
#    4. XSPH VELOCITY CORRECTION
#       ṽ_i = v_i + ε Σ_j (m_j/ρ_j)(v_j - v_i) W_ij
#
#    5. TIME INTEGRATION  (Euler-leapfrog)
#       v_i^{n+1} = v_i^n + dt * a_i^n
#       x_i^{n+1} = x_i^n + dt * ṽ_i^{n+1}
# ═════════════════════════════════════════════════════════════════════════════
def tait_pressure(rho):
    """
    Weakly compressible Tait equation of state (EOS).

    Relates local density to pressure via:
        p = B * [(ρ/ρ₀)^γ - 1],    B = ρ₀ C₀² / γ

    The stiffness coefficient B is chosen so that the reference speed of
    sound C₀ = sqrt(γ B / ρ₀) = C0.  Using C0 >> max fluid velocity keeps
    the Mach number M = V/C0 small, ensuring near-incompressible behaviour.

    Parameters
    ----------
    rho : ndarray — local particle densities [kg/m³]

    Returns
    -------
    p : ndarray — pressure [Pa]
    """
    B = cfg.RHO0 * cfg.C0**2 / cfg.GAMMA
    return B * ((rho / cfg.RHO0)**cfg.GAMMA - 1.0)


def numpy_sph_step(px, py, vx, vy, rho, mass,
                   bx, by, dt, t):
    """
    Advance the SPH system by one time step dt using NumPy (CPU).

    Parameters
    ----------
    px, py : ndarray[N]  — particle positions [m]
    vx, vy : ndarray[N]  — particle velocities [m/s]
    rho    : ndarray[N]  — particle densities [kg/m³]
    mass   : ndarray[N]  — particle masses [kg]
    bx, by : ndarray[Nb] — boundary particle positions [m]
    dt     : float       — time step [s]
    t      : float       — current simulation time [s]  (used for gravity)

    Returns
    -------
    px_new, py_new, vx_new, vy_new, rho_new — updated state arrays
    """
    N  = len(px)
    Nb = len(bx)
    h  = cfg.H
    h2 = (2.0 * h) ** 2    # square of kernel compact support radius

    # ── Step 1: External body force at current time ────────────────────────
    # Lateral acceleration: sinusoidal, simulates reaction-control thruster
    gx = cfg.SLOSH_AMP * np.sin(2*np.pi*cfg.SLOSH_FREQ * t)
    # Vertical acceleration: constant gravity, plus axial thrust after THRUST_T
    gy = -cfg.G_BASE
    if t > cfg.THRUST_T:
        gy -= cfg.THRUST_G    # main engine fires: adds downward (aft) acceleration

    # ── Step 2: Pairwise distance matrices ────────────────────────────────
    # Build NxN displacement matrices for all fluid–fluid pairs.
    # This is O(N²) in both memory and time — acceptable for N < 3000,
    # but a cell-linked list or k-d tree would be needed for larger N.
    dx_ff = px[:, None] - px[None, :]   # (N, N) x-displacement matrix
    dy_ff = py[:, None] - py[None, :]   # (N, N) y-displacement matrix
    r2_ff = dx_ff**2 + dy_ff**2         # (N, N) squared distances

    # Boolean neighbour mask: True where particles interact (within 2h, not self)
    neigh = (r2_ff < h2) & (r2_ff > 1e-12)

    # Fluid–boundary pairwise displacements (N × Nb matrices)
    dx_fb = px[:, None] - bx[None, :]
    dy_fb = py[:, None] - by[None, :]
    r2_fb = dx_fb**2 + dy_fb**2
    neigh_b = r2_fb < h2               # True where fluid particle i is near boundary b

    # ── Step 3: Density summation ──────────────────────────────────────────
    # ρ_i = Σ_{j≠i} m_j W(r_ij) + m_i W(0)   [fluid–fluid]
    #      + Σ_b   m_i W(r_ib)                 [fluid–boundary, using mirror mass]
    r_ff = np.sqrt(np.maximum(r2_ff, 1e-20))  # (N,N) distances; avoid sqrt(0)
    rho_new = np.zeros(N)
    for i in range(N):
        js = np.where(neigh[i])[0]             # indices of fluid neighbours of i
        rij = r_ff[i, js]
        rho_new[i] = np.sum(mass[js] * W(rij, h))
        rho_new[i] += mass[i] * W(np.array([0.0]), h)[0]  # self-contribution W(0)

    # Boundary particles contribute with the same mass as the fluid particle
    # (mirror-mass approach: ensures the density near the wall matches the
    # bulk density even though there are no real fluid particles outside)
    r_fb = np.sqrt(np.maximum(r2_fb, 1e-20))
    for i in range(N):
        bs = np.where(neigh_b[i])[0]
        if len(bs):
            rib = r_fb[i, bs]
            rho_new[i] += np.sum(mass[i] * W(rib, h))

    # Prevent unphysically low densities that could produce negative pressure
    rho_new = np.maximum(rho_new, cfg.RHO0 * 0.5)

    # ── Step 4: Pressure from Tait EOS ────────────────────────────────────
    p = tait_pressure(rho_new)

    # ── Step 5: Acceleration from pressure gradient + viscosity ───────────
    ax = np.zeros(N)
    ay = np.zeros(N)

    for i in range(N):
        js = np.where(neigh[i])[0]
        if len(js) == 0:
            continue                            # isolated particle — no interaction

        rij  = r_ff[i, js]
        dxij = dx_ff[i, js]
        dyij = dy_ff[i, js]

        # Kernel gradient vector ∇_i W(r_ij)
        gwx, gwy = grad_W(dxij, dyij, rij, h)

        # ── Pressure gradient (symmetric SPH form) ───────────────────────
        # SPH discretisation of -∇p/ρ:
        #   a_i^p = -Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij
        # The symmetric form conserves linear and angular momentum exactly.
        pi_rho2 = p[i]  / rho_new[i]**2   # p_i / ρ_i²
        pj_rho2 = p[js] / rho_new[js]**2  # p_j / ρ_j²  (vector over neighbours)
        fac = mass[js] * (pi_rho2 + pj_rho2)
        ax[i] -= np.sum(fac * gwx)
        ay[i] -= np.sum(fac * gwy)

        # ── Artificial viscosity (Monaghan 1992) ─────────────────────────
        # Added only for approaching pairs (v_ij · r_ij < 0) to damp shocks
        # and prevent particle penetration.
        #
        # π_ij = -α C₀ μ_ij / ρ̄_ij,    μ_ij = h (v_ij · r_ij) / (r_ij² + 0.01h²)
        #
        # The 0.01h² term in the denominator prevents singularities when r→0.
        dvx = vx[i] - vx[js]
        dvy = vy[i] - vy[js]
        vr  = dvx*dxij + dvy*dyij          # v_ij · r_ij  (scalar, same sign as approach)
        mu_ij = h * vr / (rij**2 + 0.01*h**2)
        visc_flag = vr < 0                 # only apply viscosity for approaching pairs
        rho_avg = 0.5*(rho_new[i] + rho_new[js])
        pi_ij = np.where(visc_flag,
                         -cfg.ALPHA * cfg.C0 * mu_ij / rho_avg,
                         0.0)
        ax[i] -= np.sum(mass[js] * pi_ij * gwx)
        ay[i] -= np.sum(mass[js] * pi_ij * gwy)

    # ── Step 6: Boundary repulsion (Lennard-Jones-like penalty force) ──────
    # Each fluid particle i that comes within 2*DX of a boundary particle b
    # experiences a short-range repulsive force directed away from b.
    # The Lennard-Jones form r₀⁴/r⁴ - r₀²/r² gives a steep repulsion at
    # close range and transitions to zero at r = r₀*sqrt(2) ≈ 1.41 DX.
    for i in range(N):
        bs = np.where(neigh_b[i])[0]
        if len(bs) == 0:
            continue
        rib  = r_fb[i, bs]
        dxib = dx_fb[i, bs]
        dyib = dy_fb[i, bs]
        r0   = cfg.DX                      # equilibrium distance for repulsion
        ratio = r0 / np.maximum(rib, 1e-6)
        # Force magnitude: C₀² is used to scale the force to the pressure scale
        rep  = cfg.C0**2 * (ratio**4 - ratio**2) / np.maximum(rib**2, 1e-8)
        rep  = np.where(rib < 2.0*r0, rep, 0.0)   # zero outside activation distance
        ax[i] += np.sum(rep * dxib)
        ay[i] += np.sum(rep * dyib)

    # ── Step 7: Add external body forces ──────────────────────────────────
    ax += gx
    ay += gy

    # ── Step 8: XSPH velocity correction ──────────────────────────────────
    # XSPH (Monaghan 1989) regularises particle positions by slightly
    # nudging each particle's velocity toward the local average.
    # This prevents particle clumping and improves the regularity of the
    # particle distribution.
    #   ṽ_i = v_i + ε Σ_j (m_j/ρ_j)(v_j - v_i) W_ij
    vx_corr = np.zeros(N)
    vy_corr = np.zeros(N)
    for i in range(N):
        js = np.where(neigh[i])[0]
        if len(js) == 0:
            continue
        rij = r_ff[i, js]
        wij = W(rij, h)
        vx_corr[i] = cfg.EPSILON * np.sum(
            mass[js]/rho_new[js] * (vx[js]-vx[i]) * wij)
        vy_corr[i] = cfg.EPSILON * np.sum(
            mass[js]/rho_new[js] * (vy[js]-vy[i]) * wij)

    # ── Step 9: Leapfrog (symplectic Euler) time integration ──────────────
    # Velocity is updated with the full acceleration, then positions are
    # advanced with the corrected (XSPH) velocity.
    vx_new = vx + dt * ax
    vy_new = vy + dt * ay
    px_new = px + dt * (vx_new + vx_corr)
    py_new = py + dt * (vy_new + vy_corr)

    # Failsafe position clamping — should rarely activate if DT is chosen
    # correctly; if particles escape the tank, reduce DT or increase ALPHA.
    m = 0.01
    px_new = np.clip(px_new, -cfg.TANK_W + m, cfg.TANK_W - m)
    py_new = np.clip(py_new, -cfg.TANK_H + m, cfg.TANK_H - m)

    return px_new, py_new, vx_new, vy_new, rho_new


# ═════════════════════════════════════════════════════════════════════════════
#  NVIDIA WARP GPU KERNEL DEFINITIONS
#
#  Warp kernels are Python functions decorated with @wp.kernel.  Warp's
#  ahead-of-time (AOT) compiler translates them to CUDA PTX (for GPU) or
#  LLVM IR (for CPU), allowing near-native throughput.
#
#  Key Warp concepts used here:
#    wp.tid()               — thread index (one thread per particle)
#    wp.array(dtype=float)  — typed GPU/CPU array (like a typed numpy array)
#    wp.launch(kernel, dim, inputs)  — dispatch kernel over 'dim' threads
#    wp.func                — device function callable from a kernel
#
#  The three kernels below correspond to the three main SPH stages:
#    1. density_kernel   — accumulates density ρ_i by kernel summation
#    2. force_kernel     — computes pressure + viscosity accelerations
#    3. integrate_kernel — advances positions and velocities
#
#  NOTE: These kernels use O(N²) neighbour loops.  For large N (> 10 000),
#  a spatial hash or cell-linked list should be used to reduce to O(N).
# ═════════════════════════════════════════════════════════════════════════════
if WARP_AVAILABLE:

    @wp.func
    def cubic_W(r: float, h: float) -> float:
        """
        Cubic B-spline kernel W(r, h) — Warp device function.
        Called from within GPU kernels; cannot be called from Python directly.
        Mirrors the NumPy W() function above but operates on scalar floats.
        """
        sigma = float(10.0) / (float(7.0) * float(3.14159265) * h * h)
        q = r / h
        if q < 1.0:
            return sigma * (1.0 - 1.5*q*q + 0.75*q*q*q)
        elif q < 2.0:
            d = 2.0 - q
            return sigma * 0.25 * d * d * d
        return float(0.0)

    @wp.func
    def cubic_dW(r: float, h: float) -> float:
        """
        Radial derivative dW/dr of the cubic B-spline — Warp device function.
        Returns 0 for r < 1e-8 to avoid numerical issues at coincident positions.
        """
        sigma = float(10.0) / (float(7.0) * float(3.14159265) * h * h)
        q = r / h
        if r < 1e-8:
            return float(0.0)
        if q < 1.0:
            return sigma / h * (-3.0*q + 2.25*q*q)
        elif q < 2.0:
            return sigma / h * (-0.75*(2.0 - q)*(2.0 - q))
        return float(0.0)

    @wp.kernel
    def density_kernel(
        px: wp.array(dtype=float), py: wp.array(dtype=float),
        bx: wp.array(dtype=float), by: wp.array(dtype=float),
        mass: wp.array(dtype=float),
        rho: wp.array(dtype=float),
        h: float, N: int, Nb: int
    ):
        """
        GPU kernel: density summation.

        Each thread handles one fluid particle i.  It loops over all other
        fluid particles and all boundary particles to accumulate:
            ρ_i = m_i W(0) + Σ_{j≠i} m_j W(r_ij) + Σ_b m_i W(r_ib)

        Thread index wp.tid() maps directly to particle index i.
        """
        i = wp.tid()     # particle index — one CUDA thread per particle

        # Self-contribution: W(0) = σ (the kernel value at zero distance)
        rho_i = mass[i] * cubic_W(float(0.0), h)

        # Fluid–fluid contributions (loop over all other particles)
        for j in range(N):
            if j == i:
                continue                    # skip self (already handled above)
            dx = px[i] - px[j]
            dy = py[i] - py[j]
            r  = wp.sqrt(dx*dx + dy*dy)
            if r < 2.0*h:                  # only within compact support
                rho_i += mass[j] * cubic_W(r, h)

        # Fluid–boundary contributions (mirror-mass approach)
        for b in range(Nb):
            dx = px[i] - bx[b]
            dy = py[i] - by[b]
            r  = wp.sqrt(dx*dx + dy*dy)
            if r < 2.0*h:
                rho_i += mass[i] * cubic_W(r, h)   # boundary uses fluid mass

        rho[i] = rho_i      # write result back to global array

    @wp.kernel
    def force_kernel(
        px: wp.array(dtype=float), py: wp.array(dtype=float),
        vx: wp.array(dtype=float), vy: wp.array(dtype=float),
        bx: wp.array(dtype=float), by: wp.array(dtype=float),
        mass: wp.array(dtype=float),
        rho:  wp.array(dtype=float),
        ax_out: wp.array(dtype=float),
        ay_out: wp.array(dtype=float),
        h: float, C0: float, alpha: float,
        rho0: float, gamma: float,
        gx_ext: float, gy_ext: float,
        dx0: float,
        N: int, Nb: int
    ):
        """
        GPU kernel: pressure + viscosity + boundary forces.

        For each fluid particle i:
          a) Compute pressure p_i from Tait EOS
          b) Loop over fluid neighbours j:
               - add symmetric pressure gradient term
               - add Monaghan artificial viscosity (approaching pairs only)
          c) Loop over boundary neighbours b:
               - add Lennard-Jones repulsion
          d) Add external gravity (gx_ext, gy_ext)

        Output is written to ax_out[i], ay_out[i].
        """
        i = wp.tid()

        # Tait EOS: p_i = B [(ρ_i / ρ₀)^γ - 1]
        B    = rho0 * C0 * C0 / gamma
        pi_i = B * (wp.pow(rho[i]/rho0, gamma) - float(1.0))

        ax_i = float(0.0)
        ay_i = float(0.0)

        # ── Fluid–fluid forces ───────────────────────────────────────────
        for j in range(N):
            if j == i:
                continue
            dxij = px[i] - px[j]
            dyij = py[i] - py[j]
            r    = wp.sqrt(dxij*dxij + dyij*dyij)
            if r >= 2.0*h:
                continue                        # outside compact support, skip

            pj_j = B * (wp.pow(rho[j]/rho0, gamma) - float(1.0))  # pressure at j
            dw   = cubic_dW(r, h)              # scalar dW/dr

            # Symmetric pressure gradient:  -m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij
            fac  = mass[j] * (pi_i/(rho[i]*rho[i]) + pj_j/(rho[j]*rho[j]))
            ax_i -= fac * dw / r * dxij        # ∇W_ij = (dW/dr) * r̂_ij
            ay_i -= fac * dw / r * dyij

            # ── Monaghan artificial viscosity ────────────────────────────
            # Only applied when particles approach (v_ij · r_ij < 0).
            # π_ij = -α C₀ h (v_ij · r_ij) / (ρ̄_ij (r_ij² + 0.01h²))
            vr = (vx[i]-vx[j])*dxij + (vy[i]-vy[j])*dyij
            if vr < float(0.0):                # approaching pair
                mu   = h * vr / (r*r + float(0.01)*h*h)
                piij = -alpha * C0 * mu / (float(0.5)*(rho[i]+rho[j]))
                ax_i -= mass[j] * piij * dw / r * dxij
                ay_i -= mass[j] * piij * dw / r * dyij

        # ── Boundary repulsion ───────────────────────────────────────────
        # Short-range Lennard-Jones penalty force prevents wall penetration.
        for b in range(Nb):
            dxib = px[i] - bx[b]
            dyib = py[i] - by[b]
            rib  = wp.sqrt(dxib*dxib + dyib*dyib)
            if rib < float(2.0)*dx0 and rib > float(1e-6):
                ratio = dx0 / rib
                # F ∝ C₀² (r₀⁴/r⁴ - r₀²/r²) / r² — repulsive for r < r₀√2
                rep   = C0*C0 * (ratio*ratio*ratio*ratio - ratio*ratio) / (rib*rib)
                ax_i += rep * dxib
                ay_i += rep * dyib

        # ── External body forces (gravity + maneuver) ────────────────────
        ax_out[i] = ax_i + gx_ext
        ay_out[i] = ay_i + gy_ext

    @wp.kernel
    def integrate_kernel(
        px: wp.array(dtype=float), py: wp.array(dtype=float),
        vx: wp.array(dtype=float), vy: wp.array(dtype=float),
        ax: wp.array(dtype=float), ay: wp.array(dtype=float),
        px_new: wp.array(dtype=float), py_new: wp.array(dtype=float),
        vx_new: wp.array(dtype=float), vy_new: wp.array(dtype=float),
        dt: float, tank_w: float, tank_h: float
    ):
        """
        GPU kernel: symplectic Euler time integration.

        Advances velocities and positions by one time step:
            v^{n+1} = v^n + dt * a^n
            x^{n+1} = x^n + dt * v^{n+1}   (note: uses updated velocity)

        Positions are clamped to the interior of the tank as a hard failsafe.
        Under normal operation (stable DT) the clamp should never activate.
        """
        i = wp.tid()
        vx_new[i] = vx[i] + dt * ax[i]    # velocity update
        vy_new[i] = vy[i] + dt * ay[i]
        xn = px[i] + dt * vx_new[i]       # position update with new velocity
        yn = py[i] + dt * vy_new[i]
        m = float(0.01)                    # small margin from wall face
        px_new[i] = wp.clamp(xn, -tank_w + m, tank_w - m)
        py_new[i] = wp.clamp(yn, -tank_h + m, tank_h - m)


    class WarpSPH:
        """
        Manages NVIDIA Warp GPU arrays and orchestrates kernel launches.

        This class is a thin wrapper that:
          1. Allocates Warp arrays on the appropriate device (CUDA or CPU).
          2. Provides a step() method that dispatches the three kernels in order.
          3. Provides get_state() to copy results back to NumPy for rendering.

        Double-buffering is used for positions/velocities: after each step the
        "new" and "current" buffer pointers are swapped, avoiding an extra copy.
        """
        def __init__(self, px, py, vx, vy, rho, mass, bx, by):
            self.N  = len(px)
            self.Nb = len(bx)
            # Use CUDA if a GPU is available, otherwise fall back to Warp's CPU backend
            self.device = "cuda" if wp.get_cuda_devices() else "cpu"
            print(f"[WARP] Running on: {self.device}")

            # Allocate GPU arrays — float32 is used for memory efficiency and speed
            self.px   = wp.array(px.astype(np.float32),   dtype=float, device=self.device)
            self.py   = wp.array(py.astype(np.float32),   dtype=float, device=self.device)
            self.vx   = wp.array(vx.astype(np.float32),   dtype=float, device=self.device)
            self.vy   = wp.array(vy.astype(np.float32),   dtype=float, device=self.device)
            self.rho  = wp.array(rho.astype(np.float32),  dtype=float, device=self.device)
            self.mass = wp.array(mass.astype(np.float32), dtype=float, device=self.device)
            self.bx   = wp.array(bx.astype(np.float32),   dtype=float, device=self.device)
            self.by   = wp.array(by.astype(np.float32),   dtype=float, device=self.device)

            # Acceleration buffers (written by force_kernel, read by integrate_kernel)
            self.ax   = wp.zeros(self.N, dtype=float, device=self.device)
            self.ay   = wp.zeros(self.N, dtype=float, device=self.device)

            # Double buffers for position and velocity (avoid in-place overwrite)
            self.px_n = wp.zeros(self.N, dtype=float, device=self.device)
            self.py_n = wp.zeros(self.N, dtype=float, device=self.device)
            self.vx_n = wp.zeros(self.N, dtype=float, device=self.device)
            self.vy_n = wp.zeros(self.N, dtype=float, device=self.device)

        def step(self, dt, t):
            """
            Advance simulation by one time step dt at simulation time t.

            Launches all three Warp kernels in sequence:
              density_kernel  → updates rho
              force_kernel    → updates ax, ay
              integrate_kernel → updates px_n, py_n, vx_n, vy_n
            Then pointer-swaps the new and old buffers.
            """
            N, Nb = self.N, self.Nb

            # Compute time-dependent gravity components for this step
            gx = float(cfg.SLOSH_AMP * np.sin(2*np.pi*cfg.SLOSH_FREQ * t))
            gy = float(-cfg.G_BASE - (cfg.THRUST_G if t > cfg.THRUST_T else 0.0))

            # Stage 1: density summation (reads px,py,bx,by,mass → writes rho)
            wp.launch(density_kernel,
                      dim=N, inputs=[self.px, self.py, self.bx, self.by,
                                     self.mass, self.rho,
                                     float(cfg.H), N, Nb])

            # Stage 2: force computation (reads all state → writes ax, ay)
            wp.launch(force_kernel,
                      dim=N, inputs=[self.px, self.py, self.vx, self.vy,
                                     self.bx, self.by, self.mass, self.rho,
                                     self.ax, self.ay,
                                     float(cfg.H), float(cfg.C0),
                                     float(cfg.ALPHA), float(cfg.RHO0),
                                     float(cfg.GAMMA), gx, gy,
                                     float(cfg.DX), N, Nb])

            # Stage 3: integration (reads state + ax,ay → writes px_n, py_n, vx_n, vy_n)
            wp.launch(integrate_kernel,
                      dim=N, inputs=[self.px, self.py, self.vx, self.vy,
                                     self.ax, self.ay,
                                     self.px_n, self.py_n, self.vx_n, self.vy_n,
                                     dt,
                                     float(cfg.TANK_W), float(cfg.TANK_H)])

            # Pointer swap — no data copy needed; new arrays become current
            self.px, self.px_n = self.px_n, self.px
            self.py, self.py_n = self.py_n, self.py
            self.vx, self.vx_n = self.vx_n, self.vx
            self.vy, self.vy_n = self.vy_n, self.vy

        def get_state(self):
            """
            Copy GPU arrays back to NumPy for rendering/diagnostics.
            This triggers a CUDA device→host memcpy and should be called
            sparingly (e.g. once per rendered frame, not every substep).
            """
            return (self.px.numpy(), self.py.numpy(),
                    self.vx.numpy(), self.vy.numpy(),
                    self.rho.numpy())


# ═════════════════════════════════════════════════════════════════════════════
#  UNIFIED SIMULATION RUNNER
#
#  This function initialises particles, sets up the Matplotlib animation,
#  and runs the main simulation loop.  On each animation frame it:
#    1. Advances the SPH solver for `steps_per_frame` substeps.
#    2. Updates the particle scatter plot with new positions and speeds.
#    3. Updates the telemetry panel (phase, gravity, KE, density).
#    4. Redraws the gravity direction arrow.
# ═════════════════════════════════════════════════════════════════════════════
def run_simulation():
    """
    Initialise, run and animate the SPH propellant sloshing simulation.

    Returns
    -------
    t_history  : list of float — simulation times of each rendered frame
    ke_history : list of float — fluid kinetic energy at each rendered frame
    """
    px, py, vx, vy, rho, mass, bx, by = init_particles()

    # Instantiate the appropriate solver backend
    if WARP_AVAILABLE:
        solver = WarpSPH(px, py, vx, vy, rho, mass, bx, by)
    # else: use the plain NumPy arrays directly in numpy_sph_step()

    n_frames = int(cfg.T_END * cfg.FPS)               # total number of animation frames
    dt_frame = 1.0 / cfg.FPS                          # wall-time per frame [s]
    # Number of SPH time steps per rendered frame.
    # This must be ≥ 1; more substeps → smoother physics but slower rendering.
    steps_per_frame = max(1, int(dt_frame / cfg.DT))

    # ── Build the Matplotlib figure ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#0a0a1a")
    ax_sim, ax_info = axes
    for ax in axes:
        ax.set_facecolor("#0a0a1a")

    # Main simulation view — axes limits are slightly larger than the tank
    ax_sim.set_xlim(-cfg.TANK_W*1.1, cfg.TANK_W*1.1)
    ax_sim.set_ylim(-cfg.TANK_H*1.1, cfg.TANK_H*1.1)
    ax_sim.set_aspect("equal")
    ax_sim.set_title("SPH Propellant Sloshing — Satellite Tank",
                     color="white", fontsize=13, pad=8)
    ax_sim.tick_params(colors="gray")
    for spine in ax_sim.spines.values():
        spine.set_edgecolor("#333355")

    # Tank outline rectangle (visual reference — the actual walls are enforced
    # by the boundary particles, not this rectangle)
    tank_rect = patches.Rectangle(
        (-cfg.TANK_W, -cfg.TANK_H), 2*cfg.TANK_W, 2*cfg.TANK_H,
        linewidth=2, edgecolor="#44aaff", facecolor="none", zorder=5,
        label="Tank wall")
    ax_sim.add_patch(tank_rect)

    # Show boundary particles as faint dots so the user can see the ghost zone
    ax_sim.scatter(bx, by, s=2, c="#334455", alpha=0.5, zorder=2)

    # Fluid particle scatter — coloured by speed magnitude
    speeds = np.sqrt(vx**2 + vy**2)
    sc = ax_sim.scatter(px, py, s=6, c=speeds,
                        cmap=cfg.COLORMAP, vmin=0, vmax=3.0, zorder=3)
    cbar = fig.colorbar(sc, ax=ax_sim, fraction=0.03, pad=0.02)
    cbar.set_label("Speed [m/s]", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Arrow showing the instantaneous effective gravity direction
    arrow = ax_sim.annotate("", xy=(0, 0), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="-|>", color="red",
                                             lw=2.5))
    grav_label = ax_sim.text(0.02, 0.96, "", transform=ax_sim.transAxes,
                             color="red", fontsize=9, va="top")

    # Corner text labels
    time_label = ax_sim.text(0.02, 0.02, "t = 0.00 s",
                              transform=ax_sim.transAxes,
                              color="white", fontsize=10)
    backend_label = ax_sim.text(0.98, 0.02,
        "Backend: WARP (GPU)" if WARP_AVAILABLE else "Backend: NumPy (CPU)",
        transform=ax_sim.transAxes, color="#88ccff", fontsize=8,
        ha="right")

    # ── Telemetry panel (right-hand axes) ─────────────────────────────────
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    ax_info.axis("off")
    ax_info.set_title("Mission Telemetry", color="white", fontsize=12)

    info_texts = {}
    labels = ["Phase", "t [s]", "gx [m/s²]", "gy [m/s²]",
              "Max speed", "Mean ρ/ρ₀", "ΔKE"]
    for k, lb in enumerate(labels):
        y = 0.92 - k*0.11
        ax_info.text(0.05, y, lb + ":", color="#aaaacc", fontsize=10)
        info_texts[lb] = ax_info.text(0.5, y, "—",
                                      color="white", fontsize=10)

    # Kinetic energy and time history (accumulated over all frames for plotting)
    ke_history = []
    t_history  = []

    def update(frame):
        """
        Matplotlib animation callback — called once per rendered frame.

        Advances the physics simulation by `steps_per_frame` SPH steps,
        then refreshes all visual elements.
        """
        nonlocal px, py, vx, vy, rho   # allow reassignment of outer variables

        t_now = frame * dt_frame        # simulation time at the start of this frame

        # ── Physics advance ───────────────────────────────────────────────
        t0 = time.perf_counter()        # wall-clock timer for performance display
        if WARP_AVAILABLE:
            # GPU path: run substeps entirely on the device, then copy to host
            for _ in range(steps_per_frame):
                t_sub = t_now + _ * cfg.DT
                solver.step(cfg.DT, t_sub)
            px, py, vx, vy, rho = solver.get_state()   # single host←device copy
        else:
            # CPU path: call NumPy step function for each substep
            for _ in range(steps_per_frame):
                t_sub = t_now + _ * cfg.DT
                px, py, vx, vy, rho = numpy_sph_step(
                    px, py, vx, vy, rho, mass, bx, by, cfg.DT, t_sub)
        wall = time.perf_counter() - t0    # time taken for this frame's physics

        # ── Update particle scatter ───────────────────────────────────────
        speeds = np.sqrt(vx**2 + vy**2)
        sc.set_offsets(np.column_stack([px, py]))   # update positions
        sc.set_array(speeds)                        # update colour (speed)

        # ── Update gravity arrow ──────────────────────────────────────────
        gx_now = cfg.SLOSH_AMP * np.sin(2*np.pi*cfg.SLOSH_FREQ * t_now)
        gy_now = -cfg.G_BASE - (cfg.THRUST_G if t_now > cfg.THRUST_T else 0.0)
        scale = 0.2                                 # arrow length scaling factor
        cx, cy = 0.0, cfg.TANK_H * 0.75            # arrow anchor point (upper centre)
        ex, ey = cx + gx_now*scale, cy + gy_now*scale
        arrow.set_position((ex, ey))               # arrowhead position
        arrow.xy = (cx, cy)                        # arrow tail position
        grav_label.set_text(f"g=({gx_now:.1f}, {gy_now:.1f}) m/s²")

        # ── Update telemetry text ─────────────────────────────────────────
        ke = 0.5 * np.sum(mass * (vx**2 + vy**2))  # total fluid kinetic energy
        ke_history.append(ke)
        t_history.append(t_now)
        phase = "Maneuver" if t_now > cfg.THRUST_T else "Sloshing"
        vals  = [phase, f"{t_now:.2f}", f"{gx_now:+.2f}",
                 f"{gy_now:+.2f}", f"{speeds.max():.2f} m/s",
                 f"{rho.mean()/cfg.RHO0:.4f}",
                 f"{ke:.2f} J"]
        for lb, v in zip(labels, vals):
            info_texts[lb].set_text(v)

        # Show instantaneous frame rate in the time label
        time_label.set_text(f"t = {t_now:.2f} s  ({1/max(wall,1e-4):.0f} fps)")

        # Return all artists that changed so Matplotlib can do a partial redraw
        return sc, arrow, grav_label, time_label, *info_texts.values()

    # ── Create and save/show the animation ───────────────────────────────
    print(f"[INFO] Rendering {n_frames} frames at {cfg.FPS} fps …")
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000//cfg.FPS, blit=False)

    plt.tight_layout()

    if cfg.SAVE_GIF:
        print(f"[INFO] Saving animation to {cfg.GIF_FILE} …")
        writer = animation.PillowWriter(fps=cfg.FPS)
        ani.save(cfg.GIF_FILE, writer=writer, dpi=120)
        print(f"[INFO] Saved → {cfg.GIF_FILE}")
    else:
        plt.show()

    plt.close(fig)
    return t_history, ke_history


# ═════════════════════════════════════════════════════════════════════════════
#  KINETIC ENERGY DIAGNOSTIC PLOT
#
#  Saves a standalone figure of fluid kinetic energy over time.
#  Useful for checking whether:
#    - The initial sloshing has reached a quasi-steady oscillation.
#    - The thrust event causes a measurable KE spike (slosh excitation).
#    - The simulation is numerically stable (KE should remain bounded).
# ═════════════════════════════════════════════════════════════════════════════
def plot_diagnostics(t_history, ke_history):
    """
    Plot and save the fluid kinetic energy history.

    Parameters
    ----------
    t_history  : list of float — simulation times [s]
    ke_history : list of float — kinetic energy [J] at each time
    """
    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#0a0a1a")
    ax.plot(t_history, ke_history, color="#44aaff", lw=1.5)
    # Mark the thrust event — a vertical dashed line helps correlate KE spikes
    ax.axvline(cfg.THRUST_T, color="red", lw=1, ls="--", label="Thrust on")
    ax.set_xlabel("Time [s]", color="white")
    ax.set_ylabel("Kinetic Energy [J]", color="white")
    ax.set_title("Fluid KE History — Sloshing Dynamics", color="white")
    ax.tick_params(colors="gray")
    ax.legend(facecolor="#111133", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    plt.tight_layout()
    diag_path = "ke_history.png"
    plt.savefig(diag_path, dpi=120, facecolor=fig.get_facecolor())
    print(f"[INFO] Diagnostic plot saved → {diag_path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print(" 2D SPH Satellite Propellant Tank Sloshing")
    print("=" * 60)
    t_hist, ke_hist = run_simulation()
    plot_diagnostics(t_hist, ke_hist)
    print("[DONE]")