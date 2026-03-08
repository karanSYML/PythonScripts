"""
2D SPH Satellite Propellant Tank Sloshing Simulation
=====================================================
Uses NVIDIA Warp for GPU-accelerated Smoothed Particle Hydrodynamics (SPH).
Falls back to NumPy if Warp is not installed.

Physics:
  - Weakly compressible SPH (WCSPH) with Tait equation of state
  - Viscosity: artificial viscosity (Monaghan 1992)
  - Boundary: Repulsive boundary particles (Lennard-Jones style)
  - Gravity: time-varying lateral + vertical (simulates orbital maneuver sloshing)
  - Kernel: Cubic B-spline W(r, h)

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

# ─────────────────────────────────────────────────────────────
#  Try to import NVIDIA Warp; fall back to NumPy backend
# ─────────────────────────────────────────────────────────────
try:
    import warp as wp
    wp.init()
    WARP_AVAILABLE = True
    print(f"[INFO] NVIDIA Warp {wp.__version__} detected — GPU kernels enabled.")
except ImportError:
    WARP_AVAILABLE = False
    print("[INFO] NVIDIA Warp not found — using NumPy CPU fallback.")
    print("       Install with: pip install warp-lang")

# ═══════════════════════════════════════════════════════════════
#  SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════
class SPHConfig:
    # Tank geometry (normalised units: 1 unit ≈ 0.5 m)
    TANK_W   = 2.0          # tank half-width
    TANK_H   = 2.0          # tank half-height
    FILL     = 0.55         # fill fraction (volume of fluid / tank volume)

    # SPH discretisation
    H        = 0.12         # smoothing length
    DX       = 0.065        # initial particle spacing
    RHO0     = 1000.0       # rest density  [kg/m³] (hydrazine ~1000)
    GAMMA    = 7.0          # Tait EOS exponent
    C0       = 20.0         # numerical speed of sound  [m/s]
    ALPHA    = 0.08         # artificial viscosity coefficient
    EPSILON  = 0.01         # XSPH smoothing factor

    # Time integration
    DT       = 5e-4         # time step [s]
    T_END    = 8.0          # total simulation time [s]
    SUBSTEPS = 4            # substeps per rendered frame

    # Maneuver: sinusoidal lateral + step vertical gravity
    G_BASE   = 9.81         # base gravity (vertical, pointing down)
    SLOSH_AMP   = 4.0       # lateral gravity amplitude [m/s²]
    SLOSH_FREQ  = 0.6       # sloshing excitation frequency [Hz]
    THRUST_G    = 2.0       # extra axial thrust acceleration [m/s²]
    THRUST_T    = 2.0       # time at which thrust starts [s]

    # Boundary particles
    N_BDRY_LAYERS = 2

    # Rendering
    FPS      = 30
    SAVE_GIF = True
    GIF_FILE = "sloshing.gif"
    COLORMAP = "coolwarm"   # colour by speed magnitude


cfg = SPHConfig()

# ═══════════════════════════════════════════════════════════════
#  CUBIC B-SPLINE KERNEL (2D)
# ═══════════════════════════════════════════════════════════════
SIGMA2D = 10.0 / (7.0 * np.pi * cfg.H ** 2)

def W(r, h):
    """Cubic B-spline kernel value."""
    q = r / h
    s = np.zeros_like(q)
    mask1 = q < 1.0
    mask2 = (q >= 1.0) & (q < 2.0)
    s[mask1] = 1.0 - 1.5*q[mask1]**2 + 0.75*q[mask1]**3
    s[mask2] = 0.25*(2.0 - q[mask2])**3
    return SIGMA2D * s

def dW_dr(r, h):
    """Derivative dW/dr."""
    q = r / h
    ds = np.zeros_like(q)
    mask1 = (q > 1e-8) & (q < 1.0)
    mask2 = (q >= 1.0) & (q < 2.0)
    ds[mask1] = (-3.0*q[mask1] + 2.25*q[mask1]**2)
    ds[mask2] = -0.75*(2.0 - q[mask2])**2
    return SIGMA2D / h * ds

def grad_W(rx, ry, r, h):
    """Vector gradient of kernel."""
    dw = dW_dr(r, h)
    safe_r = np.where(r > 1e-8, r, 1.0)
    return dw / safe_r * rx,  dw / safe_r * ry


# ═══════════════════════════════════════════════════════════════
#  PARTICLE INITIALISATION
# ═══════════════════════════════════════════════════════════════
def init_particles():
    """Create fluid + boundary particles for a cylindrical tank cross-section."""
    dx = cfg.DX
    W2, H2 = cfg.TANK_W, cfg.TANK_H
    fill_h = 2.0 * H2 * cfg.FILL   # height of liquid column

    # ── Fluid particles (rectangular fill at bottom) ──────────────
    xs, ys = [], []
    x = -W2 + dx
    while x < W2 - dx * 0.5:
        y = -H2 + dx
        while y < -H2 + fill_h:
            xs.append(x)
            ys.append(y)
            y += dx
        x += dx
    fluid_x = np.array(xs, dtype=np.float64)
    fluid_y = np.array(ys, dtype=np.float64)
    N_fluid = len(fluid_x)

    # ── Boundary particles (walls + floor + ceiling) ──────────────
    bx, by = [], []
    for layer in range(cfg.N_BDRY_LAYERS):
        d = (layer + 1) * dx
        # bottom wall
        x = -W2 - cfg.N_BDRY_LAYERS*dx
        while x <= W2 + cfg.N_BDRY_LAYERS*dx:
            bx.append(x); by.append(-H2 - d)
            x += dx
        # top wall
        x = -W2 - cfg.N_BDRY_LAYERS*dx
        while x <= W2 + cfg.N_BDRY_LAYERS*dx:
            bx.append(x); by.append(H2 + d)
            x += dx
        # left wall
        y = -H2
        while y <= H2:
            bx.append(-W2 - d); by.append(y)
            y += dx
        # right wall
        y = -H2
        while y <= H2:
            bx.append(W2 + d); by.append(y)
            y += dx

    bdry_x = np.array(bx, dtype=np.float64)
    bdry_y = np.array(by, dtype=np.float64)
    N_bdry = len(bdry_x)

    # Initial conditions
    vx   = np.zeros(N_fluid)
    vy   = np.zeros(N_fluid)
    rho  = np.full(N_fluid, cfg.RHO0)
    mass = (cfg.RHO0 * dx ** 2) * np.ones(N_fluid)

    print(f"[INFO] Fluid particles:    {N_fluid}")
    print(f"[INFO] Boundary particles: {N_bdry}")
    return (fluid_x, fluid_y, vx, vy, rho, mass,
            bdry_x, bdry_y)


# ═══════════════════════════════════════════════════════════════
#  NUMPY SPH BACKEND  (reference / CPU fallback)
# ═══════════════════════════════════════════════════════════════
def tait_pressure(rho):
    """Weakly compressible Tait EOS."""
    B = cfg.RHO0 * cfg.C0**2 / cfg.GAMMA
    return B * ((rho / cfg.RHO0)**cfg.GAMMA - 1.0)

def numpy_sph_step(px, py, vx, vy, rho, mass,
                   bx, by, dt, t):
    """One full SPH time step (NumPy)."""
    N  = len(px)
    Nb = len(bx)
    h  = cfg.H
    h2 = (2.0 * h) ** 2

    # ── Gravity at time t ──────────────────────────────────────
    gx = cfg.SLOSH_AMP * np.sin(2*np.pi*cfg.SLOSH_FREQ * t)
    gy = -cfg.G_BASE
    if t > cfg.THRUST_T:
        gy -= cfg.THRUST_G    # extra thrust

    # ── Build neighbour list (brute-force O(N²), fine for N<3000) ─
    # Fluid–fluid distances
    dx_ff = px[:, None] - px[None, :]   # (N,N)
    dy_ff = py[:, None] - py[None, :]
    r2_ff = dx_ff**2 + dy_ff**2
    neigh = (r2_ff < h2) & (r2_ff > 1e-12)

    # Fluid–boundary distances
    dx_fb = px[:, None] - bx[None, :]   # (N,Nb)
    dy_fb = py[:, None] - by[None, :]
    r2_fb = dx_fb**2 + dy_fb**2
    neigh_b = r2_fb < h2

    # ── Density summation ─────────────────────────────────────
    r_ff = np.sqrt(np.maximum(r2_ff, 1e-20))
    rho_new = np.zeros(N)
    for i in range(N):
        js = np.where(neigh[i])[0]
        rij = r_ff[i, js]
        rho_new[i] = np.sum(mass[js] * W(rij, h))
        rho_new[i] += mass[i] * W(np.array([0.0]), h)[0]  # self

    # Boundary contribution to density (mirror mass)
    r_fb = np.sqrt(np.maximum(r2_fb, 1e-20))
    for i in range(N):
        bs = np.where(neigh_b[i])[0]
        if len(bs):
            rib = r_fb[i, bs]
            rho_new[i] += np.sum(mass[i] * W(rib, h))

    rho_new = np.maximum(rho_new, cfg.RHO0 * 0.5)

    # ── Pressure ──────────────────────────────────────────────
    p = tait_pressure(rho_new)

    # ── Acceleration ──────────────────────────────────────────
    ax = np.zeros(N)
    ay = np.zeros(N)

    for i in range(N):
        js = np.where(neigh[i])[0]
        if len(js) == 0:
            continue
        rij = r_ff[i, js]
        dxij = dx_ff[i, js]
        dyij = dy_ff[i, js]
        gwx, gwy = grad_W(dxij, dyij, rij, h)

        # Pressure gradient (symmetric)
        pi_rho2 = p[i]  / rho_new[i]**2
        pj_rho2 = p[js] / rho_new[js]**2
        fac = mass[js] * (pi_rho2 + pj_rho2)
        ax[i] -= np.sum(fac * gwx)
        ay[i] -= np.sum(fac * gwy)

        # Artificial viscosity (Monaghan 1992)
        dvx = vx[i] - vx[js]
        dvy = vy[i] - vy[js]
        vr  = dvx*dxij + dvy*dyij
        mu_ij = h * vr / (rij**2 + 0.01*h**2)
        visc_flag = vr < 0
        rho_avg = 0.5*(rho_new[i] + rho_new[js])
        pi_ij = np.where(visc_flag,
                         -cfg.ALPHA * cfg.C0 * mu_ij / rho_avg, 0.0)
        ax[i] -= np.sum(mass[js] * pi_ij * gwx)
        ay[i] -= np.sum(mass[js] * pi_ij * gwy)

    # ── Boundary repulsion (Lennard-Jones-like) ────────────────
    for i in range(N):
        bs = np.where(neigh_b[i])[0]
        if len(bs) == 0:
            continue
        rib = r_fb[i, bs]
        dxib = dx_fb[i, bs]
        dyib = dy_fb[i, bs]
        r0   = cfg.DX
        ratio = r0 / np.maximum(rib, 1e-6)
        rep  = cfg.C0**2 * (ratio**4 - ratio**2) / np.maximum(rib**2, 1e-8)
        rep  = np.where(rib < 2.0*r0, rep, 0.0)
        ax[i] += np.sum(rep * dxib)
        ay[i] += np.sum(rep * dyib)

    # ── Add body forces ────────────────────────────────────────
    ax += gx
    ay += gy

    # ── XSPH velocity correction ──────────────────────────────
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

    # ── Leapfrog integrate ────────────────────────────────────
    vx_new = vx + dt * ax
    vy_new = vy + dt * ay
    px_new = px + dt * (vx_new + vx_corr)
    py_new = py + dt * (vy_new + vy_corr)

    # Clamp inside tank (failsafe)
    m = 0.01
    px_new = np.clip(px_new, -cfg.TANK_W + m, cfg.TANK_W - m)
    py_new = np.clip(py_new, -cfg.TANK_H + m, cfg.TANK_H - m)

    return px_new, py_new, vx_new, vy_new, rho_new


# ═══════════════════════════════════════════════════════════════
#  WARP KERNEL DEFINITIONS  (only compiled if Warp is available)
# ═══════════════════════════════════════════════════════════════
if WARP_AVAILABLE:

    @wp.func
    def cubic_W(r: float, h: float) -> float:
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
        i = wp.tid()
        rho_i = mass[i] * cubic_W(float(0.0), h)   # self
        # fluid neighbours
        for j in range(N):
            if j == i:
                continue
            dx = px[i] - px[j]
            dy = py[i] - py[j]
            r  = wp.sqrt(dx*dx + dy*dy)
            if r < 2.0*h:
                rho_i += mass[j] * cubic_W(r, h)
        # boundary neighbours
        for b in range(Nb):
            dx = px[i] - bx[b]
            dy = py[i] - by[b]
            r  = wp.sqrt(dx*dx + dy*dy)
            if r < 2.0*h:
                rho_i += mass[i] * cubic_W(r, h)
        rho[i] = rho_i

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
        i = wp.tid()
        B    = rho0 * C0 * C0 / gamma
        pi_i = B * (wp.pow(rho[i]/rho0, gamma) - float(1.0))
        ax_i = float(0.0)
        ay_i = float(0.0)

        for j in range(N):
            if j == i:
                continue
            dxij = px[i] - px[j]
            dyij = py[i] - py[j]
            r    = wp.sqrt(dxij*dxij + dyij*dyij)
            if r >= 2.0*h:
                continue
            pj_j = B * (wp.pow(rho[j]/rho0, gamma) - float(1.0))
            dw   = cubic_dW(r, h)
            fac  = mass[j] * (pi_i/(rho[i]*rho[i]) + pj_j/(rho[j]*rho[j]))
            ax_i -= fac * dw / r * dxij
            ay_i -= fac * dw / r * dyij
            # Artificial viscosity
            vr = (vx[i]-vx[j])*dxij + (vy[i]-vy[j])*dyij
            if vr < float(0.0):
                mu   = h * vr / (r*r + float(0.01)*h*h)
                piij = -alpha * C0 * mu / (float(0.5)*(rho[i]+rho[j]))
                ax_i -= mass[j] * piij * dw / r * dxij
                ay_i -= mass[j] * piij * dw / r * dyij

        # Boundary repulsion
        for b in range(Nb):
            dxib = px[i] - bx[b]
            dyib = py[i] - by[b]
            rib  = wp.sqrt(dxib*dxib + dyib*dyib)
            if rib < float(2.0)*dx0 and rib > float(1e-6):
                ratio = dx0 / rib
                rep   = C0*C0 * (ratio*ratio*ratio*ratio - ratio*ratio) / (rib*rib)
                ax_i += rep * dxib
                ay_i += rep * dyib

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
        i = wp.tid()
        vx_new[i] = vx[i] + dt * ax[i]
        vy_new[i] = vy[i] + dt * ay[i]
        xn = px[i] + dt * vx_new[i]
        yn = py[i] + dt * vy_new[i]
        m = float(0.01)
        px_new[i] = wp.clamp(xn, -tank_w + m, tank_w - m)
        py_new[i] = wp.clamp(yn, -tank_h + m, tank_h - m)


    class WarpSPH:
        """Manages Warp arrays and GPU kernel launches."""
        def __init__(self, px, py, vx, vy, rho, mass, bx, by):
            self.N  = len(px)
            self.Nb = len(bx)
            self.device = "cuda" if wp.get_cuda_devices() else "cpu"
            print(f"[WARP] Running on: {self.device}")

            self.px   = wp.array(px.astype(np.float32),   dtype=float, device=self.device)
            self.py   = wp.array(py.astype(np.float32),   dtype=float, device=self.device)
            self.vx   = wp.array(vx.astype(np.float32),   dtype=float, device=self.device)
            self.vy   = wp.array(vy.astype(np.float32),   dtype=float, device=self.device)
            self.rho  = wp.array(rho.astype(np.float32),  dtype=float, device=self.device)
            self.mass = wp.array(mass.astype(np.float32), dtype=float, device=self.device)
            self.bx   = wp.array(bx.astype(np.float32),   dtype=float, device=self.device)
            self.by   = wp.array(by.astype(np.float32),   dtype=float, device=self.device)
            self.ax   = wp.zeros(self.N, dtype=float, device=self.device)
            self.ay   = wp.zeros(self.N, dtype=float, device=self.device)
            self.px_n = wp.zeros(self.N, dtype=float, device=self.device)
            self.py_n = wp.zeros(self.N, dtype=float, device=self.device)
            self.vx_n = wp.zeros(self.N, dtype=float, device=self.device)
            self.vy_n = wp.zeros(self.N, dtype=float, device=self.device)

        def step(self, dt, t):
            N, Nb = self.N, self.Nb
            gx = float(cfg.SLOSH_AMP * np.sin(2*np.pi*cfg.SLOSH_FREQ * t))
            gy = float(-cfg.G_BASE - (cfg.THRUST_G if t > cfg.THRUST_T else 0.0))

            wp.launch(density_kernel,
                      dim=N, inputs=[self.px, self.py, self.bx, self.by,
                                     self.mass, self.rho,
                                     float(cfg.H), N, Nb])
            wp.launch(force_kernel,
                      dim=N, inputs=[self.px, self.py, self.vx, self.vy,
                                     self.bx, self.by, self.mass, self.rho,
                                     self.ax, self.ay,
                                     float(cfg.H), float(cfg.C0),
                                     float(cfg.ALPHA), float(cfg.RHO0),
                                     float(cfg.GAMMA), gx, gy,
                                     float(cfg.DX), N, Nb])
            wp.launch(integrate_kernel,
                      dim=N, inputs=[self.px, self.py, self.vx, self.vy,
                                     self.ax, self.ay,
                                     self.px_n, self.py_n, self.vx_n, self.vy_n,
                                     dt,
                                     float(cfg.TANK_W), float(cfg.TANK_H)])
            # swap buffers
            self.px, self.px_n = self.px_n, self.px
            self.py, self.py_n = self.py_n, self.py
            self.vx, self.vx_n = self.vx_n, self.vx
            self.vy, self.vy_n = self.vy_n, self.vy

        def get_state(self):
            return (self.px.numpy(), self.py.numpy(),
                    self.vx.numpy(), self.vy.numpy(),
                    self.rho.numpy())


# ═══════════════════════════════════════════════════════════════
#  UNIFIED SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════
def run_simulation():
    px, py, vx, vy, rho, mass, bx, by = init_particles()

    if WARP_AVAILABLE:
        solver = WarpSPH(px, py, vx, vy, rho, mass, bx, by)
    # else use plain arrays

    n_frames = int(cfg.T_END * cfg.FPS)
    dt_frame = 1.0 / cfg.FPS
    steps_per_frame = max(1, int(dt_frame / cfg.DT))

    # ── Matplotlib figure ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#0a0a1a")
    ax_sim, ax_info = axes
    for ax in axes:
        ax.set_facecolor("#0a0a1a")

    ax_sim.set_xlim(-cfg.TANK_W*1.1, cfg.TANK_W*1.1)
    ax_sim.set_ylim(-cfg.TANK_H*1.1, cfg.TANK_H*1.1)
    ax_sim.set_aspect("equal")
    ax_sim.set_title("SPH Propellant Sloshing — Satellite Tank",
                     color="white", fontsize=13, pad=8)
    ax_sim.tick_params(colors="gray")
    for spine in ax_sim.spines.values():
        spine.set_edgecolor("#333355")

    # Draw tank outline
    tank_rect = patches.Rectangle(
        (-cfg.TANK_W, -cfg.TANK_H), 2*cfg.TANK_W, 2*cfg.TANK_H,
        linewidth=2, edgecolor="#44aaff", facecolor="none", zorder=5,
        label="Tank wall")
    ax_sim.add_patch(tank_rect)

    # Boundary particles
    ax_sim.scatter(bx, by, s=2, c="#334455", alpha=0.5, zorder=2)

    # Fluid scatter
    speeds = np.sqrt(vx**2 + vy**2)
    sc = ax_sim.scatter(px, py, s=6, c=speeds,
                        cmap=cfg.COLORMAP, vmin=0, vmax=3.0, zorder=3)
    cbar = fig.colorbar(sc, ax=ax_sim, fraction=0.03, pad=0.02)
    cbar.set_label("Speed [m/s]", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Gravity arrow
    arrow = ax_sim.annotate("", xy=(0, 0), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="-|>", color="red",
                                             lw=2.5))
    grav_label = ax_sim.text(0.02, 0.96, "", transform=ax_sim.transAxes,
                             color="red", fontsize=9, va="top")

    # Time label
    time_label = ax_sim.text(0.02, 0.02, "t = 0.00 s",
                              transform=ax_sim.transAxes,
                              color="white", fontsize=10)
    backend_label = ax_sim.text(0.98, 0.02,
        "Backend: WARP (GPU)" if WARP_AVAILABLE else "Backend: NumPy (CPU)",
        transform=ax_sim.transAxes, color="#88ccff", fontsize=8,
        ha="right")

    # ── Info panel ────────────────────────────────────────────
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

    # History for ΔKE plot
    ke_history = []
    t_history  = []

    def update(frame):
        nonlocal px, py, vx, vy, rho

        t_now = frame * dt_frame

        # ── Integrate ─────────────────────────────────────────
        t0 = time.perf_counter()
        if WARP_AVAILABLE:
            for _ in range(steps_per_frame):
                t_sub = t_now + _ * cfg.DT
                solver.step(cfg.DT, t_sub)
            px, py, vx, vy, rho = solver.get_state()
        else:
            for _ in range(steps_per_frame):
                t_sub = t_now + _ * cfg.DT
                px, py, vx, vy, rho = numpy_sph_step(
                    px, py, vx, vy, rho, mass, bx, by, cfg.DT, t_sub)
        wall = time.perf_counter() - t0

        # ── Update scatter ────────────────────────────────────
        speeds = np.sqrt(vx**2 + vy**2)
        sc.set_offsets(np.column_stack([px, py]))
        sc.set_array(speeds)

        # ── Gravity arrow ─────────────────────────────────────
        gx_now = cfg.SLOSH_AMP * np.sin(2*np.pi*cfg.SLOSH_FREQ * t_now)
        gy_now = -cfg.G_BASE - (cfg.THRUST_G if t_now > cfg.THRUST_T else 0.0)
        scale = 0.2
        cx, cy = 0.0, cfg.TANK_H * 0.75
        ex, ey = cx + gx_now*scale, cy + gy_now*scale
        arrow.set_position((ex, ey))
        arrow.xy = (cx, cy)
        grav_label.set_text(f"g=({gx_now:.1f}, {gy_now:.1f}) m/s²")

        # ── Telemetry ─────────────────────────────────────────
        ke = 0.5 * np.sum(mass * (vx**2 + vy**2))
        ke_history.append(ke)
        t_history.append(t_now)
        phase = "Maneuver" if t_now > cfg.THRUST_T else "Sloshing"
        vals  = [phase, f"{t_now:.2f}", f"{gx_now:+.2f}",
                 f"{gy_now:+.2f}", f"{speeds.max():.2f} m/s",
                 f"{rho.mean()/cfg.RHO0:.4f}",
                 f"{ke:.2f} J"]
        for lb, v in zip(labels, vals):
            info_texts[lb].set_text(v)

        time_label.set_text(f"t = {t_now:.2f} s  ({1/max(wall,1e-4):.0f} fps)")

        return sc, arrow, grav_label, time_label, *info_texts.values()

    # ── Animate ───────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════
#  KINETIC ENERGY DIAGNOSTIC PLOT
# ═══════════════════════════════════════════════════════════════
def plot_diagnostics(t_history, ke_history):
    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#0a0a1a")
    ax.plot(t_history, ke_history, color="#44aaff", lw=1.5)
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


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print(" 2D SPH Satellite Propellant Tank Sloshing")
    print("=" * 60)
    t_hist, ke_hist = run_simulation()
    plot_diagnostics(t_hist, ke_hist)
    print("[DONE]")
