"""
flexible_spacecraft.py
======================
Simulation of a rigid-hub + flexible-appendage spacecraft under PID control.

PHYSICS MODEL
-------------
Hub (rigid body):
    J * θ̈ = u - δ * η̈

Flexible mode (modal coordinate η):
    η̈ + 2ζωₙη̇ + ωₙ²η = -δ * θ̈

Where:
    θ   = hub angle (rad)
    η   = modal deflection of flexible appendage
    u   = control torque (N·m)
    J   = hub inertia (kg·m²)
    ωₙ  = flexible mode natural frequency (rad/s)
    ζ   = modal damping ratio (very small in space!)
    δ   = coupling coefficient

PID CONTROLLER
--------------
    e   = θ_ref - θ
    u   = Kp*e + Kd*ė + Ki*∫e dt

NOTCH FILTER (biquad IIR)
-------------------------
Applied to the control signal to suppress gain at the flex frequency.

USAGE
-----
Run directly to see all four scenarios:
    python flexible_spacecraft.py

Or import and use the classes for your own experiments:
    from flexible_spacecraft import SystemParams, ControllerParams, simulate, plot_results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyArrowPatch
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.animation as animation


# ─── PARAMETER DATACLASSES ──────────────────────────────────────────────────

@dataclass
class SystemParams:
    """Physical parameters of the flexible spacecraft."""
    J: float = 10.0       # Hub moment of inertia (kg·m²)
    wn: float = 3.2       # Flexible mode natural frequency (rad/s)
    zeta: float = 0.005   # Modal damping ratio (very low — space structure)
    delta: float = 0.35   # Hub-appendage coupling coefficient


@dataclass
class ControllerParams:
    """PID controller gains."""
    Kp: float = 8.0       # Proportional gain
    Kd: float = 4.0       # Derivative gain
    Ki: float = 0.0       # Integral gain
    u_max: float = 200.0  # Actuator saturation (N·m)
    windup_limit: float = 5.0  # Anti-windup clamp on integral term


@dataclass
class NotchParams:
    """Biquad notch filter parameters."""
    enabled: bool = False
    wn: float = 3.2       # Notch center frequency (rad/s) — tune to flex freq
    Q: float = 8.0        # Quality factor (sharpness of notch)
    depth_db: float = 20.0  # Attenuation at notch center (dB)


@dataclass
class SimParams:
    """Simulation settings."""
    dt: float = 0.005         # Integration timestep (s)
    t_end: float = 30.0       # Simulation duration (s)
    theta_ref_deg: float = 10.0  # Step command (degrees)
    theta0_deg: float = 0.0   # Initial hub angle (degrees)


# ─── NOTCH FILTER ────────────────────────────────────────────────────────────

class NotchFilter:
    """
    Second-order IIR notch filter (biquad) using bilinear transform.

    Continuous-time prototype:
        H(s) = (s² + w0²) / (s² + w0/Q * s + w0²)

    Mixed with passthrough to achieve desired depth:
        H_notch(s) = alpha * H(s) + (1-alpha) * 1
    where alpha controls depth.
    """

    def __init__(self, params: NotchParams, dt: float):
        self.params = params
        self.dt = dt
        self._compute_coeffs()
        self.reset()

    def _compute_coeffs(self):
        """Compute biquad coefficients via bilinear transform."""
        w0 = self.params.wn
        Q = self.params.Q
        dt = self.dt

        # Pre-warp frequency for bilinear transform
        w0_d = 2 / dt * np.tan(w0 * dt / 2)

        cosw = np.cos(w0 * dt)
        alpha = np.sin(w0 * dt) / (2 * Q)

        # Standard notch biquad coefficients (unit gain at DC and Nyquist)
        b0 = 1.0
        b1 = -2 * cosw
        b2 = 1.0
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

        # Normalize
        self.b = np.array([b0, b1, b2]) / a0
        self.a = np.array([1.0, a1 / a0, a2 / a0])

        # Depth scaling: blend notch with all-pass
        # At center freq, notch gives 0; blend gives: depth_linear
        depth_linear = 10 ** (-self.params.depth_db / 20)
        self.depth_linear = depth_linear

    def reset(self):
        """Reset filter state (delay lines)."""
        self.x_hist = [0.0, 0.0]  # x[n-1], x[n-2]
        self.y_hist = [0.0, 0.0]  # y[n-1], y[n-2]

    def process(self, x: float) -> float:
        """Apply notch filter to input sample x, return filtered output."""
        if not self.params.enabled:
            return x

        # Direct Form II transposed
        y = (self.b[0] * x
             + self.b[1] * self.x_hist[0]
             + self.b[2] * self.x_hist[1]
             - self.a[1] * self.y_hist[0]
             - self.a[2] * self.y_hist[1])

        self.x_hist[1] = self.x_hist[0]
        self.x_hist[0] = x
        self.y_hist[1] = self.y_hist[0]
        self.y_hist[0] = y

        return y


# ─── PID CONTROLLER ──────────────────────────────────────────────────────────

class PIDController:
    """
    PID controller with anti-windup and optional notch filtering.

    The notch filter is applied to the control signal to remove
    controller authority at the flexible mode frequency — preventing
    spillover instability.
    """

    def __init__(self, ctrl: ControllerParams, notch: NotchParams, dt: float):
        self.ctrl = ctrl
        self.dt = dt
        self.notch = NotchFilter(notch, dt)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, theta_ref: float, theta: float) -> tuple[float, float]:
        """
        Compute control torque for one timestep.

        Args:
            theta_ref: Reference angle (rad)
            theta:     Current hub angle (rad)

        Returns:
            (u, error): control torque and tracking error
        """
        error = theta_ref - theta

        # Derivative (backward difference)
        error_dot = (error - self.prev_error) / self.dt
        self.prev_error = error

        # Integral with anti-windup
        self.integral = np.clip(
            self.integral + error * self.dt,
            -self.ctrl.windup_limit,
            self.ctrl.windup_limit
        )

        # PID output
        u_raw = (self.ctrl.Kp * error
                 + self.ctrl.Kd * error_dot
                 + self.ctrl.Ki * self.integral)

        # Notch filter on control signal
        u_filtered = self.notch.process(u_raw)

        # Actuator saturation
        u = np.clip(u_filtered, -self.ctrl.u_max, self.ctrl.u_max)

        return u, error


# ─── DYNAMICS ────────────────────────────────────────────────────────────────

def dynamics(state: np.ndarray, u: float, sys: SystemParams) -> np.ndarray:
    """
    Compute state derivatives for the coupled hub-flexible system.

    State vector: [θ, θ̇, η, η̇]

    Equations of motion (coupled):
        J*θ̈ = u - δ*η̈
        η̈ + 2ζωₙη̇ + ωₙ²η = -δ*θ̈

    Solving simultaneously for θ̈ and η̈:
        [  J    δ ] [θ̈]   [u                    ]
        [  δ    1 ] [η̈] = [-2ζωₙη̇ - ωₙ²η        ]

    => θ̈ = (u + δ*(2ζωₙη̇ + ωₙ²η)) / (J - δ²)
    => η̈ = (-2ζωₙη̇ - ωₙ²η - δ*θ̈)

    Args:
        state: [theta, theta_dot, eta, eta_dot]
        u:     control torque
        sys:   system parameters

    Returns:
        dstate/dt: [theta_dot, theta_ddot, eta_dot, eta_ddot]
    """
    theta, theta_dot, eta, eta_dot = state

    # Effective inertia after coupling
    J_eff = sys.J - sys.delta ** 2  # coupling reduces effective inertia

    # Solve coupled equations
    rhs_hub = u + sys.delta * (2 * sys.zeta * sys.wn * eta_dot + sys.wn**2 * eta)
    theta_ddot = rhs_hub / J_eff

    eta_ddot = (-2 * sys.zeta * sys.wn * eta_dot
                - sys.wn**2 * eta
                - sys.delta * theta_ddot)

    return np.array([theta_dot, theta_ddot, eta_dot, eta_ddot])


def rk4_step(state: np.ndarray, u: float, sys: SystemParams, dt: float) -> np.ndarray:
    """
    4th-order Runge-Kutta integration step.

    More accurate than Euler for oscillatory systems — important here
    because the flexible mode oscillates at ωₙ ~ 3 rad/s and we need
    to capture it faithfully.
    """
    k1 = dynamics(state, u, sys)
    k2 = dynamics(state + 0.5 * dt * k1, u, sys)
    k3 = dynamics(state + 0.5 * dt * k2, u, sys)
    k4 = dynamics(state + dt * k3, u, sys)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# ─── SIMULATION ──────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    """Container for simulation output."""
    t: np.ndarray
    theta: np.ndarray       # Hub angle (deg)
    theta_ref: np.ndarray   # Reference command (deg)
    theta_dot: np.ndarray   # Hub rate (deg/s)
    eta: np.ndarray         # Modal deflection
    eta_dot: np.ndarray     # Modal rate
    u: np.ndarray           # Control torque (N·m)
    error: np.ndarray       # Tracking error (deg)
    flex_energy: np.ndarray # Modal energy: 0.5*(η̇² + ωₙ²η²)
    label: str = ""
    sys: Optional[SystemParams] = None


def simulate(
    sys: SystemParams,
    ctrl: ControllerParams,
    notch: NotchParams,
    sim: SimParams,
    label: str = ""
) -> SimResult:
    """
    Run a full simulation.

    Args:
        sys:   System (plant) parameters
        ctrl:  PID controller gains
        notch: Notch filter settings
        sim:   Simulation settings
        label: Name for plots

    Returns:
        SimResult with full time histories
    """
    n_steps = int(sim.t_end / sim.dt)
    t = np.linspace(0, sim.t_end, n_steps)

    # Reference (step command)
    theta_ref_rad = sim.theta_ref_deg * np.pi / 180

    # Initial state
    state = np.array([sim.theta0_deg * np.pi / 180, 0.0, 0.0, 0.0])

    # Initialize controller
    pid = PIDController(ctrl, notch, sim.dt)

    # Storage
    theta_hist   = np.zeros(n_steps)
    theta_dot_hist = np.zeros(n_steps)
    eta_hist     = np.zeros(n_steps)
    eta_dot_hist = np.zeros(n_steps)
    u_hist       = np.zeros(n_steps)
    err_hist     = np.zeros(n_steps)
    fe_hist      = np.zeros(n_steps)

    for i in range(n_steps):
        # Control
        u, error = pid.step(theta_ref_rad, state[0])

        # Store
        theta_hist[i]     = state[0] * 180 / np.pi
        theta_dot_hist[i] = state[1] * 180 / np.pi
        eta_hist[i]       = state[2]
        eta_dot_hist[i]   = state[3]
        u_hist[i]         = u
        err_hist[i]       = error * 180 / np.pi
        fe_hist[i]        = 0.5 * (state[3]**2 + sys.wn**2 * state[2]**2)

        # Integrate
        state = rk4_step(state, u, sys, sim.dt)

        # Safety: stop if diverged
        if np.any(np.abs(state) > 1e4):
            print(f"  [{label}] Diverged at t={t[i]:.2f}s — truncating.")
            n_steps = i + 1
            t = t[:n_steps]
            theta_hist   = theta_hist[:n_steps]
            theta_dot_hist = theta_dot_hist[:n_steps]
            eta_hist     = eta_hist[:n_steps]
            eta_dot_hist = eta_dot_hist[:n_steps]
            u_hist       = u_hist[:n_steps]
            err_hist     = err_hist[:n_steps]
            fe_hist      = fe_hist[:n_steps]
            break

    return SimResult(
        t=t,
        theta=theta_hist,
        theta_ref=np.full_like(t, sim.theta_ref_deg),
        theta_dot=theta_dot_hist,
        eta=eta_hist,
        eta_dot=eta_dot_hist,
        u=u_hist,
        error=err_hist,
        flex_energy=fe_hist,
        label=label,
        sys=sys
    )


# ─── PLOTTING ────────────────────────────────────────────────────────────────

# Color palette (dark theme)
COLORS = {
    'bg':       '#0a0e17',
    'panel':    '#0f1520',
    'border':   '#1e2d45',
    'theta':    '#00d4ff',
    'ref':      '#ffcc00',
    'eta':      '#7fff6b',
    'u':        '#ff6b35',
    'fe':       '#cc77ff',
    'text':     '#c8daf0',
    'muted':    '#4a6080',
    'stable':   '#7fff6b',
    'warn':     '#ffcc00',
    'danger':   '#ff3366',
}


def apply_dark_style():
    """Apply dark GNC-lab style to matplotlib."""
    plt.rcParams.update({
        'figure.facecolor':  COLORS['bg'],
        'axes.facecolor':    COLORS['panel'],
        'axes.edgecolor':    COLORS['border'],
        'axes.labelcolor':   COLORS['text'],
        'axes.titlecolor':   COLORS['text'],
        'xtick.color':       COLORS['muted'],
        'ytick.color':       COLORS['muted'],
        'grid.color':        COLORS['border'],
        'grid.linewidth':    0.8,
        'legend.facecolor':  COLORS['panel'],
        'legend.edgecolor':  COLORS['border'],
        'legend.labelcolor': COLORS['text'],
        'text.color':        COLORS['text'],
        'font.family':       'monospace',
        'font.size':         9,
        'axes.titlesize':    10,
        'axes.labelsize':    9,
    })


def plot_results(results: list[SimResult], title: str = "Flexible Spacecraft Simulation"):
    """
    Plot time histories for one or more simulation results.

    Subplots:
        1. Hub angle θ vs reference
        2. Flex modal deflection η
        3. Control torque u
        4. Flex modal energy
    """
    apply_dark_style()
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight='bold', color='white', y=0.98)

    line_styles = ['-', '--', '-.', ':']

    for idx, r in enumerate(results):
        ls = line_styles[idx % len(line_styles)]
        lw = 1.8

        axes[0].plot(r.t, r.theta, color=COLORS['theta'], ls=ls, lw=lw, label=f'{r.label} θ (hub)')
        if idx == 0:
            axes[0].plot(r.t, r.theta_ref, color=COLORS['ref'], ls='--', lw=1.2, alpha=0.7, label='θ ref')

        axes[1].plot(r.t, r.eta, color=COLORS['eta'], ls=ls, lw=lw, label=r.label)
        axes[2].plot(r.t, r.u, color=COLORS['u'], ls=ls, lw=1.2, alpha=0.8, label=r.label)
        axes[3].plot(r.t, r.flex_energy, color=COLORS['fe'], ls=ls, lw=lw, label=r.label)

    labels = [
        ('θ (deg)', 'Hub Angle'),
        ('η (modal coord)', 'Flex Deflection'),
        ('u (N·m)', 'Control Torque'),
        ('½(η̇² + ωₙ²η²)', 'Flex Modal Energy'),
    ]

    for ax, (ylabel, title_str) in zip(axes, labels):
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title_str, loc='left', fontsize=9, color=COLORS['muted'])
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8, loc='upper right')
        ax.axhline(0, color=COLORS['border'], lw=0.8)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    return fig


def plot_bode(sys: SystemParams, ctrl: ControllerParams,
              notch: Optional[NotchParams] = None,
              dt: float = 0.005):
    """
    Plot Bode diagram of the open-loop transfer function L(jω) = C(jω)·G(jω).

    This shows:
    - The plant resonance (flexible mode pole pair)
    - Controller magnitude and phase
    - Where spillover occurs (controller has gain at flex freq)
    - How a notch filter removes gain at the resonance

    The flexible mode appears as a sharp peak in magnitude and a -180° 
    phase drop — this is what drives spillover instability when the
    controller still has significant gain there.
    """
    apply_dark_style()

    omega = np.logspace(-2, 2, 2000)  # rad/s
    s = 1j * omega

    # Plant transfer function: G(s) = 1/(J*s²) * (rigid) coupled with flex
    # Full transfer function from u to theta (approximate, ignoring coupling for Bode clarity):
    # G(s) = [1/J] * 1/s² * [1 + δ²*(s² + 2ζωₙs... ) / ...]
    # For educational purposes, show the plant with the flex mode as a notch/peak pair:
    # G_flex(s) = (s² + 2ζωₙs + ωₙ²) / (J*s²*(s² + 2ζ_ctrl*ωₙs + ωₙ²))
    # Simplified: rigid body + flex mode resonance visible in output

    J = sys.J
    wn = sys.wn
    zeta = sys.zeta
    delta = sys.delta

    # Rigid body
    G_rigid = 1 / (J * s**2)

    # Flexible mode influence (creates a resonant peak in the output)
    G_flex_num = s**2 + 2*zeta*wn*s + wn**2
    G_flex_den = s**2 + 2*zeta*wn*s + wn**2 + delta**2 * wn**2
    G_plant = G_rigid * (G_flex_den / G_flex_num)  # approximate

    # PID controller: C(s) = Kp + Ki/s + Kd*s
    # Add a small roll-off pole to make Kd proper: Kd*s / (1 + s/wc)
    wc = 50  # derivative filter cutoff
    C_pid = ctrl.Kp + ctrl.Ki / s + ctrl.Kd * s / (1 + s/wc)

    # Notch filter frequency response
    if notch and notch.enabled:
        w0 = notch.wn
        Q = notch.Q
        H_notch = (s**2 + w0**2) / (s**2 + w0/Q*s + w0**2)
        C_total = C_pid * H_notch
    else:
        C_total = C_pid
        H_notch = None

    # Open-loop
    L = C_total * G_plant

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('Bode Plot — Open-Loop L(jω) = C(jω)·G(jω)', 
                 fontsize=12, color='white')

    mag_plant = 20 * np.log10(np.abs(G_plant))
    mag_ctrl  = 20 * np.log10(np.abs(C_pid))
    mag_L     = 20 * np.log10(np.abs(L))
    phase_L   = np.angle(L, deg=True)

    ax1.semilogx(omega, mag_plant, color=COLORS['eta'], lw=1.5, ls='--',
                 label='Plant G(jω)', alpha=0.7)
    ax1.semilogx(omega, mag_ctrl, color=COLORS['u'], lw=1.5, ls='--',
                 label='PID C(jω)', alpha=0.7)
    ax1.semilogx(omega, mag_L, color=COLORS['theta'], lw=2,
                 label='Open-loop L(jω)')

    if notch and notch.enabled:
        mag_notch = 20 * np.log10(np.abs((s**2 + notch.wn**2) / 
                                          (s**2 + notch.wn/notch.Q*s + notch.wn**2)))
        ax1.semilogx(omega, mag_ctrl + mag_notch, color='#cc77ff', lw=1.5, ls=':',
                     label='C with Notch', alpha=0.9)

    # Mark flex mode frequency
    ax1.axvline(wn, color=COLORS['danger'], lw=1.5, ls=':', alpha=0.8)
    ax1.text(wn*1.05, ax1.get_ylim()[0] if ax1.get_ylim()[0] > -200 else -150,
             f'ωₙ={wn}', color=COLORS['danger'], fontsize=8, va='bottom')

    ax1.axhline(0, color=COLORS['ref'], lw=0.8, ls='--', alpha=0.5, label='0 dB')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_ylim(-120, 60)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title('Magnitude', loc='left', color=COLORS['muted'])

    ax2.semilogx(omega, phase_L, color=COLORS['theta'], lw=2, label='Phase L(jω)')
    ax2.axvline(wn, color=COLORS['danger'], lw=1.5, ls=':', alpha=0.8)
    ax2.axhline(-180, color=COLORS['ref'], lw=0.8, ls='--', alpha=0.5, label='-180°')
    ax2.set_ylabel('Phase (deg)')
    ax2.set_xlabel('Frequency (rad/s)')
    ax2.set_ylim(-360, 90)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title('Phase', loc='left', color=COLORS['muted'])

    plt.tight_layout()
    return fig


def plot_spacecraft_snapshot(result: SimResult, t_snap: float = 5.0):
    """
    Draw a snapshot of the spacecraft at a given time,
    showing hub angle and flexible appendage deflection.
    """
    apply_dark_style()

    # Find closest time index
    idx = np.argmin(np.abs(result.t - t_snap))
    theta = result.theta[idx] * np.pi / 180
    eta   = result.eta[idx]
    theta_ref = result.theta_ref[idx] * np.pi / 180

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{result.label}  |  t = {t_snap:.1f}s', color='white')

    cx, cy = 0, 0
    hub_r = 20

    # Reference direction
    ref_end = (120 * np.cos(theta_ref), 120 * np.sin(theta_ref))
    ax.annotate('', xy=ref_end, xytext=(cx,cy),
                arrowprops=dict(arrowstyle='->', color=COLORS['ref'], lw=1.5,
                                linestyle='dashed', alpha=0.5))

    # Hub circle
    hub = Circle((cx,cy), hub_r, fill=True, facecolor='#0d1828',
                 edgecolor=COLORS['theta'], lw=2)
    ax.add_patch(hub)

    # Flexible appendage (cantilever with mode shape)
    n_seg = 30
    beam_len = 120
    s_vals = np.linspace(0, 1, n_seg+1)
    mode_shape = np.sin(np.pi/2 * s_vals)  # cantilever first mode shape

    x_beam, y_beam = [], []
    for s, phi in zip(s_vals, mode_shape):
        local_x = s * beam_len + hub_r
        local_y = eta * phi * 30  # visual scale
        gx = cx + np.cos(theta)*local_x - np.sin(theta)*local_y
        gy = cy + np.sin(theta)*local_x + np.cos(theta)*local_y
        x_beam.append(gx)
        y_beam.append(gy)

    ax.plot(x_beam, y_beam, color=COLORS['eta'], lw=3, zorder=3)
    ax.plot(x_beam[-1], y_beam[-1], 'o', color=COLORS['u'],
            markersize=10, zorder=4)  # tip mass

    # Opposite stub
    ax.plot([cx - np.cos(theta)*hub_r, cx - np.cos(theta)*70],
            [cy - np.sin(theta)*hub_r, cy - np.sin(theta)*70],
            color=COLORS['eta'], lw=2, alpha=0.4)

    # Labels
    ax.text(0, -160, f'θ = {result.theta[idx]:.1f}°', color=COLORS['theta'],
            ha='center', fontsize=10, fontfamily='monospace')
    ax.text(0, -145, f'η = {result.eta[idx]:.4f}', color=COLORS['eta'],
            ha='center', fontsize=10, fontfamily='monospace')

    # Legend
    ax.plot([], [], color=COLORS['theta'], label='Hub / beam axis')
    ax.plot([], [], color=COLORS['eta'], label='Flexible appendage')
    ax.plot([], [], color=COLORS['ref'], ls='--', label='θ reference')
    ax.legend(loc='lower right', fontsize=8)

    return fig


# ─── PREDEFINED SCENARIOS ────────────────────────────────────────────────────

def scenario_nominal() -> tuple:
    """Nominal rigid-body PID — baseline, stable."""
    return (
        SystemParams(wn=3.2),
        ControllerParams(Kp=8, Kd=4, Ki=0),
        NotchParams(enabled=False),
        SimParams(t_end=25, theta_ref_deg=10),
        "Nominal (Kp=8)"
    )


def scenario_soft_mode() -> tuple:
    """Soft flexible mode — lower ωₙ, harder to avoid."""
    return (
        SystemParams(wn=1.2),
        ControllerParams(Kp=8, Kd=4, Ki=0),
        NotchParams(enabled=False),
        SimParams(t_end=30, theta_ref_deg=10),
        "Soft Mode (ωₙ=1.2)"
    )


def scenario_spillover() -> tuple:
    """High gain — spillover drives flexible mode unstable."""
    return (
        SystemParams(wn=3.2),
        ControllerParams(Kp=45, Kd=4, Ki=0),
        NotchParams(enabled=False),
        SimParams(t_end=20, theta_ref_deg=10),
        "Spillover (Kp=45)"
    )


def scenario_notch_fix() -> tuple:
    """Same high gain, but notch filter stabilises the system."""
    return (
        SystemParams(wn=3.2),
        ControllerParams(Kp=45, Kd=4, Ki=0),
        NotchParams(enabled=True, wn=3.2, Q=8, depth_db=20),
        SimParams(t_end=25, theta_ref_deg=10),
        "Notch Fix (Kp=45 + Notch)"
    )


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FLEXIBLE SPACECRAFT — GNC SIMULATION")
    print("=" * 60)

    scenarios = [
        scenario_nominal(),
        scenario_soft_mode(),
        scenario_spillover(),
        scenario_notch_fix(),
    ]

    results = []
    for sys, ctrl, notch, sim, label in scenarios:
        print(f"\nRunning: {label}")
        print(f"  ωₙ={sys.wn} rad/s | ζ={sys.zeta} | δ={sys.delta}")
        print(f"  Kp={ctrl.Kp} | Kd={ctrl.Kd} | Ki={ctrl.Ki}")
        print(f"  Notch: {'ON @ ' + str(notch.wn) + ' rad/s' if notch.enabled else 'OFF'}")
        r = simulate(sys, ctrl, notch, sim, label)
        results.append(r)
        final_err = abs(r.error[-1])
        max_eta   = np.max(np.abs(r.eta))
        print(f"  → Final error: {final_err:.3f}°  |  Max |η|: {max_eta:.4f}")

    print("\nGenerating plots...")

    # Plot 1: All scenarios on one figure
    fig1 = plot_results(results, "Four Scenarios — Flexible Spacecraft")

    # Plot 2: Spillover vs Notch comparison
    spillover = results[2]
    notch_fix = results[3]
    fig2 = plot_results([spillover, notch_fix], "Spillover vs Notch Fix")

    # Plot 3: Bode plot (nominal vs notch)
    sys_nom, ctrl_nom, _, _, _ = scenarios[0]
    _, _, notch_params, _, _ = scenarios[3]
    fig3 = plot_bode(sys_nom, ctrl_nom, notch=None)
    fig4 = plot_bode(sys_nom, ctrl_nom, notch=notch_params)
    fig4.suptitle('Bode Plot — With Notch Filter @ ωₙ=3.2 rad/s', 
                  fontsize=12, color='white')

    # Plot 4: Spacecraft snapshot
    fig5 = plot_spacecraft_snapshot(results[0], t_snap=5.0)
    fig6 = plot_spacecraft_snapshot(results[2], t_snap=8.0)

    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
