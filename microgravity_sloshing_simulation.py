"""
Low-g Propellant Sloshing Simulation — Microgravity Regime
===========================================================
Implements the Dodge (2000) / Jang et al. (2014) low-g slosh parameter model
for a cylindrical tank, including:

  - Bond-number-dependent slosh frequency (gravity + surface tension contributions)
  - Low-g smooth wall damping (Galileo number based)
  - Regime detection and model switching between high-g (Abramson) and low-g (Dodge)
  - Free surface stability check (critical acceleration)
  - Correction for elliptical dome tanks (barrel section)
  - 2D parameter table over fill fraction AND Bond number
  - Time-domain simulation of coupled spacecraft + slosh dynamics

References:
  [1] Abramson, H.N. (1966). The Dynamic Behavior of Liquids in Moving Containers.
      NASA SP-106.
  [2] Dodge, F.T. (2000). The New Dynamic Behavior of Liquids in Moving Containers.
      Southwest Research Institute.
  [3] Jang, J-W., Alaniz, A., Yang, L., Powers, J., Hall, C. (2014).
      Mechanical Slosh Models for Rocket-Propelled Spacecraft.
      Draper Laboratory / NASA MSFC. NTRS 20140002967.

Usage:
    python microgravity_sloshing.py
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Tank geometry ---
R0      = 0.4        # Tank inner radius [m]
dome_d  = 0.10        # Elliptical dome depth [m] (set 0 for flat bottom)
h_fill  = 0.50        # Propellant fill height [m]

# --- Propellant properties ---
rho     = 200 #880.0       # Liquid density [kg/m³]  (hydrazine-like)
sigma   = 0.050       # Surface tension [N/m]
nu_k    = 1.0e-6      # Kinematic viscosity [m²/s]

# --- Mission phase: effective axial acceleration ---
# Change this to explore different regimes
# Coast:              a_eff ~ 1e-5  (Bo << 1, surface tension dominated)
# Small RCS manoeuvre: a_eff ~ 1e-3  (Bo ~ 2)
# Station-keeping:    a_eff ~ 0.05  (Bo ~ 100)
# Orbit raising:      a_eff ~ 2.0   (Bo ~ 4000, → high-g)
a_eff   = 0.05        # Effective axial acceleration [m/s²]

# --- Spacecraft ---
I_sc    = 500.0       # Pitch inertia [kg.m²]
z_tank_centre = 0.5   # Tank centre axial position from CoM [m]

# --- Control ---
Kp = 2.0             # Attitude proportional gain [N.m/rad]
Kd = 18.0            # Attitude derivative gain [N.m/(rad/s)]

# --- Simulation ---
t_end   = 300.0       # Duration [s]
dt      = 0.05        # Output step [s]

# --- Initial conditions ---
theta0    = np.deg2rad(1.0)
thetadot0 = 0.0
delta0    = 0.0
deltadot0 = 0.0

# =============================================================================
# CONSTANTS
# =============================================================================

LAM = 1.8412          # First zero of J1'(x) — Bessel function constant

# =============================================================================
# DOME CORRECTION (barrel section only — see Jang et al. Eq. 20)
# =============================================================================

def dome_correction(R0, h, dome_d):
    """
    Map a cylindrical tank with elliptical domes to an equivalent
    flat-bottom cylindrical tank (barrel section case).
    Jang et al. (2014), Eq. 20.
    """
    if dome_d <= 0:
        return R0, h
    # For liquid in barrel section: R0* = R0, h* = h + d/3
    h_star = h + dome_d / 3.0
    R0_star = R0
    return R0_star, h_star

# =============================================================================
# BOND NUMBER AND REGIME DETECTION
# =============================================================================

def bond_number(rho, a, R0, sigma):
    """Compute Bond number."""
    if sigma <= 0 or a <= 0:
        return np.inf
    return rho * a * R0**2 / sigma

def regime(Bo):
    """Classify the hydrodynamic regime."""
    if Bo > 1000:
        return 'high-g (Abramson)'
    elif Bo >= 10:
        return 'low-g (Dodge/Jang)'
    else:
        return 'extreme low-g (CFD required)'

# =============================================================================
# FREE SURFACE STABILITY
# =============================================================================

def critical_acceleration(R0, sigma, rho):
    """
    Minimum adverse (reverse) acceleration to destabilise the free surface.
    Jang et al. (2014), Eq. 39.
    """
    return (LAM**2 - 1.0) * sigma / (rho * R0**2)

# =============================================================================
# HIGH-G ABRAMSON PARAMETERS
# =============================================================================

def abramson_params(R0, h, m_liq, a_eff):
    """
    Classical Abramson (1966) first-mode parameters for a cylindrical tank.
    Valid for Bo > 1000.
    """
    xi = h / R0
    omega1_sq = (a_eff * LAM / R0) * np.tanh(LAM * xi)
    omega1 = np.sqrt(max(omega1_sq, 1e-10))

    m1_frac = (2.0 / (LAM**2 * xi)) * np.tanh(LAM * xi)
    m1_frac = min(m1_frac, 0.95)
    m1 = m1_frac * m_liq
    m0 = m_liq - m1

    h1 = h - (R0 / LAM) * np.tanh(LAM * xi)
    h1 = max(h1, 0.01)

    return m0, m1, omega1, h1

# =============================================================================
# LOW-G DODGE / JANG PARAMETERS
# =============================================================================

def lowg_frequency(R0, h, a_eff, rho, sigma):
    """
    Low-g first-mode slosh frequency including surface tension contribution.
    Jang et al. (2014), Eq. 38 (90-degree contact angle form, engineering approximation).

    ω₁² = (a*λ/R)*tanh(λ*h/R)  +  (σ*λ²*(λ²-1))/(ρ*R³)*tanh(λ*h/R)
           ↑ gravity term              ↑ capillary (surface tension) term
    """
    xi = h / R0
    tanh_term = np.tanh(LAM * xi)

    omega_sq_grav = (a_eff * LAM / R0) * tanh_term
    omega_sq_cap  = (sigma * LAM**2 * (LAM**2 - 1.0) / (rho * R0**3)) * tanh_term

    omega_sq = omega_sq_grav + omega_sq_cap
    return np.sqrt(max(omega_sq, 1e-12)), omega_sq_grav, omega_sq_cap

def lowg_mass_fraction(R0, h, Bo):
    """
    Low-g sloshing mass fraction (first mode).
    Approximation following Jang et al. (2014) — Abramson form with
    Bond number correction factor.
    f(Bo) reduces sloshing mass at low Bond numbers because surface
    tension anchors more liquid near the walls.
    """
    xi = h / R0
    # Base Abramson fraction
    m1_frac_base = (2.0 / (LAM**2 * xi)) * np.tanh(LAM * xi)
    m1_frac_base = min(m1_frac_base, 0.95)

    # Bond number correction: at low Bo, surface tension reduces sloshing mass
    # f(Bo) = Bo / (Bo + (λ²-1))  →  1 as Bo→∞,  → 0 as Bo→0
    f_Bo = Bo / (Bo + (LAM**2 - 1.0)) if Bo > 0 else 0.0

    return m1_frac_base * f_Bo

def lowg_damping(R0, omega1, nu_k, Bo):
    """
    Low-g smooth wall damping ratio.
    Jang et al. (2014), Eq. 35, based on Galileo number.
    Valid for Bo > 10.
    """
    if omega1 <= 0:
        return 0.005

    # Galileo number
    N_GA = (R0 * omega1)**0.4647 / nu_k**2

    if N_GA <= 0:
        return 0.005

    if Bo <= 10:
        zeta = 0.83 / np.sqrt(N_GA)
    else:
        zeta = 0.83 / np.sqrt(N_GA) + 0.096 * Bo**(-3.0/5.0) / np.sqrt(N_GA)

    # Physical bounds
    zeta = np.clip(zeta, 0.0005, 0.20)
    return zeta

def lowg_attachment_height(R0, h):
    """
    Attachment height above tank bottom — same structural form as Abramson.
    Surface tension correction to h1 is second-order; Abramson formula retained.
    """
    xi = h / R0
    h1 = h - (R0 / LAM) * np.tanh(LAM * xi)
    return max(h1, 0.01)

# =============================================================================
# UNIFIED PARAMETER EXTRACTION
# =============================================================================

def slosh_params(R0, h_fill, dome_d, m_liq, a_eff, rho, sigma, nu_k):
    """
    Compute slosh parameters using Bond-number-aware model switching.
    Returns dict with all parameters and regime label.
    """
    # Dome correction
    R0_eff, h_eff = dome_correction(R0, h_fill, dome_d)

    # Bond number
    Bo = bond_number(rho, a_eff, R0_eff, sigma)
    reg = regime(Bo)

    # Critical acceleration
    a_crit = critical_acceleration(R0_eff, sigma, rho)

    if Bo > 1000:
        # High-g: classical Abramson
        m0, m1, omega1, h1 = abramson_params(R0_eff, h_eff, m_liq, a_eff)
        zeta1 = 0.005  # smooth wall default for high-g
        omega_grav = omega1
        omega_cap = 0.0
        note = "Abramson high-g"

    elif Bo >= 10:
        # Low-g: Dodge / Jang
        omega1, omega_grav_sq, omega_cap_sq = lowg_frequency(R0_eff, h_eff, a_eff, rho, sigma)
        omega_grav = np.sqrt(max(omega_grav_sq, 0))
        omega_cap  = np.sqrt(max(omega_cap_sq, 0))

        m1_frac = lowg_mass_fraction(R0_eff, h_eff, Bo)
        m1 = m1_frac * m_liq
        m0 = m_liq - m1

        h1 = lowg_attachment_height(R0_eff, h_eff)
        zeta1 = lowg_damping(R0_eff, omega1, nu_k, Bo)
        note = "Dodge/Jang low-g"

    else:
        # Extreme low-g: outside analytical validity
        # Return surface-tension-only estimate as order-of-magnitude
        omega1, omega_grav_sq, omega_cap_sq = lowg_frequency(R0_eff, h_eff, 0.0, rho, sigma)
        omega_grav = 0.0
        omega_cap  = omega1

        m1_frac = lowg_mass_fraction(R0_eff, h_eff, Bo)
        m1 = m1_frac * m_liq
        m0 = m_liq - m1
        h1 = lowg_attachment_height(R0_eff, h_eff)
        zeta1 = lowg_damping(R0_eff, omega1, nu_k, max(Bo, 0.1))
        note = "EXTRAPOLATION ONLY — CFD required (Bo < 10)"

    return {
        'Bo': Bo, 'regime': reg, 'note': note,
        'm_liq': m_liq, 'm0': m0, 'm1': m1,
        'm1_frac': m1 / m_liq,
        'omega1': omega1, 'f1_Hz': omega1 / (2*np.pi),
        'omega_grav': omega_grav, 'omega_cap': omega_cap,
        'h1': h1, 'zeta1': zeta1,
        'a_crit': a_crit, 'R0_eff': R0_eff, 'h_eff': h_eff,
    }

# =============================================================================
# PRINT PARAMETERS
# =============================================================================

m_liq = rho * np.pi * R0**2 * h_fill

p = slosh_params(R0, h_fill, dome_d, m_liq, a_eff, rho, sigma, nu_k)

print("=" * 60)
print("  LOW-g SLOSH PARAMETERS")
print("=" * 60)
print(f"  Tank radius R₀         = {R0:.3f} m")
print(f"  Dome depth             = {dome_d:.3f} m")
print(f"  Fill height h          = {h_fill:.3f} m  (h/R = {h_fill/R0:.2f})")
print(f"  Effective accel a_eff  = {a_eff:.2e} m/s²")
print(f"  Liquid density ρ       = {rho:.0f} kg/m³")
print(f"  Surface tension σ      = {sigma:.4f} N/m")
print()
print(f"  Bond number Bo         = {p['Bo']:.2f}")
print(f"  Regime                 : {p['regime']}")
print(f"  Note                   : {p['note']}")
print()
print(f"  Total propellant mass  = {m_liq:.2f} kg")
print(f"  Sloshing mass m₁       = {p['m1']:.2f} kg  ({100*p['m1_frac']:.1f}%)")
print(f"  Fixed mass m₀          = {p['m0']:.2f} kg")
print(f"  Slosh frequency ω₁     = {p['omega1']:.4f} rad/s  ({p['f1_Hz']:.4f} Hz)")
print(f"  Gravity contribution   = {p['omega_grav']:.4f} rad/s")
print(f"  Capillary contribution = {p['omega_cap']:.4f} rad/s")
print(f"  Damping ratio ζ₁       = {p['zeta1']:.5f}")
print(f"  Attachment height h₁   = {p['h1']:.3f} m above tank bottom")
print(f"  Critical adverse accel = {p['a_crit']:.2e} m/s²")
print("=" * 60)

# =============================================================================
# 2D PARAMETER SWEEP: fill fraction × Bond number
# =============================================================================

fill_fracs = np.linspace(0.05, 0.95, 40)
bo_vals    = np.logspace(-1, 4, 50)

omega_map  = np.zeros((len(fill_fracs), len(bo_vals)))
m1frac_map = np.zeros_like(omega_map)
zeta_map   = np.zeros_like(omega_map)

for i, ff in enumerate(fill_fracs):
    h_i = ff * 2 * R0   # convert fill fraction to height assuming spherical approx
    h_i = max(h_i, 0.01)
    m_i = rho * np.pi * R0**2 * h_i
    for j, Bo_j in enumerate(bo_vals):
        a_j = Bo_j * sigma / (rho * R0**2)
        pr = slosh_params(R0, h_i, dome_d, m_i, a_j, rho, sigma, nu_k)
        omega_map[i, j]  = pr['omega1']
        m1frac_map[i, j] = pr['m1_frac']
        zeta_map[i, j]   = pr['zeta1']

# =============================================================================
# EQUATIONS OF MOTION AND SIMULATION
# =============================================================================

m1    = p['m1']
omega1 = p['omega1']
z1    = p['zeta1']

z_tank_bottom = z_tank_centre - h_fill / 2.0
z_attach = z_tank_bottom + p['h1']
r_arm = z_attach

print(f"\n  Tank bottom z          = {z_tank_bottom:.3f} m from CoM")
print(f"  Slosh attach point z   = {z_attach:.3f} m from CoM")
print(f"  Moment arm r_arm       = {r_arm:.3f} m")
print("=" * 60)

def ctrl(theta, thetadot):
    return -Kp * theta - Kd * thetadot

def eom(t, state):
    theta, thetadot, delta, deltadot = state
    T = ctrl(theta, thetadot)

    slosh_forcing = m1 * r_arm * (2*z1*omega1*deltadot + omega1**2*delta)
    theta_ddot = (T + slosh_forcing) / I_sc
    delta_ddot = (-r_arm * theta_ddot
                  - 2*z1*omega1*deltadot
                  - omega1**2*delta)
    return [thetadot, theta_ddot, deltadot, delta_ddot]

x0 = [theta0, thetadot0, delta0, deltadot0]
sol = solve_ivp(eom, (0, t_end), x0,
                t_eval=np.arange(0, t_end, dt),
                method='RK45', rtol=1e-9, atol=1e-12)

t         = sol.t
theta_deg = np.rad2deg(sol.y[0])
thetadot  = np.rad2deg(sol.y[1])
delta_mm  = sol.y[2] * 1e3
T_hist    = np.array([ctrl(sol.y[0,i], sol.y[1,i]) for i in range(len(t))])

# =============================================================================
# OPEN LOOP BODE (state space)
# =============================================================================

from scipy.signal import StateSpace

A = np.array([
    [0, 1, 0, 0],
    [0, 0, m1*r_arm*omega1**2/I_sc, m1*r_arm*2*z1*omega1/I_sc],
    [0, 0, 0, 1],
    [0, 0, -(omega1**2 + m1*r_arm**2*omega1**2/I_sc),
            -(2*z1*omega1 + m1*r_arm**2*2*z1*omega1/I_sc)]
])
B = np.array([[0], [1/I_sc], [0], [-r_arm/I_sc]])
C = np.array([[1, 0, 0, 0]])
D = np.array([[0]])

sys_ss = StateSpace(A, B, C, D)
w_bode = np.logspace(-4, 1, 3000)
_, H = sys_ss.freqresp(w_bode)
H_flat = H.flatten()
mag_db = 20*np.log10(np.abs(H_flat))
phase_deg_bode = np.rad2deg(np.unwrap(np.angle(H_flat)))

# =============================================================================
# PLOTS
# =============================================================================

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    f"Low-g Sloshing Simulation — {p['note']}\n"
    f"Bo={p['Bo']:.1f}, ω₁={p['omega1']:.4f} rad/s ({p['f1_Hz']:.4f} Hz), "
    f"ζ₁={p['zeta1']:.4f}, m₁={p['m1']:.1f} kg ({100*p['m1_frac']:.0f}%), "
    f"a_eff={a_eff:.2e} m/s²",
    fontsize=11
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

# --- Attitude ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, theta_deg, color='steelblue', lw=1.3)
ax1.axhline(0, color='k', lw=0.5, ls='--')
ax1.set_xlabel('Time [s]'); ax1.set_ylabel('Attitude [deg]')
ax1.set_title('Spacecraft Attitude'); ax1.grid(True, alpha=0.3)

# --- Slosh displacement ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t, delta_mm, color='firebrick', lw=1.3)
ax2.axhline(0, color='k', lw=0.5, ls='--')
ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Slosh displacement [mm]')
ax2.set_title('Sloshing Mass Displacement'); ax2.grid(True, alpha=0.3)

# --- Control torque ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(t, T_hist, color='darkgreen', lw=1.3)
ax3.axhline(0, color='k', lw=0.5, ls='--')
ax3.set_xlabel('Time [s]'); ax3.set_ylabel('Torque [N.m]')
ax3.set_title('Control Torque'); ax3.grid(True, alpha=0.3)

# --- Bode magnitude ---
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.semilogx(w_bode, mag_db, color='steelblue', lw=1.5)
ax4.axvline(p['omega1'], color='firebrick', lw=1.2, ls='--',
            label=f"ω₁={p['omega1']:.4f} rad/s")
ax4.set_xlabel('ω [rad/s]'); ax4.set_ylabel('Magnitude [dB]')
ax4.set_title('Open-Loop Plant Bode — Magnitude')
ax4.legend(); ax4.grid(True, which='both', alpha=0.3)

# --- Bode phase ---
ax5 = fig.add_subplot(gs[1, 2])
ax5.semilogx(w_bode, phase_deg_bode, color='darkorange', lw=1.5)
ax5.axvline(p['omega1'], color='firebrick', lw=1.2, ls='--')
ax5.set_xlabel('ω [rad/s]'); ax5.set_ylabel('Phase [deg]')
ax5.set_title('Bode — Phase'); ax5.grid(True, which='both', alpha=0.3)

# --- 2D map: slosh frequency ---
ax6 = fig.add_subplot(gs[2, 0])
im6 = ax6.contourf(bo_vals, fill_fracs, omega_map / (2*np.pi),
                    levels=20, cmap='viridis')
ax6.set_xscale('log')
ax6.axvline(10, color='white', lw=1, ls='--', label='Bo=10')
ax6.axvline(1000, color='yellow', lw=1, ls='--', label='Bo=1000')
ax6.set_xlabel('Bond number Bo'); ax6.set_ylabel('Fill fraction h/2R')
ax6.set_title('Slosh Frequency f₁ [Hz]')
plt.colorbar(im6, ax=ax6)
ax6.legend(fontsize=7)
ax6.scatter([p['Bo']], [h_fill/(2*R0)], color='red', s=60, zorder=5, label='Current')

# --- 2D map: sloshing mass fraction ---
ax7 = fig.add_subplot(gs[2, 1])
im7 = ax7.contourf(bo_vals, fill_fracs, m1frac_map * 100,
                    levels=20, cmap='plasma')
ax7.set_xscale('log')
ax7.axvline(10, color='white', lw=1, ls='--')
ax7.axvline(1000, color='yellow', lw=1, ls='--')
ax7.set_xlabel('Bond number Bo'); ax7.set_ylabel('Fill fraction h/2R')
ax7.set_title('Sloshing Mass Fraction m₁/m_liq [%]')
plt.colorbar(im7, ax=ax7)
ax7.scatter([p['Bo']], [h_fill/(2*R0)], color='red', s=60, zorder=5)

# --- 2D map: damping ---
ax8 = fig.add_subplot(gs[2, 2])
im8 = ax8.contourf(bo_vals, fill_fracs, zeta_map * 100,
                    levels=20, cmap='cool')
ax8.set_xscale('log')
ax8.axvline(10, color='white', lw=1, ls='--')
ax8.axvline(1000, color='yellow', lw=1, ls='--')
ax8.set_xlabel('Bond number Bo'); ax8.set_ylabel('Fill fraction h/2R')
ax8.set_title('Damping Ratio ζ₁ [%]')
plt.colorbar(im8, ax=ax8)
ax8.scatter([p['Bo']], [h_fill/(2*R0)], color='red', s=60, zorder=5)

plt.savefig('microgravity_sloshing_simulation.png',
            dpi=150, bbox_inches='tight')
print("\nSimulation plot saved.")

# =============================================================================
# COMPARISON PLOT: High-g vs Low-g vs Surface-tension-only
# =============================================================================

fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('Model Comparison: Abramson High-g vs Dodge Low-g vs Surface Tension Only\n'
              f'R₀={R0}m, ρ={rho} kg/m³, σ={sigma} N/m', fontsize=11)

fill_range = np.linspace(0.02, 0.98, 100)
h_range    = fill_range * 2 * R0

omega_highg    = []
omega_lowg     = []
omega_cap_only = []
m1_highg       = []
m1_lowg        = []

a_highg = 2.0    # orbit raising
a_lowg  = 0.05   # station-keeping
a_coast = 0.0    # coast (surface tension only)

for h_i in h_range:
    m_i = rho * np.pi * R0**2 * h_i

    # High-g Abramson
    _, _, om_hg, _ = abramson_params(R0, h_i, m_i, a_highg)
    omega_highg.append(om_hg / (2*np.pi))
    m1_highg.append((2.0 / (LAM**2 * (h_i/R0))) * np.tanh(LAM * h_i/R0))

    # Low-g Dodge
    om_lg, _, _ = lowg_frequency(R0, h_i, a_lowg, rho, sigma)
    omega_lowg.append(om_lg / (2*np.pi))
    Bo_lg = bond_number(rho, a_lowg, R0, sigma)
    m1_lowg.append(lowg_mass_fraction(R0, h_i, Bo_lg))

    # Surface tension only
    om_cap, _, _ = lowg_frequency(R0, h_i, a_coast, rho, sigma)
    omega_cap_only.append(om_cap / (2*np.pi))

axes[0].plot(fill_range, omega_highg,    color='steelblue', lw=2, label=f'High-g (a={a_highg} m/s²)')
axes[0].plot(fill_range, omega_lowg,     color='darkorange', lw=2, label=f'Low-g (a={a_lowg} m/s²)')
axes[0].plot(fill_range, omega_cap_only, color='firebrick', lw=2, ls='--', label='Surface tension only')
axes[0].set_xlabel('Fill fraction h/2R'); axes[0].set_ylabel('Frequency [Hz]')
axes[0].set_title('Slosh Natural Frequency f₁')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].plot(fill_range, np.array(m1_highg)*100, color='steelblue', lw=2, label=f'High-g (a={a_highg})')
axes[1].plot(fill_range, np.array(m1_lowg)*100,  color='darkorange', lw=2, label=f'Low-g (a={a_lowg})')
axes[1].set_xlabel('Fill fraction h/2R'); axes[1].set_ylabel('m₁/m_liq [%]')
axes[1].set_title('Sloshing Mass Fraction')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

# Critical acceleration vs tank radius
R_range = np.linspace(0.1, 1.0, 100)
a_crit_range = (LAM**2 - 1) * sigma / (rho * R_range**2)
axes[2].plot(R_range, a_crit_range * 1e3, color='purple', lw=2)
axes[2].axhline(1.0, color='grey', lw=1, ls='--', label='1 mm/s²')
axes[2].axvline(R0, color='red', lw=1, ls='--', label=f'Current R₀={R0}m')
axes[2].set_xlabel('Tank radius R₀ [m]')
axes[2].set_ylabel('Critical adverse acceleration [mm/s²]')
axes[2].set_title('Free Surface Stability Limit\n(min adverse accel to destabilise)')
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('microgravity_model_comparison.png',
            dpi=150, bbox_inches='tight')
print("Comparison plot saved.")
print("\nAll done.")
