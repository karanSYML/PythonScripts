"""
Slosh Simulation for a Cylindrical Tank
========================================
Based on Abramson (NASA SP-106, 1966) equivalent mechanical model.

Models:
- First slosh mode only (spring-mass equivalent)
- Single-axis spacecraft attitude dynamics coupled with lateral slosh
- Spacecraft is torque-controlled (e.g. reaction wheels or thrusters)

Coordinate convention:
- z-axis: tank symmetry axis (vertical in ground test, thrust axis in flight)
- Slosh occurs laterally (x-axis here, single axis for simplicity)
- Attitude angle theta: rotation about y-axis (pitch)

Physical setup:
- Cylindrical tank of radius R, propellant fill height h
- Spacecraft inertia I_sc about pitch axis
- Tank center located at distance d from spacecraft CoM along z-axis

The slosh mass m1 is attached at height h1 above tank bottom,
which translates to a moment arm from spacecraft CoM.

Usage:
    python slosh_simulation.py

You can modify the parameters in the CONFIGURATION section below.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import bode, TransferFunction
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Tank geometry ---
R = 0.5          # Tank inner radius [m]
h_fill = 0.6     # Propellant fill height [m]  (try values from 0.1R to 1.8R)

# --- Propellant ---
rho = 200.366 #1000.0     # Propellant density [kg/m3] (water-like, e.g. hydrazine ~1004)
m_total = rho * np.pi * R**2 * h_fill   # Total propellant mass [kg]

# --- Spacecraft ---
I_sc = 500.0     # Spacecraft pitch inertia [kg.m2]
# Position of tank centre relative to spacecraft CoM [m] (along z, thrust axis)
z_tank_centre = 1.0  

# --- Damping ---
# Smooth wall: ~0.002, with baffles: ~0.01-0.05
zeta1 = 0.005    # First mode damping ratio [-]

# --- Effective acceleration (needed for pendulum length / frequency) ---
# On-orbit during a burn: e.g. 0.1 m/s2 for a small GEO thruster
# On ground test: 9.81
g_eff = 0.1      # Effective lateral gravity / acceleration [m/s2]

# --- Simulation ---
t_end = 200.0    # Simulation duration [s]
dt = 0.05        # Output time step [s]

# --- Initial conditions ---
theta0    = np.deg2rad(1.0)   # Initial attitude angle [rad]
thetadot0 = 0.0               # Initial angular rate [rad/s]
delta0    = 0.0               # Initial slosh displacement [m]
deltadot0 = 0.0               # Initial slosh velocity [m/s]

# --- Control law ---
# Simple PD attitude controller: T = -Kp*theta - Kd*thetadot
# Set both to 0 for open-loop (free response)
Kp = 50.0    # Proportional gain [N.m/rad]
Kd = 200.0   # Derivative gain [N.m/(rad/s)]

# =============================================================================
# ABRAMSON PARAMETERS FOR A CYLINDRICAL TANK (first mode)
# Reference: Abramson 1966, NASA SP-106, Chapter 2
# =============================================================================

def abramson_cylindrical(R, h, m_total, g_eff):
    """
    Compute first-mode Abramson equivalent mechanical model parameters
    for a cylindrical tank.

    Inputs:
        R       : tank radius [m]
        h       : fill height [m]
        m_total : total propellant mass [kg]
        g_eff   : effective acceleration (thrust or gravity) [m/s2]

    Returns dict with:
        m0  : fixed (non-sloshing) mass [kg]
        m1  : first mode sloshing mass [kg]
        omega1 : first mode natural frequency [rad/s]
        h1  : height of spring attachment above tank bottom [m]
        l1  : equivalent pendulum length [m]
    """
    # First zero of J1' Bessel function: lambda_11 = 1.8412
    lam = 1.8412

    # Fill ratio
    xi = h / R  # dimensionless fill height

    # Natural frequency (Abramson eq. 2.12)
    # omega^2 = (g_eff * lam / R) * tanh(lam * xi)
    omega1_sq = (g_eff * lam / R) * np.tanh(lam * xi)
    omega1 = np.sqrt(omega1_sq)

    # Sloshing mass fraction (Abramson eq. 2.14)
    # m1/m_total = (2 / (lam * xi)) * tanh(lam * xi)  [for first mode]
    # More precise form:
    m1_frac = (2.0 / (lam**2 * xi)) * np.tanh(lam * xi)
    m1_frac = min(m1_frac, 0.95)   # physical cap
    m1 = m1_frac * m_total
    m0 = m_total - m1

    # Height of equivalent spring/pendulum pivot above tank bottom (Abramson eq. 2.16)
    # h1 = h - R * tanh(lam*xi) / lam  ... simplified form
    # More standard: h1 = h - (R/lam) * tanh(lam * xi)
    h1 = h - (R / lam) * np.tanh(lam * xi)
    h1 = max(h1, 0.01)   # keep physical

    # Equivalent pendulum length
    l1 = g_eff / omega1_sq if omega1_sq > 0 else 0.0

    return {
        'm0': m0,
        'm1': m1,
        'omega1': omega1,
        'h1': h1,
        'l1': l1,
        'zeta1': zeta1,
        'm1_frac': m1_frac,
    }


params = abramson_cylindrical(R, h_fill, m_total, g_eff)

print("=" * 55)
print("  ABRAMSON PARAMETERS - Cylindrical Tank, First Mode")
print("=" * 55)
print(f"  Tank radius R          = {R:.3f} m")
print(f"  Fill height h          = {h_fill:.3f} m  (h/R = {h_fill/R:.2f})")
print(f"  Total propellant mass  = {m_total:.2f} kg")
print(f"  Effective acceleration = {g_eff:.3f} m/s2")
print()
print(f"  Fixed mass m0          = {params['m0']:.2f} kg  ({100*(1-params['m1_frac']):.1f}%)")
print(f"  Sloshing mass m1       = {params['m1']:.2f} kg  ({100*params['m1_frac']:.1f}%)")
print(f"  Slosh frequency omega1 = {params['omega1']:.4f} rad/s  ({params['omega1']/(2*np.pi):.4f} Hz)")
print(f"  Slosh period           = {2*np.pi/params['omega1']:.2f} s")
print(f"  Attachment height h1   = {params['h1']:.3f} m above tank bottom")
print(f"  Equivalent pendulum l1 = {params['l1']:.3f} m")
print(f"  Damping ratio zeta1    = {params['zeta1']:.4f}")
print("=" * 55)

# =============================================================================
# GEOMETRY: moment arm from spacecraft CoM to slosh attachment point
# =============================================================================
# Tank bottom is at z_tank_centre - h_fill/2 from spacecraft CoM
# Slosh attachment point h1 above tank bottom
z_tank_bottom = z_tank_centre - h_fill / 2.0
z_slosh_attach = z_tank_bottom + params['h1']
# Moment arm (distance from spacecraft CoM to slosh attachment point)
r_arm = z_slosh_attach   # [m], positive = above CoM

print(f"\n  Tank bottom z          = {z_tank_bottom:.3f} m from CoM")
print(f"  Slosh attach point z   = {z_slosh_attach:.3f} m from CoM")
print(f"  Moment arm r_arm       = {r_arm:.3f} m")
print("=" * 55)

# =============================================================================
# EQUATIONS OF MOTION
# =============================================================================
# State vector: x = [theta, thetadot, delta, deltadot]
#
# theta    : spacecraft pitch angle [rad]
# thetadot : spacecraft pitch rate [rad/s]
# delta    : slosh mass lateral displacement [m]
# deltadot : slosh mass lateral velocity [m/s]
#
# Coupled equations (linearized):
#
# (I_sc + m1*r_arm^2) * theta_ddot + m1*r_arm * delta_ddot = T_ctrl
# delta_ddot + 2*zeta1*omega1*deltadot + omega1^2*delta = -r_arm * theta_ddot
#
# Rearranged to explicit form by solving the 2x2 system simultaneously.

m1    = params['m1']
omega1= params['omega1']
z1    = params['zeta1']

# Effective inertia terms
I_eff = I_sc + m1 * r_arm**2   # effective inertia of spacecraft+slosh

def control_torque(theta, thetadot, t):
    """Simple PD controller."""
    return -Kp * theta - Kd * thetadot

def eom(t, state):
    theta, thetadot, delta, deltadot = state

    T = control_torque(theta, thetadot, t)

    # Solve coupled system:
    # [I_eff,    m1*r_arm] [theta_ddot ]   [T                              ]
    # [r_arm,   1        ] [delta_ddot ]   [-2*z1*omega1*deltadot - omega1^2*delta]
    #
    # From second equation: delta_ddot = -r_arm*theta_ddot - 2*z1*omega1*deltadot - omega1^2*delta
    # Substitute into first: I_eff*theta_ddot + m1*r_arm*(-r_arm*theta_ddot - ...) = T
    # => (I_eff - m1*r_arm^2)*theta_ddot = T + m1*r_arm*(2*z1*omega1*deltadot + omega1^2*delta)
    # => I_sc * theta_ddot = T + m1*r_arm*(2*z1*omega1*deltadot + omega1^2*delta)

    slosh_forcing = m1 * r_arm * (2*z1*omega1*deltadot + omega1**2*delta)
    theta_ddot = (T + slosh_forcing) / I_sc

    delta_ddot = (-r_arm * theta_ddot
                  - 2*z1*omega1*deltadot
                  - omega1**2*delta)

    return [thetadot, theta_ddot, deltadot, delta_ddot]

# =============================================================================
# INTEGRATE
# =============================================================================
t_span = (0, t_end)
t_eval = np.arange(0, t_end, dt)
x0 = [theta0, thetadot0, delta0, deltadot0]

print("\nRunning simulation...")
sol = solve_ivp(eom, t_span, x0, t_eval=t_eval, method='RK45',
                rtol=1e-8, atol=1e-10)
print("Done.")

t    = sol.t
theta     = np.rad2deg(sol.y[0])   # attitude [deg]
thetadot  = np.rad2deg(sol.y[1])   # rate [deg/s]
delta     = sol.y[2] * 1e3         # slosh displacement [mm]
deltadot  = sol.y[3] * 1e3         # slosh velocity [mm/s]

# Control torque history
T_hist = np.array([control_torque(sol.y[0,i], sol.y[1,i], t[i]) for i in range(len(t))])

# =============================================================================
# PLOT
# =============================================================================
fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    f"Slosh Simulation — Cylindrical Tank\n"
    f"R={R}m, h={h_fill}m, m_total={m_total:.1f}kg, "
    f"ω₁={params['omega1']:.3f} rad/s ({params['omega1']/(2*np.pi):.3f} Hz), "
    f"ζ₁={zeta1}, Kp={Kp}, Kd={Kd}",
    fontsize=11
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- Attitude angle ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, theta, color='steelblue', linewidth=1.2)
ax1.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Attitude angle [deg]')
ax1.set_title('Spacecraft Attitude (Pitch)')
ax1.grid(True, alpha=0.3)

# --- Angular rate ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t, thetadot, color='darkorange', linewidth=1.2)
ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Angular rate [deg/s]')
ax2.set_title('Spacecraft Angular Rate')
ax2.grid(True, alpha=0.3)

# --- Slosh displacement ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t, delta, color='firebrick', linewidth=1.2)
ax3.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Slosh displacement [mm]')
ax3.set_title('Sloshing Mass Lateral Displacement')
ax3.grid(True, alpha=0.3)

# --- Slosh velocity ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t, deltadot, color='purple', linewidth=1.2)
ax4.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Slosh velocity [mm/s]')
ax4.set_title('Sloshing Mass Lateral Velocity')
ax4.grid(True, alpha=0.3)

# --- Control torque ---
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(t, T_hist, color='darkgreen', linewidth=1.2)
ax5.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax5.set_xlabel('Time [s]')
ax5.set_ylabel('Control torque [N.m]')
ax5.set_title('Control Torque (PD Law)')
ax5.grid(True, alpha=0.3)

# --- Phase portrait: attitude vs slosh ---
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(theta, delta, color='teal', linewidth=0.8, alpha=0.8)
ax6.scatter(theta[0], delta[0], color='green', s=50, zorder=5, label='Start')
ax6.scatter(theta[-1], delta[-1], color='red', s=50, zorder=5, label='End')
ax6.set_xlabel('Attitude angle [deg]')
ax6.set_ylabel('Slosh displacement [mm]')
ax6.set_title('Phase Portrait: Attitude vs Slosh')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.savefig('slosh_simulation.png', dpi=150, bbox_inches='tight')
print("Plot saved.")

# =============================================================================
# BODE PLOT OF OPEN-LOOP PLANT (attitude / control torque)
# =============================================================================
# Transfer function from T_ctrl to theta (open loop, no feedback)
# Derived analytically from the coupled EOM:
# I_sc * theta_ddot = T + m1*r_arm*(2*z1*omega1*deltadot + omega1^2*delta)
# delta_ddot = -r_arm*theta_ddot - 2*z1*omega1*deltadot - omega1^2*delta
#
# In Laplace domain, substituting:
# Numerator and denominator of theta(s)/T(s):
#
# theta/T = (s^2 + 2*z1*omega1*s + omega1^2) /
#           (I_sc * s^2 * (s^2 + 2*z1*omega1*s + omega1^2)
#            + m1*r_arm^2 * omega1^2 * s^2 ... )
# 
# Let's build it numerically via state space -> transfer function

from scipy.signal import StateSpace, bode as sp_bode

# Build A, B, C, D matrices for open-loop plant (no control)
# State: [theta, thetadot, delta, deltadot], input: T, output: theta

a11 = 0; a12 = 1; a13 = 0; a14 = 0
a21 = 0; a22 = 0
a23 = m1 * r_arm * omega1**2 / I_sc
a24 = m1 * r_arm * 2*z1*omega1 / I_sc

a31 = 0; a32 = 0; a33 = 0; a34 = 1
# delta_ddot = -r_arm*theta_ddot - 2*z1*omega1*deltadot - omega1^2*delta
# theta_ddot = (T + m1*r_arm*(...))/I_sc  -> for open loop with T as input:
# a41 contribution from theta: 0 (theta doesn't directly appear)
a41 = 0
a42 = -r_arm / I_sc * 0   # no direct coupling through thetadot
# Actually let's be more careful:
# delta_ddot = -r_arm * theta_ddot + slosh restoring
# theta_ddot (open loop) = T/I_sc + m1*r_arm*(...)/I_sc
# So:
# delta_ddot = -r_arm*(T/I_sc + m1*r_arm*omega1^2*delta/I_sc + m1*r_arm*2*z1*omega1*deltadot/I_sc)
#              - 2*z1*omega1*deltadot - omega1^2*delta

A = np.array([
    [0,   1,   0,   0],
    [0,   0,   m1*r_arm*omega1**2/I_sc,   m1*r_arm*2*z1*omega1/I_sc],
    [0,   0,   0,   1],
    [0,   0,   -(omega1**2 + m1*r_arm**2*omega1**2/I_sc),
               -(2*z1*omega1 + m1*r_arm**2*2*z1*omega1/I_sc)]
])

B = np.array([[0],
              [1.0/I_sc],
              [0],
              [-r_arm/I_sc]])

C = np.array([[1, 0, 0, 0]])   # output: theta
D = np.array([[0]])

sys_ss = StateSpace(A, B, C, D)

w = np.logspace(-3, 1, 2000)   # rad/s
w_out, H = sys_ss.freqresp(w)
H_flat = H.flatten()
mag_db = 20*np.log10(np.abs(H_flat))
phase_deg = np.rad2deg(np.unwrap(np.angle(H_flat)))

fig2, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig2.suptitle(
    f"Open-Loop Bode Plot — Plant: θ/T_ctrl\n"
    f"Slosh mode at ω₁={params['omega1']:.3f} rad/s ({params['omega1']/(2*np.pi):.4f} Hz)",
    fontsize=11
)

ax_mag.semilogx(w, mag_db, color='steelblue', linewidth=1.5)
ax_mag.axvline(params['omega1'], color='firebrick', linewidth=1.2,
               linestyle='--', label=f"ω₁ = {params['omega1']:.3f} rad/s")
ax_mag.set_ylabel('Magnitude [dB]')
ax_mag.legend()
ax_mag.grid(True, which='both', alpha=0.3)
ax_mag.set_title('Magnitude')

ax_phase.semilogx(w, phase_deg, color='darkorange', linewidth=1.5)
ax_phase.axvline(params['omega1'], color='firebrick', linewidth=1.2, linestyle='--')
ax_phase.set_xlabel('Frequency [rad/s]')
ax_phase.set_ylabel('Phase [deg]')
ax_phase.grid(True, which='both', alpha=0.3)
ax_phase.set_title('Phase')

plt.tight_layout()
plt.savefig('slosh_bode.png', dpi=150, bbox_inches='tight')
print("Bode plot saved.")

plt.show()
print("\nAll done.")
