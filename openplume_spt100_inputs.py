"""
OpenPlume Input Generator — SPT-100 Empirical Plume Model
==========================================================
Generates plume characterisation inputs for OpenPlume (Custom / Curve Fit method)
based on the SPT-100 empirical model from Kim (1999) PEPL/Michigan thesis and
AIAA-1998-3641, scaled to BHT-350 operating conditions.

References:
  - Kim, S.W. (1999). PhD Thesis, University of Michigan (PEPL)
  - Parks & Katz (1979). JPL Technical Memorandum 33-777
  - Randolph et al. (1993). IEPC-93-093
  - Manzella et al. (1997). AIAA-1997-3054

Outputs:
  - ion_current_density.csv     : j(theta) angular profile [A/m^2] at 1m
  - ion_energy_distribution.csv : f(E, theta) IEDF at key angles
  - species_fractions.csv       : ion species fractions vs angle
  - summary_inputs.txt          : human-readable summary for OpenPlume GUI entry
  - plots (PNG)                 : visualisations of all distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# =============================================================================
# THRUSTER PARAMETERS
# =============================================================================

# --- SPT-100 reference parameters (Kim 1999, 300V / 4.5A) ---
SPT100 = {
    "name": "SPT-100 (Reference)",
    "V_discharge": 300.0,       # V
    "I_discharge": 4.5,         # A
    "beam_efficiency": 0.90,    # fraction of discharge current that is beam current
    "mean_ion_energy_frac": 0.88,  # E_mean = fraction * V_discharge (eV)
    "energy_spread_sigma": 38.0,   # eV, on-axis 1-sigma from Kim 1999
    "beam_half_angle_deg": 40.0,   # degrees, 95% enclosed current half-angle
    "cosine_power_n": 3.5,         # exponent in cos^n fit to main beam (fitted from Fig 4-6 Kim 1999)
    "cex_fraction": 0.15,          # fraction of total current in CEX wing
    "alpha2": 0.08,                # doubly charged ion fraction (Xe2+)
    "alpha3": 0.02,                # triply charged ion fraction (Xe3+)
    "ref_distance_m": 1.0,         # reference distance for j(theta) normalisation
}

# --- BHT-350 parameters (scaled from SPT-100) ---
BHT350 = {
    "name": "BHT-350 (Scaled from SPT-100)",
    "V_discharge": 300.0,       # V — identical to SPT-100 => same energy distributions
    "I_discharge": 1.0,         # A
    "beam_efficiency": 0.90,
    "mean_ion_energy_frac": 0.88,
    "energy_spread_sigma": 38.0,   # same as SPT-100 (same V_discharge)
    "beam_half_angle_deg": 40.0,   # same angular shape (same voltage class)
    "cosine_power_n": 3.5,         # same fitting exponent
    "cex_fraction": 0.15,
    "alpha2": 0.08,
    "alpha3": 0.02,
    "ref_distance_m": 0.77,        # BHT-350 Busek data at 77 cm
}

THRUSTERS = [SPT100, BHT350]

# Angular grid
THETA_DEG = np.linspace(0, 120, 500)   # 0 to 120 degrees (beyond CEX region)
THETA_RAD = np.radians(THETA_DEG)

# Ion energy grid
ENERGY_EV = np.linspace(0, 500, 1000)

# Key angles for IEDF output (matching RPA measurement angles)
IEDF_ANGLES_DEG = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def beam_current(thruster):
    """Total beam current [A]"""
    return thruster["I_discharge"] * thruster["beam_efficiency"]


def mean_ion_energy(thruster):
    """Mean ion energy [eV]"""
    return thruster["mean_ion_energy_frac"] * thruster["V_discharge"]


def ion_current_density(theta_rad, thruster, distance_m=None):
    """
    Ion current density j(theta) [A/m^2] at given distance.

    Two-component model:
      j(theta) = j_beam * cos^n(theta) + j_cex * cos^n(theta/2)

    The cos^n(theta/2) term captures the broad CEX backflow region.
    Normalised so that 2*pi * integral(j * sin(theta), 0, pi/2) = I_beam / (2*pi*r^2)
    """
    if distance_m is None:
        distance_m = thruster["ref_distance_m"]

    I_b = beam_current(thruster)
    n = thruster["cosine_power_n"]
    f_cex = thruster["cex_fraction"]

    # Normalisation: integrate cos^n over hemisphere = 2*pi / (n+1) for even-ish n
    # Use numerical normalisation for accuracy
    theta_norm = np.linspace(0, np.pi / 2, 10000)
    beam_shape = np.cos(theta_norm) ** n
    cex_shape = np.cos(theta_norm / 2) ** n
    combined = (1 - f_cex) * beam_shape + f_cex * cex_shape
    norm = 2 * np.pi * np.trapezoid(combined * np.sin(theta_norm), theta_norm)

    # Peak on-axis current density [A/m^2] at reference distance
    j0 = I_b / (norm * distance_m ** 2)

    # Angular profile
    beam_component = (1 - f_cex) * np.cos(np.clip(theta_rad, 0, np.pi / 2)) ** n
    cex_component = f_cex * np.cos(np.clip(theta_rad / 2, 0, np.pi / 2)) ** n

    j = j0 * (beam_component + cex_component)
    j[theta_rad > np.pi / 2] = j0 * f_cex * np.cos(
        np.clip(theta_rad[theta_rad > np.pi / 2] / 2, 0, np.pi / 2)
    ) ** n

    return j


def ion_energy_distribution(energy_ev, theta_deg, thruster):
    """
    Ion Energy Distribution Function f(E, theta) [1/eV].

    Modelled as a Gaussian centred on the mean beam energy, with:
      - sigma broadening linearly with angle (beam spreading away from axis)
      - mean energy shifting slightly with angle (Randolph 1993)

    Normalised so integral over E = 1.
    """
    E_mean = mean_ion_energy(thruster)
    sigma_0 = thruster["energy_spread_sigma"]

    # Energy broadens and shifts with angle (empirical from Kim 1999 RPA data)
    theta_rad = np.radians(theta_deg)
    sigma = sigma_0 * (1 + 1.5 * np.sin(theta_rad) ** 2)
    E_peak = E_mean * (1 - 0.12 * np.sin(theta_rad) ** 2)  # slight redshift off-axis

    f = np.exp(-0.5 * ((energy_ev - E_peak) / sigma) ** 2)
    f /= np.trapezoid(f, energy_ev)  # normalise
    return f


def species_fractions(theta_deg, thruster):
    """
    Ion species number fractions vs angle.
    Based on Kim 1999 ExB probe measurements.

    Xe1+ fraction drops slightly off-axis; Xe2+/Xe3+ increase slightly.
    """
    alpha2_0 = thruster["alpha2"]
    alpha3_0 = thruster["alpha3"]

    theta_rad = np.radians(theta_deg)

    # Slight increase in multiply-charged fraction off-axis (Kim 1999 Fig 6-x)
    alpha2 = alpha2_0 * (1 + 0.3 * np.sin(theta_rad) ** 2)
    alpha3 = alpha3_0 * (1 + 0.5 * np.sin(theta_rad) ** 2)

    # Cap fractions
    alpha2 = np.clip(alpha2, 0, 0.20)
    alpha3 = np.clip(alpha3, 0, 0.08)
    alpha1 = 1.0 - alpha2 - alpha3

    return alpha1, alpha2, alpha3


# =============================================================================
# GENERATE OUTPUTS
# =============================================================================

os.makedirs("openplume_outputs", exist_ok=True)

# ── 1. Ion Current Density ────────────────────────────────────────────────────
print("Generating ion current density profiles...")
jcd_data = {"theta_deg": THETA_DEG}
for t in THRUSTERS:
    j = ion_current_density(THETA_RAD, t)
    jcd_data[f"j_{t['name'].split()[0]}_{t['ref_distance_m']}m_A_per_m2"] = j

df_jcd = pd.DataFrame(jcd_data)
df_jcd.to_csv("openplume_outputs/ion_current_density.csv", index=False, float_format="%.6e")
print("  -> ion_current_density.csv saved")

# ── 2. Ion Energy Distribution ────────────────────────────────────────────────
print("Generating ion energy distributions...")
iedf_rows = []
for t in THRUSTERS:
    for angle in IEDF_ANGLES_DEG:
        f = ion_energy_distribution(ENERGY_EV, angle, t)
        for e, fval in zip(ENERGY_EV, f):
            iedf_rows.append({
                "thruster": t["name"].split()[0],
                "theta_deg": angle,
                "energy_eV": round(e, 2),
                "f_E_per_eV": fval
            })

df_iedf = pd.DataFrame(iedf_rows)
df_iedf.to_csv("openplume_outputs/ion_energy_distribution.csv", index=False, float_format="%.6e")
print("  -> ion_energy_distribution.csv saved")

# ── 3. Species Fractions ──────────────────────────────────────────────────────
print("Generating species fractions...")
spec_data = {"theta_deg": THETA_DEG}
for t in THRUSTERS:
    a1, a2, a3 = species_fractions(THETA_DEG, t)
    tag = t["name"].split()[0]
    spec_data[f"Xe1+_{tag}"] = a1
    spec_data[f"Xe2+_{tag}"] = a2
    spec_data[f"Xe3+_{tag}"] = a3

df_spec = pd.DataFrame(spec_data)
df_spec.to_csv("openplume_outputs/species_fractions.csv", index=False, float_format="%.6f")
print("  -> species_fractions.csv saved")

# ── 4. Summary Text (for OpenPlume GUI manual entry) ─────────────────────────
print("Generating summary inputs file...")
with open("openplume_outputs/summary_inputs.txt", "w") as f:
    f.write("=" * 65 + "\n")
    f.write("  OpenPlume Custom Method — Input Parameter Summary\n")
    f.write("  SPT-100 Empirical Model (Kim 1999) scaled to BHT-350\n")
    f.write("=" * 65 + "\n\n")

    for t in THRUSTERS:
        I_b = beam_current(t)
        E_mean = mean_ion_energy(t)
        f.write(f"THRUSTER: {t['name']}\n")
        f.write("-" * 45 + "\n")
        f.write(f"  Ion type                     : Xe (Xenon)\n")
        f.write(f"  Discharge voltage            : {t['V_discharge']:.1f} V\n")
        f.write(f"  Discharge current            : {t['I_discharge']:.2f} A\n")
        f.write(f"  Beam current (I_b)           : {I_b:.3f} A\n")
        f.write(f"  Mean ion energy              : {E_mean:.1f} eV\n")
        f.write(f"  Ion energy spread (1-sigma)  : {t['energy_spread_sigma']:.1f} eV\n")
        f.write(f"  Beam divergence half-angle   : {t['beam_half_angle_deg']:.1f} deg\n")
        f.write(f"  Cosine-power exponent n      : {t['cosine_power_n']:.2f}\n")
        f.write(f"  CEX current fraction         : {t['cex_fraction']*100:.1f} %\n")
        f.write(f"  Doubly charged fraction a2   : {t['alpha2']*100:.1f} %\n")
        f.write(f"  Triply charged fraction a3   : {t['alpha3']*100:.1f} %\n")
        f.write(f"  Reference distance           : {t['ref_distance_m']:.2f} m\n")
        f.write(f"\n  Angular distribution model:\n")
        f.write(f"    j(theta) = j0 * [ {1-t['cex_fraction']:.2f}*cos^{t['cosine_power_n']}(theta)\n")
        f.write(f"                     + {t['cex_fraction']:.2f}*cos^{t['cosine_power_n']}(theta/2) ]\n")
        f.write(f"\n  On-axis current density j0 at {t['ref_distance_m']:.2f} m:\n")
        j_onaxis = ion_current_density(np.array([0.0]), t)[0]
        f.write(f"    j0 = {j_onaxis:.4f} A/m^2\n\n")

    f.write("=" * 65 + "\n")
    f.write("References:\n")
    f.write("  Kim (1999) PhD Thesis, Univ. Michigan (PEPL)\n")
    f.write("  Parks & Katz (1979) JPL Technical Memo 33-777\n")
    f.write("  Randolph et al. (1993) IEPC-93-093\n")
    f.write("  Manzella et al. (1997) AIAA-1997-3054\n")

print("  -> summary_inputs.txt saved")

# =============================================================================
# PLOTS
# =============================================================================
print("Generating plots...")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#F8F9FA")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

colors = {"SPT-100": "#1B3A6B", "BHT-350": "#E8622A"}
ls_map = {"SPT-100": "-", "BHT-350": "--"}

# ── Plot 1: Ion Current Density (linear) ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("white")
for t in THRUSTERS:
    tag = t["name"].split()[0]
    j = ion_current_density(THETA_RAD, t)
    j_norm = j / j[0]
    ax1.plot(THETA_DEG, j_norm,
             color=colors[tag], lw=2, ls=ls_map[tag],
             label=f"{tag} ({t['ref_distance_m']}m)")
ax1.axvline(40, color="gray", lw=1, ls=":", alpha=0.7, label="40° half-angle")
ax1.set_xlabel("Angle from thrust axis (°)", fontsize=10)
ax1.set_ylabel("Normalised current density j/j₀", fontsize=10)
ax1.set_title("Ion Current Density Angular Profile", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.set_xlim(0, 120)
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3)

# ── Plot 2: Ion Current Density (log) ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("white")
for t in THRUSTERS:
    tag = t["name"].split()[0]
    j = ion_current_density(THETA_RAD, t)
    ax2.semilogy(THETA_DEG, j,
                 color=colors[tag], lw=2, ls=ls_map[tag],
                 label=f"{tag} ({t['ref_distance_m']}m)")
ax2.set_xlabel("Angle from thrust axis (°)", fontsize=10)
ax2.set_ylabel("Ion current density j (A/m²)", fontsize=10)
ax2.set_title("Ion Current Density — Log Scale", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_xlim(0, 120)
ax2.grid(True, alpha=0.3, which="both")

# ── Plot 3: IEDF at multiple angles (SPT-100) ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("white")
angles_to_plot = [0, 20, 40, 60, 80]
cmap = plt.cm.plasma
for i, angle in enumerate(angles_to_plot):
    f = ion_energy_distribution(ENERGY_EV, angle, SPT100)
    c = cmap(i / (len(angles_to_plot) - 1))
    ax3.plot(ENERGY_EV, f, color=c, lw=2, label=f"{angle}°")
ax3.set_xlabel("Ion energy (eV)", fontsize=10)
ax3.set_ylabel("f(E) (eV⁻¹)", fontsize=10)
ax3.set_title("Ion Energy Distribution f(E,θ) — SPT-100", fontsize=11, fontweight="bold")
ax3.legend(title="Angle", fontsize=9)
ax3.set_xlim(0, 450)
ax3.grid(True, alpha=0.3)

# ── Plot 4: Species fractions ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("white")
t = BHT350
a1, a2, a3 = species_fractions(THETA_DEG, t)
ax4.plot(THETA_DEG, a1 * 100, color="#1B3A6B", lw=2, label="Xe¹⁺")
ax4.plot(THETA_DEG, a2 * 100, color="#E8622A", lw=2, label="Xe²⁺")
ax4.plot(THETA_DEG, a3 * 100, color="#2EA460", lw=2, label="Xe³⁺")
ax4.set_xlabel("Angle from thrust axis (°)", fontsize=10)
ax4.set_ylabel("Ion species fraction (%)", fontsize=10)
ax4.set_title("Ion Species Fractions vs Angle — BHT-350", fontsize=11, fontweight="bold")
ax4.legend(fontsize=9)
ax4.set_xlim(0, 120)
ax4.set_ylim(0, 105)
ax4.grid(True, alpha=0.3)

fig.suptitle(
    "OpenPlume Input Data — SPT-100 Empirical Model Scaled to BHT-350\n"
    "Custom Curve Fit Method | 300V / 1A | Kim (1999) PEPL",
    fontsize=13, fontweight="bold", color="#1B3A6B", y=1.01
)

plt.savefig("openplume_outputs/openplume_input_plots.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  -> openplume_input_plots.png saved")

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 55)
print("All OpenPlume input files saved to: openplume_outputs/")
print("  ion_current_density.csv")
print("  ion_energy_distribution.csv")
print("  species_fractions.csv")
print("  summary_inputs.txt")
print("  openplume_input_plots.png")
print("=" * 55)
print("\nNext step: cross-check summary_inputs.txt against the")
print("OpenPlume .oPEP XML schema and populate the GUI fields.")
