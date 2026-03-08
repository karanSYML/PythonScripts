"""
Diphasic Fluid Properties for Slosh Analysis
Propellants: N2O (nitrous oxide) and C3H6 (propylene)
Temperature range: -5°C to +40°C (saturation conditions)

Properties extracted:
  - Saturation pressure          [Pa]
  - Liquid density               [kg/m³]
  - Vapor density                [kg/m³]
  - Surface tension              [N/m]
  - Liquid dynamic viscosity     [Pa·s]
  - Liquid kinematic viscosity   [m²/s]
  - Bond number (example geometry)

Usage:
  pip install CoolProp matplotlib pandas

CoolProp backend options (set BACKEND below):
  - "CoolProp"  : uses CoolProp's built-in HEOS equations of state
  - "REFPROP"   : uses your local REFPROP installation via CoolProp wrapper
                  Requires REFPROP installed and env var:
                  export RPPREFIX=/path/to/REFPROP   (typically /home/user/REFPROP)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from CoolProp.CoolProp import PropsSI

# ── Configuration ─────────────────────────────────────────────────────────────

BACKEND = "CoolProp"   # Switch to "REFPROP" to use your REFPROP installation
                        # CoolProp name strings change slightly for REFPROP backend

# Fluid name strings
# CoolProp backend  : "NitrousOxide", "Propylene"
# REFPROP backend   : "REFPROP::N2O",  "REFPROP::Propylene"
if BACKEND == "REFPROP":
    FLUIDS = {
        "N2O":  "REFPROP::N2O",
        "C3H6": "REFPROP::Propylene",
    }
else:
    FLUIDS = {
        "N2O":  "NitrousOxide",
        "C3H6": "Propylene",
    }

T_MIN_C = -5.0    # °C
T_MAX_C = 40.0    # °C
N_POINTS = 91     # resolution (~0.5 °C steps)

# Example tank geometry for Bond number calculation
# Bo = ρ_l * g * R² / σ  — adjust to your actual tank radius
TANK_RADIUS_M = 0.15   # [m]  representative value — update to your tank
G_LEVELS = {
    "1 g (ground)":  9.81,
    "0.01 g (low-g)": 0.0981,
    "0.001 g (micro-g)": 0.00981,
}

# ── Helper ────────────────────────────────────────────────────────────────────

def get_saturation_properties(fluid_name: str, T_K: float) -> dict:
    """
    Query CoolProp (or REFPROP via CoolProp) for saturation properties
    at a given temperature.

    Args:
        fluid_name : CoolProp/REFPROP fluid string
        T_K        : Temperature [K]

    Returns:
        dict of physical properties
    """
    Q_LIQ = 0  # quality = 0 → saturated liquid
    Q_VAP = 1  # quality = 1 → saturated vapor

    P_sat      = PropsSI("P",              "T", T_K, "Q", Q_LIQ, fluid_name)  # [Pa]
    rho_liq    = PropsSI("D",              "T", T_K, "Q", Q_LIQ, fluid_name)  # [kg/m³]
    rho_vap    = PropsSI("D",              "T", T_K, "Q", Q_VAP, fluid_name)  # [kg/m³]
    mu_liq     = PropsSI("V",              "T", T_K, "Q", Q_LIQ, fluid_name)  # [Pa·s]
    sigma      = PropsSI("surface_tension","T", T_K, "Q", Q_LIQ, fluid_name)  # [N/m]

    nu_liq = mu_liq / rho_liq  # kinematic viscosity [m²/s]

    return {
        "P_sat_Pa":     P_sat,
        "P_sat_bar":    P_sat / 1e5,
        "rho_liq":      rho_liq,
        "rho_vap":      rho_vap,
        "mu_liq_Pas":   mu_liq,
        "nu_liq_m2s":   nu_liq,
        "sigma_Nm":     sigma,
    }


def bond_number(rho_liq, sigma, g, R):
    """
    Bond number: Bo = ρ_l * g * R² / σ
    Bo >> 1  → gravity dominates  (classical slosh regime)
    Bo << 1  → surface tension dominates (low-g regime, Dodge models critical)
    """
    return rho_liq * g * R**2 / sigma


# ── Main computation ──────────────────────────────────────────────────────────

T_C_arr = np.linspace(T_MIN_C, T_MAX_C, N_POINTS)
T_K_arr = T_C_arr + 273.15

results = {}

for fluid_key, fluid_name in FLUIDS.items():
    rows = []
    print(f"\nComputing properties for {fluid_key} using backend: {BACKEND}")

    for T_C, T_K in zip(T_C_arr, T_K_arr):
        try:
            props = get_saturation_properties(fluid_name, T_K)
            # Append Bond numbers for each g-level
            bo_dict = {
                f"Bo_{label}": bond_number(props["rho_liq"], props["sigma_Nm"], g, TANK_RADIUS_M)
                for label, g in G_LEVELS.items()
            }
            rows.append({"T_C": T_C, **props, **bo_dict})
        except Exception as e:
            print(f"  Warning: failed at T={T_C:.1f}°C — {e}")

    results[fluid_key] = pd.DataFrame(rows)
    print(f"  Done. {len(rows)} points computed.")

# ── Print summary tables ──────────────────────────────────────────────────────

for fluid_key, df in results.items():
    print(f"\n{'='*70}")
    print(f"  {fluid_key} — Saturation Properties ({T_MIN_C}°C to {T_MAX_C}°C)")
    print(f"{'='*70}")
    display_cols = ["T_C", "P_sat_bar", "rho_liq", "rho_vap",
                    "mu_liq_Pas", "nu_liq_m2s", "sigma_Nm"]
    print(df[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.4g}"
    ))

# ── Export to CSV ─────────────────────────────────────────────────────────────

for fluid_key, df in results.items():
    fname = f"slosh_properties_{fluid_key}.csv"
    df.to_csv(fname, index=False)
    print(f"\nExported: {fname}")

# ── Plots ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Diphasic Properties for Slosh Analysis\nN₂O and C₃H₆ at Saturation",
             fontsize=14, fontweight="bold")

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {"N2O": "#1f77b4", "C3H6": "#d62728"}
LINESTYLES = {"N2O": "-", "C3H6": "--"}

def twin_plot(ax, fluid_key, df, y_col, ylabel, title, scale=1.0):
    ax.plot(df["T_C"], df[y_col] * scale,
            color=COLORS[fluid_key], ls=LINESTYLES[fluid_key],
            lw=2, label=fluid_key)
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axvspan(T_MIN_C, T_MAX_C, alpha=0.04, color="green")

# -- Plot 1: Saturation Pressure
ax1 = fig.add_subplot(gs[0, 0])
for fk, df in results.items():
    twin_plot(ax1, fk, df, "P_sat_bar", "Pressure [bar]", "Saturation Pressure")
ax1.legend()

# -- Plot 2: Liquid & Vapor Density
ax2 = fig.add_subplot(gs[0, 1])
for fk, df in results.items():
    ax2.plot(df["T_C"], df["rho_liq"], color=COLORS[fk], ls="-",  lw=2, label=f"{fk} liquid")
    ax2.plot(df["T_C"], df["rho_vap"], color=COLORS[fk], ls=":",  lw=1.5, label=f"{fk} vapor")
ax2.set_xlabel("Temperature [°C]")
ax2.set_ylabel("Density [kg/m³]")
ax2.set_title("Liquid & Vapor Density")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# -- Plot 3: Surface Tension
ax3 = fig.add_subplot(gs[1, 0])
for fk, df in results.items():
    twin_plot(ax3, fk, df, "sigma_Nm", "Surface Tension [mN/m]", "Surface Tension", scale=1e3)
ax3.legend()

# -- Plot 4: Kinematic Viscosity
ax4 = fig.add_subplot(gs[1, 1])
for fk, df in results.items():
    twin_plot(ax4, fk, df, "nu_liq_m2s", "ν [mm²/s]", "Kinematic Viscosity (liquid)", scale=1e6)
ax4.legend()

# -- Plot 5 & 6: Bond Number per g-level
bo_cols = list(G_LEVELS.keys())
ls_map = ["-", "--", ":"]

ax5 = fig.add_subplot(gs[2, 0])
ax5.set_title(f"Bond Number — N₂O  (R={TANK_RADIUS_M} m)")
for i, label in enumerate(bo_cols):
    col = f"Bo_{label}"
    ax5.semilogy(results["N2O"]["T_C"], results["N2O"][col],
                 lw=2, ls=ls_map[i], label=label)
ax5.axhline(1, color="k", lw=0.8, ls="-.", alpha=0.5, label="Bo = 1 (transition)")
ax5.set_xlabel("Temperature [°C]")
ax5.set_ylabel("Bond Number [-]")
ax5.legend(fontsize=7)
ax5.grid(True, which="both", alpha=0.3)

ax6 = fig.add_subplot(gs[2, 1])
ax6.set_title(f"Bond Number — C₃H₆  (R={TANK_RADIUS_M} m)")
for i, label in enumerate(bo_cols):
    col = f"Bo_{label}"
    ax6.semilogy(results["C3H6"]["T_C"], results["C3H6"][col],
                 lw=2, ls=ls_map[i], label=label)
ax6.axhline(1, color="k", lw=0.8, ls="-.", alpha=0.5, label="Bo = 1 (transition)")
ax6.set_xlabel("Temperature [°C]")
ax6.set_ylabel("Bond Number [-]")
ax6.legend(fontsize=7)
ax6.grid(True, which="both", alpha=0.3)

plt.savefig("slosh_fluid_properties.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: slosh_fluid_properties.png")

# ── REFPROP coupling note ─────────────────────────────────────────────────────
print("""
─────────────────────────────────────────────────────────────────
HOW TO USE WITH YOUR REFPROP INSTALLATION
─────────────────────────────────────────────────────────────────
1. Install CoolProp:
      pip install CoolProp

2. Set your REFPROP path (bash):
      export RPPREFIX=/home/<you>/REFPROP   # or wherever REFPROP is installed

3. In this script, set:
      BACKEND = "REFPROP"

   CoolProp will then route all PropsSI() calls through REFPROP's
   equations of state, giving you REFPROP accuracy with the same
   Python API. Fluid strings become e.g. "REFPROP::N2O".

4. You can also call REFPROP directly through CoolProp's low-level API:
      from CoolProp.CoolProp import AbstractState
      AS = AbstractState("REFPROP", "N2O")
      AS.update(CoolProp.CoolProp.QT_INPUTS, 0, T_K)
      rho = AS.rhomass()
      # etc.
─────────────────────────────────────────────────────────────────
""")
