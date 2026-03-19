"""
Diphasic Fluid Properties for Slosh Analysis
Plasma Propellant: Xenon
Temperature range: -25 to +40 C (saturation conditions)

Notes:
  - Xe critical point: 36.4 C (309.52 K). Saturation properties are
    undefined above this. The script automatically caps Xe at Tc.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI

# -- REFPROP path setup -------------------------------------------------------
REFPROP_PATH = os.environ.get(
    "REFPROP_PATH",
    "/home/karan.anand/Documents/PythonScripts/refprop/REFPROP/"
)
CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, REFPROP_PATH)

# -- Configuration ------------------------------------------------------------

BACKEND = "REFPROP"   # "CoolProp" or "REFPROP"
                        # N2O viscosity only available with REFPROP backend

if BACKEND == "REFPROP":
    FLUIDS = {"Xe": "REFPROP::XENON"}
else:
    FLUIDS = {"Xe": "Xenon"}

T_MIN_C = -25.0
T_MAX_C = 25.0
N_POINTS = 91            # ~0.5 C steps
XE_T_CRIT_C = 16.58     # Xe critical temperature [C]

TANK_RADIUS_M = 0.15    # [m] update to your actual tank radius
G_LEVELS = {
    "1 g (ground)":      9.81,
    "0.01 g (low-g)":    0.0981,
    "0.001 g (micro-g)": 0.00981,
}

# -- Helpers ------------------------------------------------------------------

def get_saturation_properties(fluid_name, T_K):
    """
    Query saturation properties at T_K. Each property queried independently
    so a missing model returns NaN rather than failing the whole row.
    Raises ValueError if core properties unavailable (e.g. supercritical).
    """
    Q_LIQ, Q_VAP = 0, 1

    def safe(prop, quality):
        try:
            val = PropsSI(prop, "T", T_K, "Q", quality, fluid_name)
            return val if np.isfinite(val) else np.nan
        except Exception:
            return np.nan

    P_sat   = safe("P",               Q_LIQ)  # [Pa]
    rho_liq = safe("D",               Q_LIQ)  # [kg/m3]
    rho_vap = safe("D",               Q_VAP)  # [kg/m3]
    mu_liq  = safe("V",               Q_LIQ)  # [Pa.s] -- NaN for N2O/CoolProp
    sigma   = safe("surface_tension", Q_LIQ)  # [N/m]

    if np.isnan(P_sat) or np.isnan(rho_liq):
        raise ValueError(f"Core props unavailable at T={T_K:.2f} K (supercritical?)")

    nu_liq = mu_liq / rho_liq if not np.isnan(mu_liq) else np.nan

    return {
        "P_sat_Pa":   P_sat,
        "P_sat_bar":  P_sat / 1e5,
        "rho_liq":    rho_liq,
        "rho_vap":    rho_vap,
        "mu_liq_Pas": mu_liq,
        "nu_liq_m2s": nu_liq,
        "sigma_Nm":   sigma,
    }


def bond_number(rho_liq, sigma, g, R):
    if np.isnan(sigma) or sigma == 0:
        return np.nan
    return rho_liq * g * R**2 / sigma


# -- Main computation ---------------------------------------------------------

T_C_arr = np.linspace(T_MIN_C, T_MAX_C, N_POINTS)
T_K_arr = T_C_arr + 273.15
results = {}

for fluid_key, fluid_name in FLUIDS.items():
    rows = []
    n_supercrit = 0

    # Cap Xe below its critical point
    t_max = (XE_T_CRIT_C - 0.5) if fluid_key == "XE" else T_MAX_C

    print(f"\nComputing {fluid_key} ({BACKEND})"
          + (f"  [capped at {t_max:.1f} C — critical point]" if fluid_key == "XE" else ""))

    for T_C, T_K in zip(T_C_arr, T_K_arr):
        if T_C > t_max:
            n_supercrit += 1
            continue
        try:
            props = get_saturation_properties(fluid_name, T_K)
            bo = {f"Bo_{lbl}": bond_number(props["rho_liq"], props["sigma_Nm"], g, TANK_RADIUS_M)
                  for lbl, g in G_LEVELS.items()}
            rows.append({"T_C": T_C, **props, **bo})
        except ValueError:
            n_supercrit += 1
        except Exception as e:
            print(f"  Warning T={T_C:.1f} C: {e}")

    df = pd.DataFrame(rows)
    results[fluid_key] = df
    print(f"  {len(rows)} points OK, {n_supercrit} skipped (supercritical)")

    nan_cols = [c for c in df.columns if df[c].isna().any()]
    if nan_cols:
        print(f"  NaN columns: {nan_cols}")
        if "mu_liq_Pas" in nan_cols and fluid_key == "N2O":
            print("  -> N2O viscosity requires BACKEND='REFPROP'")

# -- Print tables -------------------------------------------------------------

display_cols = ["T_C", "P_sat_bar", "rho_liq", "rho_vap",
                "mu_liq_Pas", "nu_liq_m2s", "sigma_Nm"]

for fluid_key, df in results.items():
    print(f"\n{'='*70}")
    print(f"  {fluid_key} — Saturation Properties")
    print(f"{'='*70}")
    cols = [c for c in display_cols if c in df.columns]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# -- Export CSVs --------------------------------------------------------------

for fluid_key, df in results.items():
    fname = f"xe_slosh_properties_{fluid_key}.csv"
    df.to_csv(fname, index=False)
    print(f"Exported: {fname}")

# -- Plots --------------------------------------------------------------------

fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "Diphasic Properties for Slosh Analysis - Xenon at Saturation\n"
    f"(Xe capped at {XE_T_CRIT_C} C | backend - {BACKEND})",
    fontsize=11, fontweight="bold"
)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)
COLORS = {"Xe": "#1f77b4"}
LS     = {"Xe": "-"}

def fluid_plot(ax, y_col, ylabel, title, scale=1.0):
    for fk, df in results.items():
        if df.empty or y_col not in df.columns or df[y_col].isna().all():
            continue
        ax.plot(df["T_C"], df[y_col] * scale,
                color=COLORS[fk], ls=LS[fk], lw=2, label=fk)
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

fluid_plot(fig.add_subplot(gs[0, 0]), "P_sat_bar",  "Pressure [bar]",     "Saturation Pressure")
fluid_plot(fig.add_subplot(gs[1, 0]), "sigma_Nm",   "Surface tension [mN/m]", "Surface Tension", scale=1e3)
fluid_plot(fig.add_subplot(gs[1, 1]), "nu_liq_m2s", "nu [mm2/s]",         "Kinematic Viscosity (liquid)", scale=1e6)

ax2 = fig.add_subplot(gs[0, 1])
for fk, df in results.items():
    if df.empty or "rho_liq" not in df.columns:
        continue
    ax2.plot(df["T_C"], df["rho_liq"], color=COLORS[fk], ls="-",  lw=2,   label=f"{fk} liquid")
    ax2.plot(df["T_C"], df["rho_vap"], color=COLORS[fk], ls=":",  lw=1.5, label=f"{fk} vapor")
ax2.set_xlabel("Temperature [C]")
ax2.set_ylabel("Density [kg/m3]")
ax2.set_title("Liquid & Vapor Density")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

bo_labels = list(G_LEVELS.keys())
ls_map = ["-", "--", ":"]
for idx, (fk, df) in enumerate(results.items()):
    ax = fig.add_subplot(gs[2, idx])
    ax.set_title(f"Bond Number — {fk}  (R={TANK_RADIUS_M} m)")
    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        continue
    for i, lbl in enumerate(bo_labels):
        col = f"Bo_{lbl}"
        if col in df.columns and df[col].notna().any():
            ax.semilogy(df["T_C"], df[col], lw=2, ls=ls_map[i], label=lbl)
    ax.axhline(1, color="k", lw=0.8, ls="-.", alpha=0.5, label="Bo=1 (transition)")
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Bond Number [-]")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

plt.savefig("xe_slosh_fluid_properties.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: slosh_fluid_properties.png")