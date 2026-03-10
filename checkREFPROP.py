"""
Quick script to find the correct REFPROP fluid name for propylene/C3H6,
and verify N2O viscosity is working.
Run this on your machine to identify the right fluid string.
"""
import os
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI

REFPROP_PATH = os.environ.get(
    "REFPROP_PATH",
    "/home/karan.anand/Documents/PythonScripts/refprop/REFPROP/"
)
CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, REFPROP_PATH)

T_K = 293.15  # 20 C

# Candidate names to try for C3H6
candidates = [
    "REFPROP::Propylene",
    "REFPROP::PROPYLENE",
    "REFPROP::propylene",
    "REFPROP::C3H6",
    "REFPROP::c3h6",
    "REFPROP::R1270",   # refrigerant alias
    "REFPROP::r1270",
]

print("Searching for correct REFPROP fluid string for C3H6 / Propylene...")
print(f"{'Fluid string':<30} {'Result'}")
print("-" * 60)
for name in candidates:
    try:
        p = PropsSI("P", "T", T_K, "Q", 0, name)
        print(f"{name:<30} OK  P_sat = {p/1e5:.3f} bar")
    except Exception as e:
        short = str(e)[:60]
        print(f"{name:<30} FAIL: {short}")

# Also list .FLD files present so we can see exactly what REFPROP has
import glob
fluids_dir = os.path.join(REFPROP_PATH.rstrip("/"), "FLUIDS")
fld_files = sorted(glob.glob(os.path.join(fluids_dir, "*.FLD")))
print(f"\n\nFLD files in {fluids_dir}:")
for f in fld_files:
    print(" ", os.path.basename(f))