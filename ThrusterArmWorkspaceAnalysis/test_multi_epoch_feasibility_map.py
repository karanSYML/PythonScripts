import numpy as np
from plume_impingement_pipeline import RoboticArmGeometry, StackConfig
from composite_mass_model import CompositeMassModel
from feasibility_cells import FeasibilityConfig
from feasibility_map import build_feasibility_maps, compute_pivot, print_summary

# ---------- Geometry setup ----------
arm = RoboticArmGeometry()
stack = StackConfig(
    servicer_mass=744.0, client_mass=2800.0,
    servicer_bus_x=0.9, servicer_bus_y=1.5, servicer_bus_z=0.8,
    client_bus_x=2.3, client_bus_y=3.0, client_bus_z=5.0,
    panel_span_one_side=16.0, panel_width=2.5, lar_offset_z=0.05,
)

SERVICER_YAW_DEG = -25.0

# ---- Pivot and Mass model ---
pivot = compute_pivot(arm, stack, servicer_yaw_deg=SERVICER_YAW_DEG)
mass = CompositeMassModel.from_json(stack=stack)

# ---- Feasibility Configuration ----
config = FeasibilityConfig.from_json()
n_hat_ee = np.array([0.1455, 0.9189, 0.3666]) # from feasibilty_inputs.json

# ---- Run ---
results = build_feasibility_maps(
    arm, mass, stack, pivot, n_hat_ee,
    servicer_yaw_deg=SERVICER_YAW_DEG,
    config=config,
    directions=['N', '-N', 'E', '-W'],
    verbose=True
)

print_summary(results)

