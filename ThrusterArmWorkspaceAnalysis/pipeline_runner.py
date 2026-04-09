from plume_impingement_pipeline import *
from openplume_exporter import export_openplume_cases
import numpy as np


# 1. Configure thruster and material
thruster = ThrusterParams(discharge_voltage=300.0, mass_flow_rate=5e-6)
material = MaterialParams(thickness_um=10.0)
  
# 2. Create pipeline
pipeline = PlumePipeline(thruster, material)                  

# 3. Define your sweep
gen = pipeline.generator

gen.set_param_range("client_mass", np.array([1500.0, 2000.0, 2500.0, 3000.0]))
# gen.set_param_range("shoulder_yaw_deg", np.array([0, 30, 60, 90]))
gen.set_param_range("mission_duration_yr", np.array([3.0, 5.0, 7.0]))
  
# 4. Generate cases (fix everything else)
fixed = {"link_ratio": 0.75,
    "arm_reach_m": 2.62,
    "servicer_mass": 735.0,
    "panel_span_one_side": 16.0,
    "firing_duration_s": 15000.0,
    "panel_tracking_deg": 0.0,
    "shoulder_yaw_deg": 0
}

cases = gen.generate_reduced_matrix(fixed, ["client_mass", "mission_duration_yr"])

# 5. Run
results = pipeline.run_sweep(cases,verbose=True)

# Generate results and heatmaps
output_dir = "pipeline_runner_output"
os.makedirs(output_dir, exist_ok=True)

print("\n [GENERATING HEATMAPS]")

f1 = generate_heatmaps(results, "client_mass", "mission_duration_yr", metric="max_erosion_um", output_dir=output_dir, thickness_um=10.0)

# 6.Inspect     
print(pipeline.summary())
pipeline.export_results_csv(os.path.join(output_dir, "results.csv"))

op_cases = pipeline.get_openplume_cases()

print(f" Cases for OpenPlume simulation: {len(op_cases)}")

out_dir = "pipeline_runner_output/openplume_cases"

manifest = export_openplume_cases(
    op_cases, output_dir=out_dir, thruster=thruster, material=material,
)

