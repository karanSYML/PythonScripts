from plume_impingement_pipeline import *
import numpy as np                                                                                                                              
   
# 1. Configure thruster and material
thruster = ThrusterParams(discharge_voltage=300.0, mass_flow_rate=5e-6)
material = MaterialParams(thickness_um=25.0)
  
# 2. Create pipeline
pipeline = PlumePipeline(thruster, material)                  

# 3. Define your sweep
gen = pipeline.generator
gen.set_param_range("arm_reach_m", np.array([2.0, 3.0, 4.0, 5.0]))
gen.set_param_range("shoulder_yaw_deg", np.array([0, 30, 60, 90]))
  
# 4. Generate cases (fix everything else)
fixed = {"link_ratio": 0.5,
    "client_mass": 2500.0,
    "servicer_mass": 735.0,
    "panel_span_one_side": 16.0,
    "firing_duration_s": 25000.0,
    "mission_duration_yr": 5.0,
    "panel_tracking_deg": 0.0,
    }               

cases = gen.generate_reduced_matrix(fixed, ["arm_reach_m","shoulder_yaw_deg"])                                                                                                                                                          
# 5. Run
results = pipeline.run_sweep(cases,verbose=True)

# 6.Inspect     
print(pipeline.summary())
pipeline.export_results_csv("results.csv")