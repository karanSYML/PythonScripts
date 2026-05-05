import numpy as np
from plume_impingement_pipeline import StackConfig
from composite_mass_model import CompositeMassModel

# from . import plume_impingement_pipeline
# from . import composite_mass_model

stack = StackConfig(servicer_mass=744.0, client_mass=2800.0, 
                    servicer_bus_x=0.9, servicer_bus_y=1.5, servicer_bus_z=0.8,
                    client_bus_x=2.3, client_bus_y=3.0, client_bus_z=5.0,
                    panel_span_one_side=16.0, panel_width=2.5, lar_offset_z=0.05)


mass = CompositeMassModel.from_json(stack=stack)

epochs = [0, 456, 913, 1369, 1825]
print("Epoch [day] | CoG (X, Y, Z) [m] | Migration [cm]")
for tau in epochs:
    cog = mass.p_CoG_LAR(tau)
    mig = mass.cog_migration_magnitude(tau) * 100
    print(f" {tau:5.0f} | ({cog[0]:+.3f}, {cog[1]:+.3f}, {cog[2]:+.3f}) | {mig:.2f}")

print(f"\n Propellant Exhausted: day {mass.propellant_exhausted_day():.0f}")
print(f"Suggested epoch spacing for eps=5cm: {mass.suggested_epoch_spacing(0.05):.0f} days")