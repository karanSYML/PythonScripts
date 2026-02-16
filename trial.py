from mission_scenario import Scenario, build_maneuvers;
from propulsion_budget import budget_from_dry_mass, print_report; 

s=Scenario(); 

m,_=build_maneuvers(s, mode='computed'); 
r=budget_from_dry_mass(s.dry_mass_kg, m); 
print_report(r)