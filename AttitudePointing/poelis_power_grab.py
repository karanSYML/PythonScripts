from poelis_sdk import PoelisClient
import numpy as np

# Create client
poelis_client = PoelisClient(
    api_key="poelis_sdk_KzCVmOrA1wn-XDQqwaWJAJ7ewRZuGhdyIO14Rx3wQCw",    # API Keys
)

og3 = poelis_client.browser.og3.og003_technical_data.v6
og3_power = og3.system.power
og3_propulsion = og3.system.propulsion

solarArray = og3_power.solararray
batteries = og3_power.batteries

# Power consumption with PPS
ppu = og3_propulsion.pps.powerprocessingunit
ppu_peak_power = ppu.peak_power_consumption.value

ppu_avg_power_consumption = np.array([ppu.average_power_consumption_min.value, ppu.average_power_consumption_max.value])

print("=== Solar Array ===")
for name in solarArray.list_properties().names:
    try:
        prop = getattr(solarArray, name)
        print(f"  {name}: {prop.value} {getattr(prop, 'unit', '')}")
    except Exception as e:
        print(f"  {name}: <{e}>")

print("\n=== Batteries ===")
for name in batteries.list_properties().names:
    try:
        prop = getattr(batteries, name)
        print(f"  {name}: {prop.value} {getattr(prop, 'unit', '')}")
    except Exception as e:
        print(f"  {name}: <{e}>")

print("\n=== PPU ===")
print(f"  peak_power_consumption: {ppu_peak_power}")
print(f"  avg_power_consumption (min/max): {ppu_avg_power_consumption}")


# Power consumption for RCS steady state
rcs_idle_power = 4.0
rcs_manual_heating_100_duty_cycle = 83.9
rcs_preparing_100_duty_cycle = 97.6
rcs_preparing_70_duty_cycle = 73.6
rcs_preparing_0_duty_cycle = 17.7
rcs_armed = 6.0
rcs_cold_gas_firing_4th = 10.1
rcs_hot_gas_firing_4th = 14.1

print("\n=== RCS (hardcoded) ===")
print(f"  idle: {rcs_idle_power} W")
print(f"  armed: {rcs_armed} W")
print(f"  cold gas firing (4th): {rcs_cold_gas_firing_4th} W")
print(f"  hot gas firing (4th): {rcs_hot_gas_firing_4th} W")
print(f"  preparing (100% duty): {rcs_preparing_100_duty_cycle} W")
print(f"  preparing (70% duty): {rcs_preparing_70_duty_cycle} W")

# Camera payload
print("\n=== Camera / Payload ===")
og3_payload = og3.system.payload
all_names = og3_payload.list_properties().names
print(f"  all properties: {all_names}")
for child_name in all_names:
    try:
        child = getattr(og3_payload, child_name)
        print(f"\n  [{child_name}]")
        for prop_name in child.list_properties().names:
            try:
                prop = getattr(child, prop_name)
                print(f"    {prop_name}: {prop.value} {getattr(prop, 'unit', '')}")
            except Exception as e:
                print(f"    {prop_name}: <{e}>")
    except Exception as e:
        print(f"  {child_name}: <{e}>")
