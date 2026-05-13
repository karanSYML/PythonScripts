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

print(solarArray.list_properties().names.dtype)
print(batteries.list_properties().names)

# Power consumption for RCS steady state
rcs_idle_power = 4.0
rcs_manual_heating_100_duty_cycle = 83.9
rcs_preparing_100_duty_cycle = 97.6
rcs_preparing_70_duty_cycle = 73.6
rcs_preparing_0_duty_cycle = 17.7
rcs_armed = 6.0
rcs_cold_gas_firing_4th = 10.1
rcs_hot_gas_firing_4th = 14.1
