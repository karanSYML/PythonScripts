# sweep Kp and measure max flex energy — find the stability boundary
from flexible_spacecraft import *

kp_values = np.linspace(2, 60, 30)
max_energies = []

for kp in kp_values:
    r = simulate(SystemParams(), ControllerParams(Kp=kp, Kd=4),
                 NotchParams(), SimParams(t_end=20), label=f"Kp={kp:.1f}")
    max_energies.append(np.max(r.flex_energy))

plt.semilogy(kp_values, max_energies)
plt.xlabel("Kp"); plt.ylabel("Max Flex Energy"); plt.title("Stability Boundary")
plt.show()