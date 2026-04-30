"""
environment.py
==============

GEO environmental modulators (physical-parameter group 4):

  * GEO ambient plasma (quiescent + substorm) — sets floating potentials,
    differential charging, and adds a small contribution to ion flux.
  * Thermal cycling — concurrent fatigue stressor that combines with
    erosion to drive the actual interconnect failure mode.

The GEO plasma background contribution to direct sputtering is normally
small compared to thruster CEX during firing, but it sets the *baseline*
floating potential of unbiased surfaces and can, during substorms, drive
differential charging that locally enhances ion impact energy on
spacecraft-conductive paths.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

KB_EV = 8.617e-5  # eV/K


@dataclass
class GEOEnvironment:
    """
    Simple representation of the GEO plasma environment.

    Quiescent values (LANL fits):
        n_e ~ 1e6 m^-3, T_e ~ 1 keV
    Substorm values (worst-case):
        n_e ~ 1e6 m^-3, T_e ~ 5-15 keV, hot population

    floating_potential_offset is what an isolated dielectric on the spacecraft
    sees relative to the local plasma potential during the chosen condition.
    """
    n_e: float = 1.0e6                 # electron density [m^-3]
    Te_eV: float = 1000.0              # electron temperature [eV]
    in_substorm: bool = False
    substorm_floating_offset: float = -8000.0  # V (severe charging case)

    def floating_potential(self) -> float:
        """Approximate floating potential of an isolated conductor [V]."""
        if self.in_substorm:
            return self.substorm_floating_offset
        # Quiescent: V_f ~ -3.5 * Te for Maxwellian electrons / Xe-ish ions
        return -3.5 * self.Te_eV

    def background_ion_flux(self) -> float:
        """
        Ambient ion flux (per m^2 per s). Useful to compare against thruster-
        driven flux. For GEO this is typically negligible vs CEX during firing
        but matters during long quiet periods.
        """
        # Bohm flux estimate
        v_b = np.sqrt(self.Te_eV * 1.602e-19 / (131.293 * 1.66e-27))
        return 0.6 * self.n_e * v_b


@dataclass
class ThermalCycling:
    """
    Thermal cycling concurrent with erosion. Doesn't modify Y(E,theta) directly
    but is needed for the actual failure-mode prediction (combined fatigue and
    cross-section thinning of interconnects).

    n_cycles_per_year   : eclipse + maneuver thermal cycles per year
    delta_T             : temperature swing per cycle [K]
    fatigue_exponent    : Coffin-Manson exponent for the interconnect material
                          (typical 2.0 for Ag-based interconnects)
    """
    n_cycles_per_year: float = 90.0     # GEO eclipse season + maneuvers
    delta_T: float = 120.0              # K
    fatigue_exponent: float = 2.0

    def life_fraction_per_year_thermal(self, n_cycles_to_failure: float = 5e4) -> float:
        """Fraction of fatigue life consumed per year by thermal cycling alone."""
        return self.n_cycles_per_year / n_cycles_to_failure

    def coupled_failure_factor(self, thinning_fraction: float) -> float:
        """
        Heuristic combined factor: fatigue life decreases as the interconnect
        cross-section thins (stress goes up).

        N_f / N_f0 ~ (A / A0)^(fatigue_exponent)
        """
        return max(1.0 - thinning_fraction, 1e-6) ** self.fatigue_exponent
