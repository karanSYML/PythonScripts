"""
erosion.py
==========

Master erosion integrator.

For each (thruster, interconnect) pair, evaluate

    dh/dt = (1 / n_atomic) * sum_species f_species *
            integral over E of  Y(E, theta_inc; species -> material)
                                * j_i(E) / (Z_species * e) dE

where:
  Y(E, theta) is the FullYield (energy + angular dep + sub-threshold floor),
  j_i(E) is the local ion energy spectrum [A/m^2/eV] = j_total * IEDF.pdf(E),
  the species sum accounts for Xe+, Xe2+, Xe3+ with the right energy boost
    (Xe2+ effectively impacts at 2*E for the same per-charge potential),
  the sheath model adds the local string-bias-induced energy boost to the
    CEX population.

Returns thinning rate in m/s and accumulated material loss over a duration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

from .geometry import SatelliteGeometry, Interconnect, ThrusterPlacement, SolarArray
from .yields import FullYield, EcksteinPreuss, EcksteinAngular
from .plume import PlumeState, SpeciesFractions
from .materials import get_projectile, Material


E_CHARGE = 1.602176634e-19


@dataclass
class ErosionResult:
    """Per-interconnect erosion result."""
    thruster_index: int
    array_index: int
    interconnect_index: int
    material: str
    j_i: float                 # local ion current density [A/m^2]
    incidence_angle_deg: float
    mean_E_eV: float
    sheath_boost_eV: float
    erosion_rate_m_s: float    # local thickness loss rate
    total_thinning_m: float    # over the firing duration
    fluence_ions_m2: float     # cumulative ion fluence
    fraction_remaining: float  # 1 - thinning / initial thickness


class ErosionIntegrator:
    """
    Couples geometry + plume + materials + yield model + sheath into the
    erosion-rate calculation.

    Parameters
    ----------
    yield_model    : a FullYield combining energy and angular dependence
    include_xe2    : include Xe2+ contribution
    include_xe3    : include Xe3+ contribution
    apply_sheath   : add sheath_potential_eV to the CEX population energies
                     for biased-array calculations
    """

    def __init__(
        self,
        yield_model: Optional[FullYield] = None,
        include_xe2: bool = True,
        include_xe3: bool = False,
        apply_sheath: bool = True,
    ):
        self.yield_model = yield_model or FullYield(
            energy_model=EcksteinPreuss(),
            angular_model=EcksteinAngular(),
            subthreshold_floor=0.0,
        )
        self.include_xe2 = include_xe2
        self.include_xe3 = include_xe3
        self.apply_sheath = apply_sheath

    # -- core per-species rate ---------------------------------------------

    def _species_rate(
        self,
        species_proj_name: str,    # "Xe" or "Xe2" or "Xe3"
        species_charge: int,
        f_current: float,
        plume_state: PlumeState,
        material: Material,
        sheath_eV: float,
        E_multiplier: float = 1.0,
    ) -> float:
        """
        Integrate Y(E,theta) * (j_i / (Z e)) * f(E) dE for one species.
        Returns atom flux [atoms / m^2 / s] sputtered off the surface by
        this species.
        """
        if f_current <= 0.0 or plume_state.j_i <= 0.0:
            return 0.0

        proj = get_projectile(species_proj_name)
        E_grid = plume_state.iedf.E_grid * E_multiplier
        pdf = plume_state.iedf.pdf / E_multiplier  # preserve normalization
        # Optionally shift CEX-tail energies by the sheath drop.
        if self.apply_sheath and sheath_eV > 0:
            # Apply the sheath boost to all ions (it's an absolute add); for
            # primary beam ions at hundreds of eV the boost is small, for CEX
            # ions at tens of eV it's the dominant contribution.
            E_grid = E_grid + sheath_eV

        Y = self.yield_model(
            E_grid, plume_state.incidence_angle, proj, material
        )
        # Integrand: yield (atoms/ion) * differential ion-number flux
        # ion-number flux = j_total * f_current / (Z e) (split by species)
        # spectral ion-number flux at E = above * pdf(E)
        n_flux_per_E = (
            plume_state.j_i * f_current
            / (species_charge * E_CHARGE)
            * pdf
        )
        atom_flux = float(np.trapezoid(Y * n_flux_per_E, E_grid))
        return atom_flux

    def _interconnect_rate(
        self,
        plume_state: PlumeState,
        ic: Interconnect,
        sheath_eV: float,
    ) -> Dict[str, float]:
        """
        Atom flux summed over ion species, plus useful diagnostics.
        """
        material = ic.material()
        species = plume_state.species

        atom_flux = 0.0
        atom_flux += self._species_rate(
            "Xe", 1, species.f_xe1, plume_state, material, sheath_eV, E_multiplier=1.0
        )
        if self.include_xe2:
            atom_flux += self._species_rate(
                "Xe2", 2, species.f_xe2, plume_state, material, sheath_eV,
                E_multiplier=2.0,
            )
        if self.include_xe3 and species.f_xe3 > 0:
            atom_flux += self._species_rate(
                "Xe", 3, species.f_xe3, plume_state, material, sheath_eV,
                E_multiplier=3.0,
            )

        # Convert atom flux to thinning rate
        thinning_rate = atom_flux / material.n_atomic   # m/s

        return {
            "atom_flux": atom_flux,
            "thinning_rate": thinning_rate,
            "mean_E": plume_state.iedf.mean_energy(),
            "incidence_deg": float(np.rad2deg(plume_state.incidence_angle)),
        }

    # -- public API --------------------------------------------------------

    def evaluate(
        self,
        geometry: SatelliteGeometry,
        firing_duration_s: float,
    ) -> List[ErosionResult]:
        """
        Compute erosion at every interconnect for all currently-active
        thrusters (those listed in `geometry.thrusters`), assuming continuous
        firing for `firing_duration_s` seconds. For mission-level duty cycling,
        use mission.LifetimeAnalysis.
        """
        results: List[ErosionResult] = []

        for ti, thr in enumerate(geometry.thrusters):
            for ai, arr in enumerate(geometry.solar_arrays):
                for ii, ic in enumerate(arr.interconnects):
                    p_body = arr.interconnect_position_body(ic)
                    n_body = arr.interconnect_normal_body(ic)
                    state = thr.evaluate_plume_at(p_body, n_body)
                    if state.j_i <= 0.0 or state.incidence_angle >= np.pi / 2:
                        continue

                    sheath_boost = 0.0
                    if self.apply_sheath:
                        sheath_boost = geometry.sheath.added_energy_eV(
                            ic.string_position
                        )

                    diag = self._interconnect_rate(state, ic, sheath_boost)
                    rate = diag["thinning_rate"]
                    total = rate * firing_duration_s
                    fluence = state.j_i / E_CHARGE * firing_duration_s

                    results.append(ErosionResult(
                        thruster_index=ti,
                        array_index=ai,
                        interconnect_index=ii,
                        material=ic.material_name,
                        j_i=state.j_i,
                        incidence_angle_deg=diag["incidence_deg"],
                        mean_E_eV=diag["mean_E"],
                        sheath_boost_eV=sheath_boost,
                        erosion_rate_m_s=rate,
                        total_thinning_m=total,
                        fluence_ions_m2=fluence,
                        fraction_remaining=max(
                            0.0, 1.0 - total / ic.exposed_thickness
                        ),
                    ))
        return results

    def total_thinning_per_interconnect(
        self,
        geometry: SatelliteGeometry,
        firing_duration_s: float,
    ) -> Dict[tuple, float]:
        """
        Sum across thrusters: returns dict keyed by (array_idx, ic_idx)
        with cumulative thinning [m] over the firing duration.
        """
        results = self.evaluate(geometry, firing_duration_s)
        out: Dict[tuple, float] = {}
        for r in results:
            key = (r.array_index, r.interconnect_index)
            out[key] = out.get(key, 0.0) + r.total_thinning_m
        return out
