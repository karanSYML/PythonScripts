"""
plume.py
========

Local plume state at the impingement point. This is "physical-parameter group 1"
from the earlier discussion: the things that determine the boundary conditions
of the erosion integral, specifically:

  * j_i(theta)              — angular distribution of ion current density
  * IEDF f(E, theta)        — ion energy distribution at the impingement point
  * species fractions       — Xe+, Xe2+, Xe3+ current fractions
  * CEX wing                — slow charge-exchange ion population at large angles
  * neutral density n_n     — drives CEX production rate

All vectors of E and theta are in SI units (eV for energy, radians for angle,
A/m^2 for current density). Plume models can be evaluated at any (theta, range)
in the thruster body frame.

References
----------
Hofer & Walker, "Survey of xenon ion sputter yield data and fits relevant to
electric propulsion spacecraft integration," IEPC-2017-60.
Boyd & Dressler, J. Appl. Phys. 92 (2002) — CEX modeling.
Goebel & Katz, Fundamentals of Electric Propulsion (JPL series), Ch. 7.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple
import numpy as np

# Physical constants
E_CHARGE = 1.602176634e-19   # C
M_XE = 131.293 * 1.66054e-27 # kg
KB = 1.380649e-23            # J/K
SIGMA_CEX_XE = 5.5e-19       # m^2, total Xe+ + Xe -> Xe + Xe+ CEX cross-section
                             # (Pullins, Dressler ~ 50 eV; weakly E-dependent)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

@dataclass
class IEDF:
    """
    Ion energy distribution function at a point in the plume.

    Stored as a (E_grid, pdf_grid) tuple, normalised so that integral(pdf) dE = 1.
    """
    E_grid: np.ndarray   # [eV]
    pdf: np.ndarray      # [1/eV]

    def __post_init__(self):
        self.E_grid = np.asarray(self.E_grid, dtype=float)
        self.pdf = np.asarray(self.pdf, dtype=float)
        # renormalise
        norm = np.trapezoid(self.pdf, self.E_grid)
        if norm > 0:
            self.pdf = self.pdf / norm

    def mean_energy(self) -> float:
        return float(np.trapezoid(self.E_grid * self.pdf, self.E_grid))

    @classmethod
    def gaussian(cls, E_peak: float, sigma: float,
                 E_min: float = 1.0, E_max: Optional[float] = None,
                 n_points: int = 200) -> "IEDF":
        """Gaussian IEDF, useful for a primary beam (peak ~ V_d)."""
        if E_max is None:
            E_max = E_peak + 6.0 * sigma
        E = np.linspace(E_min, E_max, n_points)
        pdf = np.exp(-0.5 * ((E - E_peak) / sigma) ** 2)
        return cls(E_grid=E, pdf=pdf)

    @classmethod
    def cex_population(cls, E_max_cex: float = 50.0,
                       n_points: int = 200) -> "IEDF":
        """
        Low-energy CEX population: ions born cold and accelerated through the
        local sheath. Modeled as a falling exponential between 1 and E_max_cex
        eV; tune E_max_cex to the local sheath potential.
        """
        E = np.linspace(1.0, E_max_cex, n_points)
        pdf = np.exp(-E / (E_max_cex / 3.0))
        return cls(E_grid=E, pdf=pdf)

    @classmethod
    def composite(cls, primary: "IEDF", cex: "IEDF",
                  cex_fraction: float) -> "IEDF":
        """
        Combined IEDF = (1 - f_cex) * primary + f_cex * cex, evaluated on the
        union of the two energy grids. f_cex is the fractional ion-current
        contribution from CEX at the impingement location.
        """
        E_union = np.unique(np.concatenate([primary.E_grid, cex.E_grid]))
        p1 = np.interp(E_union, primary.E_grid, primary.pdf, left=0.0, right=0.0)
        p2 = np.interp(E_union, cex.E_grid, cex.pdf, left=0.0, right=0.0)
        pdf = (1.0 - cex_fraction) * p1 + cex_fraction * p2
        return cls(E_grid=E_union, pdf=pdf)


@dataclass
class SpeciesFractions:
    """
    Ion-species current fractions at the source. Doubly-charged ions hit at
    twice the energy, and have higher sputter yield per ion than singly-charged
    at the same per-charge potential drop.

    For a typical Hall thruster operating at 300 V:
        Xe+  ~ 0.78,  Xe2+ ~ 0.20,  Xe3+ ~ 0.02 (current fractions)
    Higher discharge voltage -> higher multiply-charged fraction.
    """
    f_xe1: float = 0.85
    f_xe2: float = 0.13
    f_xe3: float = 0.02

    def normalize(self) -> "SpeciesFractions":
        s = self.f_xe1 + self.f_xe2 + self.f_xe3
        return SpeciesFractions(self.f_xe1 / s, self.f_xe2 / s, self.f_xe3 / s)

    def number_fractions(self) -> Tuple[float, float, float]:
        """Convert current fractions to number-density fractions (n_i)."""
        # n_i = (j_i / Z_i) / sum(j_k / Z_k)
        n1 = self.f_xe1 / 1.0
        n2 = self.f_xe2 / 2.0
        n3 = self.f_xe3 / 3.0
        s = n1 + n2 + n3
        return n1 / s, n2 / s, n3 / s


# ---------------------------------------------------------------------------
# Plume models
# ---------------------------------------------------------------------------

@dataclass
class PlumeState:
    """
    Local plume state evaluated at a specific point (theta, r) relative to
    the thruster, in the thruster body frame.

    Attributes
    ----------
    j_i              : ion current density at the surface [A/m^2]
    iedf             : ion energy distribution function (composite primary+CEX)
    species          : species fractions
    incidence_angle  : geometric angle of the local plume direction relative
                       to the surface normal of the impingement target [rad].
                       This is what feeds the angular-yield model.
    n_neutrals       : local neutral xenon density [m^-3], used only if you
                       want to include further CEX production along the line
                       of sight to a downstream surface
    """
    j_i: float
    iedf: IEDF
    species: SpeciesFractions
    incidence_angle: float
    n_neutrals: float = 0.0


class PlumeModel(ABC):
    """Abstract plume model; given (theta, r) returns a PlumeState."""

    @abstractmethod
    def evaluate(self, theta_rad: float, r_m: float,
                 surface_normal_angle_rad: float = 0.0) -> PlumeState:
        ...


@dataclass
class HallThrusterPlume(PlumeModel):
    """
    Semi-empirical Hall-thruster plume model.

    Parameters
    ----------
    V_d              : discharge voltage [V]
    I_beam           : beam current [A]
    mdot_neutral     : neutral mass flow rate [kg/s] (drives CEX)
    half_angle_90    : 90% beam half-angle [rad] (typically 25-45 deg)
    cex_wing_amp     : amplitude of CEX wing relative to peak ion current
    cex_wing_width   : angular width of CEX wing [rad]
    species          : species fractions at the source
    sheath_potential : local sheath potential [V] for CEX ions at the surface
                       (added to their birth energy on biased surfaces)
    """

    V_d: float
    I_beam: float
    mdot_neutral: float
    half_angle_90: float = np.deg2rad(35.0)
    cex_wing_amp: float = 0.02
    cex_wing_width: float = np.deg2rad(40.0)
    species: SpeciesFractions = field(default_factory=SpeciesFractions)
    sheath_potential: float = 20.0   # V

    def _angular_current_density(self, theta_rad: float, r_m: float) -> Tuple[float, float]:
        """
        Returns (j_primary, j_cex) at angle theta from thruster axis, range r.

        Primary beam: cos^n distribution truncated at half_angle_90.
        CEX wing: broad Gaussian centered on the thruster axis with much
        wider angular extent.
        """
        # Cosine-power exponent giving 90% within half_angle_90
        # Approximate: for cos^n, 90% is contained within theta where
        # cos^(n+1)(theta) = 0.1; n ~ log(0.1)/log(cos(half_angle))
        n_exp = np.log(0.1) / np.log(np.cos(self.half_angle_90))
        n_exp = max(n_exp, 1.0)

        # Reference current density on axis at 1 m
        j_axis_1m = self.I_beam * (n_exp + 1.0) / (2.0 * np.pi * 1.0 ** 2)

        cos_t = np.cos(theta_rad) if abs(theta_rad) < np.pi / 2 else 0.0
        primary = j_axis_1m * (cos_t ** n_exp) / (r_m ** 2) if cos_t > 0 else 0.0

        # CEX wing: Gaussian in theta, falls as 1/r^2 (point-source approximation)
        cex = (
            j_axis_1m * self.cex_wing_amp
            * np.exp(-0.5 * (theta_rad / self.cex_wing_width) ** 2)
            / (r_m ** 2)
        )
        return primary, cex

    def evaluate(self, theta_rad: float, r_m: float,
                 surface_normal_angle_rad: float = 0.0) -> PlumeState:
        j_primary, j_cex = self._angular_current_density(theta_rad, r_m)
        j_total = j_primary + j_cex
        f_cex = j_cex / j_total if j_total > 0 else 0.0

        # Primary beam IEDF: peaks near V_d with ~10% energy spread
        primary_iedf = IEDF.gaussian(
            E_peak=self.V_d, sigma=max(0.10 * self.V_d, 5.0)
        )
        # CEX IEDF: low-energy population accelerated by sheath
        cex_iedf = IEDF.cex_population(
            E_max_cex=max(self.sheath_potential * 1.5, 30.0)
        )
        iedf = IEDF.composite(primary_iedf, cex_iedf, f_cex)

        # Crude neutral-density estimate: thermal expansion from the exit plane
        # n_n(r) ~ (mdot/M_xe) / (4 pi r^2 v_thermal)
        T_neutral = 600.0  # K, typical Hall thruster discharge channel
        v_th = np.sqrt(8.0 * KB * T_neutral / (np.pi * M_XE))
        n_n = self.mdot_neutral / (M_XE * 4.0 * np.pi * r_m ** 2 * v_th + 1e-30)

        return PlumeState(
            j_i=j_total,
            iedf=iedf,
            species=self.species,
            incidence_angle=surface_normal_angle_rad,
            n_neutrals=n_n,
        )


@dataclass
class GriddedIonPlume(PlumeModel):
    """
    Gridded ion engine plume. Beam is much more collimated than HET (typical
    half_angle_90 ~ 15-20 deg), but CEX wing is comparable.

    V_b is the net beam voltage (typically 1100-1800 V for NEXT, NSTAR).
    """

    V_b: float
    I_beam: float
    mdot_neutral: float
    half_angle_90: float = np.deg2rad(20.0)
    cex_wing_amp: float = 0.015
    cex_wing_width: float = np.deg2rad(50.0)
    species: SpeciesFractions = field(default_factory=lambda: SpeciesFractions(0.92, 0.08, 0.0))
    sheath_potential: float = 15.0

    def evaluate(self, theta_rad: float, r_m: float,
                 surface_normal_angle_rad: float = 0.0) -> PlumeState:
        # Reuse Hall thruster geometry with GIE parameters
        equiv = HallThrusterPlume(
            V_d=self.V_b, I_beam=self.I_beam, mdot_neutral=self.mdot_neutral,
            half_angle_90=self.half_angle_90, cex_wing_amp=self.cex_wing_amp,
            cex_wing_width=self.cex_wing_width, species=self.species,
            sheath_potential=self.sheath_potential,
        )
        return equiv.evaluate(theta_rad, r_m, surface_normal_angle_rad)
