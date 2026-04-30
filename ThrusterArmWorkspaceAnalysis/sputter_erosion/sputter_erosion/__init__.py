"""
sputter_erosion
===============

Modular framework for computing erosion of spacecraft surfaces (in particular
solar-array interconnects) due to plasma-thruster plume impingement in the GEO
environment.

Architectural layers
--------------------
materials      : Target / projectile properties + Eckstein / Y-T / Tartz fits
yields         : Sputter-yield models (Y(E, theta, ion -> target))
plume          : Local plume state at the impingement point (j_i, IEDF, species, CEX)
geometry       : Satellite / thruster / array geometry, line-of-sight, sheath bias
environment    : GEO plasma background, thermal cycling, contamination
erosion        : Master integrator dh/dt = (1/rho) integral Y * j_i/e * cos(theta) dE dOmega
monte_carlo    : Bayesian / parametric uncertainty propagation (Zameshin & Sturm 2022 style)
mission        : Mission-level wrapper (duty cycle, lifetime, sensitivity)

Author : built for AK's Systems Simulation workflow, companion to `geo_rpo`.
"""

from .materials import Material, Projectile, MATERIALS, PROJECTILES
from .yields import (
    YieldModel,
    YamamuraTawara,
    EcksteinPreuss,
    Seah2005,
    AngularDependence,
    YamamuraAngular,
    EcksteinAngular,
)
from .plume import (
    PlumeState,
    HallThrusterPlume,
    GriddedIonPlume,
    IEDF,
    SpeciesFractions,
)
from .geometry import (
    Vector3,
    ThrusterPlacement,
    SolarArray,
    Interconnect,
    SatelliteGeometry,
    SheathModel,
)
from .environment import GEOEnvironment, ThermalCycling
from .erosion import ErosionIntegrator, ErosionResult
from .monte_carlo import ParameterPosterior, MonteCarloErosion
from .mission import MissionProfile, LifetimeAnalysis, FiringPhase

__version__ = "0.1.0"

__all__ = [
    # materials
    "Material", "Projectile", "MATERIALS", "PROJECTILES",
    # yields
    "YieldModel", "YamamuraTawara", "EcksteinPreuss", "Seah2005",
    "AngularDependence", "YamamuraAngular", "EcksteinAngular",
    # plume
    "PlumeState", "HallThrusterPlume", "GriddedIonPlume", "IEDF", "SpeciesFractions",
    # geometry
    "Vector3", "ThrusterPlacement", "SolarArray", "Interconnect",
    "SatelliteGeometry", "SheathModel",
    # environment
    "GEOEnvironment", "ThermalCycling",
    # erosion
    "ErosionIntegrator", "ErosionResult",
    # uncertainty
    "ParameterPosterior", "MonteCarloErosion",
    # mission
    "MissionProfile", "LifetimeAnalysis", "FiringPhase",
]
