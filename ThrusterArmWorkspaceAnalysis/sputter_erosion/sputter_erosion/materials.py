"""
materials.py
============

Projectile and target material data, plus fitted sputter-yield parameters from
the literature (Yamamura-Tawara 1996, Eckstein-Preuss 2003, Tartz 2011 EP-relevant
data, Zameshin & Sturm 2022 Bayesian re-fits).

Atomic / surface-binding numbers are pulled from standard references
(Y. Yamamura & H. Tawara, At. Data Nucl. Data Tables 62, 149 (1996);
W. Eckstein, Calculated Sputtering, Reflection and Range Values, IPP-Report 9/132;
NPL Seah tables for Xe). Where multiple values exist, the EP-community-preferred
value is used and noted in the comment.

These tables are deliberately conservative and traceable; they are NOT intended
as a substitute for project-specific test data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# Projectiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Projectile:
    """Incident-ion species."""
    name: str
    Z: int                 # atomic number
    M: float               # atomic mass [u]
    Es_self: float = 0.0   # self surface-binding energy [eV] (0 for inert gases)


PROJECTILES: Dict[str, Projectile] = {
    "Xe":  Projectile("Xe",  54, 131.293, 0.0),
    "Xe2": Projectile("Xe2", 54, 131.293, 0.0),  # doubly charged xenon
    "Kr":  Projectile("Kr",  36,  83.798, 0.0),
    "Ar":  Projectile("Ar",  18,  39.948, 0.0),
    "I":   Projectile("I",   53, 126.904, 1.07),  # iodine, EP-relevant
}


# ---------------------------------------------------------------------------
# Target materials
# ---------------------------------------------------------------------------

@dataclass
class YamamuraParameters:
    """
    Parameters for the Yamamura-Tawara (1996) energy-dependence fit.

    Y(E) = 0.42 * (alpha_star * Q * K * sn(eps)) / (Us * (1 + 0.35 * Us * se(eps)))
           * (1 - sqrt(Eth / E))**s

    Q       : material-dependent linear scaling
    s       : threshold-shape exponent (~2.5 for most metals)
    Eth     : sputtering threshold energy [eV]
    Us      : surface binding energy [eV] (commonly the heat of sublimation)
    """
    Q: float
    s: float
    Eth: float
    Us: float


@dataclass
class EcksteinParameters:
    """
    Parameters for the Eckstein-Preuss (2003) revised fit.

    Y(E) = q * sn_KrC(eps) * (E/Eth - 1)**mu /
           (lam/w(eps) + (E/Eth - 1)**mu)

    q, lam, mu : energy-dependence parameters
    Eth        : threshold energy [eV]
    """
    q: float
    lam: float
    mu: float
    Eth: float


@dataclass
class BayesianPosterior:
    """
    Mean and standard deviation for sputter-yield fit parameters from Bayesian
    MCMC re-fits (Zameshin & Sturm 2022, plus EP-community estimates where
    direct posteriors are unavailable).

    Use these to feed monte_carlo.ParameterPosterior for V&V uncertainty
    propagation. Where no published posterior exists, defaults are conservative
    Gaussians around the point estimate.
    """
    Q_mean: float
    Q_std: float
    s_mean: float
    s_std: float
    Eth_mean: float
    Eth_std: float


@dataclass
class Material:
    """
    Target material with all relevant physical constants and fitted parameters
    for a specific projectile species (typically Xe for EP applications).
    """
    name: str
    Z: int                              # atomic number
    M: float                            # atomic mass [u]
    rho: float                          # mass density [kg/m^3]
    n_atomic: float                     # atomic number density [atoms/m^3]
    Us: float                           # surface binding energy [eV]
    Eb_bulk: float = 0.0                # bulk binding energy [eV] (Hofsass 2023)

    # Fitted parameters per projectile (key = projectile name)
    yt_params: Dict[str, YamamuraParameters] = field(default_factory=dict)
    eckstein_params: Dict[str, EcksteinParameters] = field(default_factory=dict)
    bayesian: Dict[str, BayesianPosterior] = field(default_factory=dict)

    def has_yt(self, projectile: str) -> bool:
        return projectile in self.yt_params

    def has_eckstein(self, projectile: str) -> bool:
        return projectile in self.eckstein_params


# ---------------------------------------------------------------------------
# Material library — focused on EP / spacecraft integration
# ---------------------------------------------------------------------------
# Notes on the Ag entry (the headline interconnect material):
#   - Y-T params: from the original Yamamura-Tawara (1996) tables.
#   - Eckstein params: anchored to Tartz et al. EPJ D 61 (2011) low-energy
#     Xe -> Ag dataset (<1500 eV) plus Eckstein 2007 IPP-Report 9/132 fits.
#   - Bayesian posteriors: extrapolated from Zameshin & Sturm (2022)
#     methodology; where Ar-data-only posteriors exist, projected to Xe via
#     the Q(Z2) scaling per Yamamura-Nakagawa-Enoki (1984).
# These numerical values should be re-checked against your in-house tank data.

MATERIALS: Dict[str, Material] = {
    "Ag": Material(
        name="Ag", Z=47, M=107.868,
        rho=10490.0, n_atomic=5.86e28,
        Us=2.97, Eb_bulk=2.95,
        yt_params={
            "Xe":  YamamuraParameters(Q=2.10, s=2.5, Eth=23.0,  Us=2.97),
            "Xe2": YamamuraParameters(Q=2.10, s=2.5, Eth=23.0,  Us=2.97),
        },
        eckstein_params={
            "Xe":  EcksteinParameters(q=4.20, lam=0.45, mu=1.80, Eth=23.0),
            "Xe2": EcksteinParameters(q=4.20, lam=0.45, mu=1.80, Eth=23.0),
        },
        bayesian={
            "Xe": BayesianPosterior(
                Q_mean=2.10, Q_std=0.21,
                s_mean=2.5,  s_std=0.18,
                Eth_mean=23.0, Eth_std=4.0,
            ),
        },
    ),
    "Mo": Material(
        name="Mo", Z=42, M=95.95,
        rho=10280.0, n_atomic=6.42e28,
        Us=6.83, Eb_bulk=6.81,
        yt_params={
            "Xe":  YamamuraParameters(Q=0.85, s=2.8, Eth=49.0,  Us=6.83),
            "Xe2": YamamuraParameters(Q=0.85, s=2.8, Eth=49.0,  Us=6.83),
        },
        eckstein_params={
            "Xe":  EcksteinParameters(q=1.45, lam=0.40, mu=1.95, Eth=49.0),
        },
        bayesian={
            "Xe": BayesianPosterior(
                Q_mean=0.85, Q_std=0.09,
                s_mean=2.8,  s_std=0.20,
                Eth_mean=49.0, Eth_std=6.0,
            ),
        },
    ),
    "Al": Material(
        name="Al", Z=13, M=26.982,
        rho=2700.0, n_atomic=6.02e28,
        Us=3.39, Eb_bulk=3.35,
        yt_params={
            "Xe": YamamuraParameters(Q=1.10, s=2.5, Eth=27.0, Us=3.39),
        },
        eckstein_params={
            "Xe": EcksteinParameters(q=1.95, lam=0.42, mu=1.85, Eth=27.0),
        },
        bayesian={
            "Xe": BayesianPosterior(
                Q_mean=1.10, Q_std=0.13,
                s_mean=2.5,  s_std=0.20,
                Eth_mean=27.0, Eth_std=4.5,
            ),
        },
    ),
    "Ti": Material(
        name="Ti", Z=22, M=47.867,
        rho=4506.0, n_atomic=5.67e28,
        Us=4.85, Eb_bulk=4.85,
        yt_params={
            "Xe": YamamuraParameters(Q=0.55, s=2.5, Eth=33.0, Us=4.85),
        },
        eckstein_params={
            "Xe": EcksteinParameters(q=0.95, lam=0.40, mu=1.85, Eth=33.0),
        },
        bayesian={
            "Xe": BayesianPosterior(
                Q_mean=0.55, Q_std=0.07,
                s_mean=2.5,  s_std=0.20,
                Eth_mean=33.0, Eth_std=5.0,
            ),
        },
    ),
    "W": Material(
        name="W", Z=74, M=183.84,
        rho=19250.0, n_atomic=6.32e28,
        Us=8.68, Eb_bulk=8.66,
        yt_params={
            "Xe": YamamuraParameters(Q=0.65, s=2.8, Eth=42.0, Us=8.68),
        },
        eckstein_params={
            "Xe": EcksteinParameters(q=1.10, lam=0.38, mu=1.90, Eth=42.0),
        },
        bayesian={
            "Xe": BayesianPosterior(
                Q_mean=0.65, Q_std=0.08,
                s_mean=2.8,  s_std=0.22,
                Eth_mean=42.0, Eth_std=5.5,
            ),
        },
    ),
    # Polymer / dielectric — Kapton (yield much lower; use with caution,
    # mainly drives mass-loss not interconnect erosion)
    "Kapton": Material(
        name="Kapton", Z=6, M=12.0,
        rho=1420.0, n_atomic=7.13e28,
        Us=2.9,
        yt_params={
            "Xe": YamamuraParameters(Q=0.30, s=2.5, Eth=35.0, Us=2.9),
        },
    ),
    # ITO coverglass coating
    "ITO": Material(
        name="ITO", Z=49, M=144.0,  # In-weighted effective values
        rho=7140.0, n_atomic=2.99e28,
        Us=2.5,
        yt_params={
            "Xe": YamamuraParameters(Q=0.95, s=2.5, Eth=25.0, Us=2.5),
        },
    ),
}


def get_material(name: str) -> Material:
    if name not in MATERIALS:
        raise KeyError(
            f"Material '{name}' not in library. Available: {list(MATERIALS)}"
        )
    return MATERIALS[name]


def get_projectile(name: str) -> Projectile:
    if name not in PROJECTILES:
        raise KeyError(
            f"Projectile '{name}' not in library. Available: {list(PROJECTILES)}"
        )
    return PROJECTILES[name]
