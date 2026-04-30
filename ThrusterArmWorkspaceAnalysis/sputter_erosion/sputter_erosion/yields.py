"""
yields.py
=========

Sputter-yield models Y(E, theta; ion -> target).

Implements:
  * Yamamura-Tawara (1996) — the EP-community workhorse, weak near threshold.
  * Eckstein-Preuss (2003) — improved near-threshold behaviour, recommended
    for the CEX-dominated regime that drives interconnect erosion.
  * Seah (2005, with NPL revisions) — analytic Q(target), useful when the
    target is not in the Y-T tables.

Angular dependence:
  * YamamuraAngular  — original f / theta_opt form (1981).
  * EcksteinAngular  — Garcia-Rosales/Eckstein revised form, valid for heavy
                       projectiles and self-sputtering.

All yields returned as (atoms / incident ion). Incidence angle theta is measured
from the surface normal, in radians.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from .materials import Material, Projectile


# ---------------------------------------------------------------------------
# Reduced-energy and stopping-power utilities
# ---------------------------------------------------------------------------

def lindhard_reduced_energy(E_eV: float, projectile: Projectile, target: Material) -> float:
    """
    Lindhard reduced energy epsilon (dimensionless), used by both Y-T and
    Eckstein fits. Standard ZBL screening.

        a_L = 0.4685 * (Z1^(2/3) + Z2^(2/3))^(-1/2)   [Angstrom]
        eps = a_L * M2 * E / (Z1 * Z2 * (M1+M2) * (Z1^(2/3)+Z2^(2/3))^(1/2)) * 1/14.4
    """
    Z1, Z2 = projectile.Z, target.Z
    M1, M2 = projectile.M, target.M
    return (
        0.03255 * M2 * E_eV
        / ((M1 + M2) * Z1 * Z2 * np.sqrt(Z1 ** (2.0 / 3.0) + Z2 ** (2.0 / 3.0)))
    )


def sn_KrC(eps: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Reduced nuclear stopping cross-section, Kr-C potential
    (Wilson, Haggmark, Biersack 1977 fit).
    """
    eps = np.asarray(eps, dtype=float)
    return (
        0.5 * np.log(1.0 + 1.2288 * eps)
        / (eps + 0.1728 * np.sqrt(eps) + 0.008 * eps ** 0.1504)
    )


def sn_TF(eps: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Reduced nuclear stopping cross-section, Thomas-Fermi potential."""
    eps = np.asarray(eps, dtype=float)
    return 3.441 * np.sqrt(eps) * np.log(eps + 2.718) / (
        1.0 + 6.355 * np.sqrt(eps) + eps * (6.882 * np.sqrt(eps) - 1.708)
    )


# ---------------------------------------------------------------------------
# Energy-dependence models (normal incidence)
# ---------------------------------------------------------------------------

class YieldModel(ABC):
    """Abstract base class for energy-dependent sputter-yield models."""

    name: str = "abstract"

    @abstractmethod
    def yield_normal(
        self,
        E_eV: Union[float, np.ndarray],
        projectile: Projectile,
        target: Material,
    ) -> Union[float, np.ndarray]:
        """Sputter yield at normal incidence [atoms/ion]."""
        ...

    def __call__(self, E_eV, projectile, target):
        return self.yield_normal(E_eV, projectile, target)


class YamamuraTawara(YieldModel):
    """
    Yamamura-Tawara (1996) empirical fit:

        Y(E) = 0.42 * (alpha* Q K sn(eps)) / (Us (1 + 0.35 Us se(eps)))
               * (1 - sqrt(Eth/E))**s

    The se(eps) electronic-stopping correction is small for heavy-on-metal at
    EP energies and is dropped in the standard EP simplification (see Goebel &
    Katz, Fundamentals of EP). Set `include_electronic=True` to retain it.
    """

    name = "Yamamura-Tawara (1996)"

    def __init__(self, include_electronic: bool = False):
        self.include_electronic = include_electronic

    @staticmethod
    def _alpha_star(M2_over_M1: float) -> float:
        """Yamamura-Tawara alpha* function (energy-transfer factor)."""
        if M2_over_M1 < 0.5:
            return 0.2
        # canonical Y-T form
        return 0.10 + 0.155 * M2_over_M1 ** 0.73

    def yield_normal(self, E_eV, projectile, target):
        E = np.asarray(E_eV, dtype=float)
        params = target.yt_params.get(projectile.name)
        if params is None:
            raise KeyError(
                f"No Y-T parameters for {projectile.name} -> {target.name}"
            )
        Q, s, Eth, Us = params.Q, params.s, params.Eth, params.Us

        eps = lindhard_reduced_energy(E, projectile, target)
        sn = sn_KrC(eps)
        M2_over_M1 = target.M / projectile.M
        alpha_star = self._alpha_star(M2_over_M1)

        # K is the conversion from reduced to absolute units; set so the
        # fit reproduces the published yields. EP-community standard:
        K = 8.478 * projectile.Z * target.Z / (
            np.sqrt(projectile.Z ** (2.0 / 3.0) + target.Z ** (2.0 / 3.0))
            * (projectile.M + target.M)
        )

        prefactor = 0.42 * alpha_star * Q * K * sn / Us

        # threshold factor — the part that misbehaves near Eth
        with np.errstate(invalid="ignore"):
            ratio = np.where(E > Eth, 1.0 - np.sqrt(Eth / E), 0.0)
        threshold = np.power(ratio, s)

        Y = prefactor * threshold
        return np.where(E > Eth, Y, 0.0)


class EcksteinPreuss(YieldModel):
    """
    Eckstein-Preuss (2003) revised fit, J. Nucl. Mater. 320, 209.

        Y(E) = q * sn_KrC(eps) * (E/Eth - 1)**mu /
               (lam / w(eps) + (E/Eth - 1)**mu)

    where w(eps) ~ eps + 0.1728*sqrt(eps) + 0.008*eps^0.1504  (the denominator
    of sn_KrC); in practice the lam parameter absorbs the exact choice of w.

    Recommended for low-energy / near-threshold work, i.e. for the CEX-dominated
    flux on solar interconnects.
    """

    name = "Eckstein-Preuss (2003)"

    def yield_normal(self, E_eV, projectile, target):
        E = np.asarray(E_eV, dtype=float)
        params = target.eckstein_params.get(projectile.name)
        if params is None:
            raise KeyError(
                f"No Eckstein parameters for {projectile.name} -> {target.name}"
            )
        q, lam, mu, Eth = params.q, params.lam, params.mu, params.Eth

        eps = lindhard_reduced_energy(E, projectile, target)
        sn = sn_KrC(eps)

        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.where(E > Eth, E / Eth - 1.0, 0.0)
            w = eps + 0.1728 * np.sqrt(eps) + 0.008 * eps ** 0.1504
            num = q * sn * np.power(x, mu)
            den = lam / np.where(w > 0, w, 1.0) + np.power(x, mu)
            Y = np.where(den > 0, num / den, 0.0)
        return np.where(E > Eth, Y, 0.0)


class Seah2005(YieldModel):
    """
    Seah (2005) accurate semi-empirical equation, Surf. Interface Anal. 37, 444.

    Implements the 'general' analytic form that derives Q from tabulated
    physical data (atomic density, sublimation energy) rather than a per-target
    Q fit. Useful when the target is not in the Y-T or Eckstein parameter sets.

    For Xe the Seah Q expression is:
        Q_Xe = (Us / 7.0)**(-0.65) * (Z2 / 30.0)**0.4 * (n_at / 6e28)**0.25

    (these coefficients are an EP-relevant simplification of Seah's full
    Z2 expansion; see NPL tables for the official numbers).
    """

    name = "Seah (2005, simplified)"

    def yield_normal(self, E_eV, projectile, target):
        E = np.asarray(E_eV, dtype=float)
        # Seah Q estimator
        Q_seah = (
            (target.Us / 7.0) ** (-0.65)
            * (target.Z / 30.0) ** 0.4
            * (target.n_atomic / 6.0e28) ** 0.25
        )
        # threshold from Bohdansky-style estimate
        gamma = 4.0 * projectile.M * target.M / (projectile.M + target.M) ** 2
        if gamma > 0.3:
            Eth = target.Us / (gamma * (1.0 - gamma))
        else:
            Eth = 8.0 * target.Us * (projectile.M / target.M) ** (2.0 / 5.0)

        eps = lindhard_reduced_energy(E, projectile, target)
        sn = sn_KrC(eps)

        with np.errstate(invalid="ignore"):
            ratio = np.where(E > Eth, 1.0 - np.sqrt(Eth / E), 0.0)
        # Matsunami-style exponent ~2.5
        Y = 0.42 * Q_seah * sn * np.power(ratio, 2.5) / target.Us
        return np.where(E > Eth, Y, 0.0)


# ---------------------------------------------------------------------------
# Angular dependence
# ---------------------------------------------------------------------------

class AngularDependence(ABC):
    """Abstract: f_theta = Y(E, theta) / Y(E, 0)."""

    name: str = "abstract"

    @abstractmethod
    def factor(
        self,
        theta_rad: Union[float, np.ndarray],
        E_eV: float,
        projectile: Projectile,
        target: Material,
    ) -> Union[float, np.ndarray]:
        ...


class YamamuraAngular(AngularDependence):
    """
    Yamamura (1981) angular factor:

        Y(theta)/Y(0) = (cos theta)**(-f) * exp[-f * (1/cos(theta_opt) - 1/cos theta) * cos(theta_opt)]

    Valid for light-to-medium projectiles. For Xe -> Ag the mass ratio is
    favorable and this form is acceptable up to ~70 deg. Beyond that, switch
    to EcksteinAngular.

    f and theta_opt are typically ~1.5 and ~60-70 deg for Xe->metal at EP energies.
    """

    name = "Yamamura (1981)"

    def __init__(self, f: float = 1.7, theta_opt_deg: float = 65.0):
        self.f = f
        self.theta_opt = np.deg2rad(theta_opt_deg)

    def factor(self, theta_rad, E_eV, projectile, target):
        theta = np.asarray(theta_rad, dtype=float)
        # clamp to avoid 1/cos blowup
        theta = np.clip(theta, 0.0, np.deg2rad(85.0))
        cos_t = np.cos(theta)
        cos_topt = np.cos(self.theta_opt)
        # Y-T form
        return (
            np.power(cos_t, -self.f)
            * np.exp(-self.f * cos_topt * (1.0 / cos_topt - 1.0 / cos_t))
        )


class EcksteinAngular(AngularDependence):
    """
    Garcia-Rosales / Eckstein (1994) revised angular dependence:

        Y(theta)/Y(0) = (cos(theta_star))**(-c) *
                        exp[b * (1 - 1/cos(theta_star)) * cos(theta_opt)]

    where theta_star = pi/2 * (theta - theta_0) / (pi/2 - theta_0).

    Better behaviour at large theta and for heavy projectiles / low mass
    ratios, and consistent with the Eckstein energy fit near threshold.
    """

    name = "Eckstein (1994)"

    def __init__(self, b: float = 1.8, c: float = 1.5,
                 theta_opt_deg: float = 65.0, theta_0_deg: float = 0.0):
        self.b = b
        self.c = c
        self.theta_opt = np.deg2rad(theta_opt_deg)
        self.theta_0 = np.deg2rad(theta_0_deg)

    def factor(self, theta_rad, E_eV, projectile, target):
        theta = np.asarray(theta_rad, dtype=float)
        theta = np.clip(theta, self.theta_0, np.deg2rad(85.0))

        denom = np.pi / 2.0 - self.theta_0
        theta_star = (np.pi / 2.0) * (theta - self.theta_0) / denom
        cos_ts = np.cos(theta_star)
        cos_topt = np.cos(self.theta_opt)

        return (
            np.power(np.clip(cos_ts, 1e-3, 1.0), -self.c)
            * np.exp(self.b * cos_topt * (1.0 - 1.0 / np.clip(cos_ts, 1e-3, 1.0)))
        )


# ---------------------------------------------------------------------------
# Composed (E, theta) yield
# ---------------------------------------------------------------------------

@dataclass
class FullYield:
    """
    Combines an energy-dependence model with an angular-dependence model.

    Y(E, theta) = Y_normal(E) * f_theta(theta, E)

    Sub-threshold floor:
      Several EP measurements (Mantenieks 2001, Doerner) report finite yields
      below the nominal Eth, attributed to surface roughness / tail effects.
      `subthreshold_floor` adds a small constant yield (atoms/ion) for E in
      [Eth_floor, Eth] to bracket this effect; default 0 (off).
    """

    energy_model: YieldModel
    angular_model: AngularDependence
    subthreshold_floor: float = 0.0   # atoms/ion for sub-threshold tail
    Eth_floor_frac: float = 0.5       # floor extends from frac*Eth up to Eth

    def __call__(self, E_eV, theta_rad, projectile, target):
        Y0 = self.energy_model.yield_normal(E_eV, projectile, target)
        f = self.angular_model.factor(theta_rad, E_eV, projectile, target)
        Y = Y0 * f

        if self.subthreshold_floor > 0.0:
            # add a smooth floor in the sub-threshold tail
            params = target.eckstein_params.get(projectile.name) \
                  or target.yt_params.get(projectile.name)
            if params is not None:
                Eth = params.Eth
                E = np.asarray(E_eV, dtype=float)
                in_tail = (E < Eth) & (E > self.Eth_floor_frac * Eth)
                tail = np.where(
                    in_tail,
                    self.subthreshold_floor
                    * (E - self.Eth_floor_frac * Eth)
                    / (Eth - self.Eth_floor_frac * Eth),
                    0.0,
                )
                Y = Y + tail * f
        return Y
