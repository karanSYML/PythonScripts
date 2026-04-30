"""
monte_carlo.py
==============

Bayesian / parametric uncertainty propagation through the erosion integral,
in the spirit of Zameshin & Sturm (2022) for the Y-T model parameters and
extending naturally to Eckstein (q, lam, mu, Eth).

This is the V&V layer: rather than carrying ad-hoc factor-of-two margins,
we propagate posterior distributions of the dominant fit parameters through
the full geometric / plume / yield pipeline and report distributions of
interconnect lifetime.

Mirrors the MC structure of `geo_rpo`: a sampler builds N realisations of
material parameter sets, the existing ErosionIntegrator is called on each,
and statistics are aggregated.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import copy
import numpy as np

from .materials import Material, BayesianPosterior, EcksteinParameters, YamamuraParameters
from .geometry import SatelliteGeometry
from .erosion import ErosionIntegrator, ErosionResult


@dataclass
class ParameterPosterior:
    """
    Per-(material, projectile) posterior over the dominant fit parameters.
    Defaults to independent Gaussians on Q, s, Eth (Zameshin & Sturm 2022
    showed Q-Eth correlations are non-trivial; an optional rho_Q_Eth argument
    introduces a Gaussian copula here).
    """
    posterior: BayesianPosterior
    rho_Q_Eth: float = -0.4   # mild negative correlation (typical MCMC fit)

    def sample(self, rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
        """Draw n correlated samples of (Q, s, Eth)."""
        # Q, Eth correlated; s independent
        cov = np.array([
            [1.0, self.rho_Q_Eth],
            [self.rho_Q_Eth, 1.0],
        ])
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((2, n))
        zc = L @ z
        Q = self.posterior.Q_mean + zc[0] * self.posterior.Q_std
        Eth = self.posterior.Eth_mean + zc[1] * self.posterior.Eth_std
        s = self.posterior.s_mean + rng.standard_normal(n) * self.posterior.s_std

        # Truncate to physical (Q > 0, Eth > 0, s > 0.5)
        Q = np.clip(Q, 0.05, None)
        Eth = np.clip(Eth, 1.0, None)
        s = np.clip(s, 0.5, None)
        return {"Q": Q, "s": s, "Eth": Eth}


@dataclass
class MonteCarloErosion:
    """
    Monte Carlo wrapper around an ErosionIntegrator.

    Posteriors is a dict keyed by material name -> ParameterPosterior. For each
    sample the corresponding Material entry's yt_params and eckstein_params are
    perturbed accordingly.
    """
    integrator: ErosionIntegrator
    posteriors: Dict[str, ParameterPosterior]
    n_samples: int = 500
    seed: int = 42

    def _apply_sample(self, material: Material, projectile: str,
                      Q: float, s: float, Eth: float):
        """Mutate material in place to use the sampled parameters."""
        if material.has_yt(projectile):
            old = material.yt_params[projectile]
            material.yt_params[projectile] = YamamuraParameters(
                Q=Q, s=s, Eth=Eth, Us=old.Us
            )
        if material.has_eckstein(projectile):
            old = material.eckstein_params[projectile]
            # Eckstein doesn't use Q/s directly; map Q -> q (linear scaling)
            # and inherit lam, mu, but update Eth.
            q_new = old.q * (Q / max(self.posteriors[material.name].posterior.Q_mean, 1e-6))
            material.eckstein_params[projectile] = EcksteinParameters(
                q=q_new, lam=old.lam, mu=old.mu, Eth=Eth,
            )

    def run(self, geometry: SatelliteGeometry,
            firing_duration_s: float,
            projectile: str = "Xe") -> Dict:
        """
        Returns a dict with:
          'samples_thinning' : (N, n_interconnects) array of total thinning
          'mean_thinning'    : per-interconnect mean
          'p5', 'p50', 'p95' : per-interconnect percentiles
          'keys'             : list of (array_idx, ic_idx) keys for the columns
        """
        rng = np.random.default_rng(self.seed)

        # Collect baseline materials present on any interconnect
        materials_in_use: Dict[str, Material] = {}
        for arr in geometry.solar_arrays:
            for ic in arr.interconnects:
                m = ic.material()
                materials_in_use.setdefault(m.name, m)

        # Pre-sample all parameter realisations
        samples = {
            name: self.posteriors[name].sample(rng, self.n_samples)
            for name in materials_in_use
            if name in self.posteriors
        }

        # Establish column ordering on first run
        baseline = self.integrator.total_thinning_per_interconnect(
            geometry, firing_duration_s
        )
        keys = sorted(baseline.keys())
        out = np.zeros((self.n_samples, len(keys)))

        # Snapshot original parameters so we can restore them
        snapshot = {
            name: (
                copy.deepcopy(materials_in_use[name].yt_params),
                copy.deepcopy(materials_in_use[name].eckstein_params),
            )
            for name in materials_in_use
        }

        try:
            for k in range(self.n_samples):
                for name, mat in materials_in_use.items():
                    if name not in samples:
                        continue
                    self._apply_sample(
                        mat, projectile,
                        Q=samples[name]["Q"][k],
                        s=samples[name]["s"][k],
                        Eth=samples[name]["Eth"][k],
                    )
                run = self.integrator.total_thinning_per_interconnect(
                    geometry, firing_duration_s
                )
                for j, key in enumerate(keys):
                    out[k, j] = run.get(key, 0.0)
        finally:
            # Restore original parameters
            for name, (yt, ek) in snapshot.items():
                materials_in_use[name].yt_params = yt
                materials_in_use[name].eckstein_params = ek

        return {
            "samples_thinning": out,
            "mean_thinning": out.mean(axis=0),
            "p5": np.percentile(out, 5, axis=0),
            "p50": np.percentile(out, 50, axis=0),
            "p95": np.percentile(out, 95, axis=0),
            "keys": keys,
        }
