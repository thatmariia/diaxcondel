"""
Synthetic distance distributions for theoretical investigations.

These providers do not correspond to any anatomical atlas; they exist to
test predictions that link the spectral exponent of the simulated PSD to
the connection-length distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from diaxcondel._rng import RNGLike, as_generator
from diaxcondel._typing import FloatArray

from ..regions import RegionSet


@dataclass(frozen=True)
class PowerLawDistances:
    r"""
    Random-graph distances drawn from ``p(d) ∝ d^beta`` on ``[d_min, d_max]``.

    Parameters
    ----------
    n_regions : int
        Number of regions to generate.
    beta : float
        Power-law exponent. ``beta = -1`` gives a log-uniform distribution
        (predicting :math:`S(f) \\propto f^{-1}`), ``beta = 0`` gives uniform
        (predicting :math:`S(f) \\propto f^{-2}`).
    d_min, d_max : float
        Distance bounds in centimetres. Must satisfy ``0 < d_min < d_max``.
    rng : RNGLike, optional
        Seed or generator. ``None`` uses a fresh OS-seeded generator.

    Notes
    -----
    Constructs a fully connected, symmetric distance matrix with zero
    diagonal. Distances on each off-diagonal entry are drawn i.i.d. from
    the specified power law.
    """

    n_regions: int
    beta: float
    d_min: float
    d_max: float
    rng: RNGLike = None

    def __post_init__(self) -> None:  # noqa: D105
        if self.n_regions < 2:
            raise ValueError("n_regions must be >= 2")
        if not (0 < self.d_min < self.d_max):
            raise ValueError("require 0 < d_min < d_max")

    @cached_property
    def regions(self) -> RegionSet:  # noqa: D102
        return RegionSet.from_codes([f"R{i:04d}" for i in range(self.n_regions)])

    def matrix(self) -> FloatArray:  # noqa: D102
        return self._matrix

    @cached_property
    def _matrix(self) -> FloatArray:
        """Return a deterministic cached distance matrix."""
        gen = as_generator(self.rng)
        n = self.n_regions
        n_pairs = n * (n - 1) // 2
        samples = self._sample_power_law(n_pairs, gen)
        out = np.zeros((n, n))
        idx = np.triu_indices(n, k=1)
        out[idx] = samples
        out = out + out.T
        out.setflags(write=False)
        return out

    def _sample_power_law(self, size: int, gen: np.random.Generator) -> FloatArray:
        """Sample from ``p(d) ∝ d^beta`` on ``[d_min, d_max]`` by inverse CDF."""
        u = gen.random(size)
        if np.isclose(self.beta, -1.0):
            # log-uniform: F(d) = log(d/d_min) / log(d_max/d_min)
            return self.d_min * (self.d_max / self.d_min) ** u
        b = self.beta + 1.0
        return ((self.d_max**b - self.d_min**b) * u + self.d_min**b) ** (1 / b)
