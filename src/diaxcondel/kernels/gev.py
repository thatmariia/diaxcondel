"""
Generalized Extreme Value distribution helpers.

A thin wrapper around scipy's GEV (`scipy.stats.genextreme`) with parameter
naming: ``mu`` (location), ``sigma`` (scale > 0), ``xi`` (shape).
Note that scipy's ``c`` parameter is the *negative* of the shape parameter
``xi`` in the convention used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from diaxcondel._typing import FloatArray


@dataclass(frozen=True, slots=True)
class GEVDiameter:
    """
    Generalized Extreme Value distribution over axon diameters.

    Parameters
    ----------
    mu : float
        Location parameter (typical units: micrometres).
    sigma : float
        Scale parameter, must be positive.
    xi : float
        Shape parameter. ``xi < 0`` (Weibull domain) gives a finite upper
        bound.
    """

    mu: float
    sigma: float
    xi: float

    def __post_init__(self) -> None:  # noqa: D105
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def pdf(self, x: FloatArray) -> FloatArray:
        """
        Evaluate the GEV PDF.

        Parameters
        ----------
        x : FloatArray
            Points at which to evaluate.

        Returns
        -------
        FloatArray
            PDF values; zero outside the support.
        """
        # scipy uses c = -xi in our convention.
        return np.asarray(stats.genextreme.pdf(x, c=-self.xi, loc=self.mu, scale=self.sigma))

    def sample(self, size: int, rng: np.random.Generator) -> FloatArray:
        """
        Draw axon diameters.

        Parameters
        ----------
        size : int
            Number of draws.
        rng : numpy.random.Generator
            Randomness source.

        Returns
        -------
        FloatArray of shape (size,)
            Sampled diameters in micrometres.
        """
        return np.asarray(
            stats.genextreme.rvs(-self.xi, loc=self.mu, scale=self.sigma, size=size, random_state=rng),
            dtype=float,
        )

    def parameters(self) -> dict[str, float]:
        """Return the distribution's free scalar parameters."""
        return {"mu": self.mu, "sigma": self.sigma, "xi": self.xi}
