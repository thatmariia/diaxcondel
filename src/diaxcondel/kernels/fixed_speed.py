"""
Fixed-conduction-speed delay kernel.

The inverse-GEV kernel spreads delays because axon diameters within a bundle
vary. This kernel is the control condition for that assumption: every axon
conducts at the same speed, so a connection of length ``d`` has a single
delay ``d / v``, optionally smeared by a Gaussian jitter to keep the discrete
kernel from falling between samples.

It is deliberately simple — one speed, one jitter — and useful for asking
which spectral features come from the *spread* of delays rather than from
their typical value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diaxcondel._typing import FloatArray


@dataclass(frozen=True, slots=True)
class FixedSpeedKernel:
    """
    Delay kernel for a single conduction speed.

    Parameters
    ----------
    speed_m_s : float
        Conduction speed in metres per second. Typical myelinated
        cortico-cortical values lie between 1 and 15 m/s.
    jitter_ms : float
        Standard deviation of a Gaussian spread around the nominal delay, in
        milliseconds. Must be positive; a small value (a few tenths of a
        millisecond) keeps the discretised kernel well behaved.

    Notes
    -----
    The density is Gaussian in delay, truncated to positive delays and
    renormalised on the requested grid, so the discrete kernel always sums to
    (approximately) one within the lag horizon.
    """

    speed_m_s: float = 6.0
    jitter_ms: float = 1.0

    def __post_init__(self) -> None:  # noqa: D105
        if self.speed_m_s <= 0:
            raise ValueError("speed_m_s must be positive")
        if self.jitter_ms <= 0:
            raise ValueError("jitter_ms must be positive")

    def _pdf(self, delays_s: FloatArray, distance_cm: float) -> FloatArray:
        """Evaluate the Gaussian delay density for one connection length."""
        delays_s = np.asarray(delays_s, dtype=float)
        if np.any(delays_s <= 0):
            raise ValueError("delays must be strictly positive")
        if distance_cm < 0:
            raise ValueError("distance_cm must be non-negative")
        if distance_cm == 0:
            return np.zeros_like(delays_s)
        mean_s = (distance_cm * 1e-2) / self.speed_m_s
        sigma_s = self.jitter_ms * 1e-3
        density = np.exp(-0.5 * ((delays_s - mean_s) / sigma_s) ** 2) / (sigma_s * np.sqrt(2 * np.pi))
        return np.where(np.isfinite(density), density, 0.0)

    def pdf(self, delays_s: FloatArray, distance_cm: float) -> FloatArray:
        """
        Evaluate the delay density on a grid for one connection length.

        Parameters
        ----------
        delays_s : FloatArray of shape (P,)
            Delay grid in seconds.
        distance_cm : float
            Connection length in centimetres.

        Returns
        -------
        FloatArray of shape (P,)
            Density values.
        """
        return self._pdf(delays_s, distance_cm)

    def pdf_batched(self, delays_s: FloatArray, distances_cm: FloatArray) -> FloatArray:
        """
        Evaluate delay densities for many connection lengths at once.

        Parameters
        ----------
        delays_s : FloatArray of shape (P,)
            Delay grid in seconds.
        distances_cm : FloatArray of shape (M,)
            Connection lengths in centimetres.

        Returns
        -------
        FloatArray of shape (M, P)
            Density values per connection.
        """
        delays = np.asarray(delays_s, dtype=float)[None, :]
        distances = np.asarray(distances_cm, dtype=float).reshape(-1, 1)
        if np.any(delays <= 0):
            raise ValueError("delays must be strictly positive")
        if np.any(distances < 0):
            raise ValueError("distances_cm must be non-negative")
        mean_s = (distances * 1e-2) / self.speed_m_s
        sigma_s = self.jitter_ms * 1e-3
        density = np.exp(-0.5 * ((delays - mean_s) / sigma_s) ** 2) / (sigma_s * np.sqrt(2 * np.pi))
        return np.where(np.isfinite(density), density, 0.0)

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {"speed_m_s": self.speed_m_s, "jitter_ms": self.jitter_ms}
