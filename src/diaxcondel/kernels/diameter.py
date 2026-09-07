r"""
Axon-diameter distributions and the delay kernels they induce.

Conduction speed grows with axon calibre (Hursh, 1939), so a bundle whose
axons differ in diameter delivers one signal over a *range* of delays. Which
range depends on how diameters are distributed, and the literature fits
several families to the same histological data — Sepehrband et al. (2016)
compare generalised extreme value, gamma, log-normal and Rayleigh fits of
corpus-callosum diameters.

Every family here plugs into the same change of variables. For a diameter
density :math:`q_\\phi` and speed :math:`v = k\\phi`, the delay over a length
:math:`d` is :math:`\\tau = \\lambda / \\phi` with :math:`\\lambda = d / (100k)`
in units of s·µm, so

.. math::

    q_\\tau(t) = \\frac{\\lambda}{t^2} \\, q_\\phi\\!\\left(\\frac{\\lambda}{t}\\right).

Choosing a family therefore changes the *shape* of the delay distribution —
how sharply peaked it is, how heavy its slow tail — while the peak stays tied
to the most common diameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from scipy import stats

from diaxcondel._rng import RNGLike, as_generator
from diaxcondel._typing import FloatArray


@runtime_checkable
class DiameterDistribution(Protocol):
    """Protocol for a distribution over axon diameters in micrometres."""

    def pdf(self, x: FloatArray) -> FloatArray:
        """
        Evaluate the diameter density.

        Parameters
        ----------
        x : FloatArray
            Diameters in micrometres.

        Returns
        -------
        FloatArray
            Density values; zero outside the support.
        """
        ...

    def sample(self, size: int, rng: np.random.Generator) -> FloatArray:
        """
        Draw diameters, for constructions with no closed form.

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
        ...

    def parameters(self) -> dict[str, float]:
        """Return the distribution's free scalar parameters."""
        ...


def positive_diameters(
    distribution: DiameterDistribution,
    size: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = 20,
) -> FloatArray:
    """
    Draw physically possible axon calibres.

    Some fitted calibre distributions — the generalised extreme value among
    them — put a little probability below zero. A negative diameter has no
    meaning, so draws are filtered and topped up until enough positive ones
    are available. The same truncation is implicit in the delay density,
    which is only ever evaluated at positive calibres.

    Parameters
    ----------
    distribution : DiameterDistribution
        Calibre distribution to sample.
    size : int
        Number of positive draws required.
    rng : numpy.random.Generator
        Randomness source.
    max_attempts : int
        How many times to top up before giving up.

    Returns
    -------
    FloatArray of shape (size,)
        Strictly positive calibres in micrometres.

    Raises
    ------
    RuntimeError
        If the distribution almost never produces a positive calibre.
    """
    collected: list[FloatArray] = []
    total = 0
    for _ in range(max_attempts):
        draws = np.asarray(distribution.sample(max(size, 128), rng), dtype=float)
        positive = draws[draws > 0]
        if positive.size:
            collected.append(positive)
            total += positive.size
        if total >= size:
            return np.concatenate(collected)[:size]
    raise RuntimeError("the calibre distribution almost never produces a positive diameter; check its parameters")


@dataclass(frozen=True, slots=True)
class GammaDiameter:
    """
    Gamma-distributed axon diameters.

    A right-skewed family with a light tail and no upper bound, and the shape
    most often used when a single positive-valued distribution is wanted.

    Parameters
    ----------
    mean_um : float
        Mean diameter in micrometres.
    shape : float
        Gamma shape parameter; the spread is ``mean_um / sqrt(shape)``, so
        large values approach a single calibre.
    """

    mean_um: float = 0.3
    shape: float = 4.0

    def __post_init__(self) -> None:  # noqa: D105
        if self.mean_um <= 0:
            raise ValueError("mean_um must be positive")
        if self.shape <= 0:
            raise ValueError("shape must be positive")

    def pdf(self, x: FloatArray) -> FloatArray:  # noqa: D102
        return np.asarray(stats.gamma.pdf(x, a=self.shape, scale=self.mean_um / self.shape))

    def sample(self, size: int, rng: np.random.Generator) -> FloatArray:  # noqa: D102
        return np.asarray(
            stats.gamma.rvs(a=self.shape, scale=self.mean_um / self.shape, size=size, random_state=rng),
            dtype=float,
        )

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {"mean_um": self.mean_um, "shape": self.shape}


@dataclass(frozen=True, slots=True)
class LognormalDiameter:
    """
    Log-normally distributed axon diameters.

    The heaviest-tailed of the families here: a small fraction of very thick,
    very fast axons coexists with a bulk of thin ones, which shows up as a
    sharp early edge in the delay distribution.

    Parameters
    ----------
    median_um : float
        Median diameter in micrometres.
    sigma_log : float
        Standard deviation of ``log(diameter)``; larger values spread the
        calibres over more orders of magnitude.
    """

    median_um: float = 0.3
    sigma_log: float = 0.5

    def __post_init__(self) -> None:  # noqa: D105
        if self.median_um <= 0:
            raise ValueError("median_um must be positive")
        if self.sigma_log <= 0:
            raise ValueError("sigma_log must be positive")

    def pdf(self, x: FloatArray) -> FloatArray:  # noqa: D102
        return np.asarray(stats.lognorm.pdf(x, s=self.sigma_log, scale=self.median_um))

    def sample(self, size: int, rng: np.random.Generator) -> FloatArray:  # noqa: D102
        return np.asarray(
            stats.lognorm.rvs(s=self.sigma_log, scale=self.median_um, size=size, random_state=rng),
            dtype=float,
        )

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {"median_um": self.median_um, "sigma_log": self.sigma_log}


@dataclass(frozen=True, slots=True)
class RayleighDiameter:
    """
    Rayleigh-distributed axon diameters.

    A one-parameter family, included because it is among those fitted to
    histological diameter data; its shape is fixed, so it is the strictest
    of the alternatives.

    Parameters
    ----------
    scale_um : float
        Rayleigh scale in micrometres; the mode sits at this value and the
        mean at ``scale_um * sqrt(pi / 2)``.
    """

    scale_um: float = 0.25

    def __post_init__(self) -> None:  # noqa: D105
        if self.scale_um <= 0:
            raise ValueError("scale_um must be positive")

    def pdf(self, x: FloatArray) -> FloatArray:  # noqa: D102
        return np.asarray(stats.rayleigh.pdf(x, scale=self.scale_um))

    def sample(self, size: int, rng: np.random.Generator) -> FloatArray:  # noqa: D102
        return np.asarray(stats.rayleigh.rvs(scale=self.scale_um, size=size, random_state=rng), dtype=float)

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {"scale_um": self.scale_um}


@dataclass(frozen=True, slots=True)
class DiameterDelayKernel:
    """
    Delay kernel induced by any diameter distribution.

    Parameters
    ----------
    diameter : DiameterDistribution
        Distribution over axon diameters in micrometres.
    speed_factor : float
        Conduction speed per micrometre of diameter, in metres per second
        (Hursh's relation ``v = speed_factor * diameter``).

    Notes
    -----
    Calibre distributions fitted to histology can put a little probability
    below zero; those draws are impossible, so the delay density is only ever
    evaluated at positive calibres. For the reference parameters that leaves
    about a thousandth of the distribution unused, which shows up as a delay
    kernel whose discrete mass falls just short of one.
    """

    diameter: DiameterDistribution
    speed_factor: float = 6.0

    def __post_init__(self) -> None:  # noqa: D105
        if self.speed_factor <= 0:
            raise ValueError("speed_factor must be positive")

    def pdf(self, delays_s: FloatArray, distance_cm: float) -> FloatArray:
        """
        Evaluate the delay density for one connection length.

        Parameters
        ----------
        delays_s : FloatArray of shape (P,)
            Delay grid in seconds; must be strictly positive.
        distance_cm : float
            Connection length in centimetres.

        Returns
        -------
        FloatArray of shape (P,)
            Density values.
        """
        return self.pdf_batched(delays_s, np.asarray([distance_cm], dtype=float))[0]

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
        lam = (distances * 1e-2) / self.speed_factor  # (M, 1), units s*um
        phi = lam / delays
        jacobian = lam / np.square(delays)
        density = jacobian * self.diameter.pdf(phi)
        return np.where(np.isfinite(density), density, 0.0)

    def relay_kernel(self, *, n_samples: int = 5000, rng: RNGLike = 20260505) -> TwoLegDelayKernel:
        """
        Return the two-leg counterpart of this kernel, for relayed paths.

        A relayed connection accumulates the delay of both legs, and the sum
        of two inverse-diameter delays has no closed form. The counterpart
        keeps this kernel's diameter distribution and conduction speed and
        evaluates the sum by simulation, so the two variants describe the
        same axons.

        Parameters
        ----------
        n_samples : int
            Monte-Carlo sample count for the two-leg delay density.
        rng : RNGLike
            Seed for that step; fixed by default so builds are reproducible.

        Returns
        -------
        TwoLegDelayKernel
            Kernel for relayed connections.
        """
        return TwoLegDelayKernel(
            diameter=self.diameter,
            speed_factor=self.speed_factor,
            n_samples=n_samples,
            rng=rng,
        )

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {**self.diameter.parameters(), "speed_factor": self.speed_factor}


@dataclass(frozen=True, slots=True)
class TwoLegDelayKernel:
    """
    Delay kernel for connections relayed through an intermediate region.

    A relayed path is two axonal legs in series, each with its own diameter
    drawn from the same distribution, so its delay is the sum of two
    inverse-diameter delays. That sum has no closed form; it is evaluated by
    sampling both legs and smoothing the result with a Gaussian kernel
    density estimate.

    Parameters
    ----------
    diameter : DiameterDistribution
        Diameter distribution shared by both legs.
    speed_factor : float
        Conduction speed per micrometre of diameter, in metres per second.
    n_samples : int
        Number of Monte-Carlo samples per leg.
    bandwidth : str or float
        Kernel-density bandwidth, passed to :class:`scipy.stats.gaussian_kde`.
    rng : RNGLike
        Seed for the sampling step. Fix it for reproducible model building.

    Notes
    -----
    Densities are cached per pair of leg lengths on the instance, so a model
    with many relayed connections of similar length pays the sampling cost
    once per distinct pair.
    """

    diameter: DiameterDistribution
    speed_factor: float = 6.0
    n_samples: int = 5000
    bandwidth: str | float = "scott"
    rng: RNGLike = None
    _kde_cache: dict[tuple[float, float], stats.gaussian_kde] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:  # noqa: D105
        if self.speed_factor <= 0:
            raise ValueError("speed_factor must be positive")
        if self.n_samples < 100:
            raise ValueError("n_samples must be >= 100")
        if isinstance(self.bandwidth, (int, float)) and self.bandwidth <= 0:
            raise ValueError("numeric bandwidth must be positive")

    def _direct(self) -> DiameterDelayKernel:
        """Return the single-leg kernel with the same axons and speed."""
        return DiameterDelayKernel(diameter=self.diameter, speed_factor=self.speed_factor)

    def pdf(self, delays_s: FloatArray, distance_cm: float) -> FloatArray:
        """
        Evaluate the single-leg delay density, for connections to the relay itself.

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
        return self._direct().pdf(delays_s, distance_cm)

    def pdf_batched(self, delays_s: FloatArray, distances_cm: FloatArray) -> FloatArray:
        """
        Evaluate single-leg delay densities for many connection lengths.

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
        return self._direct().pdf_batched(delays_s, distances_cm)

    def pdf_two_leg(self, delays_s: FloatArray, leg1_cm: float, leg2_cm: float) -> FloatArray:
        """
        Evaluate the delay density of a two-leg path.

        Parameters
        ----------
        delays_s : FloatArray of shape (P,)
            Delay grid in seconds.
        leg1_cm, leg2_cm : float
            Length of each leg in centimetres.

        Returns
        -------
        FloatArray of shape (P,)
            Density values, clipped at zero to suppress estimation artefacts.
        """
        if leg1_cm < 0 or leg2_cm < 0:
            raise ValueError("leg distances must be non-negative")
        if leg1_cm == 0:
            return self.pdf(delays_s, leg2_cm)
        if leg2_cm == 0:
            return self.pdf(delays_s, leg1_cm)
        first, second = round(leg1_cm, 6), round(leg2_cm, 6)
        key = (min(first, second), max(first, second))
        kde = self._kde_cache.get(key)
        if kde is None:
            kde = self._build_kde(*key)
            self._kde_cache[key] = kde
        return np.clip(np.asarray(kde(delays_s)), 0.0, None)

    def _build_kde(self, leg1_cm: float, leg2_cm: float) -> stats.gaussian_kde:
        """Sample both legs and smooth the distribution of their summed delay."""
        gen = as_generator(self.rng)
        first = positive_diameters(self.diameter, self.n_samples, gen)
        second = positive_diameters(self.diameter, self.n_samples, gen)
        delay_1 = (leg1_cm * 1e-2) / (self.speed_factor * first)
        delay_2 = (leg2_cm * 1e-2) / (self.speed_factor * second)
        return stats.gaussian_kde(delay_1 + delay_2, bw_method=self.bandwidth)

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {**self.diameter.parameters(), "speed_factor": self.speed_factor}


@dataclass(frozen=True, slots=True)
class GammaDelayKernel:
    """
    Delay kernel specified directly in delay space, as a gamma distribution.

    Neural-field models often place a gamma (or alpha-function) spread on the
    delay itself rather than deriving it from axon calibres. This kernel does
    that: the mean delay is length over speed, and the shape parameter sets
    how tightly delays cluster around it.

    Parameters
    ----------
    speed_m_s : float
        Mean conduction speed in metres per second.
    shape : float
        Gamma shape parameter; the spread is ``mean / sqrt(shape)``, so large
        values approach a fixed delay and ``1`` gives an exponential spread.

    Notes
    -----
    Unlike the diameter-based kernels this one has no hard lower bound on
    delay, so a small amount of probability always sits at implausibly short
    delays. That difference is the point of having both.
    """

    speed_m_s: float = 6.0
    shape: float = 4.0

    def __post_init__(self) -> None:  # noqa: D105
        if self.speed_m_s <= 0:
            raise ValueError("speed_m_s must be positive")
        if self.shape <= 0:
            raise ValueError("shape must be positive")

    def pdf(self, delays_s: FloatArray, distance_cm: float) -> FloatArray:
        """
        Evaluate the delay density for one connection length.

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
        return self.pdf_batched(delays_s, np.asarray([distance_cm], dtype=float))[0]

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
        scale = np.where(mean_s > 0, mean_s / self.shape, 1.0)
        density = np.asarray(stats.gamma.pdf(delays, a=self.shape, scale=scale))
        return np.where(np.isfinite(density) & (mean_s > 0), density, 0.0)

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {"speed_m_s": self.speed_m_s, "shape": self.shape}
