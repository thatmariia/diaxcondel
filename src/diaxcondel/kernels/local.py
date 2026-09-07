"""
Local (receiver-side) delay kernels.

Transmission kernels describe how long a spike takes to travel *between*
regions. They say nothing about what happens once it arrives: postsynaptic
potentials rise and decay over a few milliseconds, signals cross cortical
layers, and short-range U-fibre connections that were collapsed into the
macroscopic region add their own travel time.

That delay is a property of the **receiving** region, not of the connection,
and it applies identically to every input — including a region's own
recurrent excitation. Its effect on the model is a convolution: the lag
weights of every incoming connection are convolved with the local delay
distribution, which shifts and broadens the effective delays (and so lowers
resonance frequencies).

Kernels here therefore expose a discrete probability mass over lag *offsets*
``0, 1, ..., n_lags - 1``, ready to convolve with a transmission kernel.
Offset zero means "no additional delay".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy import stats

from diaxcondel._typing import FloatArray


@runtime_checkable
class LocalDelayKernel(Protocol):
    """Protocol for receiver-side (postsynaptic) delay distributions."""

    def mass(self, dt_s: float, n_lags: int) -> FloatArray:
        """
        Return the discrete delay mass over lag offsets ``0 .. n_lags - 1``.

        Parameters
        ----------
        dt_s : float
            Sample period in seconds.
        n_lags : int
            Number of lag offsets to return.

        Returns
        -------
        FloatArray of shape (n_lags,)
            Non-negative masses summing to one.
        """
        ...

    def parameters(self) -> dict[str, float]:
        """Return the kernel's free scalar parameters as a dict."""
        ...


@dataclass(frozen=True, slots=True)
class NoLocalDelay:
    """
    Instantaneous postsynaptic response: no local delay at all.

    This is the assumption of the original model, where local dynamics are
    absorbed into the noise term. Convolving with it leaves transmission
    kernels unchanged.
    """

    def mass(self, dt_s: float, n_lags: int) -> FloatArray:  # noqa: D102, ARG002
        out = np.zeros(int(n_lags))
        out[0] = 1.0
        return out

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {}


@dataclass(frozen=True, slots=True)
class GammaLocalDelay:
    """
    Gamma-distributed local delay, for PSP rise/decay and layer crossing.

    Parameterised by its mean and shape rather than by rate and scale,
    because the mean is the quantity with a physiological reading: AMPA-type
    postsynaptic potentials peak a few milliseconds after arrival.

    Parameters
    ----------
    mean_ms : float
        Mean local delay in milliseconds. Must be positive.
    shape : float
        Gamma shape parameter ``k``. The distribution has standard deviation
        ``mean_ms / sqrt(k)``, so larger values give a more sharply timed
        response; ``k = 1`` is an exponential decay.

    Notes
    -----
    The continuous density is sampled on the lag grid and renormalised to
    sum to one, so the total synaptic gain is preserved exactly and only its
    timing is redistributed. If the grid is too coarse to resolve the delay
    (``mean_ms`` shorter than one sample), almost all mass lands on lag zero
    and the kernel is effectively :class:`NoLocalDelay`.
    """

    mean_ms: float = 3.0
    shape: float = 4.0

    def __post_init__(self) -> None:  # noqa: D105
        if self.mean_ms <= 0:
            raise ValueError("mean_ms must be positive")
        if self.shape <= 0:
            raise ValueError("shape must be positive")

    def mass(self, dt_s: float, n_lags: int) -> FloatArray:  # noqa: D102
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if n_lags < 1:
            raise ValueError("n_lags must be at least 1")
        lags_s = np.arange(int(n_lags)) * dt_s
        scale_s = (self.mean_ms * 1e-3) / self.shape
        density = np.asarray(stats.gamma.pdf(lags_s, a=self.shape, scale=scale_s), dtype=float)
        density = np.where(np.isfinite(density), density, 0.0)
        total = density.sum()
        if total <= 0:  # pragma: no cover - only for absurd parameter values
            out = np.zeros(int(n_lags))
            out[0] = 1.0
            return out
        return density / total

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {"mean_ms": self.mean_ms, "shape": self.shape}


@dataclass(frozen=True, slots=True)
class BiexponentialLocalDelay:
    """
    Postsynaptic response with separate rise and decay times.

    The shape neural-mass models use for a synaptic potential: a difference
    of two exponentials, ``exp(-t / tau_decay) - exp(-t / tau_rise)``, which
    rises over the shorter constant and falls over the longer one. Unlike a
    gamma it lets the two be set independently, so a fast AMPA-like rise can
    be combined with a slow decay.

    Parameters
    ----------
    rise_ms : float
        Rise time constant in milliseconds. Must be positive and smaller
        than ``decay_ms``.
    decay_ms : float
        Decay time constant in milliseconds.

    Notes
    -----
    The mean delay is ``rise_ms + decay_ms``, so a 1 ms rise with a 6 ms
    decay shifts every arrival about 7 ms later. As with every kernel here,
    the sampled shape is renormalised to sum to one, so only the timing of
    the synaptic gain is redistributed and never its total.
    """

    rise_ms: float = 1.0
    decay_ms: float = 6.0

    def __post_init__(self) -> None:  # noqa: D105
        if self.rise_ms <= 0 or self.decay_ms <= 0:
            raise ValueError("rise_ms and decay_ms must be positive")
        if self.rise_ms >= self.decay_ms:
            raise ValueError("rise_ms must be shorter than decay_ms")

    def mass(self, dt_s: float, n_lags: int) -> FloatArray:  # noqa: D102
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if n_lags < 1:
            raise ValueError("n_lags must be at least 1")
        lags_s = np.arange(int(n_lags)) * dt_s
        shape = np.exp(-lags_s / (self.decay_ms * 1e-3)) - np.exp(-lags_s / (self.rise_ms * 1e-3))
        shape = np.clip(shape, 0.0, None)
        total = shape.sum()
        if total <= 0:  # pragma: no cover - only when the grid cannot resolve the rise
            out = np.zeros(int(n_lags))
            out[0] = 1.0
            return out
        return shape / total

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {"rise_ms": self.rise_ms, "decay_ms": self.decay_ms}


def local_delay_mass(kernel: LocalDelayKernel | None, dt_s: float, n_lags: int) -> FloatArray:
    """
    Return the discrete local-delay mass, tolerating ``None``.

    Parameters
    ----------
    kernel : LocalDelayKernel or None
        Kernel to discretise. ``None`` behaves like :class:`NoLocalDelay`.
    dt_s : float
        Sample period in seconds.
    n_lags : int
        Number of lag offsets.

    Returns
    -------
    FloatArray of shape (n_lags,)
        Probability mass over lag offsets.
    """
    if kernel is None:
        return NoLocalDelay().mass(dt_s, n_lags)
    return np.asarray(kernel.mass(dt_s, n_lags), dtype=float)
