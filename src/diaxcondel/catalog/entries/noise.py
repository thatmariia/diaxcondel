"""
Noise entries: the stochastic drive that stands in for local dynamics.
"""

from __future__ import annotations

from typing import Annotated

from diaxcondel.simulate.noise import NoiseFn, get_noise_fn

from ..param import Param
from ..registry import register


@register("noise", "white", label="White (flat spectrum)", tags=("reference",))
def white_noise_fn(
    std: Annotated[float, Param(minimum=0.0, maximum=10.0, step=0.1)] = 1.0,
) -> NoiseFn:
    """
    Independent Gaussian noise, equally strong at every frequency.

    The reference choice, and the one to keep while studying what the network
    does: any structure in the output spectrum then comes from the delayed
    connections rather than from the input.

    Parameters
    ----------
    std : float
        Strength of the input at each region. It scales the whole spectrum
        without changing its shape, so it mostly matters when comparing to
        recorded amplitudes.

    Returns
    -------
    NoiseFn
        Callable ``(n_nodes, n_samples, sample_rate_hz, rng) -> (T, N)``.
    """
    return get_noise_fn("white", std=std)


@register("noise", "pink", label="Pink (1/f)", tags=("alternative",))
def pink_noise_fn(
    std: Annotated[float, Param(minimum=0.0, maximum=10.0, step=0.1)] = 1.0,
) -> NoiseFn:
    """
    Independent Gaussian noise whose power falls as ``1/f``.

    Use it to check which spectral features survive when the input itself
    already has a sloped spectrum.

    Parameters
    ----------
    std : float
        Strength of the input at each region.

    Returns
    -------
    NoiseFn
        Callable ``(n_nodes, n_samples, sample_rate_hz, rng) -> (T, N)``.
    """
    return get_noise_fn("pink", std=std)


@register("noise", "colored", label="Colored (1/f^alpha)", tags=("alternative",))
def colored_noise_fn(
    std: Annotated[float, Param(minimum=0.0, maximum=10.0, step=0.1)] = 1.0,
    exponent: Annotated[float, Param(minimum=0.0, maximum=3.0, step=0.1)] = 1.0,
) -> NoiseFn:
    """
    Gaussian noise with a chosen spectral slope.

    Parameters
    ----------
    std : float
        Strength of the input at each region.
    exponent : float
        Slope of the input spectrum: power falls as ``1 / f**exponent``.
        ``0`` is white, ``1`` pink, ``2`` brown. Whatever the network adds
        appears on top of this.

    Returns
    -------
    NoiseFn
        Callable ``(n_nodes, n_samples, sample_rate_hz, rng) -> (T, N)``.
    """
    return get_noise_fn("colored", std=std, exponent=exponent)


@register("noise", "correlated", label="Shared across regions", tags=("alternative", "control"))
def correlated_noise_fn(
    std: Annotated[float, Param(minimum=0.0, maximum=10.0, step=0.1)] = 1.0,
    correlation: Annotated[float, Param(minimum=0.0, maximum=1.0, step=0.05)] = 0.3,
    exponent: Annotated[float, Param(minimum=0.0, maximum=3.0, step=0.1, advanced=True)] = 0.0,
) -> NoiseFn:
    """
    Input that every region partly shares.

    Independent input is an assumption, not a measurement: with it, any
    coherence between regions must have travelled along a connection. Shared
    input produces coherence on its own, without any coupling at all, so
    raising this is the control that shows how much of a coherence pattern
    the connections are actually responsible for.

    Parameters
    ----------
    std : float
        Strength of the input at each region, unchanged by the sharing.
    correlation : float
        Fraction of the input each region has in common with every other.
        ``0`` is independent input; ``1`` drives the whole network with one
        signal, giving coherence one everywhere.
    exponent : float
        Spectral slope of the input, as for the coloured noise. ``0`` is
        flat.

    Returns
    -------
    NoiseFn
        Callable ``(n_nodes, n_samples, sample_rate_hz, rng) -> (T, N)``.
    """
    return get_noise_fn("correlated", std=std, correlation=correlation, exponent=exponent)
