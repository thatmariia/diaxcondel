"""
Delay kernels: how a connection's length becomes a distribution of arrival times.

The reference construction follows Steeghs-Turchina et al. (2025) in two
steps. First, the axons of a bundle differ in calibre, and that distribution is
chosen separately, in the ``diameter`` slot. Second, conduction speed grows
linearly with calibre, ``v = speed_factor * diameter`` (Hursh, 1939), so a
connection of length ``d`` delivers its signal after ``d / v``: one delay per
axon, and therefore a distribution of delays whose shape follows from the
calibre distribution by change of variables.

The alternatives here skip the calibre step and put a spread on delay or
speed directly. They are controls: if a spectral feature survives them, it
does not depend on how calibres are distributed.
"""

from __future__ import annotations

from typing import Annotated

from diaxcondel.kernels.base import DelayKernel
from diaxcondel.kernels.diameter import DiameterDelayKernel, DiameterDistribution, GammaDelayKernel
from diaxcondel.kernels.fixed_speed import FixedSpeedKernel

from ..param import Param
from ..registry import register


@register(
    "kernel",
    "hursh",
    label="Speed proportional to axon calibre (reference)",
    tags=("reference",),
    reference="Hursh (1939) velocity relation, as used by Steeghs-Turchina et al. (2025)",
)
def hursh_kernel(
    diameter: DiameterDistribution,
    speed_factor: Annotated[float, Param(unit="m/s/um", minimum=0.5, maximum=15.0, step=0.5)] = 6.0,
) -> DelayKernel:
    """
    Turn axon calibres into delays through a linear speed relation.

    Each axon conducts at ``speed_factor`` metres per second for every
    micrometre of its diameter, so a bundle whose calibres are spread out
    delivers one signal over a spread of delays. The typical delay of a
    connection is its length divided by the typical speed, and it is that
    delay, through the round trip between two regions, that sets which
    frequencies the network favours.

    Parameters
    ----------
    diameter : DiameterDistribution
        Axon-calibre distribution (chosen in the ``diameter`` slot; supplied
        by the builder).
    speed_factor : float
        Conduction speed per micrometre of calibre, in metres per second.
        The original paper fixes it at 6; the plausible range quoted in the
        literature is roughly 3 to 9, and it scales every delay inversely:
        doubling it halves all delays and doubles every resonance frequency.

    Returns
    -------
    DelayKernel
        Delay kernel for direct connections; a relayed model automatically
        switches to its two-leg counterpart, keeping the same calibres and
        speed.
    """
    return DiameterDelayKernel(diameter=diameter, speed_factor=speed_factor)


@register(
    "kernel",
    "gamma_delay",
    label="Gamma-distributed delays (no calibres)",
    tags=("control",),
    reference="Delay spreads as used in neural field models",
)
def gamma_delay_kernel(
    speed_m_s: Annotated[float, Param(unit="m/s", minimum=0.5, maximum=30.0, step=0.5)] = 6.0,
    shape: Annotated[float, Param(minimum=1.0, maximum=30.0, step=0.5)] = 4.0,
) -> DelayKernel:
    """
    Spread delays directly, without going through axon calibres.

    Neural field models often assume the delay itself is gamma-distributed
    around length over speed. Compared with the calibre-based construction
    this puts a little probability at implausibly short delays, and none of
    its parameters refer to anatomy, which is what makes it a useful
    control.

    Parameters
    ----------
    speed_m_s : float
        Mean conduction speed of the whole bundle, in metres per second. Note
        that this is a *speed*, not a speed per micrometre: with calibres
        around 0.25 um, a calibre-based speed factor of 6 corresponds to
        roughly 1.5 m/s here.
    shape : float
        How tightly delays cluster around the mean: the spread is
        ``mean / sqrt(shape)``, so large values approach a fixed delay.

    Returns
    -------
    DelayKernel
        Delay kernel for direct connections.
    """
    return GammaDelayKernel(speed_m_s=speed_m_s, shape=shape)


@register(
    "kernel",
    "fixed_speed",
    label="One speed for every axon (no calibres)",
    tags=("control",),
)
def fixed_speed_kernel(
    speed_m_s: Annotated[float, Param(unit="m/s", minimum=0.5, maximum=30.0, step=0.5)] = 6.0,
    jitter_ms: Annotated[float, Param(unit="ms", minimum=0.1, maximum=20.0, step=0.1)] = 1.0,
) -> DelayKernel:
    """
    Give every axon the same speed, so each connection has a single delay.

    The control condition for the calibre-based kernel: keeping the typical
    delay but removing its spread shows how much of the spectrum depends on
    delays being distributed rather than fixed.

    Parameters
    ----------
    speed_m_s : float
        Conduction speed shared by all axons, in metres per second. Delay is
        length divided by it, so it sets every resonance frequency directly.
    jitter_ms : float
        Small spread around that delay, in milliseconds. It keeps the
        discretised kernel from falling between samples; larger values
        interpolate towards a broad kernel.

    Returns
    -------
    DelayKernel
        Narrow, single-peaked delay kernel.
    """
    return FixedSpeedKernel(speed_m_s=speed_m_s, jitter_ms=jitter_ms)
