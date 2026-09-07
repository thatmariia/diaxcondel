"""
Local delay: what happens after a signal arrives at its target region.

Transmission kernels cover travel time between regions. A signal then has to
cross the postsynaptic machinery of the receiving region. Potentials rise
and decay, layers are crossed, short-range fibres inside the region add their
own travel time. That delay belongs to the receiver, applies to every input
including a region's own recurrent excitation, and enters the model as a
convolution with each incoming connection's lag weights.
"""

from __future__ import annotations

from typing import Annotated

from diaxcondel.kernels.local import (
    BiexponentialLocalDelay,
    GammaLocalDelay,
    LocalDelayKernel,
    NoLocalDelay,
)

from ..param import Param
from ..registry import register


@register("local_delay", "none", label="No local delay", tags=("reference",))
def no_local_delay() -> LocalDelayKernel:
    """
    Ignore postsynaptic processing time, as in the original model.

    Local dynamics are absorbed into the noise term instead. Recurrent
    self-excitation cannot be used in this setting, because feedback with no
    delay is instantaneous.

    Returns
    -------
    LocalDelayKernel
        A kernel that leaves transmission delays untouched.
    """
    return NoLocalDelay()


@register("local_delay", "gamma", label="Gamma-distributed local delay", tags=("extension",))
def gamma_local_delay(
    mean_ms: Annotated[float, Param(unit="ms", minimum=0.5, maximum=50.0, step=0.5)] = 3.0,
    shape: Annotated[float, Param(minimum=1.0, maximum=20.0, step=0.5)] = 4.0,
) -> LocalDelayKernel:
    """
    Postsynaptic delay of a few milliseconds, spread by a gamma distribution.

    Every incoming connection is delayed by this much on top of its travel
    time, which lowers resonance frequencies and smooths the delay
    distribution. It is also what makes recurrent self-excitation possible.

    Parameters
    ----------
    mean_ms : float
        Average local delay. Fast excitatory synapses peak a few
        milliseconds after arrival; larger values shift the whole spectrum
        towards lower frequencies.
    shape : float
        How tightly the delay is timed: the spread is ``mean_ms / sqrt(shape)``,
        so large values approach a fixed delay and ``1`` gives an exponential
        decay.

    Returns
    -------
    LocalDelayKernel
        Gamma-shaped local delay.
    """
    return GammaLocalDelay(mean_ms=mean_ms, shape=shape)


@register("local_delay", "biexponential", label="Synaptic rise and decay", tags=("alternative",))
def biexponential_local_delay(
    rise_ms: Annotated[float, Param(unit="ms", minimum=0.1, maximum=20.0, step=0.1)] = 1.0,
    decay_ms: Annotated[float, Param(unit="ms", minimum=0.5, maximum=100.0, step=0.5)] = 6.0,
) -> LocalDelayKernel:
    """
    Postsynaptic potential with separate rise and decay times.

    The shape neural-mass models use for a synaptic response: fast rise,
    slower decay. Setting the two independently is what distinguishes it from
    the gamma kernel, and it matters when a receptor's decay is much longer
    than its rise.

    Parameters
    ----------
    rise_ms : float
        Rise time constant, in milliseconds. Must be shorter than the decay.
    decay_ms : float
        Decay time constant, in milliseconds. Together the two set the mean
        added delay, ``rise_ms + decay_ms``, which shifts every arrival later
        and so lowers every resonance frequency.

    Returns
    -------
    LocalDelayKernel
        Receiver-side delay distribution.
    """
    return BiexponentialLocalDelay(rise_ms=rise_ms, decay_ms=decay_ms)
