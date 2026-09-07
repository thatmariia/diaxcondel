"""
Transfer entries: the element-wise map applied to lagged activity.
"""

from __future__ import annotations

from typing import Annotated

from diaxcondel.model.transfer import CentredSigmoid, HardSaturation, Identity, ThresholdLinear, TransferFn

from ..param import Param
from ..registry import register


@register("transfer", "identity", label="Identity (linear model)", tags=("reference",))
def identity_transfer() -> TransferFn:
    """
    No output transfer: the model stays linear.

    Keeps the closed-form transfer function, and with it the analytical PSD
    and the Whittle likelihood.

    Returns
    -------
    TransferFn
        The identity map.
    """
    return Identity()


@register(
    "transfer",
    "centred_sigmoid",
    label="Centred sigmoid (saturating)",
    tags=("nonlinear",),
    reference="Sigmoidal synaptic-action to firing-rate mapping",
)
def centred_sigmoid_transfer(
    slope: Annotated[float, Param(minimum=0.1, maximum=20.0, step=0.1)] = 4.0,
) -> TransferFn:
    """
    Bounded sigmoid transfer ``0.5 * tanh(slope * x / 2)``.

    Saturation lets the network run at or above the linear stability
    boundary without diverging. There is no closed-form spectrum for a
    non-linear model, so spectra must be estimated from simulated signals.

    Parameters
    ----------
    slope : float
        Slope at the origin is ``slope / 4``; ``slope = 4`` matches the unit
        gain of the linear model at small amplitudes.

    Returns
    -------
    TransferFn
        The centred sigmoid.
    """
    return CentredSigmoid(slope=slope)


@register("transfer", "hard_saturation", label="Hard limit (saturating)", tags=("nonlinear", "control"))
def hard_saturation_transfer(
    limit: Annotated[float, Param(minimum=0.01, maximum=100.0, step=0.01)] = 1.0,
) -> TransferFn:
    """
    Pass activity through unchanged until it hits a ceiling.

    The control for the sigmoid: it bounds activity in the same way but adds
    no compression below the limit, so anything the two do differently is
    caused by the smooth squashing rather than by saturation itself. Like the
    sigmoid it keeps the model bounded past the linear stability boundary,
    and like it, it has no closed-form spectrum.

    Parameters
    ----------
    limit : float
        Largest magnitude passed on. Below it the model behaves exactly
        linearly; the smaller it is relative to the noise, the more often the
        network runs into the ceiling.

    Returns
    -------
    TransferFn
        The clipping transfer.
    """
    return HardSaturation(limit=limit)


@register("transfer", "threshold_linear", label="Rectified (positive only)", tags=("nonlinear",))
def threshold_linear_transfer(
    threshold: Annotated[float, Param(minimum=-10.0, maximum=10.0, step=0.1)] = 0.0,
) -> TransferFn:
    """
    Pass on only what rises above a threshold.

    Firing rates cannot go negative, and rate models encode that with a
    rectifying transfer. It is the one transfer here that is not symmetric
    about zero, so activity gains a non-zero mean and the model is no longer
    comparable to the linear one in absolute level, only in its
    fluctuations. It also does not bound activity from above, so it needs a
    stationary operating point.

    Parameters
    ----------
    threshold : float
        Level below which nothing is passed on. ``0`` rectifies at the
        operating point; positive values also suppress small fluctuations,
        which makes the network's response depend on input strength.

    Returns
    -------
    TransferFn
        The rectifying transfer.
    """
    return ThresholdLinear(threshold=threshold)
