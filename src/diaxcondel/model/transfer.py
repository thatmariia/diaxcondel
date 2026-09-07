"""
Output-transfer functions ``F(H)``.

In the macroscopic neural-field formulation, the next-step contribution of
each region is a transformation ``F`` of the (lag-weighted) synaptic action
``H``. The original VAR model of Steeghs-Turchina et al. (2025) takes ``F``
to be the identity; a saturating model uses a centred sigmoid

    F_c(H; s) = 1 / (1 + exp(-s H)) - 1/2

which is anti-symmetric around the origin and provides amplitude
saturation. This module collects such transfer functions as small frozen
dataclasses implementing the :class:`TransferFn` protocol.

A :class:`TransferFn` is consumed by
:class:`diaxcondel.model.var.nonlinear.NonlinearVAR`. For inference it must
also expose its scalar parameters via
:meth:`parameters` so that they can be packed into a
:class:`diaxcondel.model.parameter_spec.ParameterSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from diaxcondel._typing import FloatArray


@runtime_checkable
class TransferFn(Protocol):
    """Protocol for scalar output-transfer functions applied element-wise."""

    def __call__(self, x: FloatArray) -> FloatArray:
        """
        Apply the transfer function element-wise.

        Parameters
        ----------
        x : FloatArray
            Pre-transfer values of arbitrary shape.

        Returns
        -------
        FloatArray
            Same shape as ``x``.
        """
        ...

    def derivative(self, x: FloatArray) -> FloatArray:
        """
        Element-wise derivative ``F'(x)``.

        Used for local Jacobians (stability analysis at non-zero operating
        points; not needed for forward simulation).

        Parameters
        ----------
        x : FloatArray
            Points at which to evaluate.

        Returns
        -------
        FloatArray
            Derivative values, same shape as ``x``.
        """
        ...

    def parameters(self) -> dict[str, float]:
        """Return scalar parameters as a dict (used by parameter specs)."""
        ...


@dataclass(frozen=True, slots=True)
class Identity:
    """
    Identity transfer ``F(x) = x``.

    Notes
    -----
    Linear VAR systems use this implicitly. It exists here so that
    :class:`diaxcondel.model.var.nonlinear.NonlinearVAR` with ``transfer_fn=Identity()``
    reduces exactly to a linear VAR — a useful test path.
    """

    def __call__(self, x: FloatArray) -> FloatArray:  # noqa: D102
        return x

    def derivative(self, x: FloatArray) -> FloatArray:  # noqa: D102
        return np.ones_like(x)

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {}


@dataclass(frozen=True, slots=True)
class CentredSigmoid:
    r"""
    Centred sigmoid transfer ``F_c(x; s) = 1 / (1 + exp(-s x)) - 1/2``.

    Parameters
    ----------
    slope : float
        The slope parameter ``s``. The derivative at the origin is
        ``s / 4``; with ``slope = 4`` the linearised gain equals 1, matching
        the implicit gain of :class:`Identity` and so making the linear and
        non-linear models comparable at small amplitudes.

    Notes
    -----
    Bounded output in ``(-1/2, 1/2)`` provides automatic amplitude
    saturation, which permits operation at or above the linear stability
    boundary (spectral radius :math:`\\geq 1`) without divergence, where
    spectral peaks are sharpest.
    """

    slope: float = 4.0

    def __post_init__(self) -> None:  # noqa: D105
        if self.slope <= 0:
            raise ValueError("slope must be positive")

    def __call__(self, x: FloatArray) -> FloatArray:  # noqa: D102
        # Numerically stable: avoid overflow in exp by handling sign.
        sx = self.slope * x
        # 1/(1+exp(-sx)) - 1/2 = 0.5 * tanh(sx / 2)
        return 0.5 * np.tanh(0.5 * sx)

    def derivative(self, x: FloatArray) -> FloatArray:  # noqa: D102
        # d/dx [0.5 * tanh(s x / 2)] = (s/4) * sech^2(s x / 2)
        sx_half = 0.5 * self.slope * x
        tanh = np.tanh(sx_half)
        return 0.25 * self.slope * (1.0 - tanh**2)

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {"slope": self.slope}


@dataclass(frozen=True, slots=True)
class HardSaturation:
    r"""
    Hard-limiting transfer ``F(x) = clip(x, -limit, +limit)``.

    The control for :class:`CentredSigmoid`: it saturates at the same kind of
    ceiling but leaves everything below it exactly linear, so a difference
    between the two is a statement about smooth compression rather than about
    saturation as such. Anti-symmetric, so the model stays zero-mean.

    Parameters
    ----------
    limit : float
        Largest magnitude the transfer will pass. Must be positive.
    """

    limit: float = 1.0

    def __post_init__(self) -> None:  # noqa: D105
        if self.limit <= 0:
            raise ValueError("limit must be positive")

    def __call__(self, x: FloatArray) -> FloatArray:  # noqa: D102
        return np.clip(x, -self.limit, self.limit)

    def derivative(self, x: FloatArray) -> FloatArray:  # noqa: D102
        return (np.abs(x) < self.limit).astype(float)

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {"limit": self.limit}


@dataclass(frozen=True, slots=True)
class ThresholdLinear:
    r"""
    Rectifying transfer ``F(x) = max(x - threshold, 0)``.

    Firing rates cannot fall below zero, and rate models built on that fact
    use a rectified-linear transfer. Unlike the other transfers here it is
    not anti-symmetric: only positive excursions propagate, so activity
    acquires a non-zero mean and the spectrum of the fluctuations sits on top
    of it. Compare it against the linear model on the fluctuations, not on
    absolute level.

    Parameters
    ----------
    threshold : float
        Level below which nothing is passed on. ``0`` rectifies at the
        operating point; positive values also silence small fluctuations.
    """

    threshold: float = 0.0

    def __call__(self, x: FloatArray) -> FloatArray:  # noqa: D102
        return np.clip(x - self.threshold, 0.0, None)

    def derivative(self, x: FloatArray) -> FloatArray:  # noqa: D102
        return (x > self.threshold).astype(float)

    def parameters(self) -> dict[str, float]:  # noqa: D102
        return {"threshold": self.threshold}
