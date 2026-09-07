"""
Delay-kernel protocol.

A delay kernel maps a connection length (in cm) to a probability density
function over transmission delay (in seconds). Implementations encode the
relationship ``v = k * phi`` (Hursh, 1939) and the chosen distribution over
axon diameters.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from diaxcondel._typing import FloatArray


@runtime_checkable
class DelayKernel(Protocol):
    """Protocol for delay-distribution kernels."""

    def pdf(self, delays_s: FloatArray, distance_cm: float) -> FloatArray:
        """
        Evaluate the delay PDF on a grid for a single connection length.

        Parameters
        ----------
        delays_s : FloatArray of shape (P,)
            Delay grid in seconds (must be strictly positive).
        distance_cm : float
            Connection length in centimetres.

        Returns
        -------
        FloatArray of shape (P,)
            PDF values evaluated on ``delays_s``.
        """
        ...

    def pdf_batched(self, delays_s: FloatArray, distances_cm: FloatArray) -> FloatArray:
        """Default implementation falls back to per-pair calls."""
        out = np.empty((distances_cm.size, delays_s.size))
        for k, d in enumerate(distances_cm.ravel()):
            out[k] = self.pdf(delays_s, float(d))
        return out

    def parameters(self) -> dict[str, float]:
        """
        Return the kernel's free scalar parameters as a dict.

        Used by :class:`diaxcondel.model.parameter_spec.ParameterSpec` for
        inference. Names should be stable across versions.
        """
        ...
