"""
Non-linear VAR dynamics with a configurable output-transfer function.

Extends the linear case ``h(t) = Phi @ h(t-) + epsilon`` to

    h(t) = sum_{p=1}^{P} Phi[p] @ F(h(t - p)) + epsilon(t),

where ``F`` is an element-wise :class:`diaxcondel.model.transfer.TransferFn`
(typically :class:`CentredSigmoid`). Closed-form spectral analysis is not
available in general; :meth:`transfer` returns ``None`` and callers must
fall back to simulation-based PSD estimation.

A useful sanity check: with ``transfer = Identity()`` and slope-matched
parameters, :class:`NonlinearVAR` reproduces :class:`LinearVAR` outputs
exactly (up to floating-point rounding).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diaxcondel._typing import ComplexArray, FloatArray
from diaxcondel.model.transfer import Identity, TransferFn


@dataclass(frozen=True, slots=True)
class NonlinearVAR:
    """
    Non-linear VAR(P) dynamics with element-wise output transfer.

    Parameters
    ----------
    phi : FloatArray of shape (P, N, N)
        Lag-coefficient tensor.
    transfer : TransferFn
        Element-wise output-transfer function. Default :class:`Identity`
        (in which case behaviour matches :class:`LinearVAR`).

    Notes
    -----
    The :meth:`transfer` method (frequency-domain) is unrelated to the
    :attr:`transfer` attribute (element-wise output-transfer function). The
    namespace collision is unfortunate but follows established conventions
    in both signal processing and neural-field modelling. We disambiguate
    in docstrings; when in doubt, the frequency-domain method is on the
    :class:`diaxcondel.model.dynamics.Dynamics` protocol.
    """

    phi: FloatArray
    transfer_fn: TransferFn = Identity()

    def __post_init__(self) -> None:  # noqa: D105
        if self.phi.ndim != 3 or self.phi.shape[1] != self.phi.shape[2]:
            raise ValueError(f"phi must have shape (P, N, N); got {self.phi.shape}")

    @property
    def n_nodes(self) -> int:  # noqa: D102
        return int(self.phi.shape[1])

    @property
    def n_lags(self) -> int:  # noqa: D102
        return int(self.phi.shape[0])

    def step(
        self,
        history: FloatArray,
        noise: FloatArray,
        stim: FloatArray,
    ) -> FloatArray:  # noqa: D102
        """Advance the nonlinear VAR by one sample."""
        # Apply F element-wise to the lagged state, then mix with Phi.
        f_hist = self.transfer_fn(history)
        return np.einsum("pij,pj->i", self.phi, f_hist) + noise + stim

    def transfer(self, freqs_hz: FloatArray, dt_s: float) -> ComplexArray | None:  # noqa: D102
        # If F is the identity, the model is linear and we can return the
        # closed-form transfer. Otherwise no closed form is available.
        if isinstance(self.transfer_fn, Identity):
            n = self.n_nodes
            p = np.arange(1, self.n_lags + 1)
            phase = np.exp(-2j * np.pi * np.outer(freqs_hz, p) * dt_s)
            weighted = np.einsum("fp,pij->fij", phase, self.phi.astype(np.complex128))
            eye = np.eye(n, dtype=np.complex128)
            return np.linalg.inv(eye - weighted)
        return None
