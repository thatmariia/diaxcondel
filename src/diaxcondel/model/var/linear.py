# src/diaxcondel/model/linear_var.py
"""
Linear vector autoregressive (VAR) dynamics.

Implements the model of Steeghs-Turchina et al. (2025): for ``N`` regions
and ``P`` lags,

    h(t) = sum_{p=1}^{P} Phi[p] @ h(t - p) + epsilon(t)

where ``Phi[p] = ell[p] * gamma`` is the lag-``p`` coefficient matrix,
``ell[p]`` is the delay-PDF entry and ``gamma`` the connectivity matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diaxcondel._typing import ComplexArray, FloatArray


@dataclass(frozen=True, slots=True)
class LinearVAR:
    """
    Linear VAR(P) dynamics.

    Parameters
    ----------
    phi : FloatArray of shape (P, N, N)
        Lag-``p`` coefficient matrix ``Phi[p - 1]`` for ``p = 1, ..., P``.

    Notes
    -----
    Construction is normally done via
    :func:`diaxcondel.model.build.build_linear_var`, which assembles ``Phi``
    from a connectome and a delay kernel.
    """

    phi: FloatArray

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
        """Advance the linear VAR by one sample."""
        # history: (P, N), Phi: (P, N, N) -> contract over (P, N_in)
        # Equivalent to sum_p Phi[p] @ history[p].
        return np.einsum("pij,pj->i", self.phi, history) + noise + stim

    def transfer(self, freqs_hz: FloatArray, dt_s: float) -> ComplexArray:  # noqa: D102
        # T(f) = (I - sum_p Phi[p] * exp(-i 2 pi f p dt))^{-1}
        n = self.n_nodes
        p = np.arange(1, self.n_lags + 1)
        # phase[f, p] = exp(-i 2 pi f p dt)
        phase = np.exp(-2j * np.pi * np.outer(freqs_hz, p) * dt_s)
        # weighted_phi[f, i, j] = sum_p phase[f, p] * phi[p, i, j]
        weighted = np.einsum("fp,pij->fij", phase, self.phi.astype(np.complex128))
        eye = np.eye(n, dtype=np.complex128)
        return np.linalg.inv(eye - weighted)

    def companion_matrix(self) -> FloatArray:
        """
        Return the ``(NP, NP)`` companion matrix of the VAR system.

        Used for stability analysis and for the vectorized state-space
        simulation kernel.

        Returns
        -------
        FloatArray of shape (NP, NP)
            Companion matrix ``A`` such that
            ``[h(t), h(t-1), ..., h(t-P+1)] = A @ [h(t-1), ..., h(t-P)]``
            in the absence of noise.
        """
        n, p = self.n_nodes, self.n_lags
        a = np.zeros((n * p, n * p))
        # Top block-row: Phi_1, Phi_2, ..., Phi_P.
        for k in range(p):
            a[:n, k * n : (k + 1) * n] = self.phi[k]
        # Sub-diagonal identity blocks.
        if p > 1:
            a[n:, : (p - 1) * n] = np.eye((p - 1) * n)
        return a
