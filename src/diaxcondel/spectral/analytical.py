"""
Closed-form power spectral density from VAR coefficients.

For a linear VAR driven by zero-mean noise with covariance ``Sigma``, the
PSD matrix is

    S(f) = T(f) @ Sigma @ T(f)^H,

where ``T(f) = (I - sum_p Phi[p] exp(-i 2 pi f p dt))^{-1}`` is the transfer
matrix. This module evaluates ``S(f)`` directly from a :class:`Dynamics`
that exposes a closed-form transfer.
"""

from __future__ import annotations

import numpy as np

from diaxcondel._typing import FloatArray
from diaxcondel.model.dynamics import Dynamics


def analytical_psd(
    dynamics: Dynamics,
    freqs_hz: FloatArray,
    dt_s: float,
    *,
    noise_cov: FloatArray | None = None,
) -> FloatArray:
    """
    Compute the analytical PSD matrix from a dynamics' transfer function.

    Parameters
    ----------
    dynamics : Dynamics
        Must implement :meth:`Dynamics.transfer` (returning a non-``None``
        complex array). Otherwise raises ``ValueError``.
    freqs_hz : FloatArray of shape (F,)
        Frequencies at which to evaluate the PSD.
    dt_s : float
        Sampling period.
    noise_cov : FloatArray of shape (N, N), optional
        Driving-noise covariance. ``None`` is interpreted as identity.

    Returns
    -------
    FloatArray of shape (F, N, N)
        Real part of ``T(f) @ Sigma @ T(f)^H``. The off-diagonal imaginary
        part (corresponding to phase lag) is preserved in the underlying
        complex transfer; if you need it, call :meth:`Dynamics.transfer`
        directly.
    """
    t = dynamics.transfer(freqs_hz, dt_s)
    if t is None:
        raise ValueError("dynamics has no closed-form transfer; use empirical PSD")
    n = dynamics.n_nodes
    sigma = np.eye(n) if noise_cov is None else noise_cov
    # S = T Sigma T^H, batched over freqs.
    s = np.einsum("fij,jk,fmk->fim", t, sigma, np.conj(t))
    return np.real(s)
