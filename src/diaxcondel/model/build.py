"""
Builders for concrete :class:`Dynamics` instances.

These are the only functions in the package that compose a connectome, a
delay kernel, and simulation parameters into a coefficient tensor. Keep all
that logic here; downstream code (simulation, spectral, inference) accepts
a generic :class:`Dynamics`.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import fftconvolve

from diaxcondel._typing import FloatArray
from diaxcondel.connectome.distance import DistanceProvider, LengthMixture, RelayedDistances
from diaxcondel.connectome.weights import ConnectivityProvider
from diaxcondel.kernels.base import DelayKernel
from diaxcondel.kernels.local import LocalDelayKernel, local_delay_mass

from .params import SimulationParams
from .stability import scale_to_spectral_radius, shrink_to_stationary
from .transfer import Identity, TransferFn
from .var import LinearVAR, NonlinearVAR


def _discretize_kernel(pdf_values: FloatArray, dt_s: float) -> FloatArray:
    """Convert a continuous PDF on a regular grid into a discrete probability mass."""
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    return pdf_values * dt_s


def _validate_square_matrix(name: str, matrix: FloatArray, n: int) -> None:
    """Validate a finite ``(N, N)`` matrix."""
    if matrix.shape != (n, n):
        raise ValueError(f"{name} shape {matrix.shape} != ({n}, {n})")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")


def build_lag_tensor(
    distances: DistanceProvider,
    weights: ConnectivityProvider,
    kernel: DelayKernel,
    params: SimulationParams,
    *,
    local_kernel: LocalDelayKernel | None = None,
    self_weight: float = 0.0,
    use_length_mixture: bool = True,
) -> FloatArray:
    """
    Build the ``(P, N, N)`` lag tensor for a linear VAR.

    Parameters
    ----------
    distances : DistanceProvider
        Inter-region distances (cm). May be a :class:`RelayedDistances`, in
        which case the kernel must describe two-leg paths (it must expose
        ``pdf_two_leg``).
    weights : ConnectivityProvider
        Connectivity weights ``gamma``. Must share the same region set as
        ``distances``.
    kernel : DelayKernel
        Delay-distribution kernel.
    params : SimulationParams
        Simulation parameters; only ``n_lags`` and ``dt_s`` are used here.
    local_kernel : LocalDelayKernel, optional
        Receiver-side delay (postsynaptic potential rise and decay, layer
        crossing, collapsed short-range fibres). Every incoming connection of
        a region is convolved with it, which shifts and broadens the
        effective delays. ``None`` reproduces the original model, where local
        dynamics are absorbed into the noise.
    self_weight : float
        Weight of each region's own recurrent excitation, added to the
        diagonal of ``gamma``. Requires ``local_kernel``: without a local
        delay the feedback would be instantaneous, which a VAR cannot
        represent.
    use_length_mixture : bool
        When the distance provider carries a
        :class:`~diaxcondel.connectome.distance.LengthMixture` (merged
        regions), evaluate the delay kernel at every member length and mix
        the results, weighted by how many fibres each member carries.
        ``False`` uses one representative length per connection instead,
        which is cheaper and narrower.

    Returns
    -------
    FloatArray of shape (P, N, N)
        Lag tensor ``Phi`` such that ``Phi[p - 1, i, j] = ell[p, i, j] * gamma[i, j]``,
        with the local delay convolved in and self-connections on the
        diagonal when requested.

    Raises
    ------
    ValueError
        If region sets disagree, shapes mismatch the relay/kernel pairing, or
        ``self_weight`` is requested without a local delay.
    """
    if distances.regions.codes != weights.regions.codes:
        raise ValueError("distance and weight providers must share the same RegionSet")

    g = np.asarray(weights.matrix(), dtype=float)
    n = g.shape[0]
    _validate_square_matrix("weights", g, len(weights.regions))
    if np.any(g < 0):
        raise ValueError("weights must be non-negative")

    if self_weight < 0:
        raise ValueError("self_weight must be non-negative")
    if not np.any(g) and self_weight == 0:
        warnings.warn(
            "every coupling weight is zero: the regions are not connected, so each one simply "
            "reproduces its own input. That is the uncoupled control if you meant it, and a "
            "mis-specified connectivity source otherwise",
            stacklevel=2,
        )

    p = params.n_lags
    delays = np.arange(1, p + 1) * params.dt_s
    phi = np.zeros((p, n, n), dtype=float)
    offdiag = ~np.eye(n, dtype=bool)
    # Local delay over lag offsets 0..P; offset 0 means "no extra delay".
    local_mass = local_delay_mass(local_kernel, params.dt_s, p + 1)

    is_relayed = isinstance(distances, RelayedDistances)
    if is_relayed and not hasattr(kernel, "pdf_two_leg"):
        raise ValueError(
            "relayed distances need a kernel that can describe a two-leg path; "
            f"{type(kernel).__name__} cannot. Build it from the kernel's relay_kernel()."
        )

    mixture = getattr(distances, "mixture", None) if use_length_mixture else None

    if is_relayed:
        relay_kernel = kernel
        base_d = np.asarray(distances.base.matrix(), dtype=float)
        _validate_square_matrix("base distance matrix", base_d, n)
        if np.any(base_d < 0):
            raise ValueError("distances must be non-negative")
        relay_idx = distances.base.regions.index(distances.relay_code)
        for i in range(n):
            for j in range(n):
                if i == j or g[i, j] == 0.0:
                    continue
                if i == relay_idx or j == relay_idx:
                    # Direct (single-leg) connection to/from the relay itself.
                    pdf = relay_kernel.pdf(delays, base_d[i, j])
                else:
                    leg1, leg2 = base_d[i, relay_idx], base_d[relay_idx, j]
                    pdf = relay_kernel.pdf_two_leg(delays, leg1, leg2)
                pdf = np.asarray(pdf, dtype=float)
                if not np.all(np.isfinite(pdf)) or np.any(pdf < 0):
                    raise ValueError(f"kernel returned invalid PDF values for the path {j} -> {i}")
                phi[:, i, j] = _discretize_kernel(pdf, params.dt_s) * g[i, j]
    elif mixture is not None and len(mixture):
        d = np.asarray(distances.matrix(), dtype=float)
        _validate_square_matrix("distance matrix", d, n)
        active = offdiag & (g != 0.0)
        phi = _mixture_lag_tensor(mixture, kernel, delays, params.dt_s, g, active, p, n)
    else:
        d = np.asarray(distances.matrix(), dtype=float)
        _validate_square_matrix("distance matrix", d, n)
        if np.any(d < 0):
            raise ValueError("distances must be non-negative")
        active = offdiag & (g != 0.0)
        if np.any(d[active] <= 0):
            raise ValueError("non-zero off-diagonal weights require positive distances")
        if np.any(active):
            pdf = np.asarray(kernel.pdf_batched(delays, d[active]), dtype=float)
            expected = (int(np.count_nonzero(active)), p)
            if pdf.shape != expected:
                raise ValueError(f"kernel.pdf_batched returned shape {pdf.shape}, expected {expected}")
            if not np.all(np.isfinite(pdf)):
                raise ValueError("kernel returned non-finite PDF values")
            if np.any(pdf < 0):
                raise ValueError("kernel returned negative PDF values")
            phi[:, active] = (_discretize_kernel(pdf, params.dt_s) * g[active, None]).T

    if local_kernel is not None:
        phi = _convolve_local_delay(phi, local_mass)

    if self_weight > 0:
        phi = _add_self_connections(phi, local_mass, self_weight, local_kernel is not None)

    return phi


def _mixture_lag_tensor(
    mixture: LengthMixture,
    kernel: DelayKernel,
    delays: FloatArray,
    dt_s: float,
    weights: FloatArray,
    active: FloatArray,
    p: int,
    n: int,
) -> FloatArray:
    """
    Build the lag tensor from every member length behind each connection.

    Each merged connection's delay distribution is the weighted mixture of
    its members' distributions — the physically direct statement that some
    fibres of the pathway are longer than others — rather than the single
    distribution of an averaged length.
    """
    phi = np.zeros((p, n, n), dtype=float)
    rows = np.asarray(mixture.rows, dtype=int)
    cols = np.asarray(mixture.cols, dtype=int)
    lengths = np.asarray(mixture.lengths_cm, dtype=float)
    member_weights = np.asarray(mixture.weights, dtype=float)

    keep = active[rows, cols] | active[cols, rows]
    rows, cols, lengths, member_weights = rows[keep], cols[keep], lengths[keep], member_weights[keep]
    if rows.size == 0:
        return phi
    if not np.any(member_weights > 0):
        member_weights = np.ones_like(member_weights)

    pair_id, inverse = np.unique(np.stack([rows, cols], axis=1), axis=0, return_inverse=True)
    inverse = np.reshape(inverse, -1)  # NumPy has returned this both flat and column-shaped
    n_pairs = pair_id.shape[0]
    accumulated = np.zeros((n_pairs, p), dtype=float)
    normaliser = np.zeros(n_pairs, dtype=float)

    # Chunk the kernel evaluation: a fine parcellation can have tens of
    # thousands of member lengths, and (K, P) at once would be large.
    chunk = max(1, int(2**22 // max(p, 1)))
    for start in range(0, rows.size, chunk):
        stop = min(start + chunk, rows.size)
        pdf = np.asarray(kernel.pdf_batched(delays, lengths[start:stop]), dtype=float)
        if not np.all(np.isfinite(pdf)) or np.any(pdf < 0):
            raise ValueError("kernel returned invalid PDF values for a merged connection")
        np.add.at(accumulated, inverse[start:stop], pdf * member_weights[start:stop, None])
        np.add.at(normaliser, inverse[start:stop], member_weights[start:stop])

    good = normaliser > 0
    accumulated[good] /= normaliser[good, None]
    covered = np.zeros_like(active)
    for index in np.flatnonzero(good):
        i, j = int(pair_id[index, 0]), int(pair_id[index, 1])
        mixed = _discretize_kernel(accumulated[index], dt_s)
        if active[i, j]:
            phi[:, i, j] = mixed * weights[i, j]
            covered[i, j] = True
        if active[j, i]:
            phi[:, j, i] = mixed * weights[j, i]
            covered[j, i] = True

    # A coupling with no member length behind it has no delay to travel, so
    # it would silently vanish from the model. Say so instead.
    orphaned = int(np.count_nonzero(active & ~covered))
    if orphaned:
        raise ValueError(
            f"{orphaned} connection(s) carry weight but no measured length, so no delay can be "
            "assigned to them; mask the connectivity to the measured connections, or switch off "
            "the length mixture to use one representative length per pair"
        )
    return phi


def _convolve_local_delay(phi: FloatArray, local_mass: FloatArray) -> FloatArray:
    """
    Convolve every connection's lag weights with the local delay mass.

    The convolution runs along the lag axis for all ``N * N`` connections at
    once, as a single banded accumulation: at atlas scale there are tens of
    thousands of connections, and one call per connection dominates the
    build.
    """
    p = phi.shape[0]
    mass = np.asarray(local_mass, dtype=float)
    # Mass pushed beyond the lag horizon is lost, the same truncation the
    # transmission kernel already undergoes.
    convolved = fftconvolve(phi, mass[:, None, None], mode="full", axes=0)[:p]
    # The transform leaves rounding dust of order 1e-16 where the direct sum
    # would give an exact zero; lag weights are non-negative by construction.
    return np.clip(convolved, 0.0, None)


def _add_self_connections(
    phi: FloatArray,
    local_mass: FloatArray,
    self_weight: float,
    has_local_kernel: bool,
) -> FloatArray:
    """Put recurrent self-excitation on the diagonal, delayed by the local kernel."""
    if not has_local_kernel:
        raise ValueError(
            "self_weight requires a local delay kernel: with no local delay the recurrent "
            "input would arrive instantaneously, which a VAR cannot represent"
        )
    # Lag offset 0 is instantaneous and cannot enter the VAR; the remaining
    # mass is renormalised so the self-gain equals self_weight exactly.
    delayed = local_mass[1:]
    total = float(delayed.sum())
    if total <= 0:
        raise ValueError(
            "the local delay kernel puts all of its mass at lag zero, so a self-connection "
            "would be instantaneous; use a longer local delay or a higher sample rate"
        )
    if local_mass[0] > 0.01:
        warnings.warn(
            f"{100 * local_mass[0]:.0f}% of the local delay is shorter than one sample and was "
            "redistributed; raise the sample rate to resolve it",
            stacklevel=3,
        )
    phi = phi.copy()
    n = phi.shape[1]
    diagonal = self_weight * (delayed / total)
    for i in range(n):
        phi[:, i, i] += diagonal
    return phi


def build_var(
    distances: DistanceProvider,
    weights: ConnectivityProvider,
    kernel: DelayKernel,
    params: SimulationParams,
    *,
    transfer_fn: TransferFn | None = None,
    local_kernel: LocalDelayKernel | None = None,
    self_weight: float = 0.0,
    use_length_mixture: bool = True,
    target_spectral_radius: float | None = None,
    enforce_stationarity: bool = True,
    max_shrink_iter: int = 100,
    shrink_factor: float = 0.9,
) -> LinearVAR | NonlinearVAR:
    """
    Build VAR dynamics from anatomical inputs, linear or saturating.

    Parameters
    ----------
    distances, weights, kernel, params
        See :func:`build_lag_tensor`.
    transfer_fn : TransferFn, optional
        Element-wise output transfer. ``None`` or :class:`Identity` gives a
        :class:`LinearVAR` with a closed-form spectrum; any other transfer
        gives a :class:`NonlinearVAR`, whose spectra must be estimated from
        simulated signals.
    local_kernel : LocalDelayKernel, optional
        Receiver-side delay convolved into every incoming connection; see
        :func:`build_lag_tensor`.
    self_weight : float
        Recurrent self-excitation weight; requires ``local_kernel``.
    use_length_mixture : bool
        Mix delay distributions over the member lengths of merged
        connections; see :func:`build_lag_tensor`.
    target_spectral_radius : float, optional
        Scale every coupling by one common factor so the model sits at this
        spectral radius. It fixes the distance from criticality — which is
        what sets how sharp the resonances are — without touching relative
        connectivity or any delay. ``None`` leaves the coupling as the
        components specify it.
    enforce_stationarity : bool
        If ``True``, iteratively shrink the lag tensor until the *linearised*
        VAR is stationary. If ``False``, return the unshrunk tensor.
    max_shrink_iter : int
        Maximum number of stationarity-shrink iterations.
    shrink_factor : float
        Multiplicative factor applied to all lag coefficients per iteration.

    Returns
    -------
    LinearVAR or NonlinearVAR
        Constructed dynamics.

    Notes
    -----
    Stationarity is a property of the linear system. A saturating transfer
    keeps trajectories bounded even beyond that boundary, so
    ``enforce_stationarity=False`` is a legitimate choice for non-linear
    models — and a divergence risk for linear ones.

    ``target_spectral_radius`` above one therefore only makes sense together
    with ``enforce_stationarity=False`` and a saturating transfer; otherwise
    the stationarity shrink immediately undoes it.
    """
    phi = build_lag_tensor(
        distances,
        weights,
        kernel,
        params,
        local_kernel=local_kernel,
        self_weight=self_weight,
        use_length_mixture=use_length_mixture,
    )
    if target_spectral_radius is not None:
        phi, factor = scale_to_spectral_radius(phi, target_spectral_radius)
        warnings.warn(
            f"scaled every coupling by {factor:.4f} to reach a spectral radius of "
            f"{target_spectral_radius:g}; relative connectivity and delays are unchanged",
            stacklevel=2,
        )
    if enforce_stationarity:
        phi, n_iter = shrink_to_stationary(phi, max_iter=max_shrink_iter, factor=shrink_factor)
        if n_iter > 0:
            effective_shrink = shrink_factor**n_iter
            warnings.warn(
                f"shrank phi {n_iter} times to reach stationarity "
                f"(effective coupling reduced by factor {effective_shrink:.4f}); "
                "consider reducing connectivity weights or speed_factor",
                stacklevel=2,
            )
    if transfer_fn is None or isinstance(transfer_fn, Identity):
        return LinearVAR(phi=phi)
    return NonlinearVAR(phi=phi, transfer_fn=transfer_fn)


def build_linear_var(
    distances: DistanceProvider,
    weights: ConnectivityProvider,
    kernel: DelayKernel,
    params: SimulationParams,
    *,
    enforce_stationarity: bool = True,
    max_shrink_iter: int = 100,
    shrink_factor: float = 0.9,
) -> LinearVAR:
    """Build a :class:`LinearVAR` from anatomical inputs.

    Parameters
    ----------
    distances, weights, kernel, params
        See :func:`build_lag_tensor`.
    enforce_stationarity : bool
        If ``True``, iteratively shrink the lag tensor until the VAR is
        stationary. If ``False``, return the unshrunk tensor.
    max_shrink_iter : int
        Maximum number of stationarity-shrink iterations.
    shrink_factor : float
        Multiplicative factor applied to all lag coefficients per iteration.

    Returns
    -------
    LinearVAR
        Constructed dynamics.
    """
    model = build_var(
        distances,
        weights,
        kernel,
        params,
        transfer_fn=None,
        enforce_stationarity=enforce_stationarity,
        max_shrink_iter=max_shrink_iter,
        shrink_factor=shrink_factor,
    )
    assert isinstance(model, LinearVAR)  # noqa: S101 - transfer_fn=None always yields LinearVAR
    return model
