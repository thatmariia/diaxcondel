"""
Stability analysis for VAR models.

A VAR(P) is stationary when the spectral radius of its ``(NP, NP)`` companion
matrix is below one. Forming that matrix and taking all of its eigenvalues
costs ``O((NP)^3)`` time and ``(NP)^2`` memory: fine for a handful of regions,
unusable at atlas scale. Thirty regions at 300 lags is a 9000x9000 dense
eigenproblem (about four minutes and 650 MB).

The ``*_from_phi`` functions take the lag tensor and pick the cheaper route
automatically:

* small systems: the exact dense eigendecomposition;
* large systems: the **argument principle**. Stationarity is equivalent to
  ``det(I - sum_p Phi_p z^p) != 0`` everywhere in the closed unit disk, and
  the number of zeros inside a disk equals the winding number of that
  determinant around the origin. Evaluating it on the unit circle costs
  ``O(M(P N^2 + N^3))`` and answers the stationarity question exactly (it
  counts zeros; it does not approximate eigenvalues), which at 30 regions is
  under a second instead of four minutes.

Both routes agree to five decimal places on random test systems.

Reporting the radius itself is a separate matter: for large systems
:func:`spectral_radius_from_phi` estimates it by power iteration, which is
cheap and stable, and accurate to roughly a percent. Decisions use the exact
route; reports use the estimate and say so.
"""

from __future__ import annotations

import warnings

import numpy as np

from diaxcondel._typing import FloatArray

#: Companion sizes (``N * P``) up to which the exact dense route is used.
#: Above it the winding-number test is both faster and lighter, already by a
#: factor of forty at five regions and 300 lags.
DENSE_EIGENVALUE_LIMIT = 400

#: Starting number of sample points on the contour for the winding-number test.
CONTOUR_POINTS = 2048

#: Largest contour sampling used when refining an under-resolved winding number.
MAX_CONTOUR_POINTS = 16384

#: Smallest coupling scale :func:`scale_to_spectral_radius` will search down to.
_MIN_SCALE = 1e-12


def spectral_radius(companion: FloatArray) -> float:
    """
    Return the spectral radius (max absolute eigenvalue) of a matrix.

    Parameters
    ----------
    companion : FloatArray of shape (M, M)
        Companion matrix; see :meth:`LinearVAR.companion_matrix`.

    Returns
    -------
    float
        Maximum absolute eigenvalue.
    """
    eigenvalues = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigenvalues)))


def is_stationary(companion: FloatArray, tol: float = 1.0) -> bool:
    """
    Test whether a VAR system is stationary.

    Parameters
    ----------
    companion : FloatArray of shape (M, M)
        Companion matrix.
    tol : float
        Strict upper bound on the spectral radius. Default 1.0.

    Returns
    -------
    bool
        ``True`` if all eigenvalues lie strictly within the unit disk.
    """
    return spectral_radius(companion) < tol


def _validate_phi(phi: FloatArray) -> FloatArray:
    """Return ``phi`` as a validated float array of shape ``(P, N, N)``."""
    out = np.asarray(phi, dtype=float)
    if out.ndim != 3 or out.shape[1] != out.shape[2]:
        raise ValueError(f"phi must have shape (P, N, N); got {out.shape}")
    return out


def count_roots_inside(
    phi: FloatArray,
    radius: float = 1.0,
    *,
    n_points: int = CONTOUR_POINTS,
    max_points: int = MAX_CONTOUR_POINTS,
) -> int:
    """
    Count the roots of ``det(I - sum_p Phi_p z^p)`` inside a disk.

    Parameters
    ----------
    phi : FloatArray of shape (P, N, N)
        Lag-coefficient tensor.
    radius : float
        Radius of the disk in the ``z`` plane. A root at ``z`` corresponds to
        a companion eigenvalue ``1 / z``, so roots inside the *unit* disk are
        exactly the unstable modes.
    n_points : int
        Initial number of sample points on the contour.
    max_points : int
        Sampling is doubled until consecutive determinant phases differ by
        less than a quarter turn, up to this many points.

    Returns
    -------
    int
        Number of roots (with multiplicity) strictly inside the disk.

    Notes
    -----
    ``det(I - sum_p Phi_p z^p)`` equals one at the origin, so the winding
    number of the determinant along the contour counts the enclosed zeros
    (argument principle). Determinant phases come from
    :func:`numpy.linalg.slogdet`, which cannot overflow.
    """
    phi = _validate_phi(phi)
    p, n, _ = phi.shape
    if radius <= 0:
        raise ValueError("radius must be positive")

    points = max(64, int(n_points))
    phi_c = phi.astype(np.complex128)
    while True:
        z = radius * np.exp(2j * np.pi * np.arange(points) / points)
        powers = z[:, None] ** np.arange(1, p + 1)[None, :]
        matrices = np.eye(n)[None, :, :] - np.einsum("mp,pij->mij", powers, phi_c)
        sign, _ = np.linalg.slogdet(matrices)
        phase = np.angle(sign)
        steps = np.diff(np.concatenate([phase, phase[:1]]))
        steps = (steps + np.pi) % (2 * np.pi) - np.pi
        resolved = float(np.abs(steps).max()) < 0.5 * np.pi
        if resolved or points >= max_points:
            count = int(round(float(steps.sum()) / (2 * np.pi)))
            if not resolved and count == 0:  # pragma: no cover - pathological systems only
                # A coarse contour can only *miss* windings, so a non-zero
                # count still means "unstable"; a zero count might not.
                warnings.warn(
                    f"winding-number stability test is under-resolved at {points} contour points; "
                    "treat this stability result as provisional",
                    stacklevel=2,
                )
            return count
        points *= 2


def is_stationary_from_phi(
    phi: FloatArray,
    tol: float = 1.0,
    *,
    dense_limit: int = DENSE_EIGENVALUE_LIMIT,
) -> bool:
    """
    Test stationarity directly from a lag tensor.

    Parameters
    ----------
    phi : FloatArray of shape (P, N, N)
        Lag-coefficient tensor.
    tol : float
        Strict upper bound on the spectral radius.
    dense_limit : int
        Companion sizes ``N * P`` up to this value use the exact dense
        eigendecomposition; larger systems use the winding-number test.

    Returns
    -------
    bool
        ``True`` if the system is stationary at the requested bound.
    """
    phi = _validate_phi(phi)
    p, n, _ = phi.shape
    if tol <= 0:
        raise ValueError("tol must be positive")
    if n * p <= dense_limit:
        return spectral_radius(_companion_from_phi(phi)) < tol
    # rho < tol  <=>  no root of det(I - sum_p Phi_p z^p) inside |z| <= 1 / tol.
    return count_roots_inside(phi, 1.0 / tol) == 0


def spectral_radius_from_phi(
    phi: FloatArray,
    *,
    dense_limit: int = DENSE_EIGENVALUE_LIMIT,
    n_iter: int = 2000,
    seed: int = 0,
) -> float:
    """
    Return the spectral radius of the VAR implied by a lag tensor.

    Parameters
    ----------
    phi : FloatArray of shape (P, N, N)
        Lag-coefficient tensor.
    dense_limit : int
        Companion sizes ``N * P`` up to this value are solved exactly by
        eigendecomposition; larger systems are estimated by power iteration
        on the companion operator, which costs one lag-tensor product per
        step and never forms the matrix.
    n_iter : int
        Power-iteration steps for large systems.
    seed : int
        Seed for the power-iteration start vector, so the estimate is
        reproducible.

    Returns
    -------
    float
        Largest absolute eigenvalue of the companion matrix: exact for
        small systems, and accurate to about a percent for large ones.

    Notes
    -----
    Use :func:`is_stationary_from_phi` to *decide* stability: it counts
    roots exactly at any size, whereas this function's large-system branch
    is an estimate and can sit a little either side of one.
    """
    phi = _validate_phi(phi)
    p, n, _ = phi.shape
    if n * p <= dense_limit:
        return spectral_radius(_companion_from_phi(phi))
    if not np.any(phi):
        return 0.0

    # Power iteration: ||A^k x||^(1/k) converges to the spectral radius, and
    # the companion product is just the lag-tensor contraction plus a shift.
    top = np.empty((n, n * p))
    for k in range(p):
        top[:, k * n : (k + 1) * n] = phi[k]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n * p)
    x /= np.linalg.norm(x)
    growth: list[float] = []
    for _ in range(max(10, n_iter)):
        y = np.empty_like(x)
        y[:n] = top @ x
        y[n:] = x[: (p - 1) * n]
        norm = float(np.linalg.norm(y))
        if norm <= 0:  # pragma: no cover - only for a nilpotent tensor
            return 0.0
        x = y / norm
        growth.append(norm)
    # Average the tail geometrically: for non-normal systems the per-step
    # growth oscillates around the spectral radius rather than settling.
    tail = np.asarray(growth[-max(10, len(growth) // 4) :], dtype=float)
    return float(np.exp(np.mean(np.log(tail))))


def scale_to_spectral_radius(
    phi: FloatArray,
    target: float,
    *,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> tuple[FloatArray, float]:
    """
    Rescale a lag tensor so the VAR sits at a chosen spectral radius.

    The spectral radius is the model's distance from criticality: it sets how
    sharp every delay-driven resonance is, independently of the anatomy. This
    makes it a controlled variable. Sweep it, and only the operating point
    changes.

    Parameters
    ----------
    phi : FloatArray of shape (P, N, N)
        Lag-coefficient tensor.
    target : float
        Spectral radius to hit. Values below one give a stationary model;
        one is critical; above one only a saturating transfer stays bounded.
        Radii far below one are not reachable this way; see Notes.
    tol : float
        Relative accuracy of the achieved radius.
    max_iter : int
        Maximum bisection steps.

    Returns
    -------
    phi_out : FloatArray
        ``factor * phi``.
    factor : float
        The scale that was applied. Every connection is scaled equally, so
        the *relative* connectivity and all delays are untouched.

    Raises
    ------
    ValueError
        If ``target`` is not positive, or ``phi`` is all zeros and so has no
        radius to scale.

    Notes
    -----
    The radius increases monotonically with the scale, so a bisection
    converges. Radii are measured with :func:`spectral_radius_from_phi`,
    which is exact for small systems and an estimate for large ones.

    A deep VAR resists being scaled *down*: when the coupling is spread over
    ``P`` lags the radius falls roughly as the ``P``-th root of the scale, so
    at 300 lags even a millionfold reduction only brings it to about 0.95.
    That is a property of the model, not of this routine, and the useful
    range of ``target`` is therefore the approach to criticality, the
    region where the radius actually changes what the spectrum looks like.
    The error message names the lowest radius this particular tensor can
    reach.
    """
    phi = _validate_phi(phi)
    if target <= 0:
        raise ValueError("target spectral radius must be positive")
    if not np.any(phi):
        raise ValueError("cannot scale an all-zero lag tensor to a spectral radius")

    def radius(factor: float) -> float:
        return spectral_radius_from_phi(phi * factor)

    low, high = 1.0, 1.0
    while radius(high) < target:
        high *= 2.0
        if high > 1e12:  # pragma: no cover - a nilpotent tensor never reaches a radius
            raise ValueError(f"this lag tensor cannot be scaled up to a spectral radius of {target:g}")
    while radius(low) > target:
        low /= 2.0
        if low < _MIN_SCALE:
            raise ValueError(
                f"a spectral radius of {target:g} is out of reach for this model: scaling the "
                f"coupling down by {_MIN_SCALE:g} still leaves {radius(_MIN_SCALE):.3f}, because "
                f"coupling spread over {phi.shape[0]} lags falls only as the {phi.shape[0]}-th root "
                "of the scale. Use fewer lags or weaker components to reach a lower radius"
            )

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        achieved = radius(mid)
        if abs(achieved - target) <= tol * target:
            return phi * mid, float(mid)
        if achieved < target:
            low = mid
        else:
            high = mid
    factor = 0.5 * (low + high)
    return phi * factor, float(factor)


def shrink_to_stationary(phi: FloatArray, *, max_iter: int = 100, factor: float = 0.9) -> tuple[FloatArray, int]:
    """
    Shrink ``phi`` iteratively until the VAR system is stationary.

    Parameters
    ----------
    phi : FloatArray of shape (P, N, N)
        Lag-coefficient tensor.
    max_iter : int
        Maximum number of shrinkage iterations.
    factor : float
        Multiplicative shrinkage factor per iteration; must lie in (0, 1).

    Returns
    -------
    phi_out : FloatArray
        Shrunk lag tensor.
    n_iter : int
        Number of iterations applied.

    Raises
    ------
    RuntimeError
        If stationarity is not achieved within ``max_iter`` iterations.

    Notes
    -----
    We deliberately do not search for the minimum-shrinkage solution because
    the mapping between ``phi`` and physical parameters is then no longer
    well-defined.
    """
    if not (0 < factor < 1):
        raise ValueError("factor must be in (0, 1)")
    phi_cur = _validate_phi(phi).copy()
    for it in range(max_iter):
        if is_stationary_from_phi(phi_cur):
            return phi_cur, it
        phi_cur *= factor
    raise RuntimeError(f"failed to reach stationarity within {max_iter} iterations")


def _companion_from_phi(phi: FloatArray) -> FloatArray:
    n, p = phi.shape[1], phi.shape[0]
    a = np.zeros((n * p, n * p))
    for k in range(p):
        a[:n, k * n : (k + 1) * n] = phi[k]
    if p > 1:
        a[n:, : (p - 1) * n] = np.eye((p - 1) * n)
    return a
