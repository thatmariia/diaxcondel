"""
Network layout: seeing the model's geometry rather than its matrix.

A distance matrix is hard to read as a picture of a network. Placing the
regions so that their drawn separations approximate their conduction
distances turns it into one — which regions sit far from everything, which
cluster, and where the long connections run.

Two mature pieces of scipy do the work: shortest paths fill in the distance
between regions that are not directly connected, and an eigendecomposition
gives the classical multidimensional-scaling embedding of the completed
matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import shortest_path

from diaxcondel._typing import FloatArray


def complete_distances(distances_cm: FloatArray) -> tuple[FloatArray, bool]:
    """
    Fill in distances between regions with no direct connection.

    Parameters
    ----------
    distances_cm : FloatArray of shape (N, N)
        Distances, where zero off the diagonal means "no direct connection".

    Returns
    -------
    completed : FloatArray of shape (N, N)
        Distances with missing pairs replaced by the shortest path through
        the network. Pairs in different components fall back to the longest
        finite path, so a layout can still be drawn.
    connected : bool
        Whether the network was connected to begin with.
    """
    matrix = np.asarray(distances_cm, dtype=float)
    graph = np.where(matrix > 0, matrix, np.inf)
    np.fill_diagonal(graph, 0.0)
    geodesic = np.asarray(shortest_path(graph, method="D", directed=False), dtype=float)
    finite = np.isfinite(geodesic)
    connected = bool(finite.all())
    if not connected:
        fallback = float(geodesic[finite].max()) if finite.any() else 1.0
        geodesic = np.where(finite, geodesic, fallback * 1.5)
    return geodesic, connected


def stress_layout(
    distances_cm: FloatArray,
    *,
    min_separation: float = 0.0,
    iterations: int = 200,
    n_dimensions: int = 2,
) -> tuple[FloatArray, float]:
    """
    Lay out regions by stress majorisation, with a floor on separations.

    Classical scaling gets the large-scale geometry right but piles closely
    connected regions on top of each other, which makes a labelled figure
    unreadable. Imposing a minimum target separation and then minimising the
    stress spreads those regions apart while leaving the rest of the layout
    where the distances put it.

    Parameters
    ----------
    distances_cm : FloatArray of shape (N, N)
        Distances between regions in centimetres.
    min_separation : float
        Smallest target separation, in centimetres. Pairs closer than this
        are drawn at this separation instead; ``0`` reproduces plain scaling.
    iterations : int
        Number of majorisation steps.
    n_dimensions : int
        Dimensionality of the embedding.

    Returns
    -------
    coordinates : FloatArray of shape (N, n_dimensions)
        Layout coordinates.
    distortion : float
        Median relative difference between drawn and true separations, over
        pairs with a measured distance — how much the picture lies.
    """
    completed, _ = complete_distances(distances_cm)
    n = completed.shape[0]
    if n < 2:
        return np.zeros((n, n_dimensions)), 0.0

    targets = np.maximum(completed, min_separation)
    np.fill_diagonal(targets, 0.0)
    coordinates = mds_layout(targets, n_dimensions=n_dimensions)

    for _ in range(max(0, iterations)):
        difference = coordinates[:, None, :] - coordinates[None, :, :]
        drawn = np.linalg.norm(difference, axis=-1)
        np.fill_diagonal(drawn, np.inf)
        ratio = np.divide(targets, drawn, out=np.zeros_like(targets), where=np.isfinite(drawn))
        weights = -ratio
        np.fill_diagonal(weights, 0.0)
        np.fill_diagonal(weights, -weights.sum(axis=1))
        coordinates = (weights @ coordinates) / n

    difference = coordinates[:, None, :] - coordinates[None, :, :]
    drawn = np.linalg.norm(difference, axis=-1)
    measured = np.asarray(distances_cm, dtype=float) > 0
    distortion = (
        float(np.median(np.abs(drawn[measured] - distances_cm[measured]) / distances_cm[measured]))
        if np.any(measured)
        else 0.0
    )
    return coordinates, distortion


def mds_layout(distances_cm: FloatArray, *, n_dimensions: int = 2) -> FloatArray:
    """
    Place regions so drawn separations approximate their distances.

    Parameters
    ----------
    distances_cm : FloatArray of shape (N, N)
        Distances between regions in centimetres.
    n_dimensions : int
        Dimensionality of the embedding.

    Returns
    -------
    FloatArray of shape (N, n_dimensions)
        Coordinates from classical multidimensional scaling. The axes carry
        no anatomical meaning: only the *relative* separations do, and even
        those are approximate, because a set of brain-wide fibre lengths
        does not embed exactly in a plane.
    """
    completed, _ = complete_distances(distances_cm)
    n = completed.shape[0]
    if n < 2:
        return np.zeros((n, n_dimensions))
    squared = completed**2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:n_dimensions]
    scales = np.sqrt(np.clip(values[order], 0.0, None))
    return vectors[:, order] * scales
