"""
Connectivity sources: how strongly regions drive one another.

A connectivity source fills the weight matrix ``gamma``, where entry
``[i, j]`` is the coupling from source ``j`` onto target ``i``. It is chosen
independently of the distances, so empirical geometry can be combined with a
theoretical coupling rule and the other way round.

Every entry receives the model's ``regions`` and its ``distances`` as build
context, which is what lets rules depend on connection length and lets atlas
weights check that they describe the same regions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import numpy as np

from diaxcondel.connectome.atlases.manual import steeghs_2025_weights
from diaxcondel.connectome.distance import DistanceProvider
from diaxcondel.connectome.manual import manual_connectivity as build_manual_connectivity
from diaxcondel.connectome.regions import RegionSet
from diaxcondel.connectome.weights import ManualConnectivity

from ..param import Param
from ..registry import register

_STEEGHS_CODES = ("FL", "PL", "OL", "TL", "T")


def _measured(distances: DistanceProvider) -> np.ndarray:
    """Return the boolean mask of pairs that have a measured connection."""
    matrix = np.asarray(distances.matrix(), dtype=float)
    mask = matrix > 0
    np.fill_diagonal(mask, False)
    return mask


def _normalise_total(weights: np.ndarray, coupling: float) -> np.ndarray:
    """Scale weights so the mean total input per region equals ``coupling``."""
    totals = weights.sum(axis=1)
    active = totals[totals > 0]
    if active.size == 0 or coupling <= 0:
        return weights * 0.0 if coupling <= 0 else weights
    return weights * (coupling / float(active.mean()))


@register(
    "connectivity",
    "steeghs_2025",
    label="Tuned five-lobe weights (Steeghs-Turchina 2025)",
    tags=("reference", "small"),
    reference="Steeghs-Turchina et al. (2025), Fig. 3a",
)
def steeghs_2025_connectivity(
    regions: RegionSet,
    distances: DistanceProvider,
    scale: Annotated[float, Param(minimum=0.0, maximum=3.0, step=0.05)] = 1.0,
) -> ManualConnectivity:
    """
    Use the hand-tuned coupling matrix of the original five-region model.

    Strong cortico-cortical weights (up to 1.0 between parietal and
    occipital) and weak thalamic ones (0.083). Only defined for the five
    regions FL, PL, OL, TL, T.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder).
    distances : DistanceProvider
        Model distances (supplied by the builder; unused here).
    scale : float
        Multiplies every weight. Raising it strengthens coupling and pushes
        the system towards its stability boundary, where spectral peaks
        sharpen.

    Returns
    -------
    ManualConnectivity
        The published weight matrix, scaled.
    """
    if regions.codes != _STEEGHS_CODES:
        raise ValueError(
            "these weights are defined for the five regions "
            f"{list(_STEEGHS_CODES)}, but the distances describe {list(regions.codes)[:6]}; "
            "pick a connectivity rule that works for any regions (uniform, distance decay, manual)"
        )
    return ManualConnectivity(
        regions=regions,
        weights=steeghs_2025_weights().matrix() * scale,
        name="tuned five-lobe weights",
        metadata={"source": "Steeghs-Turchina et al. (2025)", "scale": scale},
    )


@register(
    "connectivity",
    "siibra",
    label="Atlas connectivity (siibra)",
    requires=("siibra",),
    tags=("atlas", "empirical"),
    reference="Domhof et al. (2022) connectomes, via siibra-python",
)
def siibra_connectivity(
    regions: RegionSet,
    distances: DistanceProvider,
    source: Literal["streamline_counts", "functional", "uniform"] = "streamline_counts",
    normalization: Literal["max", "mean", "none"] = "max",
    scale: Annotated[float, Param(minimum=0.0, maximum=3.0, step=0.05)] = 1.0,
    threshold: Annotated[float, Param(minimum=0.0, maximum=1.0, step=0.01)] = 0.0,
    fc_exponent: Annotated[float, Param(minimum=0.1, maximum=4.0, step=0.1, advanced=True)] = 1.5,
    paradigm: Annotated[str, Param(advanced=True)] = "",
) -> ManualConnectivity:
    """
    Coupling measured in the same atlas the distances came from.

    Requires atlas distances: the regions, subject, and merge level of that
    query are replayed so the two matrices describe the same network.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder).
    distances : DistanceProvider
        Model distances (supplied by the builder); must carry an atlas query.
    source : {"streamline_counts", "functional", "uniform"}
        What to use as coupling strength. ``"streamline_counts"`` is the
        number of reconstructed fibres between regions;  ``"functional"`` is
        resting-state correlation raised to a power; ``"uniform"`` gives
        every anatomically connected pair the same weight, which isolates the
        effect of the geometry.
    normalization : {"max", "mean", "none"}
        Rescales the matrix so the strongest (or average) connection equals
        one, which makes ``scale`` the single interpretable coupling knob.
        ``"none"`` keeps raw streamline counts, whose absolute size is
        arbitrary.
    scale : float
        Overall coupling multiplier applied after normalisation.
    threshold : float
        Drops connections weaker than this fraction of the strongest one.
        Sparsifies the network, removing weak links that may be tractography
        noise.
    fc_exponent : float
        Exponent applied to functional connectivity before use. Values above
        one emphasise the strongest correlations.
    paradigm : str
        Which resting-state run to use for functional connectivity; empty
        takes the first available.

    Returns
    -------
    ManualConnectivity
        Atlas weights over exactly the model's regions.
    """
    from diaxcondel.connectome.atlases.siibra_atlas import siibra_weights

    return siibra_weights(
        distances,
        source=source,
        paradigm=paradigm,
        fc_exponent=fc_exponent,
        normalization=normalization,
        scale=scale,
        threshold=threshold,
    )


@register("connectivity", "uniform", label="Uniform coupling", tags=("theory",))
def uniform_connectivity(
    regions: RegionSet,
    distances: DistanceProvider,
    coupling: Annotated[float, Param(minimum=0.0, maximum=2.0, step=0.01)] = 0.9,
    only_measured: Annotated[bool, Param(advanced=True)] = True,
) -> ManualConnectivity:
    """
    Every connection equally strong, with a fixed total input per region.

    The weight of each connection is ``coupling`` divided by the number of
    connections a region has, so the total drive arriving at a region stays
    fixed as the network grows. This is the setting in which geometry alone
    decides the spectrum: all differences between connections come from their
    lengths.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder).
    distances : DistanceProvider
        Model distances (supplied by the builder), used to find which pairs
        are connected at all.
    coupling : float
        Total incoming coupling per region. Values approaching 1 put the
        linearised system close to its stability boundary, where resonances
        are sharp; the builder shrinks the coupling if it goes unstable
        (unless a saturating transfer is used instead).
    only_measured : bool
        Restrict connections to pairs with a positive distance. Turning it
        off would connect every pair, which needs a distance for every pair.

    Returns
    -------
    ManualConnectivity
        Uniform weights.
    """
    n = len(regions)
    mask = _measured(distances) if only_measured else ~np.eye(n, dtype=bool)
    counts = mask.sum(axis=1)
    weights = np.zeros((n, n))
    for i in range(n):
        if counts[i] > 0:
            weights[i, mask[i]] = coupling / counts[i]
    return ManualConnectivity(
        regions=regions,
        weights=weights,
        name="uniform coupling",
        metadata={"source": "rule", "rule": "uniform", "coupling": coupling},
    )


@register("connectivity", "distance_decay", label="Coupling that falls with distance", tags=("theory",))
def distance_decay_connectivity(
    regions: RegionSet,
    distances: DistanceProvider,
    form: Literal["exponential", "power"] = "exponential",
    length_constant_cm: Annotated[float, Param(unit="cm", minimum=0.5, maximum=30.0, step=0.5)] = 8.0,
    exponent: Annotated[float, Param(minimum=0.0, maximum=4.0, step=0.1)] = 1.0,
    coupling: Annotated[float, Param(minimum=0.0, maximum=2.0, step=0.01)] = 0.9,
    max_distance_cm: Annotated[float, Param(unit="cm", minimum=0.0, maximum=50.0, step=1.0)] = 0.0,
) -> ManualConnectivity:
    """
    Nearby regions coupled more strongly than distant ones.

    Anatomical connection strength falls off with distance. This rule imposes
    that fall-off directly, so long connections still carry their long delays
    but contribute less power, the opposite emphasis to uniform coupling.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder).
    distances : DistanceProvider
        Model distances (supplied by the builder).
    form : {"exponential", "power"}
        ``"exponential"`` uses ``exp(-d / length_constant_cm)``, which
        suppresses long connections sharply; ``"power"`` uses
        ``d ** -exponent``, which keeps a heavier tail.
    length_constant_cm : float
        Distance over which exponential coupling falls by a factor of e.
    exponent : float
        Exponent of the power-law fall-off. ``0`` reproduces uniform
        coupling; ``2`` is a steep fall-off.
    coupling : float
        Total incoming coupling per region after the fall-off is applied, so
        this stays comparable with the uniform rule.
    max_distance_cm : float
        Cut connections longer than this. ``0`` keeps them all. Useful for
        asking which frequencies disappear when long-range fibres are
        removed.

    Returns
    -------
    ManualConnectivity
        Distance-dependent weights.
    """
    matrix = np.asarray(distances.matrix(), dtype=float)
    mask = _measured(distances)
    if max_distance_cm > 0:
        mask = mask & (matrix <= max_distance_cm)
    weights = np.zeros_like(matrix)
    if form == "exponential":
        weights[mask] = np.exp(-matrix[mask] / length_constant_cm)
    elif form == "power":
        weights[mask] = np.power(matrix[mask], -exponent)
    else:  # pragma: no cover - Literal guards this
        raise ValueError(f"unknown fall-off form {form!r}")
    weights = _normalise_total(weights, coupling)
    return ManualConnectivity(
        regions=regions,
        weights=weights,
        name=f"{form} distance decay",
        metadata={
            "source": "rule",
            "rule": f"{form} decay",
            "length_constant_cm": length_constant_cm,
            "exponent": exponent,
            "coupling": coupling,
            "max_distance_cm": max_distance_cm or None,
        },
    )


@register("connectivity", "random_sparse", label="Random sparse coupling", tags=("theory",))
def random_sparse_connectivity(
    regions: RegionSet,
    distances: DistanceProvider,
    density: Annotated[float, Param(minimum=0.01, maximum=1.0, step=0.01)] = 0.3,
    spread: Annotated[float, Param(minimum=0.0, maximum=2.0, step=0.05)] = 0.5,
    coupling: Annotated[float, Param(minimum=0.0, maximum=2.0, step=0.01)] = 0.9,
    seed: int = 20260505,
) -> ManualConnectivity:
    """
    Connect a random subset of pairs, with log-normally distributed strengths.

    Real connectomes are sparse and their weights span orders of magnitude.
    This rule reproduces both features without committing to an atlas, which
    makes it a useful null model.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder).
    distances : DistanceProvider
        Model distances (supplied by the builder), used to find which pairs
        could be connected.
    density : float
        Fraction of the possible connections that exist.
    spread : float
        Standard deviation of ``log`` weight. ``0`` makes every surviving
        connection equally strong; ``1`` spreads strengths over roughly two
        orders of magnitude.
    coupling : float
        Total incoming coupling per region after the draw.
    seed : int
        Seed for the draw.

    Returns
    -------
    ManualConnectivity
        Sparse, symmetric weights.
    """
    rng = np.random.default_rng(seed)
    mask = _measured(distances)
    n = len(regions)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1) & mask
    keep = upper & (rng.random((n, n)) < density)
    weights = np.zeros((n, n))
    draws = rng.lognormal(mean=0.0, sigma=spread, size=(n, n))
    weights[keep] = draws[keep]
    weights = weights + weights.T
    weights = _normalise_total(weights, coupling)
    return ManualConnectivity(
        regions=regions,
        weights=weights,
        name="random sparse",
        metadata={"source": "rule", "rule": "random sparse", "density": density, "spread": spread, "seed": seed},
    )


@register("connectivity", "from_file", label="Load weights from a file", tags=("empirical", "manual"))
def file_connectivity_source(
    regions: RegionSet,
    path: Annotated[str, Param(widget="text")] = "",
    scale: Annotated[float, Param(minimum=0.0, maximum=100.0, step=0.1)] = 1.0,
    normalization: Annotated[Literal["none", "max", "mean"], Param(advanced=True)] = "none",
    symmetrize: Annotated[bool, Param(advanced=True)] = False,
) -> ManualConnectivity:
    """
    Read coupling weights from your own file.

    The companion to loading distances from a file: point at a weight matrix
    over the same regions, in the same order. It must be square and match the
    number of regions the distance source produced.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder).
    path : str
        Path to the weight matrix: ``.npy``/``.npz``, or delimited text with
        one row per line. Entry ``[i, j]`` is the drive from region ``j`` to
        region ``i``.
    scale : float
        Multiplier applied to every weight. Coupling strength sets how close
        the model runs to instability, so this is the usual knob for moving
        the operating point when the file's units are arbitrary.
    normalization : {"none", "max", "mean"}
        Divide by the largest or the mean weight before scaling, which makes
        ``scale`` mean the same thing across matrices of different units.
    symmetrize : bool
        Average the matrix with its transpose. Leave it off for a directed
        connectome.

    Returns
    -------
    ManualConnectivity
        Weights over the model's regions, with the diagonal cleared.
    """
    from diaxcondel.connectome.manual import load_matrix

    if not path.strip():
        raise ValueError("give the path to a weight-matrix file")
    weights = load_matrix(path)
    n = len(regions)
    if weights.shape[0] != n:
        raise ValueError(
            f"the file holds a {weights.shape[0]}x{weights.shape[0]} matrix but the model has {n} regions; "
            "load the distances and the weights from matching files"
        )
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if symmetrize:
        weights = 0.5 * (weights + weights.T)
    if normalization == "max" and weights.max() > 0:
        weights = weights / weights.max()
    elif normalization == "mean" and weights.mean() > 0:
        weights = weights / weights.mean()
    weights = weights * float(scale)
    np.fill_diagonal(weights, 0.0)
    return ManualConnectivity(
        regions=regions,
        weights=weights,
        name=f"file: {Path(path).name}",
        metadata={"source": str(path), "scale": scale, "normalization": normalization},
    )


@register("connectivity", "manual", label="Type in the weights", tags=("manual",))
def manual_connectivity_source(
    regions: RegionSet,
    distances: DistanceProvider,
    weights: Annotated[str, Param(widget="matrix")] = "",
    symmetrize: Annotated[bool, Param(advanced=True)] = False,
) -> ManualConnectivity:
    """
    Coupling weights entered by hand.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder); their number fixes the
        matrix size and their order its rows and columns.
    distances : DistanceProvider
        Model distances (supplied by the builder; unused here).
    weights : str
        The weight matrix as JSON (``[[0, 0.5], [0.5, 0]]``) or rows of
        numbers. Entry ``[i, j]`` is the coupling from region ``j`` onto
        region ``i``, so rows are targets. Empty starts from all zeros.
        Weights are dimensionless: 1.0 is a strong connection, and the whole
        matrix is what the stationarity check scales if the network runs hot.
    symmetrize : bool
        Average the matrix with its transpose, making every connection
        reciprocal.

    Returns
    -------
    ManualConnectivity
        Validated weights with a zero diagonal.
    """
    return build_manual_connectivity(regions, weights, symmetrize=symmetrize)
