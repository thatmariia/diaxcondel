"""
Distance sources: where the model's regions and their separations come from.

A distance source defines the model's region set and the conduction distance
(in centimetres) between every pair. Together with the delay kernel it fixes
the delays, and delays are what this model turns into spectral structure, so
this is the choice with the most consequences.

Distances and connectivity are chosen separately: any source here can be
paired with any entry in :mod:`diaxcondel.catalog.entries.connectivity`,
provided the connectivity rule can describe these regions (atlas weights need
atlas regions; uniform, distance-decay, and manual weights fit anything).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import numpy as np

from diaxcondel.connectome.atlases.manual import steeghs_2025_distances, steeghs_2025_regions
from diaxcondel.connectome.atlases.synthetic import PowerLawDistances
from diaxcondel.connectome.distance import ManualDistances
from diaxcondel.connectome.manual import manual_distances as build_manual_distances
from diaxcondel.connectome.regions import RegionSet

from ..param import Param
from ..registry import register


@register(
    "distances",
    "steeghs_2025",
    label="Five lobes (Steeghs-Turchina 2025)",
    tags=("reference", "small"),
    reference="Steeghs-Turchina et al. (2025), NeuroImage 321:121418",
)
def steeghs_2025_distance_source(
    scale: Annotated[float, Param(minimum=0.2, maximum=3.0, step=0.05)] = 1.0,
) -> ManualDistances:
    """
    Streamline lengths between frontal, parietal, occipital, temporal lobes and thalamus.

    The five-region geometry of the original paper, with distances between
    5 and 9.5 cm.

    Parameters
    ----------
    scale : float
        Stretches or shrinks the whole geometry. Delays scale with it, so
        halving the distances doubles every resonance frequency; useful for
        asking how much of a spectral feature is set by brain size alone.

    Returns
    -------
    ManualDistances
        Five regions with distances in centimetres.
    """
    base = steeghs_2025_distances()
    return ManualDistances(
        regions=steeghs_2025_regions(),
        matrix_cm=base.matrix() * scale,
        name="five lobes",
        metadata={
            "source": "Steeghs-Turchina et al. (2025)",
            "measure": "streamline length",
            "units": "cm",
            "scale": scale,
        },
    )


@register(
    "distances",
    "siibra",
    label="Atlas streamline lengths (siibra)",
    requires=("siibra",),
    tags=("atlas", "empirical"),
    reference="Domhof et al. (2022) tractography, via siibra-python",
)
def siibra_distance_source(
    parcellation: Annotated[
        str,
        Param(choices=("julich 3.1", "julich 3.0.3", "julich 2.9"), open_choices=True, widget="select"),
    ] = "julich 3.1",
    nodes: Annotated[tuple[str, ...], Param(widget="hidden")] = (),
    merge_level: Annotated[int | None, Param(minimum=0, maximum=8, widget="hidden")] = 3,
    split_hemispheres: bool = False,
    keep_parts: Annotated[tuple[str, ...], Param(widget="hidden")] = (),
    cohort: Annotated[str, Param(choices=("HCP", "1000BRAINS"), open_choices=True, widget="select")] = "HCP",
    subject: str = "",
    subject_aggregation: Literal["mean", "median", "trimmed_mean"] = "mean",
    member_weighting: Annotated[Literal["streamline_counts", "uniform"], Param(advanced=True)] = "streamline_counts",
    drop_isolated: Annotated[bool, Param(advanced=True)] = True,
    trim_fraction: Annotated[float, Param(minimum=0.0, maximum=0.49, step=0.05, advanced=True)] = 0.1,
) -> ManualDistances:
    """
    Mean streamline lengths between atlas regions, in centimetres.

    Tractography in a cohort of healthy subjects gives the length of the
    fibre bundle connecting each pair of atlas regions. Pairs with no
    streamlines have no connection at all in the model.

    Parameters
    ----------
    parcellation : str
        Which brain parcellation to use. Julich-Brain v3.1 has 414 regions
        (266 cortical areas plus thalamic and other subcortical parcels);
        v3.0.3 has 314.
    nodes : tuple of str
        The regions to model, named one by one from anywhere in the atlas
        tree: ``("frontal lobe", "parietal lobe", "thalamus")`` gives three
        nodes (two lobes and the whole thalamus) and drops everything
        else. Append a depth to split one of them further, as
        ``"frontal lobe:1"``. Leave it empty to take the whole brain at one
        level instead (see ``merge_level``).
    merge_level : int
        How far down the atlas's own region tree to keep detail, when no
        regions are named above. Level 0 is one node for the whole brain;
        each level splits it further. For Julich-Brain, level 2 separates
        cortex from subcortical structures, level 3 gives lobes, level 4
        gyri, level 5 cytoarchitectonic areas. Leave it unset to keep every
        region, which is usually too large to simulate. ``nodes`` says the
        same thing and more, so the playground offers only that; this stays
        for scripts that want one level everywhere.
    split_hemispheres : bool
        Keep left and right versions of a region as separate nodes. Doubles
        the node count and lets interhemispheric delays act.
    keep_parts : tuple of str
        Which parts of the brain to model, named as the atlas names them:
        ``"cerebral cortex"`` for a cortex-only model. Naming the regions in
        ``nodes`` filters as well as groups, so this is for scripts that want
        to filter and merge by level separately.
    cohort : str
        Which subject cohort the tractography comes from.
    subject : str
        A single subject id (``"000"``, ``"001"``, ...) to model one
        participant. Empty combines the whole cohort, which smooths over
        individual variation.
    subject_aggregation : {"mean", "median", "trimmed_mean"}
        How subject matrices are combined when no single subject is chosen.
        The median and trimmed mean are less sensitive to outlier subjects.
    member_weighting : {"streamline_counts", "uniform"}
        When regions are merged, a pathway between two merged regions is made
        of many finer connections of different lengths. This sets how much
        each of them contributes to the merged pathway's delay distribution:
        in proportion to the streamlines it carries, or equally. It changes
        no physical quantity (not speed, not calibre), only how much of the
        signal travels each distance.
    drop_isolated : bool
        Remove regions the tractography never reaches; they would otherwise
        sit in the model as silent nodes.
    trim_fraction : float
        Fraction of subjects trimmed from each tail before averaging, when
        ``subject_aggregation`` is ``"trimmed_mean"``.

    Returns
    -------
    ManualDistances
        Distances in centimetres, tagged with the atlas query so matching
        atlas connectivity can be loaded for the same regions.
    """
    from diaxcondel.connectome.atlases.siibra_atlas import siibra_distances

    return siibra_distances(
        parcellation,
        cohort=cohort,
        subject=subject,
        subject_aggregation=subject_aggregation,
        trim_fraction=trim_fraction,
        nodes=nodes,
        merge_level=merge_level,
        split_hemispheres=split_hemispheres,
        keep_parts=keep_parts,
        member_weighting=member_weighting,
        drop_isolated=drop_isolated,
    )


@register(
    "distances",
    "power_law",
    label="Random network, power-law lengths",
    tags=("synthetic", "theory"),
)
def power_law_distance_source(
    n_regions: Annotated[int, Param(minimum=2, maximum=200)] = 12,
    beta: Annotated[float, Param(minimum=-3.0, maximum=2.0, step=0.1)] = -1.0,
    d_min_cm: Annotated[float, Param(unit="cm", minimum=0.1, maximum=20.0, step=0.1)] = 0.5,
    d_max_cm: Annotated[float, Param(unit="cm", minimum=0.2, maximum=40.0, step=0.5)] = 25.0,
    seed: int = 20260505,
) -> ManualDistances:
    """
    Fully connected network whose connection lengths follow a power law.

    Every pair gets a length drawn independently from ``p(d) ~ d ** beta`` on
    ``[d_min_cm, d_max_cm]``. Because delay is proportional to length, the
    exponent controls how the network's delays, and hence its resonances,
    are spread across the frequency axis.

    Parameters
    ----------
    n_regions : int
        Number of nodes. Cost grows with the square of this number.
    beta : float
        Exponent of the length distribution. ``-1`` is log-uniform: equal
        probability per decade of length, so short and long connections are
        equally represented on a log scale. ``0`` is uniform in length,
        weighting long connections more. Positive values concentrate the
        network on its longest connections.
    d_min_cm, d_max_cm
        Shortest and longest possible connection, in centimetres. Their ratio
        sets how many octaves of delay the network spans; the absolute values
        set where that band sits.
    seed : int
        Seed for the draw. The same seed always gives the same network.

    Returns
    -------
    ManualDistances
        Synthetic distances in centimetres.
    """
    if d_min_cm >= d_max_cm:
        raise ValueError(f"d_min_cm ({d_min_cm}) must be smaller than d_max_cm ({d_max_cm})")
    provider = PowerLawDistances(n_regions=n_regions, beta=beta, d_min=d_min_cm, d_max=d_max_cm, rng=seed)
    return ManualDistances(
        regions=provider.regions,
        matrix_cm=provider.matrix(),
        name=f"power law (beta={beta:g})",
        metadata={
            "source": "synthetic",
            "law": "p(d) ~ d^beta",
            "beta": beta,
            "d_min_cm": d_min_cm,
            "d_max_cm": d_max_cm,
            "seed": seed,
            "units": "cm",
        },
    )


@register(
    "distances",
    "power_law_with_hub",
    label="Power-law cortex + subcortical hub",
    tags=("synthetic", "theory"),
)
def power_law_hub_distance_source(
    n_regions: Annotated[int, Param(minimum=2, maximum=200)] = 12,
    beta: Annotated[float, Param(minimum=-3.0, maximum=2.0, step=0.1)] = -1.0,
    d_min_cm: Annotated[float, Param(unit="cm", minimum=0.1, maximum=20.0, step=0.1)] = 0.5,
    d_max_cm: Annotated[float, Param(unit="cm", minimum=0.2, maximum=40.0, step=0.5)] = 25.0,
    hub_distance_cm: Annotated[float, Param(unit="cm", minimum=0.5, maximum=20.0, step=0.5)] = 6.0,
    hub_spread_cm: Annotated[float, Param(unit="cm", minimum=0.0, maximum=10.0, step=0.5)] = 1.0,
    hub_code: str = "HUB",
    seed: int = 20260505,
) -> ManualDistances:
    """
    Synthetic cortex with power-law lengths, plus one hub connected to every node.

    A cortical sheet whose pairwise lengths span a wide range, plus a single
    subcortical station (a thalamus stand-in) sitting at a well-defined
    distance from all of it. The cortical part spreads delays widely; the hub
    adds one sharply defined delay shared by every region.

    Parameters
    ----------
    n_regions : int
        Number of cortical nodes; the hub is added on top.
    beta : float
        Exponent of the cortical length distribution (see the power-law
        source).
    d_min_cm, d_max_cm
        Shortest and longest cortico-cortical connection, in centimetres.
    hub_distance_cm : float
        Mean distance from the hub to each cortical node.
    hub_spread_cm : float
        Standard deviation of that distance across nodes. Zero puts every
        node at exactly the same distance from the hub.
    hub_code : str
        Region code for the hub, e.g. ``"T"`` for thalamus. Use it as the
        relay region to route cortico-cortical traffic through the hub.
    seed : int
        Seed for both the cortical draw and the hub distances.

    Returns
    -------
    ManualDistances
        ``n_regions + 1`` regions, hub last.
    """
    if d_min_cm >= d_max_cm:
        raise ValueError(f"d_min_cm ({d_min_cm}) must be smaller than d_max_cm ({d_max_cm})")
    provider = PowerLawDistances(n_regions=n_regions, beta=beta, d_min=d_min_cm, d_max=d_max_cm, rng=seed)
    cortex = provider.matrix()
    rng = np.random.default_rng(seed + 1)
    hub = np.abs(rng.normal(hub_distance_cm, hub_spread_cm, size=n_regions)) if hub_spread_cm > 0 else None
    hub_distances = hub if hub is not None else np.full(n_regions, hub_distance_cm)
    hub_distances = np.clip(hub_distances, 0.1, None)

    n = n_regions + 1
    matrix = np.zeros((n, n))
    matrix[:n_regions, :n_regions] = cortex
    matrix[:n_regions, n_regions] = hub_distances
    matrix[n_regions, :n_regions] = hub_distances

    codes = [*provider.regions.codes, hub_code]
    return ManualDistances(
        regions=RegionSet.from_codes(codes),
        matrix_cm=matrix,
        name=f"power law + {hub_code}",
        metadata={
            "source": "synthetic",
            "law": "p(d) ~ d^beta plus one hub",
            "beta": beta,
            "hub_code": hub_code,
            "hub_distance_cm": hub_distance_cm,
            "seed": seed,
            "units": "cm",
        },
    )


@register("distances", "ring", label="Ring of evenly spaced regions", tags=("synthetic", "theory"))
def ring_distance_source(
    n_regions: Annotated[int, Param(minimum=3, maximum=200)] = 16,
    circumference_cm: Annotated[float, Param(unit="cm", minimum=5.0, maximum=200.0, step=5.0)] = 60.0,
) -> ManualDistances:
    """
    Regions equally spaced on a ring, with distances measured along it.

    A geometry with no randomness: distances take only ``n_regions / 2``
    distinct values, evenly spaced from the nearest-neighbour separation up
    to half the circumference. Useful as a controlled comparison for networks
    whose length distribution is broad or irregular.

    Parameters
    ----------
    n_regions : int
        Number of nodes on the ring.
    circumference_cm : float
        Length of the ring in centimetres; the human cortex unfolded is
        roughly 60–80 cm around.

    Returns
    -------
    ManualDistances
        Ring distances in centimetres.
    """
    if circumference_cm <= 0:
        raise ValueError("circumference_cm must be positive")
    index = np.arange(n_regions)
    steps = np.abs(index[:, None] - index[None, :])
    steps = np.minimum(steps, n_regions - steps)
    matrix = steps * (circumference_cm / n_regions)
    return ManualDistances(
        regions=RegionSet.from_codes([f"R{i + 1:02d}" for i in range(n_regions)]),
        matrix_cm=matrix.astype(float),
        name="ring",
        metadata={"source": "synthetic", "geometry": "ring", "circumference_cm": circumference_cm, "units": "cm"},
    )


@register("distances", "lattice", label="Regular grid of regions", tags=("synthetic", "theory"))
def lattice_distance_source(
    rows: Annotated[int, Param(minimum=1, maximum=20)] = 4,
    columns: Annotated[int, Param(minimum=1, maximum=20)] = 4,
    spacing_cm: Annotated[float, Param(unit="cm", minimum=0.1, maximum=10.0, step=0.1)] = 2.0,
) -> ManualDistances:
    """
    Regions on a regular grid, with straight-line distances between them.

    The geometry neural-field models assume: a flat sheet of tissue where
    distance, and therefore delay, grows smoothly with separation. Unlike a
    ring it has a boundary and two dimensions, so the delay spectrum is
    richer and edge regions differ from interior ones.

    Parameters
    ----------
    rows, columns : int
        Grid size. Their product is the number of regions, which is also the
        model's dimension; keep it small enough to simulate.
    spacing_cm : float
        Centre-to-centre distance between neighbouring grid points. It scales
        every delay, and so every resonance frequency, inversely.

    Returns
    -------
    ManualDistances
        Euclidean distances between grid points, in centimetres.
    """
    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be at least 1")
    if rows * columns < 2:
        raise ValueError("a model needs at least two regions")
    if spacing_cm <= 0:
        raise ValueError("spacing_cm must be positive")
    grid = np.array([(r, c) for r in range(rows) for c in range(columns)], dtype=float) * spacing_cm
    matrix = np.linalg.norm(grid[:, None, :] - grid[None, :, :], axis=-1)
    codes = [f"R{r + 1}C{c + 1}" for r in range(rows) for c in range(columns)]
    return ManualDistances(
        regions=RegionSet.from_codes(codes),
        matrix_cm=matrix,
        name=f"{rows}x{columns} lattice",
        metadata={
            "source": "synthetic",
            "geometry": "lattice",
            "rows": rows,
            "columns": columns,
            "spacing_cm": spacing_cm,
            "units": "cm",
        },
    )


@register("distances", "from_file", label="Load distances from a file", tags=("empirical", "manual"))
def file_distance_source(
    path: Annotated[str, Param(widget="text")] = "",
    codes_path: Annotated[str, Param(widget="text")] = "",
    scale_to_cm: Annotated[float, Param(minimum=1e-4, maximum=1000.0, advanced=True)] = 1.0,
    symmetrize: Annotated[bool, Param(advanced=True)] = True,
) -> ManualDistances:
    """
    Read connection lengths from your own file.

    For a connectome the package cannot fetch: export the length matrix from
    whatever produced it and point here. The specification stores the path,
    not the numbers, so a saved experiment stays small and still says exactly
    which data it used.

    Parameters
    ----------
    path : str
        Path to the length matrix: ``.npy``/``.npz``, or delimited text
        (comma, tab, or whitespace) with one row per line. Entry ``[i, j]``
        is the length of the connection between regions ``i`` and ``j``; a
        zero off the diagonal means no connection.
    codes_path : str
        Optional path to region names, one per line or comma-separated.
        Empty names the regions ``R1, R2, ...``.
    scale_to_cm : float
        Multiplier turning the file's units into centimetres: ``0.1`` for
        millimetres, ``1`` if the file is already in centimetres. Getting it
        wrong scales every delay, so check the reported median length.
    symmetrize : bool
        Average the matrix with its transpose, which fills in a matrix given
        only as an upper triangle.

    Returns
    -------
    ManualDistances
        Distances in centimetres.
    """
    from diaxcondel.connectome.manual import load_codes, load_matrix

    if not path.strip():
        raise ValueError("give the path to a distance-matrix file")
    matrix = load_matrix(path) * float(scale_to_cm)
    if np.any(matrix < 0):
        raise ValueError("distances must be non-negative")
    if symmetrize:
        matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    codes = load_codes(codes_path) if codes_path.strip() else tuple(f"R{i + 1}" for i in range(matrix.shape[0]))
    if len(codes) != matrix.shape[0]:
        raise ValueError(f"{len(codes)} region codes for a {matrix.shape[0]}x{matrix.shape[0]} matrix")
    return ManualDistances(
        regions=RegionSet.from_codes(list(codes)),
        matrix_cm=matrix,
        name=f"file: {Path(path).name}",
        metadata={
            "source": str(path),
            "measure": "connection length",
            "units": "cm",
            "scale_to_cm": scale_to_cm,
            "median_length_cm": float(np.median(matrix[matrix > 0])) if np.any(matrix > 0) else 0.0,
        },
    )


@register("distances", "manual", label="Type in the distances", tags=("manual",))
def manual_distance_source(
    matrix_cm: Annotated[str, Param(widget="matrix", unit="cm")] = "[[0, 8], [8, 0]]",
    codes: Annotated[str, Param(widget="text")] = "",
    symmetrize: Annotated[bool, Param(advanced=True)] = True,
) -> ManualDistances:
    """
    Distances entered by hand, for a connectome from a paper or a sketch.

    Parameters
    ----------
    matrix_cm : str
        The distance matrix in centimetres, as JSON (``[[0, 8], [8, 0]]``) or
        as rows of numbers. Entry ``[i, j]`` is the length of the connection
        between region ``i`` and region ``j``; a zero off the diagonal means
        those regions are not connected at all.
    codes : str
        Comma-separated region names, e.g. ``"FL, PL, OL"``. Leave empty to
        get ``R1, R2, ...``. They must match the matrix size, and they are
        what you refer to when targeting a stimulus or picking a relay.
    symmetrize : bool
        Average the matrix with its transpose, so filling in only the upper
        triangle is enough.

    Returns
    -------
    ManualDistances
        Validated distances in centimetres.
    """
    return build_manual_distances(matrix_cm, codes, symmetrize=symmetrize)
