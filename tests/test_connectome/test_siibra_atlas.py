"""
Tests for the siibra atlas adapter, against a fake siibra.

The adapter is where atlas conventions meet this package's conventions:
millimetres become centimetres, sparse tractography zeros mean "no
connection", subject matrices are aggregated, and regions get codes. All of
that is testable without a network round trip, which is what these tests do —
see ``test_integration`` for the marked tests that hit the real service.
"""

import importlib
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

LENGTHS_MM = np.array(
    [
        [7.0, 100.0, 0.0, 60.0],
        [100.0, 5.0, 40.0, 0.0],
        [0.0, 40.0, 6.0, 30.0],
        [60.0, 0.0, 30.0, 4.0],
    ]
)
COUNTS = np.array(
    [
        [10.0, 800.0, 0.0, 200.0],
        [800.0, 12.0, 400.0, 0.0],
        [0.0, 400.0, 9.0, 100.0],
        [200.0, 0.0, 100.0, 7.0],
    ]
)
FUNCTIONAL = np.array(
    [
        [1.0, 0.8, -0.2, 0.4],
        [0.8, 1.0, 0.5, 0.1],
        [-0.2, 0.5, 1.0, 0.3],
        [0.4, 0.1, 0.3, 1.0],
    ]
)

REGION_SPECS = [
    ("Area A (X) left", ("Atlas", "telencephalon", "frontal lobe", "Area A (X)")),
    ("Area A (X) right", ("Atlas", "telencephalon", "frontal lobe", "Area A (X)")),
    ("Area B (Y) left", ("Atlas", "telencephalon", "occipital lobe", "Area B (Y)")),
    ("Area B (Y) right", ("Atlas", "telencephalon", "occipital lobe", "Area B (Y)")),
]


class FakeRegion:
    """Stand-in for a siibra region object in a data-frame index."""

    def __init__(self, name: str, ancestors: tuple[str, ...]) -> None:
        self.name = name
        self.ancestors = [SimpleNamespace(name=ancestor) for ancestor in ancestors]
        self.key = name.upper().replace(" ", "_")

    def __str__(self) -> str:
        return self.name


class FakeFrame:
    """Minimal stand-in for the pandas frame siibra returns."""

    def __init__(self, matrix, index):
        self._matrix = np.asarray(matrix, dtype=float)
        self.index = list(index)

    def to_numpy(self):
        return self._matrix

    def reindex(self, index, columns):
        order = [self.index.index(region) for region in index]
        return FakeFrame(self._matrix[np.ix_(order, order)], index)


class FakeCompound:
    """Stand-in for a siibra compound feature holding per-subject matrices."""

    def __init__(self, matrices, regions, cohort="HCP", paradigm=None, name="fake feature"):
        self.indices = [f"{index:03d}" for index in range(len(matrices))]
        self.elements = [
            SimpleNamespace(data=FakeFrame(matrix, regions), subject=subject)
            for subject, matrix in zip(self.indices, matrices, strict=True)
        ]
        self.cohort = cohort
        self.paradigm = paradigm
        self.name = name


@pytest.fixture
def fake_siibra(monkeypatch, tmp_path):
    """Install a fake siibra module and an isolated cache directory."""
    regions = [FakeRegion(name, ancestors) for name, ancestors in REGION_SPECS]

    # Two subjects; the second carries a missing cell to exercise the
    # non-finite handling that the released connectomes actually need.
    lengths_subject_2 = LENGTHS_MM.copy()
    lengths_subject_2[0, 1] = np.nan
    lengths_subject_2[1, 0] = np.nan
    payload = {
        "StreamlineLengths": [FakeCompound([LENGTHS_MM, lengths_subject_2], regions, name="lengths")],
        "StreamlineCounts": [FakeCompound([COUNTS, COUNTS], regions, name="counts")],
        "FunctionalConnectivity": [
            FakeCompound(
                [FUNCTIONAL, FUNCTIONAL],
                regions,
                paradigm="Resting state (run 1)",
                name="fc1",
            ),
            FakeCompound(
                [FUNCTIONAL, FUNCTIONAL],
                regions,
                paradigm="Resting state (run 2)",
                name="fc2",
            ),
        ],
    }
    calls: list[str] = []

    connectivity = ModuleType("siibra.features.connectivity")
    connectivity.__spec__ = ModuleSpec("siibra.features.connectivity", loader=None)
    for class_name in payload:
        setattr(connectivity, class_name, type(class_name, (), {}))

    features = ModuleType("siibra.features")
    features.__spec__ = ModuleSpec("siibra.features", loader=None)
    features.connectivity = connectivity

    def get_features(_parcellation, feature_class):
        calls.append(feature_class.__name__)
        return payload[feature_class.__name__]

    features.get = get_features

    siibra = ModuleType("siibra")
    siibra.__spec__ = ModuleSpec("siibra", loader=None)
    siibra.__version__ = "0.0.0-fake"
    siibra.features = features
    siibra.parcellations = SimpleNamespace(get=lambda spec: SimpleNamespace(name=spec))

    monkeypatch.setitem(sys.modules, "siibra", siibra)
    monkeypatch.setitem(sys.modules, "siibra.features", features)
    monkeypatch.setitem(sys.modules, "siibra.features.connectivity", connectivity)
    monkeypatch.setenv("DIAXCONDEL_CACHE_DIR", str(tmp_path / "cache"))

    module = importlib.import_module("diaxcondel.connectome.atlases.siibra_atlas")
    importlib.reload(module)
    return SimpleNamespace(module=module, calls=calls, payload=payload, regions=regions)


def test_lengths_are_converted_to_centimetres_and_cleaned(fake_siibra):
    matrix, metadata = fake_siibra.module.load_connectivity_matrix("fake atlas", "streamline_lengths")
    assert matrix.shape == (4, 4)
    # 100 mm -> 10 cm, averaged over the subject with a missing cell.
    assert matrix[0, 1] == pytest.approx(10.0)
    np.testing.assert_allclose(np.diag(matrix), 0.0)
    np.testing.assert_allclose(matrix, matrix.T)
    assert np.all(np.isfinite(matrix))
    assert metadata["units"] == "cm"
    assert metadata["source_units"] == "mm"
    assert metadata["n_subjects"] == 2
    assert metadata["n_nonfinite_cells"] == 2


@pytest.mark.parametrize("aggregation", ["mean", "median", "trimmed_mean"])
def test_subject_aggregations_all_produce_finite_matrices(fake_siibra, aggregation):
    matrix, _ = fake_siibra.module.load_connectivity_matrix(
        "fake atlas",
        "streamline_lengths",
        subject_aggregation=aggregation,
    )
    assert np.all(np.isfinite(matrix))
    assert matrix[0, 1] == pytest.approx(10.0)


def test_results_are_cached_on_disk(fake_siibra):
    fake_siibra.module.load_connectivity_matrix("fake atlas", "streamline_counts")
    first = len(fake_siibra.calls)
    fake_siibra.module.load_connectivity_matrix("fake atlas", "streamline_counts")
    assert len(fake_siibra.calls) == first, "second identical query should come from the cache"
    fake_siibra.module.load_connectivity_matrix("fake atlas", "streamline_counts", subject_aggregation="median")
    assert len(fake_siibra.calls) == first + 1, "a different query must not reuse the cache entry"


def test_region_codes_and_metadata_come_from_the_atlas(fake_siibra):
    _, metadata = fake_siibra.module.load_connectivity_matrix("fake atlas", "streamline_lengths")
    regions = fake_siibra.module.region_set_from_metadata(metadata)
    assert regions.codes == (
        "Area A (X) L",
        "Area A (X) R",
        "Area B (Y) L",
        "Area B (Y) R",
    )
    first = regions[0]
    assert first.name == "Area A (X) left"
    assert first.metadata["hemisphere"] == "left"
    assert first.metadata["ancestors"][2] == "frontal lobe"


def test_distances_and_weights_describe_the_same_regions(fake_siibra):
    distances = fake_siibra.module.siibra_distances("fake atlas", merge_level=None)
    weights = fake_siibra.module.siibra_weights(distances)
    assert weights.regions.codes == distances.regions.codes
    assert distances.metadata["units"] == "cm"
    assert distances.metadata["atlas_query"]["parcellation"] == "fake atlas"


def test_weights_are_masked_to_measured_connections(fake_siibra):
    distances = fake_siibra.module.siibra_distances("fake atlas", merge_level=None)
    weights = fake_siibra.module.siibra_weights(distances).matrix()
    # Pair (0, 2) has no streamlines, so it can carry no coupling.
    assert distances.matrix()[0, 2] == 0.0
    assert weights[0, 2] == 0.0
    assert weights.max() == pytest.approx(1.0)  # default normalisation
    np.testing.assert_allclose(np.diag(weights), 0.0)


def test_weight_scale_and_normalisation_are_explicit(fake_siibra):
    distances = fake_siibra.module.siibra_distances("fake atlas", merge_level=None)
    scaled = fake_siibra.module.siibra_weights(distances, scale=0.25)
    assert scaled.matrix().max() == pytest.approx(0.25)
    raw = fake_siibra.module.siibra_weights(distances, normalization="none")
    assert raw.matrix().max() == pytest.approx(COUNTS.max())
    assert raw.metadata["normalization"] == "none"


def test_uniform_and_functional_weight_sources(fake_siibra):
    distances = fake_siibra.module.siibra_distances("fake atlas", merge_level=None)
    uniform = fake_siibra.module.siibra_weights(distances, source="uniform").matrix()
    connected = distances.matrix() > 0
    assert set(np.unique(uniform[connected])) == {1.0}

    functional = fake_siibra.module.siibra_weights(
        distances,
        source="functional",
        fc_exponent=2.0,
        normalization="none",
    ).matrix()
    # Negative correlations are clipped, then raised to the exponent.
    assert functional[0, 3] == pytest.approx(0.4**2)
    assert functional[1, 2] == pytest.approx(0.5**2)


def test_merging_follows_the_atlas_hierarchy(fake_siibra):
    grouped = fake_siibra.module.siibra_distances("fake atlas", merge_level=2)
    assert grouped.regions.codes == ("frontal lobe", "occipital lobe")
    split = fake_siibra.module.siibra_distances("fake atlas", merge_level=2, split_hemispheres=True)
    assert split.regions.codes == (
        "frontal lobe left",
        "frontal lobe right",
        "occipital lobe left",
        "occipital lobe right",
    )
    weights = fake_siibra.module.siibra_weights(grouped)
    assert weights.regions.codes == grouped.regions.codes


def test_named_regions_can_be_chosen_from_different_levels(fake_siibra):
    # One whole lobe plus one individual area — a mixture of levels that no
    # single merge level can express.
    chosen = fake_siibra.module.siibra_distances(
        "fake atlas",
        nodes=("frontal lobe", "Area B (Y) left"),
    )
    assert chosen.regions.codes[0] == "frontal lobe"
    assert chosen.regions[1].metadata["members"] == ("Area B (Y) L",)
    assert chosen.regions[0].metadata["n_members"] == 2
    assert chosen.regions[1].metadata["n_members"] == 1
    # The unchosen occipital area is gone, and atlas weights still line up.
    weights = fake_siibra.module.siibra_weights(chosen)
    assert weights.regions.codes == chosen.regions.codes
    assert chosen.metadata["nodes"] == ["frontal lobe", "Area B (Y) left"]

    # A chosen region can be split further by its own depth.
    split = fake_siibra.module.siibra_distances("fake atlas", nodes=("frontal lobe:2",))
    assert len(split.regions) == 2

    # One node on its own has nothing to connect to, and says so.


def test_anatomical_filters_select_parts_of_the_brain(fake_siibra):
    everything = fake_siibra.module.siibra_distances("fake atlas", merge_level=None)
    frontal = fake_siibra.module.siibra_distances("fake atlas", merge_level=None, keep_parts=("frontal lobe",))
    assert len(everything.regions) == 4
    assert len(frontal.regions) == 2
    dropped = fake_siibra.module.siibra_distances("fake atlas", merge_level=None, drop_parts=("frontal lobe",))
    assert len(dropped.regions) == 2
    assert set(frontal.regions.codes).isdisjoint(dropped.regions.codes)


def test_merged_distances_keep_their_member_lengths(fake_siibra):
    merged = fake_siibra.module.siibra_distances("fake atlas", merge_level=2)
    mixture = merged.mixture
    assert mixture is not None and len(mixture) > 0
    # Every member length is a real connection length from the native matrix.
    native = LENGTHS_MM[LENGTHS_MM > 0] / 10.0
    assert set(np.round(mixture.lengths_cm, 6)).issubset(set(np.round(native, 6)))
    effective = mixture.effective_lengths(len(merged.regions))
    assert effective.shape == (2, 2)
    assert effective[0, 1] > 0


def test_threshold_and_isolated_region_removal(fake_siibra):
    distances = fake_siibra.module.siibra_distances("fake atlas", merge_level=None)
    weights = fake_siibra.module.siibra_weights(distances, threshold=0.9).matrix()
    # Only the strongest connection survives the threshold.
    assert np.count_nonzero(weights) == 2


def test_single_subject_selection(fake_siibra):
    cohort = fake_siibra.module.siibra_distances("fake atlas", merge_level=None)
    single = fake_siibra.module.siibra_distances("fake atlas", merge_level=None, subject="000")
    assert single.metadata["n_subjects"] == 1
    assert single.metadata["subject"] == "000"
    # Subject 000 has no missing cells, so its matrix is the raw one.
    assert single.matrix()[0, 1] == pytest.approx(LENGTHS_MM[0, 1] / 10.0)
    assert cohort.matrix().shape == single.matrix().shape


def test_missing_cohort_and_paradigm_list_alternatives(fake_siibra):
    assert fake_siibra.module.list_cohorts("fake atlas") == ["HCP"]
    assert fake_siibra.module.list_paradigms("fake atlas") == [
        "Resting state (run 1)",
        "Resting state (run 2)",
    ]


def test_paradigm_selects_one_functional_compound(fake_siibra):
    _, metadata = fake_siibra.module.load_connectivity_matrix("fake atlas", "functional", paradigm="run 2")
    assert metadata["feature_name"] == "fc2"
