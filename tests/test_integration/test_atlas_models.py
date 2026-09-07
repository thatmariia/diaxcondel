"""
Live checks against the real siibra service.

These download hundreds of subject matrices, so they only run with
``pytest --run-network`` (and the ``atlas`` extra installed). The offline
behaviour of the adapter is covered in ``tests/test_connectome``.
"""

import numpy as np
import pytest

from diaxcondel.connectome.bundle import source_metadata
from diaxcondel.experiment import (
    build_model,
    compute_spectra,
    region_summaries,
    run_experiment,
)
from diaxcondel.experiment.presets import get_preset

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def julich_lobes():
    pytest.importorskip("siibra", reason="needs the atlas extra")
    from diaxcondel.connectome.atlases.siibra_atlas import siibra_connectome

    return siibra_connectome("julich 3.1", merge_level=3)


def test_atlas_connectome_has_plausible_geometry(julich_lobes):
    assert 5 <= julich_lobes.n_regions <= 40
    assert "frontal lobe" in julich_lobes.codes
    distances = julich_lobes.distance_matrix()
    weights = julich_lobes.weight_matrix()
    connected = weights > 0
    median_cm = float(np.median(distances[connected]))
    assert 1.0 < median_cm < 25.0, f"implausible median connection length: {median_cm} cm"
    assert weights.max() == pytest.approx(1.0)
    provenance = dict(source_metadata(julich_lobes.distances))
    assert provenance["units"] == "cm"
    assert provenance["n_subjects"] >= 100
    assert provenance["atlas_query"]["parcellation"] == "julich 3.1"


def test_hemisphere_split_doubles_the_node_count(julich_lobes):
    from diaxcondel.connectome.atlases.siibra_atlas import siibra_distances

    split = siibra_distances("julich 3.1", merge_level=3, split_hemispheres=True)
    assert len(split.regions) > julich_lobes.n_regions
    assert any(code.endswith(" left") for code in split.regions.codes)


def test_cortex_only_filter_and_single_subject(julich_lobes):
    from diaxcondel.connectome.atlases.siibra_atlas import list_subjects, siibra_distances

    cortex = siibra_distances("julich 3.1", merge_level=4, keep_parts=("cerebral cortex",))
    assert 20 < len(cortex.regions) < 200
    assert all("cerebral cortex" not in code for code in cortex.regions.codes)

    subjects = list_subjects("julich 3.1")
    assert len(subjects) > 100
    single = siibra_distances("julich 3.1", merge_level=3, subject=subjects[0])
    assert single.metadata["n_subjects"] == 1
    # One participant is not the cohort average, but the geometry is comparable.
    assert single.regions.codes == julich_lobes.codes
    assert not np.allclose(single.matrix(), julich_lobes.distance_matrix())


def test_merged_atlas_pathways_carry_their_member_lengths():
    from diaxcondel.connectome.atlases.siibra_atlas import siibra_distances

    merged = siibra_distances("julich 3.1", merge_level=3)
    mixture = merged.mixture
    assert mixture is not None and len(mixture) > 1000
    effective = mixture.effective_lengths(len(merged.regions))
    geometric = merged.matrix()
    shown = (effective > 0) & (geometric > 0)
    # Fibre-weighted lengths differ from the plain geometric mean, because the
    # members carrying the most streamlines are not of average length.
    ratios = effective[shown] / geometric[shown]
    assert np.median(ratios) > 1.0
    assert float(ratios.max()) > 1.5


def test_parcellation_info_carries_documentation_links():
    from diaxcondel.connectome.atlases.siibra_atlas import parcellation_info

    info = parcellation_info("julich 3.1")
    assert "Julich" in info["name"]
    assert info["n_regions"] > 100
    assert any(url.startswith("http") for url in info["urls"])


def test_atlas_preset_builds_and_runs():
    pytest.importorskip("siibra", reason="needs the atlas extra")
    spec = get_preset("atlas_lobes").with_simulation(sample_rate_hz=250, sim_seconds=4.0, burnin_seconds=1.0, n_lags=75)
    built = build_model(spec)
    assert built.n_regions > 5
    result = run_experiment(spec, built=built)
    summaries = region_summaries(result, compute_spectra(result, segment_seconds=2.0))
    assert len(summaries) == built.n_regions
    assert all(np.isfinite(item.peak_hz) for item in summaries)
