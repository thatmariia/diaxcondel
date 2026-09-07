"""Tests for the component catalog: introspection, validation, building."""

from typing import Annotated, Literal

import pytest

from diaxcondel import catalog
from diaxcondel.catalog.param import Param, infer_fields
from diaxcondel.catalog.registry import ComponentSpec


def sample_factory(
    count: int = 3,
    ratio: Annotated[float, Param(unit="cm", minimum=0.0, maximum=2.0)] = 0.5,
    mode: Literal["a", "b"] = "a",
    flag: bool = False,
    codes: Annotated[tuple[str, ...], Param(widget="regions")] = (),
) -> dict:
    """
    Summary of the sample factory.

    Parameters
    ----------
    count : int
        How many things.
    ratio : float
        A bounded ratio.
    mode : {"a", "b"}
        Which mode.
    flag : bool
        Whether to flag.
    codes : tuple of str
        Region codes.

    Returns
    -------
    dict
        The parameters, echoed.
    """
    return {"count": count, "ratio": ratio, "mode": mode, "flag": flag, "codes": codes}


def test_field_inference_covers_types_bounds_and_docs():
    fields, context = infer_fields(sample_factory)
    assert context == ()
    by_name = {f.name: f for f in fields}
    assert by_name["count"].kind == "int"
    assert by_name["count"].description == "How many things."
    assert by_name["ratio"].kind == "float"
    assert (
        by_name["ratio"].minimum,
        by_name["ratio"].maximum,
        by_name["ratio"].unit,
    ) == (0.0, 2.0, "cm")
    assert by_name["mode"].kind == "choice"
    assert by_name["mode"].choices == ("a", "b")
    assert by_name["flag"].kind == "bool"
    assert by_name["codes"].kind == "str_list"
    assert by_name["codes"].widget == "regions"


def test_parameters_without_defaults_become_context():
    def factory(regions: str, scale: float = 1.0) -> str:
        """Summary.

        Parameters
        ----------
        regions : str
            Injected by the builder.
        scale : float
            User option.
        """
        return regions * int(scale)

    fields, context = infer_fields(factory)
    assert context == ("regions",)
    assert [f.name for f in fields] == ["scale"]


def _spec() -> ComponentSpec:
    fields, context = infer_fields(sample_factory)
    return ComponentSpec(
        slot="kernel",
        name="sample",
        label="Sample",
        summary="",
        factory=sample_factory,
        fields=fields,
        context=context,
    )


def test_coerce_fills_defaults_and_validates():
    spec = _spec()
    assert spec.coerce()["count"] == 3
    assert spec.coerce({"count": "7"})["count"] == 7


def test_build_reports_missing_and_unexpected_context():
    spec = _spec()
    assert spec.build({"count": 2})["count"] == 2


def test_every_registered_entry_has_metadata_and_valid_defaults():
    for entry in catalog.iter_entries():
        assert entry.label
        assert entry.summary, f"{entry.key} has no summary"
        assert entry.coerce() == entry.defaults()
        for field in entry.fields:
            assert field.description, f"{entry.key}.{field.name} is undocumented"


@pytest.mark.parametrize(
    ("slot", "name"),
    [
        ("distances", "steeghs_2025"),
        ("distances", "power_law"),
        ("distances", "power_law_with_hub"),
        ("distances", "ring"),
        ("distances", "manual"),
        ("kernel", "fixed_speed"),
        ("kernel", "gamma_delay"),
        ("diameter", "gev"),
        ("diameter", "gamma"),
        ("diameter", "lognormal"),
        ("diameter", "rayleigh"),
        ("local_delay", "none"),
        ("local_delay", "gamma"),
        ("noise", "white"),
        ("noise", "pink"),
        ("noise", "colored"),
        ("transfer", "identity"),
        ("transfer", "centred_sigmoid"),
    ],
)
def test_offline_entries_build_with_defaults(slot, name):
    assert catalog.build(slot, name) is not None


@pytest.mark.parametrize("name", ["steeghs_2025", "uniform", "distance_decay", "random_sparse", "manual"])
def test_connectivity_entries_build_over_the_given_regions(name):
    distances = catalog.build("distances", "steeghs_2025")
    weights = catalog.build("connectivity", name, regions=distances.regions, distances=distances)
    assert weights.regions.codes == distances.regions.codes
    assert weights.matrix().shape == (5, 5)


def test_drive_and_modulation_need_regions(sim_params):
    from diaxcondel.connectome.atlases.manual import steeghs_2025_regions

    regions = steeghs_2025_regions()
    drive = catalog.build("drive", "gaussian_pulse", {"targets": ("OL",)}, regions=regions)
    assert [t.code for t in drive.targets] == ["OL"]
    modulation = catalog.build("modulation", "step_gain", {"targets": ("OL",), "gain": 2.0}, regions=regions)
    assert modulation.factor.shape == (5, 5)
