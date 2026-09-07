"""Tests for choosing model regions by name rather than by hierarchy level."""

from diaxcondel.connectome.grouping import (
    NodeSpec,
    browse_regions,
    describe_overlaps,
    node_labels,
    parse_nodes,
    selection_coverage,
)
from diaxcondel.connectome.regions import Region, RegionSet


def _atlas() -> RegionSet:
    """A miniature atlas whose branches sit at different depths."""
    return RegionSet(
        (
            Region("f1", "Area F1", {"ancestors": ("brain", "cortex", "frontal lobe", "superior frontal")}),
            Region("f2", "Area F2", {"ancestors": ("brain", "cortex", "frontal lobe", "inferior frontal")}),
            Region("o1", "Area O1", {"ancestors": ("brain", "cortex", "occipital lobe", "V1")}),
            Region("t1", "MD nucleus", {"ancestors": ("brain", "subcortex", "thalamus"), "hemisphere": "left"}),
            Region("t2", "LGN", {"ancestors": ("brain", "subcortex", "thalamus"), "hemisphere": "right"}),
            Region("s1", "STN", {"ancestors": ("brain", "subcortex", "subthalamus")}),
            Region("c1", "Lobule I", {"ancestors": ("brain", "cerebellum")}),
        )
    )


def test_parse_nodes_reads_names_depths_and_mappings():
    assert parse_nodes(["frontal lobe", "thalamus:2"]) == (
        NodeSpec("frontal lobe", 0),
        NodeSpec("thalamus", 2),
    )
    assert parse_nodes({"thalamus": 1}) == (NodeSpec("thalamus", 1),)
    assert parse_nodes(()) == ()
    # A colon that is not a depth belongs to the region name.
    assert parse_nodes(["Area 4p: left"]) == (NodeSpec("Area 4p: left", 0),)
    assert NodeSpec("thalamus", 2).as_text() == "thalamus:2"
    assert NodeSpec("thalamus").as_text() == "thalamus"


def test_regions_can_be_chosen_from_different_levels():
    labels, unmatched = node_labels(_atlas(), ("frontal lobe", "parietal lobe", "thalamus"))
    # Two lobes and the whole thalamus: no single merge level expresses this.
    assert labels == {
        "f1": "frontal lobe",
        "f2": "frontal lobe",
        "t1": "thalamus",
        "t2": "thalamus",
    }
    # Occipital cortex and cerebellum were not chosen, so they leave the model.
    assert "o1" not in labels
    assert "c1" not in labels
    assert unmatched == ("parietal lobe",)


def test_a_chosen_region_can_be_split_by_its_own_depth():
    labels, _ = node_labels(_atlas(), ("frontal lobe:1", "thalamus"))
    assert labels["f1"] == "superior frontal"
    assert labels["f2"] == "inferior frontal"
    assert labels["t1"] == labels["t2"] == "thalamus"


def test_depth_past_the_atlas_leaves_resolves_single_regions():
    labels, _ = node_labels(_atlas(), ("thalamus:9",))
    assert labels == {"t1": "t1", "t2": "t2"}


def test_the_most_specific_choice_wins():
    labels, unmatched = node_labels(_atlas(), ("cortex", "occipital lobe:1"))
    assert unmatched == ()
    assert labels["f1"] == labels["f2"] == "cortex"
    assert labels["o1"] == "V1"  # the deeper choice refines only its own branch


def test_a_name_the_atlas_has_matches_only_itself():
    # "thalamus" is a real region, so it must not drag in the subthalamus...
    labels, _ = node_labels(_atlas(), ("thalamus",))
    assert set(labels) == {"t1", "t2"}
    # ...while free text that names no region searches, and resolves to the
    # highest region containing it.
    loose, unmatched = node_labels(_atlas(), ("thalam",))
    assert unmatched == ()
    assert loose == {"t1": "thalamus", "t2": "thalamus", "s1": "subthalamus"}


def test_hemispheres_can_be_kept_apart():
    labels, _ = node_labels(_atlas(), ("thalamus",), split_hemispheres=True)
    assert labels == {"t1": "thalamus left", "t2": "thalamus right"}


def test_nothing_selected_is_an_error_and_a_typo_is_reported():
    labels, unmatched = node_labels(_atlas(), ("hippocampus",))
    assert labels == {}
    assert unmatched == ("hippocampus",)


def test_browsing_lists_regions_by_level_and_by_name():
    choices = browse_regions(_atlas(), level=2)
    # Level 2 holds the lobes and the thalamus — and a cerebellar leaf, because
    # branches of the atlas do not all run equally deep.
    assert [choice.name for choice in choices] == [
        "frontal lobe",
        "occipital lobe",
        "thalamus",
        "subthalamus",
        "Lobule I",
    ]
    assert {choice.name: choice.n_members for choice in choices}["frontal lobe"] == 2

    found = browse_regions(_atlas(), contains="thalam")
    assert [(choice.name, choice.level, choice.n_members) for choice in found] == [
        ("thalamus", 2, 2),
        ("subthalamus", 2, 1),
    ]

    # Levels are only a way of looking: a search reaches every one of them.
    every = browse_regions(_atlas())
    assert ("cerebellum", 1) in {(choice.name, choice.level) for choice in every}
    assert ("Lobule I", 2) in {(choice.name, choice.level) for choice in every}
    assert all(choice.name != "Lobule I" for choice in browse_regions(_atlas(), include_leaves=False))


def test_overlapping_selections_are_reported_with_what_happens():
    # A branch and something inside it: legal, since the deeper entry wins
    # where they meet, but worth saying out loud.
    messages = describe_overlaps(_atlas(), ("cortex", "occipital lobe"))
    assert len(messages) == 1
    assert "'cortex' contains 'occipital lobe'" in messages[0]
    assert "1 region in common" in messages[0]

    assert describe_overlaps(_atlas(), ("frontal lobe", "thalamus")) == ()
    assert describe_overlaps(_atlas(), ("thalamus", "thalamus")) == ("'thalamus' is selected twice.",)


def test_coverage_judges_each_selection_on_its_own():
    # node_labels lets entries compete; coverage reports what each would take.
    coverage = selection_coverage(_atlas(), ("cortex", "occipital lobe"))
    assert set(coverage["cortex"]) == {"f1", "f2", "o1"}
    assert set(coverage["occipital lobe"]) == {"o1"}
