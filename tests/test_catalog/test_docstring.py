"""Tests for NumPy-docstring parameter extraction."""

from diaxcondel.catalog._docstring import parse_parameter_docs


def test_extracts_single_parameter():
    doc = """
    Summary line.

    Parameters
    ----------
    alpha : float
        The first parameter.

    Returns
    -------
    int
        Something else.
    """
    assert parse_parameter_docs(doc) == {"alpha": "The first parameter."}


def test_shared_description_for_grouped_names():
    doc = """
    Summary.

    Parameters
    ----------
    fmin_hz, fmax_hz : float
        Inclusive frequency range.
    """
    docs = parse_parameter_docs(doc)
    assert docs["fmin_hz"] == "Inclusive frequency range."
    assert docs["fmax_hz"] == "Inclusive frequency range."


def test_multiline_description_is_collapsed():
    doc = """
    Summary.

    Parameters
    ----------
    beta : float
        First line
        second line.
    """
    assert parse_parameter_docs(doc) == {"beta": "First line second line."}


def test_stops_at_next_section():
    doc = """
    Summary.

    Parameters
    ----------
    a : int
        Description of a.

    Notes
    -----
    b : int
        Not a parameter.
    """
    assert list(parse_parameter_docs(doc)) == ["a"]


def test_missing_section_and_empty_docstring():
    assert parse_parameter_docs(None) == {}
    assert parse_parameter_docs("Just a summary.") == {}
