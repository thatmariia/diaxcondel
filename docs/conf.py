"""Sphinx configuration for the diaxcondel documentation."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "diaxcondel"
author = "Mariia Steeghs-Turchina"
copyright = "2026, Mariia Steeghs-Turchina"
release = importlib.metadata.version("diaxcondel")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

# Optional extras are not installed on the documentation runner, and autodoc
# imports every module it documents.
autodoc_mock_imports = ["siibra", "streamlit", "plotly", "specparam", "pandas"]

autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True, "member-order": "bysource"}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
always_use_bars_union = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"diaxcondel {version}"
html_static_path = []
html_theme_options = {
    "source_repository": "https://github.com/thatmariia/diaxcondel",
    "source_branch": "dev",
    "source_directory": "docs/",
}

nitpick_ignore_regex = [("py:class", r".*\.(FloatArray|ComplexArray|RNGLike)")]

# pydantic's own field annotations are not resolvable from here, and the
# duplicates come from names re-exported at package level.
suppress_warnings = ["ref.python", "autosummary"]
