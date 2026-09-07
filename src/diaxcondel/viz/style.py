"""
A shared look for every figure in the package.

One palette, one set of axis conventions, one place to change them. The
colours are the Okabe–Ito qualitative set, which stays distinguishable for
the common forms of colour vision deficiency and in greyscale print.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Qualitative palette (Okabe & Ito), safe for colour vision deficiency.
PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#666666",
)

#: Sequential colormap for magnitudes that start at zero (weights, coherence).
SEQUENTIAL = "viridis"

#: Sequential colormap for distances and delays.
DISTANCE_MAP = "magma_r"

#: Muted colour for reference lines and annotations.
ANNOTATION = "#444444"

RC_PARAMS: Mapping[str, Any] = {
    "figure.facecolor": "white",
    "figure.dpi": 110,
    "axes.facecolor": "white",
    "axes.edgecolor": "#B0B0B0",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.labelcolor": "#222222",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "medium",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=list(PALETTE)),
    "grid.color": "#E4E4E4",
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "lines.solid_capstyle": "round",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "font.size": 9,
    "image.cmap": SEQUENTIAL,
    "savefig.bbox": "tight",
}


def use_style() -> None:
    """
    Apply the package's figure style globally.

    Call it once in a notebook or script to make your own plots match the
    ones the package produces.
    """
    plt.rcParams.update(dict(RC_PARAMS))


@contextmanager
def styled() -> Iterator[None]:
    """
    Apply the package's figure style for the duration of a ``with`` block.

    Yields
    ------
    None
        Inside the block, matplotlib uses the package style; outside it, the
        caller's own settings are untouched.
    """
    with plt.rc_context(dict(RC_PARAMS)):
        yield


#: Accent colour for plots that do not distinguish regions by colour.
ACCENT = PALETTE[0]

#: Line styles cycled alongside the palette when curves outnumber colours.
LINE_STYLES: tuple[str, ...] = ("-", "--", ":")


def region_colors(n_regions: int) -> list[str]:
    """
    Return colours for stacked or ordered displays.

    Parameters
    ----------
    n_regions : int
        How many colours are needed.

    Returns
    -------
    list of str
        The qualitative palette for small sets; for larger ones, samples of a
        perceptually uniform ramp, which reads as an ordered strip rather
        than as a set of unrelated colours.
    """
    if n_regions <= len(PALETTE):
        return list(PALETTE[:n_regions])
    colormap = plt.get_cmap("viridis")
    return [mpl.colors.to_hex(colormap(index / max(1, n_regions - 1))) for index in range(n_regions)]


def region_styles(n_regions: int) -> list[tuple[str, str]]:
    """
    Return colour and line-style pairs for overlaid curves.

    Parameters
    ----------
    n_regions : int
        How many curves will be drawn.

    Returns
    -------
    list of (str, str)
        Colour and line style per curve. Colours cycle through the palette
        and the line style changes on each cycle, so up to twenty-four curves
        stay individually identifiable — beyond that, highlight a few rather
        than drawing them all in colour.
    """
    styles: list[tuple[str, str]] = []
    for index in range(n_regions):
        colour = PALETTE[index % len(PALETTE)]
        dash = LINE_STYLES[(index // len(PALETTE)) % len(LINE_STYLES)]
        styles.append((colour, dash))
    return styles
