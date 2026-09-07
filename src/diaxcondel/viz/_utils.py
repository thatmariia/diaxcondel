import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def resolve_axes(ax: Axes | None, *, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax  # type: ignore[return-value]
