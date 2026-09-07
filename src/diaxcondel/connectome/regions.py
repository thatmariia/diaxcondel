"""
Region and region-set primitives.

A :class:`Region` is an immutable label for a brain area. A
:class:`RegionSet` is an ordered, deduplicated collection of regions and
defines the index convention used by every matrix in the package: row/column
``i`` of any ``(N, N)`` matrix corresponds to ``regions[i]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from ._meta import dataclass_state, freeze_metadata, restore_dataclass_state


@dataclass(frozen=True, slots=True, eq=False)
class Region:
    """
    Immutable identifier for a brain region.

    Parameters
    ----------
    code : str
        Short label, e.g. ``"PL"`` for parietal lobe. Used in plots and logs.
    name : str
        Human-readable name, e.g. ``"parietal lobe"``.
    metadata : Mapping[str, object], optional
        Free-form metadata (atlas id, MNI centroid, hemispheric side, ...).
        Not used by the model; preserved for downstream tooling.

    Notes
    -----
    Equality and hashing use ``code`` only. Two regions with the same code
    but different names are considered the same region — a deliberate choice
    so that loaders from different atlases can map onto a common index
    namespace without bookkeeping.
    """

    code: str
    name: str = ""
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:  # noqa: D105
        if not self.code:
            raise ValueError("Region.code must be a non-empty string")
        # Freeze a copy so post-construction mutation of the original dict
        # cannot affect this Region.
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    def __getstate__(self) -> dict[str, object]:
        """Return picklable state; the read-only metadata view becomes a dict."""
        return dataclass_state(self)

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore from :meth:`__getstate__`, re-freezing the metadata mapping."""
        restore_dataclass_state(self, state)

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return isinstance(other, Region) and self.code == other.code

    def __hash__(self) -> int:  # noqa: D105
        return hash(self.code)


@dataclass(frozen=True, slots=True, eq=False)
class RegionSet:
    """
    Ordered, deduplicated collection of regions.

    The index of each region in :attr:`regions` is the row/column of every
    matrix in the package referring to it.

    Parameters
    ----------
    regions : tuple of Region
        Ordered regions. Order is significant.

    Raises
    ------
    ValueError
        If any region code is duplicated.
    """

    regions: tuple[Region, ...]

    def __post_init__(self) -> None:  # noqa: D105
        codes = [r.code for r in self.regions]
        if len(set(codes)) != len(codes):
            duplicates = {c for c in codes if codes.count(c) > 1}
            raise ValueError(f"duplicate region codes: {sorted(duplicates)}")

    @classmethod
    def from_codes(cls, codes: list[str], names: list[str] | None = None) -> Self:
        """
        Build a :class:`RegionSet` from parallel lists of codes (and names).

        Parameters
        ----------
        codes : list of str
            Region codes, in the desired order.
        names : list of str, optional
            Names parallel to ``codes``. If ``None``, names are set equal to
            codes.

        Returns
        -------
        RegionSet
            Newly constructed region set.
        """
        if names is None:
            names = list(codes)
        if len(names) != len(codes):
            raise ValueError("codes and names must have the same length")
        return cls(tuple(Region(code=c, name=n) for c, n in zip(codes, names, strict=True)))

    def __len__(self) -> int:  # noqa: D105
        return len(self.regions)

    def __getitem__(self, key: int | str) -> Region:  # noqa: D105
        if isinstance(key, int):
            return self.regions[key]
        for r in self.regions:
            if r.code == key:
                return r
        raise KeyError(key)

    def index(self, code: str) -> int:
        """
        Return the integer index of the region with the given code.

        Parameters
        ----------
        code : str
            Region code.

        Returns
        -------
        int
            Index in :attr:`regions`.

        Raises
        ------
        KeyError
            If ``code`` is not present.
        """
        for i, r in enumerate(self.regions):
            if r.code == code:
                return i
        raise KeyError(code)

    @property
    def codes(self) -> tuple[str, ...]:
        """Return the tuple of region codes in order."""
        return tuple(r.code for r in self.regions)
