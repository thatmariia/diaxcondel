"""
Manual / pre-loaded atlas data.

Convenience constructors for the small five-region atlas used in the
original paper.
"""

from __future__ import annotations

import numpy as np

from ..distance import ManualDistances
from ..regions import RegionSet
from ..weights import ManualConnectivity

# Streamline-length distances (cm) for {FL, PL, OL, TL, T}
_STEEGHS_2025_DISTANCES_CM = np.array(
    [
        [0.000, 8.769, 7.978, 5.955, 5.041],
        [8.769, 0.000, 8.558, 7.357, 7.544],
        [7.978, 8.558, 0.000, 8.906, 9.453],
        [5.955, 7.357, 8.906, 0.000, 6.573],
        [5.041, 7.544, 9.453, 6.573, 0.000],
    ]
)

# Custom-tuned weights.
_STEEGHS_2025_WEIGHTS = np.array(
    [
        [0.000, 0.667, 0.667, 0.250, 0.083],
        [0.667, 0.000, 1.000, 0.625, 0.083],
        [0.667, 1.000, 0.000, 0.667, 0.083],
        [0.250, 0.625, 0.667, 0.000, 0.083],
        [0.083, 0.083, 0.083, 0.083, 0.000],
    ]
)


def steeghs_2025_regions() -> RegionSet:
    """
    Return the five-region set used in Steeghs-Turchina et al. (2025).

    Returns
    -------
    RegionSet
        Regions in canonical order ``[FL, PL, OL, TL, T]``.
    """
    return RegionSet.from_codes(
        codes=["FL", "PL", "OL", "TL", "T"],
        names=["frontal lobe", "parietal lobe", "occipital lobe", "temporal lobe", "thalamus"],
    )


def steeghs_2025_distances() -> ManualDistances:
    """Return the streamline-length distances (cm) from the original paper."""
    return ManualDistances(regions=steeghs_2025_regions(), matrix_cm=_STEEGHS_2025_DISTANCES_CM)


def steeghs_2025_weights() -> ManualConnectivity:
    """Return the custom-tuned resting-state weights from the original paper."""
    return ManualConnectivity(regions=steeghs_2025_regions(), weights=_STEEGHS_2025_WEIGHTS)
