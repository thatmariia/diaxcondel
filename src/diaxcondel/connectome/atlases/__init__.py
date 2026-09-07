"""Built-in and optional atlas-backed connectome providers."""

from __future__ import annotations

from .manual import steeghs_2025_distances, steeghs_2025_regions, steeghs_2025_weights
from .synthetic import PowerLawDistances

__all__ = [
    "PowerLawDistances",
    "steeghs_2025_distances",
    "steeghs_2025_regions",
    "steeghs_2025_weights",
]
