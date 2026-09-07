"""
Shared type aliases for diaxcondel.

The package uses NumPy arrays as the runtime representation. We keep the
aliases thin.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Plain aliases. We deliberately do not encode shape in the type system —
# shape contracts live in docstrings and are checked at runtime where it
# matters (matrix builders, simulation entry points).
type FloatArray = npt.NDArray[np.floating]
type ComplexArray = npt.NDArray[np.complexfloating]
type IntArray = npt.NDArray[np.integer]
type BoolArray = npt.NDArray[np.bool_]
