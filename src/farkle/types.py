"""Shared type aliases for the Farkle project.

This module centralizes a few simple ``TypeAlias`` definitions used across
the code base.  ``Int64Arr1D`` is provided as a convenience for NumPy arrays
of ``np.int64`` that are expected, by convention, to be one-dimensional.
"""

from typing import Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

Counts6: TypeAlias = Tuple[int, int, int, int, int, int]
FacesT: TypeAlias = Tuple[int, ...]
Int64Arr1D: TypeAlias = npt.NDArray[np.int64]
