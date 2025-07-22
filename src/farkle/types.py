"""Shared type aliases for the Farkle project.

This module centralizes a few simple ``TypeAlias`` definitions used across
the code base.  ``Int64Array1D`` is provided as a convenience for NumPy arrays
of ``np.int64`` that are expected, by convention, to be one-dimensional.
"""

from __future__ import annotations

from typing import Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

SixFaceCounts: TypeAlias = Tuple[int, int, int, int, int, int]  # counts for faces 1-6
FacesSequence: TypeAlias = Tuple[int, ...]  # ordered dice faces
Int64Array1D: TypeAlias = npt.NDArray[np.int64]  # 1-D array of 64-bit ints
DiceRoll: TypeAlias = list[int]
