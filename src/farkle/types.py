# src/farkle/types.py  (new tiny helper, purely for typing)
from typing import Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

SixFaceCounts: TypeAlias = Tuple[int, int, int, int, int, int]  # counts for faces 1-6
FacesSequence: TypeAlias = Tuple[int, ...]  # ordered dice faces
Int64Array1D: TypeAlias = npt.NDArray[np.int64]  # 1-D array of 64-bit ints
