# src/farkle/types.py  (new tiny helper, purely for typing)
from typing import Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

Counts6: TypeAlias = Tuple[int, int, int, int, int, int]
FacesT : TypeAlias = Tuple[int, ...]
Int64Arr1D: TypeAlias = npt.NDArray[np.int64]   