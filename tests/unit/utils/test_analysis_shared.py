import numpy as np
import pytest

from farkle.utils.analysis_shared import tiers_to_map, to_int, to_stat_value, try_to_int


def test_try_to_int_handles_numpy_scalar_bool_invalid_and_non_finite() -> None:
    assert try_to_int(np.int64(7)) == 7
    assert try_to_int(True) == 1
    assert try_to_int(False) == 0
    assert try_to_int("not-an-int") is None
    assert try_to_int(float("nan")) is None
    assert try_to_int(float("inf")) is None


def test_to_int_rejects_non_integral_float() -> None:
    with pytest.raises(ValueError, match="non-integral float"):
        to_int(3.25)


def test_to_stat_value_non_finite_float_and_object_fallback() -> None:
    assert to_stat_value(float("nan")) is None
    assert to_stat_value(float("-inf")) is None

    class Obj:
        def __str__(self) -> str:
            return "obj-repr"

    assert to_stat_value(Obj()) == "obj-repr"


def test_tiers_to_map_rejects_duplicate_strategy() -> None:
    with pytest.raises(ValueError, match="duplicate strategy"):
        tiers_to_map([["A", "B"], ["B", "C"]])

