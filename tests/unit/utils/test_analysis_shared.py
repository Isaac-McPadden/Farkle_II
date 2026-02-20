import numpy as np
import pytest

from farkle.utils.analysis_shared import (
    as_float,
    as_int,
    is_na,
    tiers_to_map,
    to_int,
    to_stat_value,
    try_to_int,
)


def test_try_to_int_handles_numpy_scalar_bool_invalid_and_non_finite() -> None:
    assert try_to_int(np.int64(7)) == 7
    assert try_to_int(True) == 1
    assert try_to_int(False) == 0
    assert try_to_int("not-an-int") is None
    assert try_to_int(float("nan")) is None
    assert try_to_int(float("inf")) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  42  ", 42),
        ("\t-7\n", -7),
        (" 003 ", 3),
        (" true ", None),
        ("False", None),
        ("nan", None),
        (" NaN ", None),
    ],
)
def test_try_to_int_string_inputs(value: str, expected: int | None) -> None:
    assert try_to_int(value) == expected


def test_try_to_int_unsupported_object_returns_none() -> None:
    class Unsupported:
        pass

    assert try_to_int(Unsupported()) is None


def test_to_int_rejects_non_integral_float() -> None:
    with pytest.raises(ValueError, match="non-integral float"):
        to_int(3.25)


@pytest.mark.parametrize("value", [" 12 ", "nan", "true", None, np.nan])
def test_to_int_strictly_rejects_non_integer_scalars(value: object) -> None:
    with pytest.raises(ValueError, match="cannot convert"):
        to_int(value)


def test_to_stat_value_non_finite_float_and_object_fallback() -> None:
    assert to_stat_value(float("nan")) is None
    assert to_stat_value(float("-inf")) is None

    class Obj:
        def __str__(self) -> str:
            return "obj-repr"

    assert to_stat_value(Obj()) == "obj-repr"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" true ", " true "),
        ("False", "False"),
        ("  99  ", "  99  "),
        (np.int64(8), 8),
        (np.float64(3.5), 3.5),
        (np.float64(np.nan), None),
        ("nan", "nan"),
        ("NaN", "NaN"),
    ],
)
def test_to_stat_value_preserves_strings_and_coerces_numpy_scalars(
    value: object, expected: int | float | str | None
) -> None:
    assert to_stat_value(value) == expected


def test_tiers_to_map_rejects_duplicate_strategy() -> None:
    with pytest.raises(ValueError, match="duplicate strategy"):
        tiers_to_map([["A", "B"], ["B", "C"]])


def test_tiers_to_map_rejects_duplicate_strategy_ids_after_str_coercion() -> None:
    with pytest.raises(ValueError, match="duplicate strategy"):
        tiers_to_map([["101"], [101]])


def test_tiers_to_map_handles_empty_tiers_with_stable_ranks() -> None:
    assert tiers_to_map([[], ["A"], [], ["B", "C"]]) == {
        "A": 2,
        "B": 4,
        "C": 4,
    }


def test_is_na_accepts_common_scalar_missing_values() -> None:
    assert is_na(None) is True
    assert is_na(np.nan) is True
    assert is_na(np.float64(np.nan)) is True
    assert is_na(np.int64(2)) is False


def test_as_float_and_as_int_reject_na_and_non_numeric() -> None:
    with pytest.raises(ValueError, match="NA value"):
        as_float(np.nan)
    with pytest.raises(ValueError, match="NA value"):
        as_int(np.nan)

    with pytest.raises(TypeError, match="numeric scalar"):
        as_float("1.2")
    with pytest.raises(TypeError, match="numeric scalar"):
        as_int("2")


def test_as_int_requires_integral_float_values() -> None:
    assert as_int(np.float64(4.0)) == 4
    with pytest.raises(ValueError, match="non-integral float"):
        as_int(4.5)


def test_as_float_accepts_numpy_and_python_numeric_scalars() -> None:
    assert as_float(np.int64(3)) == pytest.approx(3.0)
    assert as_float(np.float32(1.5)) == pytest.approx(1.5)
    assert as_float(True) == pytest.approx(1.0)
