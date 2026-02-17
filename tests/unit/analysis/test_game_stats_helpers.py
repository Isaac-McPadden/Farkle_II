from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from farkle.analysis import game_stats


@pytest.mark.parametrize(
    ("quantile", "expected"),
    [
        (0.0, 1.0),
        (0.5, 2.0),
        (0.9, 3.0),
        (1.0, 3.0),
    ],
)
def test_weighted_quantile_boundaries_and_midpoints(quantile: float, expected: float) -> None:
    values = np.array([3.0, 1.0, 2.0], dtype=float)
    weights = np.array([1.0, 1.0, 1.0], dtype=float)

    assert game_stats._weighted_quantile(values, weights, quantile) == pytest.approx(expected)


def test_weighted_quantile_and_moments_handle_zero_total_weight() -> None:
    values = np.array([1.0, 3.0, 5.0], dtype=float)
    weights = np.zeros_like(values)

    assert np.isnan(game_stats._weighted_quantile(values, weights, 0.25))
    assert np.isnan(game_stats._weighted_mean(values, weights))
    assert np.isnan(game_stats._weighted_std(values, weights))


def test_weighted_mean_and_std_with_non_uniform_weights() -> None:
    values = np.array([2.0, 4.0, 10.0], dtype=float)
    weights = np.array([1.0, 3.0, 1.0], dtype=float)

    assert game_stats._weighted_mean(values, weights) == pytest.approx(4.8)
    assert game_stats._weighted_std(values, weights) == pytest.approx(2.7129319933)


def test_second_highest_supports_missing_and_single_column_rows() -> None:
    scores = np.array(
        [
            [10.0, 7.0, 2.0],
            [5.0, np.nan, 3.0],
            [np.nan, np.nan, 12.0],
        ],
        dtype=float,
    )

    result = game_stats._second_highest(scores)
    assert result.tolist() == pytest.approx([7.0, 3.0, np.nan], nan_ok=True)

    single_col = np.array([[1.0], [np.nan]], dtype=float)
    single_result = game_stats._second_highest(single_col)
    assert np.isnan(single_result).all()


@pytest.mark.parametrize(
    ("counts", "quantile", "expected"),
    [
        ({1: 2, 3: 1, 10: 1}, 0.0, 1),
        ({1: 2, 3: 1, 10: 1}, 0.5, 1),
        ({1: 2, 3: 1, 10: 1}, 0.75, 3),
        ({1: 2, 3: 1, 10: 1}, 1.0, 10),
        ({}, 0.5, None),
        ({5: 0}, 0.5, None),
    ],
)
def test_quantile_from_counts(counts: dict[int, int], quantile: float, expected: int | None) -> None:
    assert game_stats._quantile_from_counts(counts, quantile) == expected


def test_integer_downcast_helpers_cover_uint8_int32_int64() -> None:
    tiny, tiny_arrow = game_stats._select_int_dtype(255)
    med, med_arrow = game_stats._select_int_dtype(70_000)
    large, large_arrow = game_stats._select_int_dtype(np.iinfo(np.int32).max + 1)

    assert tiny == np.dtype(np.uint8)
    assert med == np.dtype(np.int32)
    assert large == np.dtype(np.int64)
    assert tiny_arrow == pa.uint8()
    assert med_arrow == pa.int32()
    assert large_arrow == pa.int64()

    frame = pd.DataFrame(
        {
            "n_players": [2, 3],
            "observations": [1000, 2000],
            "non_int": [2.5, 3.2],
            "huge": [np.iinfo(np.int32).max + 10, np.iinfo(np.int32).max + 20],
        }
    )

    out = game_stats._downcast_integer_stats(
        frame, columns=("n_players", "observations", "non_int", "huge")
    )

    assert out["n_players"].dtype == np.uint8
    assert out["observations"].dtype == np.int32
    assert out["non_int"].dtype == frame["non_int"].dtype
    assert out["huge"].dtype == np.int64


@pytest.fixture(
    params=[
        (pa.uint8(), "UInt8"),
        (pa.int32(), "Int32"),
        (pa.int64(), "Int64"),
    ]
)
def strategy_dtype_case(request: pytest.FixtureRequest) -> tuple[pa.DataType, str]:
    return request.param


def test_coerce_strategy_dtype_fixture(strategy_dtype_case: tuple[pa.DataType, str]) -> None:
    strategy_type, expected_dtype = strategy_dtype_case
    frame = pd.DataFrame({"strategy": [1, "2", 3.0, None], "n_players": [2, 2, 3, 3]})

    out = game_stats._coerce_strategy_dtype(frame, strategy_type)

    assert out["strategy"].dtype.name == expected_dtype
    assert out["strategy"].tolist()[:3] == [1, 2, 3]
    assert out["strategy"].isna().iloc[3]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (np.int64(2), 2),
        (3.0, 3),
    ],
)
def test_strategy_key_to_int_accepts_numeric_inputs(value: object, expected: int) -> None:
    assert game_stats._strategy_key_to_int(value) == expected


@pytest.mark.parametrize("value", [None, "bad", float("nan")])
def test_strategy_key_to_int_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="invalid strategy scalar"):
        game_stats._strategy_key_to_int(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (2.0, 2.0),
        ("3", "3"),
        (b"4", "4"),
        (None, pd.NA),
    ],
)
def test_strategy_stat_value_handles_int_float_string_and_null(
    value: object, expected: object
) -> None:
    result = game_stats._strategy_stat_value(value)
    if expected is pd.NA:
        assert result is pd.NA
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("game-count", "game-count"),
        ("gamecount", "game-count"),
        ("equal_k", "equal-k"),
        ("EQUAL", "equal-k"),
        ("custom", "config"),
    ],
)
def test_normalize_pooling_scheme_aliases(raw: str, normalized: str) -> None:
    assert game_stats._normalize_pooling_scheme(raw) == normalized


def test_normalize_pooling_scheme_raises_for_invalid_values() -> None:
    with pytest.raises(ValueError, match="Unknown pooling scheme"):
        game_stats._normalize_pooling_scheme("mystery-weighting")
