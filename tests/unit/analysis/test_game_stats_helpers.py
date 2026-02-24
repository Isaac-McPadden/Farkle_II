from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from farkle.analysis import game_stats
from farkle.config import AppConfig, IOConfig, SimConfig


def _write_rows(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path)


def _fixture_rows_two_players() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 4,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 120,
                "P2_score": 110,
            },
            {
                "seat_ranks": ["P2", "P1"],
                "n_rounds": 8,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 50,
                "P2_score": 200,
            },
            {
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 12,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 300,
                "P2_score": 100,
            },
        ]
    )


def _make_per_n(tmp_path: Path, n_players: int = 2) -> Path:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2, 3]))
    rows = _fixture_rows_two_players()
    per_n = cfg.ingested_rows_curated(n_players)
    _write_rows(per_n, rows)
    return per_n


def test_strategy_conversion_helpers(tmp_path: Path) -> None:
    per_n = _make_per_n(tmp_path)
    arrow_type = game_stats._strategy_arrow_type([(2, per_n)])
    assert pa.types.is_int64(arrow_type)

    assert game_stats._strategy_numpy_dtype(pa.int8()) == np.dtype("int8")
    assert game_stats._strategy_numpy_dtype(pa.uint16()) == np.dtype("uint16")
    assert game_stats._strategy_numpy_dtype(pa.float32()) == np.dtype("int64")

    assert isinstance(game_stats._strategy_pandas_dtype(pa.uint8()), pd.UInt8Dtype)
    assert isinstance(game_stats._strategy_pandas_dtype(pa.int16()), pd.Int16Dtype)

    assert game_stats._strategy_key_to_int(np.int64(9)) == 9
    with pytest.raises(ValueError, match="invalid strategy scalar"):
        bad_value: Any = None
        game_stats._strategy_key_to_int(bad_value)

    assert game_stats._strategy_stat_value(np.float32(2.5)) == pytest.approx(2.5)
    assert game_stats._strategy_stat_value(b"abc") == "abc"
    bad_value: Any = pd.NA
    assert game_stats._strategy_stat_value(bad_value) is pd.NA

    raw = pd.DataFrame({"strategy": [1, 2, None]})
    coerced = game_stats._coerce_strategy_dtype(raw, pa.uint8())
    assert str(coerced["strategy"].dtype) == "UInt8"
    assert pd.isna(coerced.loc[2, "strategy"])


def test_pooling_helpers_and_weighted_stats() -> None:
    aliases = {
        "game-count": "game-count",
        "GameCount": "game-count",
        "count": "game-count",
        "equal_k": "equal-k",
        "equal": "equal-k",
        "config-provided": "config",
        "custom": "config",
    }
    for raw, expected in aliases.items():
        assert game_stats._normalize_pooling_scheme(raw) == expected

    with pytest.raises(ValueError, match="Unknown pooling scheme"):
        game_stats._normalize_pooling_scheme("bad")

    players = pd.Series([2, 2, 3], dtype="int64")
    w_count = game_stats._pooling_weights_for_rows(players, pooling_scheme="game-count", weights_by_k={})
    assert w_count.tolist() == [1.0, 1.0, 1.0]

    w_equal = game_stats._pooling_weights_for_rows(players, pooling_scheme="equal-k", weights_by_k={})
    assert w_equal.tolist() == [0.5, 0.5, 1.0]

    w_config = game_stats._pooling_weights_for_rows(
        players,
        pooling_scheme="config",
        weights_by_k={2: 0.4, 3: 0.6},
    )
    assert w_config.tolist() == [0.2, 0.2, 0.6]

    with pytest.raises(ValueError, match="Unknown pooling scheme"):
        game_stats._pooling_weights_for_rows(players, pooling_scheme="oops", weights_by_k={})

    assert np.isnan(game_stats._weighted_quantile(np.array([]), np.array([]), 0.5))
    assert game_stats._weighted_quantile(np.array([7.0]), np.array([1.0]), 0.5) == 7.0
    assert game_stats._weighted_quantile(np.array([1.0, 10.0]), np.array([1.0, 9.0]), 0.5) == 10.0
    assert np.isnan(game_stats._weighted_quantile(np.array([1.0]), np.array([0.0]), 0.5))

    values = np.array([1.0, 4.0, 7.0])
    weights = np.array([1.0, 2.0, 1.0])
    assert game_stats._weighted_mean(values, weights) == pytest.approx(4.0)
    assert game_stats._weighted_std(values, weights) == pytest.approx(np.sqrt(4.5))
    assert np.isnan(game_stats._weighted_mean(values, np.zeros(3)))
    assert np.isnan(game_stats._weighted_std(values, np.zeros(3)))


def test_integer_and_stat_helpers_boundaries() -> None:
    scores = np.array([[5.0, 1.0], [8.0, np.nan], [np.nan, np.nan]])
    second = game_stats._second_highest(scores)
    assert second[0] == 1
    assert np.isnan(second[1])

    hist: dict[int, int] = {}
    game_stats._update_int_histogram(hist, np.array([1.2, 1.7, np.nan, 3]))
    assert hist == {1: 1, 2: 1, 3: 1}

    counts = {1: 2, 3: 1, 10: 1}
    assert game_stats._quantile_from_counts(counts, 0.0) == 1
    assert game_stats._quantile_from_counts(counts, 0.5) == 1
    assert game_stats._quantile_from_counts(counts, 1.0) == 10
    assert game_stats._quantile_from_counts({}, 0.5) is None

    dtype_u8, arrow_u8 = game_stats._select_int_dtype(255)
    assert dtype_u8 == np.dtype(np.uint8)
    assert pa.types.is_uint8(arrow_u8)
    dtype_i32, arrow_i32 = game_stats._select_int_dtype(256)
    assert dtype_i32 == np.dtype(np.int32)
    assert pa.types.is_int32(arrow_i32)

    frame = pd.DataFrame(
        {
            "n_players": [2.0, 300.0, 12.0],
            "observations": [5.0, 2**40, 9.0],
            "other": [1.5, 2.5, 3.5],
        }
    )
    downcast = game_stats._downcast_integer_stats(frame, columns=("n_players", "observations"))
    assert downcast["n_players"].dtype == np.int32
    assert downcast["observations"].dtype == np.int64
    assert downcast["other"].dtype == np.float64

    with_nan = pd.DataFrame({"observations": [1.0, np.nan]})
    with pytest.raises(pd.errors.IntCastingNaNError):
        game_stats._downcast_integer_stats(with_nan, columns=("observations",))


def test_margin_and_round_helpers_with_edge_cases(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    scores = np.array([[10.0, 6.0], [5.0, np.nan], [np.nan, np.nan]])
    with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
        margin, spread = game_stats._compute_margin_arrays(scores)
    assert margin[0] == 4
    assert spread[0] == 4
    assert np.isnan(margin[1])
    assert np.isnan(spread[2])

    df = pd.DataFrame({"P1_score": [10, 7], "P2_score": [7, 1], "P3_score": [0, np.nan]})
    margin_cols = game_stats._compute_margin_columns(df, ["P1_score", "P2_score", "P3_score"])
    assert list(margin_cols.columns) == ["margin_runner_up", "score_spread"]
    assert margin_cols["margin_runner_up"].tolist()[0] == 3

    empty_margin = game_stats._summarize_margins([], thresholds=(50,))
    assert empty_margin["observations"] == 0
    prob_margin: Any = empty_margin["prob_margin_runner_up_le_50"]
    assert np.isnan(prob_margin)

    single_margin = game_stats._summarize_margins([25], thresholds=(10, 25))
    assert single_margin["observations"] == 1
    assert single_margin["prob_margin_runner_up_le_25"] == 1.0

    mixed_round_values: Any = [1, "2", np.nan, 20]
    mixed_rounds = game_stats._summarize_rounds(mixed_round_values)
    assert mixed_rounds["observations"] == 3
    assert mixed_rounds["prob_rounds_ge_20"] == pytest.approx(1 / 3)

    combined = tmp_path / "combined.parquet"
    _write_rows(combined, _fixture_rows_two_players())
    global_stats = game_stats._global_stats(combined)
    assert not global_stats.empty
    assert set(global_stats["summary_level"]) == {"n_players"}

    class DummyDataset:
        schema = type("Schema", (), {"names": ["n_rounds"]})()

        @staticmethod
        def to_table(_columns=None):
            return pa.Table.from_pandas(pd.DataFrame({"n_rounds": [1, 2]}))

    with caplog.at_level("WARNING"):
        out = pytest.MonkeyPatch.context()
        with out as mp:
            mp.setattr(game_stats.ds, "dataset", lambda _path: DummyDataset())
            out_df = game_stats._global_stats(Path("dummy"))
    assert out_df.empty
    assert "missing seat_ranks" in caplog.text


def test_rare_event_helpers_and_threshold_resolution(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))
    rows = _fixture_rows_two_players()
    per_n = cfg.ingested_rows_curated(2)
    _write_rows(per_n, rows)

    per_n_inputs = [(2, per_n)]
    rare_path = tmp_path / "rare.parquet"
    detail_path = tmp_path / "rare_detail.parquet"
    summary_path = tmp_path / "rare_summary.parquet"

    flags_written = game_stats._rare_event_flags(
        per_n_inputs,
        thresholds=(25, 100),
        target_score=150,
        output_path=rare_path,
        codec=cfg.parquet_codec,
    )
    details_written = game_stats._rare_event_details(
        per_n_inputs,
        thresholds=(25, 100),
        target_score=150,
        output_path=detail_path,
        codec=cfg.parquet_codec,
    )
    summary_written = game_stats._rare_event_summary(
        per_n_inputs,
        thresholds=(25, 100),
        target_score=150,
        output_path=summary_path,
        codec=cfg.parquet_codec,
    )

    assert flags_written > 0
    assert details_written > 0
    assert summary_written > 0
    assert set(pd.read_parquet(rare_path)["summary_level"]) >= {"game", "strategy", "n_players"}
    assert set(pd.read_parquet(detail_path)["summary_level"]) == {"game"}
    assert set(pd.read_parquet(summary_path)["summary_level"]) >= {"strategy", "n_players"}

    strat_sums, global_sums, rows_available, max_flag, max_obs = game_stats._collect_rare_event_counts(
        per_n_inputs,
        thresholds=(25,),
        target_score=150,
    )
    assert rows_available > 0
    assert max_flag <= max_obs
    assert (1, 2) in strat_sums
    assert 2 in global_sums

    margins, targets = game_stats._collect_rare_event_histograms(
        per_n_inputs,
        need_margins=True,
        need_targets=True,
    )
    assert margins
    assert targets

    resolved = game_stats._resolve_rare_event_thresholds(
        per_n_inputs,
        thresholds=(5,),
        target_score=400,
        margin_quantile=0.5,
        target_rate=0.5,
    )
    assert resolved[0] != (5,)
    assert resolved[1] != 400

    explicit = game_stats._resolve_rare_event_thresholds(
        per_n_inputs,
        thresholds=(7, 9),
        target_score=250,
        margin_quantile=None,
        target_rate=None,
    )
    assert explicit == ((7, 9), 250)

    # No score columns -> histograms absent; thresholds should stay explicit.
    no_scores = cfg.ingested_rows_curated(3)
    _write_rows(no_scores, pd.DataFrame({"P1_strategy": [1], "n_rounds": [1]}))
    fallback = game_stats._resolve_rare_event_thresholds(
        [(3, no_scores)],
        thresholds=(11,),
        target_score=222,
        margin_quantile=0.8,
        target_rate=0.2,
    )
    assert fallback == ((11,), 222)

    with pytest.raises(ValueError, match="between 0 and 1"):
        game_stats._resolve_rare_event_thresholds(
            per_n_inputs,
            thresholds=(1,),
            target_score=10,
            margin_quantile=1.0,
            target_rate=None,
        )

def test_additional_branch_coverage_paths(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    # strategy type fallback
    no_strategy = tmp_path / "nostrategy.parquet"
    _write_rows(no_strategy, pd.DataFrame({"n_rounds": [1, 2]}))
    assert pa.types.is_int64(game_stats._strategy_arrow_type([(2, no_strategy)]))

    # dtype branches
    assert isinstance(game_stats._strategy_pandas_dtype(pa.uint16()), pd.UInt16Dtype)
    assert isinstance(game_stats._strategy_pandas_dtype(pa.uint32()), pd.UInt32Dtype)
    assert isinstance(game_stats._strategy_pandas_dtype(pa.uint64()), pd.UInt64Dtype)
    assert isinstance(game_stats._strategy_pandas_dtype(pa.int8()), pd.Int8Dtype)
    assert isinstance(game_stats._strategy_pandas_dtype(pa.int32()), pd.Int32Dtype)
    assert isinstance(game_stats._strategy_pandas_dtype(pa.float32()), pd.Int64Dtype)
    assert game_stats._strategy_numpy_dtype(pa.int16()) == np.dtype("int16")
    assert game_stats._strategy_numpy_dtype(pa.int32()) == np.dtype("int32")
    assert game_stats._strategy_numpy_dtype(pa.uint8()) == np.dtype("uint8")
    assert game_stats._strategy_numpy_dtype(pa.uint32()) == np.dtype("uint32")
    assert game_stats._strategy_numpy_dtype(pa.uint64()) == np.dtype("uint64")

    assert game_stats._strategy_stat_value(True) == 1
    bad_value: Any = object()
    strategy_repr = game_stats._strategy_stat_value(bad_value)
    assert isinstance(strategy_repr, str)
    assert strategy_repr.startswith("<object object")

    # coerce early returns
    assert game_stats._coerce_strategy_dtype(pd.DataFrame(), pa.int64()).empty
    no_col = pd.DataFrame({"x": [1]})
    pd.testing.assert_frame_equal(game_stats._coerce_strategy_dtype(no_col, pa.int64()), no_col)

    # pooling config missing keys warning
    with caplog.at_level("WARNING"):
        weights = game_stats._pooling_weights_for_rows(
            pd.Series([2, 4]), pooling_scheme="config", weights_by_k={2: 1.0}
        )
    assert weights.tolist() == [1.0, 0.0]
    assert "Missing pooling weights" in caplog.text

    assert game_stats._weighted_quantile(np.array([3.0, 6.0]), np.array([1.0, 1.0]), 0.0) == 3.0
    assert game_stats._weighted_quantile(np.array([3.0, 6.0]), np.array([1.0, 1.0]), 1.0) == 6.0

    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))
    rows = _fixture_rows_two_players()
    per_n = cfg.ingested_rows_curated(2)
    _write_rows(per_n, rows)

    pooled = game_stats._pooled_strategy_stats([(2, per_n)], pooling_scheme="equal-k", weights_by_k={})
    assert not pooled.empty
    pooled_margin = game_stats._pooled_margin_stats(
        [(2, per_n)], thresholds=(10, 100), pooling_scheme="equal-k", weights_by_k={}
    )
    assert not pooled_margin.empty

    # weights zero -> rows filtered out (weight_total <= 0)
    pooled_zero = game_stats._pooled_strategy_stats([(2, per_n)], pooling_scheme="config", weights_by_k={})
    assert pooled_zero.empty

    with caplog.at_level("WARNING"):
        assert game_stats._per_strategy_stats([(2, no_strategy)]).empty
        assert game_stats._per_strategy_margin_stats([(2, no_strategy)], thresholds=(10,)).empty
    assert "missing strategy columns" in caplog.text

    no_score = tmp_path / "noscore.parquet"
    _write_rows(no_score, pd.DataFrame({"P1_strategy": [1], "P2_strategy": [2]}))
    with caplog.at_level("WARNING"):
        margin_empty = game_stats._per_strategy_margin_stats([(2, no_score)], thresholds=(10,))
    assert margin_empty.empty
    assert "missing seat score columns" in caplog.text

    assert game_stats._rare_event_flags(
        [(2, no_score)], thresholds=(10,), target_score=100, output_path=tmp_path / "x.parquet", codec=cfg.parquet_codec
    ) == 0
    assert game_stats._rare_event_details(
        [(2, no_score)], thresholds=(10,), target_score=100, output_path=tmp_path / "y.parquet", codec=cfg.parquet_codec
    ) == 0
    assert game_stats._rare_event_summary(
        [(2, no_score)], thresholds=(10,), target_score=100, output_path=tmp_path / "z.parquet", codec=cfg.parquet_codec
    ) == 0

def test_low_level_numeric_edge_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_scores = np.empty((0, 0))
    margin, spread = game_stats._compute_margin_arrays(empty_scores)
    assert margin.size == 0 and spread.size == 0

    assert game_stats._second_highest(np.empty((0, 0))).size == 0
    single_col = game_stats._second_highest(np.array([[5.0], [np.nan]]))
    assert np.isnan(single_col).all()

    counts: dict[int, int] = {}
    game_stats._update_int_histogram(counts, np.array([]))
    game_stats._update_int_histogram(counts, np.array([np.nan]))
    assert counts == {}

    assert game_stats._quantile_from_counts({1: 0, 2: 0}, 0.5) is None
    assert game_stats._select_int_dtype(2**40)[0] == np.dtype(np.int64)

    assert game_stats._downcast_integer_stats(pd.DataFrame(), columns=("x",)).empty
    nonint = pd.DataFrame({"x": [1.5, 2.5]})
    unchanged = game_stats._downcast_integer_stats(nonint, columns=("x", "missing"))
    assert unchanged["x"].dtype == np.float64

    class DummyDataset:
        schema = type("Schema", (), {"names": ["n_rounds", "seat_ranks", "P1_score"]})()

        @staticmethod
        def to_table(_columns=None):
            return pa.Table.from_pandas(
                pd.DataFrame(
                    {
                        "n_rounds": [3, 5],
                        "seat_ranks": [["P1", "P2"], None],
                    }
                )
            )

    monkeypatch.setattr(game_stats.ds, "dataset", lambda _path: DummyDataset())
    out = game_stats._global_stats(Path("dummy"))
    assert not out.empty
    assert set(out["n_players"].astype(int).tolist()) == {1, 2}

def test_margin_pooling_and_strategy_empty_paths(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))

    no_strategy = tmp_path / "m_no_strategy.parquet"
    _write_rows(no_strategy, pd.DataFrame({"P1_score": [10], "P2_score": [8]}))
    no_score = tmp_path / "m_no_score.parquet"
    _write_rows(no_score, pd.DataFrame({"P1_strategy": [1], "P2_strategy": [2]}))
    nan_scores = tmp_path / "m_nan_scores.parquet"
    _write_rows(
        nan_scores,
        pd.DataFrame(
            {
                "P1_strategy": [1, 1],
                "P2_strategy": [2, 2],
                "P1_score": [np.nan, np.nan],
                "P2_score": [np.nan, np.nan],
            }
        ),
    )

    with caplog.at_level("WARNING"):
        pooled_missing_strategy = game_stats._pooled_margin_stats(
            [(2, no_strategy)], thresholds=(10,), pooling_scheme="equal-k", weights_by_k={}
        )
        pooled_missing_score = game_stats._pooled_margin_stats(
            [(2, no_score)], thresholds=(10,), pooling_scheme="equal-k", weights_by_k={}
        )
    assert pooled_missing_strategy.empty
    assert pooled_missing_score.empty

    with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
        pooled_nan = game_stats._pooled_margin_stats(
            [(2, nan_scores)], thresholds=(10,), pooling_scheme="equal-k", weights_by_k={}
        )
    with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
        per_strategy_nan = game_stats._per_strategy_margin_stats([(2, nan_scores)], thresholds=(10,))
    assert pooled_nan.empty
    assert per_strategy_nan.empty
