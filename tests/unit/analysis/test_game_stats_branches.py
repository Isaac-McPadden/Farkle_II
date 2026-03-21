from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis import game_stats
from farkle.config import AnalysisConfig, AppConfig, IOConfig, SimConfig


@pytest.mark.parametrize(
    ("arrow_type", "numpy_dtype", "pandas_dtype"),
    [
        (pa.int8(), np.dtype("int8"), pd.Int8Dtype),
        (pa.int16(), np.dtype("int16"), pd.Int16Dtype),
        (pa.int32(), np.dtype("int32"), pd.Int32Dtype),
        (pa.int64(), np.dtype("int64"), pd.Int64Dtype),
        (pa.uint8(), np.dtype("uint8"), pd.UInt8Dtype),
        (pa.uint16(), np.dtype("uint16"), pd.UInt16Dtype),
        (pa.uint32(), np.dtype("uint32"), pd.UInt32Dtype),
        (pa.uint64(), np.dtype("uint64"), pd.UInt64Dtype),
        (pa.string(), np.dtype("int64"), pd.Int64Dtype),
    ],
)
def test_strategy_dtype_helpers_cover_all_numeric_paths(
    arrow_type: pa.DataType,
    numpy_dtype: np.dtype[np.generic],
    pandas_dtype: type[pd.api.extensions.ExtensionDtype],
) -> None:
    assert game_stats._strategy_numpy_dtype(arrow_type) == numpy_dtype
    assert isinstance(game_stats._strategy_pandas_dtype(arrow_type), pandas_dtype)


def test_strategy_arrow_type_and_scalar_helpers_cover_edge_cases(tmp_path) -> None:
    parquet_path = tmp_path / "rows.parquet"
    pq.write_table(
        pa.Table.from_pydict({"P1_strategy": pa.array([1, 2], type=pa.uint8())}),
        parquet_path,
    )
    no_strategy_path = tmp_path / "no_strategy.parquet"
    pq.write_table(pa.Table.from_pydict({"value": [1]}), no_strategy_path)

    assert game_stats._strategy_arrow_type([(2, parquet_path)]) == pa.uint8()
    assert game_stats._strategy_arrow_type([(2, no_strategy_path)]) == pa.int64()

    assert game_stats._to_python_scalar((np.int64(7),), field="strategy") == 7
    assert game_stats._to_python_scalar(pd.NA) is None
    with pytest.raises(ValueError, match="invalid strategy scalar"):
        game_stats._to_python_scalar((1, 2), field="strategy")

    assert game_stats._require_scalar(np.int64(9), field="value") == 9
    with pytest.raises(ValueError, match="invalid value scalar"):
        game_stats._require_scalar(pd.NA, field="value")

    assert game_stats._strategy_key_to_int(np.int64(4), field="strategy") == 4
    with pytest.raises(ValueError, match="invalid strategy scalar"):
        game_stats._strategy_key_to_int("not-an-int", field="strategy")

    class CustomValue:
        def __str__(self) -> str:
            return "custom-value"

    assert game_stats._strategy_stat_value(b"abc") == "abc"
    assert game_stats._strategy_stat_value("xyz") == "xyz"
    assert game_stats._strategy_stat_value(True) == 1
    assert game_stats._strategy_stat_value(3) == 3
    assert game_stats._strategy_stat_value(3.5) == 3.5
    assert game_stats._strategy_stat_value(CustomValue()) == "custom-value"


def test_coerce_strategy_dtype_and_run_per_k_fanout_behaviors() -> None:
    frame = pd.DataFrame({"strategy": ["1", "2", None], "other": [1, 2, 3]})
    coerced = game_stats._coerce_strategy_dtype(frame, pa.uint8())
    assert str(coerced["strategy"].dtype) == "UInt8"

    empty = pd.DataFrame()
    assert game_stats._coerce_strategy_dtype(empty, pa.uint8()).empty
    no_strategy = pd.DataFrame({"other": [1]})
    pd.testing.assert_frame_equal(
        game_stats._coerce_strategy_dtype(no_strategy, pa.uint8()),
        no_strategy,
    )

    assert game_stats._run_per_k_fanout([]) is None

    succeeded = Future()
    succeeded.set_result(None)
    assert game_stats._run_per_k_fanout([succeeded]) is None

    failed = Future()
    failed.set_exception(RuntimeError("boom"))
    pending = Future()

    with pytest.raises(RuntimeError, match="boom"):
        game_stats._run_per_k_fanout([pending, failed])

    assert pending.cancelled()


def test_global_stats_to_table_fallback_filters_non_integer_player_counts(
    monkeypatch,
) -> None:
    table = pa.Table.from_pydict(
        {
            "seat_ranks": [["P1", "P2"], ["P1", "P2"], ["P1", "P2", "P3"]],
            "n_rounds": [4, 8, 12],
            "n_players": [2.0, 2.5, float("inf")],
        }
    )

    class DummyDataset:
        schema = type("Schema", (), {"names": ["seat_ranks", "n_rounds", "n_players"]})()

        @staticmethod
        def to_table(*args, **kwargs):
            if kwargs:
                raise TypeError("legacy positional API")
            return table

    monkeypatch.setattr(game_stats.ds, "dataset", lambda _path: DummyDataset())
    monkeypatch.setattr(game_stats, "n_players_from_schema", lambda _schema: 99)

    result = game_stats._global_stats(pa.scalar("unused"))

    assert set(result["n_players"].astype(int).tolist()) == {2}
    assert result.loc[result["n_players"] == 2, "observations"].item() == 1


@dataclass
class _DummyStageLog:
    started: bool = False
    missing_calls: list[tuple[str, dict[str, object]]] | None = None

    def __post_init__(self) -> None:
        if self.missing_calls is None:
            self.missing_calls = []

    def start(self) -> None:
        self.started = True

    def missing_input(self, message: str, **kwargs: object) -> None:
        assert self.missing_calls is not None
        self.missing_calls.append((message, kwargs))


def _make_run_cfg(
    tmp_path: Path,
    *,
    n_players_list: list[int] | None = None,
    n_jobs: int = 1,
    rare_event_write_details: bool = False,
) -> AppConfig:
    return AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(n_players_list=n_players_list or [2, 3], seed=7),
        analysis=AnalysisConfig(
            n_jobs=n_jobs,
            rare_event_write_details=rare_event_write_details,
        ),
    )


def test_run_logs_missing_input_when_no_curated_rows_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_log = _DummyStageLog()
    cfg = _make_run_cfg(tmp_path)

    monkeypatch.setattr(game_stats, "stage_logger", lambda *args, **kwargs: stage_log)
    monkeypatch.setattr(game_stats, "_discover_per_n_inputs", lambda _cfg: [])

    game_stats.run(cfg, force=False)

    assert stage_log.started is True
    assert stage_log.missing_calls == [
        ("no curated parquet files found", {"analysis_dir": str(cfg.analysis_dir)})
    ]


def test_run_covers_parallel_empty_output_and_rare_event_detail_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_log = _DummyStageLog()
    cfg = _make_run_cfg(tmp_path, n_jobs=4, rare_event_write_details=True)
    per_n_path = tmp_path / "inputs" / "rows_2p.parquet"
    per_n_path.parent.mkdir(parents=True, exist_ok=True)
    per_n_path.write_text("stub", encoding="utf-8")

    compute_calls: list[int] = []
    empty_output_calls: list[int] = []
    pooled_calls: list[tuple[int, ...]] = []
    rare_calls: list[tuple[str, int | None]] = []
    done_calls: list[Path] = []

    monkeypatch.setattr(game_stats, "stage_logger", lambda *args, **kwargs: stage_log)
    monkeypatch.setattr(game_stats, "_discover_per_n_inputs", lambda _cfg: [(2, per_n_path)])
    monkeypatch.setattr(game_stats, "_resolve_analysis_workers", lambda _cfg: 4)
    monkeypatch.setattr(game_stats, "stage_is_up_to_date", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        game_stats,
        "_write_empty_per_k_outputs",
        lambda **kwargs: empty_output_calls.append(kwargs["k"]),
    )
    monkeypatch.setattr(
        game_stats,
        "_compute_k_game_stats",
        lambda **kwargs: compute_calls.append(kwargs["k"]),
    )
    monkeypatch.setattr(
        game_stats,
        "_pool_completed_k_game_stats",
        lambda **kwargs: pooled_calls.append(tuple(kwargs["configured_k_values"])),
    )
    monkeypatch.setattr(
        game_stats,
        "_resolve_rare_event_thresholds",
        lambda *args, **kwargs: ((11,), 222),
    )
    monkeypatch.setattr(
        game_stats,
        "_rare_event_flags",
        lambda *args, **kwargs: rare_calls.append(("summary", kwargs["n_workers"])) or 5,
    )
    monkeypatch.setattr(
        game_stats,
        "_rare_event_details",
        lambda *args, **kwargs: rare_calls.append(("details", None)) or 3,
    )
    monkeypatch.setattr(
        game_stats,
        "write_stage_done",
        lambda path, **kwargs: done_calls.append(path),
    )

    game_stats.run(cfg, force=False)

    assert stage_log.started is True
    assert empty_output_calls == [3]
    assert compute_calls == [2]
    assert pooled_calls == [tuple(cfg.agreement_players())]
    assert rare_calls == [("summary", 2), ("details", None)]
    assert len(done_calls) == 3


def test_run_raises_when_rare_event_summary_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_log = _DummyStageLog()
    cfg = _make_run_cfg(tmp_path, n_jobs=1, rare_event_write_details=False, n_players_list=[2])
    per_n_path = tmp_path / "inputs" / "rows_2p.parquet"
    per_n_path.parent.mkdir(parents=True, exist_ok=True)
    per_n_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(game_stats, "stage_logger", lambda *args, **kwargs: stage_log)
    monkeypatch.setattr(game_stats, "_discover_per_n_inputs", lambda _cfg: [(2, per_n_path)])
    monkeypatch.setattr(game_stats, "_resolve_analysis_workers", lambda _cfg: 1)
    monkeypatch.setattr(game_stats, "stage_is_up_to_date", lambda *args, **kwargs: False)
    monkeypatch.setattr(game_stats, "_compute_k_game_stats", lambda **kwargs: None)
    monkeypatch.setattr(game_stats, "_pool_completed_k_game_stats", lambda **kwargs: None)
    monkeypatch.setattr(
        game_stats,
        "_resolve_rare_event_thresholds",
        lambda *args, **kwargs: ((9,), 111),
    )
    monkeypatch.setattr(game_stats, "_rare_event_flags", lambda *args, **kwargs: 0)

    with pytest.raises(RuntimeError, match="no rare events available to summarize"):
        game_stats.run(cfg, force=False)


def test_run_raises_when_rare_event_details_are_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_log = _DummyStageLog()
    cfg = _make_run_cfg(tmp_path, n_jobs=1, rare_event_write_details=True, n_players_list=[2])
    per_n_path = tmp_path / "inputs" / "rows_2p.parquet"
    per_n_path.parent.mkdir(parents=True, exist_ok=True)
    per_n_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(game_stats, "stage_logger", lambda *args, **kwargs: stage_log)
    monkeypatch.setattr(game_stats, "_discover_per_n_inputs", lambda _cfg: [(2, per_n_path)])
    monkeypatch.setattr(game_stats, "_resolve_analysis_workers", lambda _cfg: 1)
    monkeypatch.setattr(game_stats, "stage_is_up_to_date", lambda *args, **kwargs: False)
    monkeypatch.setattr(game_stats, "_compute_k_game_stats", lambda **kwargs: None)
    monkeypatch.setattr(game_stats, "_pool_completed_k_game_stats", lambda **kwargs: None)
    monkeypatch.setattr(
        game_stats,
        "_resolve_rare_event_thresholds",
        lambda *args, **kwargs: ((9,), 111),
    )
    monkeypatch.setattr(game_stats, "_rare_event_flags", lambda *args, **kwargs: 1)
    monkeypatch.setattr(game_stats, "_rare_event_details", lambda *args, **kwargs: 0)
    monkeypatch.setattr(game_stats, "write_stage_done", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="no rare events available to detail"):
        game_stats.run(cfg, force=False)


def test_compute_k_game_stats_handles_no_strategy_and_checkpoint_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyBatch:
        def __init__(self, frame: pd.DataFrame) -> None:
            self._frame = frame
            self.num_rows = len(frame)

        def to_pandas(self, *, categories=None) -> pd.DataFrame:
            return self._frame.copy()

    class DummyScanner:
        def __init__(self, batches: list[DummyBatch]) -> None:
            self._batches = batches

        def to_batches(self):
            return iter(self._batches)

    class DummyDataset:
        def __init__(self, names: list[str], batches: list[DummyBatch], total_rows: int) -> None:
            self.schema = type("Schema", (), {"names": names})()
            self._batches = batches
            self._total_rows = total_rows

        def count_rows(self) -> int:
            return self._total_rows

        def scanner(self, *, columns=None, batch_size=None) -> DummyScanner:
            return DummyScanner(self._batches)

    progress_calls: list[int] = []

    class DummyProgressLogger:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def maybe_log(self, processed_rows: int, *, detail: str, extra: dict[str, object]) -> None:
            assert "batches" in detail
            assert "strategy_groups" in extra
            progress_calls.append(processed_rows)

    artifact_writes: list[Path] = []
    legacy_calls: list[int] = []
    stage_done_calls: list[Path] = []
    stage_dir = tmp_path / "stage"
    input_path = tmp_path / "input.parquet"

    no_strategy_ds = DummyDataset(["n_rounds"], [], 0)
    monkeypatch.setattr(game_stats.ds, "dataset", lambda _path: no_strategy_ds)
    game_stats._compute_k_game_stats(
        k=2,
        input_path=input_path,
        stage_dir=stage_dir,
        thresholds=(10,),
        codec="zstd",
        config_sha="cfg",
        stage_config_sha="stage",
        cache_key_version=1,
    )
    assert not any(stage_dir.rglob("*.parquet"))

    populated_batches = [DummyBatch(pd.DataFrame())]
    populated_batches.extend(
        [
            DummyBatch(
                pd.DataFrame(
                    {
                        "n_rounds": [3],
                        "P1_strategy": [1],
                        "P2_strategy": [2],
                        "P1_score": [100],
                        "P2_score": [90],
                    }
                )
            )
            for _ in range(25)
        ]
    )
    populated_ds = DummyDataset(
        ["n_rounds", "P1_strategy", "P2_strategy", "P1_score", "P2_score"],
        populated_batches,
        total_rows=25,
    )
    monkeypatch.setattr(game_stats.ds, "dataset", lambda _path: populated_ds)
    monkeypatch.setattr(game_stats, "ScheduledProgressLogger", DummyProgressLogger)
    monkeypatch.setattr(
        game_stats,
        "write_parquet_atomic",
        lambda table, path, codec: artifact_writes.append(path),
    )
    monkeypatch.setattr(
        game_stats,
        "_write_per_k_game_length",
        lambda **kwargs: legacy_calls.append(kwargs["k"]),
    )
    monkeypatch.setattr(
        game_stats,
        "_write_per_k_margin",
        lambda **kwargs: legacy_calls.append(kwargs["k"]),
    )
    monkeypatch.setattr(
        game_stats,
        "write_stage_done",
        lambda path, **kwargs: stage_done_calls.append(path),
    )

    game_stats._compute_k_game_stats(
        k=2,
        input_path=input_path,
        stage_dir=stage_dir,
        thresholds=(10, 20),
        codec="zstd",
        config_sha="cfg",
        stage_config_sha="stage",
        cache_key_version=1,
    )

    checkpoint_path = (stage_dir / "2p" / "game_stats.2p.parquet").with_suffix(".checkpoint.json")
    assert checkpoint_path.exists()
    assert checkpoint_path.read_text(encoding="utf-8") == '{"batches": 25, "k": 2}'
    assert artifact_writes == [stage_dir / "2p" / "game_stats.2p.parquet"]
    assert legacy_calls == [2, 2]
    assert stage_done_calls == [stage_dir / "2p" / "game_stats.2p.done.json"]
    assert progress_calls[-1] == 25


def test_low_level_stat_helper_edge_case_branches() -> None:
    unweighted = game_stats._UnweightedAccumulator()
    game_stats._update_unweighted_accumulator(unweighted, np.array([np.nan, np.inf]))
    assert unweighted.count == 0
    game_stats._update_unweighted_accumulator(unweighted, np.array([1.0, 2.0, 2.0]))
    assert unweighted.hist == {1.0: 1, 2.0: 2}

    assert game_stats._hist_value_at_rank([(1.0, 1)], 10) == 1.0
    assert np.isnan(game_stats._quantile_linear_from_hist({}, count=0, quantile=0.5))
    assert game_stats._quantile_linear_from_hist({1.0: 1, 3.0: 1}, count=2, quantile=0.0) == 1.0
    assert game_stats._quantile_linear_from_hist({1.0: 1, 3.0: 1}, count=2, quantile=1.0) == 3.0
    assert np.isnan(game_stats._probability_le_from_hist({1.0: 1}, count=0, threshold=1.0))
    assert np.isnan(game_stats._probability_ge_from_hist({1.0: 1}, count=0, threshold=1.0))
    assert all(np.isnan(value) for value in game_stats._mean_std_from_unweighted(game_stats._UnweightedAccumulator()))

    weighted = game_stats._WeightedAccumulator()
    game_stats._update_weighted_accumulator(weighted, np.array([np.nan]), row_weight=1.0)
    game_stats._update_weighted_accumulator(weighted, np.array([1.0, 2.0]), row_weight=float("nan"))
    game_stats._update_weighted_accumulator(weighted, np.array([1.0, 3.0]), row_weight=0.5)
    assert weighted.hist == {1.0: 0.5, 3.0: 0.5}
    assert np.isnan(game_stats._weighted_quantile_from_hist({}, weight_total=0.0, quantile=0.5))
    assert (
        game_stats._weighted_quantile_from_hist({1.0: 1.0, 5.0: 2.0}, weight_total=3.0, quantile=0.0)
        == 1.0
    )
    assert (
        game_stats._weighted_quantile_from_hist({1.0: 1.0, 5.0: 2.0}, weight_total=3.0, quantile=1.0)
        == 5.0
    )
    assert game_stats._weighted_quantile_from_hist({1.0: 1.0}, weight_total=2.0, quantile=0.75) == 1.0
    assert np.isnan(game_stats._weighted_probability_le_from_hist({1.0: 1.0}, weight_total=0.0, threshold=1.0))
    assert all(np.isnan(value) for value in game_stats._mean_std_from_weighted(game_stats._WeightedAccumulator()))

    binned = game_stats._BinnedAccumulator()
    game_stats._update_binned_accumulator(binned, np.array([np.nan]))
    assert binned.count == 0
    game_stats._update_binned_accumulator(binned, np.array([10.0, 40.0]))
    assert all(np.isnan(value) for value in game_stats._mean_std_from_binned(game_stats._BinnedAccumulator()))
    assert np.isnan(game_stats._quantile_from_binned(game_stats._BinnedAccumulator(), 0.5))
    assert game_stats._quantile_from_binned(binned, 0.0) == 10.0
    assert game_stats._quantile_from_binned(binned, 1.0) == 40.0
    corrupt_binned = game_stats._BinnedAccumulator(
        count=5,
        total=0.0,
        total_sq=0.0,
        min_value=0.0,
        max_value=50.0,
        bins={0: 1},
    )
    assert game_stats._quantile_from_binned(corrupt_binned, 0.9) == 50.0
    assert np.isnan(game_stats._probability_le_from_binned(binned, float("nan")))

    assert (
        game_stats._pooling_row_weight(
            pooling_scheme="game-count",
            n_players=2,
            row_count=0,
            weights_by_k={},
        )
        == 0.0
    )
    assert (
        game_stats._pooling_row_weight(
            pooling_scheme="game-count",
            n_players=2,
            row_count=3,
            weights_by_k={},
        )
        == 1.0
    )
    assert (
        game_stats._pooling_row_weight(
            pooling_scheme="equal-k",
            n_players=2,
            row_count=4,
            weights_by_k={},
        )
        == 0.25
    )
    assert (
        game_stats._pooling_row_weight(
            pooling_scheme="config",
            n_players=3,
            row_count=2,
            weights_by_k={3: 0.6},
        )
        == pytest.approx(0.3)
    )
    with pytest.raises(ValueError, match="Unknown pooling scheme"):
        game_stats._pooling_row_weight(
            pooling_scheme="bad",
            n_players=2,
            row_count=1,
            weights_by_k={},
        )
