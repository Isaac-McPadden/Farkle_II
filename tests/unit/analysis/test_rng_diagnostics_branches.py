from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from farkle.analysis import rng_diagnostics


class _StubProgressLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def maybe_log(self, completed: int, *, detail: str | None = None, **_: object) -> bool:
        self.calls.append((completed, detail or ""))
        return True


class _FakeBatch:
    def __init__(self, frame: pd.DataFrame, *, num_rows: int | None = None) -> None:
        self._frame = frame
        self.num_rows = len(frame) if num_rows is None else num_rows

    def to_pandas(self, *, categories: list[str] | None = None) -> pd.DataFrame:
        del categories
        return self._frame.copy()


class _FakeScanner:
    def __init__(self, batches: list[_FakeBatch]) -> None:
        self._batches = batches

    def to_batches(self):
        return iter(self._batches)


class _FakeDataset:
    def __init__(self, batches: list[_FakeBatch]) -> None:
        self._batches = batches

    def scanner(self, **_: object) -> _FakeScanner:
        return _FakeScanner(self._batches)


def test_iter_prepared_and_melted_batches_skip_empty_batches() -> None:
    populated = pd.DataFrame(
        {
            "game_seed": [2, 1],
            "n_rounds": [7, 5],
            "winner_seat": ["P2", "P1"],
            "P1_strategy": ["A", "C"],
            "P2_strategy": ["B", "D"],
        }
    )
    dataset = _FakeDataset(
        [
            _FakeBatch(pd.DataFrame(), num_rows=0),
            _FakeBatch(pd.DataFrame(columns=populated.columns), num_rows=1),
            _FakeBatch(populated),
        ]
    )

    prepared = list(
        rng_diagnostics._iter_prepared_batches(
            dataset,
            columns=["game_seed", "n_rounds", "winner_seat", "P1_strategy", "P2_strategy"],
            winner_col="winner_seat",
            strat_cols=["P1_strategy", "P2_strategy"],
            batch_size=10,
            arrow_threads=1,
        )
    )
    melted = list(
        rng_diagnostics._iter_melted_batches(
            dataset,
            columns=["game_seed", "n_rounds", "winner_seat", "P1_strategy", "P2_strategy"],
            winner_col="winner_seat",
            strat_cols=["P1_strategy", "P2_strategy"],
            batch_size=10,
            arrow_threads=1,
        )
    )

    assert len(prepared) == 1
    assert prepared[0]["winner_strategy"].tolist() == ["C", "B"]
    assert prepared[0]["matchup"].tolist() == ["C | D", "A | B"]
    assert prepared[0]["n_players"].tolist() == [2, 2]

    assert len(melted) == 1
    assert set(melted[0]["strategy"].tolist()) == {"A", "B", "C", "D"}
    assert melted[0]["win_indicator"].sum() == 2


def test_collect_diagnostics_streaming_compact_caps_matchups_and_logs_progress(
    caplog,
) -> None:
    batch = pd.DataFrame(
        {
            "matchup": ["A | B", "A | C", "A | B"],
            "n_players": [2, 2, 2],
            "winner_strategy": ["A", "A", "B"],
            "n_rounds": [5, 7, 9],
            "P1_strategy": ["A", "A", "A"],
            "P2_strategy": ["B", "C", "B"],
        }
    )
    progress_logger = _StubProgressLogger()

    with caplog.at_level(logging.WARNING):
        diagnostics, melted_rows = rng_diagnostics._collect_diagnostics_streaming_compact(
            [pd.DataFrame(), batch],
            strat_cols=["P1_strategy", "P2_strategy", "P3_strategy"],
            lags=(1,),
            progress_logger=progress_logger,
            max_matchup_groups=1,
        )

    assert melted_rows == 6
    assert not diagnostics.empty
    assert progress_logger.calls
    assert any(
        record.message == "rng-diagnostics matchup grouping capped"
        for record in caplog.records
    )


def test_accumulators_and_rows_from_group_state_cover_none_paths() -> None:
    empty_acc = rng_diagnostics._LagCorrelationAccumulator()
    assert empty_acc.autocorr() is None

    constant_acc = rng_diagnostics._LagCorrelationAccumulator()
    constant_acc.update(1.0, 1.0)
    constant_acc.update(1.0, 1.0)
    assert constant_acc.autocorr() is None

    metric_acc = rng_diagnostics._MetricStreamAccumulator((1,))
    metric_acc.extend(pd.Series([1.0, np.nan, 1.0]))
    assert metric_acc.n_obs == 2
    assert metric_acc.autocorr(1) is None

    group_state = rng_diagnostics._GroupStreamAccumulator((1,))
    group_state.extend(pd.DataFrame({"win_indicator": [1, 1], "n_rounds": [10, 10]}))
    rows = rng_diagnostics._rows_from_group_state(
        summary_level="strategy",
        strategy="A",
        n_players=2,
        lags=(1,),
        group_state=group_state,
    )

    assert rows == []


def test_collect_diagnostics_streaming_matches_batch_implementation() -> None:
    melted = pd.DataFrame(
        {
            "game_seed": [1, 2, 3, 4, 5, 6],
            "strategy": ["A"] * 6,
            "matchup": ["A | B"] * 6,
            "n_players": [2] * 6,
            "win_indicator": [1, 0, 1, 0, 1, 0],
            "n_rounds": [2, 4, 6, 8, 10, 12],
        }
    )
    progress_logger = _StubProgressLogger()

    streaming, melted_rows = rng_diagnostics._collect_diagnostics_streaming(
        [melted.iloc[:3].copy(), melted.iloc[3:].copy()],
        lags=(1, 2),
        progress_logger=progress_logger,
    )
    expected = rng_diagnostics._collect_diagnostics(melted, lags=(1, 2))

    pd.testing.assert_frame_equal(
        streaming.reset_index(drop=True),
        expected.reset_index(drop=True),
    )
    assert melted_rows == len(melted)
    assert progress_logger.calls
