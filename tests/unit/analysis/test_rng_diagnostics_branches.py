from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest

from farkle.analysis import rng_diagnostics
from farkle.utils.progress import ScheduledProgressLogger
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose


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
            "root_seed": [9, 9],
            "k": [2, 2],
            "shuffle_index": [1, 0],
            "game_index": [0, 0],
            "game_seed": [2, 1],
            "rng_scheme_version": [RNG_SCHEME_VERSION, RNG_SCHEME_VERSION],
            "rng_purpose_namespace": [
                int(RandomPurpose.TOURNAMENT_GAME),
                int(RandomPurpose.TOURNAMENT_GAME),
            ],
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
            cast(ds.Dataset, dataset),
            columns=list(populated.columns),
            winner_col="winner_seat",
            strat_cols=["P1_strategy", "P2_strategy"],
            batch_size=10,
            arrow_threads=1,
        )
    )
    melted = list(
        rng_diagnostics._iter_melted_batches(
            cast(ds.Dataset, dataset),
            columns=list(populated.columns),
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
            "root_seed": [9, 9, 9],
            "k": [2, 2, 2],
            "shuffle_index": [0, 1, 2],
            "game_index": [0, 0, 0],
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
            progress_logger=cast(ScheduledProgressLogger, progress_logger),
            max_matchup_groups=1,
        )

    assert melted_rows == 6
    assert not diagnostics.empty
    assert progress_logger.calls
    assert any(
        record.message == "rng-diagnostics matchup grouping capped" for record in caplog.records
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


def _prepared_hand_frame() -> pd.DataFrame:
    """Return deliberately unordered games for an independent lag oracle."""

    return pd.DataFrame(
        {
            "root_seed": [9] * 6,
            "k": [2] * 6,
            "shuffle_index": [2, 0, 1, 0, 2, 1],
            "game_index": [1, 0, 0, 1, 0, 1],
            "rng_scheme_version": [RNG_SCHEME_VERSION] * 6,
            "rng_purpose_namespace": [int(RandomPurpose.TOURNAMENT_GAME)] * 6,
            "matchup": ["A | B"] * 6,
            "n_players": [2] * 6,
            "winner_strategy": ["B", "A", "B", "B", pd.NA, "A"],
            "n_rounds": [9, 2, 1, 5, 4, 7],
            "P1_strategy": ["B", "A", "A", "B", "A", "B"],
            "P2_strategy": ["A", "B", "B", "A", "B", "A"],
        }
    )


def _prepare_arrow_frames(frames: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    columns = [
        "root_seed",
        "k",
        "shuffle_index",
        "game_index",
        "rng_scheme_version",
        "rng_purpose_namespace",
        "n_rounds",
        "winner_strategy",
        "P1_strategy",
        "P2_strategy",
    ]
    dataset = _FakeDataset([_FakeBatch(frame) for frame in frames])
    return list(
        rng_diagnostics._iter_prepared_batches(
            cast(ds.Dataset, dataset),
            columns=columns,
            winner_col="winner_strategy",
            strat_cols=["P1_strategy", "P2_strategy"],
            batch_size=2,
            arrow_threads=2,
        )
    )


def _lag_one(values: Sequence[float]) -> float:
    earlier = np.asarray(values[:-1], dtype=float)
    later = np.asarray(values[1:], dtype=float)
    return float(np.corrcoef(earlier, later)[0, 1])


def test_one_frame_semantic_sequence_matches_hand_oracle() -> None:
    frame = _prepared_hand_frame()
    prepared = _prepare_arrow_frames([frame])

    diagnostics, melted_rows = rng_diagnostics._collect_diagnostics_streaming_compact(
        prepared,
        strat_cols=["P1_strategy", "P2_strategy"],
        lags=(1,),
        progress_logger=None,
        max_matchup_groups=None,
    )

    assert melted_rows == 12
    hand_sequences = {
        "A": {
            "win_indicator": [1, 0, 0, 1, 0, 0],
            "n_rounds": [2, 5, 1, 7, 4, 9],
        },
        "B": {
            "win_indicator": [0, 1, 1, 0, 0, 1],
            "n_rounds": [2, 5, 1, 7, 4, 9],
        },
    }
    strategy_rows = diagnostics.loc[diagnostics["summary_level"] == "strategy"]
    assert len(strategy_rows) == 4
    for strategy, metrics in hand_sequences.items():
        for metric, values in metrics.items():
            row = strategy_rows.loc[
                (strategy_rows["strategy"] == strategy) & (strategy_rows["metric"] == metric)
            ].iloc[0]
            assert row["autocorr"] == pytest.approx(_lag_one(values), abs=1e-15)
            assert row["lagged_pairs"] == 5
            half_width = 1.96 / np.sqrt(5)
            assert row["zero_centered_descriptive_reference_band_lower"] == pytest.approx(
                -half_width
            )
            assert row["zero_centered_descriptive_reference_band_upper"] == pytest.approx(
                half_width
            )
            assert row["sequence_order"] == "root_seed,k,shuffle_index,game_index,seat_index"
            assert "do not establish or refute independence" in row["note"]


def test_fragmented_batches_and_seats_match_one_frame_oracle() -> None:
    frame = _prepared_hand_frame()
    one_frame, one_frame_rows = rng_diagnostics._collect_diagnostics_streaming_compact(
        [frame],
        strat_cols=["P1_strategy", "P2_strategy"],
        lags=(1, 2),
        progress_logger=None,
        max_matchup_groups=None,
    )
    fragments = [
        frame.iloc[[4, 1]].copy(),
        frame.iloc[[5]].copy(),
        frame.iloc[[0, 3]].copy(),
        frame.iloc[[2]].copy(),
    ]
    prepared_fragments = _prepare_arrow_frames(fragments)

    fragmented, fragmented_rows = rng_diagnostics._collect_diagnostics_streaming_compact(
        prepared_fragments,
        strat_cols=["P1_strategy", "P2_strategy"],
        lags=(1, 2),
        progress_logger=None,
        max_matchup_groups=None,
    )
    globally_ordered = pd.concat(
        list(
            rng_diagnostics._iter_globally_ordered_game_batches(
                prepared_fragments,
                strat_cols=["P1_strategy", "P2_strategy"],
                output_batch_size=2,
            )
        ),
        ignore_index=True,
    )
    seats = rng_diagnostics._merge_seats_in_semantic_order(
        globally_ordered,
        strat_cols=["P1_strategy", "P2_strategy"],
    )

    pd.testing.assert_frame_equal(
        fragmented.reset_index(drop=True),
        one_frame.reset_index(drop=True),
        check_exact=False,
        rtol=1e-15,
        atol=1e-15,
    )
    assert fragmented_rows == one_frame_rows == 12
    assert list(
        seats[["shuffle_index", "game_index", "seat_index"]].itertuples(index=False, name=None)
    ) == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
        (2, 0, 0),
        (2, 0, 1),
        (2, 1, 0),
        (2, 1, 1),
    ]
