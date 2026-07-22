"""Hand-checkable engine and raw-row safety-limit outcome oracles."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.game.engine import FarkleGame, FarklePlayer, TerminationStatus
from farkle.simulation.run_tournament import _require_outcome
from farkle.simulation.simulation import (
    _play_game,
    simulation_rows_to_table,
    validate_simulation_row,
)
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION


def _strategies(count: int) -> list[ThresholdStrategy]:
    return [
        ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=100 + seat)
        for seat in range(count)
    ]


def _players(count: int) -> list[FarklePlayer]:
    return [
        FarklePlayer(
            f"P{seat + 1}",
            strategy,
            rng=np.random.Generator(np.random.PCG64DXSM(1_000 + seat)),
        )
        for seat, strategy in enumerate(_strategies(count))
    ]


def _scripted_turns(points: dict[str, int]) -> Callable[..., None]:
    def take_turn(
        player: FarklePlayer,
        target_score: int,  # noqa: ARG001
        *,
        final_round: bool = False,  # noqa: ARG001
        score_to_beat: int = 0,  # noqa: ARG001
    ) -> None:
        points_scored = points[player.name]
        player.n_turns += 1
        player.n_rolls += 2
        player.n_farkles += 1
        player.n_hot_dice += 1
        player.highest_turn = max(player.highest_turn, points_scored)
        player.score += points_scored

    return take_turn


def _provenance(k: int) -> dict[str, int]:
    return {
        "root_seed": 17,
        "k": k,
        "shuffle_index": 3,
        "game_index": 4,
        "deterministic_batch_id": 1,
        "shuffle_seed": 99,
        "game_seed": 123,
        "rng_scheme_version": RNG_SCHEME_VERSION,
        "rng_purpose_namespace": int(RandomPurpose.TOURNAMENT_GAME),
    }


def test_forced_zero_zero_two_player_safety_limit_has_no_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FarklePlayer, "take_turn", _scripted_turns({"P1": 0, "P2": 0}))

    metrics = FarkleGame(_players(2), target_score=100).play(max_rounds=3)

    assert metrics.game.termination_status is TerminationStatus.SAFETY_LIMIT
    assert metrics.game.hit_safety_limit is True
    assert metrics.winner is None
    assert metrics.winning_score is None
    assert metrics.game.margin is None
    assert metrics.game.n_rounds == 3
    assert set(metrics.players) == {"P1", "P2"}
    for stats in metrics.players.values():
        assert stats.score == 0
        assert stats.rank is None
        assert stats.loss_margin is None
        assert stats.n_turns == 3
        assert stats.rolls == 6
        assert stats.farkles == 3
        assert stats.hot_dice == 3
        assert stats.hit_max_rounds == 1


def test_non_tied_safety_limit_still_has_no_winner_or_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FarklePlayer, "take_turn", _scripted_turns({"P1": 2, "P2": 1}))

    metrics = FarkleGame(_players(2), target_score=100).play(max_rounds=3)

    assert [metrics.players[seat].score for seat in ("P1", "P2")] == [6, 3]
    assert metrics.winner is None
    assert metrics.game.termination_status is TerminationStatus.SAFETY_LIMIT
    assert [metrics.players[seat].rank for seat in ("P1", "P2")] == [None, None]


@pytest.mark.parametrize("n_players", [2, 3])
def test_normal_completed_games_have_one_winner_and_permutation_ranks(
    monkeypatch: pytest.MonkeyPatch, n_players: int
) -> None:
    points = {"P1": 100, "P2": 200, "P3": 50}
    monkeypatch.setattr(FarklePlayer, "take_turn", _scripted_turns(points))

    metrics = FarkleGame(_players(n_players), target_score=100).play(max_rounds=3)
    ranks = [metrics.players[f"P{seat}"].rank for seat in range(1, n_players + 1)]

    assert metrics.game.termination_status is TerminationStatus.COMPLETED
    assert metrics.game.hit_safety_limit is False
    assert metrics.winner == "P2"
    assert metrics.winning_score == 200
    assert ranks.count(1) == 1
    assert sorted(ranks) == list(range(1, n_players + 1))


def test_nullable_safety_outcome_round_trips_through_typed_parquet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(FarklePlayer, "take_turn", _scripted_turns({"P1": 0, "P2": 0}))
    row = dict(
        _play_game(
            123,
            _strategies(2),
            target_score=100,
            provenance=_provenance(2),
            max_rounds=2,
        )
    )

    assert row["termination_status"] == "safety_limit"
    assert row["hit_safety_limit"] is True
    assert row["outcome_schema_version"] == OUTCOME_SCHEMA_VERSION
    assert row["winner_seat"] is None
    assert row["winner_strategy"] is None
    assert row["winning_score"] is None
    assert row["victory_margin"] is None
    assert row["seat_ranks"] == [None, None]
    assert row["P1_rank"] is None and row["P2_rank"] is None
    assert row["P1_n_turns"] == 2 and row["P2_n_turns"] == 2

    table = simulation_rows_to_table([row], 2)
    for field_name, field_type in {
        "winner_seat": pa.string(),
        "winner_strategy": pa.int32(),
        "winning_score": pa.int32(),
        "victory_margin": pa.int32(),
        "P1_rank": pa.int8(),
        "P2_rank": pa.int8(),
    }.items():
        field = table.schema.field(field_name)
        assert field.type == field_type
        assert field.nullable is True

    output = tmp_path / "safety.parquet"
    write_parquet_atomic(table, output)
    restored = pq.read_table(output)
    assert restored.schema == table.schema
    assert restored.to_pylist() == [row]


def test_normal_completed_raw_row_has_consistent_outcome_fields() -> None:
    row = dict(
        _play_game(
            123,
            _strategies(2),
            target_score=200,
            provenance=_provenance(2),
        )
    )

    validate_simulation_row(row)
    assert row["termination_status"] == "completed"
    assert row["hit_safety_limit"] is False
    assert row["winner_seat"] in {"P1", "P2"}
    assert row["winner_strategy"] is not None
    assert sorted([row["P1_rank"], row["P2_rank"]]) == [1, 2]


def test_completed_row_without_exactly_one_winner_is_rejected() -> None:
    row = dict(
        _play_game(
            123,
            _strategies(2),
            target_score=200,
            provenance=_provenance(2),
        )
    )
    row["P1_rank"] = 2
    row["P2_rank"] = 2

    with pytest.raises(ValueError, match="exactly one winner"):
        validate_simulation_row(row)


def test_safety_limit_row_claiming_a_winner_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FarklePlayer, "take_turn", _scripted_turns({"P1": 0, "P2": 0}))
    row = dict(
        _play_game(
            123,
            _strategies(2),
            target_score=100,
            provenance=_provenance(2),
            max_rounds=1,
        )
    )
    row["winner_seat"] = "P1"
    row["winner_strategy"] = row["P1_strategy"]

    with pytest.raises(ValueError, match="cannot claim a winner"):
        validate_simulation_row(row)
    with pytest.raises(RuntimeError, match="fabricates a winner"):
        _require_outcome(row, source="test row")
