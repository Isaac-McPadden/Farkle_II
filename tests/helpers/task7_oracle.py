"""Independent raw-row arithmetic checks for Task 7's full workflow products."""

from __future__ import annotations

import math
from collections import defaultdict

import pyarrow.parquet as pq
import pytest

from farkle.config import AppConfig


def assert_task7_products(cfg: AppConfig) -> None:
    ranges = []
    for k in sorted(set(cfg.sim.n_players_list)):
        games = pq.read_table(cfg.ingested_rows_curated(k)).to_pylist()
        expected = defaultdict(list)
        for game in games:
            for seat in range(1, k + 1):
                expected[
                    (
                        game["deterministic_batch_id"],
                        game[f"P{seat}_strategy"],
                        seat,
                        game["termination_status"],
                    )
                ].append((game[f"P{seat}_score"], game[f"P{seat}_n_turns"], game["n_rounds"]))
        for row in pq.read_table(cfg.seat_behavior_batch_path(k)).to_pylist():
            values = expected.pop(
                (
                    row["deterministic_batch_id"],
                    row["strategy"],
                    row["seat"],
                    row["termination_status"],
                ),
                [],
            )
            assert row["exposures"] == len(values)
            assert row["score_sum"] == sum(v[0] for v in values)
            assert row["turns_sum"] == sum(v[1] for v in values)
            assert row["rounds_sum"] == sum(v[2] for v in values)
            assert row["score_per_turn_sum"] == pytest.approx(
                math.fsum(s / t if t else 0 for s, t, _ in values)
            )
            assert row["score_per_round_sum"] == pytest.approx(
                math.fsum(s / r if r else 0 for s, _, r in values)
            )
        assert not expected
        completed = [g for g in games if g["termination_status"] == "completed"]
        turns = [[g[f"P{s}_n_turns"] for s in range(1, k + 1)] for g in completed]
        mean_range = math.fsum(max(t) - min(t) for t in turns) / len(turns) if turns else None
        ranges.append(mean_range)
        opportunities = pq.read_table(cfg.turn_opportunity_path(k)).to_pylist()
        assert opportunities[0]["completed_games"] == len(completed)
        assert (
            opportunities[0]["mean_turn_range"] == pytest.approx(mean_range)
            if mean_range is not None
            else opportunities[0]["mean_turn_range"] is None
        )
        total = sum(sum(t) for t in turns)
        for row in opportunities[1:]:
            seat = row["seat"]
            count = sum(t[seat - 1] for t in turns)
            assert row["turns_sum"] == count
            if total:
                assert row["turn_share"] == pytest.approx(count / total)
        assert len(pq.read_table(cfg.parameter_responses_path(k))) <= 250
        assert len(pq.read_table(cfg.parameter_nonadditivity_path(k))) <= 1728
        assert "oof_nonadditivity" not in pq.read_schema(cfg.parameter_nonadditivity_path(k)).names
    equal = pq.read_table(cfg.turn_opportunity_equal_k_path()).to_pylist()[0]
    assert equal["weights"] == [1 / len(ranges)] * len(ranges)
    if all(v is not None for v in ranges):
        assert equal["mean_turn_range_equal_k"] == pytest.approx(math.fsum(ranges) / len(ranges))
    else:
        assert equal["mean_turn_range_equal_k"] is None
