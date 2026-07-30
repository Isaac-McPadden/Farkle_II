"""Authenticated real-game oracle for the raw two-root simulation boundary."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.raw_simulation_oracle import (
    ORACLE_PLAYER_COUNTS,
    ORACLE_ROOTS,
    load_tiny_oracle_config,
    oracle_game_profile,
    run_raw_simulation_roots,
)

from farkle.orchestration.run_contexts import SeedRunContext, load_run_context
from farkle.simulation import runner
from farkle.simulation.game_profile import GameProfile
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.manifest import iter_manifest
from farkle.utils.random import (
    RNG_SCHEME_VERSION,
    RandomPurpose,
    coordinate_entropy,
    coordinate_seed,
)
from farkle.utils.schema_helpers import (
    OUTCOME_SCHEMA_VERSION,
    TOURNAMENT_METHOD_VERSION,
    raw_simulation_schema_for,
)
from farkle.utils.stage_completion import CompletionState

EXPECTED_ROWS = {
    (11, 2, 0, 0): ([0, 2], "safety_limit", None, 0, 0, [0, 0]),
    (11, 2, 0, 1): ([1, 3], "completed", 1, 2, 5, [1950, 1100]),
    (11, 2, 1, 0): ([2, 1], "completed", 2, 2, 4, [500, 0]),
    (11, 2, 1, 1): ([0, 3], "completed", 0, 1, 2, [600, 0]),
    (11, 4, 0, 0): ([0, 1, 2, 3], "completed", 2, 1, 4, [700, 0, 800, 0]),
    (11, 4, 1, 0): ([3, 2, 1, 0], "completed", 3, 1, 5, [3050, 2900, 0, 0]),
    (22, 2, 0, 0): ([3, 0], "completed", 3, 1, 2, [600, 0]),
    (22, 2, 0, 1): ([1, 2], "completed", 2, 1, 2, [500, 1100]),
    (22, 2, 1, 0): ([2, 0], "completed", 2, 1, 2, [950, 0]),
    (22, 2, 1, 1): ([3, 1], "completed", 3, 1, 3, [750, 550]),
    (22, 4, 0, 0): ([1, 2, 0, 3], "completed", 2, 1, 5, [0, 700, 0, 0]),
    (22, 4, 1, 0): ([0, 2, 1, 3], "completed", 1, 1, 4, [700, 0, 1100, 0]),
}

EXPECTED_CELL_COUNTS = {
    (11, 2): (4, 3, 1, 3),
    (11, 4): (2, 2, 0, 2),
    (22, 2): (4, 4, 0, 4),
    (22, 4): (2, 2, 0, 2),
}

EXPECTED_STRATEGY_COUNTS = {
    0: (8, 7, 1, 1, 7),
    1: (8, 8, 0, 2, 6),
    2: (8, 7, 1, 5, 3),
    3: (8, 8, 0, 3, 5),
}

EXPECTED_BATCH_COUNTS = {
    (11, 2, 0): (2, 1, 1, [1]),
    (11, 2, 1): (2, 2, 0, [2, 0]),
    (11, 4, 0): (1, 1, 0, [2]),
    (11, 4, 1): (1, 1, 0, [3]),
    (22, 2, 0): (2, 2, 0, [3, 2]),
    (22, 2, 1): (2, 2, 0, [2, 3]),
    (22, 4, 0): (1, 1, 0, [2]),
    (22, 4, 1): (1, 1, 0, [1]),
}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_raw_rows(
    contexts: tuple[SeedRunContext, SeedRunContext],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], pa.Schema]]:
    rows: list[dict[str, Any]] = []
    schemas: dict[tuple[int, int], pa.Schema] = {}
    for context in contexts:
        for k in ORACLE_PLAYER_COUNTS:
            row_dir = context.config.simulation_row_dir(k)
            assert row_dir is not None
            manifest = row_dir / "manifest.jsonl"
            records = list(iter_manifest(manifest))
            assert len(records) == 2
            cell_tables: list[pa.Table] = []
            for record in records:
                shard = row_dir / str(record["path"])
                table = pq.read_table(shard)
                assert table.schema == raw_simulation_schema_for(k)
                cell_tables.append(table)
            cell = pa.concat_tables(cell_tables)
            schemas[(context.seed, k)] = cell.schema
            rows.extend(cell.to_pylist())
    rows.sort(
        key=lambda row: (
            int(row["root_seed"]),
            int(row["k"]),
            int(row["shuffle_index"]),
            int(row["game_index"]),
        )
    )
    return rows, schemas


def _logical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: row[key]
            for key in (
                "root_seed",
                "k",
                "shuffle_index",
                "game_index",
                "deterministic_batch_id",
                "shuffle_seed",
                "game_seed",
                "rng_scheme_version",
                "rng_purpose_namespace",
                "termination_status",
                "hit_safety_limit",
                "outcome_schema_version",
                "winner_seat",
                "winner_strategy",
                "seat_ranks",
                "winning_score",
                "victory_margin",
                "n_rounds",
                *(
                    field
                    for seat in range(1, int(row["k"]) + 1)
                    for field in (
                        f"P{seat}_strategy",
                        f"P{seat}_score",
                        f"P{seat}_rank",
                        f"P{seat}_n_turns",
                        f"P{seat}_hit_max_rounds",
                    )
                ),
            )
        }
        for row in rows
    ]


def _assert_raw_oracle(rows: list[dict[str, Any]]) -> None:
    assert len(rows) == 12
    attempted_by_strategy: Counter[int] = Counter()
    completed_by_strategy: Counter[int] = Counter()
    safety_by_strategy: Counter[int] = Counter()
    wins_by_strategy: Counter[int] = Counter()
    cell_counts: Counter[tuple[int, int, str]] = Counter()
    batch_counts: Counter[tuple[int, int, int, str]] = Counter()
    batch_winners: dict[tuple[int, int, int], list[int]] = {}
    total_turns = 0
    total_scores = 0

    for row in rows:
        root = int(row["root_seed"])
        k = int(row["k"])
        shuffle_index = int(row["shuffle_index"])
        game_index = int(row["game_index"])
        key = (root, k, shuffle_index, game_index)
        strategies, status, winner_strategy, rounds, turns, scores = EXPECTED_ROWS[key]
        assert [row[f"P{seat}_strategy"] for seat in range(1, k + 1)] == strategies
        assert row["termination_status"] == status
        assert row["winner_strategy"] == winner_strategy
        assert row["n_rounds"] == rounds
        assert sum(row[f"P{seat}_n_turns"] for seat in range(1, k + 1)) == turns
        assert [row[f"P{seat}_score"] for seat in range(1, k + 1)] == scores
        assert row["outcome_schema_version"] == OUTCOME_SCHEMA_VERSION
        assert row["rng_scheme_version"] == RNG_SCHEME_VERSION
        assert row["rng_purpose_namespace"] == int(RandomPurpose.TOURNAMENT_GAME)
        assert row["deterministic_batch_id"] == shuffle_index
        assert row["shuffle_seed"] == coordinate_seed(
            RandomPurpose.TOURNAMENT_SHUFFLE,
            root_seed=root,
            k=k,
            shuffle_index=shuffle_index,
            dtype=np.uint32,
        )
        assert row["game_seed"] == coordinate_seed(
            RandomPurpose.TOURNAMENT_GAME,
            root_seed=root,
            k=k,
            shuffle_index=shuffle_index,
            game_index=game_index,
            dtype=np.uint32,
        )
        assert coordinate_entropy(
            RandomPurpose.TOURNAMENT_PLAYER,
            root_seed=root,
            k=k,
            shuffle_index=shuffle_index,
            game_index=game_index,
            seat_index=0,
        ) != coordinate_entropy(
            RandomPurpose.TOURNAMENT_PLAYER,
            root_seed=root,
            k=k,
            shuffle_index=shuffle_index,
            game_index=game_index,
            seat_index=1,
        )

        cell_counts[(root, k, "attempted")] += 1
        batch = (root, k, shuffle_index)
        batch_counts[(*batch, "attempted")] += 1
        total_turns += turns
        total_scores += sum(scores)
        for strategy in strategies:
            attempted_by_strategy[strategy] += 1
        if status == "completed":
            cell_counts[(root, k, "completed")] += 1
            cell_counts[(root, k, "wins")] += 1
            batch_counts[(*batch, "completed")] += 1
            assert winner_strategy is not None
            wins_by_strategy[winner_strategy] += 1
            batch_winners.setdefault(batch, []).append(winner_strategy)
            for strategy in strategies:
                completed_by_strategy[strategy] += 1
        else:
            cell_counts[(root, k, "safety")] += 1
            batch_counts[(*batch, "safety")] += 1
            for strategy in strategies:
                safety_by_strategy[strategy] += 1
            assert row["hit_safety_limit"] is True
            assert row["winner_seat"] is None
            assert row["winner_strategy"] is None
            assert row["winning_score"] is None
            assert row["victory_margin"] is None
            assert row["seat_ranks"] == [None] * k
            assert all(row[f"P{seat}_rank"] is None for seat in range(1, k + 1))
            assert all(row[f"P{seat}_hit_max_rounds"] is True for seat in range(1, k + 1))

    assert total_turns == 38
    assert total_scores == 18_550
    for batch, (attempted, completed, safety, winners) in EXPECTED_BATCH_COUNTS.items():
        assert batch_counts[(*batch, "attempted")] == attempted
        assert batch_counts[(*batch, "completed")] == completed
        assert batch_counts[(*batch, "safety")] == safety
        assert batch_winners.get(batch, []) == winners
    for (root, k), (attempted, completed, safety, wins) in EXPECTED_CELL_COUNTS.items():
        assert cell_counts[(root, k, "attempted")] == attempted
        assert cell_counts[(root, k, "completed")] == completed
        assert cell_counts[(root, k, "safety")] == safety
        assert cell_counts[(root, k, "wins")] == wins
        assert attempted == completed + safety
        assert wins == completed

    assert sum(cell_counts[(*cell, "attempted")] for cell in EXPECTED_CELL_COUNTS) == 12
    assert sum(cell_counts[(*cell, "completed")] for cell in EXPECTED_CELL_COUNTS) == 11
    assert sum(cell_counts[(*cell, "safety")] for cell in EXPECTED_CELL_COUNTS) == 1
    assert sum(wins_by_strategy.values()) == 11
    for strategy, (attempted, completed, safety, wins, losses) in EXPECTED_STRATEGY_COUNTS.items():
        assert attempted_by_strategy[strategy] == attempted
        assert completed_by_strategy[strategy] == completed
        assert safety_by_strategy[strategy] == safety
        assert wins_by_strategy[strategy] == wins
        assert attempted - wins == losses


def _assert_authenticated_simulations(
    contexts: tuple[SeedRunContext, SeedRunContext],
    *,
    tmp_path: Path,
    profile_sha256: str,
) -> None:
    for context in contexts:
        persisted_context = load_run_context(
            context.run_context_path,
            active_config_path=context.active_config_path,
        )
        assert persisted_context["run_context_contract_version"] == 1
        assert persisted_context["public_config_sha256"] == context.config.config_sha
        assert persisted_context["run_lineage_sha256"] == context.config._run_lineage_sha256
        assert len(str(persisted_context["run_context_sha256"])) == 64
        assert len(str(persisted_context["public_config_sha256"])) == 64
        assert len(str(persisted_context["run_lineage_sha256"])) == 64
        assert len(str(persisted_context["code_identity"]["commit"])) == 40
        assert persisted_context["lineage_extensions"] == {"game_profile_sha256": profile_sha256}
        assert Path(persisted_context["resolved_paths"]["results_root"]).is_absolute()
        assert _is_relative_to(context.results_root.resolve(), tmp_path.resolve())

        active_done = json.loads(
            context.active_config_path.with_suffix(".done.json").read_text(encoding="utf-8")
        )
        assert active_done["config_sha"] == context.config.config_sha
        assert active_done["round_trip_config_sha"] == context.config.config_sha
        assert active_done["active_config_sha256"] == sha256_file(context.active_config_path)

        for k in ORACLE_PLAYER_COUNTS:
            assert runner.simulation_is_complete(context.config, k)
            done_path = runner.simulation_done_path(context.config, k)
            stamp = json.loads(done_path.read_text(encoding="utf-8"))
            assert stamp["schema_version"] == 4
            assert stamp["lifecycle_contract_version"] == 1
            assert stamp["stage"] == "simulation"
            assert stamp["status"] == "success"
            assert stamp["completion_state"] == CompletionState.COMPLETE_VALID.value
            assert stamp["config_sha"] == context.config.config_sha
            assert stamp["run_lineage_sha256"] == context.config._run_lineage_sha256
            assert len(stamp["stage_config_sha"]) == 64
            assert len(stamp["freshness_sha256"]) == 64
            assert len(stamp["stage_identity_sha256"]) == 64
            assert stamp["freshness_key"]["game_profile_sha256"] == profile_sha256
            assert stamp["game_profile_sha256"] == profile_sha256
            assert stamp["rng_scheme_version"] == RNG_SCHEME_VERSION
            assert stamp["outcome_schema_version"] == OUTCOME_SCHEMA_VERSION
            assert stamp["tournament_method_version"] == TOURNAMENT_METHOD_VERSION
            assert stamp["n_players"] == k
            assert stamp["num_shuffles"] == 2
            assert stamp["shuffles_per_batch"] == 1
            assert stamp["n_strategies"] == 4
            assert stamp["code_identity"] == persisted_context["code_identity"]

            for role in ("input_identities", "output_identities"):
                identities = stamp[role]
                assert identities
                for identity, raw_path in zip(
                    identities,
                    stamp["inputs" if role == "input_identities" else "outputs"],
                    strict=True,
                ):
                    path = Path(raw_path).resolve()
                    assert _is_relative_to(path, context.results_root.resolve())
                    assert identity["kind"] == "file"
                    assert identity["byte_length"] == path.stat().st_size
                    assert identity["content_sha256"] == sha256_file(path)
                    assert len(identity["content_sha256"]) == 64
                    assert identity["sidecar_sha256"] is None

            workload = json.loads(
                (context.config.n_dir(k) / "simulation_workload_plan.json").read_text(
                    encoding="utf-8"
                )
            )
            assert workload["strategy_count"] == 4
            assert workload["batch_count"] == 2
            assert workload["shuffles_per_batch"] == 1
            assert workload["required_shuffles"] == 2
            assert workload["games_per_shuffle"] == 4 // k

            checkpoint = pickle.loads(context.config.checkpoint_path(k).read_bytes())
            outcome = checkpoint["outcome_counts"]
            attempted, completed, safety, wins = EXPECTED_CELL_COUNTS[(context.seed, k)]
            assert outcome["games_attempted"] == attempted
            assert outcome["games_completed"] == completed
            assert outcome["games_safety_limit"] == safety
            assert sum(checkpoint["win_totals"].values()) == wins
            assert sum(outcome["attempted_exposures"].values()) == k * attempted
            assert sum(outcome["completed_exposures"].values()) == k * completed
            assert sum(outcome["safety_limit_exposures"].values()) == k * safety
            assert checkpoint["meta"]["game_profile_sha256"] == profile_sha256
            assert checkpoint["meta"]["rng_scheme_version"] == RNG_SCHEME_VERSION
            assert checkpoint["meta"]["rng_bit_generator"] == "PCG64DXSM"

            metric_rows = pq.read_table(context.config.metrics_path(k)).to_pylist()
            assert sum(int(row["wins"]) for row in metric_rows) == completed
            assert sum(int(row["attempted_exposures"]) for row in metric_rows) == k * attempted
            assert sum(int(row["completed_exposures"]) for row in metric_rows) == k * completed
            assert sum(int(row["safety_limit_exposures"]) for row in metric_rows) == k * safety
            for metric_row in metric_rows:
                assert int(metric_row["losses"]) == (
                    int(metric_row["attempted_exposures"]) - int(metric_row["wins"])
                )

            row_dir = context.config.simulation_row_dir(k)
            assert row_dir is not None
            assert all(
                record["game_profile_sha256"] == profile_sha256
                for record in iter_manifest(row_dir / "manifest.jsonl")
            )
            metric_chunk_dir = (
                context.config.n_dir(k) / f"{k}p_{context.config.sim.metric_chunk_dir}"
            )
            assert all(
                record["game_profile_sha256"] == profile_sha256
                for record in iter_manifest(metric_chunk_dir / "metrics_manifest.jsonl")
            )


def test_authenticated_raw_two_root_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known_bad_root = (
        Path(__file__).resolve().parents[2] / "data" / "results_seed_pair_32_33"
    ).resolve()
    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any):
        resolved = path.resolve()
        if resolved == known_bad_root or known_bad_root in resolved.parents:
            raise AssertionError(f"raw oracle accessed known-bad fast-run tree: {resolved}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    config_path, cfg = load_tiny_oracle_config(tmp_path)
    assert config_path.parent == tmp_path
    assert tuple(cfg.sim.seed_list or ()) == ORACLE_ROOTS
    assert tuple(cfg.sim.n_players_list) == ORACLE_PLAYER_COUNTS
    with pytest.raises(ValueError, match="100 equal batches"):
        cfg.validate_statistical_contract(require_two_roots=True)

    profile = oracle_game_profile()
    contexts = run_raw_simulation_roots(cfg, profile)
    rows, schemas = _load_raw_rows(contexts)
    assert all(schema == raw_simulation_schema_for(k) for (_root, k), schema in schemas.items())
    _assert_raw_oracle(rows)
    _assert_authenticated_simulations(
        contexts,
        tmp_path=tmp_path,
        profile_sha256=profile.sha256,
    )

    changed_profile = GameProfile(default_target_score=101)
    with pytest.raises(ValueError, match="authenticated run lineage"):
        runner.run_tournament(
            contexts[0].config,
            oracle_game_profile=changed_profile,
        )

    first_logical_rows = _logical_rows(rows)
    for context in contexts:
        runner.run_tournament(
            context.config,
            force=True,
            oracle_game_profile=profile,
        )
    replay_rows, _ = _load_raw_rows(contexts)
    assert _logical_rows(replay_rows) == first_logical_rows
    _assert_authenticated_simulations(
        contexts,
        tmp_path=tmp_path,
        profile_sha256=profile.sha256,
    )

    assert not any(context.analysis_root.exists() for context in contexts)
