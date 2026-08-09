"""Assertions for raw-derived tournament and real H2H analysis through agreement."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tests.helpers.raw_simulation_oracle import (
    ORACLE_PLAYER_COUNTS,
    FrozenFamilySnapshot,
)

from farkle import analysis
from farkle.analysis import combine, hgb_feat, trueskill
from farkle.analysis.all_player_metrics import all_player_batch_schema
from farkle.analysis.candidate_family import freeze_h2h_candidate_family
from farkle.analysis.h2h_schedule import (
    H2H_METHOD_VERSION,
    execute_h2h_schedule,
)
from farkle.analysis.release_audit import audit_sidecar_completeness
from farkle.config import ArtifactScope
from farkle.orchestration.run_contexts import (
    RootPairRunContext,
    SeedRunContext,
    load_run_context,
)
from farkle.simulation.game_profile import GameProfile
from farkle.utils.artifact_contract import (
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.authenticated_contract import (
    load_authenticated_sidecar,
    validate_authenticated_artifact_unbound,
)
from farkle.utils.schema_helpers import expected_schema_for, raw_simulation_schema_for
from farkle.utils.stage_completion import (
    CompletionState,
    freshness_sha256,
    resolve_stage_state,
)

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
EXPECTED_H2H_BLOCKS = {
    (0, 11, 0): (2, 1, 1, 1, 0, 1, "complete"),
    (0, 11, 1): (1, 1, 0, 0, 1, 0, "complete"),
    (0, 22, 0): (1, 1, 0, 1, 0, 0, "complete"),
    (0, 22, 1): (1, 1, 0, 0, 1, 0, "complete"),
    (1, 11, 0): (2, 0, 2, 0, 0, 1, "unresolved_nonviable"),
    (1, 11, 1): (1, 1, 0, 0, 1, 0, "complete"),
    (1, 22, 0): (1, 1, 0, 0, 1, 0, "complete"),
    (1, 22, 1): (1, 1, 0, 0, 1, 0, "complete"),
    (2, 11, 0): (1, 1, 0, 1, 0, 0, "complete"),
    (2, 11, 1): (1, 1, 0, 0, 1, 0, "complete"),
    (2, 22, 0): (1, 1, 0, 0, 1, 0, "complete"),
    (2, 22, 1): (1, 1, 0, 0, 1, 0, "complete"),
}
_COORDINATE_COLUMNS = ["root_seed", "k", "shuffle_index", "game_index"]
_SCOPES = {scope.value for scope in ArtifactScope}


def _raw_cell_table(context: SeedRunContext, k: int) -> pa.Table:
    row_dir = context.config.simulation_row_dir(k)
    assert row_dir is not None
    shards = sorted(row_dir.glob("rows_*.parquet"))
    assert len(shards) == 2
    tables = [pq.read_table(path) for path in shards]
    assert all(table.schema == raw_simulation_schema_for(k) for table in tables)
    return pa.concat_tables(tables)


def _coordinates(table: pa.Table) -> set[tuple[int, int, int, int]]:
    return {
        (
            int(row["root_seed"]),
            int(row["k"]),
            int(row["shuffle_index"]),
            int(row["game_index"]),
        )
        for row in table.select(_COORDINATE_COLUMNS).to_pylist()
    }


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _sum(frame: pd.DataFrame, column: str) -> int:
    return _as_int(frame[column].sum())


def _assert_cell_conservation(
    frame: pd.DataFrame,
    *,
    attempted: int,
    completed: int,
    safety: int,
    wins: int,
    k: int,
) -> None:
    exposures = k * attempted
    assert _sum(frame, "raw_player_game_exposures") == exposures
    assert _sum(frame, "raw_completed_player_game_exposures") == k * completed
    assert _sum(frame, "raw_safety_limit_player_game_exposures") == k * safety
    assert _sum(frame, "raw_wins") == wins
    assert _sum(frame, "raw_losses") == exposures - wins
    assert attempted == completed + safety
    assert exposures == k * completed + k * safety
    assert wins == completed


def assert_root_pipeline_oracle(
    contexts: tuple[SeedRunContext, SeedRunContext],
) -> None:
    """Assert row identity, counts, estimands, diagnostics, TrueSkill, and HGB."""

    strategy_counts: Counter[int] = Counter()
    strategy_completed: Counter[int] = Counter()
    strategy_safety: Counter[int] = Counter()
    strategy_wins: Counter[int] = Counter()

    for context in contexts:
        cfg = context.config
        expected_stage_dirs = [
            "00_ingest",
            "01_curate",
            "02_combine",
            "03_metrics",
            "04_game_stats",
            "05_trueskill",
            "06_hgb",
            "07_screening",
        ]
        assert sorted(path.name for path in cfg.analysis_dir.iterdir() if path.is_dir()) == (
            expected_stage_dirs
        )
        assert cfg.analysis.disable_rng_diagnostics is True
        assert cfg.stage_layout.folder_for("rng_diagnostics") is None

        concat_coordinates: set[tuple[int, int, int, int]] = set()
        for k in ORACLE_PLAYER_COUNTS:
            attempted, completed, safety, wins = EXPECTED_CELL_COUNTS[(context.seed, k)]
            raw = _raw_cell_table(context, k)
            ingested = pq.read_table(cfg.ingested_rows_raw(k))
            curated = pq.read_table(cfg.ingested_rows_curated(k))
            partition = pq.read_table(cfg.combined_rows_by_k(k))
            expected_coordinates = _coordinates(raw)
            assert raw.num_rows == ingested.num_rows == curated.num_rows == partition.num_rows
            assert raw.num_rows == attempted
            assert ingested.schema == curated.schema == raw_simulation_schema_for(k)
            assert _coordinates(ingested) == _coordinates(curated) == expected_coordinates
            assert _coordinates(partition) == expected_coordinates
            concat_coordinates.update(expected_coordinates)

            raw_rows = raw.to_pylist()
            for row in raw_rows:
                strategies = [int(row[f"P{seat}_strategy"]) for seat in range(1, k + 1)]
                strategy_counts.update(strategies)
                if row["termination_status"] == "completed":
                    strategy_completed.update(strategies)
                    strategy_wins[int(row["winner_strategy"])] += 1
                else:
                    strategy_safety.update(strategies)

            metrics = pq.read_table(cfg.metrics_all_player_batch_path(k))
            assert metrics.schema == all_player_batch_schema()
            metric_frame = metrics.to_pandas()
            assert len(metric_frame) == 8
            assert (
                metric_frame.groupby("deterministic_batch_id")["raw_player_game_exposures"]
                .sum()
                .eq(k * (4 // k))
                .all()
            )
            _assert_cell_conservation(
                metric_frame,
                attempted=attempted,
                completed=completed,
                safety=safety,
                wins=wins,
                k=k,
            )
            assert (
                metric_frame["raw_player_game_exposures"]
                == metric_frame["raw_completed_player_game_exposures"]
                + metric_frame["raw_safety_limit_player_game_exposures"]
            ).all()
            assert (
                metric_frame["raw_losses"]
                == metric_frame["raw_player_game_exposures"] - metric_frame["raw_wins"]
            ).all()

            performance = pq.read_table(cfg.performance_by_k_path(k)).to_pandas()
            assert set(performance["strategy"].astype(int)) == {0, 1, 2, 3}
            assert performance["raw_exposures"].eq(2).all()
            assert _sum(performance, "raw_wins") == wins
            assert _sum(performance, "raw_losses") == k * attempted - wins
            assert _sum(performance, "raw_completed_exposures") == k * completed
            assert _sum(performance, "raw_safety_limit_exposures") == k * safety
            assert all(
                math.isclose(
                    _as_float(row.win_rate),
                    _as_int(row.raw_wins) / _as_int(row.raw_exposures),
                )
                for row in performance.itertuples()
            )

            seat_counts = pq.read_table(cfg.seat_batch_counts_path(k)).to_pandas()
            assert _sum(seat_counts, "raw_exposures") == k * attempted
            assert _sum(seat_counts, "raw_completed_exposures") == k * completed
            assert _sum(seat_counts, "raw_safety_limit_exposures") == k * safety
            assert _sum(seat_counts, "raw_wins") == wins

            game_stats = pq.read_table(
                cfg.game_stats_stage_dir / "by_k" / f"{k}p" / f"game_stats.{k}p.parquet"
            ).to_pandas()
            game_row = game_stats.loc[game_stats["summary_level"].eq("n_players")].iloc[0]
            assert int(game_row["observations"]) == attempted
            assert int(game_row["completed_observations"]) == completed
            assert int(game_row["safety_limit_observations"]) == safety

            ratings = pq.read_table(
                cfg.trueskill_rating_path(k, root_seed=context.seed)
            ).to_pandas()
            assert len(ratings) == 4
            assert set(ratings["rating_status"]) == {"evidence_backed_completed_games"}
            assert ratings["cell_games_attempted"].eq(attempted).all()
            assert ratings["cell_games_completed"].eq(completed).all()
            assert ratings["cell_games_excluded_safety_limit"].eq(safety).all()
            assert ratings["cell_performed_updates"].eq(completed).all()
            assert (
                ratings["strategy_attempted_exposures"]
                == ratings["strategy_completed_exposures"]
                + ratings["strategy_excluded_safety_limit_exposures"]
            ).all()
            assert (
                ratings["strategy_performed_updates"] == ratings["strategy_completed_exposures"]
            ).all()
            rating_done = json.loads(
                (
                    cfg.trueskill_rating_path(k, root_seed=context.seed).parent
                    / f"ratings_{k}_seed{context.seed}.done.json"
                ).read_text(encoding="utf-8")
            )
            assert rating_done["method_version"] == 3
            assert rating_done["attempted_games"] == attempted
            assert rating_done["completed_games"] == completed
            assert rating_done["excluded_safety_limit_games"] == safety
            assert rating_done["performed_update_games"] == completed

            folds = pq.read_table(cfg.hgb_fold_metrics_path(k)).to_pandas()
            predictions = pq.read_table(cfg.hgb_predictive_scores_path(k)).to_pandas()
            importance = pq.read_table(cfg.hgb_importance_path(k)).to_pandas()
            assert set(folds["fold"].astype(int)) == {0, 1}
            assert folds["train_strategies"].eq(2).all()
            assert folds["heldout_strategies"].eq(2).all()
            assert set(predictions["strategy"].astype(int)) == {0, 1, 2, 3}
            assert len(importance) == 10

        combined = pa.Table.from_batches(list(combine.scan_concat_ks(cfg)))
        # ``combine`` publishes the rectangular schema selected by the
        # configured maximum, not the production default.  The tiny oracle
        # deliberately bounds this to four seats.
        assert combined.schema == expected_schema_for(cfg.combine_max_players)
        assert combined.num_rows == 6
        assert _coordinates(combined) == concat_coordinates

        across = pq.read_table(cfg.performance_across_k_path()).to_pandas()
        assert set(across["required_k_count"]) == {2}
        assert set(across["support_k_count"]) == {2}
        assert across["complete_support"].all()
        assert across["raw_attempted_exposures"].eq(4).all()
        for row in across.itertuples():
            by_k_scores = [
                _as_float(
                    pq.read_table(cfg.performance_by_k_path(k))
                    .to_pandas()
                    .set_index("strategy")
                    .at[_as_int(row.strategy), "chance_delta"]
                )
                for k in ORACLE_PLAYER_COUNTS
            ]
            assert math.isclose(_as_float(row.equal_k_score), sum(by_k_scores) / 2)
        across_sidecar = validate_artifact_sidecar(cfg.performance_across_k_path())
        assert across_sidecar.k_aggregation_method == "equal_k"
        assert across_sidecar.required_player_counts == [2, 4]

        margin = pq.read_table(cfg.game_stats_concat_path("margin_stats.parquet")).to_pandas()
        for k in ORACLE_PLAYER_COUNTS:
            attempted, completed, safety, _wins = EXPECTED_CELL_COUNTS[(context.seed, k)]
            cell = margin.loc[margin["n_players"].eq(k)]
            assert _sum(cell, "attempted_observations") == k * attempted
            assert _sum(cell, "completed_observations") == k * completed
            assert _sum(cell, "safety_limit_observations") == k * safety
            assert _sum(cell, "observations") == k * completed
            assert set(cell["observational_unit"]) == {
                "seated_strategy_exposure_per_completed_game"
            }

        ts_diagnostics = pq.read_table(cfg.trueskill_screening_diagnostics_path()).to_pandas()
        assert set(ts_diagnostics["k"].astype(int)) == {2, 4}
        assert ts_diagnostics["prior_only_strategy_count"].eq(0).all()
        assert ts_diagnostics["prior_only_strategies"].eq("").all()
        for row in ts_diagnostics.itertuples():
            attempted, completed, safety, _wins = EXPECTED_CELL_COUNTS[
                (context.seed, _as_int(row.k))
            ]
            assert _as_int(row.attempted_games) == attempted
            assert _as_int(row.completed_games) == completed
            assert _as_int(row.excluded_safety_limit_games) == safety
            assert _as_int(row.performed_update_games) == completed
            assert row.rating_status == "evidence_backed"

        proposals = pq.read_table(cfg.hgb_future_proposals_path()).to_pandas()
        assert not proposals["included_in_current_analysis"].any()
        hgb_done = json.loads((cfg.hgb_stage_dir / "hgb.done.json").read_text(encoding="utf-8"))
        assert hgb_done["state"] == CompletionState.COMPLETE_VALID.value
        hgb_sidecar = load_authenticated_sidecar(cfg.hgb_fold_metrics_path(2))
        assert hgb_sidecar.versions.method_versions["hgb_method_version"] == 2
        assert hgb_sidecar.versions.method_versions["hgb_rng_method_version"] == 2
        assert cfg.hgb.heldout_folds == 2
        assert cfg.hgb.permutation_repeats == 1
        assert cfg.hgb.n_estimators == 1
        assert cfg.hgb.max_depth == 1

    assert {
        strategy: (
            strategy_counts[strategy],
            strategy_completed[strategy],
            strategy_safety[strategy],
            strategy_wins[strategy],
            strategy_counts[strategy] - strategy_wins[strategy],
        )
        for strategy in range(4)
    } == EXPECTED_STRATEGY_COUNTS


def assert_pair_candidate_oracle(pair_context: RootPairRunContext) -> None:
    """Assert root stability, pair TrueSkill, and pre-viability family freezing."""

    cfg = pair_context.config
    assert sorted(path.name for path in cfg.analysis_dir.iterdir() if path.is_dir()) == [
        "00_root_stability",
        "01_trueskill",
        "02_candidate_freeze",
    ]
    across = pq.read_table(cfg.root_combined_performance_across_k_path()).to_pandas()
    combined = across.loc[across["estimate_scope"].eq("combined_roots")].set_index("strategy")
    assert combined["complete_support"].all()
    assert combined["support_k_count"].eq(2).all()
    for strategy, (attempted, completed, safety, wins, losses) in EXPECTED_STRATEGY_COUNTS.items():
        row = cast(pd.Series, combined.loc[strategy])
        assert _as_int(row["raw_attempted_exposures"]) == attempted
        assert _as_int(row["raw_completed_exposures"]) == completed
        assert _as_int(row["raw_safety_limit_exposures"]) == safety
        assert _as_int(row["raw_wins"]) == wins
        assert _as_int(row["raw_losses"]) == losses

    drift = pq.read_table(cfg.root_half_drift_path()).to_pandas()
    assert len(drift) == 24
    assert drift["expected_mcse"].isna().all()
    assert drift["standardized_drift"].isna().all()

    contribution = pq.read_table(cfg.trueskill_candidate_contribution_path()).to_pandas()
    assert len(contribution) == 4
    assert contribution["rating_cells_present"].eq(4).all()
    assert contribution["rating_cells_required"].eq(4).all()
    assert contribution["complete_support"].all()
    assert _sum(contribution, "attempted_exposures_total") == 32
    assert _sum(contribution, "completed_exposures_total") == 30
    assert _sum(contribution, "excluded_safety_limit_exposures_total") == 2
    assert _sum(contribution, "performed_updates_total") == 30

    membership = pq.read_table(cfg.h2h_candidate_family_path()).to_pandas()
    manifest = json.loads(cfg.h2h_candidate_family_manifest_path().read_text(encoding="utf-8"))
    frozen = membership.loc[membership["final_family"], "strategy"].astype(int).tolist()
    assert frozen == [0, 1, 3]
    assert membership.loc[membership["strategy"].eq(2), "removed_by_cap"].item()
    assert membership.loc[membership["strategy"].eq(2), "removal_round"].item() == 1
    assert set(membership.loc[membership["final_family"], "family_hash"]) == {
        manifest["family_hash"]
    }
    assert manifest["candidates"] == [0, 1, 3]
    assert manifest["candidate_count"] == 3
    assert manifest["root_seeds"] == [11, 22]
    assert manifest["cutoff_rounds"] == 1
    assert manifest["projected_workload"] == {
        "unordered_pair_count": 3,
        "root_count": 2,
        "seat_order_blocks": 12,
        "selfplay_root_blocks": 6,
        "game_allocation_status": "pending_power_plan",
    }
    assert "before scheduling" in manifest["interpretation"]
    serialized = json.dumps(manifest, sort_keys=True).lower()
    assert "operationally_viable" not in serialized
    assert "completion_rate" not in serialized

    win_path = cfg.root_combined_performance_across_k_path()
    ts_path = cfg.trueskill_candidate_contribution_path()
    assert manifest["source_paths"] == {
        "win_rate": str(win_path),
        "trueskill": str(ts_path),
    }
    assert manifest["source_identity"]["win_rate"]["sha256"] == sha256_file(win_path)
    assert manifest["source_identity"]["trueskill"]["sha256"] == sha256_file(ts_path)
    assert manifest["source_identity"]["trueskill"]["sidecar_sha256"] == sha256_file(
        sidecar_path(ts_path)
    )
    assert not cfg.stage_dir("h2h_power").exists()
    assert not cfg.h2h_power_plan_path().exists()
    assert not cfg.h2h_block_manifest_path().exists()


def _attempt_prefix_sha256(row: Mapping[str, Any]) -> str:
    payload = {
        "rng_scheme_version": _as_int(row["rng_scheme_version"]),
        "purpose": _as_int(row["rng_purpose_namespace"]),
        "root_seed": _as_int(row["root_seed"]),
        "pair_id": _as_int(row["pair_id"]),
        "order": _as_int(row["order"]),
        "attempt_index_start": 0,
        "attempt_index_stop_exclusive": _as_int(row["authenticated_attempt_index_stop_exclusive"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def assert_pair_h2h_oracle(
    pair_context: RootPairRunContext,
    frozen: FrozenFamilySnapshot,
    profile: GameProfile,
    *,
    include_reporting: bool = False,
) -> None:
    """Assert real planning, block execution, inference, dominance, and agreement."""

    cfg = pair_context.config
    expected_stage_dirs = [
        "00_root_stability",
        "01_trueskill",
        "02_candidate_freeze",
        "03_h2h_power",
        "04_h2h_execute",
        "05_h2h_inference",
        "06_h2h_digest",
        "07_agreement",
    ]
    if include_reporting:
        expected_stage_dirs.append("08_reporting")
    assert sorted(path.name for path in cfg.analysis_dir.iterdir() if path.is_dir()) == (
        expected_stage_dirs
    )
    assert cfg.stage_dir("reporting").exists() is include_reporting
    for path_text, digest in (*frozen.file_sha256, *frozen.plan_file_sha256):
        assert sha256_file(Path(path_text)) == digest

    membership = pq.read_table(cfg.h2h_candidate_family_path()).to_pandas()
    manifest = json.loads(cfg.h2h_candidate_family_manifest_path().read_text(encoding="utf-8"))
    plan = json.loads(cfg.h2h_power_plan_path().read_text(encoding="utf-8"))
    schedule = pq.read_table(cfg.h2h_block_manifest_path()).to_pandas()
    assert manifest["family_hash"] == frozen.family_hash
    assert manifest["candidates"] == [0, 1, 3]
    assert membership.loc[membership["final_family"], "strategy"].astype(int).tolist() == [
        0,
        1,
        3,
    ]
    assert plan["family_hash"] == frozen.family_hash
    assert plan["candidate_count"] == 3
    assert plan["unordered_pair_count"] == 3
    assert plan["root_seeds"] == [11, 22]
    assert plan["n_completed_required_per_root_order_block"] == 1
    assert plan["max_attempts_per_root_order_block"] == 2
    assert plan["total_completed_required"] == 12
    assert plan["maximum_total_attempts"] == 24
    assert plan["total_block_count"] == 12
    assert plan["planning_state"] == CompletionState.COMPLETE_VALID.value
    assert plan["game_profile_sha256"] == profile.sha256
    assert len(schedule) == 12
    assert not schedule.duplicated(["pair_id", "root_seed", "order"]).any()
    assert set(schedule["family_hash"]) == {frozen.family_hash}
    assert set(schedule["schedule_hash"]) == {plan["schedule_hash"]}
    assert set(schedule["game_profile_sha256"]) == {profile.sha256}
    assert set(schedule["rng_scheme_version"].astype(int)) == {2}
    assert set(schedule["outcome_schema_version"].astype(int)) == {2}
    assert set(schedule["h2h_method_version"].astype(int)) == {H2H_METHOD_VERSION}
    assert set(schedule["order"].astype(int)) == {0, 1}
    assert set(schedule["order_label"]) == {"a_b", "b_a"}
    assert (
        schedule["seat1_strategy"]
        == schedule["strategy_a"].where(schedule["order"].eq(0), schedule["strategy_b"])
    ).all()
    assert (
        schedule["seat2_strategy"]
        == schedule["strategy_b"].where(schedule["order"].eq(0), schedule["strategy_a"])
    ).all()

    counts = pq.read_table(cfg.h2h_order_counts_path()).to_pandas()
    assert len(counts) == len(EXPECTED_H2H_BLOCKS) == 12
    assert not counts.duplicated(["pair_id", "root_seed", "order"]).any()
    attempts: set[tuple[int, int, int, int]] = set()
    for raw in cast(list[dict[str, Any]], counts.to_dict(orient="records")):
        key = (
            _as_int(raw["pair_id"]),
            _as_int(raw["root_seed"]),
            _as_int(raw["order"]),
        )
        expected = EXPECTED_H2H_BLOCKS[key]
        attempted = _as_int(raw["games_attempted"])
        games_completed = _as_int(raw["games_completed"])
        games_safety_limit = _as_int(raw["games_safety_limit"])
        wins_a = _as_int(raw["wins_a"])
        wins_b = _as_int(raw["wins_b"])
        wins_seat1 = _as_int(raw["wins_seat1"])
        wins_seat2 = _as_int(raw["wins_seat2"])
        order = _as_int(raw["order"])
        actual = (
            attempted,
            games_completed,
            games_safety_limit,
            wins_a,
            wins_b,
            _as_int(raw["replacement_attempt_count"]),
            str(raw["completion_status"]),
        )
        assert actual == expected
        assert attempted == games_completed + games_safety_limit
        assert wins_a + wins_b == games_completed
        assert wins_a == (wins_seat1 if order == 0 else wins_seat2)
        assert wins_b == (wins_seat2 if order == 0 else wins_seat1)
        assert _as_int(raw["authenticated_attempt_index_start"]) == 0
        assert _as_int(raw["authenticated_attempt_index_stop_exclusive"]) == attempted
        assert str(raw["attempt_coordinate_range_hash"]) == _attempt_prefix_sha256(raw)
        for attempt_index in range(attempted):
            coordinate = (*key, attempt_index)
            assert coordinate not in attempts
            attempts.add(coordinate)
    assert len(attempts) == 14
    replacement_coordinates = {coordinate for coordinate in attempts if coordinate[3] >= 1}
    assert replacement_coordinates == {(0, 11, 0, 1), (1, 11, 0, 1)}
    assert counts.loc[counts["pair_id"].eq(2), "games_attempted"].sum() == 4
    assert counts.loc[counts["pair_id"].eq(2), "games_safety_limit"].sum() == 0
    assert counts.loc[counts["pair_id"].eq(0), "games_completed"].sum() == 4
    exhausted = counts.loc[counts["completion_status"].eq("unresolved_nonviable")]
    assert len(exhausted) == 1
    assert exhausted["games_attempted"].eq(exhausted["max_attempts"]).all()
    assert (exhausted["games_completed"] < exhausted["n_completed_required"]).all()
    completed = counts.loc[counts["completion_status"].eq("complete")]
    assert completed["games_completed"].eq(completed["n_completed_required"]).all()

    state = json.loads(cfg.h2h_execution_state_path().read_text(encoding="utf-8"))
    assert state == {
        "authorized_total_game_cap": 24,
        "completed_block_count": 12,
        "execution_authorization": "ready",
        "execution_state": "complete_valid",
        "family_hash": frozen.family_hash,
        "game_profile_sha256": profile.sha256,
        "games_attempted": 14,
        "games_completed": 11,
        "games_safety_limit": 3,
        "maximum_total_attempts": 24,
        "replacement_attempt_count": 2,
        "resolved_block_count": 11,
        "schedule_hash": plan["schedule_hash"],
        "substantive_status": "unresolved_nonviable",
        "terminal_block_count": 12,
        "total_block_count": 12,
        "unresolved_block_count": 1,
    }
    assert _sum(counts, "games_attempted") == 14
    assert _sum(counts, "games_completed") == 11
    assert _sum(counts, "games_safety_limit") == 3
    assert _sum(counts, "replacement_attempt_count") == 2

    schedule_sidecar = load_authenticated_sidecar(cfg.h2h_block_manifest_path())
    assert len(schedule_sidecar.source_artifacts) == 2
    assert {source.artifact.location.stage_key for source in schedule_sidecar.source_artifacts} == {
        "candidate_freeze"
    }
    for path in sorted(cfg.h2h_block_results_dir().glob("*.parquet")):
        metadata = load_authenticated_sidecar(path)
        assert len(metadata.source_artifacts) == 1
        assert metadata.source_artifacts[0].artifact.location == (
            schedule_sidecar.artifact.location
        )
        assert metadata.method_contract.family_hash == frozen.family_hash
        assert metadata.method_contract.schedule_hash == plan["schedule_hash"]
        assert metadata.versions.method_versions["h2h_method_version"] == H2H_METHOD_VERSION

    inference = pq.read_table(cfg.h2h_pairwise_inference_path()).to_pandas().set_index("pair_id")
    assert set(inference.index.astype(int)) == {0, 1, 2}
    assert inference["multiplicity_family_member"].all()
    assert inference["decision_class"].eq("unresolved_nonviable").all()
    assert not inference["holm_reject"].any()
    assert not inference["pair_claim_eligible"].any()
    assert inference["multiplicity_method"].eq("holm").all()
    assert inference.loc[0, "formal_test_performed"]
    assert not inference.loc[1, "formal_test_performed"]
    assert inference.loc[2, "formal_test_performed"]
    assert pd.isna(inference.loc[1, "score_p_value"])
    assert pd.isna(inference.loc[1, "holm_adjusted_p"])
    assert pd.isna(inference.loc[1, "holm_order"])
    assert inference.loc[1, "no_test_p_value_convention"] == "null_reported_treated_as_one_for_holm"
    for column in (
        "q_ab",
        "q_ba",
        "d_ab",
        "ordinary_d_low",
        "ordinary_d_high",
        "simultaneous_d_low",
        "simultaneous_d_high",
        "balanced_a_wins",
        "balanced_total_games",
        "balanced_a_win_rate_alias",
    ):
        assert pd.isna(inference.loc[1, column])
    assert math.isclose(_as_float(inference.loc[0, "q_ab"]), 1.0)
    assert math.isclose(_as_float(inference.loc[0, "q_ba"]), 1.0)
    assert math.isclose(_as_float(inference.loc[0, "d_ab"]), 0.0)
    assert _as_int(inference.loc[0, "balanced_a_wins"]) == 2
    assert _as_int(inference.loc[0, "balanced_total_games"]) == 4
    assert math.isclose(_as_float(inference.loc[0, "balanced_a_win_rate_alias"]), 0.5)
    assert math.isclose(_as_float(inference.loc[2, "q_ab"]), 0.5)
    assert math.isclose(_as_float(inference.loc[2, "q_ba"]), 1.0)
    assert math.isclose(_as_float(inference.loc[2, "d_ab"]), -0.25)
    assert _as_int(inference.loc[2, "balanced_a_wins"]) == 1
    assert _as_int(inference.loc[2, "balanced_total_games"]) == 4
    assert math.isclose(_as_float(inference.loc[2, "balanced_a_win_rate_alias"]), 0.25)

    expected_candidate_status = {
        0: (10, 7, 3, 2, 0.7, False),
        1: (9, 8, 1, 1, 8 / 9, True),
        3: (9, 7, 2, 1, 7 / 9, False),
    }
    for strategy, expected in expected_candidate_status.items():
        row = inference.loc[
            (inference["strategy_a"].eq(strategy) | inference["strategy_b"].eq(strategy))
        ].iloc[0]
        prefix = "strategy_a" if _as_int(row["strategy_a"]) == strategy else "strategy_b"
        actual = (
            _as_int(row[f"{prefix}_games_attempted"]),
            _as_int(row[f"{prefix}_games_completed"]),
            _as_int(row[f"{prefix}_games_safety_limit"]),
            _as_int(row[f"{prefix}_replacement_attempt_count"]),
            _as_float(row[f"{prefix}_completion_rate"]),
            bool(row[f"{prefix}_inferentially_viable"]),
        )
        assert actual[:4] == expected[:4]
        assert math.isclose(actual[4], expected[4])
        assert actual[5] is expected[5]
        assert not bool(row[f"{prefix}_operationally_viable"])
        assert actual[4] < cfg.head2head.min_candidate_completion_rate
        assert membership.loc[membership["strategy"].eq(strategy), "final_family"].item()

    edges = pq.read_table(cfg.h2h_dominance_edges_path())
    cycles = pq.read_table(cfg.h2h_cycle_groups_path())
    fronts = pq.read_table(cfg.h2h_dominance_fronts_path()).to_pandas()
    dominance = json.loads(cfg.h2h_dominance_summary_path().read_text(encoding="utf-8"))
    assert edges.num_rows == cycles.num_rows == 0
    assert len(fronts) == 3
    assert not fronts["candidate_claim_eligible"].any()
    assert fronts["practical_wins"].eq(0).all()
    assert fronts["statistical_wins"].eq(0).all()
    assert dominance["family_hash"] == frozen.family_hash
    assert dominance["decision_counts"] == {"unresolved_nonviable": 3}
    assert dominance["practical_edge_count"] == dominance["statistical_edge_count"] == 0
    assert dominance["unique_best"] is None
    assert dominance["unique_best_claim_permitted"] is False

    agreement = pq.read_table(cfg.structure_agreement_pairs_path()).to_pandas()
    agreement_summary = json.loads(
        cfg.structure_agreement_summary_path().read_text(encoding="utf-8")
    )
    assert len(agreement) == 3
    assert set(agreement["family_hash"]) == {frozen.family_hash}
    assert agreement["decision_class"].eq("unresolved_nonviable").all()
    assert agreement["h2h_direction"].isna().all()
    assert agreement_summary["family_hash"] == frozen.family_hash
    assert agreement_summary["selection_conditioned_h2h"]["unordered_pair_count"] == 3
    assert agreement_summary["selection_conditioned_h2h"]["unresolved_nonviable_pair_count"] == 3
    assert agreement_summary["selection_conditioned_h2h"]["equivalent_pair_count"] == 0

    execute_snapshot = _data_hashes(cfg.stage_dir("h2h_execute"))
    execute_h2h_schedule(cfg, oracle_game_profile=profile)
    assert _data_hashes(cfg.stage_dir("h2h_execute")) == execute_snapshot
    for path_text, digest in (*frozen.file_sha256, *frozen.plan_file_sha256):
        assert sha256_file(Path(path_text)) == digest


def _tree_identity(path: Path) -> tuple[int, str]:
    entries = [
        {
            "relative_path": child.relative_to(path).as_posix(),
            "byte_length": child.stat().st_size,
            "content_sha256": sha256_file(child),
        }
        for child in sorted(
            (candidate for candidate in path.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.as_posix(),
        )
    ]
    return len(entries), freshness_sha256({"entries": entries})


def _assert_stamp(cfg: Any, item: Any) -> None:
    assert item.completion_stamp is not None
    stamp = json.loads(item.completion_stamp.read_text(encoding="utf-8"))
    assert stamp["lifecycle_contract_version"] == 1
    assert stamp["state"] == CompletionState.COMPLETE_VALID.value
    assert len(str(stamp["stage_identity_sha256"])) == 64
    assert stamp["outputs"]
    assert (
        resolve_stage_state(
            item.completion_stamp,
            inputs=[],
            outputs=item.required_outputs,
            cfg=cfg,
            stage=item.name,
            freshness_key=item.freshness_key,
        )
        is CompletionState.COMPLETE_VALID
    )


def _assert_sidecars(analysis_root: Path, config_sha: str, workflow_root: Path) -> None:
    assert audit_sidecar_completeness(analysis_root) == []
    sidecars = sorted(analysis_root.rglob("*.sidecar.json"))
    assert sidecars
    for metadata_path in sidecars:
        artifact = Path(str(metadata_path)[: -len(".sidecar.json")])
        metadata = validate_artifact_sidecar(artifact)
        authenticated = validate_authenticated_artifact_unbound(
            artifact,
            validate_provenance=False,
        )
        assert metadata.artifact_contract_version == 3
        assert metadata.estimand_version == 2
        assert metadata.schema_version == 2
        assert authenticated.versions.conditioning_version == 2
        assert authenticated.versions.rng_scheme_version == 2
        assert authenticated.versions.outcome_schema_version == 2
        assert metadata.artifact_sha256 == sha256_file(artifact)
        assert metadata.artifact_size_bytes == artifact.stat().st_size
        assert metadata.method_contract["procedure"] == metadata.operation
        if artifact.suffix == ".parquet":
            assert metadata.consistency_columns == pq.read_schema(artifact).names
        elif artifact.suffix == ".json":
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            if artifact.name.startswith("ratings_"):
                nested_columns = {key for value in payload.values() for key in value}
                assert set(metadata.consistency_columns) == {"strategy", *nested_columns}
            else:
                assert set(metadata.consistency_columns) == set(payload)
        scope_parts = [part for part in artifact.parts if part in _SCOPES]
        assert len(scope_parts) == 1
        assert metadata.scope == authenticated.artifact.location.scope == scope_parts[0]
        if metadata.scope == ArtifactScope.BY_K.value:
            k = authenticated.artifact.location.player_count
            assert k is not None
            assert metadata.required_player_counts == [k]
        for source in authenticated.source_artifacts:
            assert len(source.sidecar_sha256) == 64
            assert len(source.artifact.content_sha256) == 64


def assert_authenticated_analysis_graph(
    contexts: tuple[SeedRunContext, SeedRunContext],
    pair_context: RootPairRunContext,
    *,
    profile_sha256: str,
    include_reporting: bool = False,
) -> None:
    """Authenticate contexts, stages, sidecars, schemas, hashes, scopes, and lineage."""

    root_lineages: list[str] = []
    for context in contexts:
        persisted = load_run_context(
            context.run_context_path,
            active_config_path=context.active_config_path,
        )
        assert persisted["run_context_contract_version"] == 2
        assert persisted["public_config_sha256"] == context.config.config_sha
        assert persisted["lineage_extensions"] == {"game_profile_sha256": profile_sha256}
        assert Path(persisted["resolved_paths"]["results_root"]).is_absolute()
        assert len(persisted["run_context_sha256"]) == 64
        root_lineages.append(str(persisted["run_lineage_sha256"]))
        plan = analysis.build_root_stage_plan(context.config)
        for item in plan:
            _assert_stamp(context.config, item)
        config_sha = context.config.config_sha
        assert config_sha is not None
        _assert_sidecars(context.analysis_root, config_sha, pair_context.pair_root)

    pair_persisted = load_run_context(
        pair_context.run_context_path,
        active_config_path=pair_context.active_config_path,
    )
    assert pair_persisted["run_context_contract_version"] == 2
    assert pair_persisted["public_config_sha256"] == pair_context.config.config_sha
    assert pair_persisted["parent_lifecycle_roots"]
    assert len(pair_persisted["parent_lifecycle_roots"]) == 2
    assert pair_persisted["lineage_extensions"] == {"game_profile_sha256": profile_sha256}
    assert pair_persisted["run_lineage_sha256"] not in root_lineages
    pair_plan = analysis.build_root_pair_stage_plan(pair_context)
    authenticated_plan = pair_plan if include_reporting else pair_plan[:8]
    for item in authenticated_plan:
        _assert_stamp(pair_context.config, item)
    pair_config_sha = pair_context.config.config_sha
    assert pair_config_sha is not None
    _assert_sidecars(
        pair_context.analysis_root,
        pair_config_sha,
        pair_context.pair_root,
    )


def _data_hashes(stage_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(stage_dir)): sha256_file(path)
        for path in sorted(stage_dir.rglob("*"))
        if path.is_file()
        and not path.name.endswith(".done.json")
        and (path.suffix in {".parquet", ".json"} or path.name.endswith(".sidecar.json"))
    }


def assert_model_and_family_determinism(
    contexts: tuple[SeedRunContext, SeedRunContext],
    pair_context: RootPairRunContext,
) -> None:
    """Force tiny model/family reconstruction and require exact output bytes."""

    root_snapshots = {
        context.seed: {
            "trueskill": _data_hashes(context.config.trueskill_stage_dir),
            "hgb": _data_hashes(context.config.hgb_stage_dir),
        }
        for context in contexts
    }
    pair_snapshot = {
        "trueskill": _data_hashes(pair_context.config.stage_dir("trueskill")),
        "family": _data_hashes(pair_context.config.stage_dir("candidate_freeze")),
    }

    for context in contexts:
        trueskill.run(context.config, force=True)
        hgb_feat.run(context.config, force=True)
        assert (
            _data_hashes(context.config.trueskill_stage_dir)
            == root_snapshots[context.seed]["trueskill"]
        )
        assert _data_hashes(context.config.hgb_stage_dir) == root_snapshots[context.seed]["hgb"]

    trueskill.run_root_pair(pair_context.config, contexts, force=True)
    freeze_h2h_candidate_family(pair_context.config, force=True)
    assert _data_hashes(pair_context.config.stage_dir("trueskill")) == pair_snapshot["trueskill"]
    assert (
        _data_hashes(pair_context.config.stage_dir("candidate_freeze")) == pair_snapshot["family"]
    )


__all__ = [
    "assert_authenticated_analysis_graph",
    "assert_model_and_family_determinism",
    "assert_pair_candidate_oracle",
    "assert_pair_h2h_oracle",
    "assert_root_pipeline_oracle",
]
