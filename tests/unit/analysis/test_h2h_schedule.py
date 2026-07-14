from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.h2h_schedule import (
    SCORE_TEST_ID,
    execute_h2h_schedule,
    independent_score_planning_power,
    plan_h2h_schedule,
)
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import (
    write_json_artifact_atomic,
    write_parquet_artifact_atomic,
)
from farkle.utils.random import RandomPurpose, coordinate_seed


def _cfg(tmp_path: Path, *, roots: tuple[int, ...] = (11, 22)) -> AppConfig:
    sim = SimConfig(
        seed=roots[0],
        seed_list=list(roots),
        seed_pair=roots if len(roots) == 2 else None,
        n_players_list=[2, 4],
    )
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=sim,
    )
    cfg.screening.practical_delta_by_k = {2: 0.03, 4: 0.03}
    cfg.screening.delta_across_k = 0.03
    cfg.head2head.total_game_cap = None
    cfg.head2head.n_jobs = 1
    return cfg


def _write_frozen_family(cfg: AppConfig, strategies: tuple[str, ...] = ("1", "2", "3")) -> None:
    family_hash = "a" * 64
    roots = cfg.sim.seed_list or [cfg.sim.seed]
    membership = pd.DataFrame(
        {
            "strategy": list(strategies),
            "final_family": [True] * len(strategies),
            "family_hash": [family_hash] * len(strategies),
        }
    )
    manifest: dict[str, Any] = {
        "family_hash": family_hash,
        "candidates": list(strategies),
        "candidate_count": len(strategies),
        "root_seeds": roots,
        "single_root_execution": len(roots) == 1,
    }
    common: dict[str, Any] = {
        "producer": "test",
        "scope": ArtifactScope.H2H_2P,
        "source_scope": ArtifactScope.CROSS_SEED,
        "operation": "candidate_family_freeze",
        "player_counts": [2],
        "required_player_counts": [2],
        "missing_cell_policy": "fail",
        "seed_scope": "both_roots_combined" if len(roots) == 2 else "single_root",
    }
    membership_path = cfg.h2h_candidate_family_path()
    membership_table = pa.Table.from_pandas(membership, preserve_index=False)
    membership_sidecar = make_artifact_sidecar(
        cfg,
        membership_path,
        consistency_columns=membership.columns.tolist(),
        **common,
    )
    write_parquet_artifact_atomic(
        membership_table,
        membership_path,
        sidecar=membership_sidecar,
    )
    manifest_path = cfg.h2h_candidate_family_manifest_path()
    manifest_sidecar = make_artifact_sidecar(
        cfg,
        manifest_path,
        consistency_columns=list(manifest),
        **common,
    )
    write_json_artifact_atomic(manifest, manifest_path, sidecar=manifest_sidecar)


def test_score_power_is_monotone_in_games_and_effect() -> None:
    alpha = 0.02 / 10

    low_games = independent_score_planning_power(1_000, 0.56, 0.50, alpha)
    high_games = independent_score_planning_power(2_000, 0.56, 0.50, alpha)
    larger_effect = independent_score_planning_power(1_000, 0.58, 0.48, alpha)

    assert high_games > low_games
    assert larger_effect > low_games


def test_power_plan_and_two_root_block_allocation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_frozen_family(cfg)

    artifacts = plan_h2h_schedule(cfg)

    assert artifacts.schedule_state == "ready"
    assert artifacts.block_manifest is not None
    plan = json.loads(artifacts.power_plan.read_text(encoding="utf-8"))
    assert plan["score_test_id"] == SCORE_TEST_ID
    assert plan["alpha_per_pair"] == pytest.approx(0.02 / 3)
    assert plan["games_per_pair"] % 4 == 0
    assert plan["worst_scenario_achieved_power"] >= 0.80
    assert plan["previous_equal_block_size_worst_power"] < 0.80
    target_rows = [
        row for row in plan["power_validation"] if row["reported_effect"] == pytest.approx(0.03)
    ]
    assert len(target_rows) == 3
    assert min(row["achieved_power"] for row in target_rows) >= 0.80

    schedule = pq.read_table(artifacts.block_manifest).to_pandas()
    assert len(schedule) == 3 * 2 * 2
    assert set(schedule["root_seed"]) == {11, 22}
    assert set(schedule["order"]) == {0, 1}
    assert schedule.groupby("pair_id")["games_required"].nunique().eq(1).all()
    assert schedule["block_id"].is_unique
    first_pair = schedule.loc[schedule["pair_id"].eq(0)]
    assert len(first_pair) == 4
    assert first_pair["games_required"].sum() == plan["games_per_pair"]

    first = schedule.sort_values(["root_seed", "order"]).iloc[0]
    order_seed = coordinate_seed(
        RandomPurpose.H2H_GAME,
        root_seed=int(first["root_seed"]),
        k=2,
        pair_index=int(first["pair_id"]),
        order=int(first["order"]),
        game_index=0,
    )
    other = schedule.loc[
        (schedule["pair_id"] == first["pair_id"])
        & ((schedule["root_seed"] != first["root_seed"]) | (schedule["order"] != first["order"]))
    ].iloc[0]
    other_seed = coordinate_seed(
        RandomPurpose.H2H_GAME,
        root_seed=int(other["root_seed"]),
        k=2,
        pair_index=int(other["pair_id"]),
        order=int(other["order"]),
        game_index=0,
    )
    assert order_seed != other_seed

    original = pq.read_table(artifacts.block_manifest).to_pandas()
    plan_h2h_schedule(cfg, force=True)
    replay = pq.read_table(artifacts.block_manifest).to_pandas()
    pd.testing.assert_frame_equal(original, replay)


def test_power_plan_stops_before_schedule_when_cap_is_too_small(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.head2head.total_game_cap = 1
    _write_frozen_family(cfg)

    artifacts = plan_h2h_schedule(cfg)
    plan = json.loads(artifacts.power_plan.read_text(encoding="utf-8"))

    assert artifacts.schedule_state == "blocked_by_cap"
    assert artifacts.block_manifest is None
    assert not cfg.h2h_block_manifest_path().exists()
    assert plan["projected_total_games"] > 1
    assert "head2head.total_game_cap" in plan["cap_guidance"]

    cfg.head2head.total_game_cap = plan["projected_total_games"]
    resumed = plan_h2h_schedule(cfg)
    assert resumed.schedule_state == "ready"
    assert resumed.block_manifest is not None


def test_single_root_plan_is_explicit_and_even_by_order(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, roots=(11,))
    _write_frozen_family(cfg)

    artifacts = plan_h2h_schedule(cfg)
    plan = json.loads(artifacts.power_plan.read_text(encoding="utf-8"))
    block_manifest = artifacts.block_manifest
    assert block_manifest is not None
    schedule = pq.read_table(block_manifest).to_pandas()

    assert plan["single_root_execution"] is True
    assert plan["games_per_pair"] % 2 == 0
    assert len(schedule) == 3 * 2
    assert set(schedule["root_seed"]) == {11}
    validate_artifact_sidecar(
        block_manifest,
        expected={"scope": "h2h_2p", "seed_scope": "single_root"},
    )


def test_single_root_plan_respects_explicit_disable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, roots=(11,))
    cfg.head2head.allow_single_root = False
    _write_frozen_family(cfg)

    with pytest.raises(ValueError, match="single-root H2H is disabled"):
        plan_h2h_schedule(cfg)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sensitivity_deltas", (0.03,), "sensitivity_deltas"),
        ("sensitivity_deltas", (0.03, 0.04, 0.04), "sensitivity_deltas"),
        ("seat1_advantage_scenarios", (0.0, 0.03), "seat1_advantage_scenarios"),
    ],
)
def test_statistical_contract_locks_h2h_planning_scenarios(
    tmp_path: Path,
    field: str,
    value: tuple[float, ...],
    message: str,
) -> None:
    cfg = _cfg(tmp_path)
    setattr(cfg.head2head, field, value)

    with pytest.raises(ValueError, match=message):
        cfg.validate_statistical_contract(require_two_roots=True)


def test_block_checkpoints_resume_without_regenerating_completed_work(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_frozen_family(cfg, strategies=("1", "2"))
    plan_h2h_schedule(cfg)
    manifest_path = cfg.strategy_manifest_root_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"strategy_id": pa.array([], type=pa.int64())}), manifest_path)
    calls: list[str] = []

    def fake_runner(block: dict[str, Any], _manifest: Path, _chunk: int) -> dict[str, Any]:
        calls.append(str(block["block_id"]))
        games = int(block["games_required"])
        return {
            **block,
            "games_completed": games,
            "wins_seat1": games // 2,
            "wins_seat2": games - games // 2,
        }

    completed = execute_h2h_schedule(cfg, n_jobs=1, block_runner=fake_runner)

    assert len(calls) == 4
    counts = pq.read_table(completed.order_counts).to_pandas()
    assert len(counts) == 4
    assert (counts["games_completed"] == counts["games_required"]).all()
    assert all(path.exists() for path in completed.block_paths)

    def fail_if_called(
        _block: dict[str, Any], _manifest: Path, _chunk: int
    ) -> dict[str, Any]:  # pragma: no cover - completed blocks must skip
        raise AssertionError("completed immutable block was regenerated")

    replay = execute_h2h_schedule(cfg, n_jobs=1, block_runner=fail_if_called)
    pd.testing.assert_frame_equal(
        pq.read_table(completed.order_counts).to_pandas(),
        pq.read_table(replay.order_counts).to_pandas(),
    )
