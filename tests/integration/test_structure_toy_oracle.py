"""Two-root structural oracle spanning stability, H2H, and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis import h2h_schedule
from farkle.analysis.all_player_metrics import all_player_batch_schema
from farkle.analysis.candidate_family import freeze_h2h_candidate_family
from farkle.analysis.dominance import build_dominance_outputs
from farkle.analysis.h2h_inference import run_h2h_inference
from farkle.analysis.h2h_schedule import execute_h2h_schedule, plan_h2h_schedule
from farkle.analysis.release_audit import audit_sidecar_completeness
from farkle.analysis.root_stability import RootBatchCell, build_two_root_stability
from farkle.analysis.stage_registry import resolve_root_pair_stage_layout
from farkle.analysis.structure_agreement import run as run_structure_agreement
from farkle.analysis.structure_reporting import run as run_structure_reporting
from farkle.analysis.trueskill_screening import ScreeningRatingCell, build_percentile_contribution
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.orchestration.run_contexts import SEED_PAIR_ANALYSIS_DIRNAME, RunContextConfig
from farkle.utils.artifact_contract import (
    make_artifact_sidecar,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.stage_completion import CompletionState


def _toy_block_runner(
    block: dict[str, Any],
    _strategy_manifest: Path,
    _chunk_games: int,
) -> dict[str, Any]:
    """Return deterministic cyclic evidence for one immutable coordinate block."""

    games = int(block["games_required"])
    pair_id = int(block["pair_id"])
    order = int(block["order"])
    favors_first_named = pair_id in {0, 2}
    first_named_rate = 0.60 if favors_first_named else 0.40
    seat1_rate = first_named_rate if order == 0 else 1.0 - first_named_rate
    wins_seat1 = int(round(seat1_rate * games))
    return {
        **block,
        "games_completed": games,
        "wins_seat1": wins_seat1,
        "wins_seat2": games - wins_seat1,
    }


def _cfg(root: Path) -> AppConfig:
    base = AppConfig(
        io=IOConfig(results_dir_prefix=root / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2, 4]),
    )
    base.screening.practical_delta_by_k = {2: 0.03, 4: 0.03}
    base.screening.delta_across_k = 0.03
    base.screening.bootstrap_replicates = 20
    base.screening.candidate_contribution_size = 3
    base.screening.controls = []
    base.screening.mandatory_diagnostics = []
    base.head2head.candidate_cap = None
    base.head2head.total_game_cap = None
    pair_root = root / "results_seed_pair_11_22"
    return RunContextConfig.from_base(
        base,
        analysis_root=pair_root / SEED_PAIR_ANALYSIS_DIRNAME,
        stage_layout=resolve_root_pair_stage_layout(base),
    )


def _metric_row(
    *,
    root_seed: int,
    k: int,
    batch: int,
    strategy: int,
    wins: int,
) -> dict[str, object]:
    row: dict[str, object] = {field.name: 0 for field in all_player_batch_schema()}
    row.update(
        {
            "root_seed": root_seed,
            "k": k,
            "deterministic_batch_id": batch,
            "strategy": strategy,
            "raw_player_game_exposures": 10,
            "raw_wins": wins,
        }
    )
    for name in (
        "turn_return_turn_weighted",
        "turn_return_game_weighted_exact",
        "turn_return_round_proxy",
        "round_proxy_gap",
        "round_proxy_relative_gap",
        "turn_round_mismatch_prevalence",
    ):
        row[name] = None
    return row


def _write_root_cells(cfg: AppConfig, root: Path) -> list[RootBatchCell]:
    base_wins = {
        (11, 2): {1: 7, 2: 6, 3: 5},
        (22, 2): {1: 6, 2: 5, 3: 4},
        (11, 4): {1: 5, 2: 4, 3: 3},
        (22, 4): {1: 4, 2: 3, 3: 2},
    }
    cells: list[RootBatchCell] = []
    for (root_seed, k), strategy_values in sorted(base_wins.items()):
        rows = [
            _metric_row(
                root_seed=root_seed,
                k=k,
                batch=batch,
                strategy=strategy,
                wins=max(0, min(10, base + ((batch + strategy) % 3) - 1)),
            )
            for batch in range(100)
            for strategy, base in sorted(strategy_values.items())
        ]
        path = root / "inputs" / f"root_{root_seed}" / f"{k}p_metrics.parquet"
        table = pa.Table.from_pylist(rows, schema=all_player_batch_schema())
        sidecar = make_artifact_sidecar(
            cfg,
            path,
            producer="toy_oracle",
            scope=ArtifactScope.BY_K,
            source_scope=ArtifactScope.BY_K,
            operation="aggregate_player_batch_statistics",
            conditioning="unconditional",
            consistency_columns=table.schema.names,
            grouping_keys=["root_seed", "k", "deterministic_batch_id", "strategy"],
            player_counts=[k],
            required_player_counts=[k],
            missing_cell_policy="fail",
            seed_scope="single_root",
        )
        write_parquet_artifact_atomic(table, path, sidecar=sidecar)
        cells.append(RootBatchCell(root_seed=root_seed, k=k, path=path))
    return cells


def _write_trueskill_contribution(cfg: AppConfig, root: Path) -> Path:
    cells: list[ScreeningRatingCell] = []
    for root_seed in (11, 22):
        for k in (2, 4):
            path = (
                root
                / "inputs"
                / "toy_root_ratings"
                / f"root_{root_seed}"
                / f"{k}p"
                / f"ratings_{k}_seed{root_seed}.parquet"
            )
            table = pa.table(
                {
                    "strategy": [1, 2, 3],
                    "mu": [30.0 + root_seed / 100, 25.0 + k / 100, 20.0],
                    "sigma": [2.0, 2.5, 3.0],
                }
            )
            sidecar = make_artifact_sidecar(
                cfg,
                path,
                producer="toy_oracle",
                scope=ArtifactScope.BY_K,
                source_scope=ArtifactScope.BY_K,
                operation="sequential_rating",
                conditioning="finite_grid_trueskill_screening",
                consistency_columns=table.schema.names,
                grouping_keys=["strategy"],
                player_counts=[k],
                required_player_counts=[k],
                missing_cell_policy="fail",
                seed_scope="single_root",
            )
            write_parquet_artifact_atomic(table, path, sidecar=sidecar)
            cells.append(ScreeningRatingCell(root_seed, k, path))
    return build_percentile_contribution(cfg, cells)


def _prepare(cfg: AppConfig, root: Path) -> tuple[Any, Any, Any]:
    stability = build_two_root_stability(cfg, _write_root_cells(cfg, root))
    _write_trueskill_contribution(cfg, root)
    family = freeze_h2h_candidate_family(cfg)
    schedule = plan_h2h_schedule(cfg)
    manifest_path = cfg.strategy_manifest_root_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"strategy_id": [1, 2, 3]}), manifest_path)
    return stability, family, schedule


def _finish(cfg: AppConfig) -> tuple[Any, Any, dict[str, Any]]:
    inference = run_h2h_inference(cfg)
    dominance = build_dominance_outputs(cfg)
    run_structure_agreement(cfg, execution_scope="root_pair")
    run_structure_reporting(cfg, execution_scope="root_pair")
    report = json.loads(cfg.structure_report_json_path().read_text(encoding="utf-8"))
    return inference, dominance, report


def _logical_table(path: Path, keys: list[str]) -> pd.DataFrame:
    return (
        pq.read_table(path).to_pandas().sort_values(keys, kind="mergesort").reset_index(drop=True)
    )


def _assert_one_sidecar(paths: list[Path]) -> None:
    sidecars = [sidecar_path(path) for path in paths]
    assert len(sidecars) == len(set(sidecars)) == len(paths)
    for path, metadata_path in zip(paths, sidecars, strict=True):
        assert path.exists()
        assert metadata_path.exists()
        assert list(path.parent.glob(f"{path.name}.sidecar.json")) == [metadata_path]
        validate_artifact_sidecar(path)


@pytest.mark.integration
def test_two_root_multi_k_resume_matches_worker_count_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_cfg = _cfg(tmp_path / "baseline")
    baseline_stability, baseline_family, baseline_schedule = _prepare(
        baseline_cfg, tmp_path / "baseline"
    )
    monkeypatch.setattr(h2h_schedule, "_simulate_block", _toy_block_runner)
    baseline_execution = execute_h2h_schedule(
        baseline_cfg,
        n_jobs=1,
        block_runner=_toy_block_runner,
    )
    baseline_inference, baseline_dominance, baseline_report = _finish(baseline_cfg)

    resumed_cfg = _cfg(tmp_path / "resumed")
    resumed_stability, resumed_family, resumed_schedule = _prepare(
        resumed_cfg, tmp_path / "resumed"
    )
    calls = 0

    def interrupting_runner(
        block: dict[str, Any],
        manifest: Path,
        chunk_games: int,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("intentional toy interruption")
        return _toy_block_runner(block, manifest, chunk_games)

    with pytest.raises(RuntimeError, match="intentional toy interruption"):
        execute_h2h_schedule(
            resumed_cfg,
            n_jobs=1,
            block_runner=interrupting_runner,
        )
    interrupted_state = json.loads(
        resumed_cfg.h2h_execution_state_path().read_text(encoding="utf-8")
    )
    assert interrupted_state["execution_state"] == CompletionState.PARTIAL_RESUMABLE.value
    assert interrupted_state["completed_block_count"] == 2
    immutable_resumed_plan = resumed_cfg.h2h_power_plan_path().read_bytes()

    resumed_execution = execute_h2h_schedule(
        resumed_cfg,
        n_jobs=2,
        block_runner=_toy_block_runner,
    )
    resumed_inference, resumed_dominance, resumed_report = _finish(resumed_cfg)

    baseline_family_payload = json.loads(baseline_family.manifest.read_text(encoding="utf-8"))
    resumed_family_payload = json.loads(resumed_family.manifest.read_text(encoding="utf-8"))
    baseline_plan = json.loads(baseline_schedule.power_plan.read_text(encoding="utf-8"))
    resumed_plan = json.loads(resumed_schedule.power_plan.read_text(encoding="utf-8"))
    assert baseline_family_payload["family_hash"] == resumed_family_payload["family_hash"]
    assert baseline_plan["schedule_hash"] == resumed_plan["schedule_hash"]
    assert resumed_cfg.h2h_power_plan_path().read_bytes() == immutable_resumed_plan
    resumed_state = json.loads(resumed_execution.execution_state.read_text(encoding="utf-8"))
    assert resumed_state["execution_state"] == CompletionState.COMPLETE_VALID.value

    coordinate_columns = [
        "pair_id",
        "root_seed",
        "root_index",
        "order",
        "block_id",
        "schedule_hash",
    ]
    pd.testing.assert_frame_equal(
        _logical_table(baseline_schedule.block_manifest, coordinate_columns)[coordinate_columns],
        _logical_table(resumed_schedule.block_manifest, coordinate_columns)[coordinate_columns],
    )
    pd.testing.assert_frame_equal(
        _logical_table(
            baseline_execution.order_counts,
            ["pair_id", "root_index", "order"],
        ),
        _logical_table(
            resumed_execution.order_counts,
            ["pair_id", "root_index", "order"],
        ),
    )
    pd.testing.assert_frame_equal(
        _logical_table(
            baseline_inference.pairwise_inference,
            ["pair_id"],
        ),
        _logical_table(
            resumed_inference.pairwise_inference,
            ["pair_id"],
        ),
    )
    pd.testing.assert_frame_equal(
        _logical_table(baseline_dominance.fronts, ["strategy"]),
        _logical_table(resumed_dominance.fronts, ["strategy"]),
    )
    pd.testing.assert_frame_equal(
        _logical_table(baseline_dominance.cycles, ["graph_type", "strategy"]),
        _logical_table(resumed_dominance.cycles, ["graph_type", "strategy"]),
    )
    assert baseline_report["support"] == resumed_report["support"]
    assert baseline_report["claim_language"] == resumed_report["claim_language"]
    assert baseline_report["h2h"]["pair_intervals"] == resumed_report["h2h"]["pair_intervals"]
    assert baseline_report["h2h"]["unique_best_claim_permitted"] is False
    assert baseline_report["h2h"]["cycle_group_count"] >= 1

    derived = [
        *resumed_stability.all_paths,
        *resumed_family.all_paths,
        resumed_schedule.power_plan,
        resumed_schedule.block_manifest,
        resumed_execution.execution_state,
        resumed_execution.order_counts,
        *resumed_execution.block_paths,
        *resumed_inference.all_paths,
        *resumed_dominance.all_paths,
        resumed_cfg.structure_agreement_pairs_path(),
        resumed_cfg.structure_agreement_summary_path(),
        resumed_cfg.structure_report_json_path(),
        resumed_cfg.structure_report_markdown_path(),
        resumed_cfg.structure_report_plot_path(),
        resumed_cfg.migration_report_path(),
    ]
    _assert_one_sidecar(derived)
    assert audit_sidecar_completeness(resumed_cfg.analysis_dir) == []
    expected_stage_dirs = [
        "00_root_stability",
        "01_trueskill",
        "02_candidate_freeze",
        "03_h2h_power",
        "04_h2h_execute",
        "05_h2h_inference",
        "06_h2h_digest",
        "07_agreement",
        "08_reporting",
    ]
    observed_stage_dirs = sorted(
        path.name for path in resumed_cfg.analysis_dir.iterdir() if path.is_dir()
    )
    assert observed_stage_dirs == expected_stage_dirs
    assert all(any(path.iterdir()) for path in resumed_cfg.analysis_dir.iterdir() if path.is_dir())
