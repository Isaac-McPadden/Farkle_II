"""Synthetic H2H scheduler evidence. Plans issued here are ineligible for release."""

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa

from farkle.analysis.candidate_family import freeze_h2h_candidate_family
from farkle.analysis.h2h_schedule import (
    DEFAULT_H2H_CHECKPOINT_ATTEMPT_LIMIT,
    BlockRunner,
    H2HExecutionArtifacts,
    H2HScheduleArtifacts,
    _execute_h2h_schedule,
    _plan_h2h_schedule,
)
from farkle.analysis.stage_registry import resolve_root_pair_stage_layout
from farkle.analysis.trueskill_screening import TRUESKILL_CONDITIONING, trueskill_method_contract
from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import make_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.frozen_h2h import SYNTHETIC_ENGINE


def plan_oracle(cfg: AppConfig, *, force: bool = False) -> H2HScheduleArtifacts:
    return _plan_h2h_schedule(cfg, force=force, engine=SYNTHETIC_ENGINE)


def execute_oracle(
    cfg: AppConfig,
    *,
    block_runner: BlockRunner,
    n_jobs: int | None = None,
    chunk_games: int = DEFAULT_H2H_CHECKPOINT_ATTEMPT_LIMIT,
) -> H2HExecutionArtifacts:
    return _execute_h2h_schedule(
        cfg, n_jobs=n_jobs, chunk_games=chunk_games, synthetic_runner=block_runner
    )


def write_fixture_contribution(
    cfg: AppConfig,
    path: Path,
    frame: pd.DataFrame,
    *,
    scope: ArtifactScope,
    operation: str,
    seed_scope: str,
    current_trueskill_contract: bool = True,
    sources: tuple[Path, ...] = (),
) -> None:
    frame = frame.copy()
    frame["strategy"] = pd.array(frame["strategy"].tolist(), dtype="Int32")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    is_trueskill = operation == "equal_root_k_percentile_mean" and current_trueskill_contract
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=scope,
        source_scope=scope,
        source_artifacts=list(sources),
        operation=operation,
        consistency_columns=table.schema.names,
        player_counts=list(cfg.sim.n_players_list),
        required_player_counts=list(cfg.sim.n_players_list),
        missing_cell_policy="fail",
        seed_scope=seed_scope,
        conditioning=TRUESKILL_CONDITIONING if is_trueskill else "unconditional",
        method_contract=cast(
            Any,
            trueskill_method_contract(operation) if is_trueskill else None,
        ),
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar)


def publish_fixture_family(cfg: AppConfig, candidates: list[int]) -> None:
    """Freeze a disposable family through the real contribution and publication contracts."""
    pair = len(cfg.sim.seed_list or [cfg.sim.seed]) == 2
    if pair:
        cfg.set_stage_layout(resolve_root_pair_stage_layout(cfg))
    scores = [1 - i / len(candidates) for i in range(len(candidates))]
    win = pd.DataFrame({"strategy": candidates, "complete_support": True})
    if pair:
        win["estimate_scope"] = "combined_roots"
        win["across_k_score"] = scores
    else:
        win["equal_k_score"] = scores
    write_fixture_contribution(
        cfg,
        cfg.root_combined_performance_across_k_path() if pair else cfg.performance_across_k_path(),
        win,
        scope=ArtifactScope.CROSS_SEED if pair else ArtifactScope.ACROSS_K,
        operation="equal_k_mean",
        seed_scope="root_pair_stability" if pair else "single_root",
    )
    ratings = pd.DataFrame(
        {
            "strategy": candidates,
            "complete_support": True,
            "mean_percentile_rank": scores,
            "candidate_contribution_rank": range(1, len(candidates) + 1),
        }
    )
    write_fixture_contribution(
        cfg,
        cfg.trueskill_candidate_contribution_path(),
        ratings,
        scope=ArtifactScope.ACROSS_K,
        operation="equal_root_k_percentile_mean",
        seed_scope="both_roots_combined" if pair else "single_root",
    )
    cfg.screening.candidate_contribution_size = len(candidates)
    freeze_h2h_candidate_family(cfg)
