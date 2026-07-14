"""Build the canonical tournament metrics, performance, and seat artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from farkle.analysis.all_player_metrics import build_all_player_batch_metrics
from farkle.analysis.checks import check_pre_metrics
from farkle.analysis.performance import PerformanceArtifacts, build_canonical_performance
from farkle.analysis.seat_analysis import SeatAnalysisArtifacts, build_canonical_seat_analysis
from farkle.config import AppConfig
from farkle.utils.artifact_contract import sidecar_path
from farkle.utils.parallel import process_map, resolve_mp_context
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done

LOGGER = logging.getLogger(__name__)


def _build_all_player_cell(task: tuple[AppConfig, int]) -> tuple[int, Path]:
    cfg, k = task
    return k, build_all_player_batch_metrics(cfg, k)


def _require_paths(paths: Sequence[Path], *, label: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"metrics: incomplete {label} support; missing {joined}")


def _all_player_metrics(cfg: AppConfig, player_counts: Sequence[int]) -> list[Path]:
    tasks = [(cfg, int(k)) for k in player_counts]
    results = process_map(
        _build_all_player_cell,
        tasks,
        n_jobs=cfg.analysis.n_jobs,
        mp_context=resolve_mp_context(cfg.analysis.mp_start_method),
    )
    return [path for _, path in sorted(results)]


def run(cfg: AppConfig) -> None:
    """Build canonical outputs over complete configured player-count support."""

    player_counts = sorted({int(k) for k in cfg.sim.n_players_list})
    if not player_counts:
        raise ValueError("metrics: sim.n_players_list must not be empty")
    if cfg.screening.delta_across_k is None:
        raise ValueError("metrics: screening.delta_across_k must be explicitly configured")

    concatenated_rows = cfg.curated_dataset
    per_k_curated = [cfg.ingested_rows_curated(k) for k in player_counts]
    per_k_combined = [cfg.combined_rows_by_k(k) for k in player_counts]
    _require_paths([concatenated_rows], label="concat_ks")
    _require_paths(per_k_curated, label="curated by_k")
    _require_paths(per_k_combined, label="combined by_k")
    check_pre_metrics(concatenated_rows, winner_col="winner_seat")

    all_player_outputs = [cfg.metrics_all_player_batch_path(k) for k in player_counts]
    performance_outputs = [
        *(cfg.performance_by_k_path(k) for k in player_counts),
        cfg.performance_across_k_path(),
        cfg.performance_bootstrap_path(),
        cfg.performance_control_contrasts_path(),
        cfg.performance_player_count_effects_path(),
    ]
    seat_outputs = [
        *(cfg.seat_batch_counts_path(k) for k in player_counts),
        *(cfg.seat_effects_by_k_path(k) for k in player_counts),
        *(cfg.seat_population_by_k_path(k) for k in player_counts),
        cfg.seat_standardized_across_k_path(),
        cfg.seat_exposure_mixture_diagnostic_path(),
        cfg.seat_selfplay_diagnostic_path(),
        cfg.seat_mirrored_diagnostic_path(),
    ]
    outputs = [*all_player_outputs, *performance_outputs, *seat_outputs]
    inputs = [concatenated_rows, *per_k_curated, *per_k_combined]
    done = stage_done_path(cfg.metrics_stage_dir, "metrics")
    if stage_is_up_to_date(
        done,
        inputs=inputs,
        outputs=outputs,
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=outputs,
    ):
        LOGGER.info("Metrics stage up-to-date", extra={"stage": "metrics", "path": str(done)})
        return

    all_player_paths = _all_player_metrics(cfg, player_counts)
    performance: PerformanceArtifacts = build_canonical_performance(cfg, force=True)
    seats: SeatAnalysisArtifacts = build_canonical_seat_analysis(cfg, force=True)
    emitted = [*all_player_paths, *performance.all_paths, *seats.all_paths]
    _require_paths(emitted, label="derived artifact")
    _require_paths([sidecar_path(path) for path in emitted], label="derived sidecar")
    write_stage_done(
        done,
        inputs=inputs,
        outputs=emitted,
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=emitted,
    )
    LOGGER.info(
        "Metrics stage complete",
        extra={
            "stage": "metrics",
            "player_counts": player_counts,
            "artifacts": len(emitted),
        },
    )
