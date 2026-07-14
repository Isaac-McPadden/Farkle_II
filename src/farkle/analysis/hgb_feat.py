# src/farkle/analysis/hgb_feat.py
"""Configuration wrapper for held-out HGB association exploration."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from farkle.analysis import run_hgb as _hgb
from farkle.analysis import stage_logger
from farkle.config import AppConfig
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    sidecar_path,
    validate_artifact_sidecar,
)

LOGGER = logging.getLogger(__name__)


def _latest_mtime(paths: Iterable[Path]) -> float:
    """Compute the most recent modification time among existing paths.

    Args:
        paths: Files whose modification times should be checked.

    Returns:
        Most recent mtime in seconds since the epoch, or 0.0 when none exist.
    """
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes, default=0.0)


def run(cfg: AppConfig) -> None:
    """Train histogram gradient boosting feature importance models per player count.

    Args:
        cfg: Application configuration containing analysis directories and names.
    """
    stage_log = stage_logger("hgb", logger=LOGGER)
    stage_log.start()

    analysis_dir = cfg.hgb_stage_dir
    players = sorted({int(k) for k in cfg.sim.n_players_list})
    metrics_paths = [cfg.performance_by_k_path(k) for k in players]
    missing = [path for path in metrics_paths if not path.exists()]
    if missing:
        stage_log.missing_input(
            "canonical per-k performance artifact missing",
            paths=[str(path) for path in missing],
        )
        return
    for path in metrics_paths:
        validate_artifact_sidecar(path, expected={"scope": "by_k"})
    json_out = cfg.across_k_dir("hgb") / "hgb_importance.json"

    importance_paths = [cfg.hgb_importance_path(k) for k in players]
    predictive_paths = [cfg.hgb_predictive_scores_path(k) for k in players]
    fold_paths = [cfg.hgb_fold_metrics_path(k) for k in players]
    manifest_path = cfg.strategy_manifest_root_path() if players else None

    inputs = [*metrics_paths]
    if manifest_path is not None and manifest_path.exists():
        inputs.append(manifest_path)
    latest_input = _latest_mtime(inputs)
    outputs = [
        json_out,
        cfg.hgb_future_proposals_path(),
        cfg.across_k_dir("hgb") / _hgb.LONG_IMPORTANCE_NAME,
        cfg.across_k_dir("hgb") / _hgb.OVERALL_IMPORTANCE_NAME,
        *importance_paths,
        *predictive_paths,
        *fold_paths,
    ]

    outputs_look_fresh = outputs and all(
        p.exists() and sidecar_path(p).exists() and p.stat().st_mtime >= latest_input
        for p in outputs
    )
    if outputs_look_fresh:
        try:
            for path in outputs:
                validate_artifact_sidecar(path)
        except ArtifactContractError:
            outputs_look_fresh = False
    if outputs_look_fresh:
        LOGGER.info(
            "HGB feature importance up-to-date",
            extra={"stage": "hgb", "path": str(json_out)},
        )
        return

    LOGGER.info(
        "HGB feature importance running",
        extra={
            "stage": "hgb",
            "analysis_dir": str(analysis_dir),
        },
    )
    _hgb.run_hgb(
        cfg=cfg,
        metrics_paths=metrics_paths,
        manifest_path=manifest_path,
    )
    LOGGER.info(
        "HGB feature importance complete",
        extra={"stage": "hgb", "path": str(json_out)},
    )
