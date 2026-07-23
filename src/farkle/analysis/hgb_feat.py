# src/farkle/analysis/hgb_feat.py
"""Configuration wrapper for held-out HGB association exploration."""

from __future__ import annotations

import logging
from typing import Final

from farkle.analysis import run_hgb as _hgb
from farkle.analysis import stage_logger
from farkle.config import AppConfig
from farkle.utils.artifact_contract import validate_artifact_sidecar
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done

LOGGER = logging.getLogger(__name__)

HGB_METHOD_VERSION: Final = 2
HGB_RNG_METHOD_VERSION: Final = 2
HGB_FOLD_CONSTRUCTION_VERSION: Final = 1


def _hgb_freshness_key(cfg: AppConfig) -> dict[str, object]:
    """Return the explicit statistical identity for HGB completion/reuse."""

    return {
        **cfg.freshness_key(),
        "hgb_method_version": HGB_METHOD_VERSION,
        "hgb_rng_method_version": HGB_RNG_METHOD_VERSION,
        "hgb_fold_construction_version": HGB_FOLD_CONSTRUCTION_VERSION,
        "target": "win_rate",
        "feature_specification": [list(specification) for specification in _hgb.FEATURE_SPECS],
        "fold_unit": "whole_strategy_configuration",
        "claim_scope": "predictive_association_not_causal",
    }


def run(cfg: AppConfig, *, force: bool = False) -> None:
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
    if manifest_path is None or not manifest_path.exists():
        stage_log.missing_input(
            "canonical strategy feature manifest missing",
            paths=[] if manifest_path is None else [str(manifest_path)],
        )
        return
    inputs = [*metrics_paths, manifest_path]
    outputs = [
        json_out,
        cfg.hgb_future_proposals_path(),
        cfg.concat_ks_dir("hgb") / _hgb.LONG_IMPORTANCE_NAME,
        cfg.across_k_dir("hgb") / _hgb.OVERALL_IMPORTANCE_NAME,
        *importance_paths,
        *predictive_paths,
        *fold_paths,
    ]
    done = stage_done_path(cfg.hgb_stage_dir, "hgb")
    freshness_key = _hgb_freshness_key(cfg)
    if not force and stage_is_up_to_date(
        done,
        inputs=inputs,
        outputs=outputs,
        cfg=cfg,
        stage="hgb",
        freshness_key=freshness_key,
        sidecar_artifacts=outputs,
    ):
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
    missing_outputs = [path for path in outputs if not path.exists()]
    if missing_outputs:
        raise RuntimeError(f"HGB did not publish required outputs: {missing_outputs}")
    write_stage_done(
        done,
        inputs=inputs,
        outputs=outputs,
        cfg=cfg,
        stage="hgb",
        freshness_key=freshness_key,
        sidecar_artifacts=outputs,
    )
    LOGGER.info(
        "HGB feature importance complete",
        extra={"stage": "hgb", "path": str(json_out)},
    )
