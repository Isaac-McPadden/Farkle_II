# src/farkle/analysis/interseed_analysis.py
"""Run only the cross-seed analysis stages and record a summary."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from farkle.analysis import stage_logger
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)

SUMMARY_NAME = "interseed_summary.json"


def run(cfg: AppConfig, *, force: bool = False, run_stages: bool = True) -> None:
    """Execute or summarize variance/meta/TrueSkill/agreement interseed analytics."""

    stage_log = stage_logger("interseed", logger=LOGGER)
    stage_log.start()

    if not cfg.analysis.run_interseed:
        LOGGER.info(
            "Interseed analysis skipped",
            extra={"stage": "interseed", "reason": "run_interseed=False"},
        )
        return

    statuses: dict[str, dict[str, Any]] = {}

    variance_enabled = True
    meta_enabled = True
    trueskill_enabled = cfg.analysis.run_trueskill and not cfg.analysis.disable_trueskill
    agreement_enabled = cfg.analysis.run_agreement and not cfg.analysis.disable_agreement

    if variance_enabled and run_stages:
        from farkle.analysis import variance

        variance.run(cfg, force=force)
    statuses["variance"] = {
        "enabled": variance_enabled,
        "outputs": _existing_paths(_variance_outputs(cfg)),
    }

    if meta_enabled and run_stages:
        from farkle.analysis import meta

        meta.run(cfg, force=force)
    statuses["meta"] = {
        "enabled": meta_enabled,
        "outputs": _existing_paths(_meta_outputs(cfg)),
    }

    if trueskill_enabled and run_stages:
        from farkle.analysis import trueskill

        trueskill.run(cfg)
    statuses["trueskill"] = {
        "enabled": trueskill_enabled,
        "outputs": _existing_paths(_trueskill_outputs(cfg)),
    }

    if agreement_enabled and run_stages:
        from farkle.analysis import agreement

        agreement.run(cfg)
    statuses["agreement"] = {
        "enabled": agreement_enabled,
        "outputs": _existing_paths(_agreement_outputs(cfg)),
    }

    summary_path = cfg.interseed_stage_dir / SUMMARY_NAME
    done_path = stage_done_path(cfg.interseed_stage_dir, "interseed")
    inputs = sorted({Path(path) for path in _flatten_output_paths(statuses)})

    if not force and stage_is_up_to_date(
        done_path,
        inputs=inputs,
        outputs=[summary_path],
        config_sha=cfg.config_sha,
    ):
        LOGGER.info(
            "Interseed summary up-to-date",
            extra={"stage": "interseed", "path": str(summary_path)},
        )
        return

    payload = {
        "config_sha": cfg.config_sha,
        "run_interseed": cfg.analysis.run_interseed,
        "stages": statuses,
    }
    with atomic_path(str(summary_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    write_stage_done(
        done_path,
        inputs=inputs,
        outputs=[summary_path],
        config_sha=cfg.config_sha,
    )

    LOGGER.info(
        "Interseed summary written",
        extra={"stage": "interseed", "path": str(summary_path)},
    )


def _existing_paths(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if path.exists()]


def _flatten_output_paths(statuses: dict[str, dict[str, Any]]) -> list[str]:
    outputs: list[str] = []
    for details in statuses.values():
        outputs.extend(details.get("outputs", []))
    return outputs


def _variance_outputs(cfg: AppConfig) -> list[Path]:
    return [
        cfg.variance_output_path("variance.parquet"),
        cfg.variance_output_path("variance_summary.parquet"),
        cfg.variance_output_path("variance_components.parquet"),
    ]


def _meta_outputs(cfg: AppConfig) -> list[Path]:
    outputs: list[Path] = []
    for players in sorted({int(n) for n in cfg.sim.n_players_list}):
        outputs.append(cfg.meta_output_path(players, f"strategy_summary_{players}p_meta.parquet"))
        outputs.append(cfg.meta_output_path(players, f"meta_{players}p.json"))
    return outputs


def _trueskill_outputs(cfg: AppConfig) -> list[Path]:
    outputs = [
        cfg.trueskill_path("ratings_pooled.parquet"),
        cfg.trueskill_path("ratings_pooled.json"),
        cfg.trueskill_stage_dir / "tiers.json",
    ]
    outputs.extend(sorted(cfg.trueskill_pooled_dir.glob("ratings_pooled_seed*.parquet")))
    outputs.extend(sorted(cfg.trueskill_stage_dir.glob("ratings_pooled_seed*.parquet")))
    return outputs


def _agreement_outputs(cfg: AppConfig) -> list[Path]:
    players_list = sorted({int(n) for n in cfg.sim.n_players_list}) or [0]
    return [cfg.agreement_output_path(players) for players in players_list]
