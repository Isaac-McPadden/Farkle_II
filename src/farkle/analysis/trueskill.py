# src/farkle/analysis/trueskill.py
"""Wrapper to trigger TrueSkill tier generation within the analysis pipeline."""
from __future__ import annotations

import logging

from farkle.analysis import run_trueskill, stage_logger
from farkle.config import AppConfig

LOGGER = logging.getLogger(__name__)


def run(cfg: AppConfig) -> None:
    """Thin wrapper around the legacy script so the new pipeline stays small."""
    stage_log = stage_logger("trueskill", logger=LOGGER)
    stage_log.start()

    if not cfg.curated_parquet.exists():
        stage_log.missing_input("curated parquet missing", path=str(cfg.curated_parquet))
        return

    out = cfg.preferred_tiers_path()
    target = cfg.trueskill_stage_dir / "tiers.json"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        LOGGER.info(
            "TrueSkill results up-to-date",
            extra={"stage": "trueskill", "path": str(out)},
        )
        return

    LOGGER.info(
        "TrueSkill analysis running",
        extra={"stage": "trueskill", "analysis_dir": str(cfg.analysis_dir)},
    )
    run_trueskill.run_trueskill_all_seeds(cfg)
    LOGGER.info(
        "TrueSkill analysis complete",
        extra={"stage": "trueskill", "path": str(target)},
    )
