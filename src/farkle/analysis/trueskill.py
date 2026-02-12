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

    curated_parquet = cfg.curated_parquet
    if not curated_parquet.exists():
        candidates = [str(path) for path in cfg.curated_parquet_candidates()]
        payload = {
            "path": str(curated_parquet),
            "candidate_paths": candidates,
        }
        interseed_root = cfg.interseed_input_dir
        if interseed_root is not None:
            payload["interseed_input_root"] = str(interseed_root)
        stage_log.missing_input("curated parquet missing", **payload)
        return

    out = cfg.preferred_tiers_path()
    if cfg.interseed_input_dir is not None:
        # Avoid short-circuiting interseed runs based on upstream seed artifacts.
        out = cfg.trueskill_stage_dir / "tiers.json"
    target = cfg.trueskill_stage_dir / "tiers.json"
    if out.exists() and out.stat().st_mtime >= curated_parquet.stat().st_mtime:
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
