# src/farkle/analysis/trueskill.py
from __future__ import annotations

import logging

from farkle.analysis import run_trueskill
from farkle.config import AppConfig

LOGGER = logging.getLogger(__name__)


def run(cfg: AppConfig) -> None:
    """Thin wrapper around the legacy script so the new pipeline stays small."""
    out = cfg.analysis_dir / "tiers.json"
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
    run_trueskill.run_trueskill(root=cfg.analysis_dir, dataroot=cfg.results_dir)
    LOGGER.info(
        "TrueSkill analysis complete",
        extra={"stage": "trueskill", "path": str(out)},
    )
