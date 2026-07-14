# src/farkle/analysis/trueskill.py
"""Run canonical per-root/per-k TrueSkill screening diagnostics."""
from __future__ import annotations

import logging

from farkle.analysis import run_trueskill, stage_logger
from farkle.config import AppConfig

LOGGER = logging.getLogger(__name__)


def run(cfg: AppConfig) -> None:
    """Build per-root/per-k ratings and their screening-only contribution."""
    stage_log = stage_logger("trueskill", logger=LOGGER)
    stage_log.start()

    curated_parquet = cfg.curated_parquet
    roots = tuple(cfg.sim.seed_list or [cfg.sim.seed])
    if len(roots) == 1 and not curated_parquet.exists():
        raise FileNotFoundError(
            f"TrueSkill requires canonical concatenated rows: {curated_parquet}"
        )

    out = cfg.trueskill_candidate_contribution_path()
    target = out
    if len(roots) == 1 and out.exists() and out.stat().st_mtime >= curated_parquet.stat().st_mtime:
        LOGGER.info(
            "TrueSkill results up-to-date",
            extra={"stage": "trueskill", "path": str(out)},
        )
        return

    LOGGER.info(
        "TrueSkill analysis running",
        extra={"stage": "trueskill", "analysis_dir": str(cfg.analysis_dir)},
    )
    run_trueskill.run_trueskill_root(cfg)
    LOGGER.info(
        "TrueSkill analysis complete",
        extra={"stage": "trueskill", "path": str(target)},
    )
