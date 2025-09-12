from __future__ import annotations

import logging

from farkle.analysis import run_trueskill
from farkle.analysis.analysis_config import PipelineCfg

log = logging.getLogger(__name__)


def run(cfg: PipelineCfg) -> None:
    """Thin wrapper around the legacy script so the new pipeline stays small."""
    out = cfg.analysis_dir / "tiers.json"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        log.info("TrueSkill: results up-to-date - skipped")
        return

    log.info("TrueSkill: running in-process")
    run_trueskill.main(
        [
            "--dataroot",
            str(cfg.analysis_dir),
            "--root",
            str(cfg.analysis_dir),
        ]
    )
