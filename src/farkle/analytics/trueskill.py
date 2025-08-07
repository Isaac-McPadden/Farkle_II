from __future__ import annotations

import logging

from farkle import run_trueskill as _rt
from farkle.analysis_config import PipelineCfg

log = logging.getLogger(__name__)


def run(cfg: PipelineCfg) -> None:
    """Thin wrapper around the legacy script so the new pipeline stays small."""
    out = cfg.results_dir / "tiers.json"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        log.info("TrueSkill: results up-to-date - skipped")
        return

    log.info("TrueSkill: running in-process")
    _rt.main(
        [
            "--dataroot",
            str(cfg.results_dir),
            "--root",
            str(cfg.results_dir),
        ]
    )
