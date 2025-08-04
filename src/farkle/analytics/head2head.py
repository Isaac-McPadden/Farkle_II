from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from farkle.analysis_config import PipelineCfg

log = logging.getLogger(__name__)
SCRIPT = Path(__file__).resolve().parents[1] / "run_bonferroni_head2head.py"


def run(cfg: PipelineCfg) -> None:
    out = cfg.analysis_dir / "bonferroni.parquet"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        log.info("Head-to-Head: results up-to-date - skipped")
        return

    cmd = [sys.executable, str(SCRIPT), "--root", str(cfg.results_dir)]
    log.info("Head-to-Head: calling %s", " ".join(cmd))
    subprocess.check_call(cmd)
