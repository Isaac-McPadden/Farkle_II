from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from farkle.analysis_config import PipelineCfg

log = logging.getLogger(__name__)
SCRIPT = Path(__file__).resolve().parents[1] / "run_rf.py"


def run(cfg: PipelineCfg) -> None:
    out = cfg.analysis_dir / "rf_feature_importances.parquet"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        log.info("Random-Forest: results up-to-date - skipped")
        return

    cmd = [sys.executable, str(SCRIPT), "--root", str(cfg.results_dir)]
    log.info("Random-Forest: calling %s", " ".join(cmd))
    subprocess.check_call(cmd)
