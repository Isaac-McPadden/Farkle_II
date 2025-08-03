# farkle/analysis_config.py
"""
Config module for the analysis stage
# pipeline.py
from farkle.config import PipelineCfg
from farkle import ingest, metrics, analytics

def main():
    cfg = PipelineCfg(root=Path(args.root))
    ingest.run(cfg)
    metrics.run(cfg)
    analytics.run_all(cfg)     # inside it: respects cfg.run_trueskill flags
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineCfg:
    # 1. core paths
    root: Path = Path("data")
    results_glob: str = "*_players"
    analysis_subdir: str = "analysis"
    curated_rows_name: str = "game_rows.parquet"
    metrics_name: str = "metrics.parquet"

    # 2. ingest
    ingest_cols: tuple[str, ...] = field(
        default_factory=lambda: (
            "winner",
            "n_rounds",
            "winning_score",
            *[f"P{i}_strategy" for i in range(1, 13)],
        )
    )
    parquet_codec: str = "zstd"
    row_group_size: int = 64_000
    max_shard_mb: int = 512

    # 3. analytics toggles / params
    run_trueskill: bool = True
    run_head2head: bool = True
    run_rf: bool = True
    rf_n_estimators: int = 500
    trueskill_beta: float = 25 / 6

    # 4. perf
    cores = os.cpu_count()
    assert cores is not None
    n_jobs: int = max(cores - 1, 1)
    prefetch_shards: int = 2

    # 5. logging & provenance
    log_level: str = "INFO"
    log_file: Path | None = None           # e.g. Path("analysis/pipeline.log")
    manifest_name: str = "manifest.json"
    git_sha: str | None = None   # filled on __post_init__

    # --------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.git_sha is None:
            try:
                import shlex
                import subprocess

                self.git_sha = (
                    subprocess.check_output(shlex.split("git rev-parse HEAD"))
                    .decode()
                    .strip()
                )
            except Exception:
                self.git_sha = "unknown"

    # Convenience helpers
    # -------------------
    @property
    def analysis_dir(self) -> Path:
        """Directory where analysis artifacts are written."""
        return self.root / self.analysis_subdir

    @property
    def data_dir(self) -> Path:
        """Subdirectory beneath :pyattr:`analysis_dir` holding intermediate data."""
        return self.analysis_dir / "data"

    @property
    def curated_parquet(self) -> Path:
        return self.analysis_dir / "data" / self.curated_rows_name

    def to_json(self) -> str:
        return json.dumps(
            self,
            default=lambda o: str(o) if isinstance(o, Path) else o,
            indent=2,
        )

    # Convenience: return kwargs ready for setup_logging()
    def logging_params(self) -> dict[str, object]:
        return {"level": self.log_level, "log_file": self.log_file}

    # CLI -----------------------------------------------------------------
    @classmethod
    def parse_cli(cls, argv: list[str] | None = None) -> "PipelineCfg":
        """Parse command line options into a :class:`PipelineCfg` instance."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", type=Path, default=Path("data"))
        parser.add_argument(
            "--run-trueskill",
            action=argparse.BooleanOptionalAction,
            default=True,
            dest="run_trueskill",
        )
        parser.add_argument(
            "--run-head2head",
            action=argparse.BooleanOptionalAction,
            default=True,
            dest="run_head2head",
        )
        parser.add_argument(
            "--run-rf",
            action=argparse.BooleanOptionalAction,
            default=True,
            dest="run_rf",
        )
        args = parser.parse_args(argv)
        return cls(**vars(args))
