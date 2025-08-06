# farkle/analysis_config.py
"""
Config module for the analysis stage.

Typical usage::

    from farkle.analysis_config import PipelineCfg
    from farkle import ingest, curate, metrics, analytics

    cfg = PipelineCfg(root=Path(args.root))
    ingest.run(cfg)   # writes ``game_rows.raw.parquet``
    curate.run(cfg)   # adds manifest & renames to ``game_rows.parquet``
    metrics.run(cfg)
    analytics.run_all(cfg)     # inside it: respects cfg.run_trueskill flags
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class PipelineCfg:
    # 1. core paths
    results_dir: Path = Path("results_seed_0")
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
    run_hgb: bool = True
    hgb_max_iter: int = 500
    trueskill_beta: float = 25 / 6

    # 4. perf
    cores = os.cpu_count() or 1
    if cores is None:
        raise RuntimeError("Unable to determine CPU Count")
    n_jobs: int = max(cores - 1, 1)
    prefetch_shards: int = 2

    # 5. logging & provenance
    log_level: str = "INFO"
    log_file: Path | None = None           # e.g. Path("analysis/pipeline.log")
    manifest_name: str = "manifest.json"
    _git_sha: str | None = field(default=None, repr=False, init=False)

    def _load_git_sha(self) -> str:
        try:
            import subprocess
            return subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
                        ).strip()
        except Exception:
            return "unknown _load_git_sha() exception"

    @property
    def git_sha(self) -> str:
        if self._git_sha is None:
            self._git_sha = self._load_git_sha()
        return self._git_sha

    # Convenience helpers
    # -------------------
    @property
    def analysis_dir(self) -> Path:
        """Directory where analysis artifacts are written."""
        return self.results_dir / self.analysis_subdir

    @property
    def data_dir(self) -> Path:
        """Subdirectory beneath :pyattr:`analysis_dir` holding intermediate data."""
        return self.analysis_dir / "data"

    @property
    def curated_parquet(self) -> Path:
        return self.analysis_dir / "data" / self.curated_rows_name

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: str(o) if isinstance(o, Path) else o, indent=2)

    # Convenience: return kwargs ready for setup_logging()
    def logging_params(self) -> dict[str, object]:
        return {"level": self.log_level, "log_file": self.log_file}

    # ------------------------------------------------------------------
    @classmethod
    def parse_cli(
        cls, argv: Sequence[str] | None = None
    ) -> tuple["PipelineCfg", argparse.Namespace, list[str]]:
        """Parse command-line arguments into a :class:`PipelineCfg`.

        Returns a triple ``(cfg, namespace, remaining)`` where ``remaining``
        contains any extra arguments for higher-level parsers (e.g. pipeline
        subcommands).
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--results-dir",
            "--root",
            dest="results_dir",
            type=Path,
            default=Path("results_seed_0"),
            help="Directory containing raw results blocks",
        )
        parser.add_argument(
            "--analysis-subdir",
            default="analysis",
            help="Subdirectory for analysis outputs",
        )
        parser.add_argument("-v", "--verbose", action="store_true", help="Enable progress bars")
        ns, remaining = parser.parse_known_args(argv)
        cfg = cls(results_dir=ns.results_dir, analysis_subdir=ns.analysis_subdir)
        return cfg, ns, remaining
