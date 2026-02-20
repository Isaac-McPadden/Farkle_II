from __future__ import annotations

from pathlib import Path
from typing import Any

from farkle.config import AnalysisConfig, AppConfig, IOConfig, SimConfig

DEFAULT_TEST_RESULTS_DIR = Path("test-results")


def make_test_app_config(
    *,
    results_dir_prefix: Path = DEFAULT_TEST_RESULTS_DIR,
    io: IOConfig | None = None,
    sim: SimConfig | None = None,
    analysis: AnalysisConfig | None = None,
    **overrides: Any,
) -> AppConfig:
    """Build a minimal ``AppConfig`` with deterministic test defaults."""

    io_cfg = io or IOConfig(results_dir_prefix=results_dir_prefix)
    sim_cfg = sim or SimConfig(n_players_list=[2, 3], seed=7)
    analysis_cfg = analysis or AnalysisConfig()
    return AppConfig(io=io_cfg, sim=sim_cfg, analysis=analysis_cfg, **overrides)
