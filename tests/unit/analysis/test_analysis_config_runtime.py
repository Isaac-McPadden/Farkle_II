"""Runtime behavior tests for the unified :mod:`farkle.config` loader."""

from pathlib import Path

import pytest
import yaml

from farkle.config import AppConfig, IOConfig, apply_dot_overrides, load_app_config
from farkle.utils.schema_helpers import rows_for_ram


def test_load_app_config_merges_overlays(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    overlay = tmp_path / "overlay.yaml"

    base.write_text(
        yaml.safe_dump(
            {
                "io": {
                    "results_dir": str(tmp_path / "results"),
                    "append_seed": False,
                },
                "sim": {"seed": 3},
            }
        )
    )
    overlay.write_text(
        yaml.safe_dump(
            {
                "io.analysis_subdir": "custom",
                "analysis.outputs.metrics_name": "metrics.parquet",
            }
        )
    )

    cfg = load_app_config(base, overlay)

    assert cfg.results_dir == tmp_path / "results"
    assert cfg.analysis_dir == tmp_path / "results" / "custom"
    assert cfg.analysis.outputs["metrics_name"] == "metrics.parquet"


def test_load_app_config_normalizes_legacy_keys(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        yaml.safe_dump(
            {
                "io": {"analysis_dir": "analysis", "append_seed": False},
                "sim": {"n_players": 5, "collect_metrics": True},
            }
        )
    )

    cfg = load_app_config(legacy)

    assert cfg.io.analysis_subdir == "analysis"
    assert cfg.sim.n_players_list == [5]
    assert cfg.sim.expanded_metrics is True


def test_apply_dot_overrides_updates_types() -> None:
    cfg = AppConfig(io=IOConfig(results_dir=Path("results"), append_seed=False))

    apply_dot_overrides(
        cfg,
        [
            "io.analysis_subdir=staging",
            "sim.seed=9",
            "ingest.n_jobs=8",
            "io.append_seed=false",
        ],
    )

    assert cfg.io.analysis_subdir == "staging"
    assert cfg.sim.seed == 9
    assert cfg.ingest.n_jobs == 8
    assert cfg.io.append_seed is False


def test_rows_for_ram_returns_reasonable_bounds() -> None:
    assert rows_for_ram(1, 1_000) == 10_000
    expected = int((50 * 1024**2) / (20 * 4 * 1.5))
    assert rows_for_ram(50, 20) == expected
