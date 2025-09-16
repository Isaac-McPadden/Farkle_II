from __future__ import annotations

from pathlib import Path

import yaml

from farkle.config import AppConfig, apply_dot_overrides, load_app_config


BASE_CFG = Path("configs/base.yaml")


def test_load_app_config_overlay(tmp_path: Path) -> None:
    overlay = tmp_path / "local.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "sim": {
                    "n_players": 3,
                    "collect_metrics": True,
                    "row_dir": str(tmp_path / "rows"),
                },
                "analysis": {"run_trueskill": False},
                "io": {"results_dir": str(tmp_path / "out")},
            }
        )
    )
    cfg = load_app_config(BASE_CFG, overlay)
    assert cfg.sim.n_players == 3
    assert cfg.sim.collect_metrics is True
    assert cfg.sim.row_dir == tmp_path / "rows"
    assert cfg.analysis.run_trueskill is False
    assert cfg.io.results_dir == tmp_path / "out"
    # Deep merge preserves unspecified keys
    assert cfg.sim.num_shuffles == 100


def test_apply_dot_overrides(tmp_path: Path) -> None:
    cfg = load_app_config(BASE_CFG)
    pairs = [
        "sim.n_players=7",
        "analysis.run_trueskill=false",
        f"io.results_dir={tmp_path / 'results'}",
        "analysis.trueskill_beta=3.5",
        "analysis.n_jobs=4",
        "analysis.log_level=DEBUG",
        "sim.collect_metrics=true",
        f"sim.row_dir={tmp_path / 'rows'}",
    ]
    apply_dot_overrides(cfg, pairs)
    assert cfg.sim.n_players == 7
    assert cfg.sim.collect_metrics is True
    assert cfg.sim.row_dir == tmp_path / "rows"
    assert cfg.analysis.run_trueskill is False
    assert cfg.io.results_dir == tmp_path / "results"
    assert cfg.analysis.trueskill_beta == 3.5
    assert cfg.analysis.n_jobs == 4
    assert cfg.analysis.log_level == "DEBUG"
