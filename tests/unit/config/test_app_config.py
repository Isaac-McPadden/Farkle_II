"""Tests for the ``farkle.config`` helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from farkle.config import AppConfig, apply_dot_overrides, load_app_config


@pytest.fixture
def write_yaml(tmp_path: Path):
    def _write(name: str, data: object) -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data))
        return path

    return _write


def test_load_app_config_merges_overlays(write_yaml) -> None:
    base = write_yaml(
        "base.yaml",
        {
            "io": {
                "results_dir": "results",
                "append_seed": False,
            },
            "analysis": {"log_level": "INFO"},
            "ingest": {"n_jobs": 2},
        },
    )
    overlay = write_yaml(
        "overlay.yaml",
        {
            "io.analysis_subdir": "custom",
            "analysis": {"outputs": {"metrics_name": "metrics.parquet"}},
            "ingest": {"batch_rows": 200_000},
        },
    )

    cfg = load_app_config(base, overlay)

    assert cfg.io.append_seed is False
    assert cfg.analysis.outputs["metrics_name"] == "metrics.parquet"
    assert cfg.ingest.batch_rows == 200_000
    # ensure base values remain when not overridden
    assert cfg.ingest.n_jobs == 2
    assert cfg.analysis_dir == Path("results") / "custom"


def test_load_app_config_normalizes_legacy_keys(write_yaml) -> None:
    legacy = write_yaml(
        "legacy.yaml",
        {
            "io": {"analysis_dir": "analysis", "append_seed": False},
            "sim": {"n_players": 5, "collect_metrics": True},
        },
    )

    cfg = load_app_config(legacy)

    assert cfg.io.analysis_subdir == "analysis"
    assert cfg.sim.n_players_list == [5]
    assert cfg.sim.expanded_metrics is True


def test_load_app_config_appends_seed(write_yaml) -> None:
    config = write_yaml(
        "seeded.yaml",
        {
            "io": {"results_dir": "base"},
            "sim": {"seed": 7},
        },
    )

    cfg = load_app_config(config)

    assert cfg.results_dir == Path("base_seed_7")
    assert cfg.analysis_dir == Path("base_seed_7") / cfg.io.analysis_subdir


def test_load_app_config_rejects_non_mapping(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump(["not", "a", "mapping"]))

    with pytest.raises(TypeError):
        load_app_config(config)


def test_apply_dot_overrides_coerces_values() -> None:
    cfg = AppConfig()

    apply_dot_overrides(
        cfg,
        [
            "io.results_dir=/tmp/output",
            "sim.seed=9",
            "trueskill.beta=32.5",
            "io.append_seed=false",
        ],
    )

    assert cfg.io.results_dir == Path("/tmp/output")
    assert cfg.sim.seed == 9
    assert cfg.trueskill.beta == pytest.approx(32.5)
    assert cfg.io.append_seed is False


@pytest.mark.parametrize(
    "override",
    ["missing_equals", "simseed=9"],
)
def test_apply_dot_overrides_rejects_bad_pairs(override: str) -> None:
    cfg = AppConfig()

    with pytest.raises(ValueError):
        apply_dot_overrides(cfg, [override])


def test_apply_dot_overrides_unknown_option() -> None:
    cfg = AppConfig()

    with pytest.raises(AttributeError):
        apply_dot_overrides(cfg, ["sim.unknown=1"])
