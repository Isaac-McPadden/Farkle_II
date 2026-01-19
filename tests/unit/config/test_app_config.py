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

    assert cfg.analysis.outputs["metrics_name"] == "metrics.parquet"
    assert cfg.ingest.batch_rows == 200_000
    # ensure base values remain when not overridden
    assert cfg.ingest.n_jobs == 2
    assert cfg.analysis_dir == Path("results") / "custom"


def test_load_app_config_applies_analysis_controls(write_yaml) -> None:
    config = write_yaml(
        "analysis_fields.yaml",
        {
            "analysis": {
                "run_post_h2h_analysis": True,
                "run_frequentist": True,
                "run_agreement": True,
                "run_report": False,
                "head2head_target_hours": 4.5,
                "head2head_tolerance_pct": 2.5,
                "head2head_games_per_sec": 11.0,
                "tiering_seeds": [3, 7],
            }
        },
    )

    cfg = load_app_config(config)

    assert cfg.analysis.run_post_h2h_analysis is True
    assert cfg.analysis.run_frequentist is True
    assert cfg.analysis.run_agreement is True
    assert cfg.analysis.run_report is False
    assert cfg.analysis.head2head_target_hours == pytest.approx(4.5)
    assert cfg.analysis.head2head_tolerance_pct == pytest.approx(2.5)
    assert cfg.analysis.head2head_games_per_sec == pytest.approx(11.0)
    assert cfg.analysis.tiering_seeds == [3, 7]

    apply_dot_overrides(
        cfg,
        [
            "analysis.head2head_target_hours=1.25",
            "analysis.run_report=true",
            "analysis.run_agreement=false",
        ],
    )

    assert cfg.analysis.head2head_target_hours == pytest.approx(1.25)
    assert cfg.analysis.run_report is True
    assert cfg.analysis.run_agreement is False


def test_load_app_config_normalizes_legacy_keys(write_yaml) -> None:
    legacy = write_yaml(
        "legacy.yaml",
        {
            "io": {"analysis_dir": "analysis"},
            "sim": {"n_players": 5, "collect_metrics": True},
        },
    )

    cfg = load_app_config(legacy)

    assert cfg.io.analysis_subdir == "analysis"
    assert cfg.sim.n_players_list == [5]
    assert cfg.sim.expanded_metrics is True


def test_load_app_config_keeps_results_dir(write_yaml) -> None:
    config = write_yaml(
        "seeded.yaml",
        {
            "io": {"results_dir": "base"},
            "sim": {"seed": 7},
        },
    )

    cfg = load_app_config(config)

    assert cfg.results_dir == Path("base")
    assert cfg.analysis_dir == Path("base") / cfg.io.analysis_subdir


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
        ],
    )

    assert cfg.io.results_dir == Path("/tmp/output")
    assert cfg.sim.seed == 9
    assert cfg.trueskill.beta == pytest.approx(32.5)


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
