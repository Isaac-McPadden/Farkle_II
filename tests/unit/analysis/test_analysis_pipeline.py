"""Tests for the CLI-facing analysis pipeline entry point."""

from __future__ import annotations

import datetime as _dt
import json
import sys
import types
from pathlib import Path
from typing import Callable

import pytest
import yaml

pytestmark = pytest.mark.skipif(
    not hasattr(_dt, "UTC"),
    reason="analysis pipeline requires datetime.UTC (Python 3.11+)",
)

from farkle.analysis import pipeline
from farkle.config import AppConfig, IOConfig, compute_config_sha


def _patch_pipeline_stage_stubs(
    monkeypatch: pytest.MonkeyPatch,
    cfg: AppConfig,
    calls: list[str],
) -> None:
    def _record(name: str) -> Callable[[AppConfig], None]:
        def _runner(app_cfg: AppConfig, **_kwargs: object) -> None:
            calls.append(name)
            assert app_cfg.results_root == cfg.results_root

        return _runner

    monkeypatch.setattr("farkle.analysis.ingest.run", _record("ingest"), raising=True)
    monkeypatch.setattr("farkle.analysis.curate.run", _record("curate"), raising=True)
    monkeypatch.setattr("farkle.analysis.combine.run", _record("combine"), raising=True)
    monkeypatch.setattr("farkle.analysis.metrics.run", _record("metrics"), raising=True)
    monkeypatch.setattr("farkle.analysis.coverage_by_k.run", _record("coverage_by_k"), raising=True)
    monkeypatch.setattr("farkle.analysis.game_stats.run", _record("game_stats"), raising=True)
    monkeypatch.setattr(
        "farkle.analysis.seed_summaries.run", _record("seed_summaries"), raising=True
    )
    monkeypatch.setattr("farkle.analysis.seed_symmetry.run", _record("seed_symmetry"), raising=True)
    monkeypatch.setattr("farkle.analysis.variance.run", _record("variance"), raising=True)
    monkeypatch.setattr("farkle.analysis.meta.run", _record("meta"), raising=True)
    monkeypatch.setattr(
        "farkle.analysis.h2h_tier_trends.run", _record("h2h_tier_trends"), raising=True
    )
    monkeypatch.setattr(
        "farkle.analysis.rng_diagnostics.run", _record("rng_diagnostics"), raising=True
    )

    def _fake_optional_import(module: str, *, stage_log=None):  # noqa: ANN001
        mapping = {
            "farkle.analysis.trueskill": types.SimpleNamespace(run=_record("trueskill")),
            "farkle.analysis.head2head": types.SimpleNamespace(run=_record("head2head")),
            "farkle.analysis.h2h_analysis": types.SimpleNamespace(run_post_h2h=_record("post_h2h")),
            "farkle.analysis.hgb_feat": types.SimpleNamespace(run=_record("hgb")),
            "farkle.analysis.tiering_report": types.SimpleNamespace(run=_record("tiering")),
            "farkle.analysis.agreement": types.SimpleNamespace(run=_record("agreement")),
            "farkle.analysis.interseed_analysis": types.SimpleNamespace(run=_record("interseed")),
        }
        return mapping.get(module)

    monkeypatch.setattr("farkle.analysis._optional_import", _fake_optional_import, raising=True)


def _make_config(tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, AppConfig]:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_results_dir))
    cfg_path = tmp_results_dir / "configs/fast_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("io:\n  results_dir_prefix: dummy\n")

    def _load_app_config(path: Path) -> AppConfig:
        assert path == cfg_path
        return cfg

    monkeypatch.setattr("farkle.analysis.pipeline.load_app_config", _load_app_config, raising=True)
    monkeypatch.setattr("farkle.config.load_app_config", _load_app_config, raising=False)
    return cfg_path, cfg


def _inject_dummy_networkx(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = types.ModuleType("networkx")
    monkeypatch.setattr(dummy, "DiGraph", type("DiGraph", (), {}), raising=False)
    monkeypatch.setitem(sys.modules, "networkx", dummy)


def test_pipeline_creates_stage_dirs_in_new_order(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)

    stage_folders: list[str] = []
    original = cfg.stage_dir

    def _record(stage: str):
        root = original(stage)
        stage_folders.append(root.name)
        return root

    monkeypatch.setattr(cfg, "stage_dir", _record)
    monkeypatch.setattr("farkle.analysis.ingest.run", lambda _: None, raising=True)

    rc = pipeline.main(["--config", str(cfg_path), "ingest"])

    assert rc == 0
    expected = [placement.folder_name for placement in cfg.stage_layout.placements]
    assert stage_folders[: len(expected)] == expected


def test_pipeline_writes_resolved_config_and_manifest(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)

    called = False

    def _fake_ingest(app_cfg):  # noqa: ANN001
        nonlocal called
        called = True
        assert app_cfg.results_root == cfg.results_root

    monkeypatch.setattr("farkle.analysis.ingest.run", _fake_ingest, raising=True)

    rc = pipeline.main(["--config", str(cfg_path), "ingest"])
    assert rc == 0
    assert called

    analysis_dir = cfg.analysis_dir
    resolved = analysis_dir / "config.resolved.yaml"
    manifest = analysis_dir / cfg.manifest_name
    assert resolved.exists()
    assert manifest.exists()

    resolved_yaml = resolved.read_text()
    expected_sha = compute_config_sha(cfg)
    resolved_dict = yaml.safe_load(resolved_yaml)
    stage_layout = resolved_dict.get("stage_layout") if isinstance(resolved_dict, dict) else None
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert records
    run_start = next(rec for rec in records if rec.get("event") == "run_start")
    assert run_start["config_sha"] == expected_sha
    assert run_start["resolved_config"] == str(resolved)
    assert stage_layout
    assert stage_layout == cfg.stage_layout.to_resolved_layout()
    assert records[-1]["event"] == "run_end"


@pytest.mark.parametrize(
    "command, target",
    [
        ("ingest", "farkle.analysis.ingest.run"),
        ("curate", "farkle.analysis.curate.run"),
        ("combine", "farkle.analysis.combine.run"),
        ("metrics", "farkle.analysis.metrics.run"),
    ],
)
def test_pipeline_individual_commands_invoke_target(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch, command: str, target: str
) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)

    called: list[str] = []

    def _record(app_cfg):  # noqa: ANN001
        called.append(command)
        assert app_cfg.results_root == cfg.results_root

    monkeypatch.setattr(target, _record, raising=True)
    if command == "metrics":
        monkeypatch.setattr(
            "farkle.analysis.game_stats.run",
            lambda app_cfg: called.append("game_stats"),
            raising=True,
        )
    rc = pipeline.main(["--config", str(cfg_path), command])
    assert rc == 0
    expected = [command]
    if command == "metrics" and cfg.stage_layout.folder_for("game_stats"):
        expected.append("game_stats")
    assert called == expected


def test_pipeline_analytics_runs_layout_order(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)

    calls: list[str] = []

    _patch_pipeline_stage_stubs(monkeypatch, cfg, calls)

    rc = pipeline.main(["--config", str(cfg_path), "analytics"])
    assert rc == 0
    expected_analytics_order = [
        placement.definition.key
        for placement in cfg.stage_layout.placements
        if placement.definition.group == "analytics"
    ]
    assert calls == expected_analytics_order
    assert calls.index("seed_symmetry") == calls.index("head2head") + 1


def test_pipeline_all_runs_all_steps(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)

    calls: list[str] = []

    _patch_pipeline_stage_stubs(monkeypatch, cfg, calls)

    rc = pipeline.main(["--config", str(cfg_path), "all"])
    assert rc == 0
    assert calls == cfg.stage_layout.keys()


@pytest.mark.parametrize(
    "command, target",
    [
        ("ingest", "farkle.analysis.ingest.run"),
        ("combine", "farkle.analysis.combine.run"),
        ("analytics", "farkle.analysis.game_stats.run"),
    ],
)
def test_pipeline_step_failure_propagates(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch, command: str, target: str
) -> None:
    cfg_path, _ = _make_config(tmp_results_dir, monkeypatch)

    def _boom(app_cfg):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr(target, _boom, raising=True)
    if command == "analytics":
        _inject_dummy_networkx(monkeypatch)
        monkeypatch.setattr("farkle.analysis.seed_summaries.run", lambda cfg: None, raising=True)
        monkeypatch.setattr("farkle.analysis.variance.run", lambda cfg: None, raising=True)
        monkeypatch.setattr("farkle.analysis.meta.run", lambda cfg: None, raising=True)
        monkeypatch.setattr("farkle.analysis.trueskill.run", lambda cfg: None, raising=True)
        monkeypatch.setattr("farkle.analysis.head2head.run", lambda cfg: None, raising=True)
        monkeypatch.setattr("farkle.analysis.hgb_feat.run", lambda cfg: None, raising=True)

    with pytest.raises(RuntimeError):
        pipeline.main(["--config", str(cfg_path), command])


def test_pipeline_game_stats_flag(monkeypatch: pytest.MonkeyPatch, tmp_results_dir: Path) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)

    calls: list[str] = []

    def _fake_metrics(app_cfg):  # noqa: ANN001
        calls.append("metrics")
        assert app_cfg.analysis.game_stats_margin_thresholds == (700, 900)

    def _fake_game_stats(app_cfg):  # noqa: ANN001
        calls.append("game_stats")
        assert app_cfg.analysis.rare_event_target_score == 12345

    monkeypatch.setattr("farkle.analysis.metrics.run", _fake_metrics, raising=True)
    monkeypatch.setattr("farkle.analysis.game_stats.run", _fake_game_stats, raising=True)

    rc = pipeline.main(
        [
            "--config",
            str(cfg_path),
            "--margin-thresholds",
            "700",
            "900",
            "--rare-event-target",
            "12345",
            "metrics",
        ]
    )

    assert rc == 0
    assert calls == ["metrics", "game_stats"]


def test_pipeline_game_stats_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_results_dir: Path
) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)
    cfg.analysis.run_game_stats = True

    calls: list[str] = []

    def _fake_metrics(app_cfg):  # noqa: ANN001
        calls.append("metrics")
        assert app_cfg.analysis.run_game_stats is True

    def _fake_game_stats(app_cfg):  # noqa: ANN001
        calls.append("game_stats")

    monkeypatch.setattr("farkle.analysis.metrics.run", _fake_metrics, raising=True)
    monkeypatch.setattr("farkle.analysis.game_stats.run", _fake_game_stats, raising=True)

    rc = pipeline.main(["--config", str(cfg_path), "--no-game-stats", "metrics"])

    assert rc == 0
    assert calls == ["metrics", "game_stats"]


def test_pipeline_rng_diagnostics_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_results_dir: Path
) -> None:
    cfg_path, _ = _make_config(tmp_results_dir, monkeypatch)

    calls: list[str] = []

    def _fake_combine(app_cfg):  # noqa: ANN001
        calls.append("combine")

    monkeypatch.setattr("farkle.analysis.combine.run", _fake_combine, raising=True)

    rc = pipeline.main(
        [
            "--config",
            str(cfg_path),
            "--rng-diagnostics",
            "--rng-lags",
            "3",
            "--rng-lags",
            "1",
            "combine",
        ]
    )

    assert rc == 0
    assert calls == ["combine"]
