"""Tests for the CLI-facing analysis pipeline entry point."""

from __future__ import annotations

import datetime as _dt
import hashlib
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
from farkle.config import AppConfig, IOConfig


def _make_config(tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, AppConfig]:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_results_dir))
    cfg_path = tmp_results_dir / "configs/fast_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("io:\n  results_dir: dummy\n")

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
    assert stage_folders == [placement.folder_name for placement in cfg.stage_layout.placements]


def test_pipeline_writes_resolved_config_and_manifest(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)

    called = False

    def _fake_ingest(app_cfg):  # noqa: ANN001
        nonlocal called
        called = True
        assert app_cfg.results_dir == tmp_results_dir

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
    expected_sha = hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()
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
        assert app_cfg.results_dir == tmp_results_dir

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

    def _stub(name: str) -> Callable[[object], None]:
        def _inner(app_cfg, **_kwargs):  # noqa: ANN001
            calls.append(name)
            assert app_cfg.results_dir == tmp_results_dir

        return _inner

    _inject_dummy_networkx(monkeypatch)
    monkeypatch.setattr("farkle.analysis.game_stats.run", _stub("game_stats"), raising=True)
    monkeypatch.setattr("farkle.analysis.seed_summaries.run", _stub("seed_summaries"), raising=True)
    monkeypatch.setattr("farkle.analysis.variance.run", _stub("variance"), raising=True)
    monkeypatch.setattr("farkle.analysis.meta.run", _stub("meta"), raising=True)
    monkeypatch.setattr("farkle.analysis.trueskill.run", _stub("trueskill"), raising=True)
    monkeypatch.setattr("farkle.analysis.head2head.run", _stub("head2head"), raising=True)
    monkeypatch.setattr("farkle.analysis.hgb_feat.run", _stub("hgb"), raising=True)

    rc = pipeline.main(["--config", str(cfg_path), "analytics"])
    assert rc == 0
    analytics_keys = [
        placement.definition.key
        for placement in cfg.stage_layout.placements
        if placement.definition.group == "analytics"
    ]
    assert calls == analytics_keys


def test_pipeline_all_runs_all_steps(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)

    calls: list[str] = []

    def _make_stub(name: str) -> Callable[[object], None]:
        def _stub(app_cfg, **_kwargs):  # noqa: ANN001
            calls.append(name)
            assert app_cfg.results_dir == tmp_results_dir

        return _stub

    _inject_dummy_networkx(monkeypatch)
    monkeypatch.setattr("farkle.analysis.ingest.run", _make_stub("ingest"), raising=True)
    monkeypatch.setattr("farkle.analysis.curate.run", _make_stub("curate"), raising=True)
    monkeypatch.setattr("farkle.analysis.combine.run", _make_stub("combine"), raising=True)
    monkeypatch.setattr("farkle.analysis.metrics.run", _make_stub("metrics"), raising=True)
    monkeypatch.setattr("farkle.analysis.game_stats.run", _make_stub("game_stats"), raising=True)
    monkeypatch.setattr("farkle.analysis.seed_summaries.run", _make_stub("seed_summaries"), raising=True)
    monkeypatch.setattr("farkle.analysis.variance.run", _make_stub("variance"), raising=True)
    monkeypatch.setattr("farkle.analysis.meta.run", _make_stub("meta"), raising=True)
    monkeypatch.setattr("farkle.analysis.trueskill.run", _make_stub("trueskill"), raising=True)
    monkeypatch.setattr("farkle.analysis.head2head.run", _make_stub("head2head"), raising=True)
    monkeypatch.setattr("farkle.analysis.hgb_feat.run", _make_stub("hgb"), raising=True)

    rc = pipeline.main(["--config", str(cfg_path), "all"])
    assert rc == 0
    assert calls == [placement.definition.key for placement in cfg.stage_layout.placements]


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


def test_pipeline_game_stats_opt_out(monkeypatch: pytest.MonkeyPatch, tmp_results_dir: Path) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)
    cfg.analysis.run_game_stats = True

    calls: list[str] = []

    def _fake_metrics(app_cfg):  # noqa: ANN001
        calls.append("metrics")
        assert app_cfg.analysis.run_game_stats is False

    def _fake_game_stats(app_cfg):  # noqa: ANN001
        calls.append("game_stats")

    monkeypatch.setattr("farkle.analysis.metrics.run", _fake_metrics, raising=True)
    monkeypatch.setattr("farkle.analysis.game_stats.run", _fake_game_stats, raising=True)

    rc = pipeline.main([
        "--config",
        str(cfg_path),
        "--no-game-stats",
        "metrics",
    ])

    assert rc == 0
    assert calls == ["metrics"]


def test_pipeline_rng_diagnostics_flag(monkeypatch: pytest.MonkeyPatch, tmp_results_dir: Path) -> None:
    cfg_path, _ = _make_config(tmp_results_dir, monkeypatch)

    calls: list[tuple[str, tuple[int, ...] | None]] = []

    def _fake_combine(app_cfg):  # noqa: ANN001
        calls.append(("combine", None))

    def _fake_rng(
        app_cfg, *, lags: tuple[int, ...] | None = None, force: bool = False
    ):  # noqa: ANN001
        assert lags is not None
        calls.append(("rng_diagnostics", lags))
        assert tuple(sorted(lags)) == (1, 3)

    monkeypatch.setattr("farkle.analysis.combine.run", _fake_combine, raising=True)
    monkeypatch.setattr("farkle.analysis.rng_diagnostics.run", _fake_rng, raising=True)

    rc = pipeline.main([
        "--config",
        str(cfg_path),
        "--rng-diagnostics",
        "--rng-lags",
        "3",
        "--rng-lags",
        "1",
        "combine",
    ])

    assert rc == 0
    assert calls == [("combine", None), ("rng_diagnostics", (1, 3))]
