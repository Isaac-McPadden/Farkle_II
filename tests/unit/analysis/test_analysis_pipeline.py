"""Tests for the CLI-facing analysis pipeline entry point."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(_dt, "UTC"),
    reason="analysis pipeline requires datetime.UTC (Python 3.11+)",
)

from farkle.analysis import pipeline
from farkle.config import AppConfig, IOConfig


def _make_config(tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, AppConfig]:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_results_dir, append_seed=False))
    cfg_path = tmp_results_dir / "configs/fast_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("io:\n  results_dir: dummy\n")

    def _load_app_config(path: Path) -> AppConfig:
        assert path == cfg_path
        return cfg

    monkeypatch.setattr("farkle.analysis.pipeline.load_app_config", _load_app_config, raising=True)
    monkeypatch.setattr("farkle.config.load_app_config", _load_app_config, raising=False)
    return cfg_path, cfg


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
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert records
    run_start = next(rec for rec in records if rec.get("event") == "run_start")
    assert run_start["config_sha"] == expected_sha
    assert run_start["resolved_config"] == str(resolved)
    assert records[-1]["event"] == "run_end"


@pytest.mark.parametrize(
    "command, target",
    [
        ("ingest", "farkle.analysis.ingest.run"),
        ("curate", "farkle.analysis.curate.run"),
        ("combine", "farkle.analysis.combine.run"),
        ("metrics", "farkle.analysis.metrics.run"),
        ("analytics", "farkle.analysis.run_all"),
    ],
)
def test_pipeline_individual_commands_invoke_target(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch, command: str, target: str
) -> None:
    cfg_path, _ = _make_config(tmp_results_dir, monkeypatch)

    called: list[str] = []

    def _record(app_cfg):  # noqa: ANN001
        called.append(command)
        assert app_cfg.results_dir == tmp_results_dir

    monkeypatch.setattr(target, _record, raising=True)
    rc = pipeline.main(["--config", str(cfg_path), command])
    assert rc == 0
    assert called == [command]


def test_pipeline_all_runs_all_steps(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path, _ = _make_config(tmp_results_dir, monkeypatch)

    calls: list[str] = []

    def _make_stub(name: str) -> Callable[[object], None]:
        def _stub(app_cfg):  # noqa: ANN001
            calls.append(name)
            assert app_cfg.results_dir == tmp_results_dir

        return _stub

    monkeypatch.setattr("farkle.analysis.ingest.run", _make_stub("ingest"), raising=True)
    monkeypatch.setattr("farkle.analysis.curate.run", _make_stub("curate"), raising=True)
    monkeypatch.setattr("farkle.analysis.combine.run", _make_stub("combine"), raising=True)
    monkeypatch.setattr("farkle.analysis.metrics.run", _make_stub("metrics"), raising=True)
    monkeypatch.setattr("farkle.analysis.run_all", _make_stub("analytics"), raising=True)

    rc = pipeline.main(["--config", str(cfg_path), "all"])
    assert rc == 0
    assert calls == ["ingest", "curate", "combine", "metrics", "analytics"]


@pytest.mark.parametrize(
    "command, target",
    [
        ("ingest", "farkle.analysis.ingest.run"),
        ("combine", "farkle.analysis.combine.run"),
        ("analytics", "farkle.analysis.run_all"),
    ],
)
def test_pipeline_step_failure_propagates(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch, command: str, target: str
) -> None:
    cfg_path, _ = _make_config(tmp_results_dir, monkeypatch)

    def _boom(app_cfg):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr(target, _boom, raising=True)

    with pytest.raises(RuntimeError):
        pipeline.main(["--config", str(cfg_path), command])
