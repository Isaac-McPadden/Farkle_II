"""Tests for the CLI-facing analysis pipeline entry point."""

from __future__ import annotations

import datetime as _dt
import json
import types
from pathlib import Path
from typing import Callable

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(_dt, "UTC"),
    reason="analysis pipeline requires datetime.UTC (Python 3.11+)",
)

from farkle.analysis import pipeline
from farkle.analysis.analysis_config import IO, Config, Experiment


def _make_config(tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Config]:
    cfg = Config(experiment=Experiment(name="test", seed=0), io=IO(results_dir=tmp_results_dir))
    cfg_path = tmp_results_dir / "configs/fast_config.yaml"
    cfg_path.write_text("experiment:\n  name: test\n")
    original_model_dump = cfg.model_dump

    def _model_dump(self, **kwargs):  # noqa: ANN001
        data = original_model_dump(**kwargs)

        def _coerce(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: _coerce(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(_coerce(v) for v in value)
            return value

        return _coerce(data)

    object.__setattr__(cfg, "model_dump", types.MethodType(_model_dump, cfg))
    monkeypatch.setattr("farkle.analysis.pipeline.load_config", lambda path: (cfg, "deadbeef"))
    return cfg_path, cfg


def test_pipeline_writes_resolved_config_and_manifest(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)

    called = False

    def _fake_ingest(app_cfg):  # noqa: ANN001
        nonlocal called
        called = True
        assert app_cfg.analysis.results_dir == tmp_results_dir

    monkeypatch.setattr("farkle.analysis.ingest.run", _fake_ingest, raising=True)

    rc = pipeline.main(["--config", str(cfg_path), "ingest"])
    assert rc == 0
    assert called

    analysis_dir = tmp_results_dir / cfg.io.analysis_subdir
    resolved = analysis_dir / "config.resolved.yaml"
    manifest = analysis_dir / cfg.to_pipeline_cfg().manifest_name
    assert resolved.exists()
    assert manifest.exists()

    manifest_data = json.loads(manifest.read_text())
    assert manifest_data == {"config_sha": "deadbeef"}


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
        assert app_cfg.analysis.results_dir == tmp_results_dir

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
            assert app_cfg.analysis.results_dir == tmp_results_dir

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
