from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

pytest.importorskip("pydantic")
pytest.importorskip("pyarrow")

import farkle.cli.main as cli_main


@pytest.fixture(autouse=True)
def _no_setup_logging(monkeypatch):
    monkeypatch.setattr(cli_main, "setup_info_logging", lambda: None)


@pytest.fixture
def preserve_root_logger():
    root = logging.getLogger()
    level = root.level
    handlers = list(root.handlers)
    yield
    root.setLevel(level)
    root.handlers[:] = handlers


def test_main_dispatches_watch(monkeypatch, preserve_root_logger):
    captured: dict[str, object] = {}

    def fake_watch_game(*, seed):
        captured["seed"] = seed

    monkeypatch.setattr(cli_main, "watch_game", fake_watch_game)

    cli_main.main(["watch", "--seed", "123"])

    assert captured == {"seed": 123}


def test_parse_level_accepts_string(preserve_root_logger):
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    level = cli_main._parse_level("debug")
    root.setLevel(level)

    assert root.level == logging.DEBUG


def test_parse_level_accepts_int(preserve_root_logger):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    level = cli_main._parse_level(logging.ERROR)
    root.setLevel(level)

    assert root.level == logging.ERROR


def test_analyze_metrics_dispatches_rng(monkeypatch, preserve_root_logger):
    calls: list[tuple[str, tuple[int, ...] | None]] = []

    monkeypatch.setattr(cli_main.metrics, "run", lambda cfg: calls.append(("metrics", None)))

    def _fake_rng(cfg, *, lags=None, force=False):  # noqa: ANN001
        calls.append(("rng", tuple(lags) if lags is not None else None))

    monkeypatch.setattr("farkle.analysis.rng_diagnostics.run", _fake_rng, raising=True)

    cli_main.main(
        ["analyze", "metrics", "--rng-diagnostics", "--rng-lags", "2", "1", "2"]
    )

    assert ("metrics", None) in calls
    assert ("rng", (1, 2)) in calls


def test_stringify_paths_handles_nested_structures():
    nested = {
        "path": Path("/tmp/example"),
        "items": [Path("a"), {"b": Path("c")}, (Path("d"),)],
    }

    result = cli_main._stringify_paths(nested)

    assert result == {
        "path": str(Path("/tmp/example")),
        "items": ["a", {"b": "c"}, ("d",)],
    }


def test_write_active_config_persists_yaml(tmp_path: Path, monkeypatch):
    cfg = cli_main.AppConfig()
    cfg.io.results_dir = tmp_path / "results"
    cfg.analysis.outputs = {}
    cfg.sim.row_dir = tmp_path / "rows"

    cli_main._write_active_config(cfg, tmp_path)

    written = tmp_path / "active_config.yaml"
    assert written.exists()
    data = yaml.safe_load(written.read_text())
    assert data["io"]["results_dir"].endswith("results")
    assert data["sim"]["row_dir"].endswith("rows")


def test_main_time_dispatches(monkeypatch, preserve_root_logger):
    captured: dict[str, object] = {}

    def fake_measure_sim_times(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "measure_sim_times", fake_measure_sim_times)

    cli_main.main(["time", "--players", "3", "--n-games", "10", "--jobs", "2", "--seed", "9"])

    assert captured == {"n_games": 10, "players": 3, "seed": 9, "jobs": 2}


def _write_cfg(tmp_path: Path) -> Path:
    cfg = {
        "io": {"results_dir": str(tmp_path / "out")},
        "sim": {"seed": 7, "n_players_list": [2], "num_shuffles": 1, "recompute_num_shuffles": False},
    }
    path = tmp_path / "cfg.yml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_analyze_variance_dispatch(monkeypatch, tmp_path: Path, preserve_root_logger):
    called: dict[str, object] = {}

    monkeypatch.setattr(cli_main.analysis_pkg, "run_variance", lambda cfg, *, force=False: called.update(force=force))

    cfg_path = _write_cfg(tmp_path)
    cli_main.main(["--config", str(cfg_path), "analyze", "variance", "--force"])

    assert called == {"force": True}


def test_analyze_pipeline_runs_preprocess(monkeypatch, tmp_path: Path, preserve_root_logger):
    calls: list[str] = []

    monkeypatch.setattr(cli_main, "_run_preprocess", lambda cfg, **kwargs: calls.append("preprocess"))
    monkeypatch.setattr(cli_main.analysis_pkg, "run_all", lambda cfg: calls.append("run_all"))

    cfg_path = _write_cfg(tmp_path)
    cli_main.main(["--config", str(cfg_path), "analyze", "pipeline", "--compute-game-stats"])

    assert calls == ["preprocess", "run_all"]
