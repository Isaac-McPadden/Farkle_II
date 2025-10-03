from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pytest

pytest.importorskip("pyarrow")

import farkle.cli.main as cli_main
from farkle.config import AppConfig


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


def test_main_dispatches_run(monkeypatch, tmp_path: Path, preserve_root_logger):
    cfg = AppConfig()
    cfg.sim.n_players_list = [4]

    def fake_build(config_path, overrides):  # noqa: ANN001
        assert config_path is None
        assert overrides == []
        return cfg

    recorded: dict[str, object] = {}

    def fake_run_single(cfg_obj, n_players):  # noqa: ANN001
        recorded["cfg"] = cfg_obj
        recorded["n"] = n_players

    monkeypatch.setattr(cli_main, "_build_config", fake_build)
    monkeypatch.setattr(cli_main.runner, "run_single_n", fake_run_single)
    monkeypatch.setattr(cli_main.runner, "run_multi", lambda cfg_obj: (_ for _ in ()).throw(RuntimeError("run_multi should not be called")))

    cli_main.main(
        [
            "--log-level",
            "DEBUG",
            "run",
            "--metrics",
            "--row-dir",
            str(tmp_path),
        ]
    )

    assert recorded["cfg"] is cfg
    assert recorded["n"] == 4
    assert cfg.sim.expanded_metrics is True
    assert cfg.sim.row_dir == tmp_path
    assert logging.getLogger().level == logging.DEBUG

def test_main_dispatches_time(monkeypatch, preserve_root_logger):
    called = False

    def fake_measure():
        nonlocal called
        called = True

    monkeypatch.setattr(cli_main, "measure_sim_times", fake_measure)

    cli_main.main(["time"])

    assert called is True

def test_main_dispatches_watch(monkeypatch, preserve_root_logger):
    captured: dict[str, object] = {}

    def fake_watch(*, seed):
        captured["seed"] = seed

    monkeypatch.setattr(cli_main, "watch_game", fake_watch)

    cli_main.main(["watch", "--seed", "123"])

    assert captured == {"seed": 123}

@pytest.mark.parametrize(
    "subcommand, expected_order",
    [
        ("ingest", ["ingest"]),
        ("curate", ["curate"]),
        ("combine", ["combine"]),
        ("metrics", ["metrics"]),
        ("pipeline", ["ingest", "curate", "combine", "metrics"]),
    ],
)
def test_main_dispatches_analyze_variants(monkeypatch, subcommand, expected_order, preserve_root_logger):
    cfg = AppConfig()
    cfg.sim.n_players_list = [2]

    monkeypatch.setattr(cli_main, "_build_config", lambda *args, **kwargs: cfg)

    calls: List[str] = []

    def make_recorder(name: str):
        def _recorder(cfg_obj: AppConfig) -> None:
            assert cfg_obj is cfg
            calls.append(name)

        return _recorder

    for name in ("ingest", "curate", "combine", "metrics"):
        module = getattr(cli_main, name)
        monkeypatch.setattr(module, "run", make_recorder(name))

    cli_main.main(["analyze", subcommand])

    assert calls == expected_order


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
