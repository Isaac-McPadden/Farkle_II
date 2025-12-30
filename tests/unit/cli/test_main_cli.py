from __future__ import annotations

import logging

import pytest

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
