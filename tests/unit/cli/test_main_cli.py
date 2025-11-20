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
