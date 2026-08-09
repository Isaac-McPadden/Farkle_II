from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from farkle.cli import launcher
from farkle.config import AppConfig
from farkle.utils.os_memory import SETUP_FAILURE_EXIT_CODE, MemoryBoundaryError


def test_protected_command_launches_real_cli_under_boundary(monkeypatch) -> None:
    cfg = AppConfig()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(launcher, "_resource_config", lambda _argv: cfg)

    def _supervise(command, resources, *, env):
        captured["command"] = command
        captured["resources"] = resources
        captured["env"] = env
        return 17

    monkeypatch.setattr(launcher, "supervise_process", _supervise)

    result = launcher.main(["analyze", "metrics"])

    assert result == 17
    assert captured["command"][1:3] == ["-m", "farkle.cli.protected"]
    assert captured["resources"] is cfg.resources
    assert captured["env"]["OMP_NUM_THREADS"] == "1"


def test_required_setup_failure_propagates_distinct_exit_code(monkeypatch, capsys) -> None:
    monkeypatch.setattr(launcher, "_resource_config", lambda _argv: AppConfig())
    monkeypatch.setattr(
        launcher,
        "supervise_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryBoundaryError("denied")),
    )

    assert launcher.main(["two-seed-pipeline"]) == SETUP_FAILURE_EXIT_CODE
    assert "failed closed" in capsys.readouterr().err


def test_non_pipeline_command_does_not_start_boundary(monkeypatch) -> None:
    captured = SimpleNamespace(argv=None)
    from farkle.cli import main as cli_main

    monkeypatch.setattr(cli_main, "main", lambda argv: setattr(captured, "argv", argv))
    monkeypatch.setattr(
        launcher,
        "supervise_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected boundary")),
    )

    assert launcher.main(["time", "--n-games", "1"]) == 0
    assert captured.argv == ["time", "--n-games", "1"]
