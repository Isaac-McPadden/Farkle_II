import runpy

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pyarrow")

import farkle.cli.main as cli_main


def test_main_module_calls_cli(monkeypatch):
    called = False

    def fake_main():
        nonlocal called
        called = True

    monkeypatch.setattr(cli_main, "main", fake_main)
    runpy.run_module("farkle", run_name="__main__")

    assert called


@pytest.mark.parametrize("exit_code", [2, 5, 9])
def test_main_module_propagates_cli_failure_codes(monkeypatch, exit_code):
    def fake_main():
        raise SystemExit(exit_code)

    monkeypatch.setattr(cli_main, "main", fake_main)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("farkle", run_name="__main__")

    assert excinfo.value.code == exit_code
