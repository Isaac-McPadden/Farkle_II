import runpy


def test_main_module_calls_cli(monkeypatch):
    called = False

    def fake_main():
        nonlocal called
        called = True

    monkeypatch.setattr("farkle.cli.main.main", fake_main)
    runpy.run_module("farkle", run_name="__main__")

    assert called
