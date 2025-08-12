import logging
import sys
import types
from dataclasses import dataclass
from typing import List

import pytest


@dataclass
class DummyCfg:
    run_trueskill: bool
    run_head2head: bool
    run_hgb: bool


@pytest.mark.parametrize(
    "ts,h2h,hgb",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
def test_run_all_invokes_expected_modules(ts, h2h, hgb, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    calls: List[str] = []

    def make_dummy(name: str):
        def _run(cfg):  # noqa: ANN001, ARG001
            calls.append(name)
        return _run

    # Stub out heavy dependency modules before importing run_all
    stub = types.ModuleType("run_hgb")
    stub.main = lambda *args, **kwargs: None  # noqa: ARG005  # type:ignore
    monkeypatch.setitem(sys.modules, "farkle.run_hgb", stub)

    from farkle.analytics import run_all  # import after stubbing dependencies

    ts_mod = __import__("farkle.analytics.trueskill", fromlist=["run"])
    h2h_mod = __import__("farkle.analytics.head2head", fromlist=["run"])
    hgb_mod = __import__("farkle.analytics.hgb_feat", fromlist=["run"])
    monkeypatch.setattr(ts_mod, "run", make_dummy("trueskill"))
    monkeypatch.setattr(h2h_mod, "run", make_dummy("head2head"))
    monkeypatch.setattr(hgb_mod, "run", make_dummy("hgb"))

    cfg = DummyCfg(ts, h2h, hgb)

    with caplog.at_level(logging.INFO):
        run_all(cfg)  # type: ignore[arg-type]

    expected_calls = []
    if ts:
        expected_calls.append("trueskill")
        assert "Analytics: skipping trueskill" not in caplog.text
    else:
        assert "Analytics: skipping trueskill" in caplog.text
    if h2h:
        expected_calls.append("head2head")
        assert "Analytics: skipping head-to-head" not in caplog.text
    else:
        assert "Analytics: skipping head-to-head" in caplog.text
    if hgb:
        expected_calls.append("hgb")
        assert "Analytics: skipping hist gradient boosting" not in caplog.text
    else:
        assert "Analytics: skipping hist gradient boosting" in caplog.text

    assert calls == expected_calls
    assert "Analytics: starting all modules" in caplog.text
    assert "Analytics: all modules finished" in caplog.text
