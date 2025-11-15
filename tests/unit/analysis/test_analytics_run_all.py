import importlib.machinery
import logging
import sys
import types
from typing import List

import pytest

from farkle.config import AppConfig


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
def test_run_all_invokes_expected_modules(
    ts, h2h, hgb, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: List[str] = []

    def make_module(name: str, label: str) -> types.ModuleType:
        module = types.ModuleType(f"farkle.analysis.{name}")
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)

        def _run(cfg, **_):  # noqa: ANN001
            calls.append(label)

        module.run = _run  # type: ignore[attr-defined]
        return module

    labels = {
        "trueskill": "trueskill",
        "head2head": "head2head",
        "hgb_feat": "hgb",
    }

    for mod_name, label in labels.items():
        monkeypatch.setitem(
            sys.modules, f"farkle.analysis.{mod_name}", make_module(mod_name, label)
        )

    import farkle.analysis as analysis_mod  # import after stubbing dependencies

    run_all = analysis_mod.run_all

    def _seed_stub(cfg, **_):  # noqa: ANN001
        calls.append("seed_summaries")

    monkeypatch.setattr(analysis_mod, "run_seed_summaries", _seed_stub)

    cfg = AppConfig()
    cfg.analysis.run_trueskill = ts
    cfg.analysis.run_head2head = h2h
    cfg.analysis.run_hgb = hgb

    with caplog.at_level(logging.INFO):
        run_all(cfg)

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

    expected_calls.append("seed_summaries")
    assert calls == expected_calls
    assert "Analytics: starting all modules" in caplog.text
    assert "Analytics: all modules finished" in caplog.text
