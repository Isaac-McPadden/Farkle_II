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

    def make_module(
        name: str, label: str, *, func_name: str = "run", record_call: bool = True
    ) -> types.ModuleType:
        module = types.ModuleType(f"farkle.analysis.{name}")
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)

        def _run(cfg, **_):  # noqa: ANN001
            if record_call:
                calls.append(label)

        setattr(module, func_name, _run)
        return module

    monkeypatch.setitem(
        sys.modules,
        "farkle.analysis.trueskill",
        make_module("trueskill", "trueskill"),
    )
    monkeypatch.setitem(
        sys.modules,
        "farkle.analysis.head2head",
        make_module("head2head", "head2head"),
    )
    monkeypatch.setitem(
        sys.modules,
        "farkle.analysis.h2h_analysis",
        make_module("h2h_analysis", "h2h_analysis", func_name="run_post_h2h", record_call=False),
    )
    monkeypatch.setitem(
        sys.modules,
        "farkle.analysis.hgb_feat",
        make_module("hgb_feat", "hgb"),
    )
    monkeypatch.setitem(
        sys.modules,
        "farkle.analysis.tiering_report",
        make_module("tiering_report", "tiering_report", record_call=False),
    )
    monkeypatch.setitem(
        sys.modules,
        "farkle.analysis.agreement",
        make_module("agreement", "agreement"),
    )
    monkeypatch.setitem(
        sys.modules,
        "farkle.analysis.meta",
        make_module("meta", "meta", record_call=False),
    )

    import farkle.analysis as analysis_mod  # import after stubbing dependencies

    run_all = analysis_mod.run_all

    def _seed_stub(cfg, **_):  # noqa: ANN001
        calls.append("seed_summaries")

    monkeypatch.setattr(analysis_mod, "run_seed_summaries", _seed_stub)
    monkeypatch.setattr(analysis_mod, "run_variance", lambda cfg, **__: calls.append("variance"))
    monkeypatch.setattr(analysis_mod, "run_meta", lambda cfg, **__: calls.append("meta"))

    cfg = AppConfig()
    cfg.analysis.run_trueskill = ts
    cfg.analysis.run_head2head = h2h
    cfg.analysis.run_hgb = hgb
    cfg.analysis.run_post_h2h_analysis = True
    cfg.analysis.run_frequentist = True
    cfg.analysis.run_agreement = True

    with caplog.at_level(logging.INFO):
        run_all(cfg)

    expected_calls: List[str] = ["seed_summaries", "variance", "meta"]
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

    if cfg.analysis.run_agreement:
        expected_calls.append("agreement")
        assert "Analytics: skipping agreement" not in caplog.text
    else:
        assert "Analytics: skipping agreement" in caplog.text
    assert calls == expected_calls
    assert "Analytics: starting all modules" in caplog.text
    assert "Analytics: all modules finished" in caplog.text


def test_optional_import_logs_missing(caplog: pytest.LogCaptureFixture) -> None:
    import farkle.analysis as analysis_mod

    with caplog.at_level(logging.INFO):
        result = analysis_mod._optional_import("not_a_real_module")

    assert result is None
    assert "Analytics module skipped due to missing dependency" in caplog.text


def test_skip_message_logs(caplog: pytest.LogCaptureFixture) -> None:
    import farkle.analysis as analysis_mod

    with caplog.at_level(logging.INFO):
        analysis_mod._skip_message("demo", "testing")

    assert "Analytics: skipping demo" in caplog.text
