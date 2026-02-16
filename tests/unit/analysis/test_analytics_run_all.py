import itertools
import logging
from typing import Any

import pytest

from farkle.config import AppConfig

LEGACY_FLAG_NAMES = (
    "run_trueskill",
    "run_head2head",
    "run_hgb",
    "run_frequentist",
    "run_post_h2h_analysis",
    "run_agreement",
)


@pytest.mark.parametrize(
    "legacy_values",
    list(itertools.product((True, False), repeat=len(LEGACY_FLAG_NAMES))),
)
def test_run_all_uses_current_stage_order_for_all_legacy_flag_combinations(
    legacy_values: tuple[bool, ...],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import farkle.analysis as analysis_mod

    calls: list[str] = []

    def _single_seed(cfg: AppConfig, *, force: bool = False) -> None:  # noqa: ARG001
        assert force is False
        calls.append("single_seed")

    def _interseed(
        cfg: AppConfig,  # noqa: ARG001
        *,
        force: bool = False,
        manifest_path: Any = None,
        rng_lags: list[int] | None = None,
        run_rng_diagnostics: bool | None = None,
    ) -> None:
        assert force is False
        assert manifest_path is None
        assert rng_lags == [1, 3]
        assert run_rng_diagnostics is False
        calls.append("interseed")

    def _h2h_tier_trends(cfg: AppConfig, *, force: bool = False) -> None:  # noqa: ARG001
        assert force is False
        calls.append("h2h_tier_trends")

    monkeypatch.setattr(analysis_mod, "run_single_seed_analysis", _single_seed)
    monkeypatch.setattr(analysis_mod, "run_interseed_analysis", _interseed)
    monkeypatch.setattr(analysis_mod, "run_h2h_tier_trends", _h2h_tier_trends)

    # Legacy direct module hooks removed from run_all should not be called.
    monkeypatch.setattr(
        analysis_mod,
        "run_seed_summaries",
        lambda cfg, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected call")),
    )
    monkeypatch.setattr(
        analysis_mod,
        "run_variance",
        lambda cfg, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected call")),
    )
    monkeypatch.setattr(
        analysis_mod,
        "run_meta",
        lambda cfg, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected call")),
    )

    cfg = AppConfig()
    for name, value in zip(LEGACY_FLAG_NAMES, legacy_values):
        setattr(cfg.analysis, name, value)

    with caplog.at_level(logging.INFO):
        analysis_mod.run_all(cfg, run_rng_diagnostics=False, rng_lags=[1, 3])

    assert calls == ["single_seed", "interseed", "h2h_tier_trends"]
    assert "Analytics: starting all modules" in caplog.text
    assert "Analytics: all modules finished" in caplog.text


def test_run_all_does_not_invoke_removed_direct_module_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import farkle.analysis as analysis_mod

    called: list[str] = []

    monkeypatch.setattr(
        analysis_mod,
        "run_single_seed_analysis",
        lambda cfg, **kwargs: called.append("single_seed"),
    )
    monkeypatch.setattr(
        analysis_mod,
        "run_interseed_analysis",
        lambda cfg, **kwargs: called.append("interseed"),
    )
    monkeypatch.setattr(
        analysis_mod,
        "run_h2h_tier_trends",
        lambda cfg, **kwargs: called.append("h2h_tier_trends"),
    )

    for removed_hook in ("run_seed_summaries", "run_variance", "run_meta"):
        monkeypatch.setattr(
            analysis_mod,
            removed_hook,
            lambda cfg, _hook=removed_hook, **kwargs: called.append(_hook),
        )

    analysis_mod.run_all(AppConfig())

    assert called == ["single_seed", "interseed", "h2h_tier_trends"]


def test_run_all_respects_idempotent_short_circuit_when_outputs_are_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import farkle.analysis as analysis_mod

    expensive_work: list[str] = []

    def _fresh_stage(_: AppConfig, *, force: bool = False, **kwargs: object) -> None:
        if force:
            expensive_work.append("recompute")

    monkeypatch.setattr(analysis_mod, "run_single_seed_analysis", _fresh_stage)
    monkeypatch.setattr(analysis_mod, "run_interseed_analysis", _fresh_stage)
    monkeypatch.setattr(analysis_mod, "run_h2h_tier_trends", _fresh_stage)

    analysis_mod.run_all(AppConfig())

    assert expensive_work == []


def test_optional_import_logs_missing(caplog: pytest.LogCaptureFixture) -> None:
    import farkle.analysis as analysis_mod

    with caplog.at_level(logging.INFO):
        result = analysis_mod._optional_import("not_a_real_module")

    assert result is None
    assert "Analytics module skipped due to missing dependency" in caplog.text


def test_stage_logger_missing_input_logs(caplog: pytest.LogCaptureFixture) -> None:
    import farkle.analysis as analysis_mod

    with caplog.at_level(logging.INFO):
        analysis_mod.stage_logger("demo").missing_input("testing")

    assert "Analytics: skipping demo" in caplog.text
