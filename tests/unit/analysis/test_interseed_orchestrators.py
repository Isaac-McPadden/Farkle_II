"""Unit tests for interseed orchestration entry points and helper outputs."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

import farkle.analysis as analysis_pkg
from farkle.analysis import interseed_analysis
from farkle.analysis.stage_registry import resolve_interseed_stage_layout
from farkle.config import AnalysisConfig, AppConfig, IOConfig


class _StageReasonRecord(Protocol):
    stage: str
    reason: str


def _make_cfg(tmp_path: Path, *, seed_list: list[int] | None = None) -> AppConfig:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    if seed_list is not None:
        cfg.sim.seed_list = list(seed_list)
    return cfg


def _run_plan_inline(plan, context, *, raise_on_failure=True):  # noqa: ANN001, ARG001
    for item in plan:
        item.action(context.config)


def _inject_dummy_networkx(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = ModuleType("networkx")
    monkeypatch.setattr(dummy, "DiGraph", type("DiGraph", (), {}), raising=False)
    monkeypatch.setitem(sys.modules, "networkx", dummy)


def test_run_interseed_analysis_rng_diagnostics_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _make_cfg(tmp_path, seed_list=[1, 2])
    caplog.set_level(logging.INFO)

    monkeypatch.setattr("farkle.analysis.StageRunner.run", _run_plan_inline, raising=True)

    rng_calls: list[tuple[bool, tuple[int, ...] | None]] = []
    monkeypatch.setattr(
        "farkle.analysis.rng_diagnostics.run",
        lambda app_cfg, lags=None, force=False: rng_calls.append((force, tuple(lags) if lags else None)),
        raising=True,
    )

    monkeypatch.setattr("farkle.analysis.run_variance", lambda app_cfg, force=False: None, raising=True)
    monkeypatch.setattr("farkle.analysis.run_meta", lambda app_cfg, force=False: None, raising=True)
    monkeypatch.setattr("farkle.analysis._optional_import", lambda *args, **kwargs: None, raising=True)

    # explicit False (CLI disable)
    caplog.clear()
    analysis_pkg.run_interseed_analysis(
        cfg,
        run_rng_diagnostics=False,
        force=True,
        rng_lags=(2, 5),
    )
    assert rng_calls == []
    assert any(
        cast(_StageReasonRecord, rec).stage == "rng_diagnostics"
        and cast(_StageReasonRecord, rec).reason == "disabled by CLI flag"
        for rec in caplog.records
    )

    # explicit True (CLI override over config disable)
    caplog.clear()
    cfg.analysis.disable_rng_diagnostics = True
    analysis_pkg.run_interseed_analysis(
        cfg,
        run_rng_diagnostics=True,
        force=False,
        rng_lags=(1, 3),
    )
    assert rng_calls == [(False, (1, 3))]

    # None with config disable (config disable path)
    caplog.clear()
    analysis_pkg.run_interseed_analysis(cfg, run_rng_diagnostics=None)
    assert rng_calls == [(False, (1, 3))]
    assert any(
        cast(_StageReasonRecord, rec).stage == "rng_diagnostics"
        and cast(_StageReasonRecord, rec).reason == "disabled by config"
        for rec in caplog.records
    )


def test_run_interseed_analysis_skips_all_stages_when_not_interseed_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _make_cfg(tmp_path)
    caplog.set_level(logging.INFO)

    monkeypatch.setattr("farkle.analysis.StageRunner.run", _run_plan_inline, raising=True)

    called: list[str] = []
    monkeypatch.setattr(
        "farkle.analysis.run_variance",
        lambda app_cfg, force=False: called.append("variance"),
        raising=True,
    )
    monkeypatch.setattr(
        "farkle.analysis.run_meta",
        lambda app_cfg, force=False: called.append("meta"),
        raising=True,
    )
    monkeypatch.setattr(
        "farkle.analysis._optional_import",
        lambda *args, **kwargs: called.append("optional_import") or None,
        raising=True,
    )

    analysis_pkg.run_interseed_analysis(cfg)

    assert called == []
    skipped = [rec for rec in caplog.records if getattr(rec, "status", None) == "SKIPPED"]
    assert len(skipped) == 7
    assert all("interseed inputs missing" in cast(_StageReasonRecord, rec).reason for rec in skipped)


def test_run_interseed_analysis_optional_import_missing_dependency_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(tmp_path, seed_list=[11, 12])
    previous_layout = cfg._stage_layout
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))
    cfg.trueskill_path("ratings_k_weighted.parquet").write_text("x")
    cfg._stage_layout = previous_layout

    monkeypatch.setattr("farkle.analysis.StageRunner.run", _run_plan_inline, raising=True)
    monkeypatch.setattr("farkle.analysis.run_variance", lambda app_cfg, force=False: None, raising=True)
    monkeypatch.setattr("farkle.analysis.run_meta", lambda app_cfg, force=False: None, raising=True)

    optional_modules: list[str] = []

    def _missing_optional(module: str, *, stage_log=None):  # noqa: ANN001
        optional_modules.append(module)
        return None

    monkeypatch.setattr("farkle.analysis._optional_import", _missing_optional, raising=True)

    analysis_pkg.run_interseed_analysis(cfg, run_rng_diagnostics=False)

    assert optional_modules == [
        "farkle.analysis.game_stats_interseed",
        "farkle.analysis.trueskill",
        "farkle.analysis.agreement",
        "farkle.analysis.interseed_analysis",
    ]


@pytest.mark.parametrize(
    ("run_stages", "run_rng_diagnostics", "config_disables_rng", "expected_calls"),
    [
        (False, None, False, []),
        (True, False, False, ["variance", "game_stats_interseed", "meta", "trueskill", "agreement", "s_tier_stability"]),
        (True, None, True, ["variance", "game_stats_interseed", "meta", "trueskill", "agreement", "s_tier_stability"]),
        (True, True, True, ["rng_diagnostics", "variance", "game_stats_interseed", "meta", "trueskill", "agreement", "s_tier_stability"]),
    ],
)
def test_interseed_run_stage_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_stages: bool,
    run_rng_diagnostics: bool | None,
    config_disables_rng: bool,
    expected_calls: list[str],
) -> None:
    _inject_dummy_networkx(monkeypatch)
    cfg = _make_cfg(tmp_path, seed_list=[21, 22])
    cfg.analysis.disable_rng_diagnostics = config_disables_rng

    called: list[str] = []
    monkeypatch.setattr("farkle.analysis.rng_diagnostics.run", lambda app_cfg, force=False: called.append("rng_diagnostics"), raising=True)
    monkeypatch.setattr("farkle.analysis.variance.run", lambda app_cfg, force=False: called.append("variance"), raising=True)
    monkeypatch.setattr("farkle.analysis.game_stats_interseed.run", lambda app_cfg, force=False: called.append("game_stats_interseed"), raising=True)
    monkeypatch.setattr("farkle.analysis.meta.run", lambda app_cfg, force=False: called.append("meta"), raising=True)
    monkeypatch.setattr("farkle.analysis.trueskill.run", lambda app_cfg: called.append("trueskill"), raising=True)
    monkeypatch.setattr("farkle.analysis.agreement.run", lambda app_cfg: called.append("agreement"), raising=True)
    monkeypatch.setattr("farkle.analysis.interseed_analysis._run_s_tier_stability", lambda app_cfg, force=False: called.append("s_tier_stability"), raising=True)
    monkeypatch.setattr("farkle.analysis.interseed_analysis.stage_is_up_to_date", lambda *args, **kwargs: False, raising=True)

    interseed_analysis.run(
        cfg,
        force=False,
        run_stages=run_stages,
        run_rng_diagnostics=run_rng_diagnostics,
    )

    assert called == expected_calls


def test_interseed_run_stage_is_up_to_date_short_circuit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(tmp_path, seed_list=[31, 32])

    write_stage_done_called = False

    def _write_stage_done(*args, **kwargs):  # noqa: ANN001
        nonlocal write_stage_done_called
        write_stage_done_called = True

    monkeypatch.setattr("farkle.analysis.interseed_analysis.stage_is_up_to_date", lambda *args, **kwargs: True, raising=True)
    monkeypatch.setattr("farkle.analysis.interseed_analysis.write_stage_done", _write_stage_done, raising=True)

    interseed_analysis.run(cfg, run_stages=False, force=False)

    assert not list(cfg.analysis_dir.rglob(interseed_analysis.SUMMARY_NAME))
    assert write_stage_done_called is False


def test_interseed_run_force_bypasses_up_to_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(tmp_path, seed_list=[41, 42])

    write_stage_done_called = False

    def _write_stage_done(*args, **kwargs):  # noqa: ANN001
        nonlocal write_stage_done_called
        write_stage_done_called = True

    monkeypatch.setattr("farkle.analysis.interseed_analysis.stage_is_up_to_date", lambda *args, **kwargs: True, raising=True)
    monkeypatch.setattr("farkle.analysis.interseed_analysis.write_stage_done", _write_stage_done, raising=True)

    interseed_analysis.run(cfg, run_stages=False, force=True)

    summary_candidates = list(cfg.analysis_dir.rglob(interseed_analysis.SUMMARY_NAME))
    assert summary_candidates
    summary_path = summary_candidates[0]
    payload = json.loads(summary_path.read_text())
    assert payload["interseed_ready"] is True
    assert write_stage_done_called is True


def test_interseed_helper_outputs(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path),
        analysis=AnalysisConfig(agreement_include_pooled=True),
    )
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))
    cfg.sim.seed_list = [51, 52]
    cfg.sim.n_players_list = [5, 2]

    exists = cfg.interseed_stage_dir / "exists.txt"
    exists.parent.mkdir(parents=True, exist_ok=True)
    exists.write_text("ok")
    missing = cfg.interseed_stage_dir / "missing.txt"

    assert interseed_analysis._existing_paths([exists, missing]) == [str(exists)]

    statuses = {
        "a": {"outputs": ["one", "two"]},
        "b": {"outputs": ["three"]},
        "c": {},
    }
    assert interseed_analysis._flatten_output_paths(statuses) == ["one", "two", "three"]

    assert interseed_analysis._variance_outputs(cfg) == [
        cfg.variance_output_path("variance.parquet"),
        cfg.variance_output_path("variance_summary.parquet"),
        cfg.variance_output_path("variance_components.parquet"),
    ]

    assert interseed_analysis._meta_outputs(cfg) == [
        cfg.meta_output_path(2, "strategy_summary_2p_meta.parquet"),
        cfg.meta_output_path(2, "meta_2p.json"),
        cfg.meta_output_path(5, "strategy_summary_5p_meta.parquet"),
        cfg.meta_output_path(5, "meta_5p.json"),
    ]

    assert interseed_analysis._game_stats_outputs(cfg) == [
        cfg.interseed_stage_dir / "game_length_interseed.parquet",
        cfg.interseed_stage_dir / "margin_interseed.parquet",
    ]

    agreement_outputs = interseed_analysis._agreement_outputs(cfg)
    assert cfg.agreement_stage_dir / "agreement_summary.parquet" in agreement_outputs
    assert cfg.agreement_output_path_pooled() in agreement_outputs

    assert interseed_analysis._rng_outputs(cfg) == [cfg.rng_output_path("rng_diagnostics.parquet")]
    assert interseed_analysis._s_tier_stability_outputs(cfg) == [
        cfg.interseed_stage_dir / "s_tier_stability.json",
        cfg.interseed_stage_dir / "s_tier_stability.parquet",
    ]
