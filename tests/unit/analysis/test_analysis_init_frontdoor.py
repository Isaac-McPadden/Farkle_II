"""Focused tests for public analytics front-door wrappers in ``farkle.analysis``."""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

import farkle.analysis as analysis_mod
from farkle.config import AppConfig, IOConfig


@pytest.mark.parametrize(
    ("wrapper_name", "module_path"),
    [
        ("run_seed_summaries", "farkle.analysis.seed_summaries"),
        ("run_coverage_by_k", "farkle.analysis.coverage_by_k"),
        ("run_meta", "farkle.analysis.meta"),
        ("run_variance", "farkle.analysis.variance"),
        ("run_h2h_tier_trends", "farkle.analysis.h2h_tier_trends"),
    ],
)
def test_run_wrappers_call_underlying_run_with_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wrapper_name: str,
    module_path: str,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    calls: list[tuple[AppConfig, bool]] = []

    def _run(app_cfg: AppConfig, *, force: bool = False) -> None:
        calls.append((app_cfg, force))

    monkeypatch.setattr(f"{module_path}.run", _run, raising=True)

    wrapper = getattr(analysis_mod, wrapper_name)
    wrapper(cfg, force=True)

    assert calls == [(cfg, True)]


def test_optional_import_success_path() -> None:
    module = analysis_mod._optional_import("json")
    assert module is not None
    assert module.__name__ == "json"


def test_optional_import_missing_dependency_logs_stage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stage_log = analysis_mod.stage_logger("optional-stage")

    with caplog.at_level(logging.INFO):
        imported = analysis_mod._optional_import("not_a_real_module_xyz", stage_log=stage_log)

    assert imported is None
    assert any(
        rec.msg == "Analytics module skipped due to missing dependency"
        and cast(Any, rec).stage == "optional-stage"
        and cast(Any, rec).status == "SKIPPED"
        for rec in caplog.records
    )


def test_optional_import_import_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_module: str) -> ModuleType:
        raise ImportError("hard import error")

    monkeypatch.setattr("farkle.analysis.importlib.import_module", _boom, raising=True)

    with pytest.raises(ImportError, match="hard import error"):
        analysis_mod._optional_import("whatever.module")


def test_run_single_seed_analysis_stage_matrix_and_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    calls: list[str] = []
    plan_names: list[str] = []

    def _run_plan(plan, context, raise_on_failure=True):  # noqa: ANN001
        plan_names.extend(item.name for item in plan)
        return [item.action(context.config) for item in plan]

    monkeypatch.setattr("farkle.analysis.StageRunner.run", _run_plan, raising=True)
    monkeypatch.setattr("farkle.analysis.run_seed_summaries", lambda app_cfg, force=False: calls.append(f"seed:{force}"), raising=True)
    monkeypatch.setattr("farkle.analysis.run_coverage_by_k", lambda app_cfg, force=False: calls.append(f"coverage:{force}"), raising=True)
    monkeypatch.setattr("farkle.analysis.run_seed_symmetry", lambda app_cfg, force=False: calls.append(f"seed_symmetry:{force}"), raising=True)

    enabled_modules = {
        "farkle.analysis.trueskill": SimpleNamespace(run=lambda app_cfg: calls.append("trueskill")),
        "farkle.analysis.tiering_report": SimpleNamespace(run=lambda app_cfg: calls.append("tiering")),
        "farkle.analysis.head2head": SimpleNamespace(run=lambda app_cfg: calls.append("head2head")),
        "farkle.analysis.h2h_analysis": SimpleNamespace(run_post_h2h=lambda app_cfg: calls.append("post_h2h")),
    }

    def _optional(module: str, *, stage_log=None):  # noqa: ANN001
        if module == "farkle.analysis.hgb_feat":
            return None
        return enabled_modules[module]

    monkeypatch.setattr("farkle.analysis._optional_import", _optional, raising=True)

    analysis_mod.run_single_seed_analysis(cfg, force=True)

    assert plan_names == [
        "seed_summaries",
        "coverage_by_k",
        "trueskill",
        "tiering",
        "head2head",
        "seed_symmetry",
        "post_h2h",
        "hgb",
    ]

    assert calls == [
        "seed:True",
        "coverage:True",
        "trueskill",
        "tiering",
        "head2head",
        "seed_symmetry:True",
        "post_h2h",
    ]


def test_run_all_forwards_flags_and_runs_all_enabled_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    calls: list[str] = []

    monkeypatch.setattr("farkle.analysis.run_single_seed_analysis", lambda app_cfg: calls.append("single_seed"), raising=True)
    monkeypatch.setattr(
        "farkle.analysis.run_interseed_analysis",
        lambda app_cfg, *, run_rng_diagnostics=None, rng_lags=None: calls.append(
            f"interseed:{run_rng_diagnostics}:{tuple(rng_lags) if rng_lags else None}"
        ),
        raising=True,
    )
    monkeypatch.setattr("farkle.analysis.run_h2h_tier_trends", lambda app_cfg: calls.append("h2h_tier_trends"), raising=True)

    analysis_mod.run_all(cfg, run_rng_diagnostics=False, rng_lags=(2, 4))

    assert calls == ["single_seed", "interseed:False:(2, 4)", "h2h_tier_trends"]


def test_run_manifest_metadata_handles_missing_and_partial_fields(tmp_path: Path) -> None:
    cfg_missing_sha = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "a"))
    cfg_missing_sha.config_sha = None

    missing = analysis_mod._run_manifest_metadata(cfg_missing_sha)
    assert missing == {
        "results_dir": str(cfg_missing_sha.results_root),
        "analysis_dir": str(cfg_missing_sha.analysis_dir),
    }

    cfg_partial = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "b"))
    cfg_partial.config_sha = "abc123"

    partial = analysis_mod._run_manifest_metadata(cfg_partial)
    assert partial["results_dir"] == str(cfg_partial.results_root)
    assert partial["analysis_dir"] == str(cfg_partial.analysis_dir)
    assert partial["config_sha"] == "abc123"
