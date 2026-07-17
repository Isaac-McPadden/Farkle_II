"""Tests for canonical root and root-pair orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, IOConfig, SimConfig, assign_config_sha
from farkle.orchestration import two_seed_pipeline
from farkle.orchestration.run_contexts import (
    SEED_PAIR_ANALYSIS_DIRNAME,
    RootPairRunContext,
    SeedRunContext,
)


def _context(tmp_path: Path, root: int) -> SeedRunContext:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / f"root_{root}"),
        sim=SimConfig(seed=root, seed_list=[root], n_players_list=[2]),
    )
    assign_config_sha(cfg)
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    return SeedRunContext.from_config(cfg)


def test_run_per_root_analysis_stops_before_h2h(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _context(tmp_path, 9).config
    captured: dict[str, Any] = {}

    def fake_run(plan: list[Any], context: Any, raise_on_failure: bool) -> None:
        captured["plan"] = [item.name for item in plan]
        captured["label"] = context.run_label
        captured["raise"] = raise_on_failure

    monkeypatch.setattr(two_seed_pipeline.StageRunner, "run", staticmethod(fake_run))
    policy = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=1)

    two_seed_pipeline._run_per_seed_analysis(
        cfg,
        seed=9,
        force=False,
        policy_bundle=policy,
    )

    assert captured["plan"][-1] == "screening"
    assert not {"candidate_freeze", "h2h_execute", "h2h_digest"}.intersection(captured["plan"])
    assert captured["label"] == "per_seed_pipeline_9"
    assert captured["raise"] is True


def test_root_config_replaces_pair_seed_list_with_owned_root(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
    )
    policy = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=2)

    root_cfg = two_seed_pipeline._build_seed_cfg(
        cfg,
        seed_pair=(11, 22),
        seed=22,
        policy_bundle=policy,
    )

    assert root_cfg.sim.seed == 22
    assert root_cfg.sim.seed_list == [22]


def _install_root_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    failed_root: int | None = None,
) -> None:
    def fake_run_one_seed(
        _cfg: AppConfig,
        *,
        seed: int,
        **_kwargs: object,
    ) -> two_seed_pipeline._SeedRunStatus:
        context = _context(tmp_path, seed)
        if seed == failed_root:
            return two_seed_pipeline._SeedRunStatus(
                seed=seed,
                context=context,
                simulation_ok=True,
                analysis_ok=False,
                analysis_error="root analysis failed",
            )
        return two_seed_pipeline._SeedRunStatus(
            seed=seed,
            context=context,
            simulation_ok=True,
            analysis_ok=True,
        )

    monkeypatch.setattr(two_seed_pipeline, "_run_one_seed", fake_run_one_seed)
    monkeypatch.setattr(two_seed_pipeline, "validate_manifest_contract", lambda _path: None)
    monkeypatch.setattr(two_seed_pipeline, "append_manifest_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(two_seed_pipeline, "write_active_config", lambda *_args, **_kwargs: None)


def test_two_seed_pipeline_runs_pair_tail_once_at_pair_analysis_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
    )
    _install_root_results(monkeypatch, tmp_path)
    calls: list[RootPairRunContext] = []
    health: dict[str, Any] = {}
    monkeypatch.setattr(
        two_seed_pipeline.analysis,
        "run_root_pair_analysis",
        lambda context, **_: calls.append(context),
    )
    monkeypatch.setattr(
        two_seed_pipeline,
        "_write_pipeline_health",
        lambda _path, payload: health.update(payload),
    )

    two_seed_pipeline.run_pipeline(cfg, seed_pair=(11, 22))

    assert len(calls) == 1
    context = calls[0]
    pair_root = two_seed_pipeline.seed_pair_root(cfg, (11, 22))
    assert context.root_pair == (11, 22)
    assert context.analysis_root == pair_root / SEED_PAIR_ANALYSIS_DIRNAME
    assert health["status"] == "complete_success"
    assert health["pair_workflow"]["analysis_root"] == str(context.analysis_root)


def test_two_seed_pipeline_blocks_pair_tail_after_root_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
    )
    _install_root_results(monkeypatch, tmp_path, failed_root=22)
    pair_calls: list[object] = []
    health: dict[str, Any] = {}
    monkeypatch.setattr(
        two_seed_pipeline.analysis,
        "run_root_pair_analysis",
        lambda context, **_: pair_calls.append(context),
    )
    monkeypatch.setattr(
        two_seed_pipeline,
        "_write_pipeline_health",
        lambda _path, payload: health.update(payload),
    )

    with pytest.raises(RuntimeError, match="root analysis failed"):
        two_seed_pipeline.run_pipeline(cfg, seed_pair=(11, 22))

    assert pair_calls == []
    assert health["status"] == "failed"
    assert health["pair_workflow"]["status"] == "failed"


def test_worker_budget_is_split_across_concurrent_roots(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.sim.n_jobs = 8
    cfg.orchestration.parallel_seeds = True

    policy = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=2)

    assert policy.simulation.process_workers == 4
    assert policy.analysis.process_workers <= 4
