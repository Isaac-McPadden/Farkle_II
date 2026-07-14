"""Structural tests for canonical root and root-pair plans."""

from __future__ import annotations

from pathlib import Path

import pytest

import farkle.analysis as analysis_mod
from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, IOConfig, SimConfig
from farkle.orchestration.run_contexts import RootPairRunContext, SeedRunContext


def _root_context(tmp_path: Path, root: int) -> SeedRunContext:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / f"root_{root}"),
        sim=SimConfig(seed=root, seed_list=[root], n_players_list=[2, 4]),
    )
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    return SeedRunContext.from_config(cfg)


def test_root_plan_stops_after_screening_and_diagnostics(tmp_path: Path) -> None:
    cfg = _root_context(tmp_path, 11).config

    names = [item.name for item in analysis_mod.build_root_stage_plan(cfg)]

    assert names == [
        "ingest",
        "curate",
        "combine",
        "metrics",
        "game_stats",
        "rng_diagnostics",
        "trueskill",
        "hgb",
        "screening",
    ]
    assert not {"candidate_freeze", "h2h_power", "h2h_execute", "h2h_digest"}.intersection(names)


def test_single_root_tail_is_explicitly_labelled(tmp_path: Path) -> None:
    cfg = _root_context(tmp_path, 11).config

    plan = analysis_mod.build_single_root_h2h_tail_plan(cfg)

    assert [item.name for item in plan] == [
        "candidate_freeze",
        "h2h_power",
        "h2h_execute",
        "h2h_inference",
        "h2h_digest",
        "agreement",
        "reporting",
    ]
    assert {item.metadata["execution_scope"] for item in plan} == {"single_root"}
    by_name = {item.name: item for item in plan}
    assert tuple(by_name["agreement"].required_outputs) == (
        cfg.structure_agreement_pairs_path(),
        cfg.structure_agreement_summary_path(),
    )
    assert tuple(by_name["reporting"].required_outputs) == (
        cfg.structure_report_json_path(),
        cfg.structure_report_markdown_path(),
        cfg.structure_report_plot_path(),
    )


def test_root_pair_plan_runs_one_canonical_tail(tmp_path: Path) -> None:
    first = _root_context(tmp_path, 11)
    second = _root_context(tmp_path, 22)
    context = RootPairRunContext.from_root_contexts(
        (first, second),
        pair_root=tmp_path / "pair",
    )

    plan = analysis_mod.build_root_pair_stage_plan(context)

    assert [item.name for item in plan] == [
        "cross_seed",
        "trueskill",
        "candidate_freeze",
        "h2h_power",
        "h2h_execute",
        "h2h_inference",
        "h2h_digest",
        "agreement",
        "reporting",
    ]
    assert [item.name for item in plan].count("h2h_execute") == 1
    assert all(item.metadata.get("execution_scope") == "root_pair" for item in plan[2:])


def test_run_all_dispatches_only_standalone_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _root_context(tmp_path, 11).config
    calls: list[tuple[int, tuple[int, ...]]] = []
    monkeypatch.setattr(
        analysis_mod,
        "run_single_root_analysis",
        lambda inner, **_: calls.append(
            (inner.sim.seed, tuple(inner.sim.seed_list or [inner.sim.seed]))
        ),
    )

    analysis_mod.run_all(cfg)

    assert calls == [(11, (11,))]


def test_run_all_rejects_pair_and_missing_upstream_mode(tmp_path: Path) -> None:
    cfg = _root_context(tmp_path, 11).config
    cfg.sim.seed_list = [11, 22]

    with pytest.raises(ValueError, match="two-seed-pipeline"):
        analysis_mod.run_all(cfg)
    cfg.sim.seed_list = [11]
    with pytest.raises(ValueError, match="do not permit missing"):
        analysis_mod.run_all(cfg, allow_missing_upstream=True)


def test_optional_import_logs_missing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO"):
        result = analysis_mod._optional_import("not_a_real_module")

    assert result is None
    assert "Analysis module unavailable" in caplog.text
