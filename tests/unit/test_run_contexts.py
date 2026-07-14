"""Tests for root and root-pair run-context path wiring."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, IOConfig, SimConfig
from farkle.orchestration.run_contexts import RootPairRunContext, RunContextConfig, SeedRunContext


def _root_context(tmp_path: Path, root: int) -> SeedRunContext:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / f"root_{root}"),
        sim=SimConfig(seed=root, seed_list=[root], n_players_list=[2, 4]),
    )
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    return SeedRunContext.from_config(cfg)


def test_root_pair_context_uses_pair_analysis_root_and_both_roots(tmp_path: Path) -> None:
    first = _root_context(tmp_path, 11)
    second = _root_context(tmp_path, 22)
    pair_root = tmp_path / "pair"

    context = RootPairRunContext.from_root_contexts((first, second), pair_root=pair_root)

    assert context.root_pair == (11, 22)
    assert context.analysis_root == pair_root / first.config.io.analysis_subdir
    assert context.config.analysis_dir == context.analysis_root
    assert context.config.sim.seed_list == [11, 22]
    assert context.config.stage_layout.keys()[0] == "cross_seed"


def test_root_pair_context_maps_first_root_inputs_without_changing_outputs(tmp_path: Path) -> None:
    first = _root_context(tmp_path, 7)
    second = _root_context(tmp_path, 8)
    context = RootPairRunContext.from_root_contexts(
        (first, second),
        pair_root=tmp_path / "pair",
    )

    combine_folder = first.config.stage_layout.require_folder("combine")
    assert context.config.resolve_input_stage_dir("combine", "concat_ks") == (
        first.analysis_root / combine_folder / "concat_ks"
    )
    assert context.config.cross_seed_dir("cross_seed").is_relative_to(context.analysis_root)


def test_root_pair_context_rejects_duplicate_roots(tmp_path: Path) -> None:
    first = _root_context(tmp_path, 7)

    with pytest.raises(ValueError, match="distinct roots"):
        RootPairRunContext.from_root_contexts((first, first), pair_root=tmp_path / "pair")


def test_run_context_config_preserves_all_typed_settings(tmp_path: Path) -> None:
    base = _root_context(tmp_path, 11).config
    base = replace(base, screening=replace(base.screening, resolution_delta=0.021))

    run_cfg = RunContextConfig.from_base(base, analysis_root=tmp_path / "pair" / "analysis")

    assert run_cfg.screening.resolution_delta == pytest.approx(0.021)
    assert run_cfg.analysis_dir == tmp_path / "pair" / "analysis"


def test_run_context_config_analysis_dir_falls_back_to_base(tmp_path: Path) -> None:
    base = _root_context(tmp_path, 11).config

    assert RunContextConfig.from_base(base).analysis_dir == base.analysis_dir
