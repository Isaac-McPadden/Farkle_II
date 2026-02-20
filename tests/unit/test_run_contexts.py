"""Tests for two-seed run-context path wiring."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, IOConfig
from farkle.orchestration.run_contexts import InterseedRunContext, RunContextConfig, SeedRunContext


def test_interseed_context_maps_combine_to_seed_stage(tmp_path: Path) -> None:
    seed_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "seed_results"))
    seed_cfg.set_stage_layout(resolve_stage_layout(seed_cfg))
    seed_context = SeedRunContext.from_config(seed_cfg)

    interseed = InterseedRunContext.from_seed_context(
        seed_context,
        seed_pair=(11, 22),
        analysis_root=tmp_path / "pair" / "interseed_analysis",
    )

    combine_folder = seed_cfg.stage_layout.require_folder("combine")
    expected_input = seed_context.analysis_root / combine_folder / "pooled"

    assert interseed.config.resolve_input_stage_dir("combine", "pooled") == expected_input


def test_interseed_context_curated_parquet_uses_upstream_combine(tmp_path: Path) -> None:
    seed_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "seed_results"))
    seed_cfg.set_stage_layout(resolve_stage_layout(seed_cfg))
    seed_context = SeedRunContext.from_config(seed_cfg)

    combine_folder = seed_cfg.stage_layout.require_folder("combine")
    upstream_curated = (
        seed_context.analysis_root / combine_folder / "pooled" / "all_ingested_rows.parquet"
    )
    upstream_curated.parent.mkdir(parents=True, exist_ok=True)
    upstream_curated.write_text("rows")

    interseed = InterseedRunContext.from_seed_context(
        seed_context,
        seed_pair=(7, 8),
        analysis_root=tmp_path / "pair" / "interseed_analysis",
    )

    assert interseed.config.curated_parquet == upstream_curated



def test_interseed_context_raises_when_combine_missing(tmp_path: Path, monkeypatch) -> None:
    seed_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "seed_results"))
    seed_cfg.set_stage_layout(resolve_stage_layout(seed_cfg))
    seed_context = SeedRunContext.from_config(seed_cfg)

    monkeypatch.setattr(
        "farkle.analysis.stage_registry.StageLayout.folder_for",
        lambda _self, key: None if key == "combine" else "x",
    )

    with pytest.raises(KeyError, match="combine"):
        InterseedRunContext.from_seed_context(
            seed_context,
            seed_pair=(1, 2),
            analysis_root=tmp_path / "pair" / "interseed_analysis",
        )


def test_interseed_context_preserves_existing_tiering_seeds(tmp_path: Path) -> None:
    seed_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "seed_results"))
    seed_cfg = replace(seed_cfg, analysis=replace(seed_cfg.analysis, tiering_seeds=[99]))
    seed_cfg.set_stage_layout(resolve_stage_layout(seed_cfg))
    seed_context = SeedRunContext.from_config(seed_cfg)

    interseed = InterseedRunContext.from_seed_context(
        seed_context,
        seed_pair=(3, 4),
        analysis_root=tmp_path / "pair" / "interseed_analysis",
    )

    assert interseed.config.analysis.tiering_seeds == [99]


def test_run_context_config_interseed_folder_unknown_layout_returns_none(tmp_path: Path) -> None:
    base_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "seed_results"))
    run_cfg = RunContextConfig.from_base(base_cfg)
    run_cfg._interseed_input_layout_override = cast(Any, object())

    assert run_cfg._interseed_input_folder("combine") is None



def test_run_context_config_analysis_dir_falls_back_to_base(tmp_path: Path) -> None:
    base_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "seed_results"))
    run_cfg = RunContextConfig.from_base(base_cfg)

    assert run_cfg.analysis_dir == base_cfg.analysis_dir


def test_run_context_config_interseed_input_dir_override(tmp_path: Path) -> None:
    base_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "seed_results"))
    override = tmp_path / "input"
    run_cfg = RunContextConfig.from_base(base_cfg, interseed_input_dir=override)

    assert run_cfg.interseed_input_dir == override


def test_run_context_config_interseed_folder_mapping_missing_key(tmp_path: Path) -> None:
    base_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "seed_results"))
    run_cfg = RunContextConfig.from_base(base_cfg)
    run_cfg._interseed_input_layout_override = {"metrics": "metrics_dir"}

    assert run_cfg._interseed_input_folder("combine") is None
    assert run_cfg._interseed_input_folder(None) is None
