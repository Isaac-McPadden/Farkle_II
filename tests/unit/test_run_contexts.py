"""Tests for two-seed run-context path wiring."""

from __future__ import annotations

from pathlib import Path

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, IOConfig
from farkle.orchestration.run_contexts import InterseedRunContext, SeedRunContext


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
