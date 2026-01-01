"""Unit tests for the stage registry resolver."""

from __future__ import annotations

from pathlib import Path

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AnalysisConfig, AppConfig, IOConfig


def test_resolve_stage_layout_default_numbering(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_path))

    layout = resolve_stage_layout(cfg)

    expected_keys = [
        "ingest",
        "curate",
        "combine",
        "metrics",
        "game_stats",
        "seed_summaries",
        "variance",
        "meta",
        "trueskill",
        "head2head",
        "hgb",
    ]

    assert [placement.definition.key for placement in layout.placements] == expected_keys
    assert [placement.index for placement in layout.placements] == list(range(len(expected_keys)))
    assert [placement.folder_name for placement in layout.placements] == [
        placement.definition.folder_name(placement.index) for placement in layout.placements
    ]


def test_resolve_stage_layout_cli_overrides(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir=tmp_path),
        analysis=AnalysisConfig(
            run_game_stats=True,
            run_rng=False,
            run_trueskill=False,
            run_head2head=False,
            run_hgb=True,
        ),
    )

    layout = resolve_stage_layout(cfg, run_game_stats=False, run_rng=True)

    expected_keys = [
        "ingest",
        "curate",
        "combine",
        "metrics",
        "rng_diagnostics",
        "seed_summaries",
        "variance",
        "meta",
        "hgb",
    ]

    assert [placement.definition.key for placement in layout.placements] == expected_keys
    assert [placement.index for placement in layout.placements] == list(range(len(expected_keys)))
    assert all(
        placement.folder_name.startswith(f"{placement.index:02d}_") for placement in layout.placements
    )


def test_resolve_stage_layout_config_controls_rng(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir=tmp_path),
        analysis=AnalysisConfig(run_rng=True),
    )

    layout = resolve_stage_layout(cfg)

    expected_keys = [
        "ingest",
        "curate",
        "combine",
        "metrics",
        "game_stats",
        "rng_diagnostics",
        "seed_summaries",
        "variance",
        "meta",
        "trueskill",
        "head2head",
        "hgb",
    ]

    assert [placement.definition.key for placement in layout.placements] == expected_keys
