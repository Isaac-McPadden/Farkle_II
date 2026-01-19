"""Unit tests for the stage registry resolver."""

from __future__ import annotations

from pathlib import Path

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AnalysisConfig, AppConfig, IOConfig


def test_resolve_stage_layout_default_numbering(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))

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
        "tiering",
        "agreement",
    ]

    assert [placement.definition.key for placement in layout.placements] == expected_keys
    assert [placement.index for placement in layout.placements] == list(range(len(expected_keys)))
    assert [placement.folder_name for placement in layout.placements] == [
        placement.definition.folder_name(placement.index) for placement in layout.placements
    ]


def test_resolve_stage_layout_cli_overrides(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path),
        analysis=AnalysisConfig(
            disable_game_stats=True,
            disable_rng_diagnostics=False,
            disable_trueskill=True,
            disable_head2head=True,
            disable_tiering=True,
            disable_agreement=True,
        ),
    )

    layout = resolve_stage_layout(cfg)

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
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))

    cfg.analysis.disable_rng_diagnostics = True

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
        "tiering",
        "agreement",
    ]

    assert [placement.definition.key for placement in layout.placements] == expected_keys
