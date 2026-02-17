"""Unit tests for the stage registry resolver."""

from __future__ import annotations

from pathlib import Path

from farkle.analysis import stage_registry
from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AnalysisConfig, AppConfig, IOConfig


def test_resolve_stage_layout_default_numbering(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))

    layout = resolve_stage_layout(cfg)

    expected_keys = [definition.key for definition in stage_registry._REGISTRY]

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

    base_layout = resolve_stage_layout(AppConfig(io=IOConfig(results_dir_prefix=tmp_path)))
    layout = resolve_stage_layout(cfg)

    assert [placement.definition.key for placement in layout.placements] == [
        placement.definition.key for placement in base_layout.placements
    ]
    assert [placement.folder_name for placement in layout.placements] == [
        placement.folder_name for placement in base_layout.placements
    ]


def test_resolve_stage_layout_config_controls_rng(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))

    base_layout = resolve_stage_layout(cfg)
    cfg.analysis.disable_rng_diagnostics = True

    layout = resolve_stage_layout(cfg)

    assert [placement.definition.key for placement in layout.placements] == [
        placement.definition.key for placement in base_layout.placements
    ]
