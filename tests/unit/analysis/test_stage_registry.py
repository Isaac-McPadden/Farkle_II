"""Unit tests for the stage registry resolver."""

from __future__ import annotations

from pathlib import Path

from farkle.analysis import stage_registry
from farkle.analysis.stage_registry import (
    StageDefinition,
    StageLayout,
    StagePlacement,
    resolve_interseed_stage_layout,
    resolve_stage_layout,
)
from farkle.config import AnalysisConfig, AppConfig, IOConfig, SimConfig


def test_resolve_stage_layout_default_numbering(tmp_path: Path) -> None:
    cfg = AppConfig(IOConfig(results_dir_prefix=tmp_path))

    layout = resolve_stage_layout(cfg)

    expected_keys = [definition.key for definition in stage_registry._REGISTRY]

    assert [placement.definition.key for placement in layout.placements] == expected_keys
    assert [placement.index for placement in layout.placements] == list(range(len(expected_keys)))
    assert [placement.folder_name for placement in layout.placements] == [
        placement.definition.folder_name(placement.index) for placement in layout.placements
    ]


def test_resolve_stage_layout_cli_overrides(tmp_path: Path) -> None:
    cfg = AppConfig(
        IOConfig(results_dir_prefix=tmp_path),
        SimConfig(),
        AnalysisConfig(
            disable_game_stats=True,
            disable_rng_diagnostics=False,
            disable_trueskill=True,
            disable_head2head=True,
            disable_tiering=True,
            disable_agreement=True,
        ),
    )

    base_layout = resolve_stage_layout(AppConfig(IOConfig(results_dir_prefix=tmp_path)))
    layout = resolve_stage_layout(cfg)

    assert [placement.definition.key for placement in layout.placements] == [
        placement.definition.key for placement in base_layout.placements
    ]
    assert [placement.folder_name for placement in layout.placements] == [
        placement.folder_name for placement in base_layout.placements
    ]


def test_resolve_stage_layout_config_controls_rng(tmp_path: Path) -> None:
    cfg = AppConfig(IOConfig(results_dir_prefix=tmp_path))

    base_layout = resolve_stage_layout(cfg)
    cfg.analysis.disable_rng_diagnostics = True

    layout = resolve_stage_layout(cfg)

    assert [placement.definition.key for placement in layout.placements] == [
        placement.definition.key for placement in base_layout.placements
    ]


def test_stage_registry_helpers_and_interseed_layout(tmp_path: Path) -> None:
    cfg = AppConfig(IOConfig(results_dir_prefix=tmp_path))

    always_enabled = StageDefinition(key="always", group="unit", disabled_predicate=None)
    disabled_when_true = StageDefinition(
        key="disabled_when_true",
        group="unit",
        disabled_predicate=lambda _cfg: True,
    )
    enabled_when_false = StageDefinition(
        key="enabled_when_false",
        group="unit",
        disabled_predicate=lambda _cfg: False,
    )

    assert always_enabled.is_enabled(cfg) is True
    assert disabled_when_true.is_enabled(cfg) is False
    assert enabled_when_false.is_enabled(cfg) is True

    layout = StageLayout(
        placements=[
            StagePlacement(definition=always_enabled, index=0, folder_name="00_always"),
            StagePlacement(
                definition=enabled_when_false,
                index=1,
                folder_name="01_enabled_when_false",
            ),
        ]
    )

    assert layout.folder_for("missing_key") is None
    assert layout.keys() == ["always", "enabled_when_false"]
    assert layout.to_resolved_layout() == [
        {"key": "always", "folder": "00_always", "group": "unit", "index": 0},
        {
            "key": "enabled_when_false",
            "folder": "01_enabled_when_false",
            "group": "unit",
            "index": 1,
        },
    ]

    try:
        layout.require_folder("missing_key")
        raise AssertionError("Expected missing_key to raise KeyError")
    except KeyError as exc:
        assert "is not active in the resolved layout" in str(exc)

    interseed_layout = resolve_interseed_stage_layout(cfg)
    assert interseed_layout.placements[0].definition.key == "rng_diagnostics"
    assert interseed_layout.placements[0].folder_name.endswith("_rng")
    assert [placement.index for placement in interseed_layout.placements] == list(
        range(len(interseed_layout.placements))
    )
