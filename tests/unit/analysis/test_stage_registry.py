"""Unit tests for the stage registry resolver."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from farkle.analysis import stage_registry
from farkle.analysis.ingest import _process_block
from farkle.analysis.stage_registry import (
    StageDefinition,
    StageLayout,
    StagePlacement,
    resolve_root_pair_stage_layout,
    resolve_stage_layout,
)
from farkle.config import AppConfig, IOConfig


def test_resolve_stage_layout_default_numbering(tmp_path: Path) -> None:
    cfg = AppConfig(IOConfig(results_dir_prefix=tmp_path))

    layout = resolve_stage_layout(cfg)

    expected_keys = [definition.key for definition in stage_registry._REGISTRY]

    assert [placement.definition.key for placement in layout.placements] == expected_keys
    assert [placement.index for placement in layout.placements] == list(range(len(expected_keys)))
    assert [placement.folder_name for placement in layout.placements] == [
        placement.definition.folder_name(placement.index) for placement in layout.placements
    ]

    ordered_keys = [placement.definition.key for placement in layout.placements]
    assert ordered_keys[:9] == [
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
    assert not {"variance", "meta", "post_h2h", "seed_symmetry"}.intersection(ordered_keys)


def test_resolve_stage_layout_has_no_retired_stage_switches(tmp_path: Path) -> None:
    layout = resolve_stage_layout(AppConfig(IOConfig(results_dir_prefix=tmp_path)))

    assert not {
        "variance",
        "meta",
        "frequentist",
        "post_h2h",
        "seed_symmetry",
        "h2h_tier_trends",
    }.intersection(layout.keys())


def test_resolve_stage_layout_config_controls_rng(tmp_path: Path) -> None:
    cfg = AppConfig(IOConfig(results_dir_prefix=tmp_path))

    base_layout = resolve_stage_layout(cfg)
    cfg.analysis.disable_rng_diagnostics = True

    layout = resolve_stage_layout(cfg)

    assert "rng_diagnostics" in [placement.definition.key for placement in base_layout.placements]
    assert "rng_diagnostics" not in [placement.definition.key for placement in layout.placements]


def test_resolved_app_config_crosses_spawn_process_boundary(tmp_path: Path) -> None:
    cfg = AppConfig(IOConfig(results_dir_prefix=tmp_path))
    expected_keys = cfg.stage_layout.keys()

    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        future = executor.submit(
            _process_block,
            tmp_path / "invalid_block",
            cfg,
            parent_process_workers=2,
        )
        with pytest.raises(ValueError, match="invalid player-count block name"):
            future.result(timeout=30)

    assert cfg.stage_layout.keys() == expected_keys


def test_resolve_stage_layout_excludes_disabled_stages_before_numbering(tmp_path: Path) -> None:
    cfg = AppConfig(IOConfig(results_dir_prefix=tmp_path))
    registry = (
        StageDefinition(key="always", group="unit"),
        StageDefinition(
            key="disabled",
            group="unit",
            disabled_predicate=lambda _cfg: True,
        ),
        StageDefinition(key="after", group="unit"),
    )

    layout = resolve_stage_layout(cfg, registry=registry)

    assert layout.keys() == ["always", "after"]
    assert [placement.index for placement in layout.placements] == [0, 1]
    assert [placement.folder_name for placement in layout.placements] == ["00_always", "01_after"]


def test_resolve_stage_layout_errors_when_enabled_dependency_is_disabled(tmp_path: Path) -> None:
    cfg = AppConfig(IOConfig(results_dir_prefix=tmp_path))
    registry = (
        StageDefinition(
            key="upstream",
            group="unit",
            disabled_predicate=lambda _cfg: True,
        ),
        StageDefinition(
            key="downstream",
            group="unit",
            depends_on=("upstream",),
        ),
    )

    with pytest.raises(ValueError, match="depends on disabled or unknown stages"):
        resolve_stage_layout(cfg, registry=registry)


def test_resolve_stage_layout_keeps_enabled_dependencies_unchanged(tmp_path: Path) -> None:
    cfg = AppConfig(IOConfig(results_dir_prefix=tmp_path))
    registry = (
        StageDefinition(key="upstream", group="unit"),
        StageDefinition(key="downstream", group="unit", depends_on=("upstream",)),
    )

    layout = resolve_stage_layout(cfg, registry=registry)

    assert layout.keys() == ["upstream", "downstream"]


def test_stage_registry_helpers_and_root_pair_layout(tmp_path: Path) -> None:
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

    pair_layout = resolve_root_pair_stage_layout(cfg)
    assert pair_layout.keys() == [
        "cross_seed",
        "trueskill",
        "candidate_freeze",
        "head2head",
        "h2h_power",
        "h2h_execute",
        "h2h_inference",
        "h2h_digest",
        "agreement",
        "reporting",
    ]
    assert [placement.index for placement in pair_layout.placements] == list(
        range(len(pair_layout.placements))
    )
