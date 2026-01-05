# src/farkle/analysis/stage_registry.py
"""Central registry for ordered analysis stages and their metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from farkle.config import AppConfig

__all__ = [
    "StageDefinition",
    "StageLayout",
    "StagePlacement",
    "resolve_stage_layout",
]


@dataclass(frozen=True, slots=True)
class StageDefinition:
    """Declarative description of a pipeline or analytics stage."""

    key: str
    group: str
    folder_stub: str | None = None
    requires_game_stats: bool = False
    requires_rng: bool = False
    disabled_predicate: Callable[[AppConfig], bool] | None = None

    def folder_name(self, index: int) -> str:
        """Return the zero-padded folder name for this stage."""

        suffix = self.folder_stub or self.key
        return f"{index:02d}_{suffix}"

    def is_enabled(self, cfg: AppConfig, *, run_game_stats: bool, run_rng: bool) -> bool:
        """Determine whether this stage should be active for a given config."""

        predicate_disabled = (
            self.disabled_predicate(cfg)
            if self.disabled_predicate is not None
            else False
        )
        return (
            (not self.requires_game_stats or run_game_stats)
            and (not self.requires_rng or run_rng)
            and not predicate_disabled
        )


@dataclass(frozen=True, slots=True)
class StagePlacement:
    """Resolved placement of a stage within the numbered layout."""

    definition: StageDefinition
    index: int
    folder_name: str


@dataclass(slots=True)
class StageLayout:
    """Mapping from logical stage keys to numbered folder names."""

    placements: list[StagePlacement]

    def folder_for(self, key: str) -> str | None:
        """Return the numbered folder name for *key* if the stage is active."""

        for placement in self.placements:
            if placement.definition.key == key:
                return placement.folder_name
        return None

    def require_folder(self, key: str) -> str:
        """Return the numbered folder name for *key* or raise if inactive."""

        folder = self.folder_for(key)
        if folder is None:
            raise KeyError(f"Stage {key!r} is not active in the resolved layout")
        return folder

    def keys(self) -> list[str]:
        """Return the ordered list of active stage keys."""

        return [placement.definition.key for placement in self.placements]

    def to_resolved_layout(self) -> list[dict[str, str | int]]:
        """Return a serialization-friendly view of the resolved layout."""

        return [
            {
                "key": placement.definition.key,
                "folder": placement.folder_name,
                "group": placement.definition.group,
                "index": placement.index,
            }
            for placement in self.placements
        ]


# Ordered registry describing every possible stage.
_REGISTRY: tuple[StageDefinition, ...] = (
    StageDefinition("ingest", group="pipeline"),
    StageDefinition("curate", group="pipeline"),
    StageDefinition("combine", group="pipeline"),
    StageDefinition("metrics", group="pipeline"),
    StageDefinition("game_stats", group="analytics", requires_game_stats=True),
    StageDefinition("rng_diagnostics", group="analytics", folder_stub="rng", requires_rng=True),
    StageDefinition("seed_summaries", group="analytics"),
    StageDefinition("variance", group="analytics"),
    StageDefinition("meta", group="analytics"),
    StageDefinition(
        "trueskill",
        group="analytics",
        disabled_predicate=lambda cfg: cfg.analysis.disable_trueskill,
    ),
    StageDefinition(
        "head2head",
        group="analytics",
        disabled_predicate=lambda cfg: cfg.analysis.disable_head2head,
    ),
    StageDefinition(
        "hgb",
        group="analytics",
        disabled_predicate=lambda cfg: cfg.analysis.disable_hgb,
    ),
    StageDefinition(
        "tiering",
        group="analytics",
        disabled_predicate=lambda cfg: cfg.analysis.disable_tiering,
    ),
    StageDefinition(
        "agreement",
        group="analytics",
        disabled_predicate=lambda cfg: cfg.analysis.disable_agreement,
    ),
)


def resolve_stage_layout(
    cfg: AppConfig,
    *,
    run_game_stats: bool | None = None,
    run_rng: bool | None = None,
    registry: Iterable[StageDefinition] | None = None,
) -> StageLayout:
    """Filter the registry and assign sequential numbered folder names."""

    effective_game_stats = cfg.analysis.run_game_stats if run_game_stats is None else run_game_stats
    effective_run_rng = cfg.analysis.run_rng if run_rng is None else run_rng
    definitions = tuple(registry) if registry is not None else _REGISTRY

    placements: list[StagePlacement] = []
    for definition in (
        definition
        for definition in definitions
        if definition.is_enabled(
            cfg, run_game_stats=effective_game_stats, run_rng=effective_run_rng
        )
    ):
        placement_index = len(placements)
        placements.append(
            StagePlacement(
                definition=definition,
                index=placement_index,
                folder_name=definition.folder_name(placement_index),
            )
        )
    return StageLayout(placements=placements)

