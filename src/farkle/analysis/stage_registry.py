# src/farkle/analysis/stage_registry.py
"""Central registry for ordered analysis stages and their metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from farkle.config import AppConfig

__all__ = [
    "StageDefinition",
    "StageLayout",
    "StagePlacement",
    "resolve_interseed_stage_layout",
    "resolve_stage_layout",
]


@dataclass(frozen=True, slots=True)
class StageDefinition:
    """Declarative description of a pipeline or analytics stage."""

    key: str
    group: str
    folder_stub: str | None = None
    depends_on: tuple[str, ...] = ()
    disabled_predicate: Callable[[AppConfig], bool] | None = None

    def folder_name(self, index: int) -> str:
        """Return the zero-padded folder name for this stage."""

        suffix = self.folder_stub or self.key
        return f"{index:02d}_{suffix}"

    def is_enabled(self, cfg: AppConfig) -> bool:
        """Determine whether this stage should be active for a given config."""

        if self.disabled_predicate is None:
            return True
        return not self.disabled_predicate(cfg)


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
    StageDefinition("coverage_by_k", group="analytics"),
    StageDefinition("game_stats", group="analytics"),
    StageDefinition("seed_summaries", group="analytics"),
    StageDefinition("trueskill", group="analytics"),
    StageDefinition("tiering", group="analytics"),
    StageDefinition("head2head", group="analytics"),
    StageDefinition("seed_symmetry", group="analytics"),
    StageDefinition("post_h2h", group="analytics"),
    StageDefinition("hgb", group="analytics"),
    StageDefinition("variance", group="analytics"),
    StageDefinition("meta", group="analytics"),
    StageDefinition("h2h_tier_trends", group="analytics"),
    StageDefinition("agreement", group="analytics"),
    StageDefinition("interseed", group="analytics"),
)

_INTERSEED_REGISTRY: tuple[StageDefinition, ...] = (
    StageDefinition(
        "rng_diagnostics",
        group="analytics",
        folder_stub="rng",
        disabled_predicate=lambda cfg: cfg.analysis.disable_rng_diagnostics,
    ),
    StageDefinition("variance", group="analytics"),
    StageDefinition("meta", group="analytics"),
    StageDefinition("trueskill", group="analytics"),
    StageDefinition("agreement", group="analytics"),
    StageDefinition("interseed", group="analytics"),
    StageDefinition("h2h_tier_trends", group="analytics"),
)


def resolve_stage_layout(
    cfg: AppConfig,
    registry: Iterable[StageDefinition] | None = None,
    enabled_overrides: Mapping[str, bool] | None = None,
) -> StageLayout:
    """Assign sequential numbered folder names to the supplied registry."""

    definitions = tuple(registry) if registry is not None else _REGISTRY
    enabled_definitions = tuple(
        definition
        for definition in definitions
        if (
            enabled_overrides.get(definition.key, definition.is_enabled(cfg))
            if enabled_overrides is not None
            else definition.is_enabled(cfg)
        )
    )
    enabled_keys = {definition.key for definition in enabled_definitions}

    for definition in enabled_definitions:
        missing_dependencies = tuple(dep for dep in definition.depends_on if dep not in enabled_keys)
        if missing_dependencies:
            raise ValueError(
                "Enabled stage "
                f"{definition.key!r} depends on disabled or unknown stages: {', '.join(missing_dependencies)}"
            )

    placements: list[StagePlacement] = []
    for definition in enabled_definitions:
        placement_index = len(placements)
        placements.append(
            StagePlacement(
                definition=definition,
                index=placement_index,
                folder_name=definition.folder_name(placement_index),
            )
        )
    return StageLayout(placements=placements)


def resolve_interseed_stage_layout(
    cfg: AppConfig,
    *,
    run_rng_diagnostics: bool | None = None,
) -> StageLayout:
    """Resolve a stage layout containing only interseed-relevant stages."""

    overrides = None
    if run_rng_diagnostics is not None:
        overrides = {"rng_diagnostics": run_rng_diagnostics}
    return resolve_stage_layout(cfg, registry=_INTERSEED_REGISTRY, enabled_overrides=overrides)
