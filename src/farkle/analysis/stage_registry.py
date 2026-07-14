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
    "resolve_stage_definition",
    "resolve_root_pair_stage_layout",
    "resolve_stage_layout",
]


@dataclass(frozen=True, slots=True)
class StageDefinition:
    """Declarative description of a pipeline or analytics stage."""

    key: str
    group: str
    folder_stub: str | None = None
    depends_on: tuple[str, ...] = ()
    cache_scope: tuple[str, ...] = ()
    cache_key_version: int = 3
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


# Ordered root workflow. It deliberately ends at descriptive screening.
_REGISTRY: tuple[StageDefinition, ...] = (
    StageDefinition(
        "ingest",
        group="pipeline",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.seed_list",
            "sim.n_players_list",
            "ingest",
            "analysis.mp_start_method",
        ),
    ),
    StageDefinition(
        "curate",
        group="pipeline",
        cache_scope=("io", "analysis.outputs"),
    ),
    StageDefinition(
        "combine",
        group="pipeline",
        cache_scope=("io", "combine"),
    ),
    StageDefinition(
        "metrics",
        group="pipeline",
        cache_scope=(
            "io",
            "sim.n_players_list",
            "screening",
            "batching",
            "k_aggregation",
            "analysis.n_jobs",
        ),
    ),
    StageDefinition(
        "game_stats",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.seed_list",
            "sim.n_players_list",
            "k_aggregation",
            "analysis.k_weights",
            "analysis.rare_event_target_score",
            "analysis.rare_event_write_details",
            "analysis.rare_event_margin_quantile",
            "analysis.rare_event_target_rate",
            "analysis.n_jobs",
        ),
    ),
    StageDefinition(
        "rng_diagnostics",
        group="diagnostics",
        folder_stub="rng",
        cache_scope=("io", "sim.seed", "sim.n_players_list", "rng", "analysis.n_jobs"),
        disabled_predicate=lambda cfg: cfg.analysis.disable_rng_diagnostics,
    ),
    StageDefinition(
        "trueskill",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.seed_list",
            "sim.n_players_list",
            "trueskill",
        ),
    ),
    StageDefinition("hgb", group="analytics", cache_scope=("io", "hgb")),
    StageDefinition(
        "screening",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.n_players_list",
            "screening",
            "robustness",
            "artifact_contract",
        ),
    ),
    # The single-root H2H tail is present in the layout but is never part of
    # the root workflow used by a two-root run.
    StageDefinition(
        "candidate_freeze",
        group="single_root_tail",
        cache_scope=("screening", "head2head"),
    ),
    StageDefinition(
        "head2head",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.n_players_list",
            "head2head",
            "k_aggregation",
        ),
    ),
    StageDefinition("h2h_power", group="single_root_tail", cache_scope=("head2head",)),
    StageDefinition(
        "h2h_execute",
        group="single_root_tail",
        cache_scope=("head2head", "rng"),
    ),
    StageDefinition("h2h_inference", group="single_root_tail", cache_scope=("head2head",)),
    StageDefinition("h2h_digest", group="single_root_tail", cache_scope=("head2head",)),
    StageDefinition("agreement", group="single_root_tail", cache_scope=("artifact_contract",)),
    StageDefinition("reporting", group="single_root_tail", cache_scope=("artifact_contract",)),
)

_ROOT_PAIR_REGISTRY: tuple[StageDefinition, ...] = (
    StageDefinition("cross_seed", group="root_pair", cache_scope=("sim.seed_list", "robustness")),
    StageDefinition("trueskill", group="root_pair", cache_scope=("sim.seed_list", "trueskill")),
    StageDefinition("candidate_freeze", group="root_pair", cache_scope=("screening", "head2head")),
    StageDefinition("head2head", group="root_pair", cache_scope=("head2head", "rng")),
    StageDefinition("h2h_power", group="root_pair", cache_scope=("head2head",)),
    StageDefinition("h2h_execute", group="root_pair", cache_scope=("head2head", "rng")),
    StageDefinition("h2h_inference", group="root_pair", cache_scope=("head2head",)),
    StageDefinition("h2h_digest", group="root_pair", cache_scope=("head2head",)),
    StageDefinition("agreement", group="root_pair", cache_scope=("artifact_contract",)),
    StageDefinition("reporting", group="root_pair", cache_scope=("artifact_contract",)),
)

_DEFINITION_LOOKUP: dict[str, StageDefinition] = {
    definition.key: definition for definition in (*_REGISTRY, *_ROOT_PAIR_REGISTRY)
}


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
        missing_dependencies = tuple(
            dep for dep in definition.depends_on if dep not in enabled_keys
        )
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


def resolve_root_pair_stage_layout(
    cfg: AppConfig,
) -> StageLayout:
    """Resolve the one-time root-pair workflow layout."""

    return resolve_stage_layout(cfg, registry=_ROOT_PAIR_REGISTRY)


def resolve_stage_definition(key: str) -> StageDefinition:
    """Return the canonical definition for ``key``."""

    try:
        return _DEFINITION_LOOKUP[key]
    except KeyError as exc:
        raise KeyError(f"Unknown stage key {key!r}") from exc
