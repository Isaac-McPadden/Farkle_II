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


def _rng_diagnostics_disabled(cfg: AppConfig) -> bool:
    """Return whether the root-local RNG diagnostic stage is disabled."""

    return cfg.analysis.disable_rng_diagnostics


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
        ),
    ),
    StageDefinition(
        "curate",
        group="pipeline",
        depends_on=("ingest",),
        cache_scope=("io", "analysis.outputs"),
    ),
    StageDefinition(
        "combine",
        group="pipeline",
        depends_on=("curate",),
        cache_scope=("io", "combine"),
    ),
    StageDefinition(
        "metrics",
        group="pipeline",
        depends_on=("combine",),
        cache_scope=(
            "io",
            "sim.n_players_list",
            "screening",
            "batching",
            "k_aggregation",
        ),
    ),
    StageDefinition(
        "game_stats",
        group="analytics",
        depends_on=("combine",),
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
        ),
    ),
    StageDefinition(
        "rng_diagnostics",
        group="diagnostics",
        folder_stub="rng",
        depends_on=("combine",),
        cache_scope=("io", "sim.seed", "sim.n_players_list", "rng"),
        disabled_predicate=_rng_diagnostics_disabled,
    ),
    StageDefinition(
        "trueskill",
        group="analytics",
        depends_on=("curate",),
        cache_scope=(
            "io",
            "sim.seed",
            "sim.seed_list",
            "sim.n_players_list",
            "trueskill",
        ),
    ),
    StageDefinition("hgb", group="analytics", depends_on=("metrics",), cache_scope=("io", "hgb")),
    StageDefinition(
        "screening",
        group="analytics",
        depends_on=("metrics", "game_stats", "trueskill", "hgb"),
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
        depends_on=("screening",),
        cache_scope=("screening", "head2head"),
    ),
    StageDefinition(
        "h2h_power",
        group="single_root_tail",
        depends_on=("candidate_freeze",),
        cache_scope=("head2head",),
    ),
    StageDefinition(
        "h2h_execute",
        group="single_root_tail",
        depends_on=("h2h_power",),
        cache_scope=("head2head", "rng"),
    ),
    StageDefinition(
        "h2h_inference",
        group="single_root_tail",
        depends_on=("h2h_execute",),
        cache_scope=("head2head",),
    ),
    StageDefinition(
        "h2h_digest",
        group="single_root_tail",
        depends_on=("h2h_inference",),
        cache_scope=("head2head",),
    ),
    StageDefinition(
        "agreement",
        group="single_root_tail",
        depends_on=("h2h_digest",),
        cache_scope=("artifact_contract",),
    ),
    StageDefinition(
        "reporting",
        group="single_root_tail",
        depends_on=("agreement",),
        cache_scope=("artifact_contract",),
    ),
)

_ROOT_PAIR_REGISTRY: tuple[StageDefinition, ...] = (
    StageDefinition(
        "root_stability",
        group="root_pair",
        cache_scope=(
            "sim.seed_list",
            "sim.n_players_list",
            "screening",
            "robustness",
            "k_aggregation",
        ),
    ),
    StageDefinition("trueskill", group="root_pair", cache_scope=("sim.seed_list", "trueskill")),
    StageDefinition(
        "candidate_freeze",
        group="root_pair",
        depends_on=("root_stability", "trueskill"),
        cache_scope=("screening", "head2head"),
    ),
    StageDefinition(
        "h2h_power",
        group="root_pair",
        depends_on=("candidate_freeze",),
        cache_scope=("head2head",),
    ),
    StageDefinition(
        "h2h_execute",
        group="root_pair",
        depends_on=("h2h_power",),
        cache_scope=("head2head", "rng"),
    ),
    StageDefinition(
        "h2h_inference",
        group="root_pair",
        depends_on=("h2h_execute",),
        cache_scope=("head2head",),
    ),
    StageDefinition(
        "h2h_digest",
        group="root_pair",
        depends_on=("h2h_inference",),
        cache_scope=("head2head",),
    ),
    StageDefinition(
        "agreement",
        group="root_pair",
        depends_on=("h2h_digest",),
        cache_scope=("artifact_contract",),
    ),
    StageDefinition(
        "reporting",
        group="root_pair",
        depends_on=("agreement",),
        cache_scope=("artifact_contract",),
    ),
)

_DEFINITION_LOOKUP: dict[str, StageDefinition] = {
    definition.key: definition for definition in (*_REGISTRY, *_ROOT_PAIR_REGISTRY)
}
_DEFINITION_LOOKUP["simulation"] = StageDefinition(
    "simulation",
    group="simulation",
    cache_key_version=4,
    cache_scope=(
        "sim.n_players_list",
        "sim.seed",
        "sim.seed_list",
        "sim.expanded_metrics",
        "sim.row_dir",
        "sim.metric_chunk_dir",
        "sim.score_thresholds",
        "sim.dice_thresholds",
        "sim.smart_five_opts",
        "sim.smart_one_opts",
        "sim.consider_score_opts",
        "sim.consider_dice_opts",
        "sim.auto_hot_dice_opts",
        "sim.run_up_score_opts",
        "sim.include_stop_at",
        "sim.include_stop_at_heuristic",
        "screening.resolution_delta",
        "screening.interval_confidence",
        "screening.max_shuffles_per_root_k",
        "batching.target_batches",
        "batching.min_shuffles_per_batch",
        "rng",
        "artifact_contract",
    ),
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
