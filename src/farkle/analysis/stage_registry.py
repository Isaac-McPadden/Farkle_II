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
    cache_scope: tuple[str, ...] = ()
    cache_key_version: int = 2
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
    StageDefinition(
        "ingest",
        group="pipeline",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.seed_list",
            "sim.seed_pair",
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
            "metrics",
            "analysis.pooling_weights",
            "analysis.pooling_weights_by_k",
            "analysis.n_jobs",
        ),
    ),
    StageDefinition(
        "coverage_by_k",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.seed_list",
            "sim.n_players_list",
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
            "analysis.outputs",
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
            "sim.seed_pair",
            "sim.n_players_list",
            "analysis.pooling_weights",
            "analysis.pooling_weights_by_k",
            "analysis.rare_event_target_score",
            "analysis.rare_event_write_details",
            "analysis.rare_event_margin_quantile",
            "analysis.rare_event_target_rate",
            "analysis.n_jobs",
        ),
    ),
    StageDefinition(
        "seed_summaries",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "analysis.pooling_weights",
            "analysis.pooling_weights_by_k",
        ),
    ),
    StageDefinition(
        "trueskill",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.seed_list",
            "sim.seed_pair",
            "sim.n_players_list",
            "trueskill",
        ),
    ),
    StageDefinition(
        "tiering",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.n_players_list",
            "analysis.tiering_seeds",
            "analysis.tiering_z_star",
            "analysis.tiering_min_gap",
            "analysis.tiering_weights_by_k",
        ),
    ),
    StageDefinition(
        "head2head",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.n_players_list",
            "analysis.head2head_target_hours",
            "analysis.head2head_tolerance_pct",
            "analysis.head2head_games_per_sec",
            "analysis.head2head_force_calibrate",
            "analysis.tiering_z_star",
            "analysis.tiering_min_gap",
            "head2head",
            "trueskill.pooled_weights_by_k",
        ),
    ),
    StageDefinition("seed_symmetry", group="analytics", cache_scope=("io",)),
    StageDefinition(
        "post_h2h",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "head2head.tie_break_policy",
            "head2head.tie_break_seed",
            "head2head.fdr_q",
            "head2head.bonferroni_design",
            "trueskill.pooled_weights_by_k",
        ),
    ),
    StageDefinition(
        "hgb",
        group="analytics",
        cache_scope=("io", "hgb"),
    ),
    StageDefinition("variance", group="analytics", cache_scope=("io",)),
    StageDefinition(
        "meta",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.seed_list",
            "sim.seed_pair",
            "analysis.meta_random_if_I2_gt",
            "analysis.meta_max_other_seeds",
            "analysis.meta_comparison_seed",
        ),
    ),
    StageDefinition(
        "h2h_tier_trends",
        group="analytics",
        cache_scope=(
            "io",
            "sim.n_players_list",
            "analysis.h2h_tier_trends_seed_s_tier_paths",
            "analysis.h2h_tier_trends_interseed_s_tier_path",
        ),
    ),
    StageDefinition(
        "agreement",
        group="analytics",
        cache_scope=("io", "analysis.agreement_strategies", "analysis.agreement_include_pooled", "analysis.n_jobs"),
    ),
    StageDefinition(
        "interseed",
        group="analytics",
        cache_scope=("io", "sim.seed_list", "sim.seed_pair", "analysis.disable_rng_diagnostics"),
    ),
)

_INTERSEED_REGISTRY: tuple[StageDefinition, ...] = (
    StageDefinition(
        "rng_diagnostics",
        group="analytics",
        folder_stub="rng",
        cache_scope=(
            "io",
            "sim.n_players_list",
            "analysis.disable_rng_diagnostics",
            "analysis.rng_max_matchup_groups",
            "analysis.n_jobs",
        ),
        disabled_predicate=lambda cfg: cfg.analysis.disable_rng_diagnostics,
    ),
    StageDefinition("variance", group="analytics", cache_scope=("io",)),
    StageDefinition(
        "interseed_game_stats",
        group="analytics",
        folder_stub="interseed_game_stats",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.seed_list",
            "sim.seed_pair",
            "sim.n_players_list",
            "analysis.pooling_weights",
            "analysis.pooling_weights_by_k",
            "analysis.rare_event_target_score",
            "analysis.rare_event_write_details",
            "analysis.rare_event_margin_quantile",
            "analysis.rare_event_target_rate",
            "analysis.n_jobs",
        ),
    ),
    StageDefinition(
        "meta",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.seed_list",
            "sim.seed_pair",
            "analysis.meta_random_if_I2_gt",
            "analysis.meta_max_other_seeds",
            "analysis.meta_comparison_seed",
        ),
    ),
    StageDefinition(
        "trueskill",
        group="analytics",
        cache_scope=(
            "io",
            "sim.seed",
            "sim.seed_list",
            "sim.seed_pair",
            "sim.n_players_list",
            "trueskill",
        ),
    ),
    StageDefinition(
        "agreement",
        group="analytics",
        cache_scope=("io", "analysis.agreement_strategies", "analysis.agreement_include_pooled", "analysis.n_jobs"),
    ),
    StageDefinition(
        "interseed",
        group="analytics",
        cache_scope=("io", "sim.seed_list", "sim.seed_pair", "analysis.disable_rng_diagnostics"),
    ),
    StageDefinition(
        "h2h_tier_trends",
        group="analytics",
        cache_scope=(
            "io",
            "sim.n_players_list",
            "analysis.h2h_tier_trends_seed_s_tier_paths",
            "analysis.h2h_tier_trends_interseed_s_tier_path",
        ),
    ),
)

_DEFINITION_LOOKUP: dict[str, StageDefinition] = {
    definition.key: definition for definition in (*_REGISTRY, *_INTERSEED_REGISTRY)
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


def resolve_stage_definition(key: str) -> StageDefinition:
    """Return the canonical definition for ``key``."""

    try:
        return _DEFINITION_LOOKUP[key]
    except KeyError as exc:
        raise KeyError(f"Unknown stage key {key!r}") from exc
