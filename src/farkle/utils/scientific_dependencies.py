"""Scientific dependency catalog v1. Shared by publication, freshness, and prospective cross-commit comparison.

Whole modules cover method constants, worker entry points and ordered reductions. Shared
helpers are deliberately broad. The registry still owns stage names and legacy cache scopes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from farkle.config import AppConfig


@dataclass(frozen=True)
class Producer:
    scope: Literal["root", "pair"]
    stage: str
    partition: str | None = None


@dataclass(frozen=True)
class Dependencies:
    source_files: frozenset[str]
    config_scope: tuple[str, ...]
    distributions: frozenset[str]
    input_roles: frozenset[str]
    coordinate_fields: tuple[str, ...]

    def required_inputs(self, cfg: AppConfig) -> frozenset[str]:
        """Require both roots and every configured k for pair reductions of root cells."""
        roles = set(self.input_roles)
        for role in self.input_roles & {"root_batch_cells", "root_rating_cells"}:
            roles.remove(role)
            roles.update(
                f"{role}/{root}/{k}"
                for root in cfg.sim.require_seed_pair()
                for k in cfg.sim.n_players_list
            )
            if not cfg.sim.n_players_list:
                raise ValueError(f"Empty configured player-count support for {role}")
        return frozenset(roles)


def _modules(directory: str, names: str) -> set[str]:
    return {f"src/farkle/{directory}{name}.py" for name in names.split()}


# The runner, CLI and shared adapters can supply defaults, normalize strategies or
# dispatch workers dynamically. Hash their implementation even on a clean checkout.
_COMMON = (
    _modules("", "__init__ __main__ config protocols")
    | _modules("cli/", "__init__ launcher main protected preflight_worker")
    | _modules(
        "orchestration/",
        "__init__ execution_history profile_metadata run_contexts seed_utils two_seed_pipeline",
    )
    | _modules(
        "utils/",
        """__init__ aggregation analysis_shared arrow_batches artifact_contract
        artifacts authenticated_contract authenticated_graph authentication_telemetry
        completion_files copy_on_write csv_files evidence_reader frozen_h2h logging manifest os_memory parallel
        partitioned_stage progress random release_identity schema_helpers sinks source_snapshot
        stage_completion stats strategy_ids streaming_loop telemetry timing types writer yaml_helpers
        scientific_reuse scientific_dependencies scientific_identity scientific_runtime scientific_files scientific_invocation""",
    )
    | _modules("analysis/", "__init__ stage_registry stage_runner")
    | _modules("simulation/", "__init__ strategies")
    | _modules("game/", "__init__ engine scoring scoring_lookup")
)
# Standalone entry points outside scientific production.
_EXCLUDED = (
    _modules("simulation/", "time_farkle watch_game")
    | _modules("analysis/", "release_audit")
    | _modules("orchestration/", "resume_preview")
)

_ENGINE = _modules("simulation/", "simulation runner run_tournament workload_planner game_profile")
_RATING = _modules("analysis/", "trueskill run_trueskill trueskill_screening")
_METRICS = _modules(
    "analysis/",
    """metrics performance seat_analysis all_player_metrics batch_support
    checks combine outcome_distributions outcome_precision outcome_publication seat_behavior""",
)
_MODEL = _modules("analysis/", "run_hgb hgb_feat parameter_responses")
_H2H = _modules("analysis/", "h2h_schedule h2h_authorization") | _ENGINE
_INFERENCE = _H2H | _modules("analysis/", "h2h_inference")
PREFLIGHT_RECIPE = "docs/implementation/task5a_recipe.json"
PREFLIGHT_LOCK = "docs/implementation/task5b_protocol_lock.json"
_PREFLIGHT = {
    PREFLIGHT_RECIPE,
    PREFLIGHT_LOCK,
    "pytest.ini",
    "tests/conftest.py",
    "scripts/check_rng_contract.py",
    "scripts/task5b_capability.py",
    "configs/farkle_mega_config.yaml",
    "configs/validation/task5b_capability.json",
    "docs/implementation/task5b_protocol.md",
}
# A role covers a complete semantic inventory, including upstream science and
# declared order. reduction_layout includes physical row/batch boundaries when
# they can change floating-point grouping. Scheduling edges alone omit direct reads.
_STAGE_GROUPS: dict[str, tuple[set[str], str]] = {
    "simulation": (
        _ENGINE,
        "resolved_strategy_roster tournament_design",
    ),
    "ingest": (
        _ENGINE | _modules("analysis/", "ingest"),
        "row_shard_manifest workload_plan",
    ),
    "curate": (
        _modules("analysis/", "curate"),
        "ingested_rows ingest_manifest",
    ),
    "combine": (
        _modules("analysis/", "combine"),
        "curated_rows",
    ),
    "metrics": (
        _ENGINE | _METRICS,
        "combined_rows combined_manifest curated_rows workload_plans strategy_manifest reduction_layout",
    ),
    "game_stats": (
        _modules("analysis/", "game_stats roll_enumeration"),
        "curated_rows combined_manifest reduction_layout",
    ),
    "rng_diagnostics": (
        _ENGINE
        | _METRICS
        | _PREFLIGHT
        | _modules(
            "analysis/", "rng_diagnostics rng_evidence rng_monitor rng_preflight batch_monitor"
        ),
        "combined_rows combined_manifest outcome_cells workload_plans strategy_manifest effective_lags_and_mode preflight_suite preflight_environment",
    ),
    "trueskill": (
        _RATING,
        "ordered_curated_rows combined_manifest",
    ),
    "hgb": (
        _MODEL,
        "performance_cells strategy_manifest",
    ),
    "screening": (
        _modules("analysis/", "screening all_player_metrics"),
        "performance_by_k performance_across_k bootstrap",
    ),
    "root_stability": (
        _METRICS | _modules("analysis/", "root_stability"),
        "root_batch_cells reduction_layout",
    ),
    "candidate_freeze": (
        _RATING | _modules("analysis/", "candidate_family"),
        "win_rate_contribution trueskill_contribution",
    ),
    "h2h_power": (
        _H2H,
        "frozen_membership frozen_family_manifest strategy_manifest",
    ),
    "h2h_execute": (
        _H2H,
        "power_plan block_manifest strategy_manifest",
    ),
    "h2h_inference": (
        _INFERENCE,
        "order_counts power_plan block_manifest execution_counts",
    ),
    "h2h_digest": (
        _INFERENCE | _modules("analysis/", "dominance"),
        "pairwise_inference tournament_screening_scores",
    ),
    "agreement": (
        _INFERENCE | _modules("analysis/", "structure_agreement"),
        "frozen_membership pairwise_inference root_agreement",
    ),
    "reporting": (
        _INFERENCE
        | _MODEL
        | _METRICS
        | _modules(
            "analysis/",
            "analysis_guide guide_questions structure_reporting behavior_reporting migration_audit",
        ),
        "performance_cells rating_cells h2h_family h2h_results agreement behavior_cells rng_evidence guide_sources",
    ),
}
CLASSIFIED_PACKAGE_FILES = frozenset(
    _COMMON.union(_EXCLUDED, *(files for files, _ in _STAGE_GROUPS.values()))
)

# These leaf descriptors are resolved by the scoped producer's own implementation.
# They can start a lineage. Artifact inputs always carry their upstream science.
LOCAL_DESIGNS = frozenset(
    {
        "resolved_strategy_roster",
        "tournament_design",
        "effective_lags_and_mode",
        "reduction_layout",
        "unit_design",
    }
)


# Partition families inherit their owner's implementation/config envelope. Bind the
# durable unit design as well: e.g. row spans, route layout, bootstrap range, tile,
# merge level, or H2H root/pair/order. A generic 'analysis' stamp is ambiguous.
_PARTITIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "simulation": {"player_count": ("k",), "shuffle": ("k", "shuffle_index")},
    "ingest": {"ingest": ("k",)},
    "combine": {"combine": ("k",)},
    "metrics": {
        "performance_bootstrap": ("start", "stop"),
        "seat_mirrored_pairing": ("batch", "partition", "partitions"),
        # Ordered descriptors in the retained per-k plans own multi-step unit keys.
        "outcome_publication": ("k", "operation", "unit_ordinal"),
        "outcome_precision": ("k", "operation", "unit_ordinal"),
        "seat_behavior": ("k", "operation", "unit_ordinal"),
    },
    "rng_diagnostics": {
        "rng_diagnostics_count_route": ("start", "stop"),
        "rng_diagnostics_eligibility": ("partition",),
        "rng_diagnostics_stats_route": ("start", "stop"),
        "rng_diagnostics_stats": ("partition",),
    },
    "root_stability": {
        "root_stability_top_n": ("start", "stop"),
        "root_stability_joint_discrepancy": ("start", "stop"),
    },
    "h2h_execute": {"block": ("root_seed", "pair_id", "order")},
}

# The descriptor names below are generated by the three multi-pass partition writers.
_OPERATIONS = {
    "outcome_publication": r"source|certified_batches|normalized|(?:all_batch_merge|batch_merge)_[0-9]{3,}",
    "outcome_precision": r"vectors|\.|summaries",
    "seat_behavior": r"source|merge_[0-9]{4,}|certified_batches|summaries",
}


def classified_operation(producer: Producer, operation: int | str) -> bool:
    pattern = _OPERATIONS.get(producer.partition or "")
    return (
        isinstance(operation, str)
        and pattern is not None
        and re.fullmatch(pattern, operation) is not None
    )


# Public schema inventory. Adding a dataclass leaf needs an explicit disposition.
# Scoped scientific fields participate only where selected by the registry or the
# computational exceptions below. Dictionary-valued scientific fields bind all entries.
def _fields(section: str, names: str) -> set[str]:
    return {f"{section}.{name}" for name in names.split()}


SCIENTIFIC_FIELDS = frozenset(
    _fields(
        "sim",
        """seed seed_list n_players_list expanded_metrics row_dir metric_chunk_dir
        score_thresholds dice_thresholds smart_five_opts smart_one_opts consider_score_opts
        consider_dice_opts auto_hot_dice_opts run_up_score_opts include_stop_at
        include_stop_at_heuristic""",
    )
    | _fields(
        "analysis",
        """disable_rng_diagnostics game_stats_margin_thresholds
        rare_event_target_score rare_event_write_details rare_event_margin_quantile
        rare_event_target_rate rng_max_matchup_groups rng_diagnostic_lags rng_diagnostic_partitions""",
    )
    | _fields(
        "screening",
        """resolution_delta practical_delta_by_k delta_across_k
        bootstrap_replicates candidate_contribution_size controls mandatory_diagnostics
        max_shuffles_per_root_k""",
    )
    | _fields("batching", "target_batches min_shuffles_per_batch")
    | _fields("robustness", "delta_seed_stability joint_discrepancy_alpha matched_count_fractions")
    | _fields("k_aggregation", "method k_weights")
    | _fields("combine", "max_players")
    | _fields("trueskill", "beta tau draw_probability")
    | _fields(
        "head2head",
        """family_alpha target_power practical_delta sensitivity_deltas
        delta_equivalence candidate_cap min_candidate_completion_rate max_attempt_multiplier
        allow_single_root""",
    )
    | _fields(
        "hgb", "max_depth n_estimators heldout_folds permutation_repeats future_proposal_limit"
    )
)
OPERATIONAL_FIELDS = frozenset(
    _fields("io", "results_dir_prefix analysis_subdir temp_dir")
    | _fields("sim", "desired_sec_per_chunk ckpt_every_sec")
    | _fields("analysis", "log_level outputs")
    | _fields("ingest", "parquet_codec")
    | _fields("head2head", "total_game_cap")
    | _fields("screening", "projected_games_per_second")
    | set().union(
        *(
            _fields(
                f"{section}.progress_logging",
                "frequent_interval_sec info_phase_sec ongoing_interval_sec",
            )
            for section in ("sim", "analysis")
        )
    )
)
RESOURCE_FIELDS = frozenset(
    _fields("sim", "n_jobs mp_start_method")
    | _fields("analysis", "n_jobs mp_start_method")
    | _fields("ingest", "n_jobs row_group_size batch_rows")
    | _fields("head2head", "n_jobs")
    | _fields(
        "resources",
        """scheduler_memory_budget_mb process_tree_warning_threshold_mb
        aggregate_memory_hard_limit_mb minimum_system_available_memory_mb os_memory_limit_enabled
        os_memory_limit_required allow_unenforced_memory_fallback parent_process_memory_mb
        logical_cpu_budget native_threads_per_worker max_in_flight_per_worker
        rss_sample_interval_seconds estimated_worker_memory_mb stage_batch_bytes""",
    )
)
# Purpose is a release claim/label. Git, invocation and lifecycle are outside public
# configuration. Nested per-k overrides need an explicit effective-config adapter.
RELEASE_FIELDS = frozenset({"profile.purpose"})
UNRESOLVED_FIELDS = frozenset({"sim.per_n"})
# Computed fields on AppConfig and its run-context wrapper. Source lookup uses
# authenticated logical inputs. The numerical envelope covers effective threading.
PROVENANCE_FIELDS = frozenset({"config_sha", "_code_identity", "_run_lineage_sha256"})
RUNTIME_FIELDS = PROVENANCE_FIELDS | frozenset(
    {
        "_stage_layout",
        "_game_profile_sha256",
        "_resource_limiter_mode",
        "_analysis_root_override",
        "_source_root_configs",
        "_root_input_layout_override",
    }
)

# The legacy public hash drops execution sections while v3 expands raw sections
# (including io.temp_dir, ingest.n_jobs and head2head.n_jobs). Neither hash is this
# projection. Include extra leaves when an operational knob changes the computation.
_EXTRA_CONFIG = {
    "metrics": ("ingest.row_group_size",),
    "game_stats": ("resources.stage_batch_bytes.game_stats",),
    "rng_diagnostics": ("sim.n_jobs", "combine.max_players"),
    "hgb": ("resources.native_threads_per_worker",),
    "h2h_power": ("screening.max_shuffles_per_root_k",),
    "reporting": ("head2head.delta_equivalence", "screening.max_shuffles_per_root_k"),
}

# These producers establish a native worker limit as part of their execution policy.
MANAGED_NATIVE_STAGES = frozenset(
    {
        "simulation",
        "ingest",
        "metrics",
        "rng_diagnostics",
        "game_stats",
        "trueskill",
        "root_stability",
        "hgb",
        "h2h_power",
        "h2h_execute",
    }
)


# Narrow legacy whole-section scopes where the actual operation reads individual
# fields. Plan parameters travel with the frozen scientific input to execution.
_CONFIG_SCOPE_OVERRIDES = {
    "candidate_freeze": (
        "screening.candidate_contribution_size",
        "screening.controls",
        "screening.mandatory_diagnostics",
        "head2head.candidate_cap",
    ),
    "h2h_execute": (),
    "h2h_inference": (
        "head2head.family_alpha",
        "head2head.practical_delta",
        "head2head.delta_equivalence",
    ),
    "h2h_digest": (),
}


def stage_producer(cfg: AppConfig, stage: str) -> Producer:
    """Use the owning layout for duplicate names. Simulation always belongs to a root."""
    from farkle.analysis.stage_registry import resolve_scoped_stage_definition

    if stage == "simulation":
        return Producer("root", stage)
    for placement in cfg.stage_layout.placements:
        if placement.definition.key == stage:
            return Producer("pair" if placement.definition.group == "root_pair" else "root", stage)
    # Inactive stages may still be queried by tooling. A unique owner is unambiguous.
    owners = []
    scopes: tuple[Literal["root", "pair"], ...] = ("root", "pair")
    for scope in scopes:
        try:
            resolve_scoped_stage_definition(scope, stage)
        except KeyError:
            continue
        owners.append(Producer(scope, stage))
    if len(owners) != 1:
        raise ValueError(f"Cannot resolve inactive or unknown producer {stage!r} in this layout")
    return owners[0]


def dependencies_for(producer: Producer) -> Dependencies:
    """Resolve the catalog for one scoped producer. Unknown keys have no fallback."""
    from farkle.analysis.stage_registry import resolve_scoped_stage_definition

    stage = producer.stage
    definition = resolve_scoped_stage_definition(producer.scope, stage)
    files, roles = _STAGE_GROUPS[stage]
    if PREFLIGHT_RECIPE in files:
        # The locked pytest recipe exercises other producers, including a raw
        # pipeline oracle and a repository-wide RNG source audit.
        files = set(CLASSIFIED_PACKAGE_FILES)
    inputs = set(roles.split())
    if producer.scope == "pair" and stage == "trueskill":
        inputs = {"root_rating_cells"}
    if producer.scope == "pair" and stage == "reporting":
        inputs.add("root_stability")
    coordinates: tuple[str, ...] = ()
    if producer.partition is not None:
        families = _PARTITIONS.get(stage, {})
        if producer.partition not in families:
            raise KeyError(f"Unknown partition {producer!r}")
        coordinates = families[producer.partition]
        inputs.add("unit_design")
        if stage != "simulation":
            inputs.add("unit_sources")

    config_scope = _CONFIG_SCOPE_OVERRIDES.get(stage, definition.cache_scope)
    fields = {
        field
        for field in SCIENTIFIC_FIELDS
        if any(field == scope or field.startswith(scope + ".") for scope in config_scope)
    }
    fields.update(("sim.seed", "sim.seed_list", "sim.n_players_list"))
    fields.update(_EXTRA_CONFIG.get(stage, ()))
    # In addition to the compiled build/dispatch identity supplied by the caller.
    libraries = {"numpy", "pandas", "pyarrow", "scipy", "numba", "llvmlite", "threadpoolctl"}
    if "src/farkle/analysis/h2h_inference.py" in files:
        libraries.add("statsmodels")
    if files & _RATING:
        libraries.add("trueskill")
    if files & _MODEL or stage == "agreement":
        libraries.add("scikit-learn")
    if files & _MODEL or stage == "reporting":
        libraries.add("matplotlib")
    if PREFLIGHT_RECIPE in files:
        # The preflight loads tests/conftest.py and the materialized YAML overlay.
        libraries.update(("psutil", "pytest", "freezegun", "matplotlib", "PyYAML"))
    return Dependencies(
        frozenset(_COMMON | files),
        tuple(sorted(fields)),
        frozenset(libraries),
        frozenset(inputs),
        coordinates,
    )
