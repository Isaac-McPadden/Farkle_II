# src/farkle/config.py
"""Configuration schemas and helpers for the Farkle analysis pipeline.

Defines dataclasses describing I/O, simulation, and analysis settings and
includes utilities for loading and validating YAML-based application configs.
"""
from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml  # type: ignore[import-untyped]

from farkle.utils.progress import ProgressLogConfig
from farkle.utils.types import Compression, normalize_compression
from farkle.utils.yaml_helpers import expand_dotted_keys

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from farkle.analysis.stage_registry import StageLayout


LOGGER = logging.getLogger(__name__)


class ArtifactScope(str, Enum):
    """Canonical logical scopes for derived analysis artifacts."""

    BY_K = "by_k"
    CONCAT_KS = "concat_ks"
    ACROSS_K = "across_k"
    CROSS_SEED = "cross_seed"
    DIAGNOSTICS = "diagnostics"
    H2H_2P = "h2h_2p"

    @property
    def requires_player_count(self) -> bool:
        """Return whether this scope requires a concrete player count."""

        return self is ArtifactScope.BY_K

# Deprecated analysis-stage flags that no longer disable stages.
DEPRECATED_ANALYSIS_FLAGS = {
    "run_interseed",
    "disable_game_stats",
    "disable_trueskill",
    "disable_head2head",
    "disable_hgb",
    "disable_frequentist",
    "disable_agreement",
    "run_trueskill",
    "run_head2head",
    "run_rng",
    "run_game_stats",
    "run_hgb",
    "run_frequentist",
    "run_post_h2h_analysis",
    "run_agreement",
    "run_report",
}

_CANONICAL_ARTIFACT_NAMES: dict[str, str] = {
    "ratings_combined.parquet": "ratings_k_weighted.parquet",
    "ratings_combined.json": "ratings_k_weighted.json",
    "game_length_combined.parquet": "game_length_k_weighted.parquet",
    "margin_combined.parquet": "margin_k_weighted.parquet",
    "frequentist_scores.parquet": "frequentist_scores_k_weighted.parquet",
    "frequentist_combined_provenance.json": "frequentist_k_weighted_provenance.json",
    "agreement_combined.json": "agreement_k_weighted.json",
}

RETIRED_CONFIG_KEYS: dict[str, str] = {
    "sim.power_method": "screening.resolution_delta",
    "sim.recompute_num_shuffles": "screening.resolution_delta",
    "sim.power_design": "screening and head2head settings",
    "analysis.tier_z_star": "screening bootstrap summaries",
    "analysis.tier_min_gap": "screening practical_delta_by_k",
    "analysis.frequentist_weights_by_k": "k_aggregation.k_weights",
    "analysis.meta_random_if_I2_gt": "cross-seed stability diagnostics",
    "analysis.meta_max_other_seeds": "sim.seed_list",
    "analysis.meta_comparison_seed": "sim.seed_list",
    "analysis.pooling_weights": "k_aggregation.method",  # terminology-allow: legacy-config
    "analysis.pooling_weights_by_k": "k_aggregation.k_weights",  # terminology-allow: legacy-config
    "analysis.k_aggregation_method": "k_aggregation.method",
    "analysis.k_weights": "k_aggregation.k_weights",
    "analysis.agreement_include_combined": "analysis.agreement_include_across_k",
    "trueskill.pooled_weights_by_k": "trueskill.k_weights",  # terminology-allow: legacy-config
    "head2head.fdr_q": "head2head.family_alpha",
    "head2head.bonferroni_total_games_safeguard": "head2head.total_game_cap",
    "head2head.bonferroni_design": "typed head2head settings",
}

_LEGACY_ARTIFACT_NAMES: dict[str, tuple[str, ...]] = {
    canonical: (legacy,) for legacy, canonical in _CANONICAL_ARTIFACT_NAMES.items()
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses (schema)
# ─────────────────────────────────────────────────────────────────────────────

SEED_LIST_LENGTHS_BY_COMMAND: dict[str, int] = {
    "run": 1,
    "analyze": 1,
    "two-seed": 2,
    "two-seed-pipeline": 2,
}


def expected_seed_list_length(
    command: str,
    *,
    subcommand: str | None = None,
    _subcommand: str | None = None,
) -> int | None:
    """Return the expected seed-list length for a CLI command.

    ``subcommand`` is retained as a compatibility alias for older callers.
    """

    if subcommand is not None and _subcommand is not None and subcommand != _subcommand:
        raise ValueError("subcommand and _subcommand must match when both are provided")
    return SEED_LIST_LENGTHS_BY_COMMAND.get(command)


@dataclass
class IOConfig:
    """File-system locations for the application."""

    results_dir_prefix: Path = Path("results")
    analysis_subdir: str = "analysis"
    meta_analysis_dir: Path | None = None
    interseed_input_dir: Path | None = None
    interseed_input_layout: "StageLayout | Mapping[str, str] | None" = None


@dataclass
class PowerDesign:
    """Parameters controlling power analyses for tournament planning."""

    power: float = 0.8
    control: float = 0.1  # fdr_q (BH - FDR) or alpha (Bonferroni - FWER)
    detectable_lift: float = 0.03  # absolute lift in win-rate
    baseline_rate: float = 0.50
    tail: str = "two_sided"  # "one_sided" | "two_sided"
    full_pairwise: bool = True
    endpoint: str = "pairwise"  # "pairwise" | "top1"
    min_games_floor: int = 2000
    max_games_cap: int | None = None
    use_BY: bool | None = False  # if true and using BH, use q/H_m (more conservative)
    bh_target_rank: int | None = None  # target BH order statistic (i*), optional
    bh_target_frac: float | None = 0.03  # target fraction of discoveries for BH planning


@dataclass
class RNGConfig:
    """Versioned deterministic random-stream contract."""

    scheme_version: int = 1
    bit_generator: str = "PCG64DXSM"


@dataclass
class ScreeningConfig:
    """Broad tournament resolution and candidate-screening settings."""

    resolution_delta: float = 0.03
    interval_confidence: float = 0.95
    practical_delta_by_k: dict[int, float] | None = None
    delta_across_k: float | None = None
    bootstrap_replicates: int = 2_000
    candidate_contribution_size: int = 75
    controls: list[int] = field(default_factory=list)
    mandatory_diagnostics: list[int] = field(default_factory=list)


@dataclass
class BatchingConfig:
    """Deterministic shuffle-batch construction settings."""

    target_batches: int = 100
    min_shuffles_per_batch: int = 30


@dataclass
class RobustnessConfig:
    """Finite-grid robustness summaries to emit."""

    report_pareto: bool = True
    report_maximin: bool = True


@dataclass
class ArtifactContractConfig:
    """Versions participating in artifact validation and cache freshness."""

    artifact_contract_version: int = 1
    estimand_version: int = 1
    schema_version: int = 1


@dataclass
class KAggregationConfig:
    """Declared player-count aggregation contract."""

    method: str = "equal-k"
    k_weights: dict[int, float] | None = None


@dataclass
class SimConfig:
    """Simulation parameters.

    ``seed_list`` is the canonical seed container. Single-seed commands (run,
    analyze, pipeline) require exactly one seed; two-seed orchestration requires
    exactly two seeds. ``seed`` remains the legacy primary seed for single-seed
    workflows and for naming seed-suffixed results directories. ``seed_pair`` is
    retained for compatibility with two-seed orchestration configs.
    """

    n_players_list: list[int] = field(default_factory=lambda: [5])
    num_shuffles: int = 100
    seed: int = 0
    seed_list: list[int] | None = None
    """Explicit seed list (len 1 for single-seed, len 2 for two-seed)."""
    seed_pair: tuple[int, int] | None = None
    """Legacy two-seed tuple for dual-seed orchestration (validated on load)."""
    expanded_metrics: bool = False
    row_dir: Path | None = None
    metric_chunk_dir: Path | None = None
    per_n: dict[int, "SimConfig"] = field(default_factory=dict)
    power_method: str = "bh"
    recompute_num_shuffles: bool = True
    power_design: PowerDesign = field(default_factory=PowerDesign)
    n_jobs: int | None = None
    """Parallel worker setting (`None` => 1, `0` => os.cpu_count(), `>0` => explicit)."""
    mp_start_method: str | None = None
    """Multiprocessing start method for simulation executors."""
    desired_sec_per_chunk: int = 10
    ckpt_every_sec: int = 30
    progress_logging: ProgressLogConfig = field(default_factory=ProgressLogConfig)

    # Alter strategy grid
    score_thresholds: list[int] | None = None
    dice_thresholds: list[int] | None = None
    smart_five_opts: Sequence[bool] | None = None
    smart_one_opts: Sequence[bool] | None = None
    consider_score_opts: Sequence[bool] = (True, False)
    consider_dice_opts: Sequence[bool] = (True, False)
    auto_hot_dice_opts: Sequence[bool] = (True, False)
    run_up_score_opts: Sequence[bool] = (True, False)
    include_stop_at: bool = False
    include_stop_at_heuristic: bool = False

    def interseed_seed_list(self) -> list[int] | None:
        """Return the two seeds used for interseed analysis, if configured."""
        if self.seed_list is not None:
            return list(self.seed_list) if len(self.seed_list) == 2 else None
        if self.seed_pair is not None:
            return list(self.seed_pair)
        return None

    def resolve_seed_list(self, expected_len: int) -> list[int]:
        """Resolve a seed list of ``expected_len`` from available seed settings."""
        if expected_len < 1:
            raise ValueError("expected_len must be >= 1")
        if self.seed_list is not None:
            if len(self.seed_list) != expected_len:
                raise ValueError(
                    f"sim.seed_list must contain exactly {expected_len} seeds, "
                    f"got {self.seed_list!r}"
                )
            return list(self.seed_list)
        if expected_len == 1:
            return [self.seed]
        if expected_len == 2:
            if self.seed_pair is None:
                raise ValueError(
                    "sim.seed_list or sim.seed_pair must be set for two-seed orchestration"
                )
            return list(self.seed_pair)
        raise ValueError(f"Unsupported expected seed length {expected_len}")

    def populate_seed_list(self, expected_len: int) -> list[int]:
        """Populate ``seed_list`` based on current config and return it."""
        if self.seed_list is not None and self.seed_pair is not None and list(self.seed_pair) != list(self.seed_list):
            raise ValueError(
                "sim.seed_list and sim.seed_pair must match when both are set"
            )
        seeds = self.resolve_seed_list(expected_len)
        self.seed_list = list(seeds)
        if expected_len == 1:
            self.seed = seeds[0]
        elif expected_len == 2:
            if self.seed_pair is None:
                self.seed_pair = (seeds[0], seeds[1])
            if self.seed != seeds[0]:
                self.seed = seeds[0]
        return seeds

    def require_seed_pair(self) -> tuple[int, int]:
        """Return ``seed_pair`` or raise if two-seed orchestration is requested."""
        if self.seed_list is not None:
            if len(self.seed_list) != 2:
                raise ValueError(
                    f"sim.seed_list must contain exactly two seeds, got {self.seed_list!r}"
                )
            return (self.seed_list[0], self.seed_list[1])
        if self.seed_pair is None:
            raise ValueError("sim.seed_pair must be set for two-seed orchestration")
        return self.seed_pair


@dataclass
class AnalysisConfig:
    """Analysis-stage parameters controlling downstream analytics.

    Deprecated ``run_*`` and ``disable_*`` toggles are retained for legacy configs,
    but no longer disable stages (except ``disable_rng_diagnostics``). Stages now
    run (or skip) based on inputs and preconditions instead of manual toggles.
    """

    run_interseed: bool = True
    """Deprecated: interseed stages are scheduled based on available inputs."""

    # Deprecated disable_* flags ignored in favor of precondition checks.
    disable_game_stats: bool = False
    disable_trueskill: bool = False
    disable_head2head: bool = False
    disable_hgb: bool = False
    disable_frequentist: bool = False
    disable_agreement: bool = False

    disable_rng_diagnostics: bool = False
    """Disable RNG diagnostics even when interseed analytics run by default."""

    # Deprecated run_* toggles retained for legacy configs (ignored).
    run_trueskill: bool = True
    run_head2head: bool = True
    run_rng: bool = True
    run_game_stats: bool = True
    run_hgb: bool = True
    run_frequentist: bool = True
    """Plan step 6: per-seed frequentist ranking and tier reporting."""
    run_post_h2h_analysis: bool = True
    """Execute the post head-to-head clean-up pass (plan step 5)."""

    run_agreement: bool = True
    """Generate the agreement analysis between model outputs (plan step 8)."""

    agreement_strategies: tuple[str, ...] | None = None
    """Optional subset of strategies to include when computing agreement metrics."""

    agreement_include_across_k: bool = False
    """Whether agreement analysis should emit a declared cross-k comparison payload."""

    run_report: bool = True
    """Emit the final report artifacts (plan step 9)."""

    n_jobs: int = 1
    """Parallel worker setting (`0` => os.cpu_count(), `>0` => explicit)."""
    mp_start_method: str | None = None
    """Multiprocessing start method for analysis executors."""
    progress_logging: ProgressLogConfig = field(default_factory=ProgressLogConfig)
    log_level: str = "INFO"
    results_glob: str = "*_players"
    meta_random_if_I2_gt: float = 25.0
    """Switch to random-effects aggregation once I^2 crosses this threshold."""
    meta_max_other_seeds: int | None = None
    """Optional cap on comparison seeds (excluding the current sim.seed)."""
    meta_comparison_seed: int | None = None
    """Optional fixed comparison seed when limiting meta analysis partners."""
    head2head_target_hours: float = 8.0
    """Target runtime (in hours) for head-to-head autotuning; <=0 disables the feature."""

    head2head_tolerance_pct: float = 5.0
    """Allowed +/- percent variance when computing the target head-to-head runtime."""

    head2head_games_per_sec: float | None = None
    """Measured throughput override for head-to-head simulations (games/second)."""
    head2head_force_calibrate: bool = False
    frequentist_seeds: list[int] | None = None
    """Explicit seeds to use when running the frequentist ranking stage."""

    tier_z_star: float = 1.645
    """Z-threshold used when building human-readable tier bands."""
    tier_min_gap: float | None = None
    """Optional minimum absolute gap required to split adjacent tiers."""
    frequentist_weights_by_k: dict[int, float] | None = None
    """Optional per-player-count weights for combined frequentist scores."""
    # Optional outputs block may be provided in YAML
    # outputs:
    #   curated_rows_name: "game_rows.parquet"
    #   metrics_name: "metrics.parquet"
    #   manifest_name: "manifest.jsonl"
    outputs: dict[str, Any] = field(default_factory=dict)
    game_stats_margin_thresholds: tuple[int, ...] = (500, 1000)
    """Victory-margin thresholds used by game stats and rare-event summaries."""
    k_aggregation_method: str = "game-count"
    """Weighting scheme for combined game stats across player counts."""
    k_weights: dict[int, float] | None = None
    """Optional per-player-count weights for combined game-stat summaries."""
    rare_event_target_score: int = 10_000
    """Score threshold used to flag games where multiple players crossed the target."""
    rare_event_write_details: bool = False
    """Write per-game rare-event rows to a separate details parquet."""
    rare_event_margin_quantile: float | None = None
    """Optional quantile for margin-of-victory rare-event thresholds."""
    rare_event_target_rate: float | None = None
    """Optional target rate for multi-target rare-event thresholds."""
    rng_max_matchup_groups: int | None = 100_000
    """Cap matchup-strategy group states in RNG diagnostics to bound memory use."""

    h2h_tier_trends_seed_s_tier_paths: list[Path] | None = None
    """Optional explicit per-seed S-tier JSON paths for h2h tier trends."""

    h2h_tier_trends_interseed_s_tier_path: Path | None = None
    """Optional interseed-combined S-tier JSON path for h2h tier trends."""

@dataclass
class IngestConfig:
    """Ingestion tuning for streaming parquet writes."""

    row_group_size: int = 64_000
    parquet_codec: Compression = "snappy"
    batch_rows: int = 100_000
    n_jobs: int = 1
    """Parallel worker setting (`0` => os.cpu_count(), `>0` => explicit)."""


@dataclass
class CombineConfig:
    """Settings for merging per-player-count ingested data."""

    max_players: int = 12


@dataclass
class MetricsConfig:
    """Metric computation options."""

    seat_range: tuple[int, int] = (1, 12)


@dataclass
class TrueSkillConfig:
    """Hyperparameters for TrueSkill updates."""

    beta: float = 25.0
    tau: float = 0.1
    draw_probability: float = 0.0
    k_weights: dict[int, float] | None = None
    """Optional per-player-count weights for aggregation TrueSkill ratings."""


@dataclass
class Head2HeadConfig:
    """Configuration for head-to-head tournament simulations."""

    n_jobs: int = 4
    """Parallel worker setting (`0` => os.cpu_count(), `>0` => explicit)."""
    family_alpha: float = 0.02
    target_power: float = 0.80
    practical_delta: float = 0.03
    sensitivity_deltas: tuple[float, ...] = (0.03, 0.04)
    seat1_advantage_scenarios: tuple[float, ...] = (0.0, 0.03, 0.06)
    delta_equivalence: float | None = None
    candidate_cap: int | None = None
    candidate_cap_policy: str = "balanced-tail"
    total_game_cap: int | None = 100_000_000
    allow_single_root: bool = True

    # Transitional implementation fields. Config files may not set these;
    # the owning stages remove them when their replacements land.
    games_per_pair: int = 10_000
    fdr_q: float = 0.02
    bonferroni_total_games_safeguard: int | None = 100_000_000
    """Skip Bonferroni H2H when estimated total games exceed this (<=0 disables)."""
    # If you ever add a nested design block here, it will still parse:
    bonferroni_design: dict[str, Any] = field(default_factory=dict)
    tie_break_policy: str = "neutral_edge"
    """Strategy for handling tied win counts in post head-to-head analysis."""

    tie_break_seed: int | None = None
    """Optional RNG seed for deterministic tie-break simulation (defaults to sim.seed)."""

    use_tier_elites: bool = False
    """Deprecated compatibility flag; H2H always uses the combined ratings/frequentist union."""


@dataclass
class HGBConfig:
    """Hyperparameters for histogram-based gradient boosting models."""

    max_depth: int = 6
    n_estimators: int = 300


@dataclass
class OrchestrationConfig:
    """Controls top-level orchestration behavior."""

    parallel_seeds: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# AppConfig + convenience properties used by analysis code
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AppConfig:
    """Top-level configuration container."""

    io: IOConfig = field(default_factory=IOConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    combine: CombineConfig = field(default_factory=CombineConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    trueskill: TrueSkillConfig = field(default_factory=TrueSkillConfig)
    head2head: Head2HeadConfig = field(default_factory=Head2HeadConfig)
    hgb: HGBConfig = field(default_factory=HGBConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    rng: RNGConfig = field(default_factory=RNGConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    artifact_contract: ArtifactContractConfig = field(default_factory=ArtifactContractConfig)
    k_aggregation: KAggregationConfig = field(default_factory=KAggregationConfig)
    # Computed at runtime; not part of user-provided YAML
    config_sha: str | None = field(default=None, init=False, repr=False, compare=False)
    _stage_layout: "StageLayout | None" = field(
        default=None, init=False, repr=False, compare=False
    )

    # —— Paths ——
    @property
    def results_root(self) -> Path:
        """Root directory where simulation outputs are written."""
        base = Path(self.io.results_dir_prefix)
        if not base.is_absolute():
            base = Path("data") / base
        seed_suffix = f"_seed_{self.sim.seed}"
        if base.name.endswith(seed_suffix):
            return base
        return base.parent / f"{base.name}{seed_suffix}"

    @property
    def results_dir(self) -> Path:
        """Deprecated alias for :meth:`results_root`."""
        return self.results_root

    @property
    def analysis_dir(self) -> Path:
        """Directory containing derived analysis artifacts."""
        return self.results_root / self.io.analysis_subdir

    # Numbered analysis stage directories (created on access)
    @property
    def stage_layout(self) -> "StageLayout":
        """Resolved :class:`~farkle.analysis.stage_registry.StageLayout`."""

        if self._stage_layout is None:
            from farkle.analysis.stage_registry import resolve_stage_layout

            self._stage_layout = resolve_stage_layout(cast("AppConfig", self))
        return self._stage_layout

    def stage_cache_key_version(self, stage_key: str) -> int:
        """Return the cache schema version for ``stage_key``."""

        from farkle.analysis.stage_registry import resolve_stage_definition

        return int(resolve_stage_definition(stage_key).cache_key_version)

    def stage_config_sha(self, stage_key: str) -> str:
        """Return the stage-scoped cache hash for ``stage_key``."""

        return compute_stage_config_sha(self, stage_key)

    def stage_cache_meta(self, stage_key: str) -> tuple[str, int]:
        """Return ``(stage_config_sha, cache_key_version)`` for ``stage_key``."""

        return (self.stage_config_sha(stage_key), self.stage_cache_key_version(stage_key))

    def set_stage_layout(self, layout: "StageLayout") -> None:
        """Override the resolved stage layout (used by CLI orchestration)."""

        if (
            self.io.interseed_input_dir is not None
            and self.io.interseed_input_layout is None
            and self._stage_layout is not None
        ):
            self.io.interseed_input_layout = self._stage_layout
        self._stage_layout = layout

    def validate_statistical_contract(self, *, require_two_roots: bool = False) -> None:
        """Validate locked production settings before expensive work is scheduled."""

        _validate_statistical_contract(self, require_two_roots=require_two_roots)

    def resolve_stage_dir(
        self,
        key: str,
        *,
        allow_missing: bool = False,
        required_by: str | None = "analysis pipeline",
        create: bool = True,
    ) -> Path:
        """Return the resolved stage directory for ``key`` with optional fallback."""

        folder = self.stage_layout.folder_for(key)
        if folder is None:
            if not allow_missing:
                requirement = f" required by {required_by}" if required_by else ""
                raise KeyError(
                    f"Stage {key!r} is not active in the resolved layout{requirement}."
                )
            stage_root = self.analysis_dir / key
            if create and not allow_missing:
                stage_root.mkdir(parents=True, exist_ok=True)
        else:
            stage_root = self.analysis_dir / folder
            if create:
                stage_root.mkdir(parents=True, exist_ok=True)
        return stage_root

    def stage_dir(self, key: str, *, required_by: str | None = "analysis pipeline") -> Path:
        """Return the resolved stage directory for ``key`` and create it."""

        folder = self.stage_layout.folder_for(key)
        if folder is None:
            requirement = f" required by {required_by}" if required_by else ""
            raise KeyError(
                f"Stage {key!r} is not registered in the resolved layout{requirement}."
            )
        stage_root = self.analysis_dir / folder
        stage_root.mkdir(parents=True, exist_ok=True)
        return stage_root

    def stage_dir_if_active(self, key: str, *parts: str | Path) -> Path | None:
        """Return the resolved stage directory for ``key`` when active."""

        return self._stage_dir_if_active(key, *parts)

    def interseed_ready(self) -> tuple[bool, str]:
        """Return whether interseed inputs are configured and an optional reason."""

        if self.sim.interseed_seed_list() is not None:
            return True, ""
        if self.interseed_input_dir is not None:
            return True, ""
        return (
            False,
            "interseed inputs missing (requires sim.seed_list with two seeds or io.interseed_input_dir)",
        )

    def stage_subdir(self, key: str, *parts: str | Path) -> Path:
        """Resolve a stage root or nested subdirectory under ``analysis_dir``.

        Directories are created on access to keep downstream callers simple.
        """

        stage_root = self.stage_dir(key)
        path = stage_root.joinpath(*map(Path, parts)) if parts else stage_root
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def scope_dir(
        self,
        stage: str,
        scope: ArtifactScope | str,
        *,
        k: int | None = None,
        create: bool = True,
    ) -> Path:
        """Resolve one canonical logical scope beneath an active stage.

        ``by_k`` is the only scope that accepts a player count. Requiring it in
        the API prevents the former all-k integer sentinel from entering paths.
        """

        parts = self._scope_parts(scope, k=k)
        path = self.stage_dir(stage).joinpath(*parts)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _scope_parts(
        scope: ArtifactScope | str,
        *,
        k: int | None,
    ) -> tuple[str, ...]:
        """Return validated path components for one canonical scope."""

        resolved_scope = ArtifactScope(scope)
        if resolved_scope.requires_player_count:
            if isinstance(k, bool) or k is None or int(k) < 1:
                raise ValueError("by_k scope requires a concrete positive player count")
            return (resolved_scope.value, f"{int(k)}p")
        else:
            if k is not None:
                raise ValueError(f"{resolved_scope.value} scope does not accept a player count")
            return (resolved_scope.value,)

    def scope_path(
        self,
        stage: str,
        scope: ArtifactScope | str,
        filename: str | Path,
        *,
        k: int | None = None,
        create_parent: bool = True,
    ) -> Path:
        """Resolve an artifact path under a canonical stage scope."""

        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact filename must remain within its canonical scope")
        path = self.scope_dir(stage, scope, k=k, create=create_parent) / relative
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def input_scope_path(
        self,
        stage: str,
        scope: ArtifactScope | str,
        filename: str | Path,
        *,
        k: int | None = None,
    ) -> Path:
        """Resolve one canonical scoped input from the declared input layout."""

        scope_parts = self._scope_parts(scope, k=k)
        input_scope_root = self._input_stage_path(stage, *scope_parts)
        if input_scope_root is None:
            input_scope_root = self.scope_dir(stage, scope, k=k, create=False)

        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact filename must remain within its canonical scope")
        return input_scope_root / relative

    def require_scope(
        self,
        path: Path,
        *,
        stage: str,
        scope: ArtifactScope | str,
        k: int | None = None,
    ) -> Path:
        """Return ``path`` only when it belongs to the declared stage scope."""

        expected_root = self.scope_dir(stage, scope, k=k, create=False)
        try:
            path.resolve().relative_to(expected_root.resolve())
        except ValueError as exc:
            resolved_scope = ArtifactScope(scope)
            detail = f"{resolved_scope.value}/{k}p" if k is not None else resolved_scope.value
            raise ValueError(
                f"Artifact {path} does not belong to required scope {stage}:{detail}"
            ) from exc
        return path

    def by_k_dir(self, stage: str, k: int) -> Path:
        """Return the canonical per-player-count directory for ``stage``."""

        return self.scope_dir(stage, ArtifactScope.BY_K, k=k)

    def concat_ks_dir(self, stage: str) -> Path:
        """Return the row-preserving cross-k concatenation directory."""

        return self.scope_dir(stage, ArtifactScope.CONCAT_KS)

    def across_k_dir(self, stage: str) -> Path:
        """Return the directory for declared common-support cross-k estimates."""

        return self.scope_dir(stage, ArtifactScope.ACROSS_K)

    def cross_seed_dir(self, stage: str) -> Path:
        """Return the directory for root-combination and stability artifacts."""

        return self.scope_dir(stage, ArtifactScope.CROSS_SEED)

    def diagnostics_dir(self, stage: str) -> Path:
        """Return the non-estimand diagnostic directory for ``stage``."""

        return self.scope_dir(stage, ArtifactScope.DIAGNOSTICS)

    def h2h_2p_dir(self, stage: str = "head2head") -> Path:
        """Return the explicitly two-player finalist H2H directory."""

        return self.scope_dir(stage, ArtifactScope.H2H_2P)

    def per_k_subdir(self, stage: str, k: int) -> Path:
        """Deprecated alias for :meth:`by_k_dir`."""

        return self.by_k_dir(stage, k)

    def ingest_block_dir(self, k: int) -> Path:
        """Directory holding ingest artifacts for ``k`` players."""

        return self.per_k_subdir("ingest", k)

    def curate_block_dir(self, k: int) -> Path:
        """Directory holding curated artifacts for ``k`` players."""

        return self.per_k_subdir("curate", k)

    def combine_block_dir(self, k: int) -> Path:
        """Deprecated alias for :meth:`curate_block_dir`."""

        return self.curate_block_dir(k)

    def combine_combined_dir(self, k: int | None = None) -> Path:  # noqa: ARG002
        """Deprecated alias for the row-preserving concatenation scope."""

        return self.concat_ks_dir("combine")

    @property
    def combine_partitioned_dir(self) -> Path:
        """Directory holding partitioned combined curated rows."""

        return self.concat_ks_dir("combine") / "all_ingested_rows_partitioned"

    def metrics_per_k_dir(self, k: int) -> Path:
        """Directory holding metrics artifacts for ``k`` players."""
        path = self.per_k_subdir("metrics", k)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def metrics_combined_dir(self) -> Path:
        """Deprecated alias for cross-k metric estimates."""
        path = self.across_k_dir("metrics")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def game_stats_stage_dir(self) -> Path:
        """Stage directory for game-stat analytics."""

        return self.stage_subdir("game_stats")

    @property
    def game_stats_combined_dir(self) -> Path:
        """Deprecated alias for cross-k game-stat summaries."""

        return self.across_k_dir("game_stats")

    @property
    def rng_stage_dir(self) -> Path:
        """Stage directory for RNG diagnostics."""

        return self.stage_subdir("rng_diagnostics")

    @property
    def rng_combined_dir(self) -> Path:
        """Deprecated alias for non-estimand RNG diagnostics."""

        return self.diagnostics_dir("rng_diagnostics")

    @property
    def seed_summaries_stage_dir(self) -> Path:
        """Stage directory for per-seed summaries."""

        return self.stage_subdir("seed_summaries")

    def seed_summaries_dir(self, players: int) -> Path:
        """Directory holding seed summaries for ``players`` count."""

        return self.by_k_dir("seed_summaries", players)

    @property
    def variance_stage_dir(self) -> Path:
        """Stage directory for variance analytics."""

        return self.stage_subdir("variance")

    @property
    def variance_combined_dir(self) -> Path:
        """Deprecated alias for cross-root variance artifacts."""

        return self.cross_seed_dir("variance")

    @property
    def meta_stage_dir(self) -> Path:
        """Stage directory for meta-analysis outputs."""

        return self.stage_subdir("meta")

    def meta_per_k_dir(self, players: int) -> Path:
        """Primary per-player meta-analysis directory."""

        return self.by_k_dir("meta", players)

    @property
    def meta_combined_dir(self) -> Path:
        """Deprecated alias for cross-root meta-analysis artifacts."""

        return self.cross_seed_dir("meta")

    @property
    def agreement_stage_dir(self) -> Path:
        """Stage directory for cross-method agreement analytics."""

        return self.stage_subdir("agreement")

    @property
    def interseed_stage_dir(self) -> Path:
        """Stage directory for cross-seed orchestration outputs."""

        return self.stage_subdir("interseed")

    @property
    def ingest_stage_dir(self) -> Path:
        """Stage root for ingest outputs."""

        return self.stage_subdir("ingest")

    @property
    def curate_stage_dir(self) -> Path:
        """Stage root for curated row outputs."""

        return self.resolve_stage_dir("curate", allow_missing=True)

    @property
    def combine_stage_dir(self) -> Path:
        """Stage root for combined curated-row outputs."""

        return self.stage_subdir("combine")

    @property
    def metrics_stage_dir(self) -> Path:
        """Stage root for metrics outputs."""

        return self.stage_subdir("metrics")

    @property
    def trueskill_stage_dir(self) -> Path:
        """Stage root for per-seed TrueSkill outputs."""

        return self.stage_subdir("trueskill")

    @property
    def trueskill_combined_dir(self) -> Path:
        """Deprecated alias for descriptive cross-k TrueSkill outputs."""

        return self.across_k_dir("trueskill")

    @property
    def head2head_stage_dir(self) -> Path:
        """Stage root for head-to-head analysis outputs."""

        return self.stage_subdir("head2head")

    @property
    def seed_symmetry_stage_dir(self) -> Path:
        """Stage root for seed-symmetry analysis outputs."""

        return self.stage_subdir("seed_symmetry")

    @property
    def post_h2h_stage_dir(self) -> Path:
        """Stage root for post head-to-head outputs."""

        return self.stage_subdir("post_h2h")

    @property
    def hgb_stage_dir(self) -> Path:
        """Stage root for histogram gradient boosting outputs."""

        return self.stage_subdir("hgb")

    def hgb_per_k_dir(self, k: int) -> Path:
        """Per-player-count HGB output directory."""

        return self.per_k_subdir("hgb", k)

    @property
    def hgb_combined_dir(self) -> Path:
        """Deprecated alias for descriptive cross-k HGB outputs."""

        return self.across_k_dir("hgb")

    @property
    def frequentist_stage_dir(self) -> Path:
        """Stage root for frequentist ranking outputs."""

        return self.stage_subdir("frequentist")

    @property
    def meta_analysis_dir(self) -> Path:
        """Directory containing per-seed summaries combined across runs."""

        configured = self.io.meta_analysis_dir
        if configured is None:
            return self.analysis_dir
        meta_path = Path(configured)
        if meta_path.is_absolute():
            return meta_path
        if meta_path.parts and meta_path.parts[0] == "data":
            return meta_path
        # Anchor relative paths to the parent of the seed-suffixed results_root
        return self.results_root.parent / meta_path

    @property
    def interseed_input_dir(self) -> Path | None:
        """Optional analysis root used to resolve interseed input artifacts."""

        configured = self.io.interseed_input_dir
        if configured is None:
            return None
        path = Path(configured)
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] == "data":
            return path
        return self.results_root / path

    def _interseed_input_candidate(self, stage_dir: Path, filename: str) -> Path | None:
        """Return a staged input path rooted at ``interseed_input_dir`` when set."""

        input_root = self.interseed_input_dir
        if input_root is None:
            return None
        try:
            rel = stage_dir.relative_to(self.analysis_dir)
        except ValueError:
            return input_root / filename
        if not rel.parts:
            return input_root / filename
        stage_key = self._stage_key_for_folder(rel.parts[0])
        input_folder = self._interseed_input_folder(stage_key) if stage_key else None
        if input_folder is None:
            return input_root / rel / filename
        return input_root / Path(input_folder, *rel.parts[1:]) / filename

    def _input_stage_path(self, key: str, *parts: str | Path) -> Path | None:
        """Return a stage path rooted at ``interseed_input_dir`` without creating it."""

        input_root = self.interseed_input_dir
        if input_root is None:
            return None
        stage_folder = self._interseed_input_folder(key)
        if stage_folder is None:
            stage_folder = self.stage_layout.folder_for(key)
            if stage_folder is None:
                return None
        return input_root / stage_folder / Path(*parts)

    def resolve_input_stage_dir(self, key: str, *parts: str | Path) -> Path | None:
        """Resolve an input-stage directory without requiring the active output layout."""

        interseed_path = self._input_stage_path(key, *parts)
        if interseed_path is not None:
            return interseed_path
        folder = self.stage_layout.folder_for(key)
        if folder is None:
            return None
        path = self.analysis_dir / folder
        if parts:
            path = path.joinpath(*map(Path, parts))
        return path

    def _interseed_input_folder(self, key: str | None) -> str | None:
        """Return the input-layout folder name for a stage key, when configured."""

        if key is None:
            return None
        layout = self.io.interseed_input_layout
        if layout is None:
            return None
        if isinstance(layout, Mapping):
            folder = layout.get(key)
            return str(folder) if folder is not None else None
        from farkle.analysis.stage_registry import StageLayout

        if isinstance(layout, StageLayout):
            return layout.folder_for(key)
        return None

    def _stage_key_for_folder(self, folder: str) -> str | None:
        """Return the stage key for a numbered folder in the current layout."""

        for placement in self.stage_layout.placements:
            if placement.folder_name == folder:
                return placement.definition.key
        return None

    def _stage_dir_if_active(self, key: str, *parts: str | Path) -> Path | None:
        """Return the stage directory for ``key`` only when it is active."""

        folder = self.stage_layout.folder_for(key)
        if folder is None:
            return None
        path = self.analysis_dir / folder
        if parts:
            path = path.joinpath(*map(Path, parts))
        return path

    def _interseed_stage_dir(self, key: str, *parts: str | Path) -> Path | None:
        """Return the stage directory rooted at the interseed input layout."""

        input_root = self.interseed_input_dir
        if input_root is None:
            return None
        folder = self._interseed_input_folder(key)
        if folder is None:
            return None
        return input_root / folder / Path(*parts)

    @property
    def data_dir(self) -> Path:
        """Root directory for curated data under the curate stage."""

        return self.resolve_stage_dir("curate", allow_missing=True)

    def n_dir(self, n: int) -> Path:
        """Convenience accessor for a specific ``<n>_players`` directory."""
        return self.results_root / f"{n}_players"

    def checkpoint_path(self, n: int) -> Path:
        """Path to a head-to-head checkpoint for ``n`` players."""
        return self.n_dir(n) / f"{n}p_checkpoint.pkl"

    def metrics_path(self, n: int) -> Path:
        """Path to the metrics parquet for ``n`` players."""
        return self.n_dir(n) / f"{n}p_metrics.parquet"

    def strategy_manifest_path(self, n: int) -> Path:  # noqa: ARG002
        """Path to the strategy manifest parquet for the current results root."""
        return self.strategy_manifest_root_path()

    def strategy_manifest_root_path(self) -> Path:
        """Root-level strategy manifest parquet path."""
        return self.results_root / "strategy_manifest.parquet"

    # —— Ingest/streaming knobs ——
    @property
    def row_group_size(self) -> int:
        """Row-group size used when writing parquet outputs during ingest."""
        return self.ingest.row_group_size

    @property
    def parquet_codec(self) -> Compression:
        """Compression codec used for ingest parquet outputs."""
        return normalize_compression(self.ingest.parquet_codec)

    @property
    def n_jobs_ingest(self) -> int:
        """Parallel worker setting for ingestion (0 => os.cpu_count(), >0 explicit)."""
        return self.ingest.n_jobs

    @property
    def batch_rows(self) -> int:
        """Rows per batch fed to streaming writers."""
        return self.ingest.batch_rows

    # —— Handy aliases used by some modules (kept to minimize edits) ——
    @property
    def trueskill_beta(self) -> float:
        """Alias for :class:`TrueSkillConfig` ``beta`` value."""
        return self.trueskill.beta

    @property
    def hgb_max_iter(self) -> int:
        """Alias for the histogram-based gradient boosting estimator count."""
        return self.hgb.n_estimators

    @property
    def combine_max_players(self) -> int:
        """Maximum player count to combine when consolidating parquet files."""
        return self.combine.max_players

    @property
    def metrics_seat_range(self) -> tuple[int, int]:
        """Seat indices included when computing seat-level metrics."""
        return self.metrics.seat_range

    # —— Output filenames and standard derived locations ——
    @property
    def metrics_name(self) -> str:
        """Filename for combined metrics parquet outputs."""
        # prefer analysis.outputs.metrics_name if provided
        outputs = self.analysis.outputs or {}
        return str(outputs.get("metrics_name", "metrics.parquet"))

    def metrics_output_path(self, name: str | None = None) -> Path:
        """Preferred path for combined metrics artifacts under the metrics stage."""

        filename = str(self.metrics_name if name is None else name)
        path = self.metrics_combined_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def game_stats_output_path(self, name: str) -> Path:
        """Preferred path for combined game-stat outputs."""

        canonical_name = self.canonical_artifact_name(name)
        path = self.game_stats_combined_dir / canonical_name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def game_stats_input_path(self, name: str) -> Path:
        """Resolve a cross-k game-stat artifact without legacy fallback."""

        return self.input_scope_path("game_stats", ArtifactScope.ACROSS_K, name)

    def rng_output_path(self, name: str) -> Path:
        """Preferred path for combined RNG diagnostics."""

        path = self.rng_combined_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def rng_input_path(self, name: str) -> Path:
        """Resolve an RNG diagnostic artifact in its canonical scope."""

        return self.input_scope_path("rng_diagnostics", ArtifactScope.DIAGNOSTICS, name)

    def variance_output_path(self, name: str) -> Path:
        """Preferred path for combined variance analytics."""

        path = self.variance_combined_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def variance_input_path(self, name: str) -> Path:
        """Resolve a variance artifact in the cross-root scope."""

        return self.input_scope_path("variance", ArtifactScope.CROSS_SEED, name)

    def meta_output_path(self, players: int, name: str) -> Path:
        """Preferred path for per-player meta-analysis artifacts."""

        path = self.meta_per_k_dir(players) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def meta_input_path(self, players: int, name: str) -> Path:
        """Resolve a per-k meta-analysis artifact without scope fallback."""

        return self.input_scope_path("meta", ArtifactScope.BY_K, name, k=players)

    def metrics_input_path(self, name: str | None = None) -> Path:
        """Resolve a cross-k metrics artifact without scope fallback."""

        filename = str(self.metrics_name if name is None else name)
        return self.input_scope_path("metrics", ArtifactScope.ACROSS_K, filename)

    def metrics_isolated_path(self, k: int) -> Path:
        """Preferred isolated metrics parquet for ``k`` players."""

        return self.metrics_per_k_dir(k) / f"{k}p_isolated_metrics.parquet"

    def legacy_metrics_isolated_path(self, k: int) -> Path:
        """Legacy isolated metrics parquet path under ``analysis/data``."""

        return self.analysis_dir / "data" / f"{k}p" / f"{k}p_isolated_metrics.parquet"

    @property
    def curated_rows_name(self) -> str:
        """Filename for curated row-level parquet outputs."""
        outputs = self.analysis.outputs or {}
        return str(outputs.get("curated_rows_name", "game_rows.parquet"))

    @property
    def manifest_name(self) -> str:
        """Filename used for append-only manifests."""
        outputs = self.analysis.outputs or {}
        return str(outputs.get("manifest_name", "manifest.jsonl"))

    def canonical_artifact_name(self, filename: str) -> str:
        """Return the canonical filename for a possibly legacy artifact name."""

        return _CANONICAL_ARTIFACT_NAMES.get(filename, filename)

    def legacy_artifact_names(self, filename: str) -> tuple[str, ...]:
        """Return legacy aliases for a canonical artifact filename."""

        return _LEGACY_ARTIFACT_NAMES.get(filename, ())

    def _preferred_stage_path(
        self,
        stage_dir: Path | None,
        legacy_dir: Path,
        filename: str,
        *,
        stage_key: str | None = None,
        stage_parts: Iterable[str | Path] = (),
    ) -> Path:
        """Return *filename* within the stage dir, falling back to legacy when absent."""

        canonical = self.canonical_artifact_name(filename)
        legacy_names = self.legacy_artifact_names(canonical)
        stage_candidate_names = (canonical, *legacy_names)

        legacy_paths = [legacy_dir / candidate for candidate in stage_candidate_names]
        if stage_dir is not None:
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage_paths = [stage_dir / candidate for candidate in stage_candidate_names]
            for stage_path in stage_paths:
                if stage_path.exists():
                    return stage_path
            for candidate in stage_candidate_names:
                interseed_input_path = self._interseed_input_candidate(stage_dir, candidate)
                if interseed_input_path is not None and interseed_input_path.exists():
                    return interseed_input_path
            for legacy_path in legacy_paths:
                if legacy_path.exists():
                    return legacy_path
            return stage_paths[0]

        interseed_path: Path | None = None
        if stage_key is not None:
            interseed_dir = self._interseed_stage_dir(stage_key, *stage_parts)
            if interseed_dir is not None:
                for candidate in stage_candidate_names:
                    interseed_path = interseed_dir / candidate
                    if interseed_path.exists():
                        return interseed_path
        for legacy_path in legacy_paths:
            if legacy_path.exists():
                return legacy_path
        if interseed_path is not None:
            return interseed_path
        return legacy_paths[0]

    def _resolve_stage_artifact_path(
        self,
        stage_key: str,
        filename: str,
        *parts: str | Path,
        legacy_paths: Iterable[Path] = (),
    ) -> Path:
        """Resolve a stage artifact without creating directories."""

        canonical = self.canonical_artifact_name(filename)
        candidate_names = (canonical, *self.legacy_artifact_names(canonical))

        candidates: list[Path] = []
        input_dir = self._input_stage_path(stage_key, *parts)
        stage_dir = self._stage_dir_if_active(stage_key, *parts)

        # Interseed analysis reads combine artifacts produced by an upstream
        # seed run; prefer that input root deterministically when configured.
        if stage_key == "combine":
            if input_dir is not None:
                candidates.extend([input_dir / name for name in candidate_names])
            if stage_dir is not None:
                candidates.extend([stage_dir / name for name in candidate_names])
        else:
            if stage_dir is not None:
                candidates.extend([stage_dir / name for name in candidate_names])
            if input_dir is not None:
                candidates.extend([input_dir / name for name in candidate_names])

        interseed_dir = self._interseed_stage_dir(stage_key, *parts)
        if interseed_dir is not None:
            interseed_candidates = [interseed_dir / name for name in candidate_names]
            for interseed_candidate in interseed_candidates:
                if interseed_candidate not in candidates:
                    candidates.append(interseed_candidate)

        for legacy_path in legacy_paths:
            if legacy_path not in candidates:
                candidates.append(legacy_path)

        if not candidates:
            candidates.append(self.analysis_dir / canonical)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def agreement_players(self) -> list[int]:
        """Return normalized numeric player counts for agreement analysis outputs."""

        normalized: set[int] = set()
        for entry in self.sim.n_players_list:
            if isinstance(entry, bool):
                raise ValueError(f"invalid n_players_list entry: {entry!r}")
            try:
                value = int(entry)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid n_players_list entry: {entry!r}") from exc
            if value <= 0:
                raise ValueError(
                    f"n_players_list must contain positive player counts, got {entry!r}"
                )
            normalized.add(value)
        return sorted(normalized)

    def agreement_include_across_k(self) -> bool:
        """Return whether cross-k agreement output should be generated."""

        return bool(self.analysis.agreement_include_across_k)

    def agreement_across_k_output_path(self) -> Path:
        """Preferred path for cross-k agreement diagnostics."""

        return self.scope_path(
            "agreement",
            ArtifactScope.DIAGNOSTICS,
            "agreement_across_k.json",
        )

    def agreement_output_path(self, players: int) -> Path:
        """Preferred path for agreement analytics for a given player count."""

        players_int = int(players)
        filename = f"agreement_{players_int}p.json"
        return self.scope_path(
            "agreement",
            ArtifactScope.BY_K,
            filename,
            k=players_int,
        )

    def trueskill_path(self, filename: str) -> Path:
        """Resolve a TrueSkill artifact path with legacy fallback."""
        filename = self.canonical_artifact_name(filename)
        combined = filename.startswith("ratings_combined") or filename.startswith("ratings_k_weighted")
        parts = ("combined",) if combined else ()
        stage_dir = self._stage_dir_if_active("trueskill", *parts)
        return self._preferred_stage_path(
            stage_dir,
            self.analysis_dir,
            filename,
            stage_key="trueskill",
            stage_parts=parts,
        )

    def head2head_path(self, filename: str) -> Path:
        """Resolve a head-to-head artifact path with legacy fallback."""

        stage_dir = self._stage_dir_if_active("head2head")
        return self._preferred_stage_path(
            stage_dir,
            self.analysis_dir,
            filename,
            stage_key="head2head",
        )

    def post_h2h_path(self, filename: str) -> Path:
        """Resolve a post head-to-head artifact path with legacy fallback."""

        canonical = self.canonical_artifact_name(filename)
        candidate_names = (canonical, *self.legacy_artifact_names(canonical))
        candidates: list[Path] = []
        post_h2h_dir = self._stage_dir_if_active("post_h2h")
        if post_h2h_dir is not None:
            candidates.extend(post_h2h_dir / name for name in candidate_names)
        head2head_dir = self._stage_dir_if_active("head2head")
        if head2head_dir is not None:
            for name in candidate_names:
                candidate = head2head_dir / name
                if candidate not in candidates:
                    candidates.append(candidate)
        for name in candidate_names:
            candidate = self.analysis_dir / name
            if candidate not in candidates:
                candidates.append(candidate)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def frequentist_path(self, filename: str) -> Path:
        """Resolve a frequentist-stage artifact path with legacy fallback."""

        canonical = self.canonical_artifact_name(filename)
        stage_dir = self._stage_dir_if_active("frequentist")
        return self._preferred_stage_path(
            stage_dir,
            self.analysis_dir,
            canonical,
            stage_key="frequentist",
        )

    def preferred_tiers_path(self) -> Path:
        """Locate ``tiers.json`` across frequentist and TrueSkill stages."""

        analysis_path = self.analysis_dir / "tiers.json"

        frequentist_path = self._resolve_stage_artifact_path("frequentist", "tiers.json")
        if frequentist_path.exists() and frequentist_path != analysis_path:
            return frequentist_path

        trueskill_path = self._resolve_stage_artifact_path("trueskill", "tiers.json")
        if trueskill_path.exists():
            return trueskill_path

        if analysis_path.exists():
            return analysis_path
        return frequentist_path

    @property
    def curated_parquet(self) -> Path:
        """Location of the combined curated parquet spanning all player counts."""

        return self._resolve_combine_artifact_path("all_ingested_rows.parquet")

    def curated_parquet_candidates(self) -> tuple[Path, ...]:
        """Ordered candidate paths considered when resolving ``curated_parquet``."""

        return self._combine_artifact_candidates("all_ingested_rows.parquet")

    @property
    def curated_dataset(self) -> Path:
        """Path to the preferred dataset layout for combined curated rows."""

        return self._resolve_combine_dataset_path()

    def curated_dataset_candidates(self) -> tuple[Path, ...]:
        """Ordered candidate paths considered when resolving ``curated_dataset``."""

        return self._combine_dataset_candidates()

    def _combine_dataset_candidates(self) -> tuple[Path, ...]:
        """Return the declared concatenated-row dataset location."""

        path = self.input_scope_path(
            "combine",
            ArtifactScope.CONCAT_KS,
            "all_ingested_rows_partitioned",
        )
        return (path,)

    def _resolve_combine_dataset_path(self) -> Path:
        """Resolve preferred combined dataset path with monolithic fallback."""

        for candidate in self._combine_dataset_candidates():
            if candidate.exists():
                return candidate
        return self.curated_parquet

    def _combine_artifact_candidates(self, filename: str) -> tuple[Path, ...]:
        """Return the declared row-preserving concatenation artifact location."""

        return (
            self.input_scope_path("combine", ArtifactScope.CONCAT_KS, filename),
        )

    def _resolve_combine_artifact_path(self, filename: str) -> Path:
        """Resolve an artifact from the canonical row-concatenation scope."""

        candidates = self._combine_artifact_candidates(filename)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @property
    def game_stats_margin_thresholds(self) -> tuple[int, ...]:
        """Victory-margin thresholds used by game-stat summaries."""
        return tuple(self.analysis.game_stats_margin_thresholds)

    @property
    def rare_event_target_score(self) -> int:
        """Score threshold used to flag multiple players crossing the target."""
        return int(self.analysis.rare_event_target_score)

    # Per-N helper paths used by ingest/curate/metrics
    def manifest_for(self, n: int) -> Path:
        """Path to the manifest for a specific player count."""
        return self.curate_block_dir(n) / self.manifest_name

    def ingested_rows_raw(self, n: int) -> Path:
        """Path to the raw ingested parquet for ``n`` players."""
        return self.ingest_block_dir(n) / f"{n}p_ingested_rows.raw.parquet"

    def ingest_manifest(self, n: int) -> Path:
        """Path to the append-only ingest manifest for ``n`` players."""

        return self.ingested_rows_raw(n).with_suffix(".manifest.jsonl")

    def ingested_rows_curated(self, n: int) -> Path:
        """Path to the curated ingested parquet for ``n`` players."""
        return self.curate_block_dir(n) / self.curated_rows_name

    def combined_manifest_path(self) -> Path:
        """Path to the manifest accompanying ``curated_parquet``."""

        return self._resolve_combine_artifact_path("all_ingested_rows.manifest.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# Loader (one or more YAML overlays; dotted keys allowed)
# ─────────────────────────────────────────────────────────────────────────────


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` and return a new mapping."""
    result: dict[str, Any] = dict(base)
    for key, val in overlay.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(val, Mapping):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _annotation_contains(annotation: Any, target: type) -> bool:
    """Recursively inspect type annotations for the presence of ``target``."""
    if annotation is None:
        return False
    if annotation is target:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(_annotation_contains(arg, target) for arg in get_args(annotation))


def _normalize_results_dir_prefix(value: str | Path) -> Path:
    """Normalize results_dir_prefix values from legacy results_dir inputs."""
    path = Path(value)
    match = re.match(r"^(?P<base>.+)_seed_\d+$", path.name)
    if match:
        path = path.with_name(match.group("base"))
    if not path.is_absolute() and path.parts and path.parts[0] == "data" and len(path.parts) > 1:
        path = Path(*path.parts[1:])
    return path


def _normalize_seed_list(sim: SimConfig) -> None:
    """Normalize and validate ``seed_list`` for a :class:`SimConfig` instance."""
    if sim.seed_list is None:
        return
    if isinstance(sim.seed_list, (list, tuple)):
        seed_list = [int(s) for s in sim.seed_list]
    else:
        raise TypeError("sim.seed_list must be a list/tuple of integers")
    if not seed_list:
        raise ValueError("sim.seed_list must contain at least one seed")
    sim.seed_list = seed_list


def _normalize_seed_pair(sim: SimConfig, *, seed_provided: bool) -> None:
    """Normalize and validate ``seed_pair`` for a :class:`SimConfig` instance."""
    if sim.seed_pair is None:
        return
    if isinstance(sim.seed_pair, (list, tuple)):
        seed_pair = tuple(int(s) for s in sim.seed_pair)
    else:
        raise TypeError("sim.seed_pair must be a tuple/list of two integers")
    if len(seed_pair) != 2:
        raise ValueError(f"sim.seed_pair must contain exactly two seeds, got {seed_pair!r}")
    sim.seed_pair = seed_pair
    if seed_provided and sim.seed != seed_pair[0]:
        raise ValueError(
            "sim.seed must match seed_pair[0] when both are set "
            f"(seed={sim.seed}, seed_pair={seed_pair})"
        )
    if not seed_provided:
        sim.seed = seed_pair[0]


def _format_unknown_keys(unknown: Iterable[str], candidates: Iterable[str]) -> list[str]:
    """Return formatted unknown keys with optional suggestions."""
    suggestions: list[str] = []
    candidate_list = list(candidates)
    for key in sorted(unknown):
        suggestion = difflib.get_close_matches(key, candidate_list, n=1)
        if suggestion:
            suggestions.append(f"{key!r} (did you mean {suggestion[0]!r}?)")
        else:
            suggestions.append(repr(key))
    return suggestions


def _validate_config_keys(data: Mapping[str, Any]) -> None:
    """Validate that config sections and keys match dataclass schemas."""
    top_level_sections = {
        "io": IOConfig,
        "sim": SimConfig,
        "analysis": AnalysisConfig,
        "ingest": IngestConfig,
        "combine": CombineConfig,
        "metrics": MetricsConfig,
        "trueskill": TrueSkillConfig,
        "head2head": Head2HeadConfig,
        "hgb": HGBConfig,
        "orchestration": OrchestrationConfig,
        "rng": RNGConfig,
        "screening": ScreeningConfig,
        "batching": BatchingConfig,
        "robustness": RobustnessConfig,
        "artifact_contract": ArtifactContractConfig,
        "k_aggregation": KAggregationConfig,
    }
    unknown_sections = set(data) - set(top_level_sections)
    if unknown_sections:
        formatted = ", ".join(_format_unknown_keys(unknown_sections, top_level_sections))
        raise ValueError(f"Unknown top-level config section(s): {formatted}")

    for section_name, section_cls in top_level_sections.items():
        if section_name not in data:
            continue
        section = data[section_name]
        if not isinstance(section, Mapping):
            raise TypeError(f"Config section {section_name!r} must be a mapping")
        allowed_keys = {field.name for field in dataclasses.fields(section_cls)}
        unknown_keys = set(section) - allowed_keys
        if unknown_keys:
            formatted = ", ".join(_format_unknown_keys(unknown_keys, allowed_keys))
            raise ValueError(
                f"Unknown key(s) in config section {section_name!r}: {formatted}"
            )
        if section_name == "sim" and "per_n" in section:
            per_n_section = section.get("per_n")
            if not isinstance(per_n_section, Mapping):
                raise TypeError("sim.per_n must be a mapping of per-player overrides")
            for key, val in per_n_section.items():
                if not isinstance(val, Mapping):
                    continue
                per_n_unknown = set(val) - allowed_keys
                if per_n_unknown:
                    formatted = ", ".join(_format_unknown_keys(per_n_unknown, allowed_keys))
                    raise ValueError(
                        "Unknown key(s) in config section "
                        f"sim.per_n[{key!r}]: {formatted}"
                    )


def _validate_seed_sources(
    sim: SimConfig,
    *,
    seed_provided: bool,
    seed_list_provided: bool,
    seed_pair_provided: bool,
    expected_seed_len: int | None,
    context: str,
) -> None:
    """Validate seed configuration for the provided context."""
    if sim.seed_list is not None:
        if expected_seed_len is not None and len(sim.seed_list) != expected_seed_len:
            raise ValueError(
                f"{context}: sim.seed_list must contain exactly {expected_seed_len} seeds, "
                f"got {sim.seed_list!r}"
            )
        if seed_list_provided and (seed_provided or seed_pair_provided):
            LOGGER.warning(
                "%s: sim.seed_list overrides legacy sim.seed/seed_pair settings",
                context,
                extra={
                    "stage": "config",
                    "seed_list": list(sim.seed_list),
                    "seed": sim.seed,
                    "seed_pair": list(sim.seed_pair) if sim.seed_pair is not None else None,
                },
            )
        if sim.seed != sim.seed_list[0] and seed_provided:
            LOGGER.warning(
                "%s: sim.seed overwritten by sim.seed_list[0] for deterministic naming",
                context,
                extra={
                    "stage": "config",
                    "seed_list": list(sim.seed_list),
                    "seed": sim.seed,
                },
            )
        if sim.seed != sim.seed_list[0]:
            sim.seed = sim.seed_list[0]
        if sim.seed_pair is not None and list(sim.seed_pair) != list(sim.seed_list):
            raise ValueError(
                f"{context}: sim.seed_list and sim.seed_pair must match when both are set "
                f"(seed_list={sim.seed_list!r}, seed_pair={sim.seed_pair!r})"
            )
        if sim.seed_pair is None and len(sim.seed_list) == 2:
            sim.seed_pair = (sim.seed_list[0], sim.seed_list[1])


def load_app_config(*overlays: Path, seed_list_len: int | None = None) -> AppConfig:
    """Deterministically merge one or more YAML overlays into an :class:`AppConfig`.

    Files are read in the order provided, dotted keys are expanded, and later overlays
    always win. Re-loading the same sequence therefore yields identical configs, which
    keeps resume semantics predictable.
    """
    data: dict[str, Any] = {}
    for path in overlays:
        with path.open("r", encoding="utf-8") as fh:
            overlay = yaml.safe_load(fh) or {}
        if not isinstance(overlay, Mapping):
            raise TypeError(f"Config file {path} must contain a mapping")
        expanded = expand_dotted_keys(overlay)
        data = _deep_merge(data, expanded)

    for dotted_key, replacement in RETIRED_CONFIG_KEYS.items():
        section_name, option = dotted_key.split(".", 1)
        section = data.get(section_name)
        if isinstance(section, Mapping) and option in section:
            raise ValueError(
                f"Retired config key {dotted_key!r}; use {replacement!r}. "
                "Legacy statistical settings are not reinterpreted."
            )

    # Light compatibility if someone uses old keys
    if "io" in data:
        io_section_raw = data["io"]
        if isinstance(io_section_raw, Mapping):
            io_section: MutableMapping[str, Any]
            if isinstance(io_section_raw, MutableMapping):
                io_section = io_section_raw
            else:
                io_section = dict(io_section_raw)
                data["io"] = io_section
            if "analysis_dir" in io_section and "analysis_subdir" not in io_section:
                io_section["analysis_subdir"] = io_section.pop("analysis_dir")
            if "results_dir" in io_section and "results_dir_prefix" not in io_section:
                io_section["results_dir_prefix"] = _normalize_results_dir_prefix(
                    io_section.pop("results_dir")
                )
    seed_provided = False
    seed_list_provided = False
    seed_pair_provided = False
    per_n_seed_provided: dict[int, bool] = {}
    per_n_seed_list_provided: dict[int, bool] = {}
    per_n_seed_pair_provided: dict[int, bool] = {}
    if "sim" in data:
        sim_section_raw = data["sim"]
        if not isinstance(sim_section_raw, Mapping):
            raise TypeError("Config section 'sim' must be a mapping")
        sim_section: MutableMapping[str, Any]
        if isinstance(sim_section_raw, MutableMapping):
            sim_section = sim_section_raw
        else:
            sim_section = dict(sim_section_raw)
            data["sim"] = sim_section
        seed_provided = "seed" in sim_section
        seed_list_provided = "seed_list" in sim_section
        seed_pair_provided = "seed_pair" in sim_section
        if "n_players" in sim_section and "n_players_list" not in sim_section:
            sim_section["n_players_list"] = [sim_section.pop("n_players")]
        raw_players = sim_section.get("n_players_list")
        if isinstance(raw_players, list):
            numeric_players: list[int] = []
            for entry in raw_players:
                if isinstance(entry, str) and entry.strip().lower() == "combined":
                    raise ValueError(
                        "sim.n_players_list requires concrete player counts >= 2; "
                        "select cross-k work with canonical scope settings"
                    )
                try:
                    players = int(entry)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid n_players_list entry: {entry!r}") from exc
                if players < 2:
                    raise ValueError(
                        "sim.n_players_list requires concrete player counts >= 2; "
                        "select cross-k work with canonical scope settings"
                    )
                numeric_players.append(players)
            sim_section["n_players_list"] = numeric_players
        if (
            "collect_metrics" in sim_section
            and "expanded_metrics" not in sim_section
            and sim_section.pop("collect_metrics")
        ):
            sim_section["expanded_metrics"] = True
        per_n_section = sim_section.get("per_n")
        if isinstance(per_n_section, Mapping):
            for key, val in per_n_section.items():
                if isinstance(val, Mapping):
                    try:
                        per_n_key = int(key)
                    except (TypeError, ValueError):
                        continue
                    if "seed" in val:
                        per_n_seed_provided[per_n_key] = True
                    if "seed_list" in val:
                        per_n_seed_list_provided[per_n_key] = True
                    if "seed_pair" in val:
                        per_n_seed_pair_provided[per_n_key] = True
    if "analysis" in data:
        analysis_section_raw = data["analysis"]
        if isinstance(analysis_section_raw, Mapping):
            analysis_section_map: MutableMapping[str, Any]
            if isinstance(analysis_section_raw, MutableMapping):
                analysis_section_map = analysis_section_raw
            else:
                analysis_section_map = dict(analysis_section_raw)
                data["analysis"] = analysis_section_map
            deprecated = sorted(
                key for key in analysis_section_map if key in DEPRECATED_ANALYSIS_FLAGS
            )
            if deprecated:
                LOGGER.warning(
                    "Deprecated analysis flags no longer disable stages; flags ignored",
                    extra={"stage": "config", "flags": deprecated},
                )

    _validate_config_keys(data)

    def build(cls, section: Mapping[str, Any]) -> Any:
        """Instantiate a dataclass ``cls`` from a mapping of attributes."""
        obj = cls()
        typing_ns = globals().copy()
        if "StageLayout" not in typing_ns:
            stage_layout_cls: type[Any] | None
            try:
                from farkle.analysis.stage_registry import StageLayout as imported_stage_layout_cls
            except ImportError:
                stage_layout_cls = None
            else:
                stage_layout_cls = imported_stage_layout_cls
            typing_ns["StageLayout"] = stage_layout_cls
        type_hints = get_type_hints(cls, globalns=typing_ns)
        for f in dataclasses.fields(cls):
            if f.name not in section:
                continue
            val = section[f.name]
            current = getattr(obj, f.name)
            annotation = type_hints.get(f.name)

            # NEW: plain nested dataclass support (e.g., BHDesign / BonferroniDesign)
            if annotation is not None and is_dataclass(annotation) and isinstance(val, Mapping):
                val = build(annotation, val)

            # Existing: map dict[int, Dataclass] (e.g., per_n)
            if annotation is not None and get_origin(annotation) is dict:
                key_t, val_t = get_args(annotation)
                if is_dataclass(val_t):
                    val = {
                        (int(k) if key_t is int else k): (
                            build(val_t, v) if isinstance(v, Mapping) else v
                        )
                        for k, v in (val or {}).items()
                    }

            # Path coercion (works for nested too because we use type hints)
            if (isinstance(current, Path) or _annotation_contains(annotation, Path)) and isinstance(
                val, (str, Path)
            ):
                val = Path(val)

            setattr(obj, f.name, val)
        return obj

    cfg = AppConfig(
        io=build(IOConfig, data.get("io", {})),
        sim=build(SimConfig, data.get("sim", {})),
        analysis=build(AnalysisConfig, data.get("analysis", {})),
        ingest=build(IngestConfig, data.get("ingest", {})),
        combine=build(CombineConfig, data.get("combine", {})),
        metrics=build(MetricsConfig, data.get("metrics", {})),
        trueskill=build(TrueSkillConfig, data.get("trueskill", {})),
        head2head=build(Head2HeadConfig, data.get("head2head", {})),
        hgb=build(HGBConfig, data.get("hgb", {})),
        orchestration=build(OrchestrationConfig, data.get("orchestration", {})),
        rng=build(RNGConfig, data.get("rng", {})),
        screening=build(ScreeningConfig, data.get("screening", {})),
        batching=build(BatchingConfig, data.get("batching", {})),
        robustness=build(RobustnessConfig, data.get("robustness", {})),
        artifact_contract=build(
            ArtifactContractConfig, data.get("artifact_contract", {})
        ),
        k_aggregation=build(KAggregationConfig, data.get("k_aggregation", {})),
    )
    _normalize_seed_list(cfg.sim)
    _normalize_seed_pair(cfg.sim, seed_provided=seed_provided)
    _validate_seed_sources(
        cfg.sim,
        seed_provided=seed_provided,
        seed_list_provided=seed_list_provided,
        seed_pair_provided=seed_pair_provided,
        expected_seed_len=seed_list_len,
        context="load_app_config",
    )
    if cfg.sim.per_n:
        for key, sim_cfg in cfg.sim.per_n.items():
            key_int = int(key)
            _normalize_seed_list(sim_cfg)
            _normalize_seed_pair(sim_cfg, seed_provided=per_n_seed_provided.get(key_int, False))
            _validate_seed_sources(
                sim_cfg,
                seed_provided=per_n_seed_provided.get(key_int, False),
                seed_list_provided=per_n_seed_list_provided.get(key_int, False),
                seed_pair_provided=per_n_seed_pair_provided.get(key_int, False),
                expected_seed_len=None,
                context=f"load_app_config(sim.per_n[{key_int}])",
            )
    return cfg


def _coerce(value: str, current: Any, annotation: Any | None = None) -> Any:
    """Coerce ``value`` to the type of ``current``."""
    if isinstance(current, bool) or _annotation_contains(annotation, bool):
        val_lower = value.lower()
        if val_lower in {"1", "true", "yes", "on"}:
            return True
        if val_lower in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value from {value!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if (
        annotation is not None
        and _annotation_contains(annotation, int)
        and not _annotation_contains(annotation, bool)
    ):
        return int(value)
    if isinstance(current, float) or (
        annotation is not None and _annotation_contains(annotation, float)
    ):
        return float(value)
    if isinstance(current, Path) or (
        annotation is not None and _annotation_contains(annotation, Path)
    ):
        return Path(value)
    return value


def apply_dot_overrides(cfg: AppConfig, pairs: list[str]) -> AppConfig:
    """Apply ``section.option=value`` overrides to *cfg*."""
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override {pair!r}")
        key, raw = pair.split("=", 1)
        if "." not in key:
            raise ValueError(f"Invalid override {pair!r}")
        section_name, option = key.split(".", 1)
        if section_name == "io" and option == "results_dir":
            option = "results_dir_prefix"
        section = getattr(cfg, section_name)
        if not hasattr(section, option):
            raise AttributeError(f"Unknown option {option!r} in section {section_name!r}")
        current = getattr(section, option)
        if section_name == "analysis" and option in DEPRECATED_ANALYSIS_FLAGS:
            LOGGER.warning(
                "Deprecated analysis flag override provided; stages ignore it",
                extra={"stage": "config", "flag": option},
            )
        typing_ns = globals().copy()
        if "StageLayout" not in typing_ns:
            stage_layout_cls: type[Any] | None
            try:
                from farkle.analysis.stage_registry import StageLayout as imported_stage_layout_cls
            except ImportError:
                stage_layout_cls = None
            else:
                stage_layout_cls = imported_stage_layout_cls
            typing_ns["StageLayout"] = stage_layout_cls
        type_hints = get_type_hints(type(section), globalns=typing_ns)
        annotation = type_hints.get(option)
        new_value = _coerce(raw, current, annotation)
        if (
            section_name == "io"
            and option == "results_dir_prefix"
            and isinstance(new_value, (str, Path))
        ):
            new_value = _normalize_results_dir_prefix(new_value)
        setattr(section, option, new_value)
    return cfg


def _stringify_paths_for_serialization(obj: Any) -> Any:
    """Recursively convert ``Path`` objects into strings for JSON/YAML output."""

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: _stringify_paths_for_serialization(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [_stringify_paths_for_serialization(val) for val in obj]
    if isinstance(obj, tuple):
        return tuple(_stringify_paths_for_serialization(val) for val in obj)
    return obj


def effective_config_dict(cfg: AppConfig) -> dict[str, Any]:
    """Return a materialized config mapping suitable for persistence and hashing."""

    resolved = _stringify_paths_for_serialization(dataclasses.asdict(cfg))
    resolved.pop("config_sha", None)
    resolved.pop("_stage_layout", None)
    return resolved


def _drop_nested_path(payload: MutableMapping[str, Any], path: str) -> None:
    """Remove a dotted nested key from a materialized config payload if present."""

    cursor: MutableMapping[str, Any] | None = payload
    parts = path.split(".")
    for part in parts[:-1]:
        if cursor is None:
            return
        next_value = cursor.get(part)
        if not isinstance(next_value, MutableMapping):
            return
        cursor = next_value
    if cursor is not None:
        cursor.pop(parts[-1], None)


def _hashable_config_dict(cfg: AppConfig) -> dict[str, Any]:
    """Return the config payload used for deterministic cache hashing."""

    resolved = effective_config_dict(cfg)
    for path in ("sim.progress_logging", "analysis.progress_logging"):
        _drop_nested_path(resolved, path)
    return resolved


def _assign_nested_path(target: MutableMapping[str, Any], path: str, value: Any) -> None:
    """Assign ``value`` to a dotted nested key, creating intermediate mappings."""

    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, MutableMapping):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


def _extract_scope_value(payload: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    """Return ``(present, value)`` for a dotted nested key lookup."""

    cursor: Any = payload
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return False, None
        cursor = cursor[part]
    return True, cursor


def _project_effective_config(payload: Mapping[str, Any], scope_paths: Sequence[str]) -> dict[str, Any]:
    """Project a config payload down to the dotted paths used by a stage cache scope."""

    projected: dict[str, Any] = {}
    for path in scope_paths:
        present, value = _extract_scope_value(payload, path)
        if present:
            _assign_nested_path(projected, path, value)
    return projected


def _validate_statistical_contract(cfg: AppConfig, *, require_two_roots: bool) -> None:
    """Validate the locked statistical configuration contract."""

    player_counts = [int(k) for k in cfg.sim.n_players_list]
    if not player_counts or any(k < 2 for k in player_counts):
        raise ValueError("sim.n_players_list must contain player counts >= 2")
    if len(set(player_counts)) != len(player_counts):
        raise ValueError("sim.n_players_list must not contain duplicate player counts")
    if cfg.rng.scheme_version != 1 or cfg.rng.bit_generator != "PCG64DXSM":
        raise ValueError("rng must use scheme_version=1 and bit_generator='PCG64DXSM'")
    if not 0.0 < cfg.screening.resolution_delta < 1.0:
        raise ValueError("screening.resolution_delta must be between 0 and 1")
    if not 0.0 < cfg.screening.interval_confidence < 1.0:
        raise ValueError("screening.interval_confidence must be between 0 and 1")
    if cfg.batching.target_batches < 2 or cfg.batching.min_shuffles_per_batch < 1:
        raise ValueError("batching requires at least two batches and one shuffle per batch")

    practical = cfg.screening.practical_delta_by_k
    if practical is None:
        raise ValueError(
            "screening.practical_delta_by_k must explicitly cover every configured player count"
        )
    normalized_practical = {int(k): float(value) for k, value in practical.items()}
    if set(normalized_practical) != set(player_counts):
        raise ValueError("screening.practical_delta_by_k keys must match sim.n_players_list")
    if any(value <= 0.0 for value in normalized_practical.values()):
        raise ValueError("screening practical thresholds must be positive")
    if cfg.screening.delta_across_k is None or cfg.screening.delta_across_k <= 0.0:
        raise ValueError("screening.delta_across_k must be explicitly configured and positive")

    if cfg.k_aggregation.method not in {"equal-k", "declared-mapping"}:
        raise ValueError("k_aggregation.method must be 'equal-k' or 'declared-mapping'")
    if cfg.k_aggregation.method == "equal-k" and cfg.k_aggregation.k_weights is not None:
        raise ValueError("equal-k aggregation must not provide k_aggregation.k_weights")
    if cfg.k_aggregation.method == "declared-mapping":
        weights = cfg.k_aggregation.k_weights
        if weights is None or {int(k) for k in weights} != set(player_counts):
            raise ValueError("declared-mapping weights must cover configured player counts")
        if any(float(value) <= 0.0 for value in weights.values()):
            raise ValueError("declared player-count weights must be positive")
        if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-12:
            raise ValueError("declared player-count weights must sum to 1")

    roots = cfg.sim.seed_list or [cfg.sim.seed]
    if require_two_roots and len(roots) != 2:
        raise ValueError("the combined-root production workflow requires exactly two root seeds")
    if len({int(seed) for seed in roots}) != len(roots):
        raise ValueError("root seeds must be distinct")

    h2h = cfg.head2head
    if not 0.0 < h2h.family_alpha < 1.0:
        raise ValueError("head2head.family_alpha must be between 0 and 1")
    if not 0.0 < h2h.target_power < 1.0:
        raise ValueError("head2head.target_power must be between 0 and 1")
    if h2h.practical_delta <= 0.0:
        raise ValueError("head2head.practical_delta must be positive")
    if h2h.candidate_cap is not None and h2h.candidate_cap < 2:
        raise ValueError("head2head.candidate_cap must be at least 2")
    if h2h.candidate_cap_policy != "balanced-tail":
        raise ValueError("head2head.candidate_cap_policy must be 'balanced-tail'")
    if h2h.total_game_cap is not None and h2h.total_game_cap <= 0:
        raise ValueError("head2head.total_game_cap must be positive when configured")


def compute_config_sha(cfg: AppConfig) -> str:
    """Return a deterministic sha256 over the effective configuration payload."""

    canonical_json = json.dumps(
        _hashable_config_dict(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_stage_config_sha(cfg: AppConfig, stage_key: str) -> str:
    """Return a deterministic cache hash for ``stage_key`` based on its config scope."""

    from farkle.analysis.stage_registry import resolve_stage_definition

    definition = resolve_stage_definition(stage_key)
    stage_scope = _project_effective_config(
        _hashable_config_dict(cfg),
        definition.cache_scope,
    )
    canonical_json = json.dumps(
        {
            "stage": stage_key,
            "cache_key_version": definition.cache_key_version,
            "config": stage_scope,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def assign_config_sha(cfg: AppConfig) -> str:
    """Compute and persist ``cfg.config_sha`` using canonical serialization."""

    cfg.config_sha = compute_config_sha(cfg)
    return cfg.config_sha


__all__ = [
    "ArtifactScope",
    "IOConfig",
    "ProgressLogConfig",
    "SimConfig",
    "AnalysisConfig",
    "IngestConfig",
    "CombineConfig",
    "MetricsConfig",
    "TrueSkillConfig",
    "Head2HeadConfig",
    "HGBConfig",
    "PowerDesign",
    "RNGConfig",
    "ScreeningConfig",
    "BatchingConfig",
    "RobustnessConfig",
    "ArtifactContractConfig",
    "KAggregationConfig",
    "AppConfig",
    "expected_seed_list_length",
    "load_app_config",
    "apply_dot_overrides",
    "assign_config_sha",
    "compute_stage_config_sha",
    "compute_config_sha",
    "effective_config_dict",
]
