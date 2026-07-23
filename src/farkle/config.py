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
import math
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


RETIRED_CONFIG_KEYS: dict[str, str] = {
    "sim.num_shuffles": "screening.resolution_delta and batching settings",
    "sim.power_method": "screening.resolution_delta",
    "sim.recompute_num_shuffles": "screening.resolution_delta",
    "sim.power_design": "screening and head2head settings",
    "analysis.tier_z_star": "screening bootstrap summaries",
    "analysis.tier_min_gap": "screening practical_delta_by_k",
    "analysis.frequentist_weights_by_k": "k_aggregation.k_weights",
    "analysis.meta_random_if_I2_gt": "cross-seed stability diagnostics",
    "analysis.meta_max_other_seeds": "sim.seed_list",
    "analysis.meta_comparison_seed": "sim.seed_list",
    "analysis.\x70\x6f\x6f\x6c\x69\x6e\x67_weights": "k_aggregation.method",
    "analysis.\x70\x6f\x6f\x6c\x69\x6e\x67_weights_by_k": "k_aggregation.k_weights",
    "analysis.k_aggregation_method": "k_aggregation.method",
    "analysis.k_weights": "k_aggregation.k_weights",
    "analysis.agreement_include_combined": "analysis.agreement_include_across_k",
    "trueskill.\x70\x6f\x6f\x6c\x65\x64_weights_by_k": "canonical root/k TrueSkill ratings",
    "head2head.fdr_q": "head2head.family_alpha",
    "head2head.bonferroni_total_games_safeguard": "head2head.total_game_cap",
    "head2head.bonferroni_design": "typed head2head settings",
    "head2head.games_per_pair": "head2head target power and practical effect",
    "head2head.tie_break_policy": "dominance front display rules",
    "head2head.tie_break_seed": "stable strategy identifier display ordering",
    "head2head.use_tier_elites": "the frozen canonical candidate family",
    "io.analysis_dir": "io.analysis_subdir",
    "io.results_dir": "io.results_dir_prefix",
    "io.meta_analysis_dir": "canonical cross_seed artifacts under the pair root",
    "io.interseed_input_dir": "explicit root-pair run context",
    "io.interseed_input_layout": "canonical root stage layout",
    "analysis.agreement_strategies": "the frozen H2H candidate family",
    "analysis.agreement_include_across_k": "selection-conditioned structure agreement",
    "sim.n_players": "sim.n_players_list",
    "sim.collect_metrics": "sim.expanded_metrics",
    "sim.seed_pair": "sim.seed_list",
}
RETIRED_CONFIG_KEYS.update(
    {
        f"analysis.{key}": "stage preconditions and canonical orchestration"
        for key in (
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
        )
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses (schema)
# ─────────────────────────────────────────────────────────────────────────────

SEED_LIST_LENGTHS_BY_COMMAND: dict[str, int] = {
    "run": 1,
    "analyze": 1,
    "two-seed": 2,
    "two-seed-pipeline": 2,
}


def expected_seed_list_length(command: str) -> int | None:
    """Return the expected root-list length for a CLI command."""

    return SEED_LIST_LENGTHS_BY_COMMAND.get(command)


@dataclass
class IOConfig:
    """File-system locations for the application."""

    results_dir_prefix: Path = Path("results")
    analysis_subdir: str = "analysis"


@dataclass
class RNGConfig:
    """Versioned deterministic random-stream contract."""

    scheme_version: int = 2
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
    max_shuffles_per_root_k: int | None = None
    projected_games_per_second: float | None = None


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
    delta_seed_stability: float = 0.03
    joint_discrepancy_alpha: float = 0.05
    matched_count_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)


@dataclass
class ArtifactContractConfig:
    """Versions participating in artifact validation and cache freshness."""

    artifact_contract_version: int = 2
    estimand_version: int = 1
    schema_version: int = 1
    baseline_version: int = 1
    k_support_version: int = 1
    weighting_version: int = 1
    conditioning_version: int = 1
    multiplicity_version: int = 1
    candidate_family_version: int = 1


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
    exactly two seeds. ``seed`` is the active root for single-root workflows
    and for naming root-suffixed result directories.
    """

    n_players_list: list[int] = field(default_factory=lambda: [5])
    seed: int = 0
    seed_list: list[int] | None = None
    """Explicit seed list (len 1 for single-seed, len 2 for two-seed)."""
    expanded_metrics: bool = False
    row_dir: Path | None = None
    metric_chunk_dir: Path | None = None
    per_n: dict[int, "SimConfig"] = field(default_factory=dict)
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
        raise ValueError(
            f"sim.seed_list must be set for orchestration requiring {expected_len} seeds"
        )

    def populate_seed_list(self, expected_len: int) -> list[int]:
        """Populate ``seed_list`` based on current config and return it."""
        seeds = self.resolve_seed_list(expected_len)
        self.seed_list = list(seeds)
        if expected_len in {1, 2}:
            self.seed = seeds[0]
        return seeds

    def require_seed_pair(self) -> tuple[int, int]:
        """Return the two canonical roots or raise when support is incomplete."""
        if self.seed_list is not None:
            if len(self.seed_list) != 2:
                raise ValueError(
                    f"sim.seed_list must contain exactly two seeds, got {self.seed_list!r}"
                )
            return (self.seed_list[0], self.seed_list[1])
        raise ValueError("sim.seed_list must contain exactly two seeds")


@dataclass
class AnalysisConfig:
    """Analysis-stage parameters that remain outside typed method settings."""

    disable_rng_diagnostics: bool = False
    """Disable RNG diagnostics in the root-local workflow."""

    n_jobs: int = 1
    """Parallel worker setting (`0` => os.cpu_count(), `>0` => explicit)."""
    mp_start_method: str | None = None
    """Multiprocessing start method for analysis executors."""
    progress_logging: ProgressLogConfig = field(default_factory=ProgressLogConfig)
    log_level: str = "INFO"
    results_glob: str = "*_players"
    # Optional outputs block may be provided in YAML
    # outputs:
    #   curated_rows_name: "game_rows.parquet"
    #   metrics_name: "metrics.parquet"
    #   manifest_name: "manifest.jsonl"
    outputs: dict[str, Any] = field(default_factory=dict)
    game_stats_margin_thresholds: tuple[int, ...] = (500, 1000)
    """Victory-margin thresholds used by game stats and rare-event summaries."""
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
class TrueSkillConfig:
    """Hyperparameters for TrueSkill updates."""

    beta: float = 25.0
    tau: float = 0.1
    draw_probability: float = 0.0


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
    min_candidate_completion_rate: float = 0.99
    max_attempt_multiplier: float = 2.0
    total_game_cap: int | None = 100_000_000
    allow_single_root: bool = True


@dataclass
class HGBConfig:
    """Finite-grid predictive-association settings for HGB exploration."""

    max_depth: int = 6
    n_estimators: int = 300
    heldout_folds: int = 5
    permutation_repeats: int = 10
    future_proposal_limit: int = 100


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
    _stage_layout: "StageLayout | None" = field(default=None, init=False, repr=False, compare=False)
    _code_identity: Any | None = field(default=None, init=False, repr=False, compare=False)
    _run_lineage_sha256: str | None = field(default=None, init=False, repr=False, compare=False)

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

    def freshness_key(self) -> dict[str, Any]:
        """Return the versioned statistical contract used by completion stamps."""

        from farkle.utils.schema_helpers import (
            OUTCOME_SCHEMA_VERSION,
            TOURNAMENT_METHOD_VERSION,
        )

        contract = self.artifact_contract
        weights = self.k_aggregation.k_weights
        normalized_counts: set[int | str] = set()
        for value in self.sim.n_players_list:
            try:
                normalized_counts.add(int(value))
            except (TypeError, ValueError):
                normalized_counts.add(str(value))
        ordered_counts = sorted(
            normalized_counts,
            key=lambda value: (isinstance(value, str), str(value)),
        )
        return {
            "artifact_contract_version": contract.artifact_contract_version,
            "estimand_version": contract.estimand_version,
            "schema_version": contract.schema_version,
            "rng_scheme_version": self.rng.scheme_version,
            "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
            "tournament_method_version": TOURNAMENT_METHOD_VERSION,
            "baseline_version": contract.baseline_version,
            "k_support_version": contract.k_support_version,
            "weighting_version": contract.weighting_version,
            "conditioning_version": contract.conditioning_version,
            "multiplicity_version": contract.multiplicity_version,
            "candidate_family_version": contract.candidate_family_version,
            "baseline": "chance_rate_by_k",
            "required_player_counts": ordered_counts,
            "k_aggregation_method": self.k_aggregation.method,
            "k_weights": (
                None
                if weights is None
                else {str(k): float(value) for k, value in sorted(weights.items())}
            ),
            "conditioning": "unconditional_default",
            "multiplicity": "holm_h2h",
        }

    def set_stage_layout(self, layout: "StageLayout") -> None:
        """Override the resolved stage layout (used by CLI orchestration)."""

        self._stage_layout = layout

    def validate_statistical_contract(self, *, require_two_roots: bool = False) -> None:
        """Validate locked production settings before expensive work is scheduled."""

        _validate_statistical_contract(self, require_two_roots=require_two_roots)

    def stage_dir(
        self,
        key: str,
        *,
        required_by: str | None = "analysis pipeline",
        create: bool = False,
    ) -> Path:
        """Return the resolved stage directory for ``key`` without implicit I/O."""

        folder = self.stage_layout.folder_for(key)
        if folder is None:
            requirement = f" required by {required_by}" if required_by else ""
            raise KeyError(f"Stage {key!r} is not registered in the resolved layout{requirement}.")
        stage_root = self.analysis_dir / folder
        if create:
            stage_root.mkdir(parents=True, exist_ok=True)
        return stage_root

    def stage_subdir(
        self,
        key: str,
        *parts: str | Path,
        create: bool = False,
    ) -> Path:
        """Resolve a stage root or nested subdirectory under ``analysis_dir``."""

        stage_root = self.stage_dir(key, create=create)
        path = stage_root.joinpath(*map(Path, parts)) if parts else stage_root
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def scope_dir(
        self,
        stage: str,
        scope: ArtifactScope | str,
        *,
        k: int | None = None,
        create: bool = False,
    ) -> Path:
        """Resolve one canonical logical scope beneath an active stage.

        ``by_k`` is the only scope that accepts a player count. Requiring it in
        the API prevents the former all-k integer sentinel from entering paths.
        """

        parts = self._scope_parts(scope, k=k)
        path = self.stage_dir(stage, create=create).joinpath(*parts)
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
        create_parent: bool = False,
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

    def by_k_dir(self, stage: str, k: int, *, create: bool = False) -> Path:
        """Return the canonical per-player-count directory for ``stage``."""

        return self.scope_dir(stage, ArtifactScope.BY_K, k=k, create=create)

    def concat_ks_dir(self, stage: str, *, create: bool = False) -> Path:
        """Return the row-preserving cross-k concatenation directory."""

        return self.scope_dir(stage, ArtifactScope.CONCAT_KS, create=create)

    def across_k_dir(self, stage: str, *, create: bool = False) -> Path:
        """Return the directory for declared common-support cross-k estimates."""

        return self.scope_dir(stage, ArtifactScope.ACROSS_K, create=create)

    def cross_seed_dir(self, stage: str, *, create: bool = False) -> Path:
        """Return the directory for root-combination and stability artifacts."""

        return self.scope_dir(stage, ArtifactScope.CROSS_SEED, create=create)

    def diagnostics_dir(self, stage: str, *, create: bool = False) -> Path:
        """Return the non-estimand diagnostic directory for ``stage``."""

        return self.scope_dir(stage, ArtifactScope.DIAGNOSTICS, create=create)

    def h2h_2p_dir(self, stage: str, *, create: bool = False) -> Path:
        """Return the explicitly two-player finalist H2H directory."""

        return self.scope_dir(stage, ArtifactScope.H2H_2P, create=create)

    def ingest_block_dir(self, k: int) -> Path:
        """Directory holding ingest artifacts for ``k`` players."""

        return self.by_k_dir("ingest", k)

    def curate_block_dir(self, k: int) -> Path:
        """Directory holding curated artifacts for ``k`` players."""

        return self.by_k_dir("curate", k)

    @property
    def combine_partitioned_dir(self) -> Path:
        """Directory holding partitioned combined curated rows."""

        return self.concat_ks_dir("combine") / "all_ingested_rows_partitioned"

    def metrics_per_k_dir(self, k: int) -> Path:
        """Directory holding metrics artifacts for ``k`` players."""
        path = self.by_k_dir("metrics", k)
        return path

    @property
    def game_stats_stage_dir(self) -> Path:
        """Stage directory for game-stat analytics."""

        return self.stage_subdir("game_stats")

    @property
    def rng_stage_dir(self) -> Path:
        """Stage directory for RNG diagnostics."""

        return self.stage_subdir("rng_diagnostics")

    @property
    def ingest_stage_dir(self) -> Path:
        """Stage root for ingest outputs."""

        return self.stage_subdir("ingest")

    @property
    def curate_stage_dir(self) -> Path:
        """Stage root for curated row outputs."""

        return self.stage_subdir("curate")

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
    def hgb_stage_dir(self) -> Path:
        """Stage root for histogram gradient boosting outputs."""

        return self.stage_subdir("hgb")

    def hgb_per_k_dir(self, k: int) -> Path:
        """Per-player-count HGB output directory."""

        return self.by_k_dir("hgb", k)

    def hgb_importance_path(self, k: int) -> Path:
        """Held-out permutation associations for one player count."""

        return self.hgb_per_k_dir(k) / f"feature_importance_{k}p.parquet"

    def hgb_predictive_scores_path(self, k: int) -> Path:
        """Out-of-sample predictions for held-out strategy configurations."""

        return self.hgb_per_k_dir(k) / f"heldout_predictive_scores_{k}p.parquet"

    def hgb_fold_metrics_path(self, k: int) -> Path:
        """Per-fold HGB predictive scores and finite-grid support."""

        return self.hgb_per_k_dir(k) / f"heldout_fold_metrics_{k}p.parquet"

    def hgb_future_proposals_path(self) -> Path:
        """Candidate manifest reserved for a future simulation run."""

        return self.across_k_dir("hgb") / "future_simulation_proposals.parquet"

    @property
    def screening_stage_dir(self) -> Path:
        """Stage root for descriptive performance screening outputs."""

        return self.stage_subdir("screening")

    def root_input_stage_folder(self, key: str) -> str | None:
        """Return the canonical stage folder used by root-owned input artifacts."""

        return self.stage_layout.folder_for(key)

    @property
    def data_dir(self) -> Path:
        """Root directory for curated data under the curate stage."""

        return self.stage_subdir("curate")

    def n_dir(self, n: int) -> Path:
        """Convenience accessor for a specific ``<n>_players`` directory."""
        return self.results_root / f"{n}_players"

    def simulation_row_dir(self, n: int) -> Path | None:
        """Return the canonical simulation row-shard directory for player count ``n``."""

        raw_value = self.sim.row_dir
        if raw_value is None:
            return None
        raw_text = str(raw_value)
        placeholders = {"n": n, "n_players": n, "p": f"{n}p"}
        try:
            formatted = raw_text.format(**placeholders)
        except KeyError as exc:
            raise ValueError(f"unknown simulation row-dir placeholder: {exc.args[0]}") from exc
        path = Path(formatted)
        if formatted == raw_text and path.name and not path.name.startswith(f"{n}p"):
            path = path.parent / f"{n}p_{path.name}"
        if path.is_absolute():
            return path
        return self.n_dir(n) / path

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

    # —— Output filenames and standard derived locations ——
    def game_stats_output_path(self, name: str) -> Path:
        """Preferred path for across-k game-stat estimates."""

        path = self.across_k_dir("game_stats") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def game_stats_concat_path(self, name: str) -> Path:
        """Path for row-preserving concatenated per-k game-stat summaries."""

        path = self.concat_ks_dir("game_stats") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def exact_roll_distribution_path(self) -> Path:
        """Exact ordered-roll distribution diagnostic for one through six dice."""

        return self.game_stats_stage_dir / "diagnostics" / "roll_outcome_distribution_exact.parquet"

    def exact_roll_summary_path(self) -> Path:
        """Exact ordered-roll summary diagnostic for one through six dice."""

        return self.game_stats_stage_dir / "diagnostics" / "roll_summary_exact.parquet"

    def rng_output_path(self, name: str) -> Path:
        """Preferred path for combined RNG diagnostics."""

        path = self.diagnostics_dir("rng_diagnostics") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def rng_input_path(self, name: str) -> Path:
        """Resolve an RNG diagnostic artifact in its canonical scope."""

        return self.input_scope_path("rng_diagnostics", ArtifactScope.DIAGNOSTICS, name)

    def metrics_all_player_batch_path(self, k: int) -> Path:
        """Canonical unconditional player-exposure batch metrics for ``k`` players."""

        return self.metrics_per_k_dir(k) / "all_player_batch_metrics.parquet"

    def performance_by_k_path(self, k: int) -> Path:
        """Canonical chance-adjusted performance estimates for ``k`` players."""

        return self.metrics_per_k_dir(k) / "performance.parquet"

    def performance_across_k_path(self) -> Path:
        """Canonical complete-support equal-k performance estimates."""

        return self.across_k_dir("metrics") / "performance_equal_k.parquet"

    def performance_bootstrap_path(self) -> Path:
        """Joint batch-resampling summaries for equal-k performance."""

        return self.across_k_dir("metrics") / "performance_bootstrap.parquet"

    def performance_control_contrasts_path(self) -> Path:
        """Across-k contrasts against declared control strategies."""

        return self.across_k_dir("metrics") / "performance_control_contrasts.parquet"

    def performance_player_count_effects_path(self) -> Path:
        """Chance-relative player-count effect and rank diagnostics."""

        return self.diagnostics_dir("metrics") / "player_count_effects.parquet"

    def root_combined_performance_by_k_path(self, k: int) -> Path:
        """Root-specific and raw-count-combined performance for one k."""

        return (
            self.cross_seed_dir("root_stability")
            / f"performance_root_combination_{int(k)}p.parquet"
        )

    def root_combined_performance_across_k_path(self) -> Path:
        """Root-specific and combined declared-k performance scores."""

        return (
            self.cross_seed_dir("root_stability") / "performance_root_combination_across_k.parquet"
        )

    def root_discrepancies_path(self) -> Path:
        """Raw, standardized, and threshold-scaled root differences."""

        return self.cross_seed_dir("root_stability") / "root_discrepancies.parquet"

    def root_joint_discrepancy_path(self) -> Path:
        """Joint maximum-discrepancy diagnostic summary."""

        return self.cross_seed_dir("root_stability") / "root_joint_discrepancy.parquet"

    def root_rank_stability_path(self) -> Path:
        """Between-root rank correlation and movement summary."""

        return self.cross_seed_dir("root_stability") / "root_rank_stability.parquet"

    def root_top_n_stability_path(self) -> Path:
        """Between-root top-N overlap diagnostics."""

        return self.cross_seed_dir("root_stability") / "root_top_n_stability.parquet"

    def root_bootstrap_top_n_inclusion_path(self) -> Path:
        """Root-specific bootstrap top-N inclusion probabilities."""

        return self.cross_seed_dir("root_stability") / "root_bootstrap_top_n_inclusion.parquet"

    def root_control_movement_path(self) -> Path:
        """Declared-control rank and performance movement."""

        return self.cross_seed_dir("root_stability") / "root_control_movement.parquet"

    def root_shortlist_changes_path(self) -> Path:
        """Root-specific and combined practical-shortlist membership."""

        return self.cross_seed_dir("root_stability") / "root_shortlist_changes.parquet"

    def root_matched_count_convergence_path(self) -> Path:
        """Matched cumulative-batch convergence diagnostics."""

        return self.cross_seed_dir("root_stability") / "root_matched_count_convergence.parquet"

    def root_half_drift_path(self) -> Path:
        """First-half versus second-half within-root drift diagnostics."""

        return self.cross_seed_dir("root_stability") / "root_half_drift.parquet"

    def seat_batch_counts_path(self, k: int) -> Path:
        """Canonical seat wins and exposures by root, k, batch, strategy, and seat."""

        return self.metrics_per_k_dir(k) / "seat_batch_counts.parquet"

    def seat_effects_by_k_path(self, k: int) -> Path:
        """Canonical strategy-specific within-k seat effects."""

        return self.metrics_per_k_dir(k) / "seat_effects.parquet"

    def seat_population_by_k_path(self, k: int) -> Path:
        """Canonical population-wide within-k seat effects."""

        return self.metrics_per_k_dir(k) / "seat_population_effects.parquet"

    def seat_standardized_across_k_path(self) -> Path:
        """Declared-weight seat effects over identical common k support."""

        return self.across_k_dir("metrics") / "seat_effects_standardized_across_k.parquet"

    def seat_exposure_mixture_diagnostic_path(self) -> Path:
        """Secondary exposure-weighted cross-k seat diagnostic."""

        return self.metrics_stage_dir / "diagnostics" / "seat_exposure_mixture.parquet"

    def seat_selfplay_diagnostic_path(self) -> Path:
        """Self-play first-seat diagnostic path."""

        return self.metrics_stage_dir / "diagnostics" / "seat_selfplay_p1.parquet"

    def seat_mirrored_diagnostic_path(self) -> Path:
        """Paired mirrored-game diagnostic path."""

        return self.metrics_stage_dir / "diagnostics" / "seat_mirrored_games.parquet"

    def trueskill_candidate_contribution_path(self) -> Path:
        """Complete-support percentile contribution used only for screening."""

        return self.across_k_dir("trueskill") / "candidate_percentile_contribution.parquet"

    def trueskill_rating_path(self, k: int, *, root_seed: int | None = None) -> Path:
        """Canonical sequential-rating artifact for one root/player-count cell."""

        seed = int(self.sim.seed if root_seed is None else root_seed)
        return self.by_k_dir("trueskill", int(k)) / f"ratings_{int(k)}_seed{seed}.parquet"

    def trueskill_screening_diagnostics_path(self) -> Path:
        """Tau, game-order, and held-out predictive diagnostics."""

        return self.diagnostics_dir("trueskill") / "screening_diagnostics.parquet"

    def h2h_candidate_family_path(self) -> Path:
        """Frozen H2H candidate membership and admission provenance."""

        return self.h2h_2p_dir("candidate_freeze") / "candidate_family.parquet"

    def h2h_candidate_family_manifest_path(self) -> Path:
        """Immutable candidate-family identity and workload summary."""

        return self.h2h_2p_dir("candidate_freeze") / "candidate_family.json"

    def h2h_power_plan_path(self) -> Path:
        """Score-test power and root/order allocation plan."""

        return self.h2h_2p_dir("h2h_power") / "power_plan.json"

    def h2h_block_manifest_path(self) -> Path:
        """Immutable pair/root/order simulation block manifest."""

        return self.h2h_2p_dir("h2h_power") / "block_manifest.parquet"

    def h2h_execution_state_path(self) -> Path:
        """Resumable H2H execution lifecycle independent of the power plan."""

        return self.h2h_2p_dir("h2h_execute") / "execution_state.json"

    def h2h_block_results_dir(self) -> Path:
        """Directory of atomic pair/root/order block checkpoints."""

        return self.h2h_2p_dir("h2h_execute") / "blocks"

    def h2h_block_result_path(self, pair_id: int, root_seed: int, order: int) -> Path:
        """One immutable H2H pair/root/order checkpoint path."""

        return self.h2h_block_results_dir() / (
            f"pair_{int(pair_id):06d}_root_{int(root_seed)}_order_{int(order)}.parquet"
        )

    def h2h_order_counts_path(self) -> Path:
        """Validated row-preserving union of completed H2H blocks."""

        return self.h2h_2p_dir("h2h_execute") / "root_order_counts.parquet"

    def h2h_combined_order_counts_path(self) -> Path:
        """Raw H2H counts combined across roots within each seat order."""

        return self.h2h_2p_dir("h2h_inference") / "combined_order_counts.parquet"

    def h2h_pairwise_inference_path(self) -> Path:
        """Seat-adjusted score inference and multiplicity decisions."""

        return self.h2h_2p_dir("h2h_inference") / "pairwise_inference.parquet"

    def h2h_root_pairwise_diagnostics_path(self) -> Path:
        """Root-specific seat-adjusted score diagnostics."""

        return self.h2h_2p_dir("h2h_inference") / "root_pairwise_diagnostics.parquet"

    def h2h_root_agreement_path(self) -> Path:
        """Fixed-root H2H effect discrepancy and decision agreement."""

        return self.h2h_2p_dir("h2h_inference") / "root_decision_agreement.parquet"

    def h2h_dominance_edges_path(self) -> Path:
        """Practical and statistical directed dominance edges."""

        return self.h2h_2p_dir("h2h_digest") / "dominance_edges.parquet"

    def h2h_cycle_groups_path(self) -> Path:
        """Strongly connected practical and statistical cycle groups."""

        return self.h2h_2p_dir("h2h_digest") / "cycle_groups.parquet"

    def h2h_dominance_fronts_path(self) -> Path:
        """Condensation-DAG fronts with display-only within-front order."""

        return self.h2h_2p_dir("h2h_digest") / "dominance_fronts.parquet"

    def h2h_dominance_summary_path(self) -> Path:
        """Cycle, unresolved, and unique-best claim summary."""

        return self.h2h_2p_dir("h2h_digest") / "dominance_summary.json"

    def structure_agreement_pairs_path(self) -> Path:
        """Selection-conditioned pair-level method agreement."""

        return self.h2h_2p_dir("agreement") / "selection_conditioned_pairs.parquet"

    def structure_agreement_summary_path(self) -> Path:
        """Canonical win-rate, TrueSkill, H2H, and root-stability agreement."""

        return self.h2h_2p_dir("agreement") / "agreement_summary.json"

    def structure_report_json_path(self) -> Path:
        """Machine-readable canonical structure report."""

        return self.diagnostics_dir("reporting") / "structure_report.json"

    def structure_report_markdown_path(self) -> Path:
        """Human-readable canonical structure report."""

        return self.diagnostics_dir("reporting") / "structure_report.md"

    def structure_report_plot_path(self) -> Path:
        """Display-only tournament screening score plot."""

        return self.diagnostics_dir("reporting") / "tournament_screening_scores.png"

    def migration_report_path(self) -> Path:
        """Non-destructive inventory of ignored retired artifacts."""

        return self.diagnostics_dir("reporting") / "migration_report.json"

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

    @property
    def curated_parquet(self) -> Path:
        """Canonical row-preserving cross-k curated parquet."""

        return self.input_scope_path(
            "combine", ArtifactScope.CONCAT_KS, "all_ingested_rows.parquet"
        )

    @property
    def curated_dataset(self) -> Path:
        """Canonical partitioned row-preserving cross-k curated dataset."""

        return self.input_scope_path(
            "combine",
            ArtifactScope.CONCAT_KS,
            "all_ingested_rows_partitioned",
        )

    def screening_path(self, filename: str = "descriptive_screening.parquet") -> Path:
        """Return a canonical descriptive-screening artifact path."""

        return self.screening_stage_dir / filename

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

    def combined_rows_by_k(self, n: int) -> Path:
        """Canonical normalized row partition for one player count."""

        return self.combine_partitioned_dir / f"{int(n)}p_part-00000.parquet"

    def combined_manifest_path(self) -> Path:
        """Path to the manifest accompanying ``curated_parquet``."""

        return self.input_scope_path(
            "combine", ArtifactScope.CONCAT_KS, "all_ingested_rows.manifest.jsonl"
        )


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
    if "metrics" in data:
        raise ValueError(
            "Retired config section 'metrics'; canonical all-player, performance, "
            "and seat estimators have fixed schemas"
        )
    top_level_sections = {
        "io": IOConfig,
        "sim": SimConfig,
        "analysis": AnalysisConfig,
        "ingest": IngestConfig,
        "combine": CombineConfig,
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
            raise ValueError(f"Unknown key(s) in config section {section_name!r}: {formatted}")
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
                        f"Unknown key(s) in config section sim.per_n[{key!r}]: {formatted}"
                    )


def _validate_seed_sources(
    sim: SimConfig,
    *,
    seed_provided: bool,
    seed_list_provided: bool,
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
        if seed_list_provided and seed_provided:
            LOGGER.warning(
                "%s: sim.seed_list overrides sim.seed for deterministic root naming",
                context,
                extra={
                    "stage": "config",
                    "seed_list": list(sim.seed_list),
                    "seed": sim.seed,
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

    seed_provided = False
    seed_list_provided = False
    per_n_seed_provided: dict[int, bool] = {}
    per_n_seed_list_provided: dict[int, bool] = {}
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
        resolved: dict[str, Any] = {}
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

            if (
                annotation is not None
                and get_origin(annotation) is tuple
                and isinstance(val, (list, tuple))
            ):
                val = tuple(val)

            # Path coercion (works for nested too because we use type hints)
            if (isinstance(current, Path) or _annotation_contains(annotation, Path)) and isinstance(
                val, (str, Path)
            ):
                val = Path(val)

            resolved[f.name] = val
        return cls(**resolved)

    cfg = AppConfig(
        io=build(IOConfig, data.get("io", {})),
        sim=build(SimConfig, data.get("sim", {})),
        analysis=build(AnalysisConfig, data.get("analysis", {})),
        ingest=build(IngestConfig, data.get("ingest", {})),
        combine=build(CombineConfig, data.get("combine", {})),
        trueskill=build(TrueSkillConfig, data.get("trueskill", {})),
        head2head=build(Head2HeadConfig, data.get("head2head", {})),
        hgb=build(HGBConfig, data.get("hgb", {})),
        orchestration=build(OrchestrationConfig, data.get("orchestration", {})),
        rng=build(RNGConfig, data.get("rng", {})),
        screening=build(ScreeningConfig, data.get("screening", {})),
        batching=build(BatchingConfig, data.get("batching", {})),
        robustness=build(RobustnessConfig, data.get("robustness", {})),
        artifact_contract=build(ArtifactContractConfig, data.get("artifact_contract", {})),
        k_aggregation=build(KAggregationConfig, data.get("k_aggregation", {})),
    )
    _normalize_seed_list(cfg.sim)
    _validate_seed_sources(
        cfg.sim,
        seed_provided=seed_provided,
        seed_list_provided=seed_list_provided,
        expected_seed_len=seed_list_len,
        context="load_app_config",
    )
    if cfg.sim.per_n:
        for key, sim_cfg in cfg.sim.per_n.items():
            key_int = int(key)
            _normalize_seed_list(sim_cfg)
            _validate_seed_sources(
                sim_cfg,
                seed_provided=per_n_seed_provided.get(key_int, False),
                seed_list_provided=per_n_seed_list_provided.get(key_int, False),
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
        retired_key = f"{section_name}.{option}"
        if retired_key in RETIRED_CONFIG_KEYS:
            replacement = RETIRED_CONFIG_KEYS[retired_key]
            raise ValueError(f"Retired config key {retired_key!r}; use {replacement!r} instead")
        section = getattr(cfg, section_name)
        if not hasattr(section, option):
            raise AttributeError(f"Unknown option {option!r} in section {section_name!r}")
        current = getattr(section, option)
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

    public = {
        item.name: (
            dataclasses.asdict(cast(Any, value)) if dataclasses.is_dataclass(value) else value
        )
        for item in dataclasses.fields(AppConfig)
        if item.init
        for value in (getattr(cfg, item.name),)
    }
    return _stringify_paths_for_serialization(public)


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


def _project_effective_config(
    payload: Mapping[str, Any], scope_paths: Sequence[str]
) -> dict[str, Any]:
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
    if cfg.rng.scheme_version != 2 or cfg.rng.bit_generator != "PCG64DXSM":
        raise ValueError("rng must use scheme_version=2 and bit_generator='PCG64DXSM'")
    contract_versions = dataclasses.asdict(cfg.artifact_contract)
    if any(int(value) < 1 for value in contract_versions.values()):
        raise ValueError("artifact_contract versions must all be positive integers")
    if cfg.artifact_contract.artifact_contract_version != 2:
        raise ValueError("artifact_contract.artifact_contract_version is locked at 2")
    if not 0.0 < cfg.screening.resolution_delta < 1.0:
        raise ValueError("screening.resolution_delta must be between 0 and 1")
    if cfg.screening.interval_confidence != 0.95:
        raise ValueError("screening.interval_confidence is locked at 0.95")
    if cfg.screening.bootstrap_replicates < 1:
        raise ValueError("screening.bootstrap_replicates must be positive")
    if cfg.screening.candidate_contribution_size < 1:
        raise ValueError("screening.candidate_contribution_size must be positive")
    if not cfg.robustness.report_pareto or not cfg.robustness.report_maximin:
        raise ValueError("robustness must report both Pareto membership and maximin leadership")
    if cfg.robustness.delta_seed_stability <= 0.0:
        raise ValueError("robustness.delta_seed_stability must be positive")
    if not 0.0 < cfg.robustness.joint_discrepancy_alpha < 1.0:
        raise ValueError("robustness.joint_discrepancy_alpha must be between 0 and 1")
    fractions = cfg.robustness.matched_count_fractions
    if (
        not fractions
        or any(not 0.0 < fraction <= 1.0 for fraction in fractions)
        or tuple(sorted(set(fractions))) != fractions
        or fractions[-1] != 1.0
    ):
        raise ValueError(
            "robustness.matched_count_fractions must be unique increasing values in "
            "(0, 1] ending at 1"
        )
    if cfg.screening.max_shuffles_per_root_k is not None and (
        isinstance(cfg.screening.max_shuffles_per_root_k, bool)
        or not isinstance(cfg.screening.max_shuffles_per_root_k, int)
        or cfg.screening.max_shuffles_per_root_k < 1
    ):
        raise ValueError("screening.max_shuffles_per_root_k must be positive when configured")
    if cfg.screening.projected_games_per_second is not None and (
        not math.isfinite(cfg.screening.projected_games_per_second)
        or cfg.screening.projected_games_per_second <= 0.0
    ):
        raise ValueError("screening.projected_games_per_second must be positive when configured")
    if cfg.batching.target_batches != 100 or cfg.batching.min_shuffles_per_batch < 30:
        raise ValueError(
            "batching requires exactly 100 equal batches with at least 30 shuffles per batch"
        )

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
    if h2h.delta_equivalence is not None and not 0.0 < h2h.delta_equivalence < 1.0:
        raise ValueError("head2head.delta_equivalence must be between 0 and 1 when configured")
    sensitivity = tuple(float(delta) for delta in h2h.sensitivity_deltas)
    if (
        not sensitivity
        or len(set(sensitivity)) != len(sensitivity)
        or any(delta <= 0.0 for delta in sensitivity)
        or h2h.practical_delta not in sensitivity
        or 0.04 not in sensitivity
    ):
        raise ValueError(
            "head2head.sensitivity_deltas must be unique positive values containing "
            "the practical delta and 0.04"
        )
    if tuple(float(value) for value in h2h.seat1_advantage_scenarios) != (
        0.0,
        0.03,
        0.06,
    ):
        raise ValueError("head2head.seat1_advantage_scenarios is locked to 0, 0.03, 0.06")
    if h2h.candidate_cap is not None and h2h.candidate_cap < 2:
        raise ValueError("head2head.candidate_cap must be at least 2")
    if h2h.candidate_cap_policy != "balanced-tail":
        raise ValueError("head2head.candidate_cap_policy must be 'balanced-tail'")
    if not 0.0 <= h2h.min_candidate_completion_rate <= 1.0:
        raise ValueError("head2head.min_candidate_completion_rate must be between 0 and 1")
    if not math.isfinite(h2h.max_attempt_multiplier) or h2h.max_attempt_multiplier < 1.0:
        raise ValueError("head2head.max_attempt_multiplier must be finite and at least 1")
    if h2h.total_game_cap is not None and h2h.total_game_cap <= 0:
        raise ValueError("head2head.total_game_cap must be positive when configured")

    if cfg.hgb.heldout_folds < 2:
        raise ValueError("hgb.heldout_folds must be at least 2")
    if cfg.hgb.permutation_repeats < 1:
        raise ValueError("hgb.permutation_repeats must be positive")
    if cfg.hgb.future_proposal_limit < 0:
        raise ValueError("hgb.future_proposal_limit must not be negative")
    if cfg.hgb.max_depth < 1 or cfg.hgb.n_estimators < 1:
        raise ValueError("hgb model depth and iteration count must be positive")


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
            "freshness": cfg.freshness_key(),
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
    "TrueSkillConfig",
    "Head2HeadConfig",
    "HGBConfig",
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
