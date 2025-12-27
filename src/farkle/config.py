# src/farkle/config.py
"""Configuration schemas and helpers for the Farkle analysis pipeline.

Defines dataclasses describing I/O, simulation, and analysis settings and
includes utilities for loading and validating YAML-based application configs.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, get_args, get_origin, get_type_hints

import yaml  # type: ignore[import-untyped]

from farkle.utils.yaml_helpers import expand_dotted_keys

# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses (schema)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class IOConfig:
    """File-system locations for the application."""

    results_dir: Path = Path("results")
    # Keep this as a plain string in YAML to avoid Path(dict) mistakes.
    append_seed: bool = True
    analysis_subdir: str = "analysis"
    meta_analysis_dir: Path | None = None


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
class SimConfig:
    """Simulation parameters."""

    n_players_list: list[int] = field(default_factory=lambda: [5])
    num_shuffles: int = 100
    seed: int = 0
    expanded_metrics: bool = False
    row_dir: Path | None = None
    per_n: dict[int, "SimConfig"] = field(default_factory=dict)
    power_method: str = "bh"
    recompute_num_shuffles: bool = True
    power_design: PowerDesign = field(default_factory=PowerDesign)
    n_jobs: int | None = None
    desired_sec_per_chunk: int = 10
    ckpt_every_sec: int = 30

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


@dataclass
class AnalysisConfig:
    """Analysis-stage parameters controlling downstream analytics."""

    run_trueskill: bool = True
    run_head2head: bool = True
    run_game_stats: bool = True
    run_hgb: bool = True
    run_frequentist: bool = False
    """Plan step 6: frequentist / MDD-based tiering (tiering_report)."""
    run_post_h2h_analysis: bool = False
    """Execute the post head-to-head clean-up pass (plan step 5)."""

    run_agreement: bool = False
    """Generate the agreement analysis between model outputs (plan step 8)."""

    run_report: bool = True
    """Emit the final report artifacts (plan step 9)."""

    n_jobs: int = 1
    log_level: str = "INFO"
    results_glob: str = "*_players"
    meta_random_if_I2_gt: float = 25.0
    """Switch to random-effects pooling once I^2 crosses this threshold."""
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
    tiering_seeds: list[int] | None = None
    """Explicit seeds to use when running the tiering report."""

    tiering_z_star: float = 1.645
    tiering_min_gap: float | None = None
    tiering_weights_by_k: dict[int, float] | None = None
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

    @property
    def run_tiering_report(self) -> bool:
        """Deprecated alias for run_frequentist; kept for config compatibility."""
        return self.run_frequentist

    @run_tiering_report.setter
    def run_tiering_report(self, value: bool) -> None:
        """Propagate legacy setter calls to ``run_frequentist``."""
        self.run_frequentist = bool(value)


@dataclass
class IngestConfig:
    """Ingestion tuning for streaming parquet writes."""

    row_group_size: int = 64_000
    parquet_codec: str = "snappy"
    batch_rows: int = 100_000
    n_jobs: int = 1


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


@dataclass
class Head2HeadConfig:
    """Configuration for head-to-head tournament simulations."""

    n_jobs: int = 4
    games_per_pair: int = 10_000
    fdr_q: float = 0.02
    # If you ever add a nested design block here, it will still parse:
    bonferroni_design: dict[str, Any] = field(default_factory=dict)


@dataclass
class HGBConfig:
    """Hyperparameters for histogram-based gradient boosting models."""

    max_depth: int = 6
    n_estimators: int = 300


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
    # Computed at runtime; not part of user-provided YAML
    config_sha: str | None = field(default=None, init=False, repr=False, compare=False)

    # —— Paths ——
    @property
    def results_dir(self) -> Path:
        """Root directory where simulation outputs are written."""
        return self.io.results_dir

    @property
    def analysis_dir(self) -> Path:
        """Directory containing derived analysis artifacts."""
        return self.io.results_dir / self.io.analysis_subdir

    # Numbered analysis stage directories (created on access)
    def stage_subdir(self, name: str, *parts: str | Path) -> Path:
        """Resolve a stage root or nested subdirectory under ``analysis_dir``.

        Directories are created on access to keep downstream callers simple.
        """

        stage_root = self.analysis_dir / name
        stage_root.mkdir(parents=True, exist_ok=True)
        path = stage_root.joinpath(*map(Path, parts)) if parts else stage_root
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def per_k_subdir(self, stage: str, k: int) -> Path:
        """Stage helper that returns the ``<k>p`` folder under ``stage``."""

        return self.stage_subdir(stage, f"{k}p")

    def ingest_block_dir(self, k: int) -> Path:
        """Directory holding ingest artifacts for ``k`` players."""

        return self.per_k_subdir("00_ingest", k)

    def curate_block_dir(self, k: int) -> Path:
        """Directory holding curated artifacts for ``k`` players."""

        return self.per_k_subdir("01_curate", k)

    def combine_block_dir(self, k: int) -> Path:
        """Deprecated alias for :meth:`curate_block_dir`."""

        return self.curate_block_dir(k)

    def combine_pooled_dir(self, k: int | None = None) -> Path:
        """Directory holding pooled combine artifacts (legacy *k* kept for callers)."""

        return self.stage_subdir("02_combine", "pooled")

    def metrics_per_k_dir(self, k: int) -> Path:
        """Directory holding metrics artifacts for ``k`` players."""
        path = self.per_k_subdir("03_metrics", k)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def metrics_pooled_dir(self) -> Path:
        """Directory holding pooled metrics artifacts."""
        path = self.stage_subdir("03_metrics", "pooled")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def game_stats_stage_dir(self) -> Path:
        """Stage directory for game-stat analytics."""

        return self.stage_subdir("04_game_stats")

    @property
    def game_stats_pooled_dir(self) -> Path:
        """Pooled outputs for game-stat analytics."""

        return self.stage_subdir("04_game_stats", "pooled")

    @property
    def rng_stage_dir(self) -> Path:
        """Stage directory for RNG diagnostics."""

        return self.stage_subdir("05_rng")

    @property
    def rng_pooled_dir(self) -> Path:
        """Pooled outputs for RNG diagnostics."""

        return self.stage_subdir("05_rng", "pooled")

    @property
    def seed_summaries_stage_dir(self) -> Path:
        """Stage directory for per-seed summaries."""

        return self.stage_subdir("06_seed_summaries")

    def seed_summaries_dir(self, players: int) -> Path:
        """Directory holding seed summaries for ``players`` count."""

        return self.stage_subdir("06_seed_summaries", f"{players}p")

    @property
    def variance_stage_dir(self) -> Path:
        """Stage directory for variance analytics."""

        return self.stage_subdir("07_variance")

    @property
    def variance_pooled_dir(self) -> Path:
        """Pooled outputs for variance analytics."""

        return self.stage_subdir("07_variance", "pooled")

    @property
    def meta_stage_dir(self) -> Path:
        """Stage directory for meta-analysis outputs."""

        return self.stage_subdir("07_meta")

    def meta_per_k_dir(self, players: int) -> Path:
        """Primary per-player meta-analysis directory."""

        return self.stage_subdir("07_meta", f"{players}p")

    @property
    def meta_pooled_dir(self) -> Path:
        """Legacy pooled outputs for meta-analysis."""

        return self.stage_subdir("08_meta", "pooled")

    @property
    def agreement_stage_dir(self) -> Path:
        """Stage directory for cross-method agreement analytics."""

        return self.stage_subdir("13_agreement")

    @property
    def ingest_stage_dir(self) -> Path:
        return self.stage_subdir("00_ingest")

    @property
    def curate_stage_dir(self) -> Path:
        return self.stage_subdir("01_curate")

    @property
    def combine_stage_dir(self) -> Path:
        return self.stage_subdir("02_combine")

    @property
    def metrics_stage_dir(self) -> Path:
        return self.stage_subdir("03_metrics")

    @property
    def trueskill_stage_dir(self) -> Path:
        return self.stage_subdir("09_trueskill")

    @property
    def trueskill_pooled_dir(self) -> Path:
        return self.stage_subdir("09_trueskill", "pooled")

    @property
    def head2head_stage_dir(self) -> Path:
        return self.stage_subdir("10_head2head")

    @property
    def hgb_stage_dir(self) -> Path:
        return self.stage_subdir("11_hgb")

    def hgb_per_k_dir(self, k: int) -> Path:
        return self.per_k_subdir("11_hgb", k)

    @property
    def hgb_pooled_dir(self) -> Path:
        return self.stage_subdir("11_hgb", "pooled")

    @property
    def tiering_stage_dir(self) -> Path:
        return self.stage_subdir("12_tiering")

    @property
    def meta_analysis_dir(self) -> Path:
        """Directory containing per-seed summaries pooled across runs."""

        configured = self.io.meta_analysis_dir
        if configured is None:
            return self.analysis_dir
        meta_path = Path(configured)
        if meta_path.is_absolute():
            return meta_path
        # Anchor relative paths to the parent of the seed-suffixed results_dir
        return self.io.results_dir.parent / meta_path

    @property
    def data_dir(self) -> Path:
        """Root directory for curated data under ``01_curate``."""

        return self.curate_stage_dir

    def n_dir(self, n: int) -> Path:
        """Convenience accessor for a specific ``<n>_players`` directory."""
        return self.results_dir / f"{n}_players"

    def checkpoint_path(self, n: int) -> Path:
        """Path to a head-to-head checkpoint for ``n`` players."""
        return self.n_dir(n) / f"{n}p_checkpoint.pkl"

    def metrics_path(self, n: int) -> Path:
        """Path to the metrics parquet for ``n`` players."""
        return self.n_dir(n) / f"{n}p_metrics.parquet"

    # —— Ingest/streaming knobs ——
    @property
    def row_group_size(self) -> int:
        """Row-group size used when writing parquet outputs during ingest."""
        return self.ingest.row_group_size

    @property
    def parquet_codec(self) -> str:
        """Compression codec used for ingest parquet outputs."""
        return self.ingest.parquet_codec

    @property
    def n_jobs_ingest(self) -> int:
        """Parallel worker count for ingestion tasks."""
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
        """Preferred path for pooled metrics artifacts under ``03_metrics``."""

        filename = str(self.metrics_name if name is None else name)
        path = self.metrics_pooled_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def game_stats_output_path(self, name: str) -> Path:
        """Preferred path for pooled game-stat outputs."""

        path = self.game_stats_pooled_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def game_stats_input_path(self, name: str) -> Path:
        """Resolve a game-stat artifact with a legacy fallback."""

        return self._preferred_stage_path(self.game_stats_pooled_dir, self.analysis_dir, name)

    def rng_output_path(self, name: str) -> Path:
        """Preferred path for pooled RNG diagnostics."""

        path = self.rng_pooled_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def rng_input_path(self, name: str) -> Path:
        """Resolve an RNG diagnostic artifact with a legacy fallback."""

        return self._preferred_stage_path(self.rng_pooled_dir, self.analysis_dir, name)

    def variance_output_path(self, name: str) -> Path:
        """Preferred path for pooled variance analytics."""

        path = self.variance_pooled_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def variance_input_path(self, name: str) -> Path:
        """Resolve a variance artifact with a legacy fallback."""

        return self._preferred_stage_path(self.variance_pooled_dir, self.analysis_dir, name)

    def meta_output_path(self, players: int, name: str) -> Path:
        """Preferred path for per-player meta-analysis artifacts."""

        path = self.meta_per_k_dir(players) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def meta_input_path(self, players: int, name: str) -> Path:
        """Resolve a meta-analysis artifact with a legacy fallback."""

        preferred = self.meta_per_k_dir(players) / name
        if preferred.exists():
            return preferred

        for legacy_dir in (self.meta_pooled_dir, self.analysis_dir):
            legacy_path = legacy_dir / name
            if legacy_path.exists():
                return legacy_path

        return preferred

    def metrics_input_path(self, name: str | None = None) -> Path:
        """Resolve a pooled metrics artifact with a legacy fallback."""

        filename = str(self.metrics_name if name is None else name)
        preferred = self.metrics_output_path(filename)
        legacy = self.analysis_dir / filename
        if preferred.exists() or not legacy.exists():
            return preferred
        return legacy

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

    def _preferred_stage_path(self, stage_dir: Path, legacy_dir: Path, filename: str) -> Path:
        """Return *filename* within the stage dir, falling back to legacy when absent."""

        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_path = stage_dir / filename
        legacy_path = legacy_dir / filename
        if stage_path.exists() or not legacy_path.exists():
            return stage_path
        return legacy_path

    def agreement_output_path(self, players: int) -> Path:
        """Preferred path for agreement analytics for a given player count."""

        filename = f"agreement_{players}p.json"
        stage_dir = self.per_k_subdir("13_agreement", players)
        return self._preferred_stage_path(stage_dir, self.analysis_dir, filename)

    def trueskill_path(self, filename: str) -> Path:
        """Resolve a TrueSkill artifact path with legacy fallback."""
        pooled = filename.startswith("ratings_pooled")
        stage_dir = self.trueskill_pooled_dir if pooled else self.trueskill_stage_dir
        return self._preferred_stage_path(stage_dir, self.analysis_dir, filename)

    def head2head_path(self, filename: str) -> Path:
        """Resolve a head-to-head artifact path with legacy fallback."""

        return self._preferred_stage_path(self.head2head_stage_dir, self.analysis_dir, filename)

    def tiering_path(self, filename: str) -> Path:
        """Resolve a tiering artifact path with legacy fallback."""

        return self._preferred_stage_path(self.tiering_stage_dir, self.analysis_dir, filename)

    def preferred_tiers_path(self) -> Path:
        """Locate ``tiers.json`` across tiering and TrueSkill stages."""

        candidates = [
            self.tiering_stage_dir / "tiers.json",
            self.trueskill_stage_dir / "tiers.json",
            self.analysis_dir / "tiers.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @property
    def curated_parquet(self) -> Path:
        """Location of the combined curated parquet spanning all player counts."""
        pooled_dir = self.combine_pooled_dir()
        preferred = pooled_dir / "all_ingested_rows.parquet"
        candidates = [
            preferred,
            self.combine_stage_dir
            / f"{self.combine_max_players}p"
            / "pooled"
            / "all_ingested_rows.parquet",
            self.data_dir / "all_n_players_combined" / "all_ingested_rows.parquet",
            self.analysis_dir / "all_n_players_combined" / "all_ingested_rows.parquet",
            self.analysis_dir / "data" / "all_n_players_combined" / "all_ingested_rows.parquet",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return preferred

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
        preferred = self.curate_block_dir(n) / self.manifest_name
        legacy = self.analysis_dir / "data" / f"{n}p" / "manifest.jsonl"
        if preferred.exists() or not legacy.exists():
            return preferred
        return legacy

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

        parquet = self.curated_parquet
        preferred = parquet.with_suffix(".manifest.jsonl")
        legacy_candidates = [
            self.combine_stage_dir
            / f"{self.combine_max_players}p"
            / "pooled"
            / "all_ingested_rows.manifest.jsonl",
            self.combine_stage_dir / "all_n_players_combined" / "all_ingested_rows.manifest.jsonl",
            self.analysis_dir / "all_n_players_combined" / "all_ingested_rows.manifest.jsonl",
            self.analysis_dir / "data" / "all_n_players_combined" / "all_ingested_rows.manifest.jsonl",
        ]
        for candidate in legacy_candidates:
            if preferred.exists():
                break
            if candidate.exists():
                return candidate
        return preferred


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


def load_app_config(*overlays: Path) -> AppConfig:
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

    # Light compatibility if someone uses old keys
    if "io" in data:
        io_section = data["io"]
        if "analysis_dir" in io_section and "analysis_subdir" not in io_section:
            io_section["analysis_subdir"] = io_section.pop("analysis_dir")
    if "sim" in data:
        sim_section = data["sim"]
        if "n_players" in sim_section and "n_players_list" not in sim_section:
            sim_section["n_players_list"] = [sim_section.pop("n_players")]
        if (
            "collect_metrics" in sim_section
            and "expanded_metrics" not in sim_section
            and sim_section.pop("collect_metrics")
        ):
            sim_section["expanded_metrics"] = True
    if "analysis" in data:
        analysis_section = data["analysis"]
        if "run_tiering_report" in analysis_section:
            alias_val = analysis_section.pop("run_tiering_report")
            analysis_section.setdefault("run_frequentist", alias_val)

    def build(cls, section: Mapping[str, Any]) -> Any:
        """Instantiate a dataclass ``cls`` from a mapping of attributes."""
        obj = cls()
        type_hints = get_type_hints(cls)
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
    )
    if cfg.io.append_seed:
        # Append the seed number to the results_dir name
        base = str(cfg.io.results_dir)
        cfg.io.results_dir = Path(f"{base}_seed_{cfg.sim.seed}")
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
        section = getattr(cfg, section_name)
        if not hasattr(section, option):
            raise AttributeError(f"Unknown option {option!r} in section {section_name!r}")
        current = getattr(section, option)
        type_hints = get_type_hints(type(section))
        annotation = type_hints.get(option)
        new_value = _coerce(raw, current, annotation)
        setattr(section, option, new_value)
    return cfg


__all__ = [
    "IOConfig",
    "SimConfig",
    "AnalysisConfig",
    "IngestConfig",
    "CombineConfig",
    "MetricsConfig",
    "TrueSkillConfig",
    "Head2HeadConfig",
    "HGBConfig",
    "PowerDesign",
    "AppConfig",
    "load_app_config",
    "apply_dot_overrides",
]
