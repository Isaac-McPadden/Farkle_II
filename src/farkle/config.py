# src/farkle/config.py
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, get_args, get_origin, get_type_hints

import yaml

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


@dataclass
class PowerDesign:
    power: float = 0.8
    control: float = 0.1  # fdr_q (BH - FDR) or alpha (Bonferroni - FWER)
    detectable_lift: float = 0.03  # absolute lift in win-rate
    baseline_rate: float = 0.50
    tail: str = "two_sided"  # "one_sided" | "two_sided"
    full_pairwise: bool = True
    min_games_floor: int = 2000
    max_games_cap: int | None = None
    use_BY: bool | None = False  # if true and using BH, use q/H_m (more conservative)

@dataclass
class SimConfig:
    """Simulation parameters."""
    n_players_list: list[int] = field(default_factory=lambda: [5])
    num_shuffles: int = 100
    seed: int = 0
    expanded_metrics: bool = False
    row_dir: Path | None = None
    per_n: dict[int, 'SimConfig'] = field(default_factory=dict)
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


@dataclass
class AnalysisConfig:
    """Analysis-stage parameters."""
    run_trueskill: bool = True
    run_head2head: bool = True
    run_hgb: bool = True
    n_jobs: int = 1
    log_level: str = "INFO"
    results_glob: str = "*_players"
    # Optional outputs block may be provided in YAML
    # outputs:
    #   curated_rows_name: "game_rows.parquet"
    #   metrics_name: "metrics.parquet"
    #   manifest_name: "manifest.jsonl"
    outputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestConfig:
    row_group_size: int = 64_000
    parquet_codec: str = "zstd"
    batch_rows: int = 100_000
    n_jobs: int = 1


@dataclass
class CombineConfig:
    max_players: int = 12


@dataclass
class MetricsConfig:
    seat_range: tuple[int, int] = (1, 12)


@dataclass
class TrueSkillConfig:
    beta: float = 25.0
    tau: float = 0.1
    draw_probability: float = 0.0


@dataclass
class Head2HeadConfig:
    n_jobs: int = 4
    games_per_pair: int = 10_000
    fdr_q: float = 0.02
    # If you ever add a nested design block here, it will still parse:
    bonferroni_design: dict[str, Any] = field(default_factory=dict)


@dataclass
class HGBConfig:
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

    # —— Paths ——
    @property
    def results_dir(self) -> Path:
        return self.io.results_dir

    @property
    def analysis_dir(self) -> Path:
        return self.io.results_dir / self.io.analysis_subdir

    @property
    def data_dir(self) -> Path:
        return self.analysis_dir / "data"

    def n_dir(self, n: int) -> Path:
        return self.results_dir / f"{n}_players"

    def checkpoint_path(self, n: int) -> Path:
        return self.n_dir(n) / f"{n}p_checkpoint.pkl"

    def metrics_path(self, n: int) -> Path:
        return self.n_dir(n) / f"{n}p_metrics.parquet"

    # —— Ingest/streaming knobs ——
    @property
    def row_group_size(self) -> int:
        return self.ingest.row_group_size

    @property
    def parquet_codec(self) -> str:
        return self.ingest.parquet_codec

    @property
    def n_jobs_ingest(self) -> int:
        return self.ingest.n_jobs

    @property
    def batch_rows(self) -> int:
        return self.ingest.batch_rows

    # —— Handy aliases used by some modules (kept to minimize edits) ——
    @property
    def trueskill_beta(self) -> float:
        return self.trueskill.beta

    @property
    def hgb_max_iter(self) -> int:
        return self.hgb.n_estimators

    @property
    def combine_max_players(self) -> int:
        return self.combine.max_players

    @property
    def metrics_seat_range(self) -> tuple[int, int]:
        return self.metrics.seat_range

    # —— Output filenames and standard derived locations ——
    @property
    def metrics_name(self) -> str:
        # prefer analysis.outputs.metrics_name if provided
        outputs = self.analysis.outputs or {}
        return str(outputs.get("metrics_name", "metrics.parquet"))

    @property
    def curated_rows_name(self) -> str:
        outputs = self.analysis.outputs or {}
        return str(outputs.get("curated_rows_name", "game_rows.parquet"))

    @property
    def manifest_name(self) -> str:
        outputs = self.analysis.outputs or {}
        return str(outputs.get("manifest_name", "manifest.jsonl"))

    @property
    def curated_parquet(self) -> Path:
        # combined superset parquet after "combine" step
        preferred = self.data_dir / "all_n_players_combined" / "all_ingested_rows.parquet"
        legacy = self.analysis_dir / "all_n_players_combined" / "all_ingested_rows.parquet"
        if preferred.exists() or not legacy.exists():
            return preferred
        return legacy

    # Per-N helper paths used by ingest/curate/metrics
    def manifest_for(self, n: int) -> Path:
        return self.data_dir / f"{n}p" / self.manifest_name

    def ingested_rows_raw(self, n: int) -> Path:
        return self.data_dir / f"{n}p" / f"{n}p_ingested_rows.raw.parquet"

    def ingested_rows_curated(self, n: int) -> Path:
        return self.data_dir / f"{n}p" / self.curated_rows_name


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
    if annotation is None:
        return False
    if annotation is target:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(_annotation_contains(arg, target) for arg in get_args(annotation))


def load_app_config(*overlays: Path) -> AppConfig:
    """Load one or more YAML overlays and return an :class:`AppConfig`.
    Later overlays take precedence and are deep-merged. Dotted keys supported.
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
        if "collect_metrics" in sim_section and "expanded_metrics" not in sim_section and sim_section.pop("collect_metrics"):
            sim_section["expanded_metrics"] = True

    def build(cls, section: Mapping[str, Any]) -> Any:
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
                        (int(k) if key_t is int else k): build(val_t, v) if isinstance(v, Mapping) else v
                        for k, v in (val or {}).items()
                    }

            # Path coercion (works for nested too because we use type hints)
            if (isinstance(current, Path) or _annotation_contains(annotation, Path)) and isinstance(val, (str, Path)):
                val = Path(val)

            setattr(obj, f.name, val)
        return obj

    cfg =  AppConfig(
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
    if annotation is not None and _annotation_contains(annotation, int) and not _annotation_contains(annotation, bool):
        return int(value)
    if isinstance(current, float) or (annotation is not None and _annotation_contains(annotation, float)):
        return float(value)
    if isinstance(current, Path) or (annotation is not None and _annotation_contains(annotation, Path)):
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
