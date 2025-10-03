from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, get_args, get_origin, get_type_hints

import yaml


def _expand_dotted_keys(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Return a nested dict from *mapping* that may contain dotted keys."""

    result: dict[str, Any] = {}
    for raw_key, raw_value in mapping.items():
        value = (
            _expand_dotted_keys(raw_value)
            if isinstance(raw_value, Mapping)
            else raw_value
        )
        if isinstance(raw_key, str) and "." in raw_key:
            target = result
            parts = [part for part in raw_key.split(".") if part]
            if not parts:
                continue
            for part in parts[:-1]:
                current = target.get(part)
                if current is None:
                    current = {}
                    target[part] = current
                elif not isinstance(current, dict):
                    raise TypeError(
                        "Cannot expand dotted key; intermediate value is not a mapping",
                    )
                target = current
            leaf = parts[-1]
            existing = target.get(leaf)
            if isinstance(existing, dict) and isinstance(value, dict):
                existing.update(value)
            else:
                target[leaf] = value
        else:
            if isinstance(value, dict):
                existing = result.get(raw_key)
                if isinstance(existing, dict):
                    existing.update(value)
                else:
                    result[raw_key] = value
            else:
                result[raw_key] = value
    return result


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, val in overlay.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(val, Mapping):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _annotation_contains(annotation: Any, target: type) -> bool:
    if annotation is None:
        return False
    if annotation is target:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(_annotation_contains(arg, target) for arg in get_args(annotation))


def _convert_value(value: Any, annotation: Any) -> Any:
    if annotation is Any or annotation is object or annotation is None:
        return value
    if value is None:
        return None

    if is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected mapping to build {annotation.__name__}")
        return _build_dataclass(annotation, value)

    if annotation is Path:
        return value if isinstance(value, Path) else Path(value)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"Cannot parse boolean value from {value!r}")
        return bool(value)
    if annotation is int:
        if isinstance(value, bool):
            raise ValueError("Boolean value is not a valid integer override")
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is str:
        return str(value)

    origin = get_origin(annotation)
    if origin is None:
        return value

    if origin in {list, tuple, Sequence}:
        elem_type = get_args(annotation)[0] if get_args(annotation) else Any
        converted = [_convert_value(item, elem_type) for item in value]
        if origin is tuple:
            return tuple(converted)
        return list(converted)

    if origin in {set, frozenset}:
        elem_type = get_args(annotation)[0] if get_args(annotation) else Any
        converted = {_convert_value(item, elem_type) for item in value}
        return converted if origin is set else frozenset(converted)

    if origin in {dict, Mapping}:
        key_type, val_type = get_args(annotation) if get_args(annotation) else (Any, Any)
        return {
            _convert_value(k, key_type): _convert_value(v, val_type)
            for k, v in value.items()
        }

    if str(origin).endswith("Union") or str(origin).endswith("Optional"):
        for arg in get_args(annotation):
            if arg is type(None):  # noqa: E721
                if value is None:
                    return None
                continue
            try:
                return _convert_value(value, arg)
            except (TypeError, ValueError):
                continue
        return value

    return value


def _build_dataclass(cls: type[Any], section: Mapping[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    hints = get_type_hints(cls)
    for field in dataclasses.fields(cls):
        if field.name not in section:
            continue
        annotation = hints.get(field.name, field.type)
        kwargs[field.name] = _convert_value(section[field.name], annotation)
    return cls(**kwargs)


def _normalise_for_hash(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalise_for_hash(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_normalise_for_hash(v) for v in value]
        if isinstance(value, (set, frozenset)):
            items.sort()
        return items
    if isinstance(value, Path):
        return str(value)
    return value


def _compute_config_sha(cfg_dict: Mapping[str, Any]) -> str:
    payload = json.dumps(_normalise_for_hash(cfg_dict), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


@dataclass
class ExperimentConfig:
    name: str = "default_experiment"
    seed: int = 0


@dataclass
class IOConfig:
    results_dir: Path = Path("results")
    analysis_subdir: str = "analysis"


@dataclass
class StatsDesignConfig:
    enabled: bool = False
    power: float = 0.95
    alpha: float = 0.05
    fdr_q: float | None = 0.02
    detectable_lift: float = 0.03
    baseline_rate: float = 0.50
    tail: str = "two_sided"
    min_games_floor: int = 2_000
    max_games_cap: int | None = None
    recompute: bool = True


@dataclass
class SimOverrides:
    num_shuffles: int | None = None
    expanded_metrics: bool | None = None
    n_jobs: int | None = None
    row_dir: Path | None = None


@dataclass
class SimConfig:
    n_players_list: list[int] = field(
        default_factory=lambda: [2, 3, 4, 5, 6, 8, 10, 12]
    )
    num_shuffles: int = 100
    seed: int = 0
    n_jobs: int | None = None
    expanded_metrics: bool = False
    row_dir: Path | None = None
    per_n: dict[int, SimOverrides] = field(default_factory=dict)
    bh_design: StatsDesignConfig = field(default_factory=StatsDesignConfig)
    bonferroni_design: StatsDesignConfig = field(default_factory=StatsDesignConfig)

    def __post_init__(self) -> None:
        cleaned: dict[int, SimOverrides] = {}
        for raw_key, raw_val in self.per_n.items():
            key = int(raw_key)
            if isinstance(raw_val, dict):
                raw_val = SimOverrides(**raw_val)
            cleaned[key] = raw_val
        self.per_n = cleaned
        self.n_players_list = [int(n) for n in self.n_players_list]
        if self.row_dir is not None and not isinstance(self.row_dir, Path):
            self.row_dir = Path(self.row_dir)


@dataclass
class AnalysisConfig:
    run_trueskill: bool = True
    run_head2head: bool = True
    run_hgb: bool = True
    log_level: str = "INFO"
    manifest_name: str = "manifest.json"
    metrics_filename: str = "metrics.parquet"
    curated_rows_name: str = "game_rows.parquet"
    combined_filename: str = "all_ingested_rows.parquet"
    done_suffix: str = ".done.json"
    batch_rows: int = 100_000


@dataclass
class IngestConfig:
    row_group_size: int = 64_000
    n_jobs: int = 1
    parquet_codec: str = "zstd"


@dataclass
class CombineConfig:
    max_players: int = 12


@dataclass
class MetricsConfig:
    seat_range: tuple[int, int] = (1, 12)


@dataclass
class TrueSkillConfig:
    beta: float = 25.0 / 6.0
    tau: float = 0.1
    draw_probability: float = 0.0
    workers: int | None = None
    checkpoint_every_batches: int = 50


@dataclass
class HeadToHeadConfig:
    n_jobs: int = 4
    games_per_pair: int = 10_000
    bonferroni_design: StatsDesignConfig = field(default_factory=StatsDesignConfig)


@dataclass
class HGBConfig:
    max_depth: int = 6
    n_estimators: int = 300
    mode: str = "auto"
    max_ram_mb: int = 1_024


@dataclass
class AppConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    io: IOConfig = field(default_factory=IOConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    combine: CombineConfig = field(default_factory=CombineConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    trueskill: TrueSkillConfig = field(default_factory=TrueSkillConfig)
    head_to_head: HeadToHeadConfig = field(default_factory=HeadToHeadConfig)
    hgb: HGBConfig = field(default_factory=HGBConfig)
    config_sha: str | None = None

    @property
    def results_dir(self) -> Path:
        return self.io.results_dir

    @property
    def analysis_dir(self) -> Path:
        return self.results_dir / self.io.analysis_subdir

    @property
    def data_dir(self) -> Path:
        return self.analysis_dir / "data"

    def ingested_rows_raw(self, n_players: int) -> Path:
        sub = self.data_dir / f"{n_players}p"
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{n_players}p_ingested_rows.raw.parquet"

    def ingested_rows_curated(self, n_players: int) -> Path:
        return (self.data_dir / f"{n_players}p") / f"{n_players}p_ingested_rows.parquet"

    def manifest_for(self, n_players: int) -> Path:
        return (self.data_dir / f"{n_players}p") / f"manifest_{n_players}p.json"

    @property
    def curated_parquet(self) -> Path:
        legacy = self.analysis_dir / "data" / self.analysis.curated_rows_name
        combined = self.data_dir / "all_n_players_combined" / self.analysis.combined_filename
        return combined if combined.exists() or not legacy.exists() else legacy

    @property
    def metrics_name(self) -> str:
        return self.analysis.metrics_filename

    @property
    def manifest_name(self) -> str:
        return self.analysis.manifest_name

    @property
    def head2head(self) -> HeadToHeadConfig:
        return self.head_to_head


def _materialise_app_config(data: Mapping[str, Any]) -> AppConfig:
    cfg = _build_dataclass(AppConfig, data)
    cfg.config_sha = _compute_config_sha(asdict(cfg))
    return cfg


def load_app_config(*paths: Path | str | None) -> AppConfig:
    data: dict[str, Any] = {}
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError(f"Config file {path} must contain a mapping")
        expanded = _expand_dotted_keys(loaded)
        data = _deep_merge(data, expanded)
    return _materialise_app_config(data)


def apply_dot_overrides(cfg: AppConfig, overrides: Sequence[str] | None) -> AppConfig:
    if not overrides:
        return cfg

    patch: dict[str, Any] = {}
    for expr in overrides:
        if "=" not in expr:
            raise ValueError(f"Invalid override {expr!r}")
        key, raw_value = expr.split("=", 1)
        if not key:
            raise ValueError(f"Invalid override {expr!r}")
        target = patch
        parts = [part for part in key.split(".") if part]
        if not parts:
            raise ValueError(f"Invalid override {expr!r}")
        for part in parts[:-1]:
            target = target.setdefault(part, {})  # type: ignore[assignment]
        target[parts[-1]] = yaml.safe_load(raw_value)

    merged = _deep_merge(asdict(cfg), _expand_dotted_keys(patch))
    new_cfg = _materialise_app_config(merged)
    return new_cfg


__all__ = [
    "ExperimentConfig",
    "IOConfig",
    "StatsDesignConfig",
    "SimOverrides",
    "SimConfig",
    "AnalysisConfig",
    "IngestConfig",
    "CombineConfig",
    "MetricsConfig",
    "TrueSkillConfig",
    "HeadToHeadConfig",
    "HGBConfig",
    "AppConfig",
    "load_app_config",
    "apply_dot_overrides",
]
