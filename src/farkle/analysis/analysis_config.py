# farkle/analysis_config.py
"""
Config module for the analysis stage.

Typical usage::

    from farkle.analysis_config import PipelineCfg
    from farkle import ingest, curate, metrics, analytics

    cfg = PipelineCfg(root=Path(args.root))
    ingest.run(cfg)   # writes ``game_rows.raw.parquet``
    curate.run(cfg)   # adds manifest & renames to ``game_rows.parquet``
    metrics.run(cfg)
    analytics.run_all(cfg)     # inside it: respects cfg.run_trueskill flags
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Final, Mapping, Sequence, Union, get_args, get_origin

import pyarrow as pa
import yaml

try:  # pragma: no cover - exercised in tests when dependency missing
    from pydantic import BaseModel
except ModuleNotFoundError:  # pragma: no cover - fallback used in CI
    import types as _types

    def _is_model(cls: type[Any]) -> bool:
        return isinstance(cls, type) and issubclass(cls, BaseModel)

    def _coerce(value: Any, annotation: Any) -> Any:
        origin = get_origin(annotation)
        if origin is None:
            if annotation in (Any, object):
                return value
            if annotation is Path and isinstance(value, (str, bytes)):
                return Path(value)
            if _is_model(annotation):
                if isinstance(value, Mapping):
                    return annotation(**value)  # type: ignore[arg-type]
                return value
            if isinstance(annotation, type) and isinstance(value, annotation):
                return value
            return value
        if origin in (list, Sequence, tuple, set):
            args = get_args(annotation) or (Any,)
            coerced = [_coerce(v, args[0]) for v in value]
            if origin is tuple:
                return tuple(coerced)
            if origin is set:
                return set(coerced)
            return list(coerced)
        if origin in (dict, Mapping):
            key_t, val_t = get_args(annotation) or (Any, Any)
            return { _coerce(k, key_t): _coerce(v, val_t) for k, v in value.items() }
        if origin in (_types.UnionType, Union):
            for arg in get_args(annotation):
                if arg is type(None):  # noqa: E721
                    if value is None:
                        return None
                    continue
                coerced = _coerce(value, arg)
                if coerced is not value or isinstance(value, arg):
                    return coerced
            return value
        return value

    def _dump(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, list):
            return [_dump(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_dump(v) for v in value)
        if isinstance(value, set):
            return {_dump(v) for v in value}
        return value

    class BaseModel:
        def __init_subclass__(cls, **kwargs: Any) -> None:
            annotations = getattr(cls, "__annotations__", {})
            for name in annotations:
                if hasattr(cls, name):
                    default = getattr(cls, name)
                    if isinstance(default, BaseModel):
                        def _factory(template=default):
                            return template.__class__(**template.model_dump())

                        setattr(cls, name, field(default_factory=_factory))
            dataclass(cls)

            def __init__(self, **data: Any) -> None:
                for f in fields(cls):
                    if f.name in data:
                        value = data.pop(f.name)
                    elif f.default is not MISSING:
                        value = f.default
                    elif f.default_factory is not MISSING:  # type: ignore[arg-type]
                        value = f.default_factory()
                    else:
                        raise TypeError(f"Missing required field {f.name}")
                    setattr(self, f.name, _coerce(value, f.type))
                if data:
                    raise TypeError(f"Unexpected fields: {', '.join(sorted(data))}")

            cls.__init__ = __init__  # type: ignore[assignment]

            def model_dump(self) -> dict[str, Any]:
                return {f.name: _dump(getattr(self, f.name)) for f in fields(self)}

            cls.model_dump = model_dump  # type: ignore[assignment]

        def model_dump(self) -> dict[str, Any]:  # pragma: no cover
            return {}


@dataclass
class PipelineCfg:
    # 1. core paths
    results_dir: Path = Path("results_seed_0")
    results_glob: str = "*_players"
    analysis_subdir: str = "analysis"
    curated_rows_name: str = "game_rows.parquet"  # legacy single-file name
    metrics_name: str = "metrics.parquet"

    # 2. ingest
    ingest_cols: tuple[str, ...] = field(
        default_factory=lambda: (
            "winner", "n_rounds", "winning_score",  # base inputs we need
            *(
                f"P{i}_{suffix}"
                for i in range(1, 13)  # max seats supported, gets overwritten when used by ingest
                for suffix in _SEAT_TEMPLATE
            ),
        )
    )
    parquet_codec: str = "zstd"
    row_group_size: int = 64_000  # max_shard_mb removed (unused)
    batch_rows: int = 100_000     # default Arrow batch size for streaming readers
    # Ingest concurrency (1 -> serial, >1 -> process pool)
    n_jobs_ingest: int = 1

    # 3. analytics toggles / params
    run_trueskill: bool = True
    run_head2head: bool = True
    run_hgb: bool = True
    hgb_max_iter: int = 500
    trueskill_beta: float = 25 / 6

    # 3b. streaming / memory policy
    # Cap TrueSkill worker count (None → auto based on CPU; steps will clamp to keep RAM low).
    trueskill_workers: int | None = None
    # HGB feature-matrix policy: "auto" | "incore" | "memmap"
    hgb_mode: str = "auto"
    hgb_max_ram_mb: int = 1024  # soft cap for in-core fit; above this use memmap in "auto"

    # 4. perf
    cores = os.cpu_count()
    if cores is None:
        raise RuntimeError("Unable to determine CPU Count")
    n_jobs: int = max(cores - 1, 1)
    prefetch_shards: int = 2

    # 5. logging & provenance
    log_level: str = "INFO"
    log_file: Path | None = None  # e.g. Path("analysis/pipeline.log")
    manifest_name: str = "manifest.json"
    _git_sha: str | None = field(default=None, repr=False, init=False)

    def _load_git_sha(self) -> str:
        try:
            import subprocess

            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return "unknown _load_git_sha() exception"

    @property
    def git_sha(self) -> str:
        if self._git_sha is None:
            self._git_sha = self._load_git_sha()
        return self._git_sha

    # Convenience helpers
    # -------------------
    @property
    def analysis_dir(self) -> Path:
        """Directory where analysis artifacts are written."""
        return self.results_dir / self.analysis_subdir

    @property
    def data_dir(self) -> Path:
        """Subdirectory beneath :pyattr:`analysis_dir` holding intermediate data."""
        return self.analysis_dir / "data"

    # New: per-N ingested rows (raw + curated)
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
        legacy = self.analysis_dir / "data" / self.curated_rows_name
        combined = self.data_dir / "all_n_players_combined" / "all_ingested_rows.parquet"
        # Prefer combined superset if present; fallback to legacy path
        return combined if combined.exists() or not legacy.exists() else legacy
      
    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: str(o) if isinstance(o, Path) else o, indent=2)

    # Convenience: return kwargs ready for setup_logging()
    def logging_params(self) -> dict[str, object]:
        return {"level": self.log_level, "log_file": self.log_file}
    
    def wanted_ingest_cols(self, n: int) -> list[str]:
        base = ["winner", "n_rounds", "winning_score"]
        seat = [f"P{i}_{sfx}" for i in range(1, n + 1) for sfx in _SEAT_TEMPLATE]
        return base + seat

    # ------------------------------------------------------------------
    @classmethod
    def parse_cli(
        cls, argv: Sequence[str] | None = None
    ) -> tuple["PipelineCfg", argparse.Namespace, list[str]]:
        """Parse command-line arguments into a :class:`PipelineCfg`.

        Returns a triple ``(cfg, namespace, remaining)`` where ``remaining``
        contains any extra arguments for higher-level parsers (e.g. pipeline
        subcommands).
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--results-dir",
            "--root",
            dest="results_dir",
            type=Path,
            default=Path("results_seed_0"),
            help="Directory containing raw results blocks",
        )
        parser.add_argument(
            "--analysis-subdir",
            default="analysis",
            help="Subdirectory for analysis outputs",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable progress bars and DEBUG logging",
        )
        ns, remaining = parser.parse_known_args(argv)
        cfg = cls(results_dir=ns.results_dir, analysis_subdir=ns.analysis_subdir)
        if ns.verbose:
            cfg.log_level = "DEBUG"
        return cfg, ns, remaining


# ---------- static pieces -------------------------------------------------
_BASE_FIELDS: Final[list[tuple[str, pa.DataType]]] = [
    ("winner_seat", pa.string()),  # P{n} label of the winner
    ("winner_strategy", pa.string()),  # strategy string of the winner
    ("seat_ranks", pa.list_(pa.string())),  # ["P7","P1","P3",...]
    ("winning_score", pa.int32()),
    ("n_rounds", pa.int16()),
]

_SEAT_TEMPLATE: Final[dict[str, pa.DataType]] = {
    "score": pa.int32(),
    "farkles": pa.int16(),
    "rolls": pa.int16(),
    "highest_turn": pa.int16(),
    "strategy": pa.string(),
    "rank": pa.int8(),
    "loss_margin": pa.int32(),
    "smart_five_uses": pa.int16(),
    "n_smart_five_dice": pa.int16(),
    "smart_one_uses": pa.int16(),
    "n_smart_one_dice": pa.int16(),
    "hot_dice": pa.int16(),  # counts of hot-dice used this game
    # add/remove seat-level cols here once
}

# ---------- public helpers -----------------------------------------------


def expected_schema_for(n_players: int) -> pa.Schema:
    """Return the canonical schema for *n_players* seats."""
    seat_fields: list[pa.Field] = []
    for i in range(1, n_players + 1):
        for suffix, dtype in _SEAT_TEMPLATE.items():
            seat_fields.append(pa.field(f"P{i}_{suffix}", dtype))
    return pa.schema(_BASE_FIELDS + seat_fields)


_PNUM_RE = re.compile(r"^P(\d+)_")  # Regex for P<X>_


def n_players_from_schema(schema: pa.Schema) -> int:
    pnums = []
    for name in schema.names:
        m = _PNUM_RE.match(name)
        if m:
            pnums.append(int(m.group(1)))
    return max(pnums) if pnums else 0


# Convenience: estimate rows per batch from a RAM budget (MB), column count,
# and value size (bytes). Used by streaming readers when you want a dynamic size.
def rows_for_ram(target_mb: int, n_cols: int, bytes_per_val: int = 4, safety: float = 1.5) -> int:
    """Convenience: estimate rows per batch from a RAM budget (MB), column count,
    and value size (bytes). Used by streaming readers when you want a dynamic size.
    """
    return max(10_000, int((target_mb * 1024**2) / (n_cols * bytes_per_val * safety)))


# ---------- YAML-based configuration models ---------------------------------


class IO(BaseModel):
    results_dir: Path
    analysis_subdir: str = "analysis"


class Ingest(BaseModel):
    row_group_size: int = 262_144
    n_jobs: int = 1


class Combine(BaseModel):
    max_players: int = 12


class Metrics(BaseModel):
    seat_range: tuple[int, int] = (1, 12)


class TrueSkillCfg(BaseModel):
    beta: float = 25.0
    tau: float = 0.1
    draw_probability: float = 0.0


class Head2Head(BaseModel):
    n_jobs: int = 4
    games_per_pair: int = 10_000
    fdr_q: float = 0.02


class HGBCfg(BaseModel):
    max_depth: int = 6
    n_estimators: int = 300


class Experiment(BaseModel):
    name: str
    seed: int = 0


class Config(BaseModel):
    experiment: Experiment
    io: IO
    ingest: Ingest = Ingest()
    combine: Combine = Combine()
    metrics: Metrics = Metrics()
    trueskill: TrueSkillCfg = TrueSkillCfg()
    head2head: Head2Head = Head2Head()
    hgb: HGBCfg = HGBCfg()
    schema_version: int = 1
    config_sha: str | None = None

    def to_pipeline_cfg(self) -> "PipelineCfg":
        """Convert to legacy :class:`PipelineCfg` for existing callers."""

        cfg = PipelineCfg(
            results_dir=self.io.results_dir,
            analysis_subdir=self.io.analysis_subdir,
            row_group_size=self.ingest.row_group_size,
            n_jobs_ingest=self.ingest.n_jobs,
            trueskill_beta=self.trueskill.beta,
            hgb_max_iter=self.hgb.n_estimators,
        )
        if self.config_sha is not None:
            setattr(cfg, "config_sha", self.config_sha)  # noqa: B010
        return cfg


def load_config(path: Path) -> tuple[Config, str]:
    """Load YAML → Config; return ``(config, sha12)`` of resolved dict."""

    data = yaml.safe_load(path.read_text())
    cfg = Config(**data)
    dumped = json.dumps(cfg.model_dump(), sort_keys=True, default=str).encode()
    sha = hashlib.sha256(dumped).hexdigest()[:12]
    cfg.config_sha = sha
    return cfg, sha

