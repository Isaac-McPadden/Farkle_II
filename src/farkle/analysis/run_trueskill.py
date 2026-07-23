# src/farkle/analysis/run_trueskill.py
"""Compute TrueSkill ratings for Farkle strategies.

The helpers scan tournament results, update ratings with :mod:`trueskill`, and
write canonical per-root/per-k rating files for descriptive screening. Cross-k
candidate contribution is based on normalized percentile ranks, not propagated
model sigma. Historically this module exposed a standalone command line
interface; the configuration-driven CLI now calls :func:`run_trueskill`.

Outputs
-------
``by_k/<N>p/ratings_<N>_seed<root>.parquet``
    Per-root/per-k tables with columns ``{strategy, mu, sigma}``. Sigma is
    model state for screening diagnostics, not a cross-k sampling uncertainty.
``across_k/candidate_percentile_contribution.parquet``
    Complete-support mean of normalized within-root/per-k percentile ranks.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trueskill

from farkle.analysis.trueskill_screening import (
    TRUESKILL_CONDITIONING,
    TRUESKILL_METHOD_VERSION,
    ScreeningRatingCell,
    build_percentile_contribution,
    build_screening_diagnostics,
    classify_trueskill_row,
    publish_rating_cell_contract,
    trueskill_method_contract,
)
from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    ArtifactSidecar,
    ensure_artifact_sidecar_atomic,
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
)
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.authenticated_contract import CodeIdentityPolicy, resolve_code_identity
from farkle.utils.parallel import resolve_mp_context
from farkle.utils.progress import ProgressLogConfig, ScheduledProgressLogger
from farkle.utils.random import seed_everything
from farkle.utils.stage_completion import freshness_sha256
from farkle.utils.writer import atomic_path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # hop out of src/farkle
# Default location of tournament result blocks when no path is supplied
DEFAULT_DATAROOT = _REPO_ROOT / "data" / "results"

LOGGER = logging.getLogger(__name__)

DEFAULT_RATING = trueskill.Rating()  # uses env defaults
TRUESKILL_CELL_METHOD_VERSION = TRUESKILL_METHOD_VERSION
_EVIDENCE_BACKED = "evidence_backed_completed_games"
_PRIOR_ONLY = "prior_only_unrated"


class RatingArtifactPaths(TypedDict):
    """Canonical file paths for one root/k rating artifact bundle."""

    parquet: Path
    json: Path
    ckpt: Path
    checkpoint: Path
    dir: Path


class TrueSkillInitKwargs(TypedDict, total=False):
    """Typed TrueSkill constructor kwargs."""

    mu: float
    sigma: float
    beta: float
    tau: float
    draw_probability: float


_TRUE_SKILL_INIT_KEYS: tuple[str, ...] = (
    "mu",
    "sigma",
    "beta",
    "tau",
    "draw_probability",
)


def _coerce_trueskill_env_kwargs(env_kwargs: Mapping[str, object] | None) -> dict[str, float]:
    """Validate and coerce config-driven TrueSkill kwargs to floats."""

    if env_kwargs is None:
        return {}

    coerced: dict[str, float] = {}
    for key in _TRUE_SKILL_INIT_KEYS:
        value = env_kwargs.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float, str)):
            coerced[key] = float(value)
            continue
        raise TypeError(
            f"TrueSkill config value '{key}' must be float-compatible, got {type(value)!r}"
        )
    return coerced


def _per_player_dir(root: Path, player_count: str) -> Path:
    """Canonical directory for per-player-count artifacts."""

    return root / "by_k" / f"{player_count}p"


def _rating_artifact_paths(root: Path, player_count: str, suffix: str) -> RatingArtifactPaths:
    """Return canonical paths for per-player-count rating artifacts."""

    per_dir = _per_player_dir(root, player_count)
    base_name = f"ratings_{player_count}{suffix}"
    return {
        "dir": per_dir,
        "parquet": per_dir / f"{base_name}.parquet",
        "json": per_dir / f"{base_name}.json",
        "ckpt": per_dir / f"{base_name}.ckpt.json",
        "checkpoint": per_dir / f"{base_name}.checkpoint.parquet",
    }


def _iter_rating_parquets(root: Path, suffix: str) -> list[Path]:
    """Discover canonical per-player rating parquet files."""

    per_player = list((root / "by_k").glob(f"*p/ratings_*{suffix}.parquet"))
    out: list[Path] = []
    seen: set[str] = set()
    for path in per_player:
        key = path.resolve().as_posix()
        if key in seen:
            continue
        out.append(path)
        seen.add(key)

    def _sort_key(path: Path) -> tuple[float, str]:
        count = _player_count_from_stem(path.stem)
        return (count if count is not None else math.inf, path.resolve().as_posix())

    return sorted(out, key=_sort_key)


def _player_count_from_stem(stem: str) -> int | None:
    """Extract the player count from a canonical ratings filename stem."""

    match = re.fullmatch(r"ratings_(\d+)(?:_seed\d+)?", stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


@dataclass(slots=True)
class RatingStats:
    """TrueSkill state plus explicit completed-game support."""

    mu: float
    sigma: float
    strategy_attempted_exposures: int = 0
    strategy_completed_exposures: int = 0
    strategy_excluded_safety_limit_exposures: int = 0
    strategy_performed_updates: int = 0
    rating_status: str = _PRIOR_ONLY
    cell_games_attempted: int = 0
    cell_games_completed: int = 0
    cell_games_excluded_safety_limit: int = 0
    cell_performed_updates: int = 0


def _ratings_to_table(
    mapping: Mapping[str, Union[trueskill.Rating, "RatingStats", Tuple[float, float], float]],
) -> pa.Table:
    """Coerce {strategy: Rating|RatingStats|(mu,sigma)|mu} → Arrow table."""
    columns: dict[str, list[object]] = {
        "strategy": [],
        "mu": [],
        "sigma": [],
        "strategy_attempted_exposures": [],
        "strategy_completed_exposures": [],
        "strategy_excluded_safety_limit_exposures": [],
        "strategy_performed_updates": [],
        "rating_status": [],
        "cell_games_attempted": [],
        "cell_games_completed": [],
        "cell_games_excluded_safety_limit": [],
        "cell_performed_updates": [],
    }
    for k, v in mapping.items():
        if isinstance(v, RatingStats):
            mu, sigma = float(v.mu), float(v.sigma)
            stats = v
        elif isinstance(v, trueskill.Rating):
            mu, sigma = float(v.mu), float(v.sigma)
            stats = RatingStats(mu, sigma)
        elif isinstance(v, (tuple, list)) and len(v) >= 2:
            mu, sigma = float(v[0]), float(v[1])
            stats = RatingStats(mu, sigma)
        else:
            # fallback: scalar mu with default sigma (not expected here)
            mu, sigma = float(v), 0.0  # type: ignore[arg-type]
            stats = RatingStats(mu, sigma)
        columns["strategy"].append(_normalize_strategy_id(k))
        columns["mu"].append(mu)
        columns["sigma"].append(sigma)
        columns["strategy_attempted_exposures"].append(stats.strategy_attempted_exposures)
        columns["strategy_completed_exposures"].append(stats.strategy_completed_exposures)
        columns["strategy_excluded_safety_limit_exposures"].append(
            stats.strategy_excluded_safety_limit_exposures
        )
        columns["strategy_performed_updates"].append(stats.strategy_performed_updates)
        columns["rating_status"].append(stats.rating_status)
        columns["cell_games_attempted"].append(stats.cell_games_attempted)
        columns["cell_games_completed"].append(stats.cell_games_completed)
        columns["cell_games_excluded_safety_limit"].append(stats.cell_games_excluded_safety_limit)
        columns["cell_performed_updates"].append(stats.cell_performed_updates)
    return pa.table(
        {
            "strategy": pa.array(columns["strategy"], type=pa.string()),
            "mu": pa.array(columns["mu"], type=pa.float64()),
            "sigma": pa.array(columns["sigma"], type=pa.float64()),
            "strategy_attempted_exposures": pa.array(
                columns["strategy_attempted_exposures"], type=pa.int64()
            ),
            "strategy_completed_exposures": pa.array(
                columns["strategy_completed_exposures"], type=pa.int64()
            ),
            "strategy_excluded_safety_limit_exposures": pa.array(
                columns["strategy_excluded_safety_limit_exposures"], type=pa.int64()
            ),
            "strategy_performed_updates": pa.array(
                columns["strategy_performed_updates"], type=pa.int64()
            ),
            "rating_status": pa.array(columns["rating_status"], type=pa.string()),
            "cell_games_attempted": pa.array(columns["cell_games_attempted"], type=pa.int64()),
            "cell_games_completed": pa.array(columns["cell_games_completed"], type=pa.int64()),
            "cell_games_excluded_safety_limit": pa.array(
                columns["cell_games_excluded_safety_limit"], type=pa.int64()
            ),
            "cell_performed_updates": pa.array(columns["cell_performed_updates"], type=pa.int64()),
        }
    )


def _save_ratings_parquet(
    path: Path,
    ratings: Mapping[str, Union[trueskill.Rating, "RatingStats", Tuple[float, float]]],
) -> None:
    """Write ratings to parquet in a consistent schema."""
    write_parquet_atomic(_ratings_to_table(ratings), path)


def _load_ratings_parquet(path: Path) -> dict[str, "RatingStats"]:
    """Load ratings parquet → {strategy: RatingStats}."""
    schema = pq.read_schema(path)
    support_columns = [
        "strategy_attempted_exposures",
        "strategy_completed_exposures",
        "strategy_excluded_safety_limit_exposures",
        "strategy_performed_updates",
        "rating_status",
        "cell_games_attempted",
        "cell_games_completed",
        "cell_games_excluded_safety_limit",
        "cell_performed_updates",
    ]
    columns = ["strategy", "mu", "sigma", *[c for c in support_columns if c in schema.names]]
    tbl = pq.read_table(path, columns=columns)
    data = tbl.to_pydict()
    out: dict[str, RatingStats] = {}
    for index, (s, mu, sg) in enumerate(
        zip(data["strategy"], data["mu"], data["sigma"], strict=True)
    ):

        def support(
            name: str,
            default: int | str,
            row_index: int = index,
        ) -> int | str:
            values = data.get(name)
            return default if values is None else values[row_index]

        out[str(s)] = RatingStats(
            float(mu),
            float(sg),
            strategy_attempted_exposures=int(support("strategy_attempted_exposures", 0)),
            strategy_completed_exposures=int(support("strategy_completed_exposures", 0)),
            strategy_excluded_safety_limit_exposures=int(
                support("strategy_excluded_safety_limit_exposures", 0)
            ),
            strategy_performed_updates=int(support("strategy_performed_updates", 0)),
            rating_status=str(support("rating_status", _PRIOR_ONLY)),
            cell_games_attempted=int(support("cell_games_attempted", 0)),
            cell_games_completed=int(support("cell_games_completed", 0)),
            cell_games_excluded_safety_limit=int(support("cell_games_excluded_safety_limit", 0)),
            cell_performed_updates=int(support("cell_performed_updates", 0)),
        )
    return out


def _ensure_auxiliary_rating_sidecars(
    cfg: AppConfig,
    *,
    cell: ScreeningRatingCell,
    suffix: str,
    code_revision: str | None = None,
) -> None:
    """Bind auxiliary root/k rating exports produced by the streaming worker."""

    player_count = str(cell.k)
    paths = _rating_artifact_paths(cfg.trueskill_stage_dir, player_count, suffix)
    json_path = paths["json"]
    shard_path, _shard_done = _block_shard_paths(cfg.trueskill_stage_dir, player_count, suffix)

    def _sidecar(path: Path, *, operation: str, columns: list[str]) -> ArtifactSidecar:
        return make_artifact_sidecar(
            cfg,
            path,
            producer="trueskill",
            scope=ArtifactScope.BY_K,
            source_scope=ArtifactScope.BY_K,
            operation=operation,
            weighted_quantity="trueskill_mu_and_sigma",
            support_count_role="attempted_completed_excluded_and_performed_updates",
            uncertainty_method="trueskill_model_sigma_screening_only",
            replication_unit="ordered_game",
            conditioning=TRUESKILL_CONDITIONING,
            consistency_columns=columns,
            source_artifacts=[cell.ratings_path],
            grouping_keys=["strategy"],
            player_counts=[cell.k],
            required_player_counts=[cell.k],
            missing_cell_policy="fail",
            seed_scope="single_root",
            code_revision=code_revision or _trueskill_code_revision(cfg),
            method_contract=cast(
                Any,
                trueskill_method_contract(operation),
            ),
        )

    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(json_payload, dict):
        raise ValueError(f"TrueSkill JSON export must be an object: {json_path}")
    json_sidecar = _sidecar(
        json_path,
        operation="export_sequential_ratings_json",
        columns=[
            "strategy",
            *(
                list(next(iter(json_payload.values())).keys())
                if json_payload and isinstance(next(iter(json_payload.values())), dict)
                else ["mu", "sigma"]
            ),
        ],
    )
    ensure_artifact_sidecar_atomic(
        json_path,
        json_sidecar,
        expected={
            "scope": ArtifactScope.BY_K.value,
            "operation": "export_sequential_ratings_json",
            "player_counts": [cell.k],
        },
    )
    if len(json_payload) != pq.read_metadata(cell.ratings_path).num_rows:
        raise ValueError(f"TrueSkill JSON export disagrees with canonical ratings: {json_path}")

    shard_schema = pq.read_schema(shard_path)
    shard_sidecar = _sidecar(
        shard_path,
        operation="snapshot_sequential_rating_cell",
        columns=shard_schema.names,
    )
    ensure_artifact_sidecar_atomic(
        shard_path,
        shard_sidecar,
        expected={
            "scope": ArtifactScope.BY_K.value,
            "operation": "snapshot_sequential_rating_cell",
            "player_counts": [cell.k],
        },
    )


# ---------- Per-N checkpointing ----------
@dataclass
class _BlockCkpt:
    """Checkpoint progress for a single player-count block."""

    row_file: str
    row_group: int
    batch_index: int
    games_done: int
    ratings_path: str
    freshness_sha256: str = ""
    attempted_games: int = 0
    completed_games: int = 0
    excluded_safety_limit_games: int = 0
    strategy_attempted_exposures: dict[str, int] = field(default_factory=dict)
    strategy_completed_exposures: dict[str, int] = field(default_factory=dict)
    strategy_excluded_safety_limit_exposures: dict[str, int] = field(default_factory=dict)
    strategy_performed_updates: dict[str, int] = field(default_factory=dict)
    version: int = 2


@dataclass(slots=True)
class _ShardDoneStamp:
    """Completion stamp for independently resumable shards."""

    shard_key: str
    parquet_path: str
    rows: int
    created_at: float
    root_seed: int
    player_count: int
    method_version: int
    source_sha256: str
    source_sidecar_sha256: str | None
    parquet_sha256: str
    freshness_sha256: str
    sidecar_sha256: str | None
    attempted_games: int = 0
    completed_games: int = 0
    excluded_safety_limit_games: int = 0
    performed_update_games: int = 0
    version: int = 4


def _save_done_stamp(path: Path, stamp: _ShardDoneStamp) -> None:
    """Write a shard completion stamp atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(asdict(stamp), indent=2, sort_keys=True))


def _load_done_stamp(path: Path) -> _ShardDoneStamp | None:
    """Load and minimally validate a shard completion stamp."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return _ShardDoneStamp(
            shard_key=str(payload["shard_key"]),
            parquet_path=str(payload["parquet_path"]),
            rows=int(payload["rows"]),
            created_at=float(payload["created_at"]),
            root_seed=int(payload["root_seed"]),
            player_count=int(payload["player_count"]),
            method_version=int(payload["method_version"]),
            source_sha256=str(payload["source_sha256"]),
            source_sidecar_sha256=(
                None
                if payload["source_sidecar_sha256"] is None
                else str(payload["source_sidecar_sha256"])
            ),
            parquet_sha256=str(payload["parquet_sha256"]),
            freshness_sha256=str(payload["freshness_sha256"]),
            sidecar_sha256=(
                None if payload["sidecar_sha256"] is None else str(payload["sidecar_sha256"])
            ),
            attempted_games=int(payload["attempted_games"]),
            completed_games=int(payload["completed_games"]),
            excluded_safety_limit_games=int(payload["excluded_safety_limit_games"]),
            performed_update_games=int(payload["performed_update_games"]),
            version=int(payload["version"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _done_stamp_matches(
    stamp: _ShardDoneStamp | None,
    *,
    parquet_path: Path,
    source_path: Path,
    freshness: str,
    root_seed: int,
    player_count: int,
    require_sidecar: bool = True,
) -> bool:
    """Return whether a cell stamp authenticates its current source and output."""

    return bool(
        stamp is not None
        and stamp.version == 4
        and stamp.root_seed == root_seed
        and stamp.player_count == player_count
        and stamp.method_version == TRUESKILL_CELL_METHOD_VERSION
        and Path(stamp.parquet_path) == parquet_path
        and parquet_path.is_file()
        and source_path.is_file()
        and stamp.source_sha256 == sha256_file(source_path)
        and stamp.source_sidecar_sha256
        == (sha256_file(sidecar_path(source_path)) if sidecar_path(source_path).is_file() else None)
        and stamp.parquet_sha256 == sha256_file(parquet_path)
        and stamp.freshness_sha256 == freshness
        and stamp.attempted_games == stamp.completed_games + stamp.excluded_safety_limit_games
        and stamp.performed_update_games <= stamp.completed_games
        and stamp.rows == stamp.performed_update_games
        and (
            not require_sidecar
            or (
                stamp.sidecar_sha256 is not None
                and sidecar_path(parquet_path).is_file()
                and stamp.sidecar_sha256 == sha256_file(sidecar_path(parquet_path))
            )
        )
    )


def _block_done_stamp_path(root: Path, player_count: str, suffix: str) -> Path:
    """Canonical per-block done-stamp path."""

    per_dir = _per_player_dir(root, player_count)
    return per_dir / f"ratings_{player_count}{suffix}.done.json"


def _block_shard_paths(root: Path, player_count: str, suffix: str) -> tuple[Path, Path]:
    """Canonical artifact.<shard> paths for per-k shards."""

    per_dir = _per_player_dir(root, player_count)
    shard = per_dir / f"artifact.k{player_count}{suffix}"
    return shard.with_suffix(".parquet"), shard.with_suffix(".done.json")


def _save_block_ckpt(path: Path, ck: _BlockCkpt) -> None:
    """Persist a per-block checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(asdict(ck)))


def _load_block_ckpt(path: Path) -> Optional[_BlockCkpt]:
    """Load a per-block checkpoint, returning ``None`` on failure."""
    if not path.exists():
        return None
    try:
        payload_any: object = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(payload_any, dict):
        return None

    required_keys = {
        "row_file": str,
        "row_group": int,
        "batch_index": int,
        "games_done": int,
        "ratings_path": str,
        "freshness_sha256": str,
        "attempted_games": int,
        "completed_games": int,
        "excluded_safety_limit_games": int,
        "strategy_attempted_exposures": dict,
        "strategy_completed_exposures": dict,
        "strategy_excluded_safety_limit_exposures": dict,
        "strategy_performed_updates": dict,
    }
    for key, expected_type in required_keys.items():
        value = payload_any.get(key)
        if not isinstance(value, expected_type):
            return None

    version_value = payload_any.get("version")
    if version_value != 2:
        return None

    def int_dict(name: str) -> dict[str, int]:
        value = payload_any[name]
        assert isinstance(value, dict)
        if any(not isinstance(item, int) or item < 0 for item in value.values()):
            raise ValueError(f"invalid TrueSkill checkpoint support: {name}")
        return {str(key): int(item) for key, item in value.items()}

    try:
        return _BlockCkpt(
            row_file=payload_any["row_file"],
            row_group=payload_any["row_group"],
            batch_index=payload_any["batch_index"],
            games_done=payload_any["games_done"],
            ratings_path=payload_any["ratings_path"],
            freshness_sha256=payload_any["freshness_sha256"],
            attempted_games=payload_any["attempted_games"],
            completed_games=payload_any["completed_games"],
            excluded_safety_limit_games=payload_any["excluded_safety_limit_games"],
            strategy_attempted_exposures=int_dict("strategy_attempted_exposures"),
            strategy_completed_exposures=int_dict("strategy_completed_exposures"),
            strategy_excluded_safety_limit_exposures=int_dict(
                "strategy_excluded_safety_limit_exposures"
            ),
            strategy_performed_updates=int_dict("strategy_performed_updates"),
            version=version_value,
        )
    except (TypeError, ValueError):
        return None


# ---------- Single-pass streaming ----------


def _stream_batches(
    parquet_path: Path,
    columns: list[str],
    *,
    start_row_group: int = 0,
    start_batch_idx: int = 0,
    batch_rows: int = 100_000,
) -> Iterator[tuple[int, int, pa.Table]]:
    """Yield (row_group_index, batch_index, batch_table)."""
    pf = pq.ParquetFile(parquet_path)
    n_rg = pf.num_row_groups
    for rg in range(start_row_group, n_rg):
        table = pf.read_row_group(rg, columns=columns)
        # chunk the row-group into manageable batches
        for bi, batch in enumerate(table.to_batches(max_chunksize=batch_rows)):
            if rg == start_row_group and bi < start_batch_idx:
                continue
            yield rg, bi, pa.Table.from_batches([batch])


def _normalize_strategy_id(value: object) -> str:
    """Return a stable string identifier for strategy values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _classified_games_from_batch(
    batch: pa.Table,
    n: int,
) -> Iterator[tuple[str, list[str], list[int] | None]]:
    """Yield validated canonical attempts without imputing safety-limit ranks."""

    required = {
        "termination_status",
        "outcome_schema_version",
        "winner_seat",
        *(f"P{seat}_strategy" for seat in range(1, n + 1)),
        *(f"P{seat}_rank" for seat in range(1, n + 1)),
    }
    missing = sorted(required.difference(batch.column_names))
    if missing:
        raise ValueError(f"TrueSkill canonical rows lack columns: {missing}")
    for row in batch.to_pylist():
        game = classify_trueskill_row(row, n)
        yield game.termination_status, game.players, game.ranks


def _players_and_ranks_from_batch(
    batch: pa.Table,
    n: int,
) -> Iterator[tuple[list[str], list[int]]]:
    """Yield only normally completed games with valid canonical ranks."""

    for _status, players, ranks in _classified_games_from_batch(batch, n):
        if ranks is not None:
            yield players, ranks


def _rate_block_worker(
    block_dir: str,
    root_dir: str,
    suffix: str,
    batch_rows: int,
    *,
    resume: bool = True,
    checkpoint_every_batches: int = 500,
    progress_logging: ProgressLogConfig | None = None,
    env_kwargs: Mapping[str, float] | TrueSkillInitKwargs | None = None,
    row_data_dir: str | None = None,
    curated_rows_name: str | None = None,
    cell_freshness_sha256: str | None = None,
    root_seed: int = 0,
) -> tuple[str, int]:
    """
    Process one <N>_players block with optional checkpointing.
    Returns (player_count_str, n_games).
    """
    block = Path(block_dir)
    root = Path(root_dir)
    row_data_path = Path(row_data_dir) if row_data_dir else None
    if (
        cell_freshness_sha256 is None
        or re.fullmatch(r"[0-9a-f]{64}", cell_freshness_sha256) is None
    ):
        raise ValueError("TrueSkill cell freshness must be an authenticated SHA-256 digest")

    player_count = block.name.split("_")[0]
    per_player_dir = _per_player_dir(root, player_count)
    per_player_dir.mkdir(parents=True, exist_ok=True)
    keep_path = block / f"keepers_{player_count}.npy"
    keepers = np.load(keep_path).tolist() if keep_path.exists() else []
    keepers = [_normalize_strategy_id(k) for k in keepers]

    if row_data_path is None or curated_rows_name is None:
        raise ValueError("TrueSkill requires an explicit canonical curated-row directory and name")
    row_file = row_data_path / "by_k" / f"{player_count}p" / curated_rows_name
    if not row_file.exists():
        raise FileNotFoundError(f"TrueSkill canonical root/k rows missing: {row_file}")
    n = int(player_count)
    try:
        total_input_games = max(0, int(pq.read_metadata(row_file).num_rows))
    except Exception:
        total_input_games = 0

    # Up-to-date guard
    paths = _rating_artifact_paths(root, player_count, suffix)
    parquet_path = paths["parquet"]
    ck_path = paths["ckpt"]
    rk_path = paths["checkpoint"]
    json_path = paths["json"]
    done_path = _block_done_stamp_path(root, player_count, suffix)
    shard_parquet, shard_done_path = _block_shard_paths(root, player_count, suffix)

    done_stamp = _load_done_stamp(done_path)
    if resume and _done_stamp_matches(
        done_stamp,
        parquet_path=parquet_path,
        source_path=row_file,
        freshness=cell_freshness_sha256,
        root_seed=root_seed,
        player_count=n,
    ):
        assert done_stamp is not None
        LOGGER.info(
            "Authenticated TrueSkill block stamp found; skipping",
            extra={
                "stage": "trueskill",
                "block": player_count,
                "row_file": str(row_file),
                "parquet": str(parquet_path),
                "done": str(done_path),
            },
        )
        return player_count, max(0, int(done_stamp.rows))

    env = trueskill.TrueSkill(**_coerce_trueskill_env_kwargs(env_kwargs))

    start_rg = 0
    start_bi = 0
    performed_update_games = 0
    attempted_games = 0
    completed_games = 0
    excluded_safety_limit_games = 0
    ratings: dict[str, trueskill.Rating] = {strategy: env.create_rating() for strategy in keepers}
    strategy_attempted: dict[str, int] = dict.fromkeys(keepers, 0)
    strategy_completed: dict[str, int] = dict.fromkeys(keepers, 0)
    strategy_excluded: dict[str, int] = dict.fromkeys(keepers, 0)
    strategy_updates: dict[str, int] = dict.fromkeys(keepers, 0)
    if resume:
        ck = _load_block_ckpt(ck_path)
        if ck and Path(ck.row_file) == row_file and ck.freshness_sha256 == cell_freshness_sha256:
            start_rg = ck.row_group
            start_bi = ck.batch_index
            performed_update_games = ck.games_done
            attempted_games = ck.attempted_games
            completed_games = ck.completed_games
            excluded_safety_limit_games = ck.excluded_safety_limit_games
            strategy_attempted = ck.strategy_attempted_exposures
            strategy_completed = ck.strategy_completed_exposures
            strategy_excluded = ck.strategy_excluded_safety_limit_exposures
            strategy_updates = ck.strategy_performed_updates
            if Path(ck.ratings_path).exists():
                interim = _load_ratings_parquet(Path(ck.ratings_path))
                ratings = {k: env.create_rating(mu=v.mu, sigma=v.sigma) for k, v in interim.items()}

    run_completed = False
    try:
        last_ck = time.time()
        batches_since_ck = 0
        progress_logger = (
            ScheduledProgressLogger(
                LOGGER,
                label=f"TrueSkill {player_count}p",
                schedule=progress_logging or ProgressLogConfig(),
                unit="games",
                total=total_input_games,
            )
            if total_input_games > 0
            else None
        )
        schema = pq.read_schema(row_file)
        columns = list(schema.names)
        for rg, bi, batch in _stream_batches(
            row_file,
            columns,
            start_row_group=start_rg,
            start_batch_idx=start_bi,
            batch_rows=batch_rows,
        ):
            for status, players, ranks in _classified_games_from_batch(batch, n):
                attempted_games += 1
                if keepers:
                    selected = [
                        (player, None if ranks is None else ranks[index])
                        for index, player in enumerate(players)
                        if player in keepers
                    ]
                    players = [player for player, _rank in selected]
                    selected_ranks = [rank for _player, rank in selected]
                else:
                    selected_ranks = [] if ranks is None else list(ranks)

                for s in players:
                    if s not in ratings:
                        ratings[s] = env.create_rating()
                    strategy_attempted[s] = strategy_attempted.get(s, 0) + 1
                    strategy_completed.setdefault(s, 0)
                    strategy_excluded.setdefault(s, 0)
                    strategy_updates.setdefault(s, 0)
                if status == "safety_limit":
                    excluded_safety_limit_games += 1
                    for s in players:
                        strategy_excluded[s] += 1
                    continue

                completed_games += 1
                for s in players:
                    strategy_completed[s] += 1
                if ranks is None:
                    raise ValueError("completed TrueSkill game unexpectedly lacks ranks")
                if len(players) < 2:
                    continue
                if keepers:
                    if any(rank is None for rank in selected_ranks):
                        raise ValueError("completed keeper ranks must be non-null")
                    ranks = [int(cast(int, rank)) for rank in selected_ranks]
                    unique_ranks = sorted(set(ranks))
                    rank_map = {rank: index for index, rank in enumerate(unique_ranks)}
                    ranks = [rank_map[rank] for rank in ranks]
                teams = [[ratings[s]] for s in players]
                new = env.rate(teams, ranks=ranks)
                for s, t in zip(players, new, strict=False):
                    ratings[s] = t[0]
                    strategy_updates[s] += 1
                performed_update_games += 1

            batches_since_ck += 1
            if batches_since_ck >= checkpoint_every_batches or (time.time() - last_ck) > 60:
                _save_ratings_parquet(rk_path, ratings)
                _save_block_ckpt(
                    ck_path,
                    _BlockCkpt(
                        row_file=str(row_file),
                        row_group=rg,
                        batch_index=bi + 1,
                        games_done=performed_update_games,
                        ratings_path=str(rk_path),
                        freshness_sha256=cell_freshness_sha256,
                        attempted_games=attempted_games,
                        completed_games=completed_games,
                        excluded_safety_limit_games=excluded_safety_limit_games,
                        strategy_attempted_exposures=strategy_attempted,
                        strategy_completed_exposures=strategy_completed,
                        strategy_excluded_safety_limit_exposures=strategy_excluded,
                        strategy_performed_updates=strategy_updates,
                    ),
                )
                if progress_logger is not None:
                    progress_logger.maybe_log(
                        attempted_games,
                        detail=(
                            f"row group {rg + 1}, batch {bi + 1}, {len(ratings):,} ratings tracked"
                        ),
                        extra={
                            "stage": "trueskill",
                            "block": player_count,
                            "games_attempted": attempted_games,
                            "games_completed": completed_games,
                            "games_excluded_safety_limit": excluded_safety_limit_games,
                            "games_updated": performed_update_games,
                            "games_total": total_input_games,
                            "row_group": rg,
                            "batch_index": bi + 1,
                        },
                    )
                last_ck = time.time()
                batches_since_ck = 0

        if attempted_games != completed_games + excluded_safety_limit_games:
            raise ValueError("TrueSkill game support conservation failed")
        if performed_update_games > completed_games:
            raise ValueError("TrueSkill performed updates exceed completed games")
        for strategy in ratings:
            if strategy_attempted[strategy] != (
                strategy_completed[strategy] + strategy_excluded[strategy]
            ):
                raise ValueError(f"TrueSkill strategy support conservation failed: {strategy}")
            if strategy_updates[strategy] > strategy_completed[strategy]:
                raise ValueError(f"TrueSkill strategy updates exceed completed support: {strategy}")
        ratings_stats = {
            strategy: RatingStats(
                rating.mu,
                rating.sigma,
                strategy_attempted_exposures=strategy_attempted[strategy],
                strategy_completed_exposures=strategy_completed[strategy],
                strategy_excluded_safety_limit_exposures=strategy_excluded[strategy],
                strategy_performed_updates=strategy_updates[strategy],
                rating_status=(_EVIDENCE_BACKED if strategy_updates[strategy] > 0 else _PRIOR_ONLY),
                cell_games_attempted=attempted_games,
                cell_games_completed=completed_games,
                cell_games_excluded_safety_limit=excluded_safety_limit_games,
                cell_performed_updates=performed_update_games,
            )
            for strategy, rating in sorted(ratings.items())
        }
        # Atomic writes for per-N outputs
        # Write per-N ratings as Parquet (strategy, mu, sigma)
        sidecar_path(parquet_path).unlink(missing_ok=True)
        _save_ratings_parquet(parquet_path, ratings_stats)

        json_path = paths["json"]
        sidecar_path(json_path).unlink(missing_ok=True)
        with atomic_path(str(json_path)) as tmp_path:
            Path(tmp_path).write_text(
                json.dumps(
                    {strategy: asdict(stats) for strategy, stats in ratings_stats.items()},
                    sort_keys=True,
                )
            )
        _save_done_stamp(
            done_path,
            _ShardDoneStamp(
                shard_key=f"k={player_count}",
                parquet_path=str(parquet_path),
                rows=int(performed_update_games),
                created_at=time.time(),
                root_seed=root_seed,
                player_count=n,
                method_version=TRUESKILL_CELL_METHOD_VERSION,
                source_sha256=sha256_file(row_file),
                source_sidecar_sha256=(
                    sha256_file(sidecar_path(row_file))
                    if sidecar_path(row_file).is_file()
                    else None
                ),
                parquet_sha256=sha256_file(parquet_path),
                freshness_sha256=cell_freshness_sha256,
                sidecar_sha256=None,
                attempted_games=attempted_games,
                completed_games=completed_games,
                excluded_safety_limit_games=excluded_safety_limit_games,
                performed_update_games=performed_update_games,
            ),
        )
        sidecar_path(shard_parquet).unlink(missing_ok=True)
        _save_ratings_parquet(shard_parquet, ratings_stats)
        _save_done_stamp(
            shard_done_path,
            _ShardDoneStamp(
                shard_key=f"k={player_count}",
                parquet_path=str(shard_parquet),
                rows=int(performed_update_games),
                created_at=time.time(),
                root_seed=root_seed,
                player_count=n,
                method_version=TRUESKILL_CELL_METHOD_VERSION,
                source_sha256=sha256_file(row_file),
                source_sidecar_sha256=(
                    sha256_file(sidecar_path(row_file))
                    if sidecar_path(row_file).is_file()
                    else None
                ),
                parquet_sha256=sha256_file(shard_parquet),
                freshness_sha256=cell_freshness_sha256,
                sidecar_sha256=None,
                attempted_games=attempted_games,
                completed_games=completed_games,
                excluded_safety_limit_games=excluded_safety_limit_games,
                performed_update_games=performed_update_games,
            ),
        )

        run_completed = True
        return player_count, performed_update_games
    finally:
        if run_completed:
            for ck_file in (ck_path, rk_path):
                try:
                    ck_file.unlink(missing_ok=True)
                except Exception:
                    LOGGER.warning(
                        "TrueSkill checkpoint cleanup failed",
                        extra={"stage": "trueskill", "path": str(ck_file)},
                    )


def run_trueskill(
    output_seed: int = 0,
    root: Path | None = None,
    dataroot: Path | None = None,
    row_data_dir: Path | None = None,
    curated_rows_name: str | None = None,
    workers: int | None = None,
    batch_rows: int = 100_000,
    resume_per_n: bool = True,
    checkpoint_every_batches: int = 500,
    env_kwargs: Mapping[str, float] | TrueSkillInitKwargs | None = None,
    mp_start_method: str | None = None,
    progress_logging: ProgressLogConfig | None = None,
    cell_freshness_sha256: str | None = None,
) -> None:
    """Compute TrueSkill ratings for all result blocks.

    Parameters
    ----------
    output_seed: int, optional
        Value appended to output filenames so repeated runs do not overwrite
        earlier results.
    root: Path | None, optional
        Directory where rating artifacts are written. Defaults to
        ``<dataroot>/analysis/06_trueskill`` when ``None``.
    dataroot: Path | None, optional
        Directory containing tournament result blocks. When ``None`` the
        path defaults to ``<root>/results`` if ``root`` is given, otherwise
        to :data:`DEFAULT_DATAROOT`.
    row_data_dir: Path | None, optional
        Canonical curate-stage directory containing ``by_k/<k>p`` inputs.
    curated_rows_name: str | None, optional
        Canonical curated-row filename. It is required with ``row_data_dir``.

    Production side effects are canonical per-root/per-k ratings only. Formal
    cross-k model-sigma propagation is unavailable.
    """
    if (
        cell_freshness_sha256 is None
        or re.fullmatch(r"[0-9a-f]{64}", cell_freshness_sha256) is None
    ):
        raise ValueError("TrueSkill cell freshness must be an authenticated SHA-256 digest")
    if dataroot is None:
        base = Path(root) / "results" if root is not None else DEFAULT_DATAROOT
    else:
        base = Path(dataroot)

    root = Path(root) if root is not None else base / "analysis" / "06_trueskill"
    root.mkdir(parents=True, exist_ok=True)
    row_data_path = Path(row_data_dir) if row_data_dir is not None else None
    if row_data_path is None or curated_rows_name is None:
        raise ValueError("TrueSkill requires canonical row_data_dir and curated_rows_name")
    suffix = f"_seed{output_seed}" if output_seed is not None else ""
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)
    if progress_logging is None:
        progress_logging = ProgressLogConfig()

    LOGGER.info(
        "TrueSkill run start",
        extra={
            "stage": "trueskill",
            "root": str(root),
            "dataroot": str(base),
            "row_data_dir": str(row_data_path) if row_data_path else None,
            "workers": workers,
            "batch_rows": batch_rows,
            "resume": resume_per_n,
            "checkpoint_batches": checkpoint_every_batches,
            "seed": output_seed,
        },
    )

    mp_context = resolve_mp_context(mp_start_method)

    env_kwargs_float = _coerce_trueskill_env_kwargs(env_kwargs)

    blocks = sorted(base.glob("*_players"))
    # Pick a sensible default, then cap to CPUs and number of blocks
    auto_default = max(1, (os.cpu_count() or 1) - 1)
    requested = workers or auto_default
    cpu_cap = os.cpu_count() or 1
    actual_workers = max(1, min(requested, cpu_cap, len(blocks)))
    if actual_workers != requested:
        LOGGER.info(
            "TrueSkill workers capped",
            extra={
                "stage": "trueskill",
                "requested": requested,
                "actual": actual_workers,
                "cpu_cap": cpu_cap,
                "blocks": len(blocks),
            },
        )
    if actual_workers > 1 and len(blocks) > 1:
        with cf.ProcessPoolExecutor(max_workers=actual_workers, mp_context=mp_context) as ex:
            futures = {
                ex.submit(
                    _rate_block_worker,
                    str(b),
                    str(root),
                    suffix,
                    batch_rows,
                    resume=resume_per_n,
                    checkpoint_every_batches=checkpoint_every_batches,
                    progress_logging=progress_logging,
                    env_kwargs=env_kwargs_float,
                    row_data_dir=str(row_data_path) if row_data_path else None,
                    curated_rows_name=curated_rows_name,
                    cell_freshness_sha256=cell_freshness_sha256,
                    root_seed=output_seed,
                ): b
                for b in blocks
            }
            for fut in cf.as_completed(futures):
                try:
                    player_count, block_games = fut.result()
                except Exception as e:
                    bad = futures[fut]
                    LOGGER.exception(
                        "TrueSkill block failed",
                        extra={"stage": "trueskill", "block": bad.name, "error": str(e)},
                    )
                    raise RuntimeError(f"TrueSkill block failed for {bad.name}: {e}") from e
                del player_count, block_games
    else:
        for block in blocks:
            player_count, block_games = _rate_block_worker(
                str(block),
                str(root),
                suffix,
                batch_rows,
                resume=resume_per_n,
                checkpoint_every_batches=checkpoint_every_batches,
                progress_logging=progress_logging,
                env_kwargs=env_kwargs_float,
                row_data_dir=str(row_data_path) if row_data_path else None,
                curated_rows_name=curated_rows_name,
                cell_freshness_sha256=cell_freshness_sha256,
                root_seed=output_seed,
            )
            del player_count, block_games

    per_player_parquets = _iter_rating_parquets(root, suffix)
    valid_per_player_parquets: list[Path] = []
    for parquet in per_player_parquets:
        player_count_value = _player_count_from_stem(parquet.stem)
        if player_count_value is None:
            continue
        done_path = _block_done_stamp_path(root, str(player_count_value), suffix)
        done_stamp = _load_done_stamp(done_path)
        source_path = row_data_path / "by_k" / f"{player_count_value}p" / curated_rows_name
        if not _done_stamp_matches(
            done_stamp,
            parquet_path=parquet,
            source_path=source_path,
            freshness=cell_freshness_sha256,
            root_seed=output_seed,
            player_count=player_count_value,
            require_sidecar=False,
        ):
            LOGGER.info(
                "Skipping per-k parquet without a valid authenticated stamp",
                extra={
                    "stage": "trueskill",
                    "parquet": str(parquet),
                    "done": str(done_path),
                },
            )
            continue
        valid_per_player_parquets.append(parquet)

    LOGGER.info(
        "TrueSkill per-root/per-k ratings complete",
        extra={
            "stage": "trueskill",
            "root": str(root),
            "blocks": len(blocks),
            "rating_cells": len(valid_per_player_parquets),
            "cross_k_rating_propagation": "unavailable",
        },
    )


def _sorted_ratings(ratings: Mapping[str, RatingStats]) -> Mapping[str, RatingStats]:
    """Return ratings ordered by strategy for stable materialisation."""

    return dict(sorted(ratings.items(), key=lambda kv: kv[0]))


def _resolve_root_row_data_dir(cfg: AppConfig) -> Path | None:
    """Resolve the canonical curated-row directory owned by the active root."""

    input_folder = cfg.root_input_stage_folder("curate")
    if input_folder is None:
        return None
    candidate = cfg.analysis_dir / input_folder
    return candidate if candidate.exists() else None


def _trueskill_cell_freshness(cfg: AppConfig) -> str:
    """Bind resumable cells to the same scoped method, code, and run lineage."""

    code_identity = cfg._code_identity or resolve_code_identity(
        _REPO_ROOT.parent,
        policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY,
    )
    return freshness_sha256(
        {
            "stage": "trueskill_cell",
            "stage_config_sha": cfg.stage_config_sha("trueskill"),
            "cache_key_version": cfg.stage_cache_key_version("trueskill"),
            "freshness_key": cfg.freshness_key(),
            "trueskill_cell_method_version": TRUESKILL_CELL_METHOD_VERSION,
            "hyperparameters": {
                "beta": cfg.trueskill.beta,
                "tau": cfg.trueskill.tau,
                "draw_probability": cfg.trueskill.draw_probability,
            },
            "ordered_input_contract": "exact_parquet_bytes_in_declared_row_order",
            "code_identity": asdict(code_identity),
            "run_lineage_sha256": cfg._run_lineage_sha256,
        }
    )


def _trueskill_code_revision(cfg: AppConfig) -> str:
    """Return the full code identity recorded by rating sidecars."""

    code_identity = cfg._code_identity or resolve_code_identity(
        _REPO_ROOT.parent,
        policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY,
    )
    if code_identity.dirty_fingerprint_sha256 is None:
        return code_identity.commit
    return f"{code_identity.commit}:development_dirty:{code_identity.dirty_fingerprint_sha256}"


def _seal_rating_cell_completion(
    cfg: AppConfig,
    *,
    cell: ScreeningRatingCell,
    done_path: Path,
    stamp: _ShardDoneStamp,
    source_path: Path,
    freshness: str,
) -> _ShardDoneStamp:
    """Publish/recover the sidecar, then seal its exact bytes into the stamp."""

    if not _done_stamp_matches(
        stamp,
        parquet_path=cell.ratings_path,
        source_path=source_path,
        freshness=freshness,
        root_seed=cell.root_seed,
        player_count=cell.k,
        require_sidecar=False,
    ):
        raise ArtifactContractError("TrueSkill cell completion does not bind current bytes")
    publish_rating_cell_contract(
        cfg,
        cell,
        completed_artifact_sha256=stamp.parquet_sha256,
        expected_sidecar_sha256=stamp.sidecar_sha256,
        code_revision=_trueskill_code_revision(cfg),
    )
    sealed = replace(
        stamp,
        sidecar_sha256=sha256_file(sidecar_path(cell.ratings_path)),
    )
    _save_done_stamp(done_path, sealed)
    return sealed


def run_trueskill_root(cfg: AppConfig, *, force: bool = False) -> None:
    """Publish canonical root/k ratings and percentile screening artifacts."""

    cell_freshness = _trueskill_cell_freshness(cfg)

    analysis_cfg = cfg.analysis
    roots = tuple(int(root) for root in (cfg.sim.seed_list or [cfg.sim.seed]))
    if len(roots) != 1:
        raise ValueError("TrueSkill execution requires exactly one root context")
    seeds = [roots[0]]

    analysis_dir = cfg.trueskill_stage_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    root_row_data_dir = _resolve_root_row_data_dir(cfg)

    env_kwargs: dict[str, float] = _coerce_trueskill_env_kwargs(
        {
            "beta": cfg.trueskill.beta,
            "tau": cfg.trueskill.tau,
            "draw_probability": cfg.trueskill.draw_probability,
        }
    )
    screening_cells: list[ScreeningRatingCell] = []

    # Sidecar recovery runs before workers only when a prior independent cell
    # completion already binds both the unchanged rating and sidecar bytes.
    if not force and root_row_data_dir is not None:
        for players in sorted({int(k) for k in cfg.sim.n_players_list}):
            suffix = f"_seed{roots[0]}"
            paths = _rating_artifact_paths(analysis_dir, str(players), suffix)
            done_path = _block_done_stamp_path(analysis_dir, str(players), suffix)
            stamp = _load_done_stamp(done_path)
            source_path = root_row_data_dir / "by_k" / f"{players}p" / cfg.curated_rows_name
            if stamp is None or stamp.sidecar_sha256 is None:
                continue
            try:
                _seal_rating_cell_completion(
                    cfg,
                    cell=ScreeningRatingCell(
                        root_seed=roots[0],
                        k=players,
                        ratings_path=paths["parquet"],
                        game_rows_path=source_path,
                    ),
                    done_path=done_path,
                    stamp=stamp,
                    source_path=source_path,
                    freshness=cell_freshness,
                )
            except ArtifactContractError:
                LOGGER.info(
                    "TrueSkill sidecar recovery identity did not validate; cell will replay",
                    extra={"stage": "trueskill", "seed": roots[0], "players": players},
                )

    for seed in seeds:
        seed_everything(seed)
        seed_results_root = cfg.results_root
        seed_row_data_dir = root_row_data_dir
        LOGGER.info(
            "TrueSkill seed run start",
            extra={
                "stage": "trueskill",
                "seed": seed,
                "analysis_dir": str(analysis_dir),
                "dataroot": str(seed_results_root),
                "row_data_dir": str(seed_row_data_dir) if seed_row_data_dir else None,
            },
        )
        run_trueskill(
            output_seed=seed,
            root=analysis_dir,
            dataroot=seed_results_root,
            row_data_dir=seed_row_data_dir,
            curated_rows_name=cfg.curated_rows_name,
            workers=analysis_cfg.n_jobs or None,
            resume_per_n=not force,
            env_kwargs=env_kwargs,
            mp_start_method=analysis_cfg.mp_start_method,
            progress_logging=analysis_cfg.progress_logging,
            cell_freshness_sha256=cell_freshness,
        )

        seed_outputs: dict[str, Mapping[str, RatingStats]] = {}
        for parquet in _iter_rating_parquets(analysis_dir, f"_seed{seed}"):
            player_count = _player_count_from_stem(parquet.stem)
            if player_count is None:
                continue
            done_path = _block_done_stamp_path(analysis_dir, str(player_count), f"_seed{seed}")
            done_stamp = _load_done_stamp(done_path)
            game_rows_path = (
                seed_row_data_dir / "by_k" / f"{player_count}p" / cfg.curated_rows_name
                if seed_row_data_dir is not None
                else None
            )
            if game_rows_path is None or not _done_stamp_matches(
                done_stamp,
                parquet_path=parquet,
                source_path=game_rows_path,
                freshness=cell_freshness,
                root_seed=seed,
                player_count=player_count,
                require_sidecar=False,
            ):
                LOGGER.warning(
                    "Ignoring incomplete TrueSkill rating cell",
                    extra={
                        "stage": "trueskill",
                        "seed": seed,
                        "players": player_count,
                        "parquet": str(parquet),
                        "done": str(done_path),
                    },
                )
                continue
            stats = _load_ratings_parquet(parquet)
            ordered_stats = _sorted_ratings(stats)
            seed_outputs[f"{player_count}p"] = ordered_stats  # keyed for logging/debug
            screening_cell = ScreeningRatingCell(
                root_seed=seed,
                k=player_count,
                ratings_path=parquet,
                game_rows_path=game_rows_path,
            )
            assert done_stamp is not None
            done_stamp = _seal_rating_cell_completion(
                cfg,
                cell=screening_cell,
                done_path=done_path,
                stamp=done_stamp,
                source_path=game_rows_path,
                freshness=cell_freshness,
            )
            if not _done_stamp_matches(
                done_stamp,
                parquet_path=parquet,
                source_path=game_rows_path,
                freshness=cell_freshness,
                root_seed=seed,
                player_count=player_count,
            ):
                raise ArtifactContractError("TrueSkill cell failed authenticated sealing")
            screening_cells.append(screening_cell)

        if not seed_outputs:
            raise RuntimeError(f"No per-player outputs generated for seed {seed}")
        LOGGER.info(
            "TrueSkill seed run complete",
            extra={
                "stage": "trueskill",
                "seed": seed,
                "per_players": sorted(seed_outputs.keys()),
            },
        )

    if not screening_cells:
        raise RuntimeError("TrueSkill produced no canonical root/k rating cells")
    expected_cells = {(roots[0], int(k)) for k in cfg.sim.n_players_list}
    observed_cells = {(cell.root_seed, cell.k) for cell in screening_cells}
    if observed_cells != expected_cells or len(screening_cells) != len(expected_cells):
        missing = sorted(expected_cells.difference(observed_cells))
        extra = sorted(observed_cells.difference(expected_cells))
        raise RuntimeError(
            "TrueSkill must produce exactly every configured root/k cell; "
            f"missing={missing}, extra={extra}"
        )
    for cell in screening_cells:
        _ensure_auxiliary_rating_sidecars(
            cfg,
            cell=cell,
            suffix=f"_seed{cell.root_seed}",
            code_revision=_trueskill_code_revision(cfg),
        )
    contribution_path = build_percentile_contribution(cfg, screening_cells, force=force)
    diagnostics_path = build_screening_diagnostics(cfg, screening_cells, force=force)
    LOGGER.info(
        "TrueSkill screening outputs written",
        extra={
            "stage": "trueskill",
            "candidate_contribution": str(contribution_path),
            "diagnostics": str(diagnostics_path) if diagnostics_path else None,
            "rating_cells": len(screening_cells),
        },
    )
