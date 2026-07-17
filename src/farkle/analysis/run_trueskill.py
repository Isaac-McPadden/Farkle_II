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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Iterable,
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
    ScreeningRatingCell,
    build_percentile_contribution,
    build_screening_diagnostics,
    publish_rating_cell_contract,
)
from farkle.config import AppConfig
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.parallel import resolve_mp_context
from farkle.utils.progress import ProgressLogConfig, ScheduledProgressLogger
from farkle.utils.random import seed_everything
from farkle.utils.writer import atomic_path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # hop out of src/farkle
# Default location of tournament result blocks when no path is supplied
DEFAULT_DATAROOT = _REPO_ROOT / "data" / "results"

LOGGER = logging.getLogger(__name__)

DEFAULT_RATING = trueskill.Rating()  # uses env defaults


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
    """Simple TrueSkill rating stats container."""

    mu: float
    sigma: float


def _ratings_to_table(
    mapping: Mapping[str, Union[trueskill.Rating, "RatingStats", Tuple[float, float], float]],
) -> pa.Table:
    """Coerce {strategy: Rating|RatingStats|(mu,sigma)|mu} → Arrow table."""
    strategies: list[str] = []
    mus: list[float] = []
    sigmas: list[float] = []
    for k, v in mapping.items():
        if isinstance(v, (trueskill.Rating, RatingStats)):
            mu, sigma = float(v.mu), float(v.sigma)
        elif isinstance(v, (tuple, list)) and len(v) >= 2:
            mu, sigma = float(v[0]), float(v[1])
        else:
            # fallback: scalar mu with default sigma (not expected here)
            mu, sigma = float(v), 0.0  # type: ignore[arg-type]
        strategies.append(_normalize_strategy_id(k))
        mus.append(mu)
        sigmas.append(sigma)
    strategy_values = [str(s) for s in strategies]
    return pa.table(
        {
            "strategy": pa.array(strategy_values, type=pa.string()),
            "mu": pa.array(mus, type=pa.float64()),
            "sigma": pa.array(sigmas, type=pa.float64()),
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
    tbl = pq.read_table(path, columns=["strategy", "mu", "sigma"])
    data = tbl.to_pydict()
    out: dict[str, RatingStats] = {}
    for s, mu, sg in zip(data["strategy"], data["mu"], data["sigma"], strict=True):
        out[str(s)] = RatingStats(float(mu), float(sg))
    return out


# ---------- Per-N checkpointing ----------
@dataclass
class _BlockCkpt:
    """Checkpoint progress for a single player-count block."""

    row_file: str
    row_group: int
    batch_index: int
    games_done: int
    ratings_path: str
    version: int = 1


@dataclass(slots=True)
class _ShardDoneStamp:
    """Completion stamp for independently resumable shards."""

    shard_key: str
    parquet_path: str
    rows: int
    created_at: float
    version: int = 1


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
            version=int(payload.get("version", 1)),
        )
    except (KeyError, TypeError, ValueError):
        return None


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
    }
    for key, expected_type in required_keys.items():
        value = payload_any.get(key)
        if not isinstance(value, expected_type):
            return None

    version_value = payload_any.get("version", 1)
    if not isinstance(version_value, int):
        return None

    return _BlockCkpt(
        row_file=payload_any["row_file"],
        row_group=payload_any["row_group"],
        batch_index=payload_any["batch_index"],
        games_done=payload_any["games_done"],
        ratings_path=payload_any["ratings_path"],
        version=version_value,
    )


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


def _players_and_ranks_from_batch(batch: pa.Table, n: int) -> Iterator[tuple[list[str], list[int]]]:
    """Robust per-row extraction (tie-friendly precedence):
    1) Derive placements from P#_rank (preserves ties)
    2) Else, use seat_ranks (strict order, no ties)
    3) Else, fallback: winner vs (n-1) tied second.
    """
    cols = set(batch.column_names)

    def col(name: str) -> pa.ChunkedArray | None:
        """Return column from the batch when present, otherwise ``None``."""
        return batch.column(name) if name in cols else None

    sr_col = col("seat_ranks")
    ranks_list = sr_col.to_pylist() if sr_col is not None else None
    winner_col_name: str | None
    if "winner_seat" in cols:
        winner_col_name = "winner_seat"
    elif "winner" in cols:
        winner_col_name = "winner"
    else:
        winner_col_name = None
    winner_col = batch.column(winner_col_name) if winner_col_name is not None else None
    winner_list = winner_col.to_pylist() if winner_col is not None else None
    strat_col_names: list[str] = [f"P{i}_strategy" for i in range(1, n + 1)]
    rank_col_names: list[str] = [f"P{i}_rank" for i in range(1, n + 1)]
    strat_cols: list[pa.ChunkedArray | None] = [col(name) for name in strat_col_names]
    rank_cols: list[pa.ChunkedArray | None] = [col(name) for name in rank_col_names]
    n_rows = batch.num_rows
    for r in range(n_rows):
        seats = [sc[r].as_py() if sc is not None else None for sc in strat_cols]
        seats = [_normalize_strategy_id(s) if s is not None else None for s in seats]
        # 1) prefer numeric ranks (preserves ties)
        if any(rc is not None for rc in rank_cols):
            pairs = [(i, rc[r].as_py()) for i, rc in enumerate(rank_cols) if rc is not None]
            pairs = [(i, rv) for i, rv in pairs if rv is not None]
            if len(pairs) >= 2:
                pairs.sort(key=lambda x: x[1])
                uniq = sorted({rv for _, rv in pairs})
                remap = {rv: j for j, rv in enumerate(uniq)}
                players, rr = [], []
                for i, rv in pairs:
                    p = seats[i]
                    if p is None:
                        continue
                    players.append(p)
                    rr.append(remap[rv])
                if len(players) >= 2:
                    yield cast(list[str], players), rr
                    continue
        # 2) seat_ranks (strict order, no ties)
        if ranks_list is not None:
            order = ranks_list[r]
            if not order:
                continue
            order = cast(Iterable[str], order)
            players = [seats[int(s[1:]) - 1] for s in order]
            if any(p is None for p in players):  # skip incomplete rows
                continue
            yield cast(list[str], players), list(range(len(players)))
            continue
        # 3) fallback
        if winner_list is None or not winner_list[r]:
            continue
        try:
            w_idx = int(str(winner_list[r])[1:]) - 1
        except Exception:
            continue
        winner = seats[w_idx]
        if winner is None:
            continue
        losers = [s for j, s in enumerate(seats) if j != w_idx and s is not None]
        if losers:
            yield [winner] + losers, [0] + [1] * len(losers)


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
) -> tuple[str, int]:
    """
    Process one <N>_players block with optional checkpointing.
    Returns (player_count_str, n_games).
    """
    block = Path(block_dir)
    root = Path(root_dir)
    row_data_path = Path(row_data_dir) if row_data_dir else None

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
    if done_stamp is not None:
        stamped_parquet = Path(done_stamp.parquet_path)
        if stamped_parquet.exists() and parquet_path.exists() and stamped_parquet == parquet_path:
            LOGGER.info(
                "TrueSkill block shard done stamp found; skipping",
                extra={
                    "stage": "trueskill",
                    "block": player_count,
                    "row_file": str(row_file),
                    "parquet": str(parquet_path),
                    "done": str(done_path),
                },
            )
            return player_count, max(0, int(done_stamp.rows))

    if parquet_path.exists() and parquet_path.stat().st_mtime >= row_file.stat().st_mtime:
        try:
            md = pq.read_metadata(row_file)
            n_rows = md.num_rows
        except Exception:
            n_rows = 0
        LOGGER.info(
            "TrueSkill block up-to-date",
            extra={
                "stage": "trueskill",
                "block": player_count,
                "row_file": str(row_file),
                "parquet": str(parquet_path),
            },
        )
        return player_count, n_rows

    env = trueskill.TrueSkill(**_coerce_trueskill_env_kwargs(env_kwargs))

    start_rg = 0
    start_bi = 0
    n_games = 0
    ratings: dict[str, trueskill.Rating] = {}
    if resume:
        ck = _load_block_ckpt(ck_path)
        if ck and Path(ck.row_file) == row_file:
            start_rg = ck.row_group
            start_bi = ck.batch_index
            n_games = ck.games_done
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
            for players, ranks in _players_and_ranks_from_batch(batch, n):
                if keepers:
                    filt = [(p, r) for p, r in zip(players, ranks, strict=True) if p in keepers]
                    if len(filt) < 2:
                        continue
                    players = [p for p, _ in filt]
                    ranks = [r for _, r in filt]
                    uq = sorted(set(ranks))
                    rmap = {rv: j for j, rv in enumerate(uq)}
                    ranks = [rmap[r] for r in ranks]

                for s in players:
                    if s not in ratings:
                        ratings[s] = env.create_rating()
                teams = [[ratings[s]] for s in players]
                new = env.rate(teams, ranks=ranks)
                for s, t in zip(players, new, strict=False):
                    ratings[s] = t[0]
                n_games += 1

            batches_since_ck += 1
            if batches_since_ck >= checkpoint_every_batches or (time.time() - last_ck) > 60:
                _save_ratings_parquet(rk_path, ratings)
                _save_block_ckpt(
                    ck_path,
                    _BlockCkpt(
                        row_file=str(row_file),
                        row_group=rg,
                        batch_index=bi + 1,
                        games_done=n_games,
                        ratings_path=str(rk_path),
                    ),
                )
                if progress_logger is not None:
                    progress_logger.maybe_log(
                        n_games,
                        detail=(
                            f"row group {rg + 1}, batch {bi + 1}, "
                            f"{len(ratings):,} ratings tracked"
                        ),
                        extra={
                            "stage": "trueskill",
                            "block": player_count,
                            "games_done": n_games,
                            "games_total": total_input_games,
                            "row_group": rg,
                            "batch_index": bi + 1,
                        },
                    )
                last_ck = time.time()
                batches_since_ck = 0

        ratings_stats = {k: RatingStats(v.mu, v.sigma) for k, v in ratings.items()}
        # Atomic writes for per-N outputs
        # Write per-N ratings as Parquet (strategy, mu, sigma)
        _save_ratings_parquet(parquet_path, ratings_stats)

        json_path = paths["json"]
        with atomic_path(str(json_path)) as tmp_path:
            Path(tmp_path).write_text(
                json.dumps({k: {"mu": v.mu, "sigma": v.sigma} for k, v in ratings_stats.items()})
            )
        _save_done_stamp(
            done_path,
            _ShardDoneStamp(
                shard_key=f"k={player_count}",
                parquet_path=str(parquet_path),
                rows=int(n_games),
                created_at=time.time(),
            ),
        )
        _save_ratings_parquet(shard_parquet, ratings_stats)
        _save_done_stamp(
            shard_done_path,
            _ShardDoneStamp(
                shard_key=f"k={player_count}",
                parquet_path=str(shard_parquet),
                rows=int(n_games),
                created_at=time.time(),
            ),
        )

        run_completed = True
        return player_count, n_games
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
        if done_stamp is None:
            LOGGER.info(
                "Skipping per-k parquet without done stamp",
                extra={
                    "stage": "trueskill",
                    "parquet": str(parquet),
                    "done": str(done_path),
                },
            )
            continue
        if Path(done_stamp.parquet_path) != parquet:
            LOGGER.info(
                "Skipping per-k parquet with mismatched done stamp",
                extra={
                    "stage": "trueskill",
                    "parquet": str(parquet),
                    "stamped_parquet": done_stamp.parquet_path,
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


def run_trueskill_root(cfg: AppConfig) -> None:
    """Publish canonical root/k ratings and percentile screening artifacts."""

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
            env_kwargs=env_kwargs,
            mp_start_method=analysis_cfg.mp_start_method,
            progress_logging=analysis_cfg.progress_logging,
        )

        seed_outputs: dict[str, Mapping[str, RatingStats]] = {}
        for parquet in _iter_rating_parquets(analysis_dir, f"_seed{seed}"):
            players = _player_count_from_stem(parquet.stem)
            if players is None:
                continue
            done_path = _block_done_stamp_path(analysis_dir, str(players), f"_seed{seed}")
            done_stamp = _load_done_stamp(done_path)
            if done_stamp is None or Path(done_stamp.parquet_path) != parquet:
                LOGGER.warning(
                    "Ignoring incomplete TrueSkill rating cell",
                    extra={
                        "stage": "trueskill",
                        "seed": seed,
                        "players": players,
                        "parquet": str(parquet),
                        "done": str(done_path),
                    },
                )
                continue
            stats = _load_ratings_parquet(parquet)
            ordered_stats = _sorted_ratings(stats)
            seed_outputs[f"{players}p"] = ordered_stats  # keyed for logging/debug
            game_rows_path: Path | None = None
            if seed_row_data_dir is not None:
                candidate = seed_row_data_dir / "by_k" / f"{players}p" / cfg.curated_rows_name
                game_rows_path = candidate if candidate.exists() else None
            screening_cell = ScreeningRatingCell(
                root_seed=seed,
                k=players,
                ratings_path=parquet,
                game_rows_path=game_rows_path,
            )
            publish_rating_cell_contract(cfg, screening_cell)
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
    contribution_path = build_percentile_contribution(cfg, screening_cells)
    diagnostics_path = build_screening_diagnostics(cfg, screening_cells)
    LOGGER.info(
        "TrueSkill screening outputs written",
        extra={
            "stage": "trueskill",
            "candidate_contribution": str(contribution_path),
            "diagnostics": str(diagnostics_path) if diagnostics_path else None,
            "rating_cells": len(screening_cells),
        },
    )
