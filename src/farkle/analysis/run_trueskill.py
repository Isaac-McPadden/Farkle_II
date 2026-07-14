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
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import trueskill
import yaml  # type: ignore[import-untyped]
from trueskill import Rating

from farkle.analysis.trueskill_screening import (
    ScreeningRatingCell,
    build_percentile_contribution,
    build_screening_diagnostics,
    publish_rating_cell_contract,
)
from farkle.config import AppConfig, ArtifactScope
from farkle.orchestration.seed_utils import resolve_results_dir, split_seeded_results_dir
from farkle.utils.artifact_contract import make_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic, write_parquet_atomic
from farkle.utils.parallel import resolve_mp_context
from farkle.utils.progress import ProgressLogConfig, ScheduledProgressLogger
from farkle.utils.random import seed_everything
from farkle.utils.schema_helpers import n_players_from_schema
from farkle.utils.writer import atomic_path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # hop out of src/farkle
# Default location of tournament result blocks when no path is supplied
DEFAULT_DATAROOT = _REPO_ROOT / "data" / "results"

LOGGER = logging.getLogger(__name__)

DEFAULT_RATING = trueskill.Rating()  # uses env defaults


class RatingArtifactPaths(TypedDict):
    """Resolved canonical and legacy file paths for a rating artifact bundle.

    The mapping includes the preferred parquet/json/checkpoint paths, the owning
    directory, and any legacy locations that should be migrated or inspected.
    """

    parquet: Path
    json: Path
    ckpt: Path
    checkpoint: Path
    dir: Path
    legacy_parquet: list[Path]
    legacy_json: list[Path]
    legacy_ckpt: list[Path]
    legacy_checkpoint: list[Path]


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


def _ensure_new_location(dest: Path, *legacy_paths: Path) -> Path:
    """Return the canonical destination without interpreting legacy files."""

    _ = legacy_paths
    return dest


def _rating_artifact_paths(
    root: Path, player_count: str, suffix: str, *, legacy_root: Path | None = None
) -> RatingArtifactPaths:
    """Return canonical and legacy paths for per-player artifacts."""

    per_dir = _per_player_dir(root, player_count)
    legacy_candidates = [root / "data" / f"{player_count}p"]
    if legacy_root is not None and legacy_root != root:
        legacy_candidates.extend(
            [
                legacy_root / "data" / f"{player_count}p",
                legacy_root / f"{player_count}p",
                legacy_root,
            ]
        )
    base_name = f"ratings_{player_count}{suffix}"
    return {
        "dir": per_dir,
        "parquet": per_dir / f"{base_name}.parquet",
        "json": per_dir / f"{base_name}.json",
        "ckpt": per_dir / f"{base_name}.ckpt.json",
        "checkpoint": per_dir / f"{base_name}.checkpoint.parquet",
        "legacy_parquet": [base / f"{base_name}.parquet" for base in legacy_candidates],
        "legacy_json": [base / f"{base_name}.json" for base in legacy_candidates],
        "legacy_ckpt": [base / f"{base_name}.ckpt.json" for base in legacy_candidates],
        "legacy_checkpoint": [
            base / f"{base_name}.checkpoint.parquet" for base in legacy_candidates
        ],
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
    """Extract the player count from a ratings filename stem.

    Accepts stems such as ``ratings_2_seed21`` or ``ratings_2p_seed21`` to
    tolerate variations in legacy/materialised filenames.
    """

    match = re.match(r"ratings_(\d+)(?:p)?(?:_|$)", stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _find_combined_parquet(base: Path | None) -> Path | None:
    """Return path to combined ``all_ingested_rows.parquet`` if it exists.

    The *base* argument may point to a results directory or directly to the
    ``analysis/data`` directory. This helper tries a few common locations and
    returns the first existing path. ``None`` is returned when no candidate is
    found.
    """

    if base is None:
        return None
    base = Path(base)
    candidates = [
        base / "analysis" / "02_combine" / "concat_ks" / "all_ingested_rows.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


@dataclass(slots=True)
class RatingStats:
    """Simple TrueSkill rating stats container."""

    mu: float
    sigma: float


# ---------- Checkpointing ----------
@dataclass
class _TSCheckpoint:
    """Serialize streaming progress for the combined ratings pass."""

    source: str  # parquet path
    row_group: int  # next row-group to start at
    batch_index: int  # next batch index within that row-group
    games_done: int
    ratings_path: str  # where the interim ratings parquet lives
    version: int = 1


def _save_ckpt(path: Path, ck: _TSCheckpoint) -> None:
    """Persist a meta-level checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(asdict(ck)))


def _load_ckpt(path: Path) -> Optional[_TSCheckpoint]:
    """Load a previously written checkpoint if it exists and parses cleanly."""
    if not path.exists():
        return None
    try:
        payload_any: object = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(payload_any, dict):
        return None

    required_keys = {
        "source": str,
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

    return _TSCheckpoint(
        source=payload_any["source"],
        row_group=payload_any["row_group"],
        batch_index=payload_any["batch_index"],
        games_done=payload_any["games_done"],
        ratings_path=payload_any["ratings_path"],
        version=version_value,
    )

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


def _aggregation_shard_paths(combined_dir: Path, shard_key: str, suffix: str) -> tuple[Path, Path]:
    """Return parquet and done-stamp paths for an aggregation shard."""

    safe_key = shard_key.replace("/", "_")
    base = combined_dir / f"artifact.{safe_key}{suffix}"
    return base.with_suffix(".parquet"), base.with_suffix(".done.json")


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


def _rate_single_pass(
    source: Path,
    *,
    env: trueskill.TrueSkill,
    resume: bool,
    checkpoint_path: Path,
    ratings_ckpt_path: Path,
    batch_rows: int,
    checkpoint_every_batches: int = 500,
) -> tuple[dict[str, RatingStats], int]:
    """Stream all games from a single combined parquet, with checkpoint/resume."""
    # Find n from schema (max seat present)
    schema = pq.read_schema(source)
    names = set(schema.names)
    n = max(int(nm[1]) for nm in names if nm.startswith("P") and nm.endswith("_strategy"))

    # Initialise ratings & resume point
    start_rg = 0
    start_bi = 0
    games = 0
    ratings: dict[str, trueskill.Rating] = {}
    if resume:
        ck = _load_ckpt(checkpoint_path)
        if ck and Path(ck.source) == source:
            start_rg = ck.row_group
            start_bi = ck.batch_index
            games = ck.games_done
            if Path(ck.ratings_path).exists():
                # load interim ratings from parquet and build env Rating objects
                interim = _load_ratings_parquet(Path(ck.ratings_path))
                ratings = {k: env.create_rating(mu=v.mu, sigma=v.sigma) for k, v in interim.items()}

    last_ck = time.time()
    batches_since_ck = 0
    for rg, bi, batch in _stream_batches(
        source,
        columns=list(schema.names),
        start_row_group=start_rg,
        start_batch_idx=start_bi,
        batch_rows=batch_rows,
    ):
        for players, ranks in _players_and_ranks_from_batch(batch, n):
            # seed ratings
            for s in players:
                if s not in ratings:
                    ratings[s] = env.create_rating()
            teams = [[ratings[s]] for s in players]
            new = env.rate(teams, ranks=ranks)
            for s, t in zip(players, new, strict=False):
                ratings[s] = t[0]
            games += 1

        # periodic checkpoint
        batches_since_ck += 1
        if batches_since_ck >= checkpoint_every_batches or (time.time() - last_ck) > 60:
            _save_ratings_parquet(ratings_ckpt_path, ratings)
            _save_ckpt(
                checkpoint_path,
                _TSCheckpoint(
                    source=str(source),
                    row_group=rg,
                    batch_index=bi + 1,
                    games_done=games,
                    ratings_path=str(ratings_ckpt_path),
                ),
            )
            last_ck = time.time()
            batches_since_ck = 0

    # finalise
    stats = {k: RatingStats(v.mu, v.sigma) for k, v in ratings.items()}
    return stats, games


def _coerce_ratings(obj: Mapping[str, object]) -> dict[str, RatingStats]:
    """Accept {strategy: RatingStats | {'mu':..,'sigma':..} | (mu,sigma)}."""
    out: dict[str, RatingStats] = {}
    for k, v in obj.items():
        if isinstance(v, RatingStats):
            out[k] = v
        elif isinstance(v, dict) and "mu" in v and "sigma" in v:
            out[k] = RatingStats(float(v["mu"]), float(v["sigma"]))
        else:
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                mu, sigma = v[:2]
            else:
                mu, sigma = 0.0, 0.0
            out[k] = RatingStats(float(mu), float(sigma))
    return out


def _ensure_seed_ratings(
    ratings: dict[str, Rating], all_strategies: list[str], env: trueskill.TrueSkill
) -> None:
    """Guarantee every strategy has an initial rating for the current seed."""
    for strat in all_strategies:
        ratings.setdefault(strat, env.create_rating())


def _read_manifest_seed(path: Path) -> int:
    """Return the tournament seed recorded in ``manifest.yaml``.

    Parameters
    ----------
    path : Path
        Location of the manifest file. The YAML is expected to contain a
        ``seed`` entry.

    Returns
    -------
    int
        The integer seed found in the file. ``0`` is returned when the file
        does not exist or the value is missing.
    """

    try:
        data = yaml.safe_load(path.read_text())
        return int(data.get("seed", 0))
    except FileNotFoundError:
        return 0


def _update_ratings(
    games: list[list[str]],
    keepers: list[str],
    env: trueskill.TrueSkill,
) -> dict[str, RatingStats]:
    """Update strategy ratings for a block of games.

    Parameters
    ----------
    games : list[list[str]]
        Each inner list contains strategy names ordered by finishing position
        for a single game.
    keepers : list[str]
        Strategies to include in the rating calculation. If empty, all
        strategies appearing in ``games`` are rated.
    env : trueskill.TrueSkill
        The environment used to perform the TrueSkill updates.

    Returns
    -------
    dict[str, RatingStats]
        Mapping from strategy name to its RatingStats tuple.
    """
    ratings: dict[str, trueskill.Rating] = {k: env.create_rating() for k in keepers}
    all_strats: set[str] = set(keepers)
    for g in games:
        all_strats.update(g)
    _ensure_seed_ratings(ratings, list(all_strats), env)

    for game in games:
        # keep only the strategies we really want to rate
        players = [s for s in game if (not keepers or s in keepers)]
        teams = [[ratings.setdefault(s, env.create_rating())] for s in players]

        if len(teams) < 2:
            continue  # need at least two teams for a rating update

        new_teams = env.rate(teams, ranks=range(len(teams)))
        for s, team_rating in zip(players, new_teams, strict=True):
            ratings[s] = team_rating[0]

    return {k: RatingStats(r.mu, r.sigma) for k, r in ratings.items()}


def _strategy_column_names(n: int) -> list[str]:
    return [f"P{i}_strategy" for i in range(1, n + 1)]


def _iter_players_and_ranks(
    row_file: Path, n: int, batch_size: int = 100_000
) -> Iterator[tuple[list[str], list[int]]]:
    """Yield (players, ranks) per game in a streaming fashion.
    Prefers explicit placements via ``seat_ranks``; if absent, derives placements
    from ``P#_rank``; if only a winner is known, treats others as a single tied group.
    """
    dataset = ds.dataset(row_file, format="parquet")
    schema = dataset.schema
    has_seat_ranks = schema.get_field_index("seat_ranks") != -1

    if has_seat_ranks:
        seat_rank_col_name = "seat_ranks"

        strategy_column_names = _strategy_column_names(n)
        seat_rank_columns: list[str] = [seat_rank_col_name, *strategy_column_names]

        for batch in dataset.to_batches(columns=seat_rank_columns, batch_size=batch_size):
            ranks_col: pa.Array = batch.column(seat_rank_col_name)
            ranks_list = ranks_col.to_pylist()  # list[list[str]] or None

            strat_cols: list[pa.Array] = [batch.column(name) for name in strategy_column_names]

            for r, order in enumerate(ranks_list):
                if not order:
                    continue
                players = [strat_cols[int(str(seat)[1:]) - 1][r].as_py() for seat in order]
                if any(p is None for p in players):
                    continue
                yield [_normalize_strategy_id(p) for p in players], list(range(len(players)))
        return

    # seat_ranks not present → derive from P#_rank or fall back to winner + tied losers
    rank_col_names: list[str] = [f"P{i}_rank" for i in range(1, n + 1)]
    strategy_column_names = _strategy_column_names(n)
    winner_col_name: str = (
        "winner_seat" if schema.get_field_index("winner_seat") != -1 else "winner"
    )
    fallback_columns: list[str] = [
        winner_col_name,
        *rank_col_names,
        *strategy_column_names,
    ]

    for batch in dataset.to_batches(columns=fallback_columns, batch_size=batch_size):
        winner_seats = batch.column(winner_col_name).to_pylist()

        rank_cols: list[pa.Array] = [batch.column(name) for name in rank_col_names]
        fallback_strat_cols: list[pa.Array] = [batch.column(name) for name in strategy_column_names]

        ranks = [[col[i].as_py() for col in rank_cols] for i in range(len(batch))]
        strats = [[col[i].as_py() for col in fallback_strat_cols] for i in range(len(batch))]

        for winner_seat, seat_ranks, seat_strats in zip(winner_seats, ranks, strats, strict=True):
            # try to build placements from P#_rank if we have ≥2 ranks
            pairs = [(i, rv) for i, rv in enumerate(seat_ranks) if rv is not None]
            if len(pairs) >= 2:
                # sort by rank ascending; remap rank values to 0..K-1 to keep them dense
                pairs.sort(key=lambda x: x[1])
                uniq = sorted({rv for _, rv in pairs})
                remap = {rv: j for j, rv in enumerate(uniq)}
                players = []
                pranks = []
                for i, rv in pairs:
                    p = seat_strats[i]
                    if p is None:
                        continue
                    players.append(_normalize_strategy_id(p))
                    pranks.append(remap[rv])
                if len(players) >= 2:
                    yield players, pranks
                    continue

            # fallback: winner + (n-1) tied for 2nd
            if not winner_seat:
                continue
            try:
                w_idx = int(str(winner_seat)[1:]) - 1  # "P3" -> 2
            except Exception:
                continue
            winner = seat_strats[w_idx]
            if winner is None:
                continue
            losers = [
                _normalize_strategy_id(s)
                for j, s in enumerate(seat_strats)
                if j != w_idx and s is not None
            ]
            if losers:
                yield [_normalize_strategy_id(winner)] + losers, [0] + [1] * len(losers)


def _rate_stream(
    row_file: Path, n: int, keepers: list[str], env: trueskill.TrueSkill, batch_size: int = 100_000
) -> tuple[dict[str, RatingStats], int]:
    """Stream TrueSkill updates from parquet without materialising all games.
    Supports ties and multiple sources of placement (seat_ranks or P#_rank)."""
    if keepers:
        keepers = [_normalize_strategy_id(k) for k in keepers]
    ratings: dict[str, trueskill.Rating] = {k: env.create_rating() for k in keepers}
    n_games = 0
    for players, ranks in _iter_players_and_ranks(row_file, n, batch_size):
        # Optional keepers filter
        if keepers:
            filt = [(p, r) for p, r in zip(players, ranks, strict=True) if p in keepers]
            if len(filt) < 2:
                continue
            players = [p for p, _ in filt]
            ranks = [r for _, r in filt]
            # make ranks dense again in case we dropped teams
            uniq = sorted(set(ranks))
            rmap = {rv: j for j, rv in enumerate(uniq)}
            ranks = [rmap[r] for r in ranks]
        # seed ratings on the fly and rate with ranks=
        for s in players:
            ratings.setdefault(s, env.create_rating())
        teams = [[ratings[s]] for s in players]
        new_teams = env.rate(teams, ranks=ranks)
        for s, team_rating in zip(players, new_teams, strict=False):
            ratings[s] = team_rating[0]
        n_games += 1
    return ({k: RatingStats(r.mu, r.sigma) for k, r in ratings.items()}, n_games)


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
    legacy_root = root.parent
    row_data_path = Path(row_data_dir) if row_data_dir else None

    player_count = block.name.split("_")[0]
    per_player_dir = _per_player_dir(root, player_count)
    per_player_dir.mkdir(parents=True, exist_ok=True)
    keep_path = block / f"keepers_{player_count}.npy"
    keepers = np.load(keep_path).tolist() if keep_path.exists() else []
    keepers = [_normalize_strategy_id(k) for k in keepers]

    # Locate curated parquet
    row_file = per_player_dir / f"{player_count}p_ingested_rows.parquet"
    n = int(player_count)
    if not row_file.exists():
        alt_candidates: list[Path] = []
        if row_data_path is not None:
            per_n_dir = row_data_path
            if per_n_dir.name != f"{player_count}p":
                per_n_dir = per_n_dir / f"{player_count}p"
            name_candidates = [f"{player_count}p_ingested_rows.parquet"]
            if curated_rows_name:
                name_candidates.insert(0, curated_rows_name)
            name_candidates.append("game_rows.parquet")
            seen_names: set[str] = set()
            for name in name_candidates:
                if name in seen_names:
                    continue
                seen_names.add(name)
                alt_candidates.append(per_n_dir / name)
        row_file_candidate = next((cand for cand in alt_candidates if cand.exists()), None)
        if row_file_candidate is None:
            candidate = next(block.glob("*p_rows.parquet"), None)
            if candidate is None:
                return player_count, 0
            row_file = candidate
            n = n_players_from_schema(pq.read_schema(row_file))
        else:
            row_file = row_file_candidate
    try:
        total_input_games = max(0, int(pq.read_metadata(row_file).num_rows))
    except Exception:
        total_input_games = 0

    # Up-to-date guard
    paths = _rating_artifact_paths(root, player_count, suffix, legacy_root=legacy_root)
    parquet_path = _ensure_new_location(paths["parquet"], *paths["legacy_parquet"])
    ck_path = _ensure_new_location(paths["ckpt"], *paths["legacy_ckpt"])
    rk_path = _ensure_new_location(paths["checkpoint"], *paths["legacy_checkpoint"])
    json_path = _ensure_new_location(paths["json"], *paths["legacy_json"])
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

        json_path = _ensure_new_location(paths["json"], *paths["legacy_json"])
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
        Directory containing curated per-player parquet inputs. When ``None``
        the function attempts to use ``<root.parent>/02_combine/data`` if it
        exists, falling back to ``<root.parent>/01_combine/data`` for legacy
        layouts.
    curated_rows_name: str | None, optional
        Custom filename for curated per-player inputs. When unset, the loader
        searches for ``<n>p_ingested_rows.parquet`` and ``game_rows.parquet``.

    Production side effects are canonical per-root/per-k ratings only. Formal
    cross-k model-sigma propagation is unavailable.
    """
    if dataroot is None:
        base = Path(root) / "results" if root is not None else DEFAULT_DATAROOT
    else:
        base = Path(dataroot)

    root = Path(root) if root is not None else base / "analysis" / "06_trueskill"
    root.mkdir(parents=True, exist_ok=True)
    legacy_root = root.parent
    row_data_path = Path(row_data_dir) if row_data_dir is not None else None
    if row_data_path is None:
        for inferred in (
            legacy_root / "02_combine" / "data",
            legacy_root / "01_combine" / "data",
        ):
            if inferred.exists():
                row_data_path = inferred
                break
    _read_manifest_seed(base / "manifest.yaml")
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
                    continue
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


def _resolve_seed_results_root(cfg: AppConfig, seed: int) -> Path:
    """Resolve per-seed results root for interseed processing."""

    base_root, parsed_seed = split_seeded_results_dir(cfg.results_root)
    if parsed_seed is None:
        return cfg.results_root
    return resolve_results_dir(base_root, int(seed))


def _resolve_seed_row_data_dir(cfg: AppConfig, seed_results_root: Path) -> Path | None:
    """Resolve curated row-data directory under a specific seed's analysis root."""

    analysis_root = seed_results_root / cfg.io.analysis_subdir
    candidates: list[Path] = []

    input_folder = cfg._interseed_input_folder("curate")
    if input_folder is not None:
        candidates.append(analysis_root / input_folder)

    stage_folder = cfg.stage_layout.folder_for("curate")
    if stage_folder is not None:
        stage_candidate = analysis_root / stage_folder
        if stage_candidate not in candidates:
            candidates.append(stage_candidate)

    legacy_candidate = analysis_root / "curate"
    if legacy_candidate not in candidates:
        candidates.append(legacy_candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_trueskill_all_seeds(cfg: AppConfig) -> None:
    """Publish canonical root/k ratings and percentile screening artifacts."""

    analysis_cfg = cfg.analysis
    seeds_cfg = cfg.sim.seed_list or [cfg.sim.seed]
    deduped: list[int] = []
    seen: set[int] = set()
    for raw in seeds_cfg:
        seed = int(raw)
        if seed in seen:
            continue
        seen.add(seed)
        deduped.append(seed)
    seeds = sorted(deduped)

    analysis_dir = cfg.trueskill_stage_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    concat_dir = cfg.concat_ks_dir("trueskill")
    default_row_data_dir = cfg.resolve_input_stage_dir("curate")
    if default_row_data_dir is not None and not default_row_data_dir.exists():
        default_row_data_dir = None
    seed_results_roots = {seed: _resolve_seed_results_root(cfg, seed) for seed in seeds}
    seed_row_data_dirs = {
        seed: _resolve_seed_row_data_dir(cfg, seed_results_roots[seed]) for seed in seeds
    }

    env_kwargs: dict[str, float] = _coerce_trueskill_env_kwargs(
        {
            "beta": cfg.trueskill.beta,
            "tau": cfg.trueskill.tau,
            "draw_probability": cfg.trueskill.draw_probability,
        }
    )
    concat_tables: list[pa.Table] = []
    screening_cells: list[ScreeningRatingCell] = []

    for seed in seeds:
        seed_everything(seed)
        seed_results_root = seed_results_roots.get(seed, cfg.results_root)
        seed_row_data_dir = seed_row_data_dirs.get(seed) or default_row_data_dir
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
            stats = _load_ratings_parquet(parquet)
            ordered_stats = _sorted_ratings(stats)
            seed_outputs[f"{players}p"] = ordered_stats  # keyed for logging/debug
            concat_tables.append(
                pa.table(
                    {
                        "strategy": list(ordered_stats.keys()),
                        "players": [players] * len(ordered_stats),
                        "mu": [rating.mu for rating in ordered_stats.values()],
                        "sigma": [rating.sigma for rating in ordered_stats.values()],
                        "seed": [seed] * len(ordered_stats),
                    }
                )
            )
            game_rows_path: Path | None = None
            if seed_row_data_dir is not None:
                candidates = (
                    seed_row_data_dir / "by_k" / f"{players}p" / cfg.curated_rows_name,
                    seed_row_data_dir / f"{players}p" / cfg.curated_rows_name,
                )
                game_rows_path = next((path for path in candidates if path.exists()), None)
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

    if concat_tables:
        concat_table = pa.concat_tables(concat_tables, promote_options="default")
        concat_path = concat_dir / "ratings_concat_ks.parquet"
        concat_sidecar = make_artifact_sidecar(
            cfg,
            concat_path,
            producer="trueskill",
            scope=ArtifactScope.CONCAT_KS,
            source_scope=ArtifactScope.BY_K,
            operation="concatenate",
            weighted_quantity="trueskill_mu_and_model_sigma",
            support_count_role="canonical_root_k_rating_cells",
            uncertainty_method="trueskill_model_sigma_screening_only",
            replication_unit="root_k_strategy_rating",
            conditioning="finite_grid_trueskill_screening",
            consistency_columns=concat_table.schema.names,
            source_artifacts=[cell.ratings_path for cell in screening_cells],
            grouping_keys=["seed", "players", "strategy"],
            player_counts=sorted({cell.k for cell in screening_cells}),
            required_player_counts=sorted({cell.k for cell in screening_cells}),
            missing_cell_policy="fail",
            seed_scope="both_roots_combined" if len(seeds) == 2 else "single_root",
        )
        write_parquet_artifact_atomic(
            concat_table,
            concat_path,
            sidecar=concat_sidecar,
            codec=cfg.parquet_codec,
        )

    if not screening_cells:
        raise RuntimeError("TrueSkill produced no canonical root/k rating cells")
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
