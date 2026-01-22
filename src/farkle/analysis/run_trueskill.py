# src/farkle/analysis/run_trueskill.py
"""Compute TrueSkill ratings for Farkle strategies.

The helpers in this module scan a directory of tournament results, update
ratings with the :mod:`trueskill` package, and write both per-block and pooled
rating files.  Historically this module exposed a standalone command line
interface; the new configuration-driven CLI calls :func:`run_trueskill`
directly, so the parsing layer has been removed.

Outputs
-------
``ratings_<N>.parquet`` and ``ratings_pooled.parquet``
    Parquet tables with columns ``{strategy, mu, sigma}`` written under
    ``09_trueskill/<Np>/`` and ``09_trueskill/pooled/`` respectively.
``tiers.json``
    Consolidated tier report containing TrueSkill and frequentist tiers.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
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
    Sequence,
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

from farkle.config import AppConfig
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.random import seed_everything
from farkle.utils.schema_helpers import n_players_from_schema
from farkle.utils.stats import build_tiers
from farkle.utils.tiers import write_tier_payload
from farkle.utils.writer import atomic_path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # hop out of src/farkle
# Default location of tournament result blocks when no path is supplied
DEFAULT_DATAROOT = _REPO_ROOT / "data" / "results"

LOGGER = logging.getLogger(__name__)

DEFAULT_RATING = trueskill.Rating()  # uses env defaults


class RatingArtifactPaths(TypedDict):
    parquet: Path
    json: Path
    ckpt: Path
    checkpoint: Path
    dir: Path
    legacy_parquet: list[Path]
    legacy_json: list[Path]
    legacy_ckpt: list[Path]
    legacy_checkpoint: list[Path]


def _per_player_dir(root: Path, player_count: str) -> Path:
    """Canonical directory for per-player-count artifacts."""

    return root / f"{player_count}p"


def _ensure_new_location(dest: Path, *legacy_paths: Path) -> Path:
    """Return *dest*, migrating a legacy file into place when present."""

    if dest.exists():
        return dest
    for legacy in legacy_paths:
        if legacy.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            legacy.replace(dest)
            break
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


def _iter_rating_parquets(root: Path, suffix: str, legacy_root: Path | None = None) -> list[Path]:
    """Discover per-player rating parquet files with legacy fallback."""

    search_roots = [root]
    if legacy_root is not None and legacy_root != root:
        search_roots.append(legacy_root)

    per_player: list[Path] = []
    for base in search_roots:
        per_player.extend(base.glob(f"*p/ratings_*{suffix}.parquet"))
        per_player.extend((base / "data").glob(f"*p/ratings_*{suffix}.parquet"))
        per_player.extend(base.glob(f"ratings_*{suffix}.parquet"))
    out: list[Path] = []
    seen: set[str] = set()
    for path in per_player:
        if path.stem.startswith("ratings_pooled"):
            continue
        key = path.resolve().as_posix()
        if key in seen:
            continue
        out.append(path)
        seen.add(key)
    return out


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
        base / "analysis" / "02_combine" / "pooled" / "all_ingested_rows.parquet",
        base / "analysis" / "data" / "all_n_players_combined" / "all_ingested_rows.parquet",
        base / "data" / "all_n_players_combined" / "all_ingested_rows.parquet",
        base / "all_ingested_rows.parquet",
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
        return _TSCheckpoint(**json.loads(path.read_text()))
    except Exception:
        return None


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
        return _BlockCkpt(**json.loads(path.read_text()))
    except Exception:
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
    winner_col = (
        batch.column(winner_col_name) if winner_col_name is not None else None
    )
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


def _coerce_ratings(obj: dict[str, object]) -> dict[str, RatingStats]:
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

            strat_cols: list[pa.Array] = [
                batch.column(name) for name in strategy_column_names
            ]

            for r, order in enumerate(ranks_list):
                if not order:
                    continue
                players = [
                    strat_cols[int(str(seat)[1:]) - 1][r].as_py() for seat in order
                ]
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
        fallback_strat_cols: list[pa.Array] = [
            batch.column(name) for name in strategy_column_names
        ]

        ranks = [[col[i].as_py() for col in rank_cols] for i in range(len(batch))]
        strats = [
            [col[i].as_py() for col in fallback_strat_cols] for i in range(len(batch))
        ]

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
    env_kwargs: dict | None = None,
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

    # Up-to-date guard
    paths = _rating_artifact_paths(root, player_count, suffix, legacy_root=legacy_root)
    parquet_path = _ensure_new_location(paths["parquet"], *paths["legacy_parquet"])
    ck_path = _ensure_new_location(paths["ckpt"], *paths["legacy_ckpt"])
    rk_path = _ensure_new_location(paths["checkpoint"], *paths["legacy_checkpoint"])
    json_path = _ensure_new_location(paths["json"], *paths["legacy_json"])

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

    env = trueskill.TrueSkill(**(env_kwargs or {}))

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
    env_kwargs: dict | None = None,
    pooled_weights_by_k: Mapping[int, float] | None = None,
    tiering_z: float | None = None,
    tiering_min_gap: float | None = None,
) -> None:
    """Compute TrueSkill ratings for all result blocks.

    Parameters
    ----------
    output_seed: int, optional
        Value appended to output filenames so repeated runs do not overwrite
        earlier results.
    root: Path | None, optional
        Directory where rating artifacts are written. Defaults to
        ``<dataroot>/analysis/09_trueskill`` when ``None``.
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

    Side Effects
    ------------
    ``ratings_<n>[ _seedX ].parquet``
        Pickle files containing per-block ratings for each strategy.
    ``ratings_pooled[ _seedX ].parquet``
        A pickle with ratings pooled across all blocks using a precision-weighted
        mean (optionally scaled by per-player-count weights).
    ``tiers.json``
        JSON file with league tiers derived from the pooled ratings.
    """
    if dataroot is None:
        base = Path(root) / "results" if root is not None else DEFAULT_DATAROOT
    else:
        base = Path(dataroot)

    root = Path(root) if root is not None else base / "analysis" / "09_trueskill"
    root.mkdir(parents=True, exist_ok=True)
    pooled_dir = root / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)
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
    suffix = f"_seed{output_seed}" if output_seed else ""
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

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

    pooled: dict[str, tuple[float, float]] = {}
    per_block_games: dict[str, int] = {}
    pooled_weights_by_k = dict(pooled_weights_by_k or {})

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
        with cf.ProcessPoolExecutor(max_workers=actual_workers) as ex:
            futures = {
                ex.submit(
                    _rate_block_worker,
                    str(b),
                    str(root),
                    suffix,
                    batch_rows,
                    resume=resume_per_n,
                    checkpoint_every_batches=checkpoint_every_batches,
                    env_kwargs=env_kwargs,
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
                per_block_games[player_count] = block_games
    else:
        for block in blocks:
            player_count, block_games = _rate_block_worker(
                str(block),
                str(root),
                suffix,
                batch_rows,
                resume=resume_per_n,
                checkpoint_every_batches=checkpoint_every_batches,
                env_kwargs=env_kwargs,
                row_data_dir=str(row_data_path) if row_data_path else None,
                curated_rows_name=curated_rows_name,
            )
            per_block_games[player_count] = block_games

    # Combine per-N ratings into pooled stats
    pooled_parquet = _ensure_new_location(
        pooled_dir / f"ratings_pooled{suffix}.parquet",
        root / f"ratings_pooled{suffix}.parquet",
        legacy_root / f"ratings_pooled{suffix}.parquet",
    )
    per_player_parquets = _iter_rating_parquets(root, suffix, legacy_root=legacy_root)
    if pooled_parquet.exists():
        newest = max((p.stat().st_mtime for p in per_player_parquets), default=0.0)
        if pooled_parquet.stat().st_mtime >= newest:
            LOGGER.info(
                "TrueSkill pooled outputs up-to-date",
                extra={"stage": "trueskill", "path": str(pooled_parquet)},
            )
            return

    for parquet in per_player_parquets:
        stem = parquet.stem[len("ratings_") :]
        if suffix:
            stem = stem[: -len(suffix)]
        player_count_value = _player_count_from_stem(parquet.stem)
        if player_count_value is None:
            continue
        games = per_block_games.get(str(player_count_value), 0)
        if games <= 0:
            continue
        weight = pooled_weights_by_k.get(player_count_value, 1.0)
        with parquet.open("rb"):
            ratings = _load_ratings_parquet(parquet)
        for k, v in ratings.items():
            tau = 1.0 / (v.sigma**2) if v.sigma > 0 else 0.0
            s_mu, s_tau = pooled.get(k, (0.0, 0.0))
            s_mu += weight * tau * v.mu
            s_tau += weight * tau
            pooled[k] = (s_mu, s_tau)

    # Precision-weighted pooling → (sum weight*tau*mu, sum weight*tau) → (mu, sigma)
    pooled_stats = {
        k: RatingStats(mu=(s_mu / s_tau), sigma=(s_tau**-0.5))
        for k, (s_mu, s_tau) in pooled.items()
        if s_tau > 0
    }
    # Atomic writes for pooled outputs
    # Save pooled ratings as Parquet
    _save_ratings_parquet(pooled_parquet, pooled_stats)
    pooled_json = {k: {"mu": v.mu, "sigma": v.sigma} for k, v in pooled_stats.items()}
    pooled_json_path = _ensure_new_location(
        pooled_dir / f"ratings_pooled{suffix}.json",
        root / f"ratings_pooled{suffix}.json",
        legacy_root / f"ratings_pooled{suffix}.json",
    )
    with atomic_path(str(pooled_json_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(pooled_json))
    tiers = build_tiers(
        means={k: v.mu for k, v in pooled_stats.items()},
        stdevs={k: v.sigma for k, v in pooled_stats.items()},
        z=float(tiering_z or 1.645),
        min_gap=tiering_min_gap,
    )
    tiers_path = _write_conservative_tiers(
        root, tiers, float(tiering_z or 1.645), tiering_min_gap, legacy_root=legacy_root
    )
    LOGGER.info(
        "TrueSkill run complete",
        extra={
            "stage": "trueskill",
            "root": str(root),
            "blocks": len(blocks),
            "pooled_parquet": str(pooled_parquet),
            "tiers_path": str(tiers_path),
        },
    )


def _sorted_ratings(ratings: Mapping[str, RatingStats]) -> Mapping[str, RatingStats]:
    """Return ratings ordered by strategy for stable materialisation."""

    return dict(sorted(ratings.items(), key=lambda kv: kv[0]))


def _precision_pool(runs: Iterable[Mapping[str, RatingStats]]) -> dict[str, RatingStats]:
    """Combine multiple rating mappings using precision weighting."""

    accum: dict[str, tuple[float, float]] = {}
    for mapping in runs:
        for strategy, stats in mapping.items():
            tau = 1.0 / (stats.sigma**2) if stats.sigma > 0 else 0.0
            if tau <= 0:
                continue
            sum_mu_tau, sum_tau = accum.get(strategy, (0.0, 0.0))
            sum_mu_tau += tau * stats.mu
            sum_tau += tau
            accum[strategy] = (sum_mu_tau, sum_tau)

    pooled: dict[str, RatingStats] = {}
    for strategy, (sum_mu_tau, sum_tau) in accum.items():
        if sum_tau <= 0:
            continue
        pooled[strategy] = RatingStats(mu=sum_mu_tau / sum_tau, sigma=math.sqrt(1.0 / sum_tau))
    return pooled


def _ensure_strict_mu_ordering(
    ratings: Mapping[str, RatingStats],
) -> Mapping[str, RatingStats]:
    """Log pooled TrueSkill ties and return deterministically ordered ratings."""

    sorted_by_mu = sorted(ratings.items(), key=lambda kv: kv[1].mu, reverse=True)
    tie_groups: list[list[str]] = []
    prev_mu: float | None = None
    current_group: list[str] = []

    for name, stats in sorted_by_mu:
        if prev_mu is not None and math.isclose(prev_mu, stats.mu, rel_tol=0.0, abs_tol=1e-12):
            current_group.append(name)
        else:
            if len(current_group) > 1:
                tie_groups.append(sorted(current_group))
            current_group = [name]
            prev_mu = stats.mu

    if len(current_group) > 1:
        tie_groups.append(sorted(current_group))

    if tie_groups:
        LOGGER.warning(
            "Ties detected in pooled TrueSkill means; ordering deterministically",
            extra={
                "stage": "trueskill",
                "tie_groups": tie_groups,
                "tie_count": sum(len(group) for group in tie_groups),
            },
        )

    ordered = sorted(sorted_by_mu, key=lambda kv: (-kv[1].mu, kv[0]))
    return dict(ordered)


def _json_safe_number(value: float | int) -> float | int | None:
    """Convert NaN/inf values to ``None`` for JSON serialization."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _rank_correlations_vs_pooled(
    per_seed_results: Mapping[int, Mapping[str, RatingStats]],
    pooled: Mapping[str, RatingStats],
) -> dict[int, float]:
    """Compute Spearman-style rank correlations against pooled ordering."""

    pooled_order = sorted(pooled.items(), key=lambda kv: (-kv[1].mu, kv[0]))
    pooled_rank = {name: idx for idx, (name, _) in enumerate(pooled_order, start=1)}
    correlations: dict[int, float] = {}

    for seed, stats in per_seed_results.items():
        seed_order = sorted(stats.items(), key=lambda kv: (-kv[1].mu, kv[0]))
        seed_rank = {name: idx for idx, (name, _) in enumerate(seed_order, start=1)}
        common = [name for name in pooled_rank if name in seed_rank]
        if len(common) < 2:
            correlations[seed] = math.nan
            continue

        pooled_vec = np.array([pooled_rank[name] for name in common], dtype=float)
        seed_vec = np.array([seed_rank[name] for name in common], dtype=float)
        denom = pooled_vec.std(ddof=0) * seed_vec.std(ddof=0)
        if denom == 0:
            correlations[seed] = math.nan
            continue
        corr = float(np.corrcoef(pooled_vec, seed_vec)[0, 1])
        correlations[seed] = corr

    return correlations


def _write_seed_alignment_summary(
    dest_dir: Path,
    seeds: Sequence[int],
    per_seed_results: Mapping[int, Mapping[str, RatingStats]],
    pooled: Mapping[str, RatingStats],
    *,
    write_csv: bool = False,
    write_json: bool = False,
) -> dict[str, Path | None]:
    """Align per-seed means by strategy and persist summary statistics."""

    dest_dir.mkdir(parents=True, exist_ok=True)
    seed_list = [int(s) for s in sorted(seeds)]
    strategies = sorted({name for stats in per_seed_results.values() for name in stats})

    base_cols = [
        "strategy",
        "pooled_mu",
        "mean_mu",
        "std_mu",
        "min_mu",
        "max_mu",
        "range_mu",
        "max_abs_delta",
        "seeds_present",
        "seeds_missing",
    ]
    table_columns: dict[str, list[float | int | str]] = {col: [] for col in base_cols}
    for seed in seed_list:
        table_columns[f"seed_{seed}_mu"] = []

    rows: list[dict[str, float | int | str]] = []
    for strategy in strategies:
        seed_mus: list[float] = []
        row: dict[str, float | int | str] = {"strategy": strategy}
        for seed in seed_list:
            mu_val = per_seed_results.get(seed, {}).get(strategy)
            mu = float(mu_val.mu) if mu_val is not None else math.nan
            row[f"seed_{seed}_mu"] = mu
            seed_mus.append(mu)

        pooled_mu = float(pooled.get(strategy, RatingStats(math.nan, math.nan)).mu)
        valid_mus = [m for m in seed_mus if math.isfinite(m)]
        deltas = [m - pooled_mu for m in valid_mus if math.isfinite(pooled_mu)]

        row.update(
            {
                "pooled_mu": pooled_mu,
                "mean_mu": float(np.nanmean(seed_mus)) if seed_mus else math.nan,
                "std_mu": float(np.nanstd(seed_mus)) if seed_mus else math.nan,
                "min_mu": min(valid_mus) if valid_mus else math.nan,
                "max_mu": max(valid_mus) if valid_mus else math.nan,
                "range_mu": (max(valid_mus) - min(valid_mus)) if len(valid_mus) > 1 else math.nan,
                "max_abs_delta": max(abs(delta) for delta in deltas) if deltas else math.nan,
                "seeds_present": len(valid_mus),
                "seeds_missing": len(seed_list) - len(valid_mus),
            }
        )

        for col in table_columns:
            table_columns[col].append(row.get(col, math.nan))
        rows.append(row)

    alignment_table = pa.table(table_columns)
    parquet_path = dest_dir / "seed_mu_alignment.parquet"
    write_parquet_atomic(alignment_table, parquet_path)

    rank_correlations = _rank_correlations_vs_pooled(per_seed_results, pooled)
    csv_path: Path | None = None
    if write_csv:
        csv_path = dest_dir / "seed_mu_alignment.csv"
        fieldnames = base_cols + [f"seed_{seed}_mu" for seed in seed_list]
        with atomic_path(str(csv_path)) as tmp_path, open(tmp_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                {k: ("" if (isinstance(v, float) and math.isnan(v)) else v) for k, v in row.items()}
                for row in rows
            )

    json_path: Path | None = None
    if write_json:
        json_path = dest_dir / "seed_mu_alignment.json"
        json_rows = [
            {k: _json_safe_number(v) if isinstance(v, (float, int)) else v for k, v in row.items()}
            for row in rows
        ]
        json_corr = {int(seed): _json_safe_number(value) for seed, value in rank_correlations.items()}
        payload = {
            "seeds": seed_list,
            "strategies": strategies,
            "alignment": json_rows,
            "rank_correlation_vs_pooled": json_corr,
        }
        with atomic_path(str(json_path)) as tmp_path:
            Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    corr_values = [v for v in rank_correlations.values() if math.isfinite(v)]
    LOGGER.info(
        "TrueSkill seed alignment summary complete",
        extra={
            "stage": "trueskill",
            "strategies": len(strategies),
            "seeds": seed_list,
            "alignment_parquet": str(parquet_path),
            "alignment_csv": str(csv_path) if csv_path else None,
            "alignment_json": str(json_path) if json_path else None,
            "rank_corr_min": min(corr_values) if corr_values else None,
            "rank_corr_max": max(corr_values) if corr_values else None,
        },
    )

    return {
        "parquet": parquet_path,
        "csv": csv_path,
        "json": json_path,
    }


def _write_conservative_tiers(
    root: Path,
    tiers: Mapping[str, int],
    z: float,
    min_gap: float | None,
    *,
    legacy_root: Path | None = None,
) -> Path:
    """Write consolidated TrueSkill tiers to ``tiers.json``."""

    payload = {
        "tiers": dict(tiers),
        "z": float(z),
    }
    if min_gap is not None:
        payload["min_gap"] = float(min_gap)

    tiers_path = _ensure_new_location(root / "tiers.json", (legacy_root or root) / "tiers.json")
    write_tier_payload(tiers_path, trueskill=payload, active="trueskill")
    return tiers_path


def run_trueskill_all_seeds(cfg: AppConfig) -> None:
    """Run TrueSkill for each configured seed and pool the results."""

    analysis_cfg = cfg.analysis
    outputs_cfg = analysis_cfg.outputs or {}
    seeds_cfg = analysis_cfg.tiering_seeds or [cfg.sim.seed]
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
    pooled_dir = analysis_dir / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    legacy_root = analysis_dir.parent

    env_kwargs = {
        "beta": cfg.trueskill.beta,
        "tau": cfg.trueskill.tau,
        "draw_probability": cfg.trueskill.draw_probability,
    }
    tiering_z = float(analysis_cfg.tiering_z_star or 1.645)
    tiering_min_gap = analysis_cfg.tiering_min_gap

    per_seed_results: dict[int, dict[str, RatingStats]] = {}
    per_seed_outputs: dict[int, Mapping[str, Mapping[str, RatingStats]]] = {}

    for seed in seeds:
        seed_everything(seed)
        LOGGER.info(
            "TrueSkill seed run start",
            extra={
                "stage": "trueskill",
                "seed": seed,
                "analysis_dir": str(analysis_dir),
            },
        )
        run_trueskill(
            output_seed=seed,
            root=analysis_dir,
            dataroot=cfg.results_root,
            row_data_dir=cfg.data_dir,
            curated_rows_name=cfg.curated_rows_name,
            workers=analysis_cfg.n_jobs or None,
            env_kwargs=env_kwargs,
            pooled_weights_by_k=cfg.trueskill.pooled_weights_by_k,
            tiering_z=tiering_z,
            tiering_min_gap=tiering_min_gap,
        )

        pooled_path = _ensure_new_location(
            pooled_dir / f"ratings_pooled_seed{seed}.parquet",
            analysis_dir / f"ratings_pooled_seed{seed}.parquet",
            legacy_root / f"ratings_pooled_seed{seed}.parquet",
        )
        if not pooled_path.exists():
            raise FileNotFoundError(f"Missing pooled ratings for seed {seed}: {pooled_path}")
        per_seed_results[seed] = _load_ratings_parquet(pooled_path)

        seed_outputs: dict[str, Mapping[str, RatingStats]] = {}
        for parquet in _iter_rating_parquets(analysis_dir, f"_seed{seed}", legacy_root=legacy_root):
            players = _player_count_from_stem(parquet.stem)
            if players is None:
                continue
            stats = _load_ratings_parquet(parquet)
            ordered_stats = _sorted_ratings(stats)
            dest = analysis_dir / f"trueskill_{players}p_seed{seed}.parquet"
            _save_ratings_parquet(dest, ordered_stats)
            seed_outputs[f"{players}p"] = ordered_stats  # keyed for logging/debug

        if not seed_outputs:
            raise RuntimeError(f"No per-player outputs generated for seed {seed}")
        per_seed_outputs[seed] = seed_outputs

        LOGGER.info(
            "TrueSkill seed run complete",
            extra={
                "stage": "trueskill",
                "seed": seed,
                "pooled": str(pooled_path),
                "per_players": sorted(seed_outputs.keys()),
            },
        )

    pooled_stats = _precision_pool(per_seed_results.values())
    if not pooled_stats:
        raise RuntimeError("TrueSkill pooling produced no results")
    ordered_pooled = _ensure_strict_mu_ordering(_sorted_ratings(pooled_stats))

    pooled_parquet = _ensure_new_location(
        pooled_dir / "ratings_pooled.parquet",
        analysis_dir / "ratings_pooled.parquet",
        legacy_root / "ratings_pooled.parquet",
    )
    _save_ratings_parquet(pooled_parquet, ordered_pooled)

    pooled_json_path = _ensure_new_location(
        pooled_dir / "ratings_pooled.json",
        analysis_dir / "ratings_pooled.json",
        legacy_root / "ratings_pooled.json",
    )
    with atomic_path(str(pooled_json_path)) as tmp_path:
        Path(tmp_path).write_text(
            json.dumps(
                {k: {"mu": v.mu, "sigma": v.sigma} for k, v in ordered_pooled.items()},
                indent=2,
                sort_keys=True,
            )
        )

    _write_seed_alignment_summary(
        pooled_dir,
        seeds,
        per_seed_results,
        ordered_pooled,
        write_csv=bool(outputs_cfg.get("trueskill_alignment_csv", False)),
        write_json=bool(outputs_cfg.get("trueskill_alignment_json", False)),
    )

    tiers = build_tiers(
        means={k: v.mu for k, v in ordered_pooled.items()},
        stdevs={k: v.sigma for k, v in ordered_pooled.items()},
        z=tiering_z,
        min_gap=tiering_min_gap,
    )
    tiers_path = _write_conservative_tiers(
        analysis_dir, tiers, tiering_z, tiering_min_gap, legacy_root=legacy_root
    )

    LOGGER.info(
        "TrueSkill pooled results complete",
        extra={
            "stage": "trueskill",
            "seeds": seeds,
            "pooled_parquet": str(pooled_parquet),
            "tiers_path": str(tiers_path),
        },
    )
