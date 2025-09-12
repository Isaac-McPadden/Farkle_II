# src/farkle/run_trueskill.py
"""Compute TrueSkill ratings for Farkle strategies.

The script scans a directory of tournament results, updates ratings with the
``trueskill`` package and writes per-block as well as pooled rating files.

Key CLI flags
-------------
--dataroot   Directory containing <N>_players blocks            (default: data/results)
--root       Output directory for analysis artefacts            (default: <dataroot>/analysis)

Outputs
-------
ratings_<N>.parquet(*), ratings_pooled.parquet - Parquet tables with columns {strategy, mu, sigma}
tiers.json - mapping of strategy → tier
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional, Tuple, Union, cast

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import trueskill
import yaml
from trueskill import Rating

from farkle.analysis.analysis_config import PipelineCfg, n_players_from_schema
from farkle.utils.utils import build_tiers

_REPO_ROOT = Path(__file__).resolve().parents[2]  # hop out of src/farkle
# Default location of tournament result blocks when no path is supplied
DEFAULT_DATAROOT = _REPO_ROOT / "data" / "results"

log = logging.getLogger(__name__)

DEFAULT_RATING = trueskill.Rating()  # uses env defaults
_DEFAULT_WORKERS = 1


def _find_aggregate_parquet(base: Path | None) -> Path | None:
    """Return path to aggregated ``all_ingested_rows.parquet`` if it exists.

    The *base* argument may point to a results directory or directly to the
    ``analysis/data`` directory. This helper tries a few common locations and
    returns the first existing path. ``None`` is returned when no candidate is
    found.
    """

    if base is None:
        return None
    base = Path(base)
    candidates = [
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
    source: str                 # parquet path
    row_group: int              # next row-group to start at
    batch_index: int            # next batch index within that row-group
    games_done: int
    ratings_path: str           # where the interim ratings parquet lives
    version: int = 1


def _save_ckpt(path: Path, ck: _TSCheckpoint) -> None:
    path.write_text(json.dumps(asdict(ck)))


def _load_ckpt(path: Path) -> Optional[_TSCheckpoint]:
    if not path.exists():
        return None
    try:
        return _TSCheckpoint(**json.loads(path.read_text()))
    except Exception:
        return None


def _ratings_to_table(
    mapping: Mapping[str, Union[trueskill.Rating, "RatingStats", Tuple[float, float], float]]
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
        strategies.append(k)
        mus.append(mu)
        sigmas.append(sigma)
    return pa.table(
        {
            "strategy": pa.array(strategies, type=pa.string()),
            "mu": pa.array(mus, type=pa.float64()),
            "sigma": pa.array(sigmas, type=pa.float64()),
        }
    )


def _write_parquet_atomic(table: pa.Table, path: Path, *, compression: str = "zstd") -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression=compression)
    tmp.replace(path)


def _save_ratings_parquet(
    path: Path,
    ratings: Mapping[str, Union[trueskill.Rating, "RatingStats", Tuple[float, float]]],
) -> None:
    _write_parquet_atomic(_ratings_to_table(ratings), path)


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
    row_file: str
    row_group: int
    batch_index: int
    games_done: int
    ratings_path: str
    version: int = 1


def _save_block_ckpt(path: Path, ck: _BlockCkpt) -> None:
    path.write_text(json.dumps(asdict(ck)))


def _load_block_ckpt(path: Path) -> Optional[_BlockCkpt]:
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


def _players_and_ranks_from_batch(
    batch: pa.Table, n: int
) -> Iterator[tuple[list[str], list[int]]]:
    """Robust per-row extraction (tie-friendly precedence):
       1) Derive placements from P#_rank (preserves ties)
       2) Else, use seat_ranks (strict order, no ties)
       3) Else, fallback: winner vs (n-1) tied second.
    """
    cols = set(batch.column_names)

    def col(name: str):
        return batch[name] if name in cols else None

    sr_col = col("seat_ranks")
    ranks_list = sr_col.to_pylist() if sr_col is not None else None
    w_col = col("winner_seat") or col("winner")
    winner_col = w_col.to_pylist() if w_col is not None else None
    strat_cols = [col(f"P{i}_strategy") for i in range(1, n + 1)]
    rank_cols = [col(f"P{i}_rank") for i in range(1, n + 1)]
    n_rows = batch.num_rows
    for r in range(n_rows):
        seats = [sc[r].as_py() if sc is not None else None for sc in strat_cols]
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
        if ranks_list is not None and ranks_list[r]:
            order = ranks_list[r]
            players = [seats[int(s[1:]) - 1] for s in order]
            if any(p is None for p in players):  # skip incomplete rows
                continue
            yield cast(list[str], players), list(range(len(players)))
            continue
        # 3) fallback
        if winner_col is None or not winner_col[r]:
            continue
        try:
            w_idx = int(str(winner_col[r])[1:]) - 1
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
    """Stream all games from a single aggregated parquet, with checkpoint/resume."""
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


def _ensure_seed_ratings(ratings: dict[str, Rating], all_strategies: list[str], env: trueskill.TrueSkill) -> None:
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


def _iter_players_and_ranks(row_file: Path, n: int, batch_size: int = 100_000) -> Iterator[tuple[list[str], list[int]]]:
    """Yield (players, ranks) per game in a streaming fashion.
    Prefers explicit placements via ``seat_ranks``; if absent, derives placements
    from ``P#_rank``; if only a winner is known, treats others as a single tied group.
    """
    dataset = ds.dataset(row_file, format="parquet")
    schema = dataset.schema
    has_seat_ranks = schema.get_field_index("seat_ranks") != -1

    if has_seat_ranks:
        cols = ["seat_ranks"] + [f"P{i}_strategy" for i in range(1, n + 1)]
        for batch in dataset.to_batches(columns=cols, batch_size=batch_size):
            ranks_col = batch["seat_ranks"].to_pylist()  # list[list[str]] or None
            strat_cols = [batch[f"P{i}_strategy"] for i in range(1, n + 1)]
            for r, order in enumerate(ranks_col):
                if not order:
                    continue
                # strict placement: teams already ordered; ranks are 0..K-1
                players = [strat_cols[int(seat[1:]) - 1][r].as_py() for seat in order]
                if any(p is None for p in players):
                    continue
                yield players, list(range(len(players)))
        return

    # seat_ranks not present → derive from P#_rank or fall back to winner + tied losers
    rank_cols = [f"P{i}_rank" for i in range(1, n + 1)]
    strat_cols = [f"P{i}_strategy" for i in range(1, n + 1)]
    winner_col_name = "winner_seat" if schema.get_field_index("winner_seat") != -1 else "winner"
    cols = [winner_col_name] + rank_cols + strat_cols
    for batch in dataset.to_batches(columns=cols, batch_size=batch_size):
        winner_seats = batch[winner_col_name].to_pylist()
        ranks = [[batch[c][i].as_py() for c in rank_cols] for i in range(len(batch))]
        strats = [[batch[c][i].as_py() for c in strat_cols] for i in range(len(batch))]

        for winner_seat, seat_ranks, seat_strats in zip(winner_seats, ranks, strats, strict=True):
            # try to build placements from P#_rank if we have ≥2 ranks
            pairs = [(i, rv) for i, rv in enumerate(seat_ranks) if rv is not None]
            if len(pairs) >= 2:
                # sort by rank ascending; remap rank values to 0..K-1 to keep them dense
                pairs.sort(key=lambda x: x[1])
                uniq = sorted({rv for _, rv in pairs})
                remap = {rv: j for j, rv in enumerate(uniq)}
                players = []
                pranks  = []
                for i, rv in pairs:
                    p = seat_strats[i]
                    if p is None:
                        continue
                    players.append(p)
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
            losers = [s for j, s in enumerate(seat_strats) if j != w_idx and s is not None]
            if losers:
                yield [winner] + losers, [0] + [1] * len(losers)


def _rate_stream(row_file: Path, n: int, keepers: list[str], env: trueskill.TrueSkill,
                 batch_size: int = 100_000) -> tuple[dict[str, RatingStats], int]:
    """Stream TrueSkill updates from parquet without materialising all games.
    Supports ties and multiple sources of placement (seat_ranks or P#_rank)."""
    ratings: dict[str, trueskill.Rating] = {k: env.create_rating() for k in keepers}
    n_games = 0
    for players, ranks in _iter_players_and_ranks(row_file, n, batch_size):
        # Optional keepers filter
        if keepers:
            filt = [(p, r) for p, r in zip(players, ranks, strict=True) if p in keepers]
            if len(filt) < 2:
                continue
            players = [p for p, _ in filt]
            ranks   = [r for _, r in filt]
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
) -> tuple[str, int]:
    """
    Process one <N>_players block with optional checkpointing.
    Returns (player_count_str, n_games).
    """
    block = Path(block_dir)
    root = Path(root_dir)

    player_count = block.name.split("_")[0]
    keep_path = block / f"keepers_{player_count}.npy"
    keepers = np.load(keep_path).tolist() if keep_path.exists() else []

    # Locate curated parquet
    row_file = Path(root) / "data" / f"{player_count}p" / f"{player_count}p_ingested_rows.parquet"
    if not row_file.exists():
        candidate = next(block.glob("*p_rows.parquet"), None)
        if candidate is None:
            return player_count, 0
        row_file = candidate
        n = n_players_from_schema(pq.read_schema(row_file))
    else:
        n = int(player_count)

    # Up-to-date guard
    parquet_path = root / f"ratings_{player_count}{suffix}.parquet"
    if parquet_path.exists() and parquet_path.stat().st_mtime >= row_file.stat().st_mtime:
        try:
            md = pq.read_metadata(row_file)
            n_rows = md.num_rows
        except Exception:
            n_rows = 0
        log.info("TrueSkill: %sp up-to-date - skipped", player_count)
        return player_count, n_rows

    env = trueskill.TrueSkill(**(env_kwargs or {}))
    ck_path = root / f"ratings_{player_count}{suffix}.ckpt.json"
    rk_path = root / f"ratings_{player_count}{suffix}.checkpoint.parquet"

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

    json_path = root / f"ratings_{player_count}{suffix}.json"
    tmp_json = json_path.with_name(json_path.name + ".tmp")
    tmp_json.write_text(json.dumps({k: {"mu": v.mu, "sigma": v.sigma} for k, v in ratings_stats.items()}))
    tmp_json.replace(json_path)

    return player_count, n_games

def run_trueskill(
    output_seed: int = 0,
    root: Path | None = None,
    dataroot: Path | None = None,
    workers: int | None = None,
    batch_rows: int = 100_000,
    resume_per_n: bool = True,
    checkpoint_every_batches: int = 500,
    env_kwargs: dict | None = None,
) -> None:
    """Compute TrueSkill ratings for all result blocks.

    Parameters
    ----------
    output_seed: int, optional
        Value appended to output filenames so repeated runs do not overwrite
        earlier results.
    root: Path | None, optional
        Directory where rating artifacts are written. Defaults to
        ``<dataroot>/analysis`` when ``None``.
    dataroot: Path | None, optional
        Directory containing tournament result blocks. When ``None`` the
        path defaults to ``<root>/results`` if ``root`` is given, otherwise
        to :data:`DEFAULT_DATAROOT`.

    Side Effects
    ------------
    ``ratings_<n>[ _seedX ].parquet``
        Pickle files containing per-block ratings for each strategy.
    ``ratings_pooled[ _seedX ].parquet``
        A pickle with ratings pooled across all blocks using a games-per-block
        precision-weighted mean.
    ``tiers.json``
        JSON file with league tiers derived from the pooled ratings.
    """
    if dataroot is None:
        base = Path(root) / "results" if root is not None else DEFAULT_DATAROOT
    else:
        base = Path(dataroot)

    root = Path(root) if root is not None else base / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    _read_manifest_seed(base / "manifest.yaml")
    suffix = f"_seed{output_seed}" if output_seed else ""
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    pooled: dict[str, tuple[float, float]] = {}
    per_block_games: dict[str, int] = {}

    blocks = sorted(base.glob("*_players"))
    # Pick a sensible default, then cap to CPUs and number of blocks
    auto_default = max(1, (os.cpu_count() or 1) - 1)
    requested = workers or auto_default
    cpu_cap   = (os.cpu_count() or 1)
    actual_workers = max(1, min(requested, cpu_cap, len(blocks)))
    if actual_workers != requested:
        log.info("Workers capped from %d → %d (cpus=%d, blocks=%d)",
                 requested, actual_workers, cpu_cap, len(blocks))
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
                ): b
                for b in blocks
            }
            for fut in cf.as_completed(futures):
                try:
                    player_count, block_games = fut.result()
                except Exception as e:
                    bad = futures[fut]
                    log.exception("TrueSkill failed for block %s: %s", bad, e)
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
            )
            per_block_games[player_count] = block_games
                    
    # Combine per-N ratings into pooled stats
    pooled_parquet = root / f"ratings_pooled{suffix}.parquet"
    existing_parquets = sorted(root.glob(f"ratings_*{suffix}.parquet"))
    if pooled_parquet.exists():
        newest = max((p.stat().st_mtime for p in existing_parquets), default=0.0)
        if pooled_parquet.stat().st_mtime >= newest:
            log.info("TrueSkill: pooled outputs up-to-date - skipped")
            return

    for parquet in existing_parquets:
        if parquet == pooled_parquet:
            continue
        stem = parquet.stem[len("ratings_"):]
        if suffix:
            stem = stem[: -len(suffix)]
        games = per_block_games.get(stem, 0)
        if games <= 0:
            continue
        with parquet.open("rb") as fh:
            ratings = _load_ratings_parquet(parquet)
        for k, v in ratings.items():
            tau = 1.0 / (v.sigma ** 2) if v.sigma > 0 else 0.0
            s_mu, s_tau = pooled.get(k, (0.0, 0.0))
            s_mu += tau * v.mu * games
            s_tau += tau * games
            pooled[k] = (s_mu, s_tau)

    # Precision-weighted pooling → (sum tau*mu, sum tau) → (mu, sigma)
    pooled_stats = {k: RatingStats(mu=(s_mu / s_tau), sigma=(s_tau ** -0.5))
                    for k, (s_mu, s_tau) in pooled.items() if s_tau > 0}
    # Atomic writes for pooled outputs
    # Save pooled ratings as Parquet
    _save_ratings_parquet(pooled_parquet, pooled_stats)
    pooled_json = {k: {"mu": v.mu, "sigma": v.sigma} for k, v in pooled_stats.items()}
    pooled_json_path = root / f"ratings_pooled{suffix}.json"
    tmp_pool_json = pooled_json_path.with_name(pooled_json_path.name + ".tmp")
    tmp_pool_json.write_text(json.dumps(pooled_json))
    tmp_pool_json.replace(pooled_json_path)
    tiers = build_tiers(
        means={k: v.mu for k, v in pooled_stats.items()},
        stdevs={k: v.sigma for k, v in pooled_stats.items()},
    )
    with (root / "tiers.json").open("w") as fh:
        json.dump(tiers, fh, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``run_trueskill`` command line interface.

    Parameters
    ----------
    argv : list[str] | None, optional
        Arguments to parse instead of ``sys.argv``. Each argument should be a
        separate element in the list.

    Side Effects
    ------------
    Invokes :func:`run_trueskill`, which writes rating and tier files to
    ``--root`` (default ``<dataroot>/analysis``).
    """
    parser = argparse.ArgumentParser(description="Compute TrueSkill ratings")
    parser.add_argument(
        "--output-seed", type=int, default=0,
        help="appended to filenames to avoid overwrites",
    )
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=None,
        help=(
            "Folder that holds <N>_players blocks "
            "(default: <root>/results or <repo>/data/results). "
            "Accepts absolute or relative paths."
        ),
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="output directory (default: <dataroot>/analysis)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="process <N>_players blocks in parallel (default: cpu_count-1)",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=None,
        help="Arrow batch size for streaming readers (default from PipelineCfg)",
    )
    parser.add_argument(
        "--single-pass-from",
        type=Path,
        default=None,
        help="Path to aggregated all_ingested_rows.parquet (single-pass mode).",
    )
    parser.add_argument(
        "--no-single-pass",
        action="store_true",
        help="Force legacy per-N mode even if aggregate is present.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if present (default: on).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore checkpoint and start fresh.",
    )
    parser.add_argument(
        "--no-resume-per-n",
        dest="resume_per_n",
        action="store_false",
        default=True,
        help="Disable per-N resume (default: resume).",
    )
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=500,
        help="Checkpoint every N batches (default 500).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Where to save the checkpoint JSON (default: <root>/trueskill.checkpoint.json).",
    )
    parser.add_argument(
        "--ratings-checkpoint-path",
        type=Path,
        default=None,
        help="Where to save interim ratings parquet (default: <root>/ratings_checkpoint.parquet).",
    )
    # Pull batch/worker policy from PipelineCfg (falls back to CLI if not set).
    cfg, _, _ = PipelineCfg.parse_cli([])
    cfg_workers = cfg.trueskill_workers if cfg.trueskill_workers is not None else _DEFAULT_WORKERS
    cfg_batch_rows = cfg.batch_rows

    args = parser.parse_args(argv or [])
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg_workers = args.workers or cfg_workers
    cfg_batch_rows = args.batch_rows or cfg_batch_rows

    # Decide single-pass source
    if not args.no_single_pass:
        agg = args.single_pass_from or _find_aggregate_parquet(args.dataroot or cfg.data_dir)
    else:
        agg = None

    # Prepare TrueSkill env kwargs from config (e.g., beta)
    env_kwargs = {"beta": cfg.trueskill_beta}

    # Default checkpoint locations
    if args.root:
        root = Path(args.root)
    elif args.dataroot:
        root = Path(args.dataroot) / "analysis"
    else:
        root = DEFAULT_DATAROOT / "analysis"
    ck_path = args.checkpoint_path or (root / "trueskill.checkpoint.json")
    rk_path = args.ratings_checkpoint_path or (root / "ratings_checkpoint.parquet")

    if agg and agg.exists():
        log.info("TrueSkill: single-pass over %s (resume=%s)", agg, args.resume)
        root.mkdir(parents=True, exist_ok=True)
        env = trueskill.TrueSkill(**env_kwargs)
        pooled, games = _rate_single_pass(
            agg,
            env=env,
            resume=args.resume,
            checkpoint_path=ck_path,
            ratings_ckpt_path=rk_path,
            batch_rows=cfg_batch_rows,
            checkpoint_every_batches=args.checkpoint_every_batches,
        )
        suffix = f"_seed{args.output_seed}" if args.output_seed else ""
        (root / f"ratings_pooled{suffix}.json").write_text(
            json.dumps({k: {"mu": v.mu, "sigma": v.sigma} for k, v in pooled.items()})
        )
        _save_ratings_parquet(root / f"ratings_pooled{suffix}.parquet", pooled)
        tiers = build_tiers(
            means={k: v.mu for k, v in pooled.items()},
            stdevs={k: v.sigma for k, v in pooled.items()},
        )
        with (root / "tiers.json").open("w") as fh:
            json.dump(tiers, fh, indent=2, sort_keys=True)
        log.info("TrueSkill single-pass complete: %d games, %d strategies", games, len(pooled))
        return

    log.info("TrueSkill: aggregate not found or disabled; using per-N legacy flow.")
    run_trueskill(
        output_seed=args.output_seed,
        root=args.root,
        dataroot=args.dataroot,
        workers=cfg_workers,
        batch_rows=cfg_batch_rows,
        resume_per_n=args.resume_per_n,
        checkpoint_every_batches=args.checkpoint_every_batches,
        env_kwargs=env_kwargs,
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
