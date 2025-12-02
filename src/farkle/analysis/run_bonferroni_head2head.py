# src/farkle/analysis/run_bonferroni_head2head.py
"""Bonferroni-corrected head-to-head comparison of top-tier strategies."""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import binomtest

from farkle.simulation.simulation import simulate_many_games_from_seeds
from farkle.simulation.strategies import parse_strategy
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.random import MAX_UINT32
from farkle.utils.stats import games_for_power
from farkle.utils.tiers import load_tier_payload, tier_mapping_from_payload

DEFAULT_ROOT = Path("results_seed_0")

LOGGER = logging.getLogger(__name__)


def _load_top_strategies(
    *,
    ratings_path: Path,
    metrics_path: Path,
    ratings_limit: int = 100,
    metrics_limit: int = 100,
) -> list[str]:
    """Collect fallback strategies from pooled ratings and metrics tables.

    Strategies are sorted by their respective performance indicators and trimmed to
    the provided limits before being combined into a de-duplicated list.
    """

    def _sorted_from_parquet(path: Path, sort_col: str, limit: int, label: str) -> list[str]:
        if not path.exists():
            LOGGER.warning(
                "Fallback selection skipped: missing %s parquet",
                label,
                extra={"stage": "head2head", "path": str(path)},
            )
            return []

        try:
            df = pd.read_parquet(path, columns=["strategy", sort_col])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Fallback selection skipped: failed to read %s parquet",
                label,
                extra={"stage": "head2head", "path": str(path), "error": str(exc)},
            )
            return []

        if df.empty or sort_col not in df:
            LOGGER.warning(
                "Fallback selection skipped: %s parquet missing data",
                label,
                extra={"stage": "head2head", "path": str(path)},
            )
            return []

        return (
            df.sort_values(sort_col, ascending=False)["strategy"].head(limit).dropna().tolist()
        )

    top_by_rating = _sorted_from_parquet(ratings_path, "mu", ratings_limit, "ratings")
    top_by_win_rate = _sorted_from_parquet(metrics_path, "win_rate", metrics_limit, "metrics")

    combined: list[str] = []
    seen: set[str] = set()
    for source in (top_by_rating, top_by_win_rate):
        for strategy in source:
            if strategy not in seen:
                combined.append(strategy)
                seen.add(strategy)

    LOGGER.info(
        "Fallback strategy selection prepared",
        extra={
            "stage": "head2head",
            "ratings_count": len(top_by_rating),
            "metrics_count": len(top_by_win_rate),
            "combined_count": len(combined),
        },
    )
    return combined


def _count_pair_wins(df: pd.DataFrame, strategy_a: str, strategy_b: str) -> tuple[int, int]:
    """Count how many times each strategy won within a simulated batch."""
    if "winner_strategy" in df.columns:
        winners = df["winner_strategy"].astype(str)
        counts = winners.value_counts()
        return int(counts.get(strategy_a, 0)), int(counts.get(strategy_b, 0))

    seat_column = None
    for candidate in ("winner", "winner_seat"):
        if candidate in df.columns:
            seat_column = candidate
            break
    if seat_column is None:
        raise KeyError("winner_strategy column missing and no winner identifier available")

    seat_numbers = (
        df[seat_column]
        .astype(str)
        .str.extract(r"P(?P<num>\d+)", expand=True)["num"]
        .astype("Int64")
    )
    seat_counts = seat_numbers.value_counts()
    wins_a = int(seat_counts.get(1, 0))
    wins_b = int(seat_counts.get(2, 0))
    return wins_a, wins_b


def run_bonferroni_head2head(
    *,
    seed: int = 0,
    root: Path = DEFAULT_ROOT,
    n_jobs: int = 1,
    design: Dict[str, Any] | None = None,
    completed_pair_ids: Iterable[int] | None = None,
    shard_dir: Path | None = None,
    shard_size: int = 25,
    progress_schedule: Sequence[float] | None = None,
) -> None:
    """Run pairwise games between top-tier strategies using Bonferroni tests.

    Parameters
    ----------
    seed : int, default ``0``
        Base seed for shuffling the schedule and deterministically assigning
        unique seeds to each simulated game.
    root : Path, default :data:`DEFAULT_ROOT`
        Directory containing ``tiers.json`` and where results are written.
    n_jobs : int, default ``1``
        Number of worker processes; when greater than one, games are simulated in
        parallel and queued in batches so multiple pairs advance concurrently.

    The function reads ``data/tiers.json`` to find strategies in the highest
    tier. It derives a per-pair game budget from :func:`~farkle.utils.stats.games_for_power`
    using the same one-sided tail as the exact test and writes
    ``analysis/bonferroni_pairwise.parquet``. The output schema is::

        players: int64
        seed: int64
        pair_id: int64
        a: string
        b: string
        games: int64
        wins_a: int64
        wins_b: int64
        win_rate_a: float64
        pval_one_sided: float64

    Each row summarises a pairwise matchup with deterministic seeding; the
    p-value is produced by :func:`scipy.stats.binomtest` using the ``greater``
    alternative.
    """
    LOGGER.info(
        "Bonferroni head-to-head start",
        extra={"stage": "head2head", "root": str(root), "seed": seed, "n_jobs": n_jobs},
    )
    analysis_root = Path(root / "analysis")
    sub_root = analysis_root / "04_head2head"
    sub_root.mkdir(parents=True, exist_ok=True)
    tiers_candidates = [
        analysis_root / "05_tiering" / "tiers.json",
        analysis_root / "03_trueskill" / "tiers.json",
        analysis_root / "tiers.json",
    ]
    tiers_path = next((p for p in tiers_candidates if p.exists()), tiers_candidates[0])
    pairwise_candidates = [sub_root / "bonferroni_pairwise.parquet", analysis_root / "bonferroni_pairwise.parquet"]
    pairwise_parquet = next((p for p in pairwise_candidates if p.exists()), pairwise_candidates[0])
    default_shards = sub_root / "bonferroni_pairwise_shards"
    legacy_shards = analysis_root / "bonferroni_pairwise_shards"
    shard_dir = shard_dir or (default_shards if default_shards.exists() or not legacy_shards.exists() else legacy_shards)

    if not tiers_path.exists():
        raise RuntimeError(f"Tier file not found at {tiers_path}")

    payload = load_tier_payload(tiers_path)
    prefer_source = payload.get("active") if isinstance(payload, dict) else None
    tiers = tier_mapping_from_payload(payload, prefer=str(prefer_source) if prefer_source else "")
    if not tiers:
        raise RuntimeError(f"No tiers found in {tiers_path}")
    top_val = min(tiers.values())
    elites = [s for s, t in tiers.items() if t == top_val]
    ratings_candidates = [
        analysis_root / "03_trueskill" / "ratings_pooled.parquet",
        analysis_root / "ratings_pooled.parquet",
    ]
    ratings_path = next((p for p in ratings_candidates if p.exists()), ratings_candidates[0])
    metrics_candidates = [analysis_root / "02_metrics" / "metrics.parquet", analysis_root / "metrics.parquet"]
    metrics_path = next((p for p in metrics_candidates if p.exists()), metrics_candidates[0])
    if len(elites) < 2:
        fallback = _load_top_strategies(
            ratings_path=ratings_path,
            metrics_path=metrics_path,
        )
        if fallback:
            combined = []
            seen = set()
            for name in (*elites, *fallback):
                if name not in seen:
                    combined.append(name)
                    seen.add(name)
            elites = combined
            LOGGER.info(
                "Elite tier too small; using fallback strategy union",
                extra={
                    "stage": "head2head",
                    "elite_count": len(elites),
                    "tiers_path": str(tiers_path),
                    "ratings_path": str(ratings_path),
                    "metrics_path": str(metrics_path),
                },
            )
    LOGGER.info(
        "Loaded elite strategies",
        extra={
            "stage": "head2head",
            "path": str(tiers_path),
            "strategies": len(tiers),
            "elite_count": len(elites),
        },
    )
    design_kwargs = dict(design or {})
    method = design_kwargs.pop("method", "bonferroni")
    full_pairwise = bool(design_kwargs.pop("full_pairwise", True))
    design_kwargs.setdefault("endpoint", "pairwise")
    design_kwargs.setdefault("k_players", 2)
    if method != "bonferroni":
        raise ValueError("Bonferroni head-to-head requires method='bonferroni'")
    if not full_pairwise:
        raise ValueError("Bonferroni head-to-head requires full_pairwise comparisons")
    tail = str(design_kwargs.get("tail", "one_sided")).lower().replace("-", "_")
    if tail != "one_sided":
        raise ValueError(
            "Bonferroni head-to-head uses a one-sided exact test; set tail='one_sided'",
        )
    design_kwargs["tail"] = "one_sided"
    k_players = int(design_kwargs.get("k_players", 2))
    if k_players != 2:
        raise ValueError("Bonferroni head-to-head only supports k_players=2")

    games_per_strategy = games_for_power(
        n_strategies=len(elites),
        method=method,
        full_pairwise=True,
        **design_kwargs,
    )
    opponents = max(0, len(elites) - 1)
    games_per_pair = (
        0
        if opponents == 0
        else int(math.ceil(games_per_strategy * max(1, k_players - 1) / opponents))
    )
    LOGGER.info(
        "Bonferroni head-to-head sizing",
        extra={
            "stage": "head2head",
            "games_per_strategy": games_per_strategy,
            "games_per_pair": games_per_pair,
            "elite_count": len(elites),
            "k_players": k_players,
        },
    )

    if games_per_pair <= 0:
        LOGGER.info(
            "Bonferroni head-to-head: no games scheduled",
            extra={"stage": "head2head", "elite_count": len(elites)},
        )
        return

    sorted_elites = sorted(elites)
    pair_count = math.comb(len(sorted_elites), 2)
    total_games = games_per_pair * pair_count
    LOGGER.debug(
        "Head-to-head schedule prepared",
        extra={
            "stage": "head2head",
            "pairs": pair_count,
            "games": total_games,
            "games_per_pair": games_per_pair,
        },
    )

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    pending_shard_records: list[dict[str, Any]] = []
    strategies_cache: Dict[str, Any] = {}
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_schema = pa.schema(
        [
            pa.field("players", pa.int64()),
            pa.field("seed", pa.int64()),
            pa.field("pair_id", pa.int64()),
            pa.field("a", pa.string()),
            pa.field("b", pa.string()),
            pa.field("games", pa.int64()),
            pa.field("wins_a", pa.int64()),
            pa.field("wins_b", pa.int64()),
            pa.field("win_rate_a", pa.float64()),
            pa.field("pval_one_sided", pa.float64()),
        ]
    )

    def _read_pair_ids_from_parquet(path: Path) -> set[int]:
        if not path.exists():
            return set()
        try:
            df = pd.read_parquet(path, columns=["pair_id"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Failed to load completed pair ids",
                extra={"stage": "head2head", "path": str(path), "error": str(exc)},
            )
            return set()
        if "pair_id" not in df:
            LOGGER.warning(
                "Existing parquet missing pair_id column",
                extra={"stage": "head2head", "path": str(path)},
            )
            return set()
        return {int(pid) for pid in df["pair_id"].dropna().astype(int).tolist()}

    shard_paths = sorted(shard_dir.glob("bonferroni_pairwise_shard_*.parquet"))
    shard_count = len(shard_paths)
    existing_shard_frames: list[pd.DataFrame] = []
    completed_pairs: set[int] = {int(pid) for pid in (completed_pair_ids or ())}
    for shard_path in shard_paths:
        completed_pairs.update(_read_pair_ids_from_parquet(shard_path))
        try:
            df = pd.read_parquet(shard_path)
            existing_shard_frames.append(df)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Failed to load shard data",
                extra={"stage": "head2head", "path": str(shard_path), "error": str(exc)},
            )
    existing_final_pairs = _read_pair_ids_from_parquet(pairwise_parquet)
    completed_pairs.update(existing_final_pairs)

    LOGGER.info(
        "Resuming head-to-head schedule",
        extra={
            "stage": "head2head",
            "precompleted_pairs": len(completed_pairs),
            "shards": shard_count,
        },
    )

    def flush_shard(records_to_flush: List[dict[str, Any]], shard_index: int) -> int:
        if not records_to_flush:
            return shard_index
        shard_path = shard_dir / f"bonferroni_pairwise_shard_{shard_index:04d}.parquet"
        shard_table = pa.Table.from_pylist(records_to_flush, schema=shard_schema)
        write_parquet_atomic(shard_table, shard_path)
        LOGGER.info(
            "Shard written",
            extra={
                "stage": "head2head",
                "rows": shard_table.num_rows,
                "path": str(shard_path),
            },
        )
        existing_shard_frames.append(shard_table.to_pandas())
        records_to_flush.clear()
        return shard_index + 1

    fast_interval = 30.0
    fast_phase_seconds = 600.0
    slow_interval = 600.0
    schedule: Sequence[float] = progress_schedule or [fast_interval, fast_phase_seconds, slow_interval]
    if len(schedule) != 3:
        raise ValueError(
            "progress_schedule must have three values: fast interval, fast phase seconds, slow interval",
        )
    fast_interval, fast_phase_seconds, slow_interval = (float(x) for x in schedule)
    start_time = time.monotonic()
    next_progress = start_time + fast_interval
    fast_phase_end = start_time + fast_phase_seconds

    def maybe_log_progress(completed: int) -> None:
        nonlocal next_progress
        now = time.monotonic()
        if now < next_progress:
            return
        LOGGER.info(
            "Head-to-head progress",
            extra={
                "stage": "head2head",
                "pairs_completed": completed,
                "pairs_total": pair_count,
                "elapsed_seconds": round(now - start_time, 1),
            },
        )
        if now < fast_phase_end:
            next_progress += fast_interval
        else:
            next_progress = max(next_progress, fast_phase_end) + slow_interval

    scheduled_pairs: list[tuple[int, str, str, list[int]]] = []
    processed_pairs = len(completed_pairs)
    for pair_id, (a, b) in enumerate(combinations(sorted_elites, 2)):
        if pair_id in completed_pairs:
            LOGGER.debug(
                "Skipping completed pair",
                extra={"stage": "head2head", "pair_id": pair_id, "strategy_a": a, "strategy_b": b},
            )
            maybe_log_progress(processed_pairs)
            continue
        seeds = rng.integers(0, MAX_UINT32, size=games_per_pair, dtype=np.uint32).tolist()
        LOGGER.debug(
            "Queueing head-to-head batch",
            extra={
                "stage": "head2head",
                "strategy_a": a,
                "strategy_b": b,
                "games": len(seeds),
            },
        )
        if not seeds:
            continue
        scheduled_pairs.append((pair_id, a, b, seeds))

    if not scheduled_pairs:
        shard_count = flush_shard(pending_shard_records, shard_count)
    else:
        strategy_names = {name for _, a, b, _ in scheduled_pairs for name in (a, b)}
        strategies_cache = {name: parse_strategy(name) for name in strategy_names}

        effective_jobs = n_jobs if n_jobs > 0 else 1
        if effective_jobs <= 1:
            worker_count = 1
            pair_jobs = 1
        else:
            worker_count = min(len(scheduled_pairs), max(1, effective_jobs // 2))
            pair_jobs = max(1, effective_jobs // worker_count)

        def simulate_pair(job: tuple[int, str, str, list[int]]) -> dict[str, Any]:
            pair_id, a, b, seeds = job
            df = simulate_many_games_from_seeds(
                seeds=seeds,
                strategies=[strategies_cache[a], strategies_cache[b]],
                n_jobs=pair_jobs,
            )
            wa, wb = _count_pair_wins(df, a, b)
            games_played = len(seeds)
            if wa + wb != games_played:
                raise RuntimeError(
                    f"Tie or missing outcome detected for pair ({a}, {b}); wins_a={wa} wins_b={wb} games={games_played}",
                )
            pval = binomtest(wa, games_played, alternative="greater").pvalue
            return {
                "players": k_players,
                "seed": seed,
                "pair_id": pair_id,
                "a": a,
                "b": b,
                "games": games_played,
                "wins_a": wa,
                "wins_b": wb,
                "win_rate_a": wa / games_played if games_played else math.nan,
                "pval_one_sided": pval,
            }

        LOGGER.info(
            "Dispatching head-to-head batches",
            extra={
                "stage": "head2head",
                "pending_pairs": len(scheduled_pairs),
                "workers": worker_count,
                "pair_jobs": pair_jobs,
            },
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_job = {
                executor.submit(simulate_pair, job): job for job in scheduled_pairs
            }
            for future in as_completed(future_to_job):
                job_pair_id, job_a, job_b, _ = future_to_job[future]
                record = future.result()
                records.append(record)
                pending_shard_records.append(record)
                if len(pending_shard_records) >= shard_size:
                    shard_count = flush_shard(pending_shard_records, shard_count)
                completed_pairs.add(job_pair_id)
                processed_pairs += 1
                maybe_log_progress(processed_pairs)
                LOGGER.debug(
                    "Completed head-to-head batch",
                    extra={
                        "stage": "head2head",
                        "strategy_a": job_a,
                        "strategy_b": job_b,
                        "wins_a": record["wins_a"],
                        "wins_b": record["wins_b"],
                        "pvalue": record["pval_one_sided"],
                    },
                )

    shard_count = flush_shard(pending_shard_records, shard_count)
    all_frames: list[pd.DataFrame] = []
    if records:
        all_frames.append(pd.DataFrame.from_records(records))
    all_frames.extend(existing_shard_frames)
    if pairwise_parquet.exists():
        try:
            all_frames.append(pd.read_parquet(pairwise_parquet))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Failed to load existing pairwise parquet",
                extra={"stage": "head2head", "path": str(pairwise_parquet), "error": str(exc)},
            )
    if not all_frames:
        LOGGER.info(
            "Bonferroni head-to-head: no results to write",
            extra={"stage": "head2head", "elite_count": len(elites)},
        )
        return
    combined_df = (
        pd.concat(all_frames, ignore_index=True)
        .sort_values("pair_id")
        .drop_duplicates(subset=["pair_id"], keep="last")
    )
    pairwise_table = pa.Table.from_pandas(combined_df, schema=shard_schema, preserve_index=False)
    write_parquet_atomic(pairwise_table, pairwise_parquet)
    LOGGER.info(
        "Bonferroni head-to-head results written",
        extra={
            "stage": "head2head",
            "rows": pairwise_table.num_rows,
            "path": str(pairwise_parquet),
        },
    )
