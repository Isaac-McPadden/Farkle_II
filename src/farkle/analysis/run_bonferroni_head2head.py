# src/farkle/analysis/run_bonferroni_head2head.py
"""Bonferroni-corrected head-to-head comparison of top-tier strategies."""

from __future__ import annotations

import json
import logging
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import binomtest

from farkle.simulation.simulation import simulate_many_games_from_seeds
from farkle.simulation.strategies import parse_strategy
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.random import MAX_UINT32
from farkle.utils.stats import games_for_power

DEFAULT_ROOT = Path("results_seed_0")

LOGGER = logging.getLogger(__name__)


def _load_top_strategies(
    *,
    ratings_path: Path,
    metrics_path: Path,
    ratings_limit: int = 150,
    metrics_limit: int = 150,
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
        parallel.

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
    sub_root = Path(root / "analysis")
    tiers_path = sub_root / "tiers.json"
    pairwise_parquet = sub_root / "bonferroni_pairwise.parquet"

    try:
        with tiers_path.open() as fh:
            tiers = json.load(fh)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Tier file not found at {tiers_path}") from exc
    if not tiers:
        raise RuntimeError(f"No tiers found in {tiers_path}")
    top_val = min(tiers.values())
    elites = [s for s, t in tiers.items() if t == top_val]
    ratings_path = sub_root / "ratings_pooled.parquet"
    metrics_path = sub_root / "metrics.parquet"
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
    records = []
    strategies_cache: Dict[str, Any] = {}
    for pair_id, (a, b) in enumerate(combinations(sorted_elites, 2)):
        seeds = rng.integers(0, MAX_UINT32, size=games_per_pair, dtype=np.uint32).tolist()
        LOGGER.debug(
            "Simulating head-to-head batch",
            extra={
                "stage": "head2head",
                "strategy_a": a,
                "strategy_b": b,
                "games": len(seeds),
            },
        )
        if not seeds:
            continue
        strat_a = strategies_cache.setdefault(a, parse_strategy(a))
        strat_b = strategies_cache.setdefault(b, parse_strategy(b))
        df = simulate_many_games_from_seeds(
            seeds=seeds,
            strategies=[strat_a, strat_b],
            n_jobs=n_jobs,
        )
        wa, wb = _count_pair_wins(df, a, b)
        games_played = len(seeds)
        if wa + wb != games_played:
            raise RuntimeError(
                f"Tie or missing outcome detected for pair ({a}, {b}); wins_a={wa} wins_b={wb} games={games_played}",
            )
        # One-sided: "A beats B"
        pval = binomtest(wa, games_played, alternative="greater").pvalue
        records.append(
            {
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
        )
        LOGGER.debug(
            "Completed head-to-head batch",
            extra={
                "stage": "head2head",
                "strategy_a": a,
                "strategy_b": b,
                "wins_a": wa,
                "wins_b": wb,
                "pvalue": pval,
            },
        )

    pairwise_table = pa.Table.from_pylist(
        sorted(records, key=lambda r: r["pair_id"]),
        schema=pa.schema(
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
        ),
    )
    write_parquet_atomic(pairwise_table, pairwise_parquet)
    LOGGER.info(
        "Bonferroni head-to-head results written",
        extra={
            "stage": "head2head",
            "rows": pairwise_table.num_rows,
            "path": str(pairwise_parquet),
        },
    )
