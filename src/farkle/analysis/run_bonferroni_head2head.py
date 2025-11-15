# src/farkle/analysis/run_bonferroni_head2head.py
"""Bonferroni-corrected head-to-head comparison of top-tier strategies."""

from __future__ import annotations

import math
import json
import logging
from itertools import combinations
from pathlib import Path

import pyarrow as pa
import numpy as np
from scipy.stats import binomtest

from typing import Any, Dict

from farkle.simulation.simulation import simulate_many_games_from_seeds
from farkle.simulation.strategies import parse_strategy
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.random import MAX_UINT32
from farkle.utils.stats import games_for_power

DEFAULT_ROOT = Path("results_seed_0")

LOGGER = logging.getLogger(__name__)


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
    pairings = list(combinations(sorted_elites, 2))
    rng = np.random.default_rng(seed)
    pair_seeds = []
    for pair_id, (a, b) in enumerate(pairings):
        seeds = rng.integers(0, MAX_UINT32, size=games_per_pair, dtype=np.uint32).tolist()
        pair_seeds.append((pair_id, a, b, seeds))
    total_games = games_per_pair * len(pairings)
    LOGGER.debug(
        "Head-to-head schedule prepared",
        extra={
            "stage": "head2head",
            "pairs": len(pairings),
            "games": total_games,
            "games_per_pair": games_per_pair,
        },
    )

    records = []
    strategies_cache: Dict[str, Any] = {}
    for pair_id, a, b, seeds in pair_seeds:
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
        wins = df["winner_strategy"].value_counts()
        wa = int(wins.get(a, 0))
        wb = int(wins.get(b, 0))
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
