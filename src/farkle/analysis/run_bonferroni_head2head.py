# src/farkle/run_bonferroni_head2head
"""Bonferroni-corrected head-to-head comparison of top-tier strategies."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
from scipy.stats import binomtest

from farkle.simulation.simulation import simulate_many_games_from_seeds
from farkle.simulation.strategies import parse_strategy
from farkle.utils.stats import bonferroni_pairs, games_for_power
from farkle.utils.artifacts import write_parquet_atomic

DEFAULT_ROOT = Path("results_seed_0")

LOGGER = logging.getLogger(__name__)


def run_bonferroni_head2head(*, seed: int = 0, root: Path = DEFAULT_ROOT, n_jobs: int = 1) -> None:
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
    tier.  It runs enough games for a Bonferroni-corrected binomial test on each
    matchup and writes ``data/bonferroni_pairwise.parquet`` containing win counts and
    p-values computed via :func:`scipy.stats.binomtest`.
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
    games_needed = games_for_power(len(elites), method="bonferroni", full_pairwise=True)
    schedule = bonferroni_pairs(elites, games_needed, seed)
    LOGGER.debug(
        "Head-to-head schedule prepared",
        extra={
            "stage": "head2head",
            "pairs": len(schedule.index),
            "games": len(schedule),
            "games_needed": games_needed,
        },
    )

    # Nothing to simulate (e.g., only one elite strategy)
    if schedule.empty:
        LOGGER.info(
            "Bonferroni head-to-head: no games needed",
            extra={"stage": "head2head", "elite_count": len(elites)},
        )
        return

    records = []
    for (a, b), grp in schedule.groupby(["a", "b"]):
        seeds = grp["seed"].tolist()
        LOGGER.debug(
            "Simulating head-to-head batch",
            extra={
                "stage": "head2head",
                "strategy_a": a,
                "strategy_b": b,
                "games": len(seeds),
            },
        )
        df = simulate_many_games_from_seeds(
            seeds=seeds,
            strategies=[parse_strategy(a), parse_strategy(b)],
            n_jobs=n_jobs,
        )
        wins = df["winner_strategy"].value_counts()
        wa = int(wins.get(a, 0))
        wb = int(wins.get(b, 0))
        # One-sided: "A beats B"
        pval = binomtest(wa, wa + wb, alternative="greater").pvalue
        records.append({"a": a, "b": b, "wins_a": wa, "wins_b": wb, "pvalue": pval})
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
        records,
        schema=pa.schema(
            [
                ("a", pa.string()),
                ("b", pa.string()),
                ("wins_a", pa.int64()),
                ("wins_b", pa.int64()),
                ("pvalue", pa.float64()),
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
