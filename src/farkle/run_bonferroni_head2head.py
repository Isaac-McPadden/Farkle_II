from __future__ import annotations

"""Bonferroni-corrected head-to-head comparison of top-tier strategies."""

import argparse
import json
from pathlib import Path
from typing import List

TIERS_PATH = Path("data/tiers.json")
PAIRWISE_CSV = Path("data/bonferroni_pairwise.csv")

import pandas as pd
from scipy.stats import binomtest

from .simulation import simulate_many_games
from .stats import games_for_power
from .strategies import parse_strategy
from .utils import bonferroni_pairs


def run_bonferroni_head2head(seed: int = 0) -> None:
    """Run pairwise games between top-tier strategies using Bonferroni tests.

    Parameters
    ----------
    seed : int, default ``0``
        Seed for shuffling the schedule and for each simulated game.

    The function reads ``data/tiers.json`` to find strategies in the highest
    tier.  It runs enough games for a Bonferroni-corrected binomial test on each
    matchup and writes ``data/bonferroni_pairwise.csv`` containing win counts and
    p-values computed via :func:`scipy.stats.binomtest`.
    """

    with open("TIERS_PATH") as fh:
        tiers = json.load(fh)
    top_val = min(tiers.values())
    elites = [s for s, t in tiers.items() if t == top_val]
    games_needed = games_for_power(len(elites), method="bonferroni", pairwise=True)
    schedule = bonferroni_pairs(elites, games_needed, seed)

    records = []
    for (a, b), grp in schedule.groupby(["a", "b"]):
        n_games = len(grp)
        df = simulate_many_games(
            n_games=n_games, strategies=[parse_strategy(a), parse_strategy(b)], seed=seed, n_jobs=1
        )
        wins = df["winner_strategy"].value_counts()
        wa = int(wins.get(a, 0))
        wb = int(wins.get(b, 0))
        pval = binomtest(wa, wa + wb).pvalue
        records.append({"a": a, "b": b, "wins_a": wa, "wins_b": wb, "pvalue": pval})

    out = pd.DataFrame(records)
    PAIRWISE_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(PAIRWISE_CSV, index=False)


def main(argv: List[str] | None = None) -> None:
    """Command line interface for :func:`run_bonferroni_head2head`.

    Usage
        python -m farkle.run_bonferroni_head2head [--seed N]
    """
    parser = argparse.ArgumentParser(description="Head-to-head Bonferroni analysis")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_bonferroni_head2head(seed=args.seed)


if __name__ == "__main__":
    main()
