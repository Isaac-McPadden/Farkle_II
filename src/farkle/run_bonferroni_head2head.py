# src/farkle/run_bonferroni_head2head
"""Bonferroni-corrected head-to-head comparison of top-tier strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import binomtest

from farkle.simulation import simulate_many_games
from farkle.stats import games_for_power
from farkle.strategies import parse_strategy
from farkle.utils import bonferroni_pairs

DEFAULT_ROOT = Path("data")


def run_bonferroni_head2head(seed: int = 0, root: Path = DEFAULT_ROOT) -> None:
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
    root = Path(root)
    tiers_path = root / "tiers.json"
    pairwise_csv = root / "bonferroni_pairwise.csv"

    try:
        with tiers_path.open() as fh:
            tiers = json.load(fh)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Tier file not found at {tiers_path}") from exc
    if not tiers:
        raise RuntimeError(f"No tiers found in {tiers_path}")
    top_val = min(tiers.values())
    elites = [s for s, t in tiers.items() if t == top_val]
    games_needed = games_for_power(len(elites), method="bonferroni", full_pairwise=True)
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
    pairwise_csv.parent.mkdir(exist_ok=True)
    out.to_csv(pairwise_csv, index=False)


def main(argv: List[str] | None = None) -> None:
    """Command line interface for :func:`run_bonferroni_head2head`.

    Usage
        python -m farkle.run_bonferroni_head2head [--seed N]
    """
    parser = argparse.ArgumentParser(description="Head-to-head Bonferroni analysis")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    run_bonferroni_head2head(seed=args.seed, root=args.root)


if __name__ == "__main__":
    main()
