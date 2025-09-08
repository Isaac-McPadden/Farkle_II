# src/farkle/run_bonferroni_head2head
"""Bonferroni-corrected head-to-head comparison of top-tier strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import binomtest

from farkle.simulation import simulate_many_games_from_seeds
from farkle.stats import games_for_power
from farkle.strategies import parse_strategy
from farkle.utils import bonferroni_pairs

DEFAULT_ROOT = Path("results_seed_0")


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
    games_needed = games_for_power(len(elites), method="bonferroni", full_pairwise=True)
    schedule = bonferroni_pairs(elites, games_needed, seed)

    # Nothing to simulate (e.g., only one elite strategy)
    if schedule.empty:
        print("\u2713 Bonferroni H2H: no games needed \u2014 exiting early.")
        return

    records = []
    for (a, b), grp in schedule.groupby(["a", "b"]):
        seeds = grp["seed"].tolist()
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

    out = pd.DataFrame(records)
    pairwise_parquet.parent.mkdir(exist_ok=True)

    tmp_path = pairwise_parquet.with_suffix(".tmp")
    out.to_csv(tmp_path, index=False)
    tmp_path.replace(pairwise_parquet)


def main(argv: List[str] | None = None) -> None:
    """Command line interface for :func:`run_bonferroni_head2head`.

    Usage
        python -m farkle.run_bonferroni_head2head [--seed N] [--root R] [--jobs J]
    """
    parser = argparse.ArgumentParser(description="Head-to-head Bonferroni analysis")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    args = parser.parse_args(argv)
    run_bonferroni_head2head(seed=args.seed, root=args.root, n_jobs=args.jobs)


if __name__ == "__main__":
    main()
