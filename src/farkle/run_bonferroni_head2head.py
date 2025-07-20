from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import binomtest

from .simulation import simulate_many_games_from_seeds
from .stats import games_for_power
from .strategies import parse_strategy
from .utils import bonferroni_pairs


def run_bonferroni_head2head(seed: int = 0) -> None:
    with open("data/tiers.json") as fh:
        tiers = json.load(fh)
    top_val = min(tiers.values())
    elites = [s for s, t in tiers.items() if t == top_val]
    games_needed = games_for_power(len(elites), method="bonferroni", pairwise=True)
    schedule = bonferroni_pairs(elites, games_needed, seed)

    records = []
    for (a, b), grp in schedule.groupby(["a", "b"]):
        df = simulate_many_games_from_seeds(
            seeds=grp["seed"].tolist(),
            strategies=[parse_strategy(a), parse_strategy(b)],
            n_jobs=1,
        )
        wins = df["winner_strategy"].value_counts()
        wa = int(wins.get(a, 0))
        wb = int(wins.get(b, 0))
        pval = binomtest(wa, wa + wb).pvalue
        records.append({"a": a, "b": b, "wins_a": wa, "wins_b": wb, "pvalue": pval})

    out = pd.DataFrame(records)
    Path("data").mkdir(exist_ok=True)
    out.to_csv("data/bonferroni_pairwise.csv", index=False)


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Head-to-head Bonferroni analysis")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run_bonferroni_head2head(seed=args.seed)


if __name__ == "__main__":
    main()
    