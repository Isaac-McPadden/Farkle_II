from __future__ import annotations
"""simulation.py
================
High‑level utilities for *batch* and *grid* simulations.

This is the entry point most users will reach for:

* ``generate_strategy_grid`` – produce the canonical 800‑strategy (or
  custom) grid and its accompanying ``DataFrame``.
* ``simulate_many_games`` – run *N* games, optionally in parallel, and
  return tidy metrics.
* ``aggregate_metrics`` – summarise a DataFrame of game results.
"""

from itertools import product
from typing import Sequence, Tuple, List, Dict, Any
import random
import multiprocessing as mp

import pandas as pd

from strategies import ThresholdStrategy
from engine import FarklePlayer, FarkleGame

__all__: list[str] = [
    "generate_strategy_grid",
    "experiment_size",
    "simulate_one_game",
    "simulate_many_games",
    "aggregate_metrics",
]

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def generate_strategy_grid(
    *,
    score_thresholds: Sequence[int] | None = None,
    dice_thresholds: Sequence[int] | None = None,
    smart_options: Sequence[bool] | None = None,
    consider_score_opts: Sequence[bool] | None = (True, False),
    consider_dice_opts: Sequence[bool] | None = (True, False),
) -> Tuple[List[ThresholdStrategy], pd.DataFrame]:
    """Create the Cartesian product of all parameter options.

    Returns a tuple *(strategies, meta_df)* where the first element is a
    list of fully‑constructed ``ThresholdStrategy`` instances (safe to
    pass to the engine) and the second element is a *metadata* dataframe
    recording each parameter combo.
    """
    score_thresholds = score_thresholds or list(range(200, 1050, 50))
    dice_thresholds = dice_thresholds or list(range(0, 5))
    smart_options = smart_options or [True, False]
    combos = list(product(score_thresholds, dice_thresholds, smart_options, consider_score_opts, consider_dice_opts))
    strategies = [ThresholdStrategy(st, dt, sm, cs, cd) for st, dt, sm, cs, cd in combos]
    meta = pd.DataFrame(combos, columns=["score_threshold", "dice_threshold", "smart", "consider_score", "consider_dice"])
    meta["strategy_idx"] = meta.index
    return strategies, meta


def experiment_size(
    *,
    score_thresholds: Sequence[int] | None = None,
    dice_thresholds: Sequence[int] | None = None,
    smart_options: Sequence[bool] | None = None,
    consider_score_opts: Sequence[bool] | None = (True, False),
    consider_dice_opts: Sequence[bool] | None = (True, False),
) -> int:
    """Compute *a priori* size of a strategy grid."""
    score_thresholds = score_thresholds or list(range(200, 1050, 50))
    dice_thresholds = dice_thresholds or list(range(0, 5))
    smart_options = smart_options or [True, False]
    return (
        len(score_thresholds)
        * len(dice_thresholds)
        * len(smart_options)
        * len(consider_score_opts)
        * len(consider_dice_opts)
    )

# ---------------------------------------------------------------------------
# Batch simulation helpers
# ---------------------------------------------------------------------------

def _play_game(seed: int, strategies: Sequence[ThresholdStrategy], target_score: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    players = [FarklePlayer(name=f"P{i+1}", strategy=s, rng=rng) for i, s in enumerate(strategies)]
    gm = FarkleGame(players, target_score=target_score).play()
    flat: Dict[str, Any] = {
        "winner": gm.winner,
        "winning_score": gm.winning_score,
        "n_rounds": gm.n_rounds,
    }
    for pname, stats in gm.per_player.items():
        for k, v in stats.items():
            flat[f"{pname}_{k}"] = v
    return flat


def simulate_many_games(
    *,
    n_games: int,
    strategies: Sequence[ThresholdStrategy],
    target_score: int = 10_000,
    seed: int | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run *n_games* in parallel (if *n_jobs>1*) and return tidy metrics."""
    master_rng = random.Random(seed)
    seeds = [master_rng.randint(0, 2**32 - 1) for _ in range(n_games)]
    args = [(s, strategies, target_score) for s in seeds]
    if n_jobs == 1:
        rows = [_play_game(*a) for a in args]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            rows = pool.starmap(_play_game, args)
    return pd.DataFrame(rows)


def simulate_one_game(
    *,
    strategies: Sequence[ThresholdStrategy],
    target_score: int = 10_000,
    seed: int | None = None,
):
    """Convenience wrapper around the *single* game engine."""
    rng = random.Random(seed)
    players = [FarklePlayer(name=f"P{i+1}", strategy=s, rng=rng) for i, s in enumerate(strategies)]
    return FarkleGame(players, target_score=target_score).play()


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def aggregate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a *dict* with high‑level summary statistics."""
    return {
        "games": len(df),
        "avg_rounds": df["n_rounds"].mean(),
        "winner_freq": df["winner"].value_counts().to_dict(),
    }
