# src/farkle/time_farkle.py
"""
Usage:
    python time_farkle.py [--n_games N] [--players P] [--seed S] [--jobs J]

Prints timing for a single game and for a batch of N games
using random ThresholdStrategy instances for each player.
"""

import random
import time

import pandas as pd

from farkle.simulation.simulation import (
    simulate_many_games,
    simulate_one_game,
)
from farkle.simulation.strategies import ThresholdStrategy, random_threshold_strategy


def make_random_strategies(num_players: int, seed: int | None) -> list[ThresholdStrategy]:
    """Generate random strategies for each player."""

    rng = random.Random(seed)
    return [random_threshold_strategy(rng) for _ in range(num_players)]


def measure_sim_times(
    *, n_games: int = 1000, players: int = 5, seed: int = 42, jobs: int = 1
) -> None:
    """Run timing benchmarks for one game and a batch of games.

    The function keeps the original printed output but no longer parses
    command-line arguments.  All parameters are supplied directly.
    """

    strategies = make_random_strategies(players, seed)

    t0 = time.perf_counter()
    gm = simulate_one_game(strategies=strategies, seed=seed)
    t1 = time.perf_counter()
    print("\nSingle game:")
    print(f"  Time elapsed      : {t1 - t0:.6f} s")
    winner = max(gm.players.items(), key=lambda p: p[1].score)[0]
    print(f"  Winner            : {winner}")
    print(f"  Winning score     : {gm.players[winner].score}")
    print(f"  Rounds            : {gm.game.n_rounds}")

    t0 = time.perf_counter()
    df: pd.DataFrame = simulate_many_games(
        n_games=n_games, strategies=strategies, seed=seed, n_jobs=jobs
    )
    t1 = time.perf_counter()
    print(f"\nBatch of {n_games} games (jobs={jobs}):")
    print(f"  Time elapsed      : {t1 - t0:.6f} s")
    print("  Winners breakdown :")
    print(df["winner"].value_counts().to_string())


__all__ = ["measure_sim_times", "make_random_strategies"]
