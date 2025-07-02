#!/usr/bin/env python3
"""
time_farkle.py

Usage:
    python time_farkle.py [--n_games N] [--players P] [--seed S] [--jobs J]

Prints timing for a single game and for a batch of N games
using random ThresholdStrategy instances for each player.
"""

import argparse
import random
import time

import pandas as pd

from farkle.simulation import (
    simulate_many_games,
    simulate_one_game,
)
from farkle.strategies import random_threshold_strategy


def make_random_strategies(num_players: int, seed: int | None) -> list:
    """Summary
    -------
    Generates a list of random ThresholdStrategy objects for each player.

    Inputs
    ------
    num_players: int 
    seed: int | None

    Returns
    -------
    list
        Randomly generated ThresholdStrategy objects
    """
    rng = random.Random(seed)
    return [random_threshold_strategy(rng) for _ in range(num_players)]


def measure_sim_times(argv: list[str] | None = None):
    """Summary
    -------
    Run timing benchmarks for one game and a batch of games.

    Inputs
    ------
    argv : list[str] | None, optional
        Argument list to parse instead of ``sys.argv``.

    Returns
    -------
    None
    """
    p = argparse.ArgumentParser(
        description="Time one Farkle game and a batch of N games."
    )
    p.add_argument(
        "-n", "--n_games", type=int, default=1000,
        help="Number of games to simulate in batch"
    )
    p.add_argument(
        "-p", "--players", type=int, default=5,
        help="Number of players per game"
    )
    p.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Master seed for reproducible RNG"
    )
    p.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Number of parallel processes"
    )
    args = p.parse_args(argv)

    # 1) Build a fixed roster of random strategies
    strategies = make_random_strategies(args.players, args.seed)

    # 2) Time a single game
    t0 = time.perf_counter()
    gm = simulate_one_game(
        strategies=strategies,
        seed=args.seed
    )
    t1 = time.perf_counter()
    print("\nSingle game:")
    print(f"  Time elapsed      : {t1-t0:.6f} s")
    print(f"  Winner            : {gm.winner}")
    print(f"  Winning score     : {gm.winning_score}")
    print(f"  Rounds            : {gm.n_rounds}")

    # 3) Time a batch of N games
    t0 = time.perf_counter()
    df: pd.DataFrame = simulate_many_games(
        n_games=args.n_games,
        strategies=strategies,
        seed=args.seed,
        n_jobs=args.jobs
    )
    t1 = time.perf_counter()
    print(f"\nBatch of {args.n_games} games (jobs={args.jobs}):")
    print(f"  Time elapsed      : {t1-t0:.6f} s")
    print("  Winners breakdown :")
    print(df["winner"].value_counts().to_string())
    
def main():
    measure_sim_times()

if __name__ == "__main__":
    main()
