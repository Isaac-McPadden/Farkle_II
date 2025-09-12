# src/farkle/time_farkle.py
"""
Usage:
    python time_farkle.py [--n_games N] [--players P] [--seed S] [--jobs J]

Prints timing for a single game and for a batch of N games
using random ThresholdStrategy instances for each player.
"""

import argparse
import random
import time

import pandas as pd

from farkle.simulation.simulation import (
    simulate_many_games,
    simulate_one_game,
)
from farkle.simulation.strategies import ThresholdStrategy, random_threshold_strategy


def _positive_int(value: str) -> int:
    """Return ``value`` as ``int`` if it is strictly positive."""

    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return ivalue


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the :class:`argparse.ArgumentParser` for the CLI."""
    parser = argparse.ArgumentParser(
        description="Time one Farkle game and a batch of N games.",
    )
    parser.add_argument(
        "-n",
        "--n_games",
        type=_positive_int,
        default=1000,
        help="Number of games to simulate in batch",
    )
    parser.add_argument(
        "-p",
        "--players",
        type=_positive_int,
        default=5,
        help="Number of players per game",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=_positive_int,
        default=42,
        help="Master seed for reproducible RNG",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=_positive_int,
        default=1,
        help="Number of parallel processes",
    )
    return parser


def make_random_strategies(num_players: int, seed: int | None) -> list[ThresholdStrategy]:
    """Generate random strategies for each player.

    Args:
        num_players: Number of players in the game.
        seed: Seed for deterministic randomness.

    Returns:
        list[ThresholdStrategy]:
            Randomly generated ``ThresholdStrategy`` objects.
    """
    rng = random.Random(seed)
    return [random_threshold_strategy(rng) for _ in range(num_players)]


def measure_sim_times(argv: list[str] | None = None):
    """Run timing benchmarks for one game and a batch of games.

    Args:
        argv: Optional argument list to parse instead of ``sys.argv``.

    Returns:
        None
    """
    parser = build_arg_parser()
    if argv and any(arg.startswith("--") for arg in argv):
        try:
            args = parser.parse_args(argv)
        except SystemExit as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    else:
        args = parser.parse_args(argv)

    # 1) Build a fixed roster of random strategies
    strategies = make_random_strategies(args.players, args.seed)

    # 2) Time a single game
    t0 = time.perf_counter()
    gm = simulate_one_game(strategies=strategies, seed=args.seed)
    t1 = time.perf_counter()
    print("\nSingle game:")
    print(f"  Time elapsed      : {t1 - t0:.6f} s")
    winner = max(gm.players.items(), key=lambda p: p[1].score)[0]
    print(f"  Winner            : {winner}")
    print(f"  Winning score     : {gm.players[winner].score}")
    print(f"  Rounds            : {gm.game.n_rounds}")

    # 3) Time a batch of N games
    t0 = time.perf_counter()
    df: pd.DataFrame = simulate_many_games(
        n_games=args.n_games, strategies=strategies, seed=args.seed, n_jobs=args.jobs
    )
    t1 = time.perf_counter()
    print(f"\nBatch of {args.n_games} games (jobs={args.jobs}):")
    print(f"  Time elapsed      : {t1 - t0:.6f} s")
    print("  Winners breakdown :")
    print(df["winner"].value_counts().to_string())


def main():
    measure_sim_times()


if __name__ == "__main__":
    main()
