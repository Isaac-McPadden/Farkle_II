# src/farkle/simulation/time_farkle.py
"""
Functions used timing for a single game and for a batch of N games
using random ThresholdStrategy instances for each player.  Used 
for determining chunk size when running tournaments.
"""

import logging
import random
import time

import pandas as pd

from farkle.simulation.simulation import (
    simulate_many_games,
    simulate_one_game,
)
from farkle.simulation.strategies import ThresholdStrategy, random_threshold_strategy

LOGGER = logging.getLogger(__name__)


def make_random_strategies(num_players: int, seed: int | None) -> list[ThresholdStrategy]:
    """Create randomized player strategies for a single simulation run.

    Inputs:
        num_players: Total number of ThresholdStrategy instances to generate.
        seed: Optional seed that makes the random generation reproducible.

    Returns:
        A list of ThresholdStrategy objects, one for each simulated player.
    """

    rng = random.Random(seed)
    return [random_threshold_strategy(rng) for _ in range(num_players)]


def measure_sim_times(
    *, n_games: int = 1000, players: int = 5, seed: int = 42, jobs: int = 1
) -> None:
    """Benchmark single-game and multi-game simulation performance.

    Inputs:
        n_games: Number of games to run in the batch benchmark.
        players: Number of players that participate in each simulated game.
        seed: Seed used both for strategy creation and simulation reproducibility.
        jobs: Parallel job count to use when simulating many games.

    Returns:
        None. Logging output captures timing, winners, and benchmark metadata.
    """

    LOGGER.info(
        "Simulation timing start",
        extra={
            "stage": "simulation",
            "benchmark": "time_farkle",
            "players": players,
            "seed": seed,
            "jobs": jobs,
            "n_games": n_games,
        },
    )

    strategies = make_random_strategies(players, seed)

    t0 = time.perf_counter()
    gm = simulate_one_game(strategies=strategies, seed=seed)
    t1 = time.perf_counter()
    winner = max(gm.players.items(), key=lambda p: p[1].score)[0]
    LOGGER.info(
        "Single game benchmark",
        extra={
            "stage": "simulation",
            "benchmark": "single_game",
            "players": players,
            "seed": seed,
            "elapsed_s": t1 - t0,
            "winner": winner,
            "winning_score": gm.players[winner].score,
            "rounds": gm.game.n_rounds,
        },
    )

    t0 = time.perf_counter()
    df: pd.DataFrame = simulate_many_games(
        n_games=n_games, strategies=strategies, seed=seed, n_jobs=jobs
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    gps = (n_games / elapsed) if elapsed > 0 else 0.0
    winners = df["winner"].value_counts().to_dict()
    LOGGER.info(
        "Batch benchmark",
        extra={
            "stage": "simulation",
            "benchmark": "batch",
            "players": players,
            "seed": seed,
            "jobs": jobs,
            "n_games": n_games,
            "elapsed_s": elapsed,
            "games_per_sec": gps,
            "winners": winners,
        },
    )
    LOGGER.info(
        "Simulation timing complete",
        extra={
            "stage": "simulation",
            "benchmark": "time_farkle",
            "players": players,
            "seed": seed,
            "jobs": jobs,
            "n_games": n_games,
        },
    )


__all__ = ["measure_sim_times", "make_random_strategies"]
