"""High level tournament runner using configuration objects.

The :func:`run_tournament` function acts as a thin wrapper around the
lower level helpers found in :mod:`farkle.simulation.run_tournament`.
It accepts an :class:`AppConfig` instance which mirrors the structure
used throughout the project.  Only a tiny subset of the original
configuration surface is supported – just enough for the unit tests –
but additional fields can be added in the future without touching the
public API.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from farkle.simulation.run_tournament import (
    TournamentConfig,
    _init_worker,
    _play_shuffle,
    generate_strategy_grid,
)
from farkle.utils import parallel, random, sinks

# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """Simulation related settings.

    Attributes
    ----------
    jobs:
        Number of worker processes. ``None`` lets the executor decide.
    seed:
        Master RNG seed.  Individual worker seeds are derived from it.
    n_games:
        Total number of games to simulate.  The number is rounded up so that
        all strategies get an equal number of appearances.
    checkpoints:
        Unused placeholder for future extension; kept for compatibility with
        earlier iterations of the project.
    n_players:
        Number of players in each simulated game.
    """

    jobs: int | None = None
    seed: int = 0
    n_games: int = 0
    checkpoints: int | None = None
    n_players: int = 5


@dataclass
class IOConfig:
    """Simple container for I/O paths."""

    results_dir: Path = Path("results")


@dataclass
class AppConfig:
    """Top level configuration passed to :func:`run_tournament`."""

    sim: SimConfig = field(default_factory=SimConfig)
    io: IOConfig = field(default_factory=IOConfig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_tournament(cfg: AppConfig) -> int:
    """Run a Farkle tournament according to ``cfg``.

    The function executes a Monte-Carlo tournament by repeatedly invoking
    :func:`farkle.simulation.run_tournament._play_shuffle`.  Results are
    aggregated into a ``strategy → wins`` counter which is persisted as a CSV
    file inside ``cfg.io.results_dir``.  A best-effort integer describing the
    number of games played is returned.
    """

    strategies, _ = generate_strategy_grid()
    tcfg = TournamentConfig(n_players=cfg.sim.n_players)

    games_per_shuffle = tcfg.games_per_shuffle
    if games_per_shuffle <= 0:
        raise ValueError("games_per_shuffle must be positive")

    # round up to the next whole shuffle so that each strategy participates
    n_shuffles = max(1, -(-cfg.sim.n_games // games_per_shuffle))

    seeds = random.spawn_seeds(n_shuffles, seed=cfg.sim.seed)

    win_totals: Counter[str] = Counter()
    for wins in parallel.process_map(
        _play_shuffle,
        seeds,
        n_jobs=cfg.sim.jobs,
        initializer=_init_worker,
        initargs=(strategies, tcfg, None),
    ):
        win_totals.update(wins)

    out_dir = cfg.io.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sinks.write_counter_csv(win_totals, out_dir / "win_counts.csv")

    return n_shuffles * games_per_shuffle


__all__ = ["AppConfig", "SimConfig", "IOConfig", "run_tournament"]
