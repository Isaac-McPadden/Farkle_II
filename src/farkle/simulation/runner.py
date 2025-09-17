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

import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import farkle.simulation.run_tournament as tournament_mod
from farkle.simulation.run_tournament import TournamentConfig
from farkle.utils import sinks

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
    collect_metrics:
        When ``True`` the full tournament driver records per-strategy metric
        aggregates in addition to win counts.
    row_dir:
        Optional output directory for per-game rows written as parquet shards.
    """

    jobs: int | None = None
    seed: int = 0
    n_games: int = 0
    checkpoints: int | None = None
    n_players: int = 5
    collect_metrics: bool = False
    row_dir: Path | None = None


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

    The function delegates to :func:`farkle.simulation.run_tournament.run_tournament`
    and mirrors its behaviour closely while retaining the simple configuration
    surface used by older entry points.  Win counts are written to
    ``cfg.io.results_dir / "win_counts.csv"`` and the total number of games
    simulated is returned.
    """

    tcfg = TournamentConfig(n_players=cfg.sim.n_players)

    games_per_shuffle = tcfg.games_per_shuffle
    if games_per_shuffle <= 0:
        raise ValueError("games_per_shuffle must be positive")

    # round up to the next whole shuffle so that each strategy participates
    n_shuffles = max(1, -(-cfg.sim.n_games // games_per_shuffle))

    out_dir = cfg.io.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "checkpoint.pkl"

    row_dir = cfg.sim.row_dir
    if row_dir is not None and not row_dir.is_absolute():
        row_dir = out_dir / row_dir

    tournament_mod.run_tournament(
        config=tcfg,
        global_seed=cfg.sim.seed,
        n_jobs=cfg.sim.jobs,
        num_shuffles=n_shuffles,
        checkpoint_path=checkpoint_path,
        collect_metrics=cfg.sim.collect_metrics,
        row_output_directory=row_dir,
    )

    payload = pickle.loads(checkpoint_path.read_bytes())
    if isinstance(payload, dict) and "win_totals" in payload:
        raw_counts = payload["win_totals"]
    else:
        raw_counts = payload

    if isinstance(raw_counts, Counter):
        win_totals = Counter(raw_counts)
    elif isinstance(raw_counts, Mapping):
        win_totals = Counter({str(k): int(v) for k, v in raw_counts.items()})
    else:
        raise TypeError(
            "Unexpected win_totals payload type",
            type(raw_counts),
        )

    sinks.write_counter_csv(win_totals, out_dir / "win_counts.csv")

    return n_shuffles * games_per_shuffle


__all__ = ["AppConfig", "SimConfig", "IOConfig", "run_tournament"]
