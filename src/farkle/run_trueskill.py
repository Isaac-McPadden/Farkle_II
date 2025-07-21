from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path


import numpy as np
import pandas as pd
import trueskill
import yaml

from .utils import build_tiers

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RatingStats:
    """Simple TrueSkill rating stats container."""

    mu: float
    sigma: float


def _read_manifest_seed(path: Path) -> int:
    """Return the tournament seed recorded in ``manifest.yaml``.

    Parameters
    ----------
    path : Path
        Location of the manifest file. The YAML is expected to contain a
        ``seed`` entry.

    Returns
    -------
    int
        The integer seed found in the file. ``0`` is returned when the file
        does not exist or the value is missing.
    """

    try:
        data = yaml.safe_load(path.read_text())
        return int(data.get("seed", 0))
    except FileNotFoundError:
        return 0


def _load_ranked_games(block: Path) -> list[list[str]]:
    """Load all ranked games for a results block.

    Parameters
    ----------
    block : Path
        Directory containing the results for one tournament block. The
        directory may be organised in one of three supported layouts:

        ``*_rows`` directory of Parquet files,
        a ``winners.csv`` file,
        or loose ``.parquet`` files in the block root.

    Returns
    -------
    list[list[str]]
        One list per game with strategy names ordered from first place to
        last. An empty list is returned when the block contains no readable
        data.
    """
    # ── 1. Modern layout ────────────────────────────────────────────────────
    row_dirs = sorted(block.glob("*_rows"))
    if row_dirs:
        frames: list[pd.DataFrame] = []
        for row_dir in row_dirs:
            frames.extend(pd.read_parquet(p) for p in row_dir.glob("*.parquet"))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ── 2. winners.csv fallback ────────────────────────────────────────────
    elif (block / "winners.csv").exists():
        df = pd.read_csv(block / "winners.csv")

    # ── 3. Loose parquet(s) in the block root ──────────────────────────────
    else:
        parquet_files = list(block.glob("*.parquet"))
        if parquet_files:
            frames = [pd.read_parquet(p) for p in parquet_files]
            df = pd.concat(frames, ignore_index=True)
        else:
            return []                       # nothing we know how to read

    # ----------------------------------------------------------------------
    rank_cols  = [c for c in df.columns if c.endswith("_rank")]
    strat_cols = {c[:-5]: f"{c[:-5]}_strategy" for c in rank_cols}

    games: list[list[str]] = []
    for _, row in df.iterrows():
        if rank_cols:                                   # modern per-player ranks
            ordered = sorted(rank_cols, key=row.__getitem__)
            players = [row[strat_cols[c[:-5]]] for c in ordered]
        else:                                           # winner-only rows
            if "winner_strategy" in row:
                players = [row["winner_strategy"]]
            elif "winner" in row:
                players = [row["winner"]]
            else:
                players = []        # unknown schema → ignore row
        games.append(players)

    return games


def _update_ratings(
    games: list[list[str]],
    keepers: list[str],
    env: trueskill.TrueSkill,
) -> dict[str, RatingStats]:
    """Update strategy ratings for a block of games.

    Parameters
    ----------
    games : list[list[str]]
        Each inner list contains strategy names ordered by finishing position
        for a single game.
    keepers : list[str]
        Strategies to include in the rating calculation. If empty, all
        strategies appearing in ``games`` are rated.
    env : trueskill.TrueSkill
        The environment used to perform the TrueSkill updates.

    Returns
    -------
    dict[str, RatingStats]
        Mapping from strategy name to its RatingStats tuple.
    """
    ratings: dict[str, trueskill.Rating] = {
        k: env.create_rating() for k in keepers
    }

    for game in games:
        # keep only the strategies we really want to rate
        players = [s for s in game if (not keepers or s in keepers)]
        teams   = [[ratings.setdefault(s, env.create_rating())] for s in players]

        if len(teams) < 2:
            continue  # need at least two teams for a rating update

        new_teams = env.rate(teams, ranks=range(len(teams)))
        for s, team_rating in zip(players, new_teams, strict=True):
            ratings[s] = team_rating[0]

    return {k: RatingStats(r.mu, r.sigma) for k, r in ratings.items()}


def run_trueskill(seed: int = 0) -> None:
    """Compute TrueSkill ratings for all result blocks.

    Parameters
    ----------
    seed : int, optional
        Seed used when generating the results. It is compared against the
        value stored in ``manifest.yaml`` to decide whether a suffix should be
        appended to the rating files.

    Side Effects
    ------------
    ``ratings_<n>[ _seedX ].pkl``
        Pickle files containing per-block ratings for each strategy.
    ``ratings_pooled[ _seedX ].pkl``
        A pickle with ratings pooled across all blocks.
    ``tiers.json``
        JSON file with league tiers derived from the pooled ratings.
    """

    base = Path("data/results")
    manifest_seed = _read_manifest_seed(base / "manifest.yaml")
    suffix = f"_seed{seed}" if seed != manifest_seed else ""
    env = trueskill.TrueSkill()
    pooled: Dict[str, RatingStats] = {}
    for block in sorted(base.glob("*_players")):
        player_count = block.name.split("_")[0]
        keep_path = block / f"keepers_{player_count}.npy"
        keepers = np.load(keep_path).tolist() if keep_path.exists() else []
        games = _load_ranked_games(block)
        ratings = _update_ratings(games, keepers, env)
        with (Path("data") / f"ratings_{player_count}{suffix}.pkl").open("wb") as fh:
            pickle.dump(ratings, fh)
        for k, v in ratings.items():
            if k in pooled:
                r = pooled[k]
                pooled[k] = RatingStats((r.mu + v.mu) / 2, (r.sigma + v.sigma) / 2)
            else:
                pooled[k] = v
    with (Path("data") / f"ratings_pooled{suffix}.pkl").open("wb") as fh:
        pickle.dump(pooled, fh)
    tiers = build_tiers({k: v.mu for k, v in pooled.items()}, {k: v.sigma for k, v in pooled.items()})
    Path("data").mkdir(exist_ok=True)
    with (Path("data") / "tiers.json").open("w") as fh:
        json.dump(tiers, fh, indent=2, sort_keys=True)


def main(argv: List[str] | None = None) -> None:
    """Entry point for the ``run_trueskill`` command line interface.

    Parameters
    ----------
    argv : list[str] | None, optional
        Arguments to parse instead of ``sys.argv``. Each argument should be a
        separate element in the list.

    Side Effects
    ------------
    Invokes :func:`run_trueskill`, which writes rating and tier files to the
    ``data`` directory.
    """
    parser = argparse.ArgumentParser(description="Compute TrueSkill ratings")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv or [])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_trueskill(seed=args.seed)


if __name__ == "__main__":
    main()
    