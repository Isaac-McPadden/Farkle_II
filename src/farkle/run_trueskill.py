# src/farkle/run_trueskill.py
"""Compute TrueSkill ratings for Farkle strategies.

The script scans a directory of tournament results, updates ratings with the
``trueskill`` package and writes per-block as well as pooled rating files.

Key CLI flags
-------------
--dataroot   Directory containing <N>_players blocks            (default: data/results)
--root       Output directory for analysis artefacts            (default: <dataroot>/analysis)

Outputs
-------
ratings_<N>.pkl(*), ratings_pooled.pkl – pickled RatingStats
tiers.json                       – mapping of strategy → tier
"""
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

_REPO_ROOT = Path(__file__).resolve().parents[2]  # hop out of src/farkle
# Default location of tournament result blocks when no path is supplied
DEFAULT_DATAROOT = _REPO_ROOT / "data" / "results"

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


def _read_row_shards(row_dir: Path) -> pd.DataFrame:
    """Return a DataFrame built from all Parquet files in ``row_dir``."""
    frames = [pd.read_parquet(p) for p in row_dir.glob("*.parquet")]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_winners_csv(block: Path) -> pd.DataFrame:
    """Read ``winners.csv`` from a result block."""
    return pd.read_csv(block / "winners.csv")


def _read_loose_parquets(block: Path) -> pd.DataFrame | None:
    """Return a DataFrame from loose parquet files or ``None`` when absent."""
    files = list(block.glob("*.parquet"))
    if not files:
        return None
    frames = [pd.read_parquet(p) for p in files]
    return pd.concat(frames, ignore_index=True)


def _load_ranked_games(block: Path) -> list[list[str]]:
    """Return one list per game, ordered by finishing position."""
    # Prefer the consolidated `<Np_rows>.parquet` file when available.
    row_file = next(block.glob("*p_rows.parquet"), None)
    if row_file is not None:
        df = pd.read_parquet(row_file)
    else:
        row_dirs = [p for p in block.glob("*_rows") if p.is_dir()]
        if row_dirs:
            frames = [_read_row_shards(d) for d in row_dirs]
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        elif (block / "winners.csv").exists():
            df = _read_winners_csv(block)
        else:
            df = _read_loose_parquets(block)
            if df is None:
                return []

    # ----------------------------------------------------------------------
    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    strat_cols = {c[:-5]: f"{c[:-5]}_strategy" for c in rank_cols}

    games: list[list[str]] = []
    for _, row in df.iterrows():
        if rank_cols:  # modern per-player ranks
            ordered = sorted(rank_cols, key=row.__getitem__)
            players = [row[strat_cols[c[:-5]]] for c in ordered]
        else:  # winner-only rows
            if "winner_strategy" in row:
                players = [row["winner_strategy"]]
            elif "winner" in row:
                players = [row["winner"]]
            else:
                players = []  # unknown schema → ignore row
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
    ratings: dict[str, trueskill.Rating] = {k: env.create_rating() for k in keepers}

    for game in games:
        # keep only the strategies we really want to rate
        players = [s for s in game if (not keepers or s in keepers)]
        teams = [[ratings.setdefault(s, env.create_rating())] for s in players]

        if len(teams) < 2:
            continue  # need at least two teams for a rating update

        new_teams = env.rate(teams, ranks=range(len(teams)))
        for s, team_rating in zip(players, new_teams, strict=True):
            ratings[s] = team_rating[0]

    return {k: RatingStats(r.mu, r.sigma) for k, r in ratings.items()}


def run_trueskill(
    output_seed: int = 0,
    root: Path | None = None,
    dataroot: Path | None = None,
) -> None:
    """Compute TrueSkill ratings for all result blocks.

    Parameters
    ----------
    output_seed: int, optional
        Value appended to output filenames so repeated runs do not overwrite
        earlier results.
    root: Path | None, optional
        Directory where rating artifacts are written. Defaults to
        ``<dataroot>/analysis`` when ``None``.
    dataroot: Path | None, optional
        Directory containing tournament result blocks. When ``None`` the
        path defaults to ``<root>/results`` if ``root`` is given, otherwise
        to :data:`DEFAULT_DATAROOT`.

    Side Effects
    ------------
    ``ratings_<n>[ _seedX ].pkl``
        Pickle files containing per-block ratings for each strategy.
    ``ratings_pooled[ _seedX ].pkl``
        A pickle with ratings pooled across all blocks using a games-per-block
        weighted mean.
    ``tiers.json``
        JSON file with league tiers derived from the pooled ratings.
    """
    if dataroot is None:
        base = Path(root) / "results" if root is not None else DEFAULT_DATAROOT
    else:
        base = Path(dataroot)

    root = Path(root) if root is not None else base / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    _read_manifest_seed(base / "manifest.yaml")
    suffix = f"_seed{output_seed}" if output_seed else ""
    env = trueskill.TrueSkill()
    pooled: dict[str, RatingStats] = {}
    pooled_weights: dict[str, int] = {}
    for block in sorted(base.glob("*_players")):
        player_count = block.name.split("_")[0]
        keep_path = block / f"keepers_{player_count}.npy"
        keepers = np.load(keep_path).tolist() if keep_path.exists() else []
        games = _load_ranked_games(block)
        ratings = _update_ratings(games, keepers, env)
        with (root / f"ratings_{player_count}{suffix}.pkl").open("wb") as fh:
            pickle.dump(ratings, fh)
        block_games = len(games)
        for k, v in ratings.items():
            if k in pooled:
                weight = pooled_weights[k]
                new_weight = weight + block_games
                pooled[k] = RatingStats(
                    (pooled[k].mu * weight + v.mu * block_games) / new_weight,
                    (pooled[k].sigma * weight + v.sigma * block_games) / new_weight,
                )
                pooled_weights[k] = new_weight
            else:
                pooled[k] = v
                pooled_weights[k] = block_games
    with (root / f"ratings_pooled{suffix}.pkl").open("wb") as fh:
        pickle.dump(pooled, fh)
    tiers = build_tiers(
        means={k: v.mu for k, v in pooled.items()},
        stdevs={k: v.sigma for k, v in pooled.items()},
    )
    with (root / "tiers.json").open("w") as fh:
        json.dump(tiers, fh, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``run_trueskill`` command line interface.

    Parameters
    ----------
    argv : list[str] | None, optional
        Arguments to parse instead of ``sys.argv``. Each argument should be a
        separate element in the list.

    Side Effects
    ------------
    Invokes :func:`run_trueskill`, which writes rating and tier files to
    ``--root`` (default ``<dataroot>/analysis``).
    """
    parser = argparse.ArgumentParser(description="Compute TrueSkill ratings")
    parser.add_argument(
        "--output-seed", type=int, default=0,
        help="appended to filenames to avoid overwrites",
    )
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=None,
        help=(
            "Folder that holds <N>_players blocks "
            "(default: <root>/results or <repo>/data/results). "
            "Accepts absolute or relative paths."
        ),
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="output directory (default: <dataroot>/analysis)",
    )

    args = parser.parse_args(argv or [])
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_trueskill(
        output_seed=args.output_seed,
        root=args.root,
        dataroot=args.dataroot,
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
