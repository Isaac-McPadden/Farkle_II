from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import trueskill
import yaml

from .utils import build_tiers

log = logging.getLogger(__name__)


def _read_manifest_seed(path: Path) -> int:
    try:
        data = yaml.safe_load(path.read_text())
        return int(data.get("seed", 0))
    except FileNotFoundError:
        return 0


def _read_row_shards(row_dir: Path) -> pd.DataFrame:
    """Concatenate all parquet shards inside ``row_dir``."""
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
    row_dir = next((p for p in block.glob("*_rows") if p.is_dir()), None)
    if row_dir:
        df = _read_row_shards(row_dir)
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
) -> dict[str, tuple[float, float]]:
    """
    Update ratings using TrueSkill's team API with full table rankings.
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

    return {k: (r.mu, r.sigma) for k, r in ratings.items()}


def run_trueskill(seed: int = 0) -> None:
    base = Path("data/results")
    manifest_seed = _read_manifest_seed(base / "manifest.yaml")
    suffix = f"_seed{seed}" if seed != manifest_seed else ""
    env = trueskill.TrueSkill()
    pooled: Dict[str, tuple[float, float]] = {}
    for block in sorted(base.glob("*_players")):
        n = block.name.split("_")[0]
        keep_path = block / f"keepers_{n}.npy"
        keepers = np.load(keep_path).tolist() if keep_path.exists() else []
        games = _load_ranked_games(block)
        ratings = _update_ratings(games, keepers, env)
        with (Path("data") / f"ratings_{n}{suffix}.pkl").open("wb") as fh:
            pickle.dump(ratings, fh)
        for k, v in ratings.items():
            if k in pooled:
                m, s = pooled[k]
                pooled[k] = ((m + v[0]) / 2, (s + v[1]) / 2)
            else:
                pooled[k] = v
    with (Path("data") / f"ratings_pooled{suffix}.pkl").open("wb") as fh:
        pickle.dump(pooled, fh)
    tiers = build_tiers({k: v[0] for k, v in pooled.items()}, {k: v[1] for k, v in pooled.items()})
    Path("data").mkdir(exist_ok=True)
    with (Path("data") / "tiers.json").open("w") as fh:
        json.dump(tiers, fh, indent=2, sort_keys=True)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute TrueSkill ratings")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv or [])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_trueskill(seed=args.seed)


if __name__ == "__main__":
    main()
