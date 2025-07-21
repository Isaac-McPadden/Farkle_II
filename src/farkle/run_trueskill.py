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


def _load_ranked_games(block: Path) -> list[list[str]]:
    """
    Return one list per game, containing the strategies in finishing order.

    Understands three on-disk layouts, in this priority order
        1. <block>/*_rows/*.parquet         # modern row-shard directory
        2. <block>/winners.csv              # legacy CSV with one winner col
        3. <block>/*.parquet                # legacy parquet(s) in root
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
            return []  # nothing we know how to read

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


def run_trueskill(output_seed: int = 0) -> None:
    """Compute TrueSkill ratings from previous tournament results.

    Parameters
    ----------
    output_seed:
        Value appended to output filenames so repeated runs do not overwrite
        earlier results.
    """

    base = Path("data/results")
    manifest_seed = _read_manifest_seed(base / "manifest.yaml")
    suffix = f"_seed{output_seed}" if output_seed != manifest_seed else ""
    env = trueskill.TrueSkill()
    pooled_sums: Dict[str, list[float]] = {}
    for block in sorted(base.glob("*_players")):
        n = block.name.split("_")[0]
        keep_path = block / f"keepers_{n}.npy"
        keepers = np.load(keep_path).tolist() if keep_path.exists() else []
        games = _load_ranked_games(block)
        ratings = _update_ratings(games, keepers, env)
        with (Path("data") / f"ratings_{n}{suffix}.pkl").open("wb") as fh:
            pickle.dump(ratings, fh)
        for k, v in ratings.items():
            entry = pooled_sums.setdefault(k, [0.0, 0.0, 0])
            entry[0] += v[0]
            entry[1] += v[1]
            entry[2] += 1

    pooled = {k: (mu / cnt, sig / cnt) for k, (mu, sig, cnt) in pooled_sums.items()}
    with (Path("data") / f"ratings_pooled{suffix}.pkl").open("wb") as fh:
        pickle.dump(pooled, fh)
    tiers = build_tiers({k: v[0] for k, v in pooled.items()}, {k: v[1] for k, v in pooled.items()})
    Path("data").mkdir(exist_ok=True)
    with (Path("data") / "tiers.json").open("w") as fh:
        json.dump(tiers, fh, indent=2, sort_keys=True)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute TrueSkill ratings")
    parser.add_argument(
        "--output-seed",
        type=int,
        default=0,
        help="only used to name output files",
    )
    args = parser.parse_args(argv or [])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_trueskill(output_seed=args.output_seed)


if __name__ == "__main__":
    main()
