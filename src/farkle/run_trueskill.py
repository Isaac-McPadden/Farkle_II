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


def _load_winners(block: Path) -> List[str]:
    row_dir = next(block.glob("*_rows"), None)
    if row_dir and row_dir.is_dir():
        files = list(row_dir.glob("*.parquet"))
        if files:
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            col = "winner_strategy" if "winner_strategy" in df.columns else "winner"
            return df[col].tolist()
    csv = block / "winners.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        col = "winner_strategy" if "winner_strategy" in df.columns else "winner"
        return df[col].tolist()
    return []


def _update_ratings(
    winners: List[str], keepers: List[str], env: trueskill.TrueSkill
) -> Dict[str, tuple[float, float]]:
    ratings = {k: env.create_rating() for k in keepers}
    dummy = env.create_rating()
    prev_mu = {k: r.mu for k, r in ratings.items()}
    total = len(winners)
    for i, w in enumerate(winners, 1):
        if w not in ratings:
            continue
        ratings[w], dummy = trueskill.rate_1vs1(ratings[w], dummy, env=env)
        if i % 100_000 == 0 or i == total:
            diff = [abs(ratings[k].mu - prev_mu[k]) for k in ratings]
            med = float(np.median(diff)) if diff else 0.0
            log.info("%s games → median Δμ=%.6f", i, med)
            prev_mu = {k: r.mu for k, r in ratings.items()}
            if i >= 100_000 and med < 0.005:
                break
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
        winners = _load_winners(block)
        winners = [w for w in winners if not keepers or w in keepers]
        ratings = _update_ratings(winners, keepers, env)
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
    