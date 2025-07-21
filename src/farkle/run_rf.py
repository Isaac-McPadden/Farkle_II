from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import List

IMPORTANCE_PATH = Path("data/rf_importance.json")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


def run_rf(seed: int = 0, output_path: Path = IMPORTANCE_PATH) -> None:
    metrics = pd.read_parquet("data/metrics.parquet")
    with open("data/ratings_pooled.pkl", "rb") as fh:
        ratings = pickle.load(fh)
    df_mu = pd.DataFrame({"strategy": list(ratings), "mu": [v[0] for v in ratings.values()]})
    data = metrics.merge(df_mu, on="strategy", how="inner")
    X = data.drop(columns=["strategy", "mu"])
    X = X.astype(float)
    y = data["mu"]

    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(X, y)

    imp = permutation_importance(model, X, y, n_repeats=5, random_state=seed)
    imp_dict = {c: float(s) for c, s in zip(X.columns, imp["importances_mean"], strict=False)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(imp_dict, fh, indent=2, sort_keys=True)

    figs = Path("notebooks/figs")
    figs.mkdir(parents=True, exist_ok=True)
    for col in X.columns:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Attempting to set identical low and high ylims",
            )
            disp = PartialDependenceDisplay.from_estimator(model, X, [col])
        disp.figure_.savefig(figs / f"pd_{col}.png")
        plt.close(disp.figure_)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a random forest using data/metrics.parquet and data/ratings_pooled.pkl. "
            "Run from the project root. Writes rf_importance.json to --output and partial "
            "dependence plots to notebooks/figs/."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=IMPORTANCE_PATH,
        help="Path to write rf_importance.json",
    )
    args = parser.parse_args(argv or [])
    run_rf(seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    main()
    