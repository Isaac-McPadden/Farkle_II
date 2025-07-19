from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


def run_rf(seed: int = 0) -> None:
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
    Path("data").mkdir(exist_ok=True)
    with (Path("data") / "rf_importance.json").open("w") as fh:
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
    parser = argparse.ArgumentParser(description="Random forest analysis")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv or [])
    run_rf(seed=args.seed)


if __name__ == "__main__":
    main()
    