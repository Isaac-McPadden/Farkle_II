"""Train a gradient boosting model to analyse strategy metrics.

This script reads the feature metrics and pooled ratings, fits a
``HistGradientBoostingRegressor`` to predict strategy ``mu`` values, then writes
permutation feature importances and partial dependence plots.
"""

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

# ---------------------------------------------------------------------------
# Constants for file and directory locations used in this module
# ---------------------------------------------------------------------------
METRICS_PATH = Path("data/metrics.parquet")
RATINGS_PATH = Path("data/ratings_pooled.pkl")
FIG_DIR = Path("notebooks/figs")
IMPORTANCE_PATH = Path("data/rf_importance.json")


def plot_partial_dependence(model, X, column: str, out_dir: Path) -> Path:
    """Save the partial dependence plot for ``column`` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Attempting to set identical low and high ylims",
        )
        disp = PartialDependenceDisplay.from_estimator(model, X, [column])
    out_file = out_dir / f"pd_{column}.png"
    disp.figure_.savefig(out_file)
    plt.close(disp.figure_)
    return out_file


def run_rf(seed: int = 0, output_path: Path = IMPORTANCE_PATH) -> None:
    """Train the regressor and output feature importance and plots.

    Parameters
    ----------
    seed : int, optional
        Random seed for model fitting and permutation importance.

    Reads
    -----
    ``data/metrics.parquet``
        Per-strategy feature metrics.
    ``data/ratings_pooled.pkl``
        Pickled mapping of strategy names to pooled ``(mu, sigma)`` tuples.

    Writes
    ------
    ``data/rf_importance.json``
        JSON file mapping metric names to permutation importance scores.
    ``notebooks/figs/pd_<feature>.png``
        Partial dependence plots for each metric.
    """
    metrics = pd.read_parquet(METRICS_PATH)
    with open(RATINGS_PATH, "rb") as fh:
        ratings = pickle.load(fh)
    rating_df = pd.DataFrame({"strategy": list(ratings), "mu": [v[0] for v in ratings.values()]})
    data = metrics.merge(rating_df, on="strategy", how="inner")
    X = data.drop(columns=["strategy", "mu"])
    X = X.astype(float)
    y = data["mu"]

    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(X, y)

    perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=seed)
    if len(perm_importance["importances_mean"]) != len(X.columns):
        msg = (
            "Mismatch between number of features and permutation importances: "
            f"expected {len(X.columns)}, got {len(imp['importances_mean'])}"
        )
        raise ValueError(msg)
    imp_dict = {c: float(s) for c, s in zip(X.columns, perm_importance["importances_mean"], strict=False)}
    IMPORTANCE_PATH.parent.mkdir(exist_ok=True)
    with IMPORTANCE_PATH.open("w") as fh:
        json.dump(imp_dict, fh, indent=2, sort_keys=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for col in X.columns:
        plot_partial_dependence(model, X, col, FIG_DIR)


def main(argv: List[str] | None = None) -> None:
    """Entry point for ``python -m farkle.run_rf``.

    Parameters
    ----------
    argv : List[str] | None, optional
        Command line arguments, or ``None`` to use ``sys.argv``. Only a single
        ``--seed`` option is accepted.
    """

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
