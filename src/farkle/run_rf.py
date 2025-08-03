# src/farkle/run_rf.py
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

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

# ---------------------------------------------------------------------------
# Constants for file and directory locations used in this module
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path("data")
METRICS_NAME = "metrics.parquet"
RATINGS_NAME = "ratings_pooled.pkl"
FIG_DIR = Path("notebooks/figs")


def plot_partial_dependence(model, X, column: str, out_dir: Path) -> Path:
    """Return a saved partial dependence plot for ``column``.

    Parameters
    ----------
    model : HistGradientBoostingRegressor
        Fitted model used to compute the plot.
    X : pd.DataFrame
        Feature matrix that ``model`` was trained on.
    column : str
        Name of the feature to analyse.
    out_dir : Path
        Directory in which to store the generated PNG.

    Returns
    -------
    Path
        Location of the saved plot.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ``PartialDependenceDisplay`` currently warns if integer dtypes are passed
    # for feature columns. Casting avoids the warning and future ``ValueError``
    # in scikit-learn 1.9.
    X = X.copy()
    X[column] = X[column].astype(float)
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


def run_rf(
    seed: int = 0,
    output_path: Path | None = None,
    root: Path = DEFAULT_ROOT,
) -> None:
    """Train the regressor and output feature importance and plots.

    Parameters
    ----------
    seed : int, optional
        Random seed for model fitting and permutation importance.
    output_path : Path | None, optional
        Location for the permutation importance JSON file.

    Reads
    -----
    ``<root>/metrics.parquet``
        Per-strategy feature metrics.
    ``<root>/ratings_pooled.pkl``
        Pickled mapping of strategy names to pooled ``(mu, sigma)`` tuples.

    Writes
    ------
    ``<root>/rf_importance.json``
        JSON file mapping metric names to permutation importance scores.
    ``notebooks/figs/pd_<feature>.png``
        Partial dependence plots for each metric.
    """
    root = Path(root)
    metrics_path = root / METRICS_NAME
    ratings_path = root / RATINGS_NAME

    metrics = pd.read_parquet(metrics_path)
    with open(ratings_path, "rb") as fh:
        ratings = pickle.load(fh)
    rating_df = pd.DataFrame({"strategy": list(ratings), "mu": [v.mu for v in ratings.values()]})
    data = metrics.merge(rating_df, on="strategy", how="inner")
    features = data.drop(columns=["strategy", "mu"]).astype(float)
    target = data["mu"]

    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(features, target)

    perm_importance = permutation_importance(
        model, features, target, n_repeats=5, random_state=seed
    )
    if len(perm_importance["importances_mean"]) != len(features.columns):
        msg = (
            "Mismatch between number of features and permutation importances: "
            f"expected {len(features.columns)}, got {len(perm_importance['importances_mean'])}"
        )
        raise ValueError(msg)
    imp_dict = {
        c: float(s)
        for c, s in zip(features.columns, perm_importance["importances_mean"], strict=False)
    }
    if output_path is None:
        output_path = root / "rf_importance.json"

    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(imp_dict, fh, indent=2, sort_keys=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for col in features.columns:
        plot_partial_dependence(model, features, col, FIG_DIR)


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
        default=None,
        help="Path to write rf_importance.json",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv or [])
    run_rf(seed=args.seed, output_path=args.output, root=args.root)


if __name__ == "__main__":
    main()
