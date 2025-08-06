# src/farkle/run_hgb.py
"""Train a hist gradient boosting model to analyze strategy metrics.

This script reads the feature metrics and pooled ratings, fits a
``HistGradientBoostingRegressor`` to predict strategy ``mu`` values, then writes
permutation feature importances and partial dependence plots.
"""

from __future__ import annotations

import argparse
import json
import logging
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
DEFAULT_ROOT = Path("results_seed_0")
METRICS_NAME = "metrics.parquet"
RATINGS_NAME = "ratings_pooled.pkl"
FIG_DIR = Path("notebooks/figs")
MAX_PD_PLOTS = 30


logger = logging.getLogger(__name__)


def plot_partial_dependence(model, X, column: str, out_dir: Path) -> Path:
    """Return a saved partial dependence plot for ``column``.

    Parameters
    ----------
    model : HistGradientBoostingRegressor
        Fitted model used to compute the plot.
    X : pd.DataFrame
        Feature matrix that ``model`` was trained on.
    column : str
        Name of the feature to analyze.
    out_dir : Path
        Directory in which to store the generated PNG.

    Returns
    -------
    Path
        Location of the saved plot.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ``PartialDependenceDisplay`` warns if integer dtypes are passed for feature
    # columns. ``X`` should therefore be cast to ``float`` before calling this
    # function. Casting once outside this helper avoids an expensive copy for
    # each plotted feature.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Attempting to set identical low and high ylims",
        )
        disp = PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=[column],
        )
    out_file = out_dir / f"pd_{column}.png"
    tmp_file = out_file.with_suffix(".tmp")
    disp.figure_.savefig(tmp_file, format="png")
    plt.close(disp.figure_)
    tmp_file.replace(out_file)
    return out_file


def run_hgb(
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
    ``<root>/hgb_importance.json``
        JSON file mapping metric names to permutation importance scores.
    ``notebooks/figs/pd_<feature>.png``
        Partial dependence plots for up to ``MAX_PD_PLOTS`` metrics.
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
        output_path = root / "hgb_importance.json"

    output_path.parent.mkdir(exist_ok=True)
    tmp_output = output_path.with_suffix(".tmp")
    try:
        with tmp_output.open("w") as fh:
            json.dump(imp_dict, fh, indent=2, sort_keys=True)
        tmp_output.replace(output_path)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(features.columns)
    if len(cols) > MAX_PD_PLOTS:
        logger.warning(
            "More than %d features provided (%d); only plotting the first %d",
            MAX_PD_PLOTS,
            len(cols),
            MAX_PD_PLOTS,
        )
    for col in cols[:MAX_PD_PLOTS]:
        plot_partial_dependence(model, features, col, FIG_DIR)


def main(argv: List[str] | None = None) -> None:
    """Entry point for ``python -m farkle.run_hgb``.

    Parameters
    ----------
    argv : List[str] | None, optional
        Command line arguments, or ``None`` to use ``sys.argv``. Only a single
        ``--seed`` option is accepted.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train a HistGradientBoostingRegressor using data/metrics.parquet and "
            "data/ratings_pooled.pkl. Run from the project root. Writes "
            "hgb_importance.json to --output and partial dependence plots to "
            "notebooks/figs/."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to write hgb_importance.json",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv or [])
    run_hgb(seed=args.seed, output_path=args.output, root=args.root)


if __name__ == "__main__":
    main()
