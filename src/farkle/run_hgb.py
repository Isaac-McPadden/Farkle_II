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
import re
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
RATINGS_NAME = "ratings_pooled.parquet"
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
    ``<root>/ratings_pooled.parquet``
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

    def _get_mu(v):
        if hasattr(v, "mu"):
            return v.mu
        if isinstance(v, dict) and "mu" in v:
            return v["mu"]
        if isinstance(v, (list, tuple)) and v:
            return v[0]
        raise TypeError("Unknown rating value type")

    rating_df = pd.DataFrame(
        {"strategy": list(ratings), "mu": [_get_mu(v) for v in ratings.values()]}
    )
    data = metrics.merge(rating_df, on="strategy", how="inner")

    # ------------------------------------------------------------------
    # Parse strategy parameters from the strategy name. This ensures that
    # the model only trains on configuration parameters and does not leak
    # outcome-based metrics into the feature set.
    # ------------------------------------------------------------------
    _RX = re.compile(
        r"^Strat\((?P<S>\d+),(?P<D>\d+)\)\[(?P<cs>[S-])(?P<cd>[D-])\]\[(?P<sf>[F-])(?P<so>[O-])(?P<fs>FS|FD)\]\[(?P<rb>AND|OR)\]\[(?P<hd>[H-])(?P<rs>[R-])\]$"
    )

    def _parse(name: str) -> dict[str, object]:
        m = _RX.match(name)
        if not m:
            return {}
        g = m.groupdict()
        return {
            "score_threshold": int(g["S"]),
            "dice_threshold": int(g["D"]),
            "consider_score": 1 if g["cs"] == "S" else 0,
            "consider_dice": 1 if g["cd"] == "D" else 0,
            "smart_five": 1 if g["sf"] == "F" else 0,
            "smart_one": 1 if g["so"] == "O" else 0,
            "favor_score": 1 if g["fs"] == "FS" else 0,  # 1=FavorScore, 0=FavorDice
            "require_both": 1 if g["rb"] == "AND" else 0,
            "auto_hot_dice": 1 if g["hd"] == "H" else 0,
            "run_up_score": 1 if g["rs"] == "R" else 0,
        }

    feat_df = pd.DataFrame([_parse(s) for s in data["strategy"]], index=data.index).fillna(0)
    feature_cols = [
        "score_threshold",
        "dice_threshold",
        "consider_score",
        "consider_dice",
        "smart_five",
        "smart_one",
        "favor_score",
        "require_both",
        "auto_hot_dice",
        "run_up_score",
    ]
    features = feat_df.reindex(columns=feature_cols).astype("float32")
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
            "data/ratings_pooled.parquet. Run from the project root. Writes "
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
