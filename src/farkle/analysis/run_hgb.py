# src/farkle/analysis/run_hgb.py
"""Train a hist gradient boosting model to analyze strategy metrics.

This script reads the feature metrics and pooled ratings, fits a
``HistGradientBoostingRegressor`` to predict strategy ``mu`` values, then writes
permutation feature importances and partial dependence plots.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from farkle.utils.writer import atomic_path

# ---------------------------------------------------------------------------
# Constants for file and directory locations used in this module
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path("results_seed_0")
METRICS_NAME = "metrics.parquet"
RATINGS_NAME = "ratings_pooled.parquet"
FIG_DIR = Path("notebooks/figs")
MAX_PD_PLOTS = 30


LOGGER = logging.getLogger(__name__)
logger = LOGGER


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
    try:
        with atomic_path(str(out_file)) as tmp_path:
            disp.figure_.savefig(tmp_path, format="png")
    finally:
        plt.close(disp.figure_)
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
        Parquet table of pooled ratings with columns ``{strategy, mu, sigma}``.

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

    LOGGER.info(
        "HGB regression start",
        extra={
            "stage": "hgb",
            "root": str(root),
            "seed": seed,
            "metrics_path": str(metrics_path),
            "ratings_path": str(ratings_path),
        },
    )

    metrics = pd.read_parquet(metrics_path)
    ratings_table = pq.read_table(ratings_path, columns=["strategy", "mu"])
    rating_df = ratings_table.to_pandas()
    data = metrics.merge(rating_df, on="strategy", how="inner")
    if data.empty:
        LOGGER.warning(
            "HGB regression skipped: no overlapping strategies between metrics and ratings",
            extra={
                "stage": "hgb",
                "metrics_rows": len(metrics),
                "ratings_rows": len(rating_df),
            },
        )
        return

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(output_path)) as tmp_path, Path(tmp_path).open("w") as fh:
        json.dump(imp_dict, fh, indent=2, sort_keys=True)
    LOGGER.info(
        "HGB permutation importances written",
        extra={
            "stage": "hgb",
            "path": str(output_path),
            "features": len(features.columns),
        },
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(features.columns)
    if len(cols) > MAX_PD_PLOTS:
        LOGGER.warning(
            "Too many features for partial dependence",
            extra={
                "stage": "hgb",
                "features": len(cols),
                "max_plots": MAX_PD_PLOTS,
            },
        )
    for col in cols[:MAX_PD_PLOTS]:
        plot_partial_dependence(model, features, col, FIG_DIR)
    LOGGER.info(
        "HGB regression complete",
        extra={
            "stage": "hgb",
            "path": str(output_path),
            "plots": min(len(cols), MAX_PD_PLOTS),
        },
    )
