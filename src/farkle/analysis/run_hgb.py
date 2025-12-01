# src/farkle/analysis/run_hgb.py
"""Train histogram gradient boosting models for strategy feature analysis.

This module joins the curated metrics with pooled TrueSkill ratings, derives a
feature matrix from serialized strategy literals, and fits a
``HistGradientBoostingRegressor`` for each player-count bucket.  Deterministic
permutation importances are written to ``feature_importance_<Np>.parquet`` files,
and grouped cross-validation metrics are logged when per-seed ratings are
available.  Optional partial dependence plots are emitted for notebook
exploration.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.simulation.strategies import FavorDiceOrScore, parse_strategy_for_df
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.writer import atomic_path


class PermutationImportanceResult(Protocol):
    importances_mean: np.ndarray
    importances_std: np.ndarray

# ---------------------------------------------------------------------------
# Constants for file and directory locations used in this module
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path("results_seed_0")
METRICS_NAME = "metrics.parquet"
RATINGS_NAME = "ratings_pooled.parquet"
FIG_DIR = Path("notebooks/figs")
MAX_PD_PLOTS = 30
IMPORTANCE_TEMPLATE = "feature_importance_{players}p.parquet"
OVERALL_IMPORTANCE_NAME = "feature_importance_overall.parquet"

FEATURE_SPECS: list[tuple[str, str]] = [
    ("score_threshold", "float32"),
    ("dice_threshold", "float32"),
    ("consider_score", "float32"),
    ("consider_dice", "float32"),
    ("smart_five", "float32"),
    ("smart_one", "float32"),
    ("favor_score", "float32"),
    ("require_both", "float32"),
    ("auto_hot_dice", "float32"),
    ("run_up_score", "float32"),
]

_SEED_PATTERN = re.compile(r"ratings_pooled_seed(?P<seed>\d+)\.parquet$")


LOGGER = logging.getLogger(__name__)
logger = LOGGER


def _parse_strategy_features(strategies: pd.Series) -> pd.DataFrame:
    """Return a feature matrix indexed by strategy literal."""

    unique = pd.Index(pd.unique(strategies.dropna().astype(str)))
    rows: list[dict[str, float | str]] = []
    skipped: list[str] = []
    for strategy in unique:
        try:
            parsed = parse_strategy_for_df(strategy)
        except ValueError:
            skipped.append(strategy)
            continue
        row: dict[str, float | str] = {"strategy": strategy}
        row["score_threshold"] = float(parsed.get("score_threshold", np.nan))
        row["dice_threshold"] = float(parsed.get("dice_threshold", np.nan))
        row["consider_score"] = 1.0 if parsed.get("consider_score") else 0.0
        row["consider_dice"] = 1.0 if parsed.get("consider_dice") else 0.0
        row["smart_five"] = 1.0 if parsed.get("smart_five") else 0.0
        row["smart_one"] = 1.0 if parsed.get("smart_one") else 0.0
        favor = parsed.get("favor_dice_or_score")
        row["favor_score"] = 1.0 if favor == FavorDiceOrScore.SCORE else 0.0
        row["require_both"] = 1.0 if parsed.get("require_both") else 0.0
        row["auto_hot_dice"] = 1.0 if parsed.get("auto_hot_dice") else 0.0
        row["run_up_score"] = 1.0 if parsed.get("run_up_score") else 0.0
        rows.append(row)

    if skipped:
        LOGGER.warning(
            "Skipping unparseable strategies",
            extra={"stage": "hgb", "count": len(skipped)},
        )

    if not rows:
        columns = ["strategy"] + [name for name, _dtype in FEATURE_SPECS]
        return pd.DataFrame(columns=columns).set_index("strategy")

    features = pd.DataFrame(rows).set_index("strategy")
    for name, dtype in FEATURE_SPECS:
        if name not in features.columns:
            features[name] = 0.0
        np_dtype = np.dtype(dtype)
        features[name] = features[name].astype(np_dtype)
    return features[[name for name, _dtype in FEATURE_SPECS]]


def _load_seed_targets(root: Path) -> pd.DataFrame:
    """Load per-seed TrueSkill targets used for grouped CV."""
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("ratings_pooled_seed*.parquet")):
        match = _SEED_PATTERN.match(path.name)
        if match is None:
            continue
        seed_val = int(match.group("seed"))
        table = pq.read_table(path, columns=["strategy", "mu"])
        frame = table.to_pandas()
        frame["strategy"] = frame["strategy"].astype(str)
        frame["seed"] = seed_val
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["strategy", "mu", "seed"])
    combined = pd.concat(frames, ignore_index=True)
    combined["seed"] = combined["seed"].astype(int)
    return combined


def _write_importances(path: Path, frame: pd.DataFrame) -> None:
    """Write permutation importance results to parquet atomically."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    write_parquet_atomic(table, path)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error with an empty-guard."""
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination with zero-variance guard."""
    if y_true.size == 0:
        return 0.0
    mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean) ** 2))
    if ss_tot == 0.0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - (ss_res / ss_tot))


def _run_grouped_cv(
    *,
    players: int,
    subset: pd.DataFrame,
    feature_cols: list[str],
    seed_targets: pd.DataFrame,
    random_state: int,
) -> None:
    """Run grouped cross-validation when per-seed ratings are available."""

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    if seed_targets.empty:
        LOGGER.info(
            "Grouped CV skipped: no per-seed ratings",
            extra={"stage": "hgb", "players": players},
        )
        return

    cv_frame = seed_targets.merge(subset[["strategy", *feature_cols]], on="strategy", how="inner")
    cv_frame.dropna(subset=["mu", "seed"], inplace=True)
    if cv_frame.empty:
        LOGGER.info(
            "Grouped CV skipped: insufficient overlap",
            extra={"stage": "hgb", "players": players},
        )
        return

    seeds = pd.Index(cv_frame["seed"].unique())
    if len(seeds) < 2:
        LOGGER.info(
            "Grouped CV skipped: <2 unique seeds",
            extra={"stage": "hgb", "players": players, "seeds": seeds.tolist()},
        )
        return

    n_splits = min(5, len(seeds))
    if n_splits < 2:
        LOGGER.info(
            "Grouped CV skipped: insufficient splits",
            extra={"stage": "hgb", "players": players, "seeds": seeds.tolist()},
        )
        return

    groups = cv_frame["seed"].astype(int).to_numpy()
    X = cv_frame[feature_cols].to_numpy(dtype=np.float32)
    y = cv_frame["mu"].to_numpy(dtype=np.float32)

    splitter = GroupKFold(n_splits=n_splits)
    r2_scores: list[float] = []
    mae_scores: list[float] = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = HistGradientBoostingRegressor(random_state=random_state)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        r2_scores.append(_r2(y[test_idx], preds))
        mae_scores.append(_mae(y[test_idx], preds))

    if not r2_scores:
        LOGGER.info(
            "Grouped CV skipped: empty folds",
            extra={"stage": "hgb", "players": players},
        )
        return

    LOGGER.info(
        "Grouped CV metrics",
        extra={
            "stage": "hgb",
            "players": players,
            "splits": len(r2_scores),
            "r2_mean": float(np.mean(r2_scores)),
            "r2_std": float(np.std(r2_scores)),
            "mae_mean": float(np.mean(mae_scores)),
            "mae_std": float(np.std(mae_scores)),
        },
    )


def plot_partial_dependence(model, X, column: str, out_dir: Path) -> Path:
    """Return a saved partial dependence plot for ``column``."""

    from sklearn.inspection import PartialDependenceDisplay

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    """Train the regressor and output feature importance and plots."""

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    root = Path(root)
    metrics_path = root / METRICS_NAME

    LOGGER.info(
        "HGB regression start",
        extra={
            "stage": "hgb",
            "root": str(root),
            "seed": seed,
            "metrics_path": str(metrics_path),
        },
    )

    metrics = pd.read_parquet(metrics_path)
    metrics = metrics.copy()
    metrics["strategy"] = metrics["strategy"].astype(str)
    raw_players: set[int] = set()
    if "n_players" in metrics.columns:
        raw_players.update(int(p) for p in metrics["n_players"].dropna().astype(int).unique())
        metrics.rename(columns={"n_players": "players"}, inplace=True)
    if "players" not in metrics.columns:
        metrics["players"] = 0
    raw_players.update(int(p) for p in metrics["players"].dropna().astype(int).unique())
    metrics["players"] = metrics["players"].fillna(0).astype(int)

    if "win_rate" not in metrics.columns:
        raise ValueError("metrics parquet missing win_rate column required for HGB regression")

    data = metrics

    feature_frame = _parse_strategy_features(data["strategy"])
    if feature_frame.empty:
        LOGGER.warning(
            "HGB regression skipped: no strategies produced features",
            extra={"stage": "hgb"},
        )
        return

    feature_cols = [name for name, _dtype in FEATURE_SPECS]
    data = data.join(feature_frame, on="strategy", how="inner")
    data = data.drop_duplicates(subset=["strategy", "players"]).reset_index(drop=True)
    if data.empty:
        LOGGER.warning(
            "HGB regression skipped: no rows after feature join",
            extra={"stage": "hgb"},
        )
        return

    metrics_players = (
        sorted(raw_players) if raw_players else sorted({int(p) for p in data["players"].unique()})
    )
    importance_summary: dict[str, dict[str, float]] = {}
    collected_frames: list[pd.DataFrame] = []

    for players in metrics_players:
        subset = data[data["players"] == players].copy()
        subset["strategy"] = subset["strategy"].astype(str)
        importance_path = root / IMPORTANCE_TEMPLATE.format(players=players)

        if subset.empty:
            LOGGER.info(
                "No data for player bucket",
                extra={"stage": "hgb", "players": players},
            )
            empty_frame = pd.DataFrame(
                columns=["feature", "importance_mean", "importance_std", "players"],
                dtype=float,
            )
            _write_importances(importance_path, empty_frame)
            importance_summary[f"{players}p"] = {}
            continue

        features = subset[feature_cols].astype(np.float32)
        target = subset["win_rate"].astype(np.float32)

        if len(subset) < 2:
            LOGGER.warning(
                "Insufficient rows for training",
                extra={"stage": "hgb", "players": players, "rows": len(subset)},
            )
            empty_frame = pd.DataFrame(
                columns=["feature", "importance_mean", "importance_std", "players"],
                dtype=float,
            )
            _write_importances(importance_path, empty_frame)
            importance_summary[f"{players}p"] = {}
            continue

        model = HistGradientBoostingRegressor(random_state=seed)
        model.fit(features, target)

        perm_raw = permutation_importance(
            model,
            features,
            target,
            n_repeats=10,
            random_state=seed,
        )
        perm = cast(PermutationImportanceResult, perm_raw)
        importances_mean = perm.importances_mean
        importances_std = (
            perm.importances_std
        )

        if len(importances_mean) != len(features.columns):
            msg = (
                "Mismatch between number of features and permutation importances: "
                f"expected {len(features.columns)}, got {len(importances_mean)}"
            )
            raise ValueError(msg)

        importance_summary[f"{players}p"] = {
            col: float(val) for col, val in zip(feature_cols, importances_mean, strict=False)
        }

        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance_mean": importances_mean,
                "importance_std": importances_std,
                "players": players,
            }
        )

        _write_importances(importance_path, importance_df)
        collected_frames.append(importance_df)
        LOGGER.info(
            "Permutation importances written",
            extra={
                "stage": "hgb",
                "players": players,
                "path": str(importance_path),
            },
        )

        fig_dir = FIG_DIR / f"{players}p"
        fig_dir.mkdir(parents=True, exist_ok=True)
        cols = list(feature_cols)
        if len(cols) > MAX_PD_PLOTS:
            LOGGER.warning(
                "Too many features for partial dependence",
                extra={
                    "stage": "hgb",
                    "players": players,
                    "features": len(cols),
                    "max_plots": MAX_PD_PLOTS,
                },
            )
        for col in cols[:MAX_PD_PLOTS]:
            plot_partial_dependence(model, features, col, fig_dir)

    if collected_frames:
        overall_frame = pd.concat(collected_frames, ignore_index=True)
        grouped = (
            overall_frame.groupby("feature", as_index=False)
            .agg(
                importance_mean=("importance_mean", "mean"),
                importance_std=("importance_std", "mean"),
            )
            .assign(players="overall")
        )
        _write_importances(root / OVERALL_IMPORTANCE_NAME, grouped)
        importance_summary["overall"] = {
            str(row.feature): float(row.importance_mean)
            for row in grouped.itertuples(index=False)
        }

    if output_path is None:
        output_path = root / "hgb_importance.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(output_path)) as tmp_path, Path(tmp_path).open("w") as fh:
        json.dump(importance_summary, fh, indent=2, sort_keys=True)

    LOGGER.info(
        "HGB regression complete",
        extra={
            "stage": "hgb",
            "path": str(output_path),
            "player_buckets": sorted(importance_summary.keys()),
        },
    )
