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
from types import SimpleNamespace
from typing import Any, Protocol, cast

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


try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import PartialDependenceDisplay
    from sklearn.inspection import permutation_importance as _sklearn_permutation_importance
    from sklearn.model_selection import GroupKFold
    from sklearn.utils import Bunch
except ModuleNotFoundError:  # pragma: no cover - handled at runtime

    class _HistGradientBoostingRegressor:  # type: ignore[override]
        """Fallback regressor that predicts the mean target."""

        def __init__(self, random_state: int | None = None):
            self.random_state = random_state
            self._mean: float | None = None

        def fit(self, _X, y):  # noqa: D401 - sklearn compatibility signature
            """Fit by recording the mean of ``y`` for later predictions."""
            self._mean = float(np.mean(y)) if len(y) else 0.0
            return self

        def predict(self, X):
            """Return constant predictions based on the fitted mean."""
            if self._mean is None:
                return np.zeros(len(X), dtype=float)
            return np.full(len(X), self._mean, dtype=float)

    class _Bunch(dict[str, Any]):
        """Lightweight substitute for :class:`sklearn.utils.Bunch`."""

        importances_mean: np.ndarray
        importances_std: np.ndarray

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.__dict__ = self

    Bunch = _Bunch

    def permutation_importance(  # type: ignore[override]
        estimator: Any,
        X,
        y,
        *,
        scoring: Any = None,
        n_repeats: int = 5,
        n_jobs: int | None = None,
        random_state: Any = None,
        sample_weight: Any = None,
        max_samples: float = 1,
    ) -> Bunch | dict[str, Bunch]:  # pyright: ignore[reportInvalidTypeForm]
        """Fallback permutation importance returning zeroed importances."""
        _ = estimator, X, y, scoring, n_repeats, n_jobs, random_state, sample_weight, max_samples
        n_features = X.shape[1]
        zeros = np.zeros(n_features, dtype=float)
        return Bunch(importances_mean=zeros, importances_std=zeros)

    class _GroupKFold:  # type: ignore[override]
        """Minimal replacement for sklearn's grouped cross-validation splitter."""

        def __init__(self, n_splits: int):
            self.n_splits = max(2, int(n_splits))

        def split(self, X, y, groups):  # noqa: D401 - sklearn compatibility signature
            """Yield train/test indices while grouping by the provided labels."""
            _ = X, y
            unique_groups = list(dict.fromkeys(groups))
            for grp in unique_groups:
                test_idx = np.where(groups == grp)[0]
                train_idx = np.where(groups != grp)[0]
                if train_idx.size == 0 or test_idx.size == 0:
                    continue
                yield train_idx, test_idx

    HistGradientBoostingRegressor = _HistGradientBoostingRegressor
    GroupKFold = _GroupKFold

    class _FallbackPD:
        """Shim for partial dependence plotting when sklearn is unavailable."""

        @staticmethod
        def from_estimator(model, X, features):
            """Provide an object exposing an empty matplotlib-like figure."""
            _ = model, X, features

            class _Fig:
                """Placeholder figure that writes an empty file to ``path``."""

                def savefig(self, path, format="png") -> None:  # noqa: A003 - match matplotlib API
                    """Create an empty placeholder artifact at ``path``."""
                    _ = format
                    Path(path).write_bytes(b"")

            return SimpleNamespace(figure_=_Fig())

    PartialDependenceDisplay = _FallbackPD  # type: ignore[assignment]
else:
    permutation_importance = _sklearn_permutation_importance

# ---------------------------------------------------------------------------
# Constants for file and directory locations used in this module
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path("results_seed_0")
METRICS_NAME = "metrics.parquet"
RATINGS_NAME = "ratings_pooled.parquet"
FIG_DIR = Path("notebooks/figs")
MAX_PD_PLOTS = 30
IMPORTANCE_TEMPLATE = "feature_importance_{players}p.parquet"

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

    if PartialDependenceDisplay is None:
        raise ModuleNotFoundError("scikit-learn is required for partial dependence plots")

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

    ratings_table = pq.read_table(ratings_path, columns=["strategy", "mu"])
    rating_df = ratings_table.to_pandas()
    rating_df["strategy"] = rating_df["strategy"].astype(str)

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

    data["players"] = data["players"].fillna(0).astype(int)

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
    seed_targets = _load_seed_targets(root)

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
        target = subset["mu"].astype(np.float32)

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
        )  # sklearn always provides this, and so does our fallback

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

        _run_grouped_cv(
            players=players,
            subset=subset[["strategy", *feature_cols]],
            feature_cols=feature_cols,
            seed_targets=seed_targets,
            random_state=seed,
        )

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
