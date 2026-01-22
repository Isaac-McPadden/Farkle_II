# src/farkle/analysis/run_hgb.py
"""Train histogram gradient boosting models for strategy feature analysis.

This module joins the curated metrics with pooled TrueSkill ratings, derives a
feature matrix from serialized strategy literals, and fits a
``HistGradientBoostingRegressor`` for each player-count bucket. Deterministic
permutation importances are written to ``feature_importance_<Np>.parquet`` files
under a stage-specific directory tree, and grouped cross-validation metrics are
logged when per-seed ratings are available. Optional partial dependence plots
are emitted alongside the per-player artifacts for offline exploration.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Protocol, cast

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from farkle.simulation.strategies import (
    FavorDiceOrScore,
    coerce_strategy_ids,
    parse_strategy_for_df,
    strategy_attributes_from_series,
)
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.writer import atomic_path


class PermutationImportanceResult(Protocol):
    importances_mean: np.ndarray
    importances_std: np.ndarray

# ---------------------------------------------------------------------------
# Constants for file and directory locations used in this module
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path("results_seed_0") / "analysis" / "11_hgb"
METRICS_NAME = "metrics.parquet"
RATINGS_NAME = "ratings_pooled.parquet"
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


def _select_partial_dependence_features(
    features: pd.DataFrame, *, tolerance: float = 1e-6
) -> tuple[list[str], list[str]]:
    """Return columns eligible for partial dependence plots.

    Columns with constant or near-constant ranges (within ``tolerance``) are
    skipped to avoid degenerate plots.
    """

    kept: list[str] = []
    skipped: list[str] = []
    for column in features.columns:
        values = features[column].to_numpy(dtype=np.float32)
        if values.size == 0:
            skipped.append(column)
            continue
        spread = float(np.nanmax(values) - np.nanmin(values))
        if np.isnan(spread) or spread <= tolerance:
            skipped.append(column)
            continue
        kept.append(column)

    return kept, skipped


def _parse_strategy_features(
    strategies: pd.Series, *, manifest: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return a feature matrix indexed by strategy identifier."""

    unique = pd.Series(pd.unique(strategies.dropna()))
    if unique.empty:
        columns = ["strategy"] + [name for name, _dtype in FEATURE_SPECS]
        return pd.DataFrame(columns=columns).set_index("strategy")

    def _safe_parse(value: str) -> dict:
        try:
            return parse_strategy_for_df(value)
        except ValueError:
            return {}

    attrs = strategy_attributes_from_series(unique, manifest=manifest, parse_legacy=_safe_parse)
    if attrs.empty:
        columns = ["strategy"] + [name for name, _dtype in FEATURE_SPECS]
        return pd.DataFrame(columns=columns).set_index("strategy")

    skipped = int(attrs.isna().all(axis=1).sum())
    if skipped:
        LOGGER.warning(
            "Skipping unparseable strategies",
            extra={"stage": "hgb", "count": skipped},
        )

    favor_raw = attrs["favor_dice_or_score"]
    favor_score = favor_raw.apply(
        lambda v: v == FavorDiceOrScore.SCORE or v == FavorDiceOrScore.SCORE.value
    )

    features = pd.DataFrame(
        {
            "strategy": unique,
            "score_threshold": pd.to_numeric(attrs["score_threshold"], errors="coerce"),
            "dice_threshold": pd.to_numeric(attrs["dice_threshold"], errors="coerce"),
            "consider_score": pd.to_numeric(attrs["consider_score"], errors="coerce").fillna(0.0),
            "consider_dice": pd.to_numeric(attrs["consider_dice"], errors="coerce").fillna(0.0),
            "smart_five": pd.to_numeric(attrs["smart_five"], errors="coerce").fillna(0.0),
            "smart_one": pd.to_numeric(attrs["smart_one"], errors="coerce").fillna(0.0),
            "favor_score": favor_score.astype(float),
            "require_both": pd.to_numeric(attrs["require_both"], errors="coerce").fillna(0.0),
            "auto_hot_dice": pd.to_numeric(attrs["auto_hot_dice"], errors="coerce").fillna(0.0),
            "run_up_score": pd.to_numeric(attrs["run_up_score"], errors="coerce").fillna(0.0),
        }
    ).set_index("strategy")

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
        frame["strategy"] = coerce_strategy_ids(frame["strategy"])
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
    *,
    metrics_path: Path | None = None,
    ratings_path: Path | None = None,
    manifest_path: Path | None = None,
) -> None:
    """Train the regressor and output feature importance and plots."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pooled_dir = root / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(metrics_path) if metrics_path is not None else root / METRICS_NAME
    ratings_path = Path(ratings_path) if ratings_path is not None else root / RATINGS_NAME
    manifest = None
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = pd.read_parquet(manifest_path)

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
    if "strategy" in metrics.columns:
        metrics["strategy"] = coerce_strategy_ids(metrics["strategy"])
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

    feature_frame = _parse_strategy_features(data["strategy"], manifest=manifest)
    seed_targets = _load_seed_targets(ratings_path.parent)
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
        subset["strategy"] = coerce_strategy_ids(subset["strategy"])
        per_player_dir = root / f"{players}p"
        per_player_dir.mkdir(parents=True, exist_ok=True)
        importance_path = per_player_dir / IMPORTANCE_TEMPLATE.format(players=players)

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

        _run_grouped_cv(
            players=players,
            subset=subset,
            feature_cols=feature_cols,
            seed_targets=seed_targets,
            random_state=seed,
        )

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

        fig_dir = per_player_dir / "plots"
        fig_dir.mkdir(parents=True, exist_ok=True)
        cols = list(feature_cols)
        pd_cols, skipped_cols = _select_partial_dependence_features(
            features[cols]
        )
        if skipped_cols:
            LOGGER.info(
                "Skipping near-constant features for partial dependence",
                extra={
                    "stage": "hgb",
                    "players": players,
                    "skipped_features": skipped_cols,
                },
            )
        if len(pd_cols) > MAX_PD_PLOTS:
            LOGGER.warning(
                "Too many features for partial dependence",
                extra={
                    "stage": "hgb",
                    "players": players,
                    "features": len(pd_cols),
                    "max_plots": MAX_PD_PLOTS,
                },
            )
        for col in pd_cols[:MAX_PD_PLOTS]:
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
        grouped = grouped.astype(
            {"importance_mean": "float", "importance_std": "float"}
        )
        _write_importances(pooled_dir / OVERALL_IMPORTANCE_NAME, grouped)
        overall_series = grouped.set_index("feature")["importance_mean"]
        importance_summary["overall"] = {
            str(k): float(v) for k, v in overall_series.items()
        }

    if output_path is None:
        output_path = pooled_dir / "hgb_importance.json"

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
