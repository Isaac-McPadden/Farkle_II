# src/farkle/analysis/run_hgb.py
"""Out-of-sample HGB exploration of strategy-option associations.

Every predictive score and permutation importance is evaluated on strategy
configurations excluded from model fitting.  Full-grid fits are used only for
exploratory response plots and to draft candidates for a future simulation;
they never add strategies to the current analysis population.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from farkle.simulation.strategies import (
    FavorDiceOrScore,
    ThresholdStrategy,
    coerce_strategy_ids,
    parse_strategy_for_df,
    strategy_attributes_from_series,
)
from farkle.utils.artifact_contract import make_artifact_sidecar
from farkle.utils.artifacts import (
    write_json_artifact_atomic,
    write_parquet_artifact_atomic,
)
from farkle.utils.random import RandomPurpose, coordinate_rng, coordinate_seed

if TYPE_CHECKING:
    from farkle.config import AppConfig


class PermutationImportanceResult(Protocol):
    """Protocol describing the sklearn permutation-importance payload.

    Attributes:
        importances_mean: Mean importance value for each feature column.
        importances_std: Standard deviation of the sampled importances.
    """

    importances_mean: np.ndarray
    importances_std: np.ndarray


IMPORTANCE_TEMPLATE = "feature_importance_{players}p.parquet"
PREDICTIVE_SCORES_TEMPLATE = "heldout_predictive_scores_{players}p.parquet"
FOLD_METRICS_TEMPLATE = "heldout_fold_metrics_{players}p.parquet"
OVERALL_IMPORTANCE_NAME = "feature_importance_overall.parquet"
LONG_IMPORTANCE_NAME = "feature_importance_long.parquet"
FUTURE_PROPOSALS_NAME = "future_simulation_proposals.parquet"
DEFAULT_HELDOUT_FOLDS = 5
DEFAULT_PERMUTATION_REPEATS = 10
DEFAULT_PROPOSAL_LIMIT = 100

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

LOGGER = logging.getLogger(__name__)
logger = LOGGER


@dataclass(frozen=True)
class HeldoutEvaluation:
    """Out-of-sample artifacts for one player-count grid."""

    importance: pd.DataFrame
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame


def _parse_strategy_features(
    strategies: pd.Series, *, manifest: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return a feature matrix indexed by strategy identifier."""

    unique = pd.Series(pd.unique(strategies.dropna()))
    if unique.empty:
        columns = ["strategy"] + [name for name, _dtype in FEATURE_SPECS]
        return pd.DataFrame(columns=columns).set_index("strategy")

    def _safe_parse(value: str) -> dict:
        """Parse one legacy strategy literal, swallowing unsupported variants."""
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


def _write_hgb_frame(
    path: Path,
    frame: pd.DataFrame,
    *,
    cfg: AppConfig,
    source_artifacts: Sequence[Path],
    players: Sequence[int],
    scope: str,
    operation: str,
    conditioning: str,
    grouping_keys: Sequence[str],
) -> None:
    """Write one canonical HGB artifact with its required sidecar."""

    table = pa.Table.from_pandas(frame, preserve_index=False)
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="hgb",
        scope=scope,
        source_scope="by_k",
        operation=operation,
        weighted_quantity="win_rate",
        k_aggregation_method="equal_k" if operation == "equal_k_mean" else "none",
        support_count_role="finite_strategy_grid_support",
        uncertainty_method="heldout_strategy_configuration_folds",
        replication_unit="strategy_configuration",
        conditioning=conditioning,
        consistency_columns=table.schema.names,
        source_artifacts=source_artifacts,
        grouping_keys=grouping_keys,
        player_counts=players,
        required_player_counts=players,
        missing_cell_policy="fail",
        seed_scope="single_root",
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar, codec=cfg.parquet_codec)


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


def _model_seed(root_seed: int, players: int, fold: int) -> int:
    """Return a stable sklearn-compatible seed for one HGB fit."""

    return coordinate_seed(
        RandomPurpose.HGB,
        root_seed=root_seed,
        k=players,
        replicate_index=fold,
        dtype=np.uint32,
    )


def _empty_heldout_evaluation(players: int, root_seed: int) -> HeldoutEvaluation:
    """Return typed empty frames when a finite grid cannot be split."""

    importance = pd.DataFrame(
        columns=[
            "feature",
            "association_importance_mean",
            "association_importance_fold_std",
            "association_importance_repeat_std_mean",
            "players",
            "root_seed",
            "heldout_folds",
            "finite_grid_support",
            "interpretation",
        ]
    )
    predictions = pd.DataFrame(
        columns=[
            "strategy",
            "players",
            "root_seed",
            "fold",
            "observed_win_rate",
            "predicted_win_rate",
            "residual",
            "finite_grid_support",
        ]
    )
    fold_metrics = pd.DataFrame(
        columns=[
            "players",
            "root_seed",
            "fold",
            "train_strategies",
            "heldout_strategies",
            "mae",
            "r2",
            "finite_grid_support",
        ]
    )
    for frame in (importance, predictions, fold_metrics):
        frame.attrs.update(players=players, root_seed=root_seed)
    return HeldoutEvaluation(importance, predictions, fold_metrics)


def _heldout_strategy_evaluation(
    *,
    players: int,
    subset: pd.DataFrame,
    feature_cols: list[str],
    root_seed: int,
    requested_folds: int = DEFAULT_HELDOUT_FOLDS,
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    max_depth: int | None = None,
    max_iter: int = 300,
) -> HeldoutEvaluation:
    """Evaluate prediction and importance on excluded strategy configurations."""

    support = len(subset)
    if support < 2:
        return _empty_heldout_evaluation(players, root_seed)
    if requested_folds < 2:
        raise ValueError("HGB heldout_folds must be at least 2")
    if permutation_repeats < 1:
        raise ValueError("HGB permutation_repeats must be positive")

    folds = min(requested_folds, support)
    ordered = subset.assign(_strategy_sort=subset["strategy"].astype(str)).sort_values(
        "_strategy_sort", kind="mergesort"
    )
    ordered = ordered.drop(columns="_strategy_sort").reset_index(drop=True)
    rng = coordinate_rng(RandomPurpose.HGB, root_seed=root_seed, k=players)
    shuffled_positions = rng.permutation(support)
    fold_ids = np.empty(support, dtype=np.int64)
    fold_ids[shuffled_positions] = np.arange(support, dtype=np.int64) % folds

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, int | float]] = []
    fold_importance_means: list[np.ndarray] = []
    fold_importance_repeat_stds: list[np.ndarray] = []
    all_features = ordered[feature_cols].astype(np.float32)
    all_targets = ordered["win_rate"].to_numpy(dtype=np.float32)

    for fold in range(folds):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        model_seed = _model_seed(root_seed, players, fold + 1)
        model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            max_iter=max_iter,
            random_state=model_seed,
        )
        model.fit(all_features.loc[train_mask], all_targets[train_mask])
        heldout_features = all_features.loc[test_mask]
        heldout_targets = all_targets[test_mask]
        predicted = np.asarray(model.predict(heldout_features), dtype=float)

        perm_raw = permutation_importance(
            model,
            heldout_features,
            heldout_targets,
            n_repeats=permutation_repeats,
            random_state=_model_seed(root_seed, players, folds + fold + 1),
            scoring="neg_mean_absolute_error",
        )
        perm = cast(PermutationImportanceResult, perm_raw)
        if len(perm.importances_mean) != len(feature_cols):
            raise ValueError(
                "Mismatch between number of features and permutation importances: "
                f"expected {len(feature_cols)}, got {len(perm.importances_mean)}"
            )
        fold_importance_means.append(np.asarray(perm.importances_mean, dtype=float))
        fold_importance_repeat_stds.append(np.asarray(perm.importances_std, dtype=float))

        heldout = ordered.loc[test_mask, ["strategy"]]
        for strategy, observed, estimate in zip(
            heldout["strategy"], heldout_targets, predicted, strict=True
        ):
            prediction_rows.append(
                {
                    "strategy": strategy,
                    "players": players,
                    "root_seed": root_seed,
                    "fold": fold,
                    "observed_win_rate": float(observed),
                    "predicted_win_rate": float(estimate),
                    "residual": float(observed - estimate),
                    "finite_grid_support": support,
                }
            )
        metric_rows.append(
            {
                "players": players,
                "root_seed": root_seed,
                "fold": fold,
                "train_strategies": int(train_mask.sum()),
                "heldout_strategies": int(test_mask.sum()),
                "mae": _mae(heldout_targets, predicted),
                "r2": _r2(heldout_targets, predicted),
                "finite_grid_support": support,
            }
        )

    means = np.vstack(fold_importance_means)
    repeat_stds = np.vstack(fold_importance_repeat_stds)
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "association_importance_mean": means.mean(axis=0),
            "association_importance_fold_std": (
                means.std(axis=0, ddof=1) if folds > 1 else np.zeros(len(feature_cols))
            ),
            "association_importance_repeat_std_mean": repeat_stds.mean(axis=0),
            "players": players,
            "root_seed": root_seed,
            "heldout_folds": folds,
            "finite_grid_support": support,
            "interpretation": "predictive_association_not_causal",
        }
    )
    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(metric_rows)
    return HeldoutEvaluation(importance, predictions, fold_metrics)


def _proposal_from_features(values: pd.Series) -> ThresholdStrategy | None:
    """Build a valid future candidate from one HGB feature row."""

    try:
        return ThresholdStrategy(
            score_threshold=int(values["score_threshold"]),
            dice_threshold=int(values["dice_threshold"]),
            consider_score=bool(values["consider_score"]),
            consider_dice=bool(values["consider_dice"]),
            smart_five=bool(values["smart_five"]),
            smart_one=bool(values["smart_one"]),
            favor_dice_or_score=(
                FavorDiceOrScore.SCORE if bool(values["favor_score"]) else FavorDiceOrScore.DICE
            ),
            require_both=bool(values["require_both"]),
            auto_hot_dice=bool(values["auto_hot_dice"]),
            run_up_score=bool(values["run_up_score"]),
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _future_strategy_proposals(
    *,
    players: int,
    features: pd.DataFrame,
    model: HistGradientBoostingRegressor,
    limit: int,
) -> pd.DataFrame:
    """Draft valid one-option mutations for a later simulation manifest."""

    columns = [
        "proposal_id",
        "strategy_id",
        "strategy_str",
        "players",
        *features.columns,
        "predicted_win_rate",
        "finite_grid_support",
        "proposal_status",
        "included_in_current_analysis",
    ]
    if limit < 1 or features.empty:
        return pd.DataFrame(columns=columns)

    observed = {tuple(float(value) for value in row) for row in features.to_numpy(dtype=np.float32)}
    levels = {
        column: sorted(float(value) for value in features[column].dropna().unique())
        for column in features.columns
    }
    fitted = np.asarray(model.predict(features), dtype=float)
    base_positions = np.argsort(-fitted, kind="stable")[: min(20, len(features))]
    candidates: dict[tuple[float, ...], ThresholdStrategy] = {}
    for position in base_positions:
        base = features.iloc[int(position)].copy()
        for column in features.columns:
            for level in levels[column]:
                if float(base[column]) == level:
                    continue
                candidate = base.copy()
                candidate[column] = level
                key = tuple(float(candidate[name]) for name in features.columns)
                if key in observed or key in candidates:
                    continue
                strategy = _proposal_from_features(candidate)
                if strategy is not None:
                    candidates[key] = strategy

    if not candidates:
        return pd.DataFrame(columns=columns)
    candidate_keys = sorted(candidates)
    candidate_features = pd.DataFrame(candidate_keys, columns=features.columns, dtype=np.float32)
    estimates = np.asarray(model.predict(candidate_features), dtype=float)
    order = np.argsort(-estimates, kind="stable")[:limit]
    rows: list[dict[str, object]] = []
    for position in order:
        key = candidate_keys[int(position)]
        strategy = candidates[key]
        digest_input = f"{players}|" + "|".join(f"{value:.9g}" for value in key)
        row: dict[str, object] = {
            "proposal_id": f"hgb-{sha256(digest_input.encode()).hexdigest()[:16]}",
            "strategy_id": None,
            "strategy_str": str(strategy),
            "players": players,
            "predicted_win_rate": float(estimates[int(position)]),
            "finite_grid_support": len(features),
            "proposal_status": "future_simulation_only",
            "included_in_current_analysis": False,
        }
        row.update(dict(zip(features.columns, key, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def run_hgb(
    *,
    cfg: AppConfig,
    metrics_paths: Sequence[Path],
    manifest_path: Path | None = None,
) -> None:
    """Evaluate canonical performance cells and draft future-only candidates."""

    root = cfg.hgb_stage_dir
    root.mkdir(parents=True, exist_ok=True)
    combined_dir = cfg.across_k_dir("hgb")
    combined_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.sim.seed
    heldout_folds = cfg.hgb.heldout_folds
    permutation_repeats = cfg.hgb.permutation_repeats
    max_depth = cfg.hgb.max_depth
    max_iter = cfg.hgb.n_estimators
    proposal_limit = cfg.hgb.future_proposal_limit
    manifest = None
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = pd.read_parquet(manifest_path)

    LOGGER.info(
        "HGB regression start",
        extra={
            "stage": "hgb",
            "root": str(root),
            "seed": seed,
            "metrics_paths": [str(path) for path in metrics_paths],
            "evaluation": "heldout_strategy_configurations",
        },
    )

    source_metrics = [Path(path) for path in metrics_paths]
    if not source_metrics:
        raise ValueError("HGB requires canonical per-k performance artifacts")
    metrics = pd.concat([pd.read_parquet(path) for path in source_metrics], ignore_index=True)
    metrics = metrics.copy()
    if "strategy" in metrics.columns:
        metrics["strategy"] = coerce_strategy_ids(metrics["strategy"])
    raw_players: set[int] = set()
    if "n_players" in metrics.columns:
        raw_players.update(int(p) for p in metrics["n_players"].dropna().astype(int).unique())
        metrics.rename(columns={"n_players": "players"}, inplace=True)
    if "k" in metrics.columns:
        raw_players.update(int(p) for p in metrics["k"].dropna().astype(int).unique())
        if "players" not in metrics.columns:
            metrics.rename(columns={"k": "players"}, inplace=True)
    if "players" not in metrics.columns:
        metrics["players"] = 0
    raw_players.update(int(p) for p in metrics["players"].dropna().astype(int).unique())
    metrics["players"] = metrics["players"].fillna(0).astype(int)

    if "win_rate" not in metrics.columns:
        raise ValueError("metrics parquet missing win_rate column required for HGB regression")

    data = metrics

    feature_frame = _parse_strategy_features(data["strategy"], manifest=manifest)
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
    collected_proposals: list[pd.DataFrame] = []

    for players in metrics_players:
        subset = data[data["players"] == players].copy()
        subset["strategy"] = coerce_strategy_ids(subset["strategy"])
        per_player_dir = cfg.hgb_per_k_dir(players)
        per_player_dir.mkdir(parents=True, exist_ok=True)
        importance_path = per_player_dir / IMPORTANCE_TEMPLATE.format(players=players)
        predictions_path = per_player_dir / PREDICTIVE_SCORES_TEMPLATE.format(players=players)
        fold_metrics_path = per_player_dir / FOLD_METRICS_TEMPLATE.format(players=players)

        if subset.empty:
            raise ValueError(
                f"HGB requires finite-grid performance rows for configured k={players}"
            )

        features = subset[feature_cols].astype(np.float32)
        target = subset["win_rate"].astype(np.float32)

        if len(subset) < 2:
            raise ValueError(
                "HGB held-out evaluation requires at least two strategy "
                f"configurations for k={players}; found {len(subset)}"
            )

        heldout = _heldout_strategy_evaluation(
            players=players,
            subset=subset,
            feature_cols=feature_cols,
            root_seed=seed,
            requested_folds=heldout_folds,
            permutation_repeats=permutation_repeats,
            max_depth=max_depth,
            max_iter=max_iter,
        )
        importance_df = heldout.importance
        _write_hgb_frame(
            importance_path,
            importance_df,
            cfg=cfg,
            source_artifacts=source_metrics,
            players=[players],
            scope="by_k",
            operation="heldout_permutation_importance",
            conditioning="finite_grid_predictive_association_not_causal",
            grouping_keys=["players", "feature"],
        )
        _write_hgb_frame(
            predictions_path,
            heldout.predictions,
            cfg=cfg,
            source_artifacts=source_metrics,
            players=[players],
            scope="by_k",
            operation="heldout_prediction",
            conditioning="finite_strategy_grid",
            grouping_keys=["players", "fold", "strategy"],
        )
        _write_hgb_frame(
            fold_metrics_path,
            heldout.fold_metrics,
            cfg=cfg,
            source_artifacts=source_metrics,
            players=[players],
            scope="by_k",
            operation="heldout_fold_diagnostics",
            conditioning="finite_strategy_grid",
            grouping_keys=["players", "fold"],
        )
        association_values = importance_df.set_index("feature")["association_importance_mean"]
        importance_summary[f"{players}p"] = {
            str(feature): float(value) for feature, value in association_values.items()
        }
        collected_frames.append(importance_df)
        LOGGER.info(
            "Held-out predictive associations written",
            extra={
                "stage": "hgb",
                "players": players,
                "path": str(importance_path),
                "folds": int(importance_df["heldout_folds"].iloc[0]),
                "finite_grid_support": len(subset),
            },
        )

        full_model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            max_iter=max_iter,
            random_state=_model_seed(seed, players, 0),
        )
        full_model.fit(features, target)
        proposals = _future_strategy_proposals(
            players=players,
            features=features,
            model=full_model,
            limit=proposal_limit,
        )
        if not proposals.empty:
            collected_proposals.append(proposals)

    if collected_frames:
        overall_frame = pd.concat(collected_frames, ignore_index=True)
        grouped = (
            overall_frame.groupby("feature", as_index=False)
            .agg(
                association_importance_mean=("association_importance_mean", "mean"),
                association_importance_fold_std=("association_importance_fold_std", "mean"),
                association_importance_repeat_std_mean=(
                    "association_importance_repeat_std_mean",
                    "mean",
                ),
                finite_grid_support=("finite_grid_support", "sum"),
            )
            .assign(players="overall")
        )
        grouped = grouped.astype(
            {
                "association_importance_mean": "float",
                "association_importance_fold_std": "float",
            }
        )
        grouped["interpretation"] = "predictive_association_not_causal"
        _write_hgb_frame(
            combined_dir / LONG_IMPORTANCE_NAME,
            overall_frame,
            cfg=cfg,
            source_artifacts=source_metrics,
            players=metrics_players,
            scope="concat_ks",
            operation="concatenate",
            conditioning="finite_grid_predictive_association_not_causal",
            grouping_keys=["players", "feature"],
        )
        _write_hgb_frame(
            combined_dir / OVERALL_IMPORTANCE_NAME,
            grouped,
            cfg=cfg,
            source_artifacts=source_metrics,
            players=metrics_players,
            scope="across_k",
            operation="equal_k_mean",
            conditioning="finite_grid_predictive_association_not_causal",
            grouping_keys=["feature"],
        )
        overall_series = grouped.set_index("feature")["association_importance_mean"]
        importance_summary["overall"] = {str(k): float(v) for k, v in overall_series.items()}

    proposals_path = cfg.hgb_future_proposals_path()
    proposal_frame = (
        pd.concat(collected_proposals, ignore_index=True)
        if collected_proposals
        else pd.DataFrame(
            columns=[
                "proposal_id",
                "strategy_id",
                "strategy_str",
                "players",
                *feature_cols,
                "predicted_win_rate",
                "finite_grid_support",
                "proposal_status",
                "included_in_current_analysis",
            ]
        )
    )
    _write_hgb_frame(
        proposals_path,
        proposal_frame,
        cfg=cfg,
        source_artifacts=source_metrics,
        players=metrics_players,
        scope="across_k",
        operation="future_candidate_generation",
        conditioning="future_simulation_only_not_current_analysis",
        grouping_keys=["players", "proposal_id"],
    )

    output_path = combined_dir / "hgb_importance.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = make_artifact_sidecar(
        cfg,
        output_path,
        producer="hgb",
        scope="across_k",
        source_scope="by_k",
        operation="equal_k_mean",
        weighted_quantity="heldout_permutation_association_importance",
        k_aggregation_method="equal_k",
        support_count_role="finite_strategy_grid_support",
        uncertainty_method="heldout_strategy_configuration_folds",
        replication_unit="strategy_configuration",
        conditioning="predictive_association_not_causal",
        consistency_columns=sorted(importance_summary),
        source_artifacts=source_metrics,
        grouping_keys=["feature"],
        player_counts=metrics_players,
        required_player_counts=metrics_players,
        missing_cell_policy="fail",
        seed_scope="single_root",
    )
    write_json_artifact_atomic(importance_summary, output_path, sidecar=sidecar)

    LOGGER.info(
        "HGB regression complete",
        extra={
            "stage": "hgb",
            "path": str(output_path),
            "player_buckets": sorted(importance_summary.keys()),
            "proposal_manifest": str(proposals_path),
            "interpretation": "predictive_association_not_causal",
        },
    )
