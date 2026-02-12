# src/farkle/analysis/seed_summaries.py
"""Build per-seed, per-player summaries with Wilson confidence intervals.

The summaries are derived strictly from the resolved metrics parquet written by
the :mod:`farkle.analysis.metrics` stage.  Each summary file contains the union
of strategies observed for a given ``(seed, players)`` pair; no interpolation is
performed for missing combinations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa

from farkle.analysis import stage_logger
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.simulation.strategies import coerce_strategy_ids
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.stats import wilson_ci

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"strategy", "n_players", "games", "wins"}
BASE_COLUMNS = [
    "strategy_id",
    "players",
    "seed",
    "games",
    "wins",
    "win_rate",
    "ci_lo",
    "ci_hi",
]
SUMMARY_TEMPLATE = "strategy_summary_{players}p_seed{seed}.parquet"
SEED_LONG_TEMPLATE = "seed_{seed}_summary_long.parquet"
SEED_WEIGHTED_TEMPLATE = "seed_{seed}_summary_weighted.parquet"
MEAN_NAME_OVERRIDES = {
    "n_rounds": "turns",
    "farkles": "farkles",
    "rolls": "rolls",
    "score": "score",
    "highest_turn": "highest_turn",
    "smart_five_uses": "smart_five_uses",
    "n_smart_five_dice": "smart_five_dice",
    "smart_one_uses": "smart_one_uses",
    "n_smart_one_dice": "smart_one_dice",
    "hot_dice": "hot_dice",
}


def _summary_path(cfg: AppConfig, *, players: int, seed: int) -> Path:
    filename = SUMMARY_TEMPLATE.format(players=players, seed=seed)
    stage_dir = cfg.seed_summaries_dir(players)
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir / filename


def _seed_long_summary_path(cfg: AppConfig, *, seed: int) -> Path:
    stage_dir = cfg.seed_summaries_stage_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    filename = SEED_LONG_TEMPLATE.format(seed=seed)
    return stage_dir / filename


def _seed_weighted_summary_path(cfg: AppConfig, *, seed: int) -> Path:
    stage_dir = cfg.seed_summaries_stage_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    filename = SEED_WEIGHTED_TEMPLATE.format(seed=seed)
    return stage_dir / filename


def _meta_mirror_path(cfg: AppConfig, analysis_path: Path) -> Path | None:
    """Return the mirrored meta path when shared-meta output is configured."""

    if cfg.io.meta_analysis_dir is None:
        return None
    return cfg.meta_analysis_dir / analysis_path.name


def run(cfg: AppConfig, *, force: bool = False) -> None:
    """Materialize per-seed strategy summaries with confidence intervals."""

    stage_log = stage_logger("seed_summaries", logger=LOGGER)
    stage_log.start()

    metrics_path = cfg.metrics_input_path()
    if not metrics_path.exists():
        stage_log.missing_input("metrics parquet missing", path=str(metrics_path))
        return

    metrics_frame, metrics_path = _load_metrics_frame(cfg)
    if metrics_frame.empty:
        stage_log.missing_input("metrics parquet empty", metrics_path=str(metrics_path))
        return

    seeds = sorted(metrics_frame["seed"].unique())
    player_counts = sorted(metrics_frame["players"].unique())
    done = stage_done_path(cfg.seed_summaries_stage_dir, "seed_summaries")

    expected_outputs: list[Path] = []
    for seed in seeds:
        seed_has_data = False
        for players in player_counts:
            subset = metrics_frame[
                (metrics_frame["seed"] == seed) & (metrics_frame["players"] == players)
            ]
            if subset.empty:
                continue
            seed_has_data = True
            expected_outputs.append(_summary_path(cfg, players=int(players), seed=int(seed)))
        if seed_has_data:
            expected_outputs.append(_seed_long_summary_path(cfg, seed=int(seed)))
            expected_outputs.append(_seed_weighted_summary_path(cfg, seed=int(seed)))

    expected_mirrors: list[Path] = []
    for output_path in expected_outputs:
        mirror_path = _meta_mirror_path(cfg, output_path)
        if mirror_path is not None:
            expected_mirrors.append(mirror_path)

    if not force and expected_outputs and stage_is_up_to_date(
        done, inputs=[metrics_path], outputs=expected_outputs, config_sha=cfg.config_sha
    ) and stage_is_up_to_date(
        done,
        inputs=[metrics_path],
        outputs=expected_mirrors,
        config_sha=cfg.config_sha,
    ):
        LOGGER.info(
            "Seed summaries up-to-date",
            extra={"stage": "seed_summaries", "stamp": str(done)},
        )
        return

    outputs: list[Path] = []
    mirrored_outputs: list[Path] = []
    for seed in seeds:
        seed_frames: list[pd.DataFrame] = []
        for players in player_counts:
            subset = metrics_frame[
                (metrics_frame["seed"] == seed) & (metrics_frame["players"] == players)
            ]
            if subset.empty:
                continue
            summary = _build_summary(subset, players=int(players), seed=int(seed))
            if summary.empty:
                continue
            seed_frames.append(summary)
            output_path = _summary_path(cfg, players=int(players), seed=int(seed))
            outputs.append(output_path)
            if not force and _existing_summary_matches(output_path, summary):
                LOGGER.info(
                    "Seed summary already up-to-date",
                    extra={
                        "stage": "seed_summaries",
                        "players": players,
                        "seed": seed,
                        "path": str(output_path),
                    },
                )
            else:
                _write_summary(summary, output_path)
                LOGGER.info(
                    "Seed summary written",
                    extra={
                        "stage": "seed_summaries",
                        "players": players,
                        "seed": seed,
                        "rows": len(summary),
                        "path": str(output_path),
                    },
                )

            mirrored_path = _sync_meta_summary(cfg, summary, output_path)
            if mirrored_path is not None:
                mirrored_outputs.append(mirrored_path)
        if seed_frames:
            seed_summary = _stack_seed_summaries(seed_frames, seed=int(seed))
            seed_output_path = _seed_long_summary_path(cfg, seed=int(seed))
            outputs.append(seed_output_path)
            if not force and _existing_summary_matches(seed_output_path, seed_summary):
                LOGGER.info(
                    "Seed long summary already up-to-date",
                    extra={
                        "stage": "seed_summaries",
                        "seed": seed,
                        "path": str(seed_output_path),
                    },
                )
            else:
                _write_summary(seed_summary, seed_output_path)
                LOGGER.info(
                    "Seed long summary written",
                    extra={
                        "stage": "seed_summaries",
                        "seed": seed,
                        "rows": len(seed_summary),
                        "path": str(seed_output_path),
                    },
                )
            mirrored_path = _sync_meta_summary(cfg, seed_summary, seed_output_path)
            if mirrored_path is not None:
                mirrored_outputs.append(mirrored_path)
            pooled_summary = _build_pooled_seed_summary(cfg, seed_summary, seed=int(seed))
            pooled_output_path = _seed_weighted_summary_path(cfg, seed=int(seed))
            outputs.append(pooled_output_path)
            if not pooled_summary.empty:
                if not force and _existing_summary_matches(pooled_output_path, pooled_summary):
                    LOGGER.info(
                        "Seed pooled summary already up-to-date",
                        extra={
                            "stage": "seed_summaries",
                            "seed": seed,
                            "path": str(pooled_output_path),
                        },
                    )
                else:
                    _write_summary(pooled_summary, pooled_output_path)
                    LOGGER.info(
                        "Seed pooled summary written",
                        extra={
                            "stage": "seed_summaries",
                            "seed": seed,
                            "rows": len(pooled_summary),
                            "path": str(pooled_output_path),
                        },
                    )
                mirrored_path = _sync_meta_summary(cfg, pooled_summary, pooled_output_path)
                if mirrored_path is not None:
                    mirrored_outputs.append(mirrored_path)

    all_outputs = outputs + mirrored_outputs
    if all_outputs:
        write_stage_done(
            done,
            inputs=[metrics_path],
            outputs=all_outputs,
            config_sha=cfg.config_sha,
        )


def _load_metrics_frame(cfg: AppConfig) -> tuple[pd.DataFrame, Path]:
    """Load the combined metrics parquet and standardize core columns."""
    metrics_path = cfg.metrics_input_path()
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_parquet(metrics_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"metrics parquet missing required columns: {sorted(missing)}")

    df = df.copy()
    if "seed" not in df.columns:
        df["seed"] = int(cfg.sim.seed)
    if df["seed"].isna().any():
        raise ValueError("metrics parquet contains null seed values")

    df.rename(columns={"strategy": "strategy_id", "n_players": "players"}, inplace=True)
    strategy_ids = coerce_strategy_ids(df["strategy_id"])
    numeric_ids = pd.to_numeric(strategy_ids, errors="coerce")
    if numeric_ids.isna().any():
        sample = strategy_ids[numeric_ids.isna()].iloc[0]
        raise ValueError(f"metrics parquet contains non-numeric strategy_id: {sample!r}")
    df["strategy_id"] = numeric_ids.astype("Int64")
    df["players"] = df["players"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["games"] = df["games"].fillna(0).astype(int)
    df["wins"] = df["wins"].fillna(0).astype(int)
    if (df["games"] < 0).any():
        raise ValueError("metrics parquet contains negative game counts")
    df = df.sort_values(["seed", "players", "strategy_id"], kind="mergesort").reset_index(drop=True)
    return df, metrics_path


def _build_summary(frame: pd.DataFrame, *, players: int, seed: int) -> pd.DataFrame:
    """Aggregate metrics for a seed/player subset into summary rows."""
    subset = frame.copy()
    if subset.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)

    mean_columns = [c for c in subset.columns if c.startswith("mean_")]
    totals = (
        subset.groupby("strategy_id", sort=True)
        .agg({"games": "sum", "wins": "sum"})
        .reset_index()
    )
    if mean_columns:
        weighted = _weighted_means_by_strategy(subset, mean_columns).reset_index()
        summary = totals.merge(weighted, on="strategy_id", how="left")
    else:
        summary = totals
    summary["players"] = players
    summary["seed"] = seed

    games = summary["games"].to_numpy()
    wins = summary["wins"].to_numpy()
    summary["win_rate"] = np.divide(
        wins, games, out=np.zeros_like(wins, dtype=float), where=games > 0
    )
    ci_bounds = [
        wilson_ci(int(wins_i), int(games_i)) if games_i > 0 else (0.0, 1.0)
        for wins_i, games_i in zip(wins, games, strict=False)
    ]
    summary["ci_lo"] = [bounds[0] for bounds in ci_bounds]
    summary["ci_hi"] = [bounds[1] for bounds in ci_bounds]
    if summary.empty:
        return summary
    summary = _normalize_summary(summary)
    extra_cols = [c for c in summary.columns if c not in BASE_COLUMNS]
    ordered = BASE_COLUMNS + sorted(extra_cols)
    return summary[ordered]


def _stack_seed_summaries(frames: list[pd.DataFrame], *, seed: int) -> pd.DataFrame:
    """Stack per-player summaries for a seed into a long-form table."""
    if not frames:
        return pd.DataFrame(columns=BASE_COLUMNS)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["seed"] = int(seed)
    combined["players"] = combined["players"].astype(int)
    combined["seed"] = combined["seed"].astype(int)
    combined = combined.sort_values(["players", "strategy_id"], kind="mergesort")
    return combined.reset_index(drop=True)


def _normalize_pooling_scheme(pooling_scheme: str) -> str:
    """Normalize pooling scheme names for pooled seed summaries."""
    normalized = pooling_scheme.strip().lower().replace("_", "-")
    if normalized in {"game-count", "gamecount", "count"}:
        return "game-count"
    if normalized in {"equal-k", "equalk", "equal"}:
        return "equal-k"
    if normalized in {"config", "config-provided", "custom"}:
        return "config"
    raise ValueError(f"Unknown pooling scheme: {pooling_scheme!r}")


def _pooling_weights_for_seed_summary(
    df: pd.DataFrame,
    *,
    pooling_scheme: str,
    weights_by_k: dict[int, float],
) -> pd.Series:
    """Return per-row weights for pooled seed summaries."""
    games = pd.to_numeric(df["games"], errors="coerce").fillna(0.0)
    players_numeric = pd.to_numeric(df["players"], errors="raise")
    if not np.all(players_numeric.to_numpy(dtype=float) % 1 == 0):
        raise TypeError("players column must contain integer-like values")
    players = players_numeric.astype(np.int64)
    totals = games.groupby(players).sum()
    totals_map: dict[int, float] = {}
    for k, v in totals.items():
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"Expected integer-like player count key, got {type(k).__name__}")
        k_int = int(k)
        totals_map[k_int] = float(v)

    if pooling_scheme == "game-count":
        return games.astype(float)

    if pooling_scheme == "equal-k":
        def _equal_factor(k: object) -> float:
            if not isinstance(k, (int, np.integer)):
                raise TypeError(f"Expected integer-like player count key, got {type(k).__name__}")
            k_int = int(k)
            total = totals_map.get(k_int, 0.0)
            return 1.0 / total if total > 0 else 0.0

        factors = players.map(_equal_factor)
        return games * factors

    if pooling_scheme == "config":
        missing = sorted(set(totals_map) - set(weights_by_k))
        if missing:
            LOGGER.warning(
                "Missing pooling weights for player counts; treating as zero",
                extra={"stage": "seed_summaries", "missing": missing},
            )

        def _config_factor(k: object) -> float:
            if not isinstance(k, (int, np.integer)):
                raise TypeError(f"Expected integer-like player count key, got {type(k).__name__}")
            k_int = int(k)
            total = totals_map.get(k_int, 0.0)
            if total <= 0:
                return 0.0
            return float(weights_by_k.get(k_int, 0.0)) / total

        factors = players.map(_config_factor)
        return games * factors

    raise ValueError(f"Unknown pooling scheme: {pooling_scheme!r}")


def _weighted_means_with_weights(
    frame: pd.DataFrame,
    columns: list[str],
    weights: pd.Series,
) -> pd.DataFrame:
    """Compute weighted mean columns grouped by ``strategy_id``."""
    working_frame = frame.loc[:, ["strategy_id", *columns]]
    weight_values = weights.astype(float)
    positive_weights = weight_values > 0
    weighted_terms: dict[str, pd.Series] = {
        "pooling_weight_sum": weight_values.where(positive_weights, 0.0)
    }
    aggregation: dict[str, tuple[str, str]] = {
        "pooling_weight_sum": ("pooling_weight_sum", "sum")
    }
    for column in columns:
        values = working_frame[column].astype(float)
        valid = positive_weights & values.notna()
        denominator_col = f"{column}__den"
        numerator_col = f"{column}__num"
        weighted_terms[denominator_col] = weight_values.where(valid, 0.0)
        weighted_terms[numerator_col] = values.where(valid, 0.0) * weight_values.where(valid, 0.0)
        aggregation[denominator_col] = (denominator_col, "sum")
        aggregation[numerator_col] = (numerator_col, "sum")

    grouped = (
        working_frame.assign(**weighted_terms)
        .groupby("strategy_id", sort=True)
        .agg(**aggregation)
    )
    for column in columns:
        denominator = grouped.pop(f"{column}__den")
        numerator = grouped.pop(f"{column}__num")
        grouped[column] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator.to_numpy(dtype=float), np.nan, dtype=float),
            where=denominator.to_numpy(dtype=float) > 0,
        )
    return grouped


def _build_pooled_seed_summary(
    cfg: AppConfig,
    frame: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Aggregate per-player summaries into pooled rows for a seed."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "strategy_id",
                "seed",
                "games",
                "wins",
                "win_rate",
                "ci_lo",
                "ci_hi",
                "pooling_scheme",
                "pooling_weights",
                "pooling_weight_sum",
            ]
        )
    pooling_scheme = _normalize_pooling_scheme(cfg.analysis.pooling_weights)
    weights_by_k = dict(cfg.analysis.pooling_weights_by_k or {})
    if pooling_scheme == "config" and not weights_by_k:
        raise ValueError("analysis.pooling_weights_by_k must be set for config pooling")

    mean_columns = [c for c in frame.columns if c.endswith("_mean")]
    totals = (
        frame.groupby("strategy_id", sort=True)
        .agg({"games": "sum", "wins": "sum"})
        .reset_index()
    )
    weights = _pooling_weights_for_seed_summary(
        frame,
        pooling_scheme=pooling_scheme,
        weights_by_k=weights_by_k,
    )
    weight_sums = (
        weights.groupby(frame["strategy_id"])
        .sum()
        .rename("pooling_weight_sum")
        .reset_index()
    )
    if mean_columns:
        weighted = _weighted_means_with_weights(frame, mean_columns, weights).reset_index()
        summary = totals.merge(weighted, on="strategy_id", how="left")
    else:
        summary = totals
    if "pooling_weight_sum" not in summary.columns:
        summary = summary.merge(weight_sums, on="strategy_id", how="left")

    summary["seed"] = seed
    summary["pooling_scheme"] = pooling_scheme
    summary["pooling_weights"] = (
        "{}" if not weights_by_k else json.dumps(weights_by_k, sort_keys=True)
    )

    games = summary["games"].to_numpy()
    wins = summary["wins"].to_numpy()
    summary["win_rate"] = np.divide(
        wins, games, out=np.zeros_like(wins, dtype=float), where=games > 0
    )
    ci_bounds = [
        wilson_ci(int(wins_i), int(games_i)) if games_i > 0 else (0.0, 1.0)
        for wins_i, games_i in zip(wins, games, strict=False)
    ]
    summary["ci_lo"] = [bounds[0] for bounds in ci_bounds]
    summary["ci_hi"] = [bounds[1] for bounds in ci_bounds]
    summary = _normalize_summary(summary)
    ordered = [
        "strategy_id",
        "seed",
        "games",
        "wins",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "pooling_scheme",
        "pooling_weights",
        "pooling_weight_sum",
    ]
    extra_cols = [c for c in summary.columns if c not in ordered]
    return summary[ordered + sorted(extra_cols)]


def _mean_output_name(column: str) -> str:
    """Convert a ``mean_<metric>`` column into a user-facing label."""
    base = column.removeprefix("mean_")
    label = MEAN_NAME_OVERRIDES.get(base, base)
    return f"{label}_mean"


def _weighted_means_by_strategy(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute weighted mean columns grouped by ``strategy_id``."""
    working_frame = frame.loc[:, ["strategy_id", "games", *columns]]
    weight_values = working_frame["games"].astype(float)
    positive_weights = weight_values > 0
    weighted_terms: dict[str, pd.Series] = {}
    aggregation: dict[str, tuple[str, str]] = {}
    for column in columns:
        values = working_frame[column].astype(float)
        valid = positive_weights & values.notna()
        denominator_col = f"{column}__den"
        numerator_col = f"{column}__num"
        weighted_terms[denominator_col] = weight_values.where(valid, 0.0)
        weighted_terms[numerator_col] = values.where(valid, 0.0) * weight_values.where(valid, 0.0)
        aggregation[denominator_col] = (denominator_col, "sum")
        aggregation[numerator_col] = (numerator_col, "sum")

    grouped = (
        working_frame.assign(**weighted_terms)
        .groupby("strategy_id", sort=True)
        .agg(**aggregation)
    )
    renamed_columns: dict[str, pd.Series] = {}
    for column in columns:
        denominator = grouped.pop(f"{column}__den")
        numerator = grouped.pop(f"{column}__num")
        renamed_columns[_mean_output_name(column)] = pd.Series(
            np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator.to_numpy(dtype=float), np.nan, dtype=float),
                where=denominator.to_numpy(dtype=float) > 0,
            ),
            index=grouped.index,
        )
    return grouped.assign(**renamed_columns)


def _normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce deterministic types and ordering for summary comparison."""
    normalized = df.copy()
    for col in ("strategy_id", "players", "seed", "games", "wins"):
        if col in normalized.columns:
            normalized[col] = _cast_int32_if_safe(normalized[col])
    for col in ("win_rate", "ci_lo", "ci_hi"):
        if col in normalized.columns:
            normalized[col] = normalized[col].astype(float)
    extra_cols = [c for c in normalized.columns if c not in BASE_COLUMNS]
    for col in extra_cols:
        if pd.api.types.is_numeric_dtype(normalized[col]):
            normalized[col] = normalized[col].astype(float)
    return normalized


def _cast_int32_if_safe(series: pd.Series) -> pd.Series:
    """Cast series to int32 when values fit, otherwise keep int64."""
    values = pd.to_numeric(series, errors="coerce")
    non_null = values.dropna()
    if non_null.empty:
        return values.astype(np.int64)
    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max
    if non_null.min() < int32_min or non_null.max() > int32_max:
        return values.astype(np.int64)
    return values.astype(np.int32)


def _existing_summary_matches(path: Path, new_df: pd.DataFrame) -> bool:
    """Check if an existing summary parquet matches ``new_df`` exactly."""
    if not path.exists():
        return False
    try:
        existing = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return False
    existing_norm = _normalize_summary(existing)
    new_norm = _normalize_summary(new_df)
    if list(existing_norm.columns) != list(new_norm.columns):
        return False
    try:
        pdt.assert_frame_equal(existing_norm, new_norm, check_dtype=True, check_exact=True)
        return True
    except AssertionError:
        return False


def _write_summary(df: pd.DataFrame, path: Path) -> None:
    """Write a summary frame to parquet using atomic semantics."""

    table = pa.Table.from_pandas(df, preserve_index=False)
    write_parquet_atomic(table, path)


def _sync_meta_summary(cfg: AppConfig, summary: pd.DataFrame, analysis_path: Path) -> Path | None:
    """Copy the latest summary into the shared meta directory when configured."""

    meta_path = _meta_mirror_path(cfg, analysis_path)
    if meta_path is None or meta_path == analysis_path:
        return None
    if _existing_summary_matches(meta_path, summary):
        return meta_path

    _write_summary(summary, meta_path)
    LOGGER.info(
        "Seed summary synced to meta directory",
        extra={
            "stage": "seed_summaries",
            "players": (
                summary["players"].iloc[0]
                if (not summary.empty and "players" in summary.columns)
                else None
            ),
            "seed": summary["seed"].iloc[0] if not summary.empty else None,
            "path": str(meta_path),
        },
    )
    return meta_path


__all__ = ["run"]
