# src/farkle/analysis/metrics.py
"""Aggregate curated data into per-strategy metrics and outputs.

Computes win rates and seat advantages from combined parquet shards, validates
input schemas, and emits CSV/Parquet artifacts for downstream reporting.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Hashable, Iterable, Protocol, Sequence, TypeAlias

import numpy as np
import pandas as pd
import pyarrow as pa

from farkle.analysis.all_player_metrics import build_all_player_batch_metrics
from farkle.analysis.checks import check_pre_metrics
from farkle.analysis.isolated_metrics import build_isolated_metrics
from farkle.analysis.performance import PerformanceArtifacts, build_canonical_performance
from farkle.analysis.seat_analysis import SeatAnalysisArtifacts, build_canonical_seat_analysis
from farkle.analysis.seat_stats import (
    SeatMetricConfig,
    compute_seat_advantage,
    compute_seat_metrics,
)
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.utils.aggregation import normalize_k_aggregation_method
from farkle.utils.artifacts import write_csv_atomic, write_parquet_atomic
from farkle.utils.stage_io import resolve_worker_count, select_preferred_or_legacy
from farkle.utils.writer import atomic_path

if TYPE_CHECKING:  # pragma: no cover - pandas typing is optional at runtime
    from pandas._typing import Scalar as PandasScalar

    Scalar: TypeAlias = PandasScalar
else:  # pragma: no cover - fallback for older pandas
    Scalar: TypeAlias = Hashable

LOGGER = logging.getLogger(__name__)


class _AnalysisAggregationConfig(Protocol):
    @property
    def k_aggregation_method(self) -> str: ...

    @property
    def k_weights(self) -> dict[int, float] | None: ...


class _MetricsAggregationConfig(Protocol):
    @property
    def analysis(self) -> _AnalysisAggregationConfig: ...


def run(cfg: AppConfig) -> None:
    """Compute per-strategy metrics and seat-advantage tables."""

    analysis_dir = cfg.analysis_dir
    metrics_dir = cfg.metrics_combined_dir
    data_file = cfg.curated_dataset
    out_metrics = cfg.metrics_output_path()
    out_metrics_weighted = cfg.metrics_output_path("metrics_weighted.parquet")
    out_seats = cfg.metrics_output_path("seat_advantage.csv")
    out_seats_parquet = cfg.metrics_output_path("seat_advantage.parquet")
    out_seat_metrics = cfg.metrics_output_path("seat_metrics.parquet")
    out_seat_metrics_csv = cfg.metrics_output_path("seat_metrics.csv")
    stamp = cfg.metrics_output_path("metrics.done.json")

    done = stage_done_path(cfg.metrics_stage_dir, "metrics")
    done_isolated = stage_done_path(cfg.metrics_stage_dir, "metrics_isolated")
    done_core = stage_done_path(cfg.metrics_stage_dir, "metrics_core")
    done_weighted = stage_done_path(cfg.metrics_stage_dir, "metrics_weighted")
    done_seat_advantage = stage_done_path(cfg.metrics_stage_dir, "metrics_seat_advantage")
    done_seat_metrics = stage_done_path(cfg.metrics_stage_dir, "metrics_seat_metrics")
    stamp_isolated = cfg.metrics_output_path("metrics.isolated.stamp.json")
    stamp_core = cfg.metrics_output_path("metrics.core.stamp.json")
    stamp_weighted = cfg.metrics_output_path("metrics.weighted.stamp.json")
    stamp_seat_advantage = cfg.metrics_output_path("metrics.seat_advantage.stamp.json")
    stamp_seat_metrics = cfg.metrics_output_path("metrics.seat_metrics.stamp.json")
    player_counts = sorted({int(n) for n in cfg.sim.n_players_list})
    include_players = set(player_counts)
    raw_metric_inputs = [
        cfg.results_root / f"{n}_players" / f"{n}p_metrics.parquet" for n in player_counts
    ]
    all_player_inputs = [
        cfg.ingested_rows_curated(n) for n in player_counts if cfg.ingested_rows_curated(n).exists()
    ]
    all_player_targets = [
        cfg.metrics_all_player_batch_path(n)
        for n in player_counts
        if cfg.ingested_rows_curated(n).exists()
    ]
    performance_enabled = (
        len(all_player_inputs) == len(player_counts) and cfg.screening.delta_across_k is not None
    )
    performance_targets = (
        [
            *(cfg.performance_by_k_path(n) for n in player_counts),
            cfg.performance_across_k_path(),
            cfg.performance_bootstrap_path(),
            cfg.performance_control_contrasts_path(),
        ]
        if performance_enabled
        else []
    )
    seat_inputs = [
        cfg.combined_rows_by_k(n) for n in player_counts if cfg.combined_rows_by_k(n).exists()
    ]
    seat_targets = (
        [
            *(cfg.seat_batch_counts_path(n) for n in player_counts),
            *(cfg.seat_effects_by_k_path(n) for n in player_counts),
            *(cfg.seat_population_by_k_path(n) for n in player_counts),
            cfg.seat_standardized_across_k_path(),
            cfg.seat_exposure_mixture_diagnostic_path(),
            cfg.seat_selfplay_diagnostic_path(),
            cfg.seat_mirrored_diagnostic_path(),
        ]
        if len(seat_inputs) == len(player_counts)
        else []
    )
    available_raw_inputs = [path for path in raw_metric_inputs if path.exists()]
    iso_targets = []
    for n in player_counts:
        raw_path = cfg.results_root / f"{n}_players" / f"{n}p_metrics.parquet"
        if not raw_path.exists():
            continue
        preferred = cfg.metrics_isolated_path(n)
        legacy = cfg.legacy_metrics_isolated_path(n)
        if preferred.exists():
            iso_targets.append(preferred)
        elif legacy.exists():
            iso_targets.append(legacy)
        else:
            iso_targets.append(preferred)
    outputs = [
        out_metrics,
        out_metrics_weighted,
        out_seats,
        out_seats_parquet,
        out_seat_metrics,
        out_seat_metrics_csv,
        *iso_targets,
        *all_player_targets,
        *performance_targets,
        *seat_targets,
    ]
    stage_inputs = [data_file, *raw_metric_inputs, *all_player_inputs, *seat_inputs]
    if stage_is_up_to_date(
        done,
        inputs=stage_inputs,
        outputs=outputs,
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=[*performance_targets, *seat_targets],
    ):
        LOGGER.info(
            "Metrics stage up-to-date",
            extra={"stage": "metrics", "path": str(done)},
        )
        return

    if not data_file.exists():
        raise FileNotFoundError(
            f"metrics: missing combined parquet {data_file} – run combine step first"
        )

    LOGGER.info(
        "Metrics stage start",
        extra={
            "stage": "metrics",
            "data_file": str(data_file),
            "analysis_dir": str(analysis_dir),
            "metrics_dir": str(metrics_dir),
        },
    )

    check_pre_metrics(data_file, winner_col="winner_seat")

    all_player_paths = _ensure_all_player_metrics(cfg, player_counts)
    performance_paths = _ensure_canonical_performance(cfg, all_player_paths, player_counts)
    seat_paths = _ensure_canonical_seat_analysis(cfg, seat_inputs, player_counts)

    if stage_is_up_to_date(
        done_isolated,
        inputs=[data_file, *available_raw_inputs],
        outputs=iso_targets,
        cfg=cfg,
        stage="metrics",
    ):
        iso_paths = [path for path in iso_targets if path.exists()]
        raw_inputs = raw_metric_inputs
    else:
        iso_paths, raw_inputs = _ensure_isolated_metrics(cfg, player_counts)
        _write_stamp(
            stamp_isolated,
            inputs=[data_file, *available_raw_inputs],
            outputs=iso_paths,
        )
        write_stage_done(
            done_isolated,
            inputs=[data_file, *available_raw_inputs],
            outputs=iso_paths,
            cfg=cfg,
            stage="metrics",
        )

    outputs = [
        out_metrics,
        out_metrics_weighted,
        out_seats,
        out_seats_parquet,
        out_seat_metrics,
        out_seat_metrics_csv,
        *iso_paths,
        *all_player_paths,
        *performance_paths,
        *seat_paths,
    ]

    if stage_is_up_to_date(
        done_core,
        inputs=iso_paths,
        outputs=[out_metrics],
        cfg=cfg,
        stage="metrics",
    ):
        metrics_df = pd.read_parquet(out_metrics)
    else:
        metrics_df = _collect_metrics_frames(iso_paths)
        if metrics_df.empty:
            raise RuntimeError("metrics: no isolated metric files generated")

        metrics_df = _add_win_rate_uncertainty(metrics_df)
        metrics_df = _downcast_metric_counters(metrics_df)

        metrics_table = pa.Table.from_pandas(metrics_df, preserve_index=False)
        write_parquet_atomic(metrics_table, out_metrics)
        _write_stamp(stamp_core, inputs=iso_paths, outputs=[out_metrics])
        write_stage_done(
            done_core,
            inputs=iso_paths,
            outputs=[out_metrics],
            cfg=cfg,
            stage="metrics",
        )

    if stage_is_up_to_date(
        done_weighted,
        inputs=[out_metrics],
        outputs=[out_metrics_weighted],
        cfg=cfg,
        stage="metrics",
    ):
        weighted_df = pd.read_parquet(out_metrics_weighted)
    else:
        weighted_df = _compute_weighted_metrics(metrics_df, cfg)
        weighted_table = pa.Table.from_pandas(weighted_df, preserve_index=False)
        write_parquet_atomic(weighted_table, out_metrics_weighted)
        _write_stamp(
            stamp_weighted,
            inputs=[out_metrics],
            outputs=[out_metrics_weighted],
        )
        write_stage_done(
            done_weighted,
            inputs=[out_metrics],
            outputs=[out_metrics_weighted],
            cfg=cfg,
            stage="metrics",
        )

    seat_cfg = SeatMetricConfig(seat_range=cfg.metrics_seat_range)
    if stage_is_up_to_date(
        done_seat_advantage,
        inputs=[data_file],
        outputs=[out_seats, out_seats_parquet],
        cfg=cfg,
        stage="metrics",
    ):
        seat_df = pd.read_csv(out_seats)
    else:
        LOGGER.info(
            "Seat-advantage aggregation start",
            extra={
                "stage": "metrics",
                "input": str(data_file),
                "output_csv": str(out_seats),
                "output_parquet": str(out_seats_parquet),
            },
        )
        seat_df = compute_seat_advantage(cfg, data_file, seat_cfg, include_players=include_players)
        write_csv_atomic(seat_df, out_seats)
        seat_table = pa.Table.from_pandas(seat_df, preserve_index=False)
        write_parquet_atomic(seat_table, out_seats_parquet)
        _write_stamp(
            stamp_seat_advantage,
            inputs=[data_file],
            outputs=[out_seats, out_seats_parquet],
        )
        write_stage_done(
            done_seat_advantage,
            inputs=[data_file],
            outputs=[out_seats, out_seats_parquet],
            cfg=cfg,
            stage="metrics",
        )

    if stage_is_up_to_date(
        done_seat_metrics,
        inputs=[data_file],
        outputs=[out_seat_metrics, out_seat_metrics_csv],
        cfg=cfg,
        stage="metrics",
    ):
        seat_metrics_df = pd.read_parquet(out_seat_metrics)
    else:
        seat_progress = cfg.metrics_output_path("seat_metrics.progress.json")
        LOGGER.info(
            "Seat-metrics aggregation start",
            extra={
                "stage": "metrics",
                "input": str(data_file),
                "output_parquet": str(out_seat_metrics),
                "output_csv": str(out_seat_metrics_csv),
                "progress": str(seat_progress),
            },
        )
        seat_metrics_df = compute_seat_metrics(
            data_file,
            seat_cfg,
            include_players=include_players,
            progress_path=seat_progress,
            progress_logging=cfg.analysis.progress_logging,
        )
        seat_metrics_table = pa.Table.from_pandas(seat_metrics_df, preserve_index=False)
        write_parquet_atomic(seat_metrics_table, out_seat_metrics)
        write_csv_atomic(seat_metrics_df, out_seat_metrics_csv)
        _write_stamp(
            stamp_seat_metrics,
            inputs=[data_file],
            outputs=[out_seat_metrics, out_seat_metrics_csv],
        )
        write_stage_done(
            done_seat_metrics,
            inputs=[data_file],
            outputs=[out_seat_metrics, out_seat_metrics_csv],
            cfg=cfg,
            stage="metrics",
        )

    if not metrics_df.empty:
        leader = metrics_df.sort_values(["wins", "win_rate"], ascending=False).iloc[0]
        LOGGER.info(
            "Metrics leaderboard computed",
            extra={
                "stage": "metrics",
                "top_strategy": leader["strategy"],
                "wins": int(leader["wins"]),
                "games": int(leader["games"]),
            },
        )

    _write_stamp(
        stamp,
        inputs=stage_inputs,
        outputs=[
            out_metrics,
            out_metrics_weighted,
            out_seats,
            out_seats_parquet,
            out_seat_metrics,
            out_seat_metrics_csv,
            *iso_paths,
            *all_player_paths,
            *performance_paths,
            *seat_paths,
        ],
    )
    complete_extra = {
        "stage": "metrics",
        "rows": len(metrics_df),
        "seat_rows": len(seat_df),
        "metrics_path": str(out_metrics),
        "weighted_metrics_path": str(out_metrics_weighted),
        "seat_path": str(out_seats),
        "seat_parquet": str(out_seats_parquet),
        "seat_metrics": str(out_seat_metrics),
        "all_player_batch_metrics": [str(path) for path in all_player_paths],
        "canonical_performance": [str(path) for path in performance_paths],
        "canonical_seat_analysis": [str(path) for path in seat_paths],
    }
    LOGGER.info(
        "Metrics stage complete",
        extra=complete_extra,
    )
    write_stage_done(
        done,
        inputs=stage_inputs,
        outputs=outputs,
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=[*performance_paths, *seat_paths],
    )


def _ensure_isolated_metrics(
    cfg: AppConfig, player_counts: Sequence[int]
) -> tuple[list[Path], list[Path]]:
    """Generate normalized per-player-count metrics where available.

    Args:
        cfg: Application configuration containing metrics locations.
        player_counts: Player counts to process.

    Returns:
        Tuple of normalized parquet paths discovered and the corresponding raw
        inputs checked on disk.
    """

    def _process_player_count(
        n: int,
    ) -> tuple[int, Path | None, Path, dict[str, object]]:
        """Normalize isolated metrics for one player count when inputs exist.

        Args:
            n: Player count whose isolated metrics should be prepared.

        Returns:
            Tuple of player count, resolved isolated path, raw input path, and log metadata.
        """
        raw_path = cfg.results_root / f"{n}_players" / f"{n}p_metrics.parquet"
        preferred = cfg.metrics_isolated_path(n)
        legacy = cfg.legacy_metrics_isolated_path(n)

        if not raw_path.exists():
            return n, None, raw_path, {"missing_raw": True}

        try:
            iso_path = build_isolated_metrics(cfg, n)
            return n, iso_path, raw_path, {}
        except Exception as exc:  # noqa: BLE001
            log_fields: dict[str, object] = {
                "normalization_error": str(exc),
            }
        selection = select_preferred_or_legacy(preferred, legacy)
        if selection is not None:
            if selection.used_legacy:
                log_fields["used_legacy"] = True
            return n, selection.path, raw_path, log_fields

        return n, None, raw_path, log_fields

    if not player_counts:
        return [], []

    worker_count = resolve_worker_count(
        cfg.analysis.n_jobs,
        cfg.sim.n_jobs,
        item_count=len(player_counts),
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        unordered_results = list(executor.map(_process_player_count, player_counts))

    results_by_n = {
        n: (iso_path, raw_path, log_fields)
        for n, iso_path, raw_path, log_fields in unordered_results
    }

    iso_paths: list[Path] = []
    raw_inputs: list[Path] = []
    for n in player_counts:
        iso_path, raw_path, log_fields = results_by_n[n]
        raw_inputs.append(raw_path)
        if log_fields.get("missing_raw"):
            LOGGER.warning(
                "Expanded metrics missing",
                extra={"stage": "metrics", "player_count": n, "path": str(raw_path)},
            )
            continue
        normalization_error = log_fields.get("normalization_error")
        if normalization_error is not None:
            LOGGER.warning(
                "Failed to normalize metrics parquet",
                extra={
                    "stage": "metrics",
                    "player_count": n,
                    "path": str(raw_path),
                    "error": str(normalization_error),
                },
            )
        if log_fields.get("used_legacy"):
            LOGGER.info(
                "Using legacy isolated metrics path",
                extra={
                    "stage": "metrics",
                    "player_count": n,
                    "path": str(iso_path),
                },
            )
        if iso_path is not None:
            iso_paths.append(iso_path)
    return iso_paths, raw_inputs


def _ensure_all_player_metrics(cfg: AppConfig, player_counts: Sequence[int]) -> list[Path]:
    """Build independent per-k unconditional exposure artifacts concurrently."""

    available = [n for n in player_counts if cfg.ingested_rows_curated(n).exists()]
    if not available:
        LOGGER.warning(
            "Canonical per-k curated rows unavailable; all-player batch metrics skipped",
            extra={"stage": "metrics", "player_counts": list(player_counts)},
        )
        return []
    worker_count = resolve_worker_count(
        cfg.analysis.n_jobs,
        cfg.sim.n_jobs,
        item_count=len(available),
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        outputs = list(
            executor.map(
                lambda n: build_all_player_batch_metrics(cfg, n),
                available,
            )
        )
    return [path for _, path in sorted(zip(available, outputs, strict=True))]


def _ensure_canonical_performance(
    cfg: AppConfig,
    all_player_paths: Sequence[Path],
    player_counts: Sequence[int],
) -> list[Path]:
    """Build canonical performance outputs when their locked contract is available."""

    if len(all_player_paths) != len(player_counts):
        return []
    if cfg.screening.delta_across_k is None:
        LOGGER.warning(
            "Canonical performance skipped: screening.delta_across_k is not configured",
            extra={"stage": "metrics"},
        )
        return []
    artifacts: PerformanceArtifacts = build_canonical_performance(cfg)
    return list(artifacts.all_paths)


def _ensure_canonical_seat_analysis(
    cfg: AppConfig,
    seat_inputs: Sequence[Path],
    player_counts: Sequence[int],
) -> list[Path]:
    """Build canonical seat outputs when every normalized by-k input exists."""

    if len(seat_inputs) != len(player_counts):
        LOGGER.warning(
            "Canonical seat analysis skipped: normalized by-k rows are incomplete",
            extra={"stage": "metrics", "player_counts": list(player_counts)},
        )
        return []
    artifacts: SeatAnalysisArtifacts = build_canonical_seat_analysis(cfg)
    return list(artifacts.all_paths)


def _collect_metrics_frames(paths: Iterable[Path]) -> pd.DataFrame:
    """Load multiple metrics parquets into a single DataFrame."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(
            columns=[
                "strategy",
                "n_players",
                "games",
                "wins",
                "win_rate",
                "win_prob",
                "win_conditioned_score_contribution_per_exposure",
            ]
        )
    df = pd.concat(frames, ignore_index=True)
    if "win_prob" not in df.columns:
        df["win_prob"] = df["win_rate"]
    base_cols = [
        "strategy",
        "n_players",
        "games",
        "wins",
        "win_rate",
        "win_prob",
        "win_conditioned_score_contribution_per_exposure",
    ]
    remainder = [c for c in df.columns if c not in base_cols]
    return df[base_cols + remainder]


def _add_win_rate_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """Attach standard errors and normal-approximation CIs for win rates."""

    if df.empty:
        return df

    out = df.copy()
    games = pd.to_numeric(out["games"], errors="coerce")
    win_rate = pd.to_numeric(out["win_rate"], errors="coerce")
    positive_games = games > 0

    se = pd.Series(0.0, index=out.index, dtype="float64")
    safe_games = games.where(positive_games)
    win_prob = win_rate.loc[positive_games]
    se.loc[positive_games] = ((win_prob * (1.0 - win_prob)) / safe_games.loc[positive_games]).pow(
        0.5
    )
    out["se_win_rate"] = se

    z = 1.96
    ci_lo = (win_rate - z * se).clip(lower=0.0, upper=1.0)
    ci_hi = (win_rate + z * se).clip(lower=0.0, upper=1.0)
    out["win_rate_ci_lo"] = ci_lo.where(positive_games, win_rate)
    out["win_rate_ci_hi"] = ci_hi.where(positive_games, win_rate)

    desired_order = [
        "strategy",
        "n_players",
        "games",
        "wins",
        "win_rate",
        "se_win_rate",
        "win_rate_ci_lo",
        "win_rate_ci_hi",
    ]
    remaining = [c for c in out.columns if c not in desired_order]
    return out[desired_order + remaining]


def _downcast_metric_counters(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast integer counters when safe to reduce parquet size."""
    if df.empty:
        return df
    out = df.copy()
    int_cols = {"games", "wins", "n_players", "seed"}
    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max

    for col in out.columns:
        if col in int_cols or col.endswith("_count") or col.startswith("n_"):
            if col not in out.columns:
                continue
            series = pd.to_numeric(out[col], errors="coerce")
            non_null = series.dropna()
            if non_null.empty:
                continue
            if not np.all(np.isclose(non_null, np.floor(non_null))):
                continue
            if non_null.min() < int32_min or non_null.max() > int32_max:
                continue
            out[col] = pd.to_numeric(series, downcast="integer")
    return out


def _normalize_k_aggregation_method(aggregation_method: str) -> str:
    """Backward-compatible module-local alias for aggregation normalization."""

    return normalize_k_aggregation_method(aggregation_method)


def _k_aggregation_method_for_metrics(
    df: pd.DataFrame,
    *,
    aggregation_method: str,
    weights_by_k: dict[int, float],
) -> pd.Series:
    """Return per-row weights for combined metric aggregation."""

    games = pd.to_numeric(df["games"], errors="coerce").fillna(0.0)
    n_players = df["n_players"].astype(np.int16)
    totals = games.groupby(n_players).sum()
    totals_map: dict[int, float] = {}
    for k, v in totals.items():
        assert isinstance(k, (int, np.integer))
        k_int = int(k)
        totals_map[k_int] = float(v)

    if aggregation_method == "game-count":
        return games.astype(float)

    if aggregation_method == "equal-k":

        def _equal_factor(k: int | np.integer) -> float:
            """Resolve the equal-``k`` aggregation factor for one player count."""
            total = totals_map.get(int(k), 0.0)
            return 1.0 / total if total > 0 else 0.0

        factors = n_players.map(_equal_factor)
        return games * factors

    if aggregation_method == "config":
        missing = sorted(set(totals_map) - set(weights_by_k))
        if missing:
            LOGGER.warning(
                "Missing aggregation weights for player counts; treating as zero",
                extra={"stage": "metrics", "missing": missing},
            )

        def _config_factor(k: int | np.integer) -> float:
            """Resolve the config-driven aggregation factor for one player count."""
            total = totals_map.get(int(k), 0.0)
            if total <= 0:
                return 0.0
            return float(weights_by_k.get(int(k), 0.0)) / total

        factors = n_players.map(_config_factor)
        return games * factors

    raise ValueError(f"Unknown aggregation scheme: {aggregation_method!r}")


def _combined_value_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric metric columns that should participate in combined aggregation.

    Args:
        df: Metrics frame to inspect.

    Returns:
        Numeric value columns excluding identifiers, counters, and bookkeeping fields.
    """
    exclude = {
        "strategy",
        "n_players",
        "games",
        "wins",
        "false_wins_handled",
        "missing_before_pad",
        "seed",
    }
    columns: list[str] = []
    for col in df.columns:
        if col in exclude or col.endswith("_count"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        columns.append(col)
    return columns


def _weighted_mean(values: pd.Series, weights: np.ndarray) -> float:
    """Compute a weighted mean for one metrics series.

    Args:
        values: Metric values to average.
        weights: Per-row weights aligned with ``values``.

    Returns:
        Weighted mean, or ``nan`` when no valid weighted values exist.
    """
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(numeric) & np.isfinite(weights)
    if not mask.any():
        return float("nan")
    subset_weights = weights[mask]
    total = subset_weights.sum()
    if total <= 0:
        return float("nan")
    return float(np.average(numeric[mask], weights=subset_weights))


def _compute_weighted_metrics(
    metrics_df: pd.DataFrame, cfg: _MetricsAggregationConfig
) -> pd.DataFrame:
    """Compute combined weighted metrics across player counts."""

    columns = [
        "strategy",
        "games",
        "wins",
        "win_rate",
        "win_prob",
        "win_conditioned_score_contribution_per_exposure",
        "aggregation_method",
        "k_aggregation_method",
    ]
    if metrics_df.empty:
        return pd.DataFrame(columns=columns)

    aggregation_method = normalize_k_aggregation_method(cfg.analysis.k_aggregation_method)
    weights_by_k = dict(cfg.analysis.k_weights or {})
    if aggregation_method == "config" and not weights_by_k:
        raise ValueError("analysis.k_weights must be set for config aggregation")

    weights = _k_aggregation_method_for_metrics(
        metrics_df,
        aggregation_method=aggregation_method,
        weights_by_k=weights_by_k,
    )
    value_cols = _combined_value_columns(metrics_df)

    combined_rows: list[dict[str, Scalar]] = []
    for strategy, group in metrics_df.groupby("strategy", sort=False):
        group_weights = weights.loc[group.index].to_numpy(dtype=float)
        if not np.isfinite(group_weights).any() or group_weights.sum() <= 0:
            continue
        row: dict[str, Scalar] = {
            "strategy": strategy,
            "games": int(pd.to_numeric(group["games"], errors="coerce").fillna(0.0).sum()),
            "wins": int(pd.to_numeric(group["wins"], errors="coerce").fillna(0.0).sum()),
        }
        for col in value_cols:
            row[col] = _weighted_mean(group[col], group_weights)
        combined_rows.append(row)

    combined_df = pd.DataFrame(combined_rows)
    k_aggregation_method = json.dumps(weights_by_k, sort_keys=True) if weights_by_k else "{}"
    combined_df["aggregation_method"] = aggregation_method
    combined_df["k_aggregation_method"] = k_aggregation_method
    ordered = [c for c in columns if c in combined_df.columns]
    remainder = [c for c in combined_df.columns if c not in ordered]
    return combined_df[ordered + remainder]


def _compute_seat_advantage(cfg: AppConfig, combined: Path) -> pd.DataFrame:
    """Backwards-compatible wrapper for seat-advantage calculations."""

    seat_cfg = SeatMetricConfig(seat_range=cfg.metrics_seat_range)
    include_players = {int(n) for n in cfg.sim.n_players_list}
    return compute_seat_advantage(cfg, combined, seat_cfg, include_players=include_players)


def _stamp(path: Path) -> dict[str, float | int]:
    """Capture filesystem metadata for cache stamps."""
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _write_stamp(stamp_path: Path, *, inputs: Iterable[Path], outputs: Iterable[Path]) -> None:
    """Persist a JSON stamp summarizing inputs and outputs for auditing."""
    payload = {
        "inputs": {str(p): _stamp(p) for p in inputs if p.exists()},
        "outputs": {str(p): _stamp(p) for p in outputs if p.exists()},
    }
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(stamp_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2))
