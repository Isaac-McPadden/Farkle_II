# src/farkle/analysis/game_stats.py
"""Compute descriptive statistics for game lengths and victory margins.

Reads curated row-level parquet files (per-``n`` shards and the combined
superset) and aggregates the ``n_rounds`` column. Outputs both per-strategy
statistics and a small global summary grouped by ``n_players``.

The module also flags close margins and multi-target games, emitting pooled
artifacts under ``04_game_stats/pooled`` with per-game records plus aggregated
frequencies per strategy and player-count cohort.

The module also derives per-game ``margin_runner_up`` and ``score_spread`` from
seat-level scores and writes ``04_game_stats/pooled/margin_stats.parquet`` with
per-``(strategy, n_players)`` summaries. Margin schema:

``summary_level``
    Literal "strategy" for compatibility with ``game_length.parquet``.
``strategy``
    Strategy ID taken from ``P#_strategy`` columns.
``n_players``
    Player count inferred from the shard path.
``observations``
    Number of games with at least two valid seat scores.
``mean_margin_runner_up`` / ``median_margin_runner_up`` / ``std_margin_runner_up``
    Descriptive statistics over ``margin_runner_up``.
``prob_margin_runner_up_le_500`` / ``prob_margin_runner_up_le_1000``
    Close-game shares for the given thresholds on ``margin_runner_up``.
``mean_score_spread`` / ``median_score_spread`` / ``std_score_spread``
    Descriptive statistics over ``score_spread``.
``prob_score_spread_le_500`` / ``prob_score_spread_le_1000``
    Close-game shares for the given thresholds on ``score_spread``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from pandas._libs.missing import NAType
from pandas._typing import Scalar

from farkle.analysis import stage_logger
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.utils.analysis_shared import to_int
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.schema_helpers import n_players_from_schema
from farkle.utils.types import Compression
from farkle.utils.writer import ParquetShardWriter

StatValue: TypeAlias = float | int | str | NAType
NormalizedScalar: TypeAlias = Scalar | None
ArrowColumnData: TypeAlias = np.ndarray | list[Any] | pa.Array | pa.ChunkedArray
StrategyPandasDtype: TypeAlias = (
    pd.UInt8Dtype
    | pd.UInt16Dtype
    | pd.UInt32Dtype
    | pd.UInt64Dtype
    | pd.Int8Dtype
    | pd.Int16Dtype
    | pd.Int32Dtype
    | pd.Int64Dtype
)

LOGGER = logging.getLogger(__name__)


def _strategy_arrow_type(per_n_inputs: Sequence[tuple[int, Path]]) -> pa.DataType:
    for _, path in per_n_inputs:
        ds_in = ds.dataset(path)
        for name in ds_in.schema.names:
            if name.endswith("_strategy"):
                dtype = ds_in.schema.field(name).type
                if pa.types.is_integer(dtype):
                    return dtype
    return pa.int64()


def _strategy_numpy_dtype(strategy_type: pa.DataType) -> np.dtype[Any]:
    if pa.types.is_int8(strategy_type):
        return np.dtype("int8")
    if pa.types.is_int16(strategy_type):
        return np.dtype("int16")
    if pa.types.is_int32(strategy_type):
        return np.dtype("int32")
    if pa.types.is_int64(strategy_type):
        return np.dtype("int64")
    if pa.types.is_uint8(strategy_type):
        return np.dtype("uint8")
    if pa.types.is_uint16(strategy_type):
        return np.dtype("uint16")
    if pa.types.is_uint32(strategy_type):
        return np.dtype("uint32")
    if pa.types.is_uint64(strategy_type):
        return np.dtype("uint64")
    return np.dtype("int64")


def _strategy_pandas_dtype(strategy_type: pa.DataType) -> StrategyPandasDtype:
    if pa.types.is_uint8(strategy_type):
        return pd.UInt8Dtype()
    if pa.types.is_uint16(strategy_type):
        return pd.UInt16Dtype()
    if pa.types.is_uint32(strategy_type):
        return pd.UInt32Dtype()
    if pa.types.is_uint64(strategy_type):
        return pd.UInt64Dtype()
    if pa.types.is_int8(strategy_type):
        return pd.Int8Dtype()
    if pa.types.is_int16(strategy_type):
        return pd.Int16Dtype()
    if pa.types.is_int32(strategy_type):
        return pd.Int32Dtype()
    return pd.Int64Dtype()


def _to_python_scalar(value: Scalar) -> NormalizedScalar:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _strategy_key_to_int(value: Scalar, *, field: str = "strategy") -> int:
    normalized = _to_python_scalar(value)
    if normalized is None:
        raise ValueError(f"invalid {field} scalar for int conversion: {value!r}")
    try:
        return to_int(normalized)
    except ValueError as err:
        raise ValueError(f"invalid {field} scalar for int conversion: {value!r}") from err


def _strategy_stat_value(value: Scalar) -> StatValue:
    normalized = _to_python_scalar(value)
    if normalized is None or normalized is pd.NA:
        return pd.NA
    if isinstance(normalized, bytes):
        return normalized.decode("utf-8", errors="replace")
    if isinstance(normalized, str):
        return normalized
    if isinstance(normalized, bool):
        return int(normalized)
    if isinstance(normalized, int):
        return normalized
    if isinstance(normalized, float):
        return float(normalized)
    return str(normalized)


def _coerce_strategy_dtype(df: pd.DataFrame, strategy_type: pa.DataType) -> pd.DataFrame:
    if df.empty or "strategy" not in df.columns:
        return df
    out = df.copy()
    strategy_values = cast(pd.Series, pd.to_numeric(out["strategy"], errors="coerce"))
    out["strategy"] = pd.array(
        strategy_values.to_numpy(copy=False),
        dtype=_strategy_pandas_dtype(strategy_type),
    )
    return out


def run(cfg: AppConfig, *, force: bool = False) -> None:
    """Compute game statistics and write them to parquet outputs.

    Args:
        cfg: Application configuration used to resolve file locations.
        force: When True, recompute even if the outputs appear up-to-date.
    """

    stage_log = stage_logger("game_stats", logger=LOGGER)
    stage_log.start()

    stage_dir = cfg.game_stats_stage_dir
    game_length_output = cfg.game_stats_output_path("game_length.parquet")
    margin_output = cfg.game_stats_output_path("margin_stats.parquet")
    pooled_game_length_output = cfg.game_stats_output_path("game_length_k_weighted.parquet")
    pooled_margin_output = cfg.game_stats_output_path("margin_k_weighted.parquet")
    configured_k_values = cfg.agreement_players()
    per_k_game_length_outputs = {
        k: cfg.per_k_subdir("game_stats", k) / "game_length.parquet" for k in configured_k_values
    }
    per_k_margin_outputs = {
        k: cfg.per_k_subdir("game_stats", k) / "margin_stats.parquet" for k in configured_k_values
    }
    rare_events_output = cfg.game_stats_output_path("rare_events.parquet")
    rare_events_details_output = cfg.game_stats_output_path("rare_events_details.parquet")
    stamp_path = stage_done_path(stage_dir, "game_stats")
    game_length_stamp = stage_done_path(stage_dir, "game_stats.game_length")
    margin_stamp = stage_done_path(stage_dir, "game_stats.margin")
    pooled_game_length_stamp = stage_done_path(stage_dir, "game_stats.game_length_pooled")
    pooled_margin_stamp = stage_done_path(stage_dir, "game_stats.margin_pooled")
    per_k_game_length_stamps = {
        k: stage_done_path(stage_dir, f"game_stats.game_length.{k}p") for k in configured_k_values
    }
    per_k_margin_stamps = {
        k: stage_done_path(stage_dir, f"game_stats.margin.{k}p") for k in configured_k_values
    }
    rare_events_summary_stamp = stage_done_path(stage_dir, "game_stats.rare_events_summary")
    rare_events_details_stamp = stage_done_path(stage_dir, "game_stats.rare_events_details")

    per_n_inputs = _discover_per_n_inputs(cfg)
    combined_path = cfg.curated_parquet
    input_paths: list[Path] = [p for _, p in per_n_inputs]
    if combined_path.exists():
        input_paths.append(combined_path)

    if not input_paths:
        stage_log.missing_input(
            "no curated parquet files found", analysis_dir=str(cfg.analysis_dir)
        )
        return

    write_details = cfg.analysis.rare_event_write_details
    required_outputs = [game_length_output, margin_output, rare_events_output]
    if write_details:
        required_outputs.append(rare_events_details_output)

    pooled_possible = bool(per_n_inputs)
    if pooled_possible:
        required_outputs.extend([pooled_game_length_output, pooled_margin_output])
    required_outputs.extend(per_k_game_length_outputs.values())
    required_outputs.extend(per_k_margin_outputs.values())

    if not force and stage_is_up_to_date(
        stamp_path,
        inputs=input_paths,
        outputs=required_outputs,
        config_sha=cfg.config_sha,
    ):
        LOGGER.info(
            "Game-length stats up-to-date",
            extra={
                "stage": "game_stats",
                "game_length_output": str(game_length_output),
                "margin_output": str(margin_output),
                "stamp": str(stamp_path),
            },
        )
        return

    game_length_up_to_date = not force and stage_is_up_to_date(
        game_length_stamp,
        inputs=input_paths,
        outputs=[game_length_output],
        config_sha=cfg.config_sha,
    )
    margin_up_to_date = not force and stage_is_up_to_date(
        margin_stamp,
        inputs=input_paths,
        outputs=[margin_output],
        config_sha=cfg.config_sha,
    )
    rare_events_summary_up_to_date = not force and stage_is_up_to_date(
        rare_events_summary_stamp,
        inputs=input_paths,
        outputs=[rare_events_output],
        config_sha=cfg.config_sha,
    )

    rare_events_details_up_to_date = True
    if write_details:
        rare_events_details_up_to_date = not force and stage_is_up_to_date(
            rare_events_details_stamp,
            inputs=input_paths,
            outputs=[rare_events_details_output],
            config_sha=cfg.config_sha,
        )

    pooled_game_length_up_to_date = True
    pooled_margin_up_to_date = True
    if pooled_possible:
        pooled_inputs = [p for _, p in per_n_inputs]
        pooled_game_length_up_to_date = not force and stage_is_up_to_date(
            pooled_game_length_stamp,
            inputs=pooled_inputs,
            outputs=[pooled_game_length_output],
            config_sha=cfg.config_sha,
        )
        pooled_margin_up_to_date = not force and stage_is_up_to_date(
            pooled_margin_stamp,
            inputs=pooled_inputs,
            outputs=[pooled_margin_output],
            config_sha=cfg.config_sha,
        )

    per_k_game_length_up_to_date = {
        k: (not force)
        and stage_is_up_to_date(
            per_k_game_length_stamps[k],
            inputs=input_paths,
            outputs=[per_k_game_length_outputs[k]],
            config_sha=cfg.config_sha,
        )
        for k in configured_k_values
    }
    per_k_margin_up_to_date = {
        k: (not force)
        and stage_is_up_to_date(
            per_k_margin_stamps[k],
            inputs=input_paths,
            outputs=[per_k_margin_outputs[k]],
            config_sha=cfg.config_sha,
        )
        for k in configured_k_values
    }

    if (
        game_length_up_to_date
        and margin_up_to_date
        and pooled_game_length_up_to_date
        and pooled_margin_up_to_date
        and all(per_k_game_length_up_to_date.values())
        and all(per_k_margin_up_to_date.values())
        and rare_events_summary_up_to_date
        and rare_events_details_up_to_date
    ):
        LOGGER.info(
            "Game-length stats up-to-date",
            extra={
                "stage": "game_stats",
                "game_length_output": str(game_length_output),
                "margin_output": str(margin_output),
                "stamp": str(stamp_path),
            },
        )
        return

    LOGGER.info(
        "Computing game-length stats",
        extra={
            "stage": "game_stats",
            "analysis_dir": str(cfg.analysis_dir),
            "game_length_output": str(game_length_output),
            "margin_output": str(margin_output),
            "force": force,
        },
    )

    combined_rows: int | None = None
    combined: pd.DataFrame | None = None
    if not game_length_up_to_date:
        strategy_stats = _per_strategy_stats(per_n_inputs)
        global_stats = _global_stats(combined_path) if combined_path.exists() else pd.DataFrame()

        combined = pd.concat([strategy_stats, global_stats], ignore_index=True)
        if combined.empty:
            stage_log.missing_input("no rows available to summarize")
            return
        combined = _coerce_strategy_dtype(combined, _strategy_arrow_type(per_n_inputs))
        combined = _downcast_integer_stats(combined, columns=("n_players", "observations"))

        table = pa.Table.from_pandas(combined, preserve_index=False)
        write_parquet_atomic(table, game_length_output, codec=cfg.parquet_codec)
        combined_rows = len(combined)
        write_stage_done(
            game_length_stamp,
            inputs=input_paths,
            outputs=[game_length_output],
            config_sha=cfg.config_sha,
        )

    margin_stats: pd.DataFrame | None = None
    if not margin_up_to_date:
        margin_stats = _per_strategy_margin_stats(
            per_n_inputs, thresholds=cfg.game_stats_margin_thresholds
        )
        if margin_stats.empty:
            stage_log.missing_input("no margins available to summarize")
            return
        margin_stats = _coerce_strategy_dtype(margin_stats, _strategy_arrow_type(per_n_inputs))
        margin_stats = _downcast_integer_stats(margin_stats, columns=("n_players", "observations"))

        margin_table = pa.Table.from_pandas(margin_stats, preserve_index=False)
        write_parquet_atomic(margin_table, margin_output, codec=cfg.parquet_codec)
        write_stage_done(
            margin_stamp,
            inputs=input_paths,
            outputs=[margin_output],
            config_sha=cfg.config_sha,
        )

    if not all(per_k_game_length_up_to_date.values()):
        if combined is None:
            combined = pd.read_parquet(game_length_output)
        for k in configured_k_values:
            if per_k_game_length_up_to_date[k]:
                continue
            per_k_game_length = combined.loc[combined["n_players"] == k].copy()
            per_k_game_length_table = pa.Table.from_pandas(per_k_game_length, preserve_index=False)
            write_parquet_atomic(
                per_k_game_length_table,
                per_k_game_length_outputs[k],
                codec=cfg.parquet_codec,
            )
            write_stage_done(
                per_k_game_length_stamps[k],
                inputs=input_paths,
                outputs=[per_k_game_length_outputs[k]],
                config_sha=cfg.config_sha,
            )

    if not all(per_k_margin_up_to_date.values()):
        if margin_stats is None:
            margin_stats = pd.read_parquet(margin_output)
        for k in configured_k_values:
            if per_k_margin_up_to_date[k]:
                continue
            per_k_margin = margin_stats.loc[margin_stats["n_players"] == k].copy()
            per_k_margin_table = pa.Table.from_pandas(per_k_margin, preserve_index=False)
            write_parquet_atomic(
                per_k_margin_table,
                per_k_margin_outputs[k],
                codec=cfg.parquet_codec,
            )
            write_stage_done(
                per_k_margin_stamps[k],
                inputs=input_paths,
                outputs=[per_k_margin_outputs[k]],
                config_sha=cfg.config_sha,
            )

    if pooled_possible and not (pooled_game_length_up_to_date and pooled_margin_up_to_date):
        pooling_scheme = _normalize_pooling_scheme(cfg.analysis.pooling_weights)
        weights_by_k = dict(cfg.analysis.pooling_weights_by_k or {})
        if pooling_scheme == "config" and not weights_by_k:
            raise ValueError("analysis.pooling_weights_by_k must be set for config pooling")
        LOGGER.info(
            "Computing pooled game stats",
            extra={
                "stage": "game_stats",
                "pooling_scheme": pooling_scheme,
                "weights_by_k": weights_by_k or None,
            },
        )

        if not pooled_game_length_up_to_date:
            pooled_game_length = _pooled_strategy_stats(
                per_n_inputs,
                pooling_scheme=pooling_scheme,
                weights_by_k=weights_by_k,
            )
            if pooled_game_length.empty:
                stage_log.missing_input("no pooled game-length rows available")
                return
            pooled_game_length = _coerce_strategy_dtype(
                pooled_game_length, _strategy_arrow_type(per_n_inputs)
            )
            pooled_game_length = _downcast_integer_stats(
                pooled_game_length, columns=("observations",)
            )
            pooled_table = pa.Table.from_pandas(pooled_game_length, preserve_index=False)
            write_parquet_atomic(pooled_table, pooled_game_length_output, codec=cfg.parquet_codec)
            write_stage_done(
                pooled_game_length_stamp,
                inputs=[p for _, p in per_n_inputs],
                outputs=[pooled_game_length_output],
                config_sha=cfg.config_sha,
            )

        if not pooled_margin_up_to_date:
            pooled_margin = _pooled_margin_stats(
                per_n_inputs,
                thresholds=cfg.game_stats_margin_thresholds,
                pooling_scheme=pooling_scheme,
                weights_by_k=weights_by_k,
            )
            if pooled_margin.empty:
                stage_log.missing_input("no pooled margin rows available")
                return
            pooled_margin = _coerce_strategy_dtype(
                pooled_margin, _strategy_arrow_type(per_n_inputs)
            )
            pooled_margin = _downcast_integer_stats(pooled_margin, columns=("observations",))
            pooled_margin_table = pa.Table.from_pandas(pooled_margin, preserve_index=False)
            write_parquet_atomic(pooled_margin_table, pooled_margin_output, codec=cfg.parquet_codec)
            write_stage_done(
                pooled_margin_stamp,
                inputs=[p for _, p in per_n_inputs],
                outputs=[pooled_margin_output],
                config_sha=cfg.config_sha,
            )

    if not (rare_events_summary_up_to_date and rare_events_details_up_to_date):
        resolved_thresholds, resolved_target_score = _resolve_rare_event_thresholds(
            per_n_inputs,
            thresholds=cfg.game_stats_margin_thresholds,
            target_score=cfg.rare_event_target_score,
            margin_quantile=cfg.analysis.rare_event_margin_quantile,
            target_rate=cfg.analysis.rare_event_target_rate,
        )

        if not rare_events_summary_up_to_date:
            rare_event_rows = _rare_event_flags(
                per_n_inputs,
                thresholds=resolved_thresholds,
                target_score=resolved_target_score,
                output_path=rare_events_output,
                codec=cfg.parquet_codec,
            )
            if rare_event_rows == 0:
                raise RuntimeError("game-stats: no rare events available to summarize")
            write_stage_done(
                rare_events_summary_stamp,
                inputs=input_paths,
                outputs=[rare_events_output],
                config_sha=cfg.config_sha,
            )

        if write_details and not rare_events_details_up_to_date:
            rare_event_rows = _rare_event_details(
                per_n_inputs,
                thresholds=resolved_thresholds,
                target_score=resolved_target_score,
                output_path=rare_events_details_output,
                codec=cfg.parquet_codec,
            )
            if rare_event_rows == 0:
                raise RuntimeError("game-stats: no rare events available to detail")
            write_stage_done(
                rare_events_details_stamp,
                inputs=input_paths,
                outputs=[rare_events_details_output],
                config_sha=cfg.config_sha,
            )
    write_stage_done(
        stamp_path,
        inputs=input_paths,
        outputs=required_outputs,
        config_sha=cfg.config_sha,
    )

    LOGGER.info(
        "Game-length stats written",
        extra={
            "stage": "game_stats",
            "rows": combined_rows,
            "game_length_output": str(game_length_output),
            "margin_output": str(margin_output),
        },
    )


def _discover_per_n_inputs(cfg: AppConfig) -> list[tuple[int, Path]]:
    """Return discovered per-``n`` curated parquets in ``analysis/data``."""

    inputs: list[tuple[int, Path]] = []
    for p_dir in sorted(cfg.data_dir.glob("*p")):
        if not p_dir.is_dir():
            continue
        try:
            n_players = int(p_dir.name.removesuffix("p"))
        except ValueError:
            continue

        preferred = p_dir / cfg.curated_rows_name
        legacy = p_dir / f"{n_players}p_ingested_rows.parquet"
        candidate = preferred if preferred.exists() else legacy
        if candidate.exists():
            inputs.append((n_players, candidate))
    return inputs


def _per_strategy_stats(per_n_inputs: Sequence[tuple[int, Path]]) -> pd.DataFrame:
    """Compute statistics grouped by strategy and player count."""

    grouped_rounds: dict[tuple[Scalar, int], list[np.ndarray[Any, np.dtype[np.float64]]]] = {}
    for n_players, path in per_n_inputs:
        ds_in = ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        if not strategy_cols:
            LOGGER.warning(
                "Per-N parquet missing strategy columns",
                extra={"stage": "game_stats", "path": str(path)},
            )
            continue

        scanner = ds_in.scanner(columns=["n_rounds", *strategy_cols], batch_size=65_536)
        for batch in scanner.to_batches():
            df = batch.to_pandas(categories=strategy_cols)
            if df.empty:
                continue
            melted = df.melt(id_vars=["n_rounds"], value_vars=strategy_cols, value_name="strategy")
            melted = melted.dropna(subset=["strategy"])
            if melted.empty:
                continue

            melted["n_rounds"] = pd.to_numeric(melted["n_rounds"], errors="coerce")
            melted = melted.dropna(subset=["n_rounds"])
            if melted.empty:
                continue

            batch_grouped = melted.groupby("strategy", observed=True, sort=False)["n_rounds"]
            for strategy, series in batch_grouped:
                key = (strategy, n_players)
                grouped_rounds.setdefault(key, []).append(series.to_numpy(dtype=float, copy=False))

    if not grouped_rounds:
        return pd.DataFrame(
            columns=[
                "summary_level",
                "strategy",
                "n_players",
                "observations",
                "mean_rounds",
                "median_rounds",
                "std_rounds",
                "p10_rounds",
                "p50_rounds",
                "p90_rounds",
                "prob_rounds_le_5",
                "prob_rounds_le_10",
                "prob_rounds_ge_20",
            ]
        )

    rows: list[dict[str, StatValue]] = []
    for (strategy, n_players), chunks in grouped_rounds.items():
        values = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if values.size == 0:
            continue
        rows.append(
            {
                "summary_level": "strategy",
                "strategy": _strategy_stat_value(strategy),
                "n_players": n_players,
                "observations": _strategy_key_to_int(values.size, field="observations"),
                "mean_rounds": float(np.mean(values)),
                "median_rounds": float(np.quantile(values, 0.5)),
                "std_rounds": float(np.std(values, ddof=0)),
                "p10_rounds": float(np.quantile(values, 0.1)),
                "p50_rounds": float(np.quantile(values, 0.5)),
                "p90_rounds": float(np.quantile(values, 0.9)),
                "prob_rounds_le_5": float(np.mean(values <= 5)),
                "prob_rounds_le_10": float(np.mean(values <= 10)),
                "prob_rounds_ge_20": float(np.mean(values >= 20)),
            }
        )

    stats = pd.DataFrame(rows)
    return stats[
        [
            "summary_level",
            "strategy",
            "n_players",
            "observations",
            "mean_rounds",
            "median_rounds",
            "std_rounds",
            "p10_rounds",
            "p50_rounds",
            "p90_rounds",
            "prob_rounds_le_5",
            "prob_rounds_le_10",
            "prob_rounds_ge_20",
        ]
    ]


def _normalize_pooling_scheme(pooling_scheme: str) -> str:
    """Normalize pooling scheme names for pooled game stats."""

    normalized = pooling_scheme.strip().lower().replace("_", "-")
    if normalized in {"game-count", "gamecount", "count"}:
        return "game-count"
    if normalized in {"equal-k", "equalk", "equal"}:
        return "equal-k"
    if normalized in {"config", "config-provided", "custom"}:
        return "config"
    raise ValueError(f"Unknown pooling scheme: {pooling_scheme!r}")


def _pooling_weights_for_rows(
    n_players: pd.Series,
    *,
    pooling_scheme: str,
    weights_by_k: dict[int, float],
) -> pd.Series:
    """Return per-row weights based on pooling scheme and player count."""

    counts = n_players.value_counts().to_dict()
    if pooling_scheme == "game-count":
        return pd.Series(1.0, index=n_players.index)

    if pooling_scheme == "equal-k":
        return n_players.map(lambda k: 1.0 / counts.get(k, 1))

    if pooling_scheme == "config":
        missing = sorted({to_int(k) for k in counts} - set(weights_by_k))
        if missing:
            LOGGER.warning(
                "Missing pooling weights for player counts; treating as zero",
                extra={"stage": "game_stats", "missing": missing},
            )
        return n_players.map(lambda k: float(weights_by_k.get(to_int(k), 0.0)) / counts.get(k, 1))

    raise ValueError(f"Unknown pooling scheme: {pooling_scheme!r}")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Compute a weighted quantile from value and weight arrays."""

    if values.size == 0:
        return float("nan")
    if quantile <= 0.0:
        return float(np.nanmin(values))
    if quantile >= 1.0:
        return float(np.nanmax(values))
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = weights[sorter]
    cumulative = np.cumsum(weights_sorted)
    total = cumulative[-1]
    if total <= 0:
        return float("nan")
    cutoff = quantile * total
    idx = _strategy_key_to_int(np.searchsorted(cumulative, cutoff, side="left"), field="index")
    idx = min(max(idx, 0), values_sorted.size - 1)
    return float(values_sorted[idx])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return float("nan")
    return float(np.average(values, weights=weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return float("nan")
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    return float(math.sqrt(variance))


def _pooled_strategy_stats(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    pooling_scheme: str,
    weights_by_k: dict[int, float],
) -> pd.DataFrame:
    """Compute pooled game-length statistics across player counts."""

    grouped_values: dict[Scalar, list[np.ndarray[Any, np.dtype[np.float64]]]] = {}
    grouped_weights: dict[Scalar, list[np.ndarray[Any, np.dtype[np.float64]]]] = {}
    for n_players, path in per_n_inputs:
        ds_in = ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        if not strategy_cols:
            LOGGER.warning(
                "Per-N parquet missing strategy columns",
                extra={"stage": "game_stats", "path": str(path)},
            )
            continue

        scanner = ds_in.scanner(columns=["n_rounds", *strategy_cols], batch_size=65_536)
        for batch in scanner.to_batches():
            df = batch.to_pandas(categories=strategy_cols)
            if df.empty:
                continue
            melted = df.melt(id_vars=["n_rounds"], value_vars=strategy_cols, value_name="strategy")
            melted = melted.dropna(subset=["strategy"])
            if melted.empty:
                continue
            melted["n_rounds"] = pd.to_numeric(melted["n_rounds"], errors="coerce")
            melted = melted.dropna(subset=["n_rounds"])
            if melted.empty:
                continue
            melted["n_players"] = n_players
            melted["weight"] = _pooling_weights_for_rows(
                melted["n_players"],
                pooling_scheme=pooling_scheme,
                weights_by_k=weights_by_k,
            )

            batch_grouped = melted.groupby("strategy", observed=True, sort=False)
            for strategy, group in batch_grouped:
                grouped_values.setdefault(strategy, []).append(
                    group["n_rounds"].to_numpy(dtype=float, copy=False)
                )
                grouped_weights.setdefault(strategy, []).append(
                    group["weight"].to_numpy(dtype=float, copy=False)
                )

    if not grouped_values:
        return pd.DataFrame(
            columns=[
                "summary_level",
                "strategy",
                "observations",
                "mean_rounds",
                "median_rounds",
                "std_rounds",
                "p10_rounds",
                "p50_rounds",
                "p90_rounds",
                "prob_rounds_le_5",
                "prob_rounds_le_10",
                "prob_rounds_ge_20",
            ]
        )
    pooled_rows: list[dict[str, StatValue]] = []
    for strategy, chunks in grouped_values.items():
        values = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        weight_chunks = grouped_weights.get(strategy, [])
        weights = np.concatenate(weight_chunks) if len(weight_chunks) > 1 else weight_chunks[0]
        weight_total = weights.sum()
        if not math.isfinite(weight_total) or weight_total <= 0:
            continue
        pooled_rows.append(
            {
                "summary_level": "pooled",
                "strategy": _strategy_stat_value(strategy),
                "observations": _strategy_key_to_int(values.size, field="observations"),
                "mean_rounds": _weighted_mean(values, weights),
                "median_rounds": _weighted_quantile(values, weights, 0.5),
                "std_rounds": _weighted_std(values, weights),
                "p10_rounds": _weighted_quantile(values, weights, 0.1),
                "p50_rounds": _weighted_quantile(values, weights, 0.5),
                "p90_rounds": _weighted_quantile(values, weights, 0.9),
                "prob_rounds_le_5": _weighted_mean(values <= 5, weights),
                "prob_rounds_le_10": _weighted_mean(values <= 10, weights),
                "prob_rounds_ge_20": _weighted_mean(values >= 20, weights),
            }
        )

    pooled_df = pd.DataFrame(
        pooled_rows,
        columns=[
            "summary_level",
            "strategy",
            "observations",
            "mean_rounds",
            "median_rounds",
            "std_rounds",
            "p10_rounds",
            "p50_rounds",
            "p90_rounds",
            "prob_rounds_le_5",
            "prob_rounds_le_10",
            "prob_rounds_ge_20",
        ],
    )
    return pooled_df


def _pooled_margin_stats(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
    pooling_scheme: str,
    weights_by_k: dict[int, float],
) -> pd.DataFrame:
    """Compute pooled victory-margin statistics across player counts."""

    grouped_runner: dict[Scalar, list[np.ndarray[Any, np.dtype[np.float64]]]] = {}
    grouped_spread: dict[Scalar, list[np.ndarray[Any, np.dtype[np.float64]]]] = {}
    grouped_weights: dict[Scalar, list[np.ndarray[Any, np.dtype[np.float64]]]] = {}
    for n_players, path in per_n_inputs:
        ds_in = ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        score_cols = [
            name for name in ds_in.schema.names if name.startswith("P") and name.endswith("_score")
        ]

        if not strategy_cols:
            LOGGER.warning(
                "Per-N parquet missing strategy columns",
                extra={"stage": "game_stats", "path": str(path)},
            )
            continue

        if not score_cols:
            LOGGER.warning(
                "Per-N parquet missing seat score columns; skipping margins",
                extra={"stage": "game_stats", "path": str(path)},
            )
            continue

        scanner = ds_in.scanner(columns=[*score_cols, *strategy_cols], batch_size=65_536)
        for batch in scanner.to_batches():
            df = batch.to_pandas(categories=strategy_cols)
            if df.empty:
                continue
            margin_cols = _compute_margin_columns(df, score_cols)
            df = df.assign(
                margin_runner_up=margin_cols["margin_runner_up"],
                score_spread=margin_cols["score_spread"],
            )
            melted = df.melt(
                id_vars=["margin_runner_up", "score_spread"],
                value_vars=strategy_cols,
                value_name="strategy",
            )
            melted = melted.dropna(subset=["strategy"])
            if melted.empty:
                continue

            melted["margin_runner_up"] = pd.to_numeric(melted["margin_runner_up"], errors="coerce")
            melted["score_spread"] = pd.to_numeric(melted["score_spread"], errors="coerce")
            melted = melted.dropna(subset=["margin_runner_up"])
            if melted.empty:
                continue

            melted["n_players"] = n_players
            melted["weight"] = _pooling_weights_for_rows(
                melted["n_players"],
                pooling_scheme=pooling_scheme,
                weights_by_k=weights_by_k,
            )

            batch_grouped = melted.groupby("strategy", observed=True, sort=False)
            for strategy, group in batch_grouped:
                grouped_runner.setdefault(strategy, []).append(
                    group["margin_runner_up"].to_numpy(dtype=float, copy=False)
                )
                grouped_spread.setdefault(strategy, []).append(
                    group["score_spread"].to_numpy(dtype=float, copy=False)
                )
                grouped_weights.setdefault(strategy, []).append(
                    group["weight"].to_numpy(dtype=float, copy=False)
                )

    if not grouped_runner:
        columns = [
            "summary_level",
            "strategy",
            "observations",
            "mean_margin_runner_up",
            "median_margin_runner_up",
            "std_margin_runner_up",
            *[f"prob_margin_runner_up_le_{thr}" for thr in thresholds],
            "mean_score_spread",
            "median_score_spread",
            "std_score_spread",
            *[f"prob_score_spread_le_{thr}" for thr in thresholds],
        ]
        return pd.DataFrame(columns=columns)

    pooled_rows: list[dict[str, StatValue]] = []
    for strategy, runner_chunks in grouped_runner.items():
        runner_vals = np.concatenate(runner_chunks) if len(runner_chunks) > 1 else runner_chunks[0]
        spread_chunks = grouped_spread.get(strategy, [])
        spread_vals = np.concatenate(spread_chunks) if len(spread_chunks) > 1 else spread_chunks[0]
        weight_chunks = grouped_weights.get(strategy, [])
        weights = np.concatenate(weight_chunks) if len(weight_chunks) > 1 else weight_chunks[0]
        weight_total = weights.sum()
        if not math.isfinite(weight_total) or weight_total <= 0:
            continue

        pooled_row: dict[str, StatValue] = {
            "summary_level": "pooled",
            "strategy": _strategy_stat_value(strategy),
            "observations": _strategy_key_to_int(runner_vals.size, field="observations"),
            "mean_margin_runner_up": _weighted_mean(runner_vals, weights),
            "median_margin_runner_up": _weighted_quantile(runner_vals, weights, 0.5),
            "std_margin_runner_up": _weighted_std(runner_vals, weights),
            "mean_score_spread": _weighted_mean(spread_vals, weights),
            "median_score_spread": _weighted_quantile(spread_vals, weights, 0.5),
            "std_score_spread": _weighted_std(spread_vals, weights),
        }
        for thr in thresholds:
            pooled_row[f"prob_margin_runner_up_le_{thr}"] = _weighted_mean(
                runner_vals <= thr, weights
            )
            pooled_row[f"prob_score_spread_le_{thr}"] = _weighted_mean(spread_vals <= thr, weights)
        pooled_rows.append(pooled_row)

    ordered_cols = [
        "summary_level",
        "strategy",
        "observations",
        "mean_margin_runner_up",
        "median_margin_runner_up",
        "std_margin_runner_up",
        *[f"prob_margin_runner_up_le_{thr}" for thr in thresholds],
        "mean_score_spread",
        "median_score_spread",
        "std_score_spread",
        *[f"prob_score_spread_le_{thr}" for thr in thresholds],
    ]
    return pd.DataFrame(pooled_rows, columns=ordered_cols)


def _per_strategy_margin_stats(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
) -> pd.DataFrame:
    """Compute victory-margin statistics grouped by strategy and player count."""

    grouped_runner: dict[tuple[Scalar, int], list[np.ndarray[Any, np.dtype[np.float64]]]] = {}
    grouped_spread: dict[tuple[Scalar, int], list[np.ndarray[Any, np.dtype[np.float64]]]] = {}
    for n_players, path in per_n_inputs:
        ds_in = ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        score_cols = [
            name for name in ds_in.schema.names if name.startswith("P") and name.endswith("_score")
        ]

        if not strategy_cols:
            LOGGER.warning(
                "Per-N parquet missing strategy columns",
                extra={"stage": "game_stats", "path": str(path)},
            )
            continue

        if not score_cols:
            LOGGER.warning(
                "Per-N parquet missing seat score columns; skipping margins",
                extra={"stage": "game_stats", "path": str(path)},
            )
            continue

        scanner = ds_in.scanner(columns=[*score_cols, *strategy_cols], batch_size=65_536)
        for batch in scanner.to_batches():
            df = batch.to_pandas(categories=strategy_cols)
            if df.empty:
                continue
            margin_cols = _compute_margin_columns(df, score_cols)
            df = df.assign(
                margin_runner_up=margin_cols["margin_runner_up"],
                score_spread=margin_cols["score_spread"],
            )
            melted = df.melt(
                id_vars=["margin_runner_up", "score_spread"],
                value_vars=strategy_cols,
                value_name="strategy",
            )
            melted = melted.dropna(subset=["strategy"])
            if melted.empty:
                continue

            melted["margin_runner_up"] = pd.to_numeric(melted["margin_runner_up"], errors="coerce")
            melted["score_spread"] = pd.to_numeric(melted["score_spread"], errors="coerce")
            melted = melted.dropna(subset=["margin_runner_up"])
            if melted.empty:
                continue

            batch_grouped = melted.groupby("strategy", observed=True, sort=False)
            for strategy, group in batch_grouped:
                key = (strategy, n_players)
                grouped_runner.setdefault(key, []).append(
                    group["margin_runner_up"].to_numpy(dtype=float, copy=False)
                )
                grouped_spread.setdefault(key, []).append(
                    group["score_spread"].to_numpy(dtype=float, copy=False)
                )

    if not grouped_runner:
        columns = [
            "summary_level",
            "strategy",
            "n_players",
            "observations",
            "mean_margin_runner_up",
            "median_margin_runner_up",
            "std_margin_runner_up",
            *[f"prob_margin_runner_up_le_{thr}" for thr in thresholds],
            "mean_score_spread",
            "median_score_spread",
            "std_score_spread",
            *[f"prob_score_spread_le_{thr}" for thr in thresholds],
        ]
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, StatValue]] = []
    for (strategy, n_players), runner_chunks in grouped_runner.items():
        runner_vals = np.concatenate(runner_chunks) if len(runner_chunks) > 1 else runner_chunks[0]
        spread_chunks = grouped_spread.get((strategy, n_players), [])
        spread_vals = np.concatenate(spread_chunks) if len(spread_chunks) > 1 else spread_chunks[0]
        if runner_vals.size == 0:
            continue
        row: dict[str, StatValue] = {
            "summary_level": "strategy",
            "strategy": _strategy_stat_value(strategy),
            "n_players": n_players,
            "observations": _strategy_key_to_int(runner_vals.size, field="observations"),
            "mean_margin_runner_up": float(np.mean(runner_vals)),
            "median_margin_runner_up": float(np.quantile(runner_vals, 0.5)),
            "std_margin_runner_up": float(np.std(runner_vals, ddof=0)),
            "mean_score_spread": float(np.mean(spread_vals)),
            "median_score_spread": float(np.quantile(spread_vals, 0.5)),
            "std_score_spread": float(np.std(spread_vals, ddof=0)),
        }
        for thr in thresholds:
            row[f"prob_margin_runner_up_le_{thr}"] = float(np.mean(runner_vals <= thr))
            row[f"prob_score_spread_le_{thr}"] = float(np.mean(spread_vals <= thr))
        rows.append(row)

    stats = pd.DataFrame(rows)

    ordered_cols = [
        "summary_level",
        "strategy",
        "n_players",
        "observations",
        "mean_margin_runner_up",
        "median_margin_runner_up",
        "std_margin_runner_up",
        *[f"prob_margin_runner_up_le_{thr}" for thr in thresholds],
        "mean_score_spread",
        "median_score_spread",
        "std_score_spread",
        *[f"prob_score_spread_le_{thr}" for thr in thresholds],
    ]
    return stats[ordered_cols]


def _rare_event_summary(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
    target_score: int,
    output_path: Path,
    codec: Compression,
) -> int:
    """Compute aggregated rare-event counts."""

    flags = ["multi_reached_target", *[f"margin_le_{thr}" for thr in thresholds]]
    column_order = [
        "summary_level",
        "strategy",
        "n_players",
        "margin_runner_up",
        "score_spread",
        "multi_reached_target",
        "observations",
        *[f"margin_le_{thr}" for thr in thresholds],
    ]
    (
        strategy_sums,
        global_sums,
        rows_available,
        _max_flag_count,
        max_observations,
    ) = _collect_rare_event_counts(
        per_n_inputs,
        thresholds=thresholds,
        target_score=target_score,
    )

    if rows_available == 0:
        return 0

    strategy_arrow = _strategy_arrow_type(per_n_inputs)
    flag_arrow = pa.float64()
    obs_dtype, obs_arrow = _select_int_dtype(max_observations)
    fields: list[pa.Field] = [
        pa.field("summary_level", pa.string()),
        pa.field("strategy", strategy_arrow),
        pa.field("n_players", pa.int32()),
        pa.field("margin_runner_up", pa.float64()),
        pa.field("score_spread", pa.float64()),
        pa.field("multi_reached_target", flag_arrow),
        pa.field("observations", obs_arrow),
        *[pa.field(f"margin_le_{thr}", flag_arrow) for thr in thresholds],
    ]
    schema = pa.schema(fields)
    strategy_rows: list[dict[str, object]] = []
    for (strategy, players), sums in strategy_sums.items():
        observations = sums["observations"]
        summary_row: dict[str, object] = {
            "summary_level": "strategy",
            "strategy": _strategy_stat_value(strategy),
            "n_players": players,
            "margin_runner_up": pd.NA,
            "score_spread": pd.NA,
            "observations": observations,
        }
        for flag in flags:
            summary_row[flag] = sums[flag]
        strategy_rows.append(summary_row)
    strategy_summary = pd.DataFrame(strategy_rows, columns=column_order)

    global_rows: list[dict[str, object]] = []
    for players, sums in global_sums.items():
        observations = sums["observations"]
        summary_row = {
            "summary_level": "n_players",
            "strategy": pd.NA,
            "n_players": players,
            "margin_runner_up": pd.NA,
            "score_spread": pd.NA,
            "observations": observations,
        }
        for flag in flags:
            summary_row[flag] = sums[flag]
        global_rows.append(summary_row)
    global_summary = pd.DataFrame(global_rows, columns=column_order)

    summary_df = pd.concat([strategy_summary, global_summary], ignore_index=True)
    summary_df["strategy"] = summary_df["strategy"].astype("Int64").array
    summary_df = _downcast_integer_stats(summary_df, columns=("n_players", "observations", *flags))
    summary_table = pa.Table.from_pandas(summary_df, preserve_index=False, schema=schema)
    write_parquet_atomic(summary_table, output_path, codec=codec)
    return summary_table.num_rows


def _rare_event_flags(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
    target_score: int,
    output_path: Path,
    codec: Compression,
) -> int:
    """Write combined per-game and summary rare-event rows to parquet."""

    flags = ["multi_reached_target", *[f"margin_le_{thr}" for thr in thresholds]]
    (
        strategy_sums,
        global_sums,
        rows_available,
        _max_flag_count,
        max_observations,
    ) = _collect_rare_event_counts(
        per_n_inputs,
        thresholds=thresholds,
        target_score=target_score,
    )
    if rows_available == 0:
        return 0

    strategy_arrow = _strategy_arrow_type(per_n_inputs)
    strategy_dtype = _strategy_numpy_dtype(strategy_arrow)
    flag_dtype = np.float64
    flag_arrow = pa.float64()
    obs_dtype, obs_arrow = _select_int_dtype(max_observations)
    player_dtype = np.int32
    fields: list[pa.Field] = [
        pa.field("summary_level", pa.string()),
        pa.field("strategy", strategy_arrow),
        pa.field("n_players", pa.int32()),
        pa.field("margin_runner_up", pa.float64()),
        pa.field("score_spread", pa.float64()),
        pa.field("multi_reached_target", flag_arrow),
        pa.field("observations", obs_arrow),
        *[pa.field(f"margin_le_{thr}", flag_arrow) for thr in thresholds],
    ]
    schema = pa.schema(fields)

    rows_written = 0
    writer = ParquetShardWriter(str(output_path), schema=schema, compression=codec)
    try:
        for n_players, path in per_n_inputs:
            ds_in = ds.dataset(path)
            strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
            score_cols = [
                name
                for name in ds_in.schema.names
                if name.startswith("P") and name.endswith("_score")
            ]

            if not strategy_cols:
                LOGGER.warning(
                    "Per-N parquet missing strategy columns; skipping rare events",
                    extra={"stage": "game_stats", "path": str(path)},
                )
                continue

            if not score_cols:
                LOGGER.warning(
                    "Per-N parquet missing seat score columns; skipping rare events",
                    extra={"stage": "game_stats", "path": str(path)},
                )
                continue

            columns = [*strategy_cols, *score_cols]
            scanner = ds_in.scanner(columns=columns, batch_size=65_536)
            for batch in scanner.to_batches():
                df = batch.to_pandas(categories=strategy_cols)
                if df.empty:
                    continue
                margin_cols = _compute_margin_columns(df, score_cols)
                scores = df[score_cols].apply(pd.to_numeric, errors="coerce")
                multi_target = (scores >= target_score).sum(axis=1) >= 2
                flag_series: dict[str, pd.Series] = {"multi_reached_target": multi_target}
                for thr in thresholds:
                    flag_series[f"margin_le_{thr}"] = margin_cols["margin_runner_up"] <= thr
                any_flag = flag_series["multi_reached_target"].copy()
                for thr in thresholds:
                    any_flag |= flag_series[f"margin_le_{thr}"]
                base = df.loc[any_flag, strategy_cols].copy()
                if base.empty:
                    continue
                base["margin_runner_up"] = margin_cols["margin_runner_up"][any_flag]
                base["score_spread"] = margin_cols["score_spread"][any_flag]
                base["multi_reached_target"] = flag_series["multi_reached_target"][any_flag]
                for thr in thresholds:
                    base[f"margin_le_{thr}"] = flag_series[f"margin_le_{thr}"][any_flag]
                base["n_players"] = n_players
                melted = base.melt(
                    id_vars=[
                        "margin_runner_up",
                        "score_spread",
                        "multi_reached_target",
                        "n_players",
                        *[f"margin_le_{thr}" for thr in thresholds],
                    ],
                    value_vars=strategy_cols,
                    var_name="seat",
                    value_name="strategy",
                )
                melted = melted.dropna(subset=["strategy"])
                if melted.empty:
                    continue
                melted["strategy"] = melted["strategy"].astype("category")
                grouped = melted.groupby("strategy", observed=True, sort=False)
                for strategy, group in grouped:
                    count = _strategy_key_to_int(group.shape[0], field="observations")
                    strategy_value = _strategy_key_to_int(strategy)
                    per_game_data: dict[str, ArrowColumnData] = {
                        "summary_level": np.full(count, "game", dtype=object),
                        "strategy": np.full(count, strategy_value, dtype=strategy_dtype),
                        "n_players": np.full(count, n_players, dtype=player_dtype),
                        "margin_runner_up": group["margin_runner_up"].to_numpy(dtype=float),
                        "score_spread": group["score_spread"].to_numpy(dtype=float),
                        "multi_reached_target": group["multi_reached_target"].to_numpy(
                            dtype=flag_dtype
                        ),
                        "observations": np.ones(count, dtype=obs_dtype),
                    }
                    for thr in thresholds:
                        per_game_data[f"margin_le_{thr}"] = group[f"margin_le_{thr}"].to_numpy(
                            dtype=flag_dtype
                        )

                    writer.write_batch(pa.Table.from_pydict(per_game_data, schema=schema))
                    rows_written += count

        summary_rows: list[dict[str, object]] = []
        for (strategy, players), sums in strategy_sums.items():
            observations = sums["observations"]
            summary_row: dict[str, object] = {
                "summary_level": "strategy",
                "strategy": _strategy_stat_value(strategy),
                "n_players": players,
                "margin_runner_up": pd.NA,
                "score_spread": pd.NA,
                "observations": observations,
            }
            for flag in flags:
                summary_row[flag] = sums[flag] / observations if observations else float("nan")
            summary_rows.append(summary_row)

        for players, sums in global_sums.items():
            observations = sums["observations"]
            summary_row = {
                "summary_level": "n_players",
                "strategy": pd.NA,
                "n_players": players,
                "margin_runner_up": pd.NA,
                "score_spread": pd.NA,
                "observations": observations,
            }
            for flag in flags:
                summary_row[flag] = sums[flag] / observations if observations else float("nan")
            summary_rows.append(summary_row)

        summary_df = pd.DataFrame(
            summary_rows,
            columns=[
                "summary_level",
                "strategy",
                "n_players",
                "margin_runner_up",
                "score_spread",
                "multi_reached_target",
                "observations",
                *[f"margin_le_{thr}" for thr in thresholds],
            ],
        )
        summary_df["strategy"] = summary_df["strategy"].astype("Int64").array
        summary_df = _downcast_integer_stats(summary_df, columns=("n_players", "observations"))
        summary_table = pa.Table.from_pandas(summary_df, preserve_index=False, schema=schema)
        writer.write_batch(summary_table)
        rows_written += summary_table.num_rows
        return rows_written
    finally:
        writer.close(success=rows_written > 0)


def _rare_event_details(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
    target_score: int,
    output_path: Path,
    codec: Compression,
) -> int:
    """Write per-game rare-event rows to a separate parquet."""

    strategy_arrow = _strategy_arrow_type(per_n_inputs)
    strategy_dtype = _strategy_numpy_dtype(strategy_arrow)
    flag_dtype, flag_arrow = _select_int_dtype(1)
    obs_dtype, obs_arrow = _select_int_dtype(1)
    player_dtype = np.int32
    fields: list[pa.Field] = [
        pa.field("summary_level", pa.string()),
        pa.field("strategy", strategy_arrow),
        pa.field("n_players", pa.int32()),
        pa.field("margin_runner_up", pa.float64()),
        pa.field("score_spread", pa.float64()),
        pa.field("multi_reached_target", flag_arrow),
        pa.field("observations", obs_arrow),
        *[pa.field(f"margin_le_{thr}", flag_arrow) for thr in thresholds],
    ]
    schema = pa.schema(fields)

    rows_written = 0
    writer = ParquetShardWriter(str(output_path), schema=schema, compression=codec)
    try:
        for n_players, path in per_n_inputs:
            ds_in = ds.dataset(path)
            strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
            score_cols = [
                name
                for name in ds_in.schema.names
                if name.startswith("P") and name.endswith("_score")
            ]

            if not strategy_cols:
                LOGGER.warning(
                    "Per-N parquet missing strategy columns; skipping rare events",
                    extra={"stage": "game_stats", "path": str(path)},
                )
                continue

            if not score_cols:
                LOGGER.warning(
                    "Per-N parquet missing seat score columns; skipping rare events",
                    extra={"stage": "game_stats", "path": str(path)},
                )
                continue

            columns = [*strategy_cols, *score_cols]
            scanner = ds_in.scanner(columns=columns, batch_size=65_536)
            for batch in scanner.to_batches():
                df = batch.to_pandas(categories=strategy_cols)
                if df.empty:
                    continue
                margin_cols = _compute_margin_columns(df, score_cols)
                scores = df[score_cols].apply(pd.to_numeric, errors="coerce")
                multi_target = (scores >= target_score).sum(axis=1) >= 2
                flag_series: dict[str, pd.Series] = {"multi_reached_target": multi_target}
                for thr in thresholds:
                    flag_series[f"margin_le_{thr}"] = margin_cols["margin_runner_up"] <= thr
                any_flag = flag_series["multi_reached_target"].copy()
                for thr in thresholds:
                    any_flag |= flag_series[f"margin_le_{thr}"]
                base = df.loc[any_flag, strategy_cols].copy()
                if base.empty:
                    continue
                base["margin_runner_up"] = margin_cols["margin_runner_up"][any_flag]
                base["score_spread"] = margin_cols["score_spread"][any_flag]
                base["multi_reached_target"] = flag_series["multi_reached_target"][any_flag]
                for thr in thresholds:
                    base[f"margin_le_{thr}"] = flag_series[f"margin_le_{thr}"][any_flag]
                base["n_players"] = n_players
                melted = base.melt(
                    id_vars=[
                        "margin_runner_up",
                        "score_spread",
                        "multi_reached_target",
                        "n_players",
                        *[f"margin_le_{thr}" for thr in thresholds],
                    ],
                    value_vars=strategy_cols,
                    var_name="seat",
                    value_name="strategy",
                )
                melted = melted.dropna(subset=["strategy"])
                if melted.empty:
                    continue
                melted["strategy"] = melted["strategy"].astype("category")
                grouped = melted.groupby("strategy", observed=True, sort=False)
                for strategy, group in grouped:
                    count = _strategy_key_to_int(group.shape[0], field="observations")
                    strategy_value = _strategy_key_to_int(strategy)
                    per_game_data: dict[str, ArrowColumnData] = {
                        "summary_level": np.full(count, "game", dtype=object),
                        "strategy": np.full(count, strategy_value, dtype=strategy_dtype),
                        "n_players": np.full(count, n_players, dtype=player_dtype),
                        "margin_runner_up": group["margin_runner_up"].to_numpy(dtype=float),
                        "score_spread": group["score_spread"].to_numpy(dtype=float),
                        "multi_reached_target": group["multi_reached_target"].to_numpy(
                            dtype=flag_dtype
                        ),
                        "observations": np.ones(count, dtype=obs_dtype),
                    }
                    for thr in thresholds:
                        per_game_data[f"margin_le_{thr}"] = group[f"margin_le_{thr}"].to_numpy(
                            dtype=flag_dtype
                        )

                    writer.write_batch(pa.Table.from_pydict(per_game_data, schema=schema))
                    rows_written += count
        return rows_written
    finally:
        writer.close(success=rows_written > 0)


def _collect_rare_event_counts(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
    target_score: int,
) -> tuple[dict[tuple[int, int], dict[str, int]], dict[int, dict[str, int]], int, int, int]:
    """Gather rare-event counts for downstream downcasting decisions."""
    flags = ["multi_reached_target", *[f"margin_le_{thr}" for thr in thresholds]]
    strategy_sums: dict[tuple[int, int], dict[str, int]] = {}
    global_sums: dict[int, dict[str, int]] = {}
    rows_available = 0
    for n_players, path in per_n_inputs:
        ds_in = ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        score_cols = [
            name for name in ds_in.schema.names if name.startswith("P") and name.endswith("_score")
        ]

        if not strategy_cols or not score_cols:
            continue

        columns = [*strategy_cols, *score_cols]
        scanner = ds_in.scanner(columns=columns, batch_size=65_536)
        for batch in scanner.to_batches():
            df = batch.to_pandas(categories=strategy_cols)
            if df.empty:
                continue
            margin_cols = _compute_margin_columns(df, score_cols)
            scores = df[score_cols].apply(pd.to_numeric, errors="coerce")
            multi_target = (scores >= target_score).sum(axis=1) >= 2
            flag_series: dict[str, pd.Series] = {"multi_reached_target": multi_target}
            for thr in thresholds:
                flag_series[f"margin_le_{thr}"] = margin_cols["margin_runner_up"] <= thr
            base = df[strategy_cols].copy()
            base["multi_reached_target"] = flag_series["multi_reached_target"]
            for thr in thresholds:
                base[f"margin_le_{thr}"] = flag_series[f"margin_le_{thr}"]
            base["n_players"] = n_players
            melted = base.melt(
                id_vars=[
                    "multi_reached_target",
                    "n_players",
                    *[f"margin_le_{thr}" for thr in thresholds],
                ],
                value_vars=strategy_cols,
                var_name="seat",
                value_name="strategy",
            )
            melted = melted.dropna(subset=["strategy"])
            if melted.empty:
                continue
            melted["strategy"] = melted["strategy"].astype("category")
            grouped = melted.groupby("strategy", observed=True, sort=False)
            for key, group in grouped:
                count = _strategy_key_to_int(group.shape[0], field="observations")
                rows_available += count
                norm_key: tuple[int, int] = (
                    _strategy_key_to_int(key),
                    _strategy_key_to_int(n_players, field="n_players"),
                )
                strategy_entry = strategy_sums.setdefault(
                    norm_key,
                    {"observations": 0, **dict.fromkeys(flags, 0)},
                )
                strategy_entry["observations"] += count
                for flag in flags:
                    strategy_entry[flag] += _strategy_key_to_int(group[flag].sum(), field=flag)

                global_entry = global_sums.setdefault(
                    n_players, {"observations": 0, **dict.fromkeys(flags, 0)}
                )
                global_entry["observations"] += count
                for flag in flags:
                    global_entry[flag] += _strategy_key_to_int(group[flag].sum(), field=flag)

    max_flag_count = 0
    max_observations = 0
    for sums in list(strategy_sums.values()) + list(global_sums.values()):
        max_observations = max(max_observations, sums.get("observations", 0))
        for flag in flags:
            max_flag_count = max(max_flag_count, sums.get(flag, 0))
    return strategy_sums, global_sums, rows_available, max_flag_count, max_observations


def _resolve_rare_event_thresholds(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
    target_score: int,
    margin_quantile: float | None,
    target_rate: float | None,
) -> tuple[tuple[int, ...], int]:
    """Resolve rare-event thresholds, optionally based on streaming quantiles."""
    resolved_thresholds = tuple(to_int(thr) for thr in thresholds)
    resolved_target_score = to_int(target_score)
    needs_margin = margin_quantile is not None
    needs_target = target_rate is not None
    if not needs_margin and not needs_target:
        return resolved_thresholds, resolved_target_score

    if margin_quantile is not None and not 0.0 < margin_quantile < 1.0:
        raise ValueError("rare_event_margin_quantile must be between 0 and 1")
    if target_rate is not None and not 0.0 < target_rate < 1.0:
        raise ValueError("rare_event_target_rate must be between 0 and 1")

    margin_counts, target_counts = _collect_rare_event_histograms(
        per_n_inputs,
        need_margins=needs_margin,
        need_targets=needs_target,
    )
    if margin_quantile is not None:
        quantile_threshold = _quantile_from_counts(margin_counts, margin_quantile)
        if quantile_threshold is not None:
            resolved_thresholds = (quantile_threshold,)
    if target_rate is not None:
        quantile = 1.0 - target_rate
        target_threshold = _quantile_from_counts(target_counts, quantile)
        if target_threshold is not None:
            resolved_target_score = target_threshold
    return resolved_thresholds, resolved_target_score


def _collect_rare_event_histograms(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    need_margins: bool,
    need_targets: bool,
) -> tuple[dict[int, int], dict[int, int]]:
    margin_counts: dict[int, int] = {}
    target_counts: dict[int, int] = {}
    if not need_margins and not need_targets:
        return margin_counts, target_counts

    for _, path in per_n_inputs:
        ds_in = ds.dataset(path)
        score_cols = [
            name for name in ds_in.schema.names if name.startswith("P") and name.endswith("_score")
        ]
        if not score_cols:
            continue
        scanner = ds_in.scanner(columns=score_cols, batch_size=65_536)
        for batch in scanner.to_batches():
            df = batch.to_pandas()
            if df.empty:
                continue
            scores = df[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if need_margins:
                margin_runner_up, _score_spread = _compute_margin_arrays(scores)
                _update_int_histogram(margin_counts, margin_runner_up)
            if need_targets:
                second_scores = _second_highest(scores)
                _update_int_histogram(target_counts, second_scores)
    return margin_counts, target_counts


def _compute_margin_arrays(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute margin-of-victory and spread arrays from a numeric score array."""
    if scores.size == 0:
        return scores, scores
    valid_mask = np.isfinite(scores)
    valid_counts = valid_mask.sum(axis=1)
    with np.errstate(all="ignore"):
        max_scores = np.nanmax(scores, axis=1)
        min_scores = np.nanmin(scores, axis=1)
    score_spread = max_scores - min_scores
    second_scores = _second_highest(scores)
    margin_runner_up = max_scores - second_scores
    score_spread[valid_counts < 2] = np.nan
    margin_runner_up[valid_counts < 2] = np.nan
    return margin_runner_up.astype(float), score_spread.astype(float)


def _second_highest(scores: np.ndarray) -> np.ndarray:
    """Return the second-highest score per row, or NaN when fewer than two scores."""
    if scores.size == 0:
        return scores
    valid_mask = np.isfinite(scores)
    if scores.shape[1] < 2:
        return np.full(scores.shape[0], np.nan, dtype=float)
    safe_scores = np.where(valid_mask, scores, -np.inf)
    second = np.partition(safe_scores, -2, axis=1)[:, -2]
    valid_counts = valid_mask.sum(axis=1)
    second[valid_counts < 2] = np.nan
    return second.astype(float)


def _update_int_histogram(counts: dict[int, int], values: np.ndarray) -> None:
    """Update integer histogram counts from numeric values."""
    if values.size == 0:
        return
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return
    ints = np.rint(clean).astype(int)
    unique, freq = np.unique(ints, return_counts=True)
    for value, count in zip(unique, freq, strict=True):
        counts[int(value)] = counts.get(int(value), 0) + int(count)


def _quantile_from_counts(counts: dict[int, int], quantile: float) -> int | None:
    """Return the smallest value whose CDF exceeds the quantile."""
    if not counts:
        return None
    total = sum(counts.values())
    if total <= 0:
        return None
    if quantile <= 0.0:
        return int(min(counts))
    if quantile >= 1.0:
        return int(max(counts))
    cutoff = int(math.ceil(total * quantile))
    running = 0
    for value in sorted(counts):
        running += counts[value]
        if running >= cutoff:
            return int(value)
    return int(max(counts))


def _select_int_dtype(max_value: int) -> tuple[np.dtype[np.integer], pa.DataType]:
    """Pick an integer dtype with overflow protection."""
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8), pa.uint8()
    if max_value <= np.iinfo(np.int32).max:
        return np.dtype(np.int32), pa.int32()
    return np.dtype(np.int64), pa.int64()


def _downcast_integer_stats(df: pd.DataFrame, *, columns: Sequence[str]) -> pd.DataFrame:
    """Downcast integer columns for stats outputs when safe."""
    if df.empty:
        return df
    out = df.copy()
    int32_max = np.iinfo(np.int32).max
    for col in columns:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        non_null = values.dropna()
        if non_null.empty:
            continue
        if not np.all(np.isclose(non_null, np.floor(non_null))):
            continue
        if non_null.min() >= 0 and non_null.max() <= np.iinfo(np.uint8).max:
            out[col] = values.astype(np.uint8)
        elif non_null.min() >= np.iinfo(np.int32).min and non_null.max() <= int32_max:
            out[col] = values.astype(np.int32)
        else:
            out[col] = values.astype(np.int64)
    return out


def _global_stats(combined_path: Path) -> pd.DataFrame:
    """Aggregate ``n_rounds`` across all strategies, grouped by ``n_players``."""

    ds_in = ds.dataset(combined_path)
    schema = ds_in.schema
    n_players = n_players_from_schema(schema)
    columns = ["n_rounds", "seat_ranks"]
    tbl = ds_in.to_table(columns)
    df = tbl.to_pandas()

    if "seat_ranks" not in df.columns:
        LOGGER.warning(
            "Combined parquet missing seat_ranks; skipping global game-length stats",
            extra={"stage": "game_stats", "path": str(combined_path)},
        )
        return pd.DataFrame()

    def _player_count_from_ranks(ranks: object) -> int:
        if isinstance(ranks, np.ndarray):
            if ranks.ndim == 1:
                return int(ranks.size)
            return n_players
        if isinstance(ranks, (list, tuple)):
            return len(ranks)
        return n_players

    df["n_players"] = df["seat_ranks"].apply(_player_count_from_ranks)
    grouped = df.groupby("n_players", sort=False)["n_rounds"]

    rows: list[pd.Series] = []
    for players, rounds in grouped:
        player_value = players
        if not isinstance(player_value, (int, np.integer)):
            player_value = pd.to_numeric(player_value, errors="coerce")
        if pd.isna(player_value):
            continue
        if isinstance(player_value, (np.floating, float)):
            if not np.isfinite(player_value) or not float(player_value).is_integer():
                continue
            player_value = to_int(player_value)
        elif isinstance(player_value, (np.integer, int)):
            player_value = to_int(player_value)
        else:
            continue
        stats = _summarize_rounds(rounds)
        stats.update(
            {
                "summary_level": "n_players",
                "strategy": pd.NA,
                "n_players": player_value,
            }
        )
        rows.append(pd.Series(stats))

    return pd.DataFrame(rows)


def _compute_margin_columns(df: pd.DataFrame, score_cols: Sequence[str]) -> pd.DataFrame:
    """Derive per-game margin columns from seat scores.

    ``margin_runner_up`` is ``max(score) - second_max(score)`` (the runner-up
    gap) and ``score_spread`` is ``max(score) - min(score)``. Games with fewer
    than two valid scores return ``NaN`` for both values.
    """

    scores = df.loc[:, list(score_cols)].apply(pd.to_numeric, errors="coerce")
    margin_runner_up, score_spread = _compute_margin_arrays(scores.to_numpy(dtype=float))
    return pd.DataFrame(
        {
            "margin_runner_up": margin_runner_up,
            "score_spread": score_spread,
        },
        index=df.index,
    )


def _summarize_margins(
    values: Iterable[int | float | np.integer | np.floating],
    thresholds: Sequence[int],
) -> dict[str, StatValue]:
    """Return descriptive statistics for per-game runner-up margins."""

    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    prob_keys = [f"prob_margin_runner_up_le_{thr}" for thr in thresholds]
    if series.empty:
        base: dict[str, StatValue] = {
            "observations": 0,
            "mean_margin_runner_up": float("nan"),
            "median_margin_runner_up": float("nan"),
            "std_margin_runner_up": float("nan"),
        }
        base.update({key: float("nan") for key in prob_keys})
        return base

    stats: dict[str, StatValue] = {
        "observations": _strategy_key_to_int(series.size, field="observations"),
        "mean_margin_runner_up": float(series.mean()),
        "median_margin_runner_up": float(series.median()),
        "std_margin_runner_up": float(series.std(ddof=0)),
    }
    stats.update(
        {key: float((series <= thr).mean()) for key, thr in zip(prob_keys, thresholds, strict=True)}
    )
    return stats


def _summarize_rounds(
    values: Iterable[int | float | np.integer | np.floating],
) -> dict[str, StatValue]:
    """Return descriptive statistics for the provided iterable of round counts."""

    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return {
            "observations": 0,
            "mean_rounds": float("nan"),
            "median_rounds": float("nan"),
            "std_rounds": float("nan"),
            "p10_rounds": float("nan"),
            "p50_rounds": float("nan"),
            "p90_rounds": float("nan"),
            "prob_rounds_le_5": float("nan"),
            "prob_rounds_le_10": float("nan"),
            "prob_rounds_ge_20": float("nan"),
        }

    count = _strategy_key_to_int(series.size, field="observations")
    return {
        "observations": count,
        "mean_rounds": float(series.mean()),
        "median_rounds": float(series.median()),
        "std_rounds": float(series.std(ddof=0)),
        "p10_rounds": float(series.quantile(0.10)),
        "p50_rounds": float(series.quantile(0.50)),
        "p90_rounds": float(series.quantile(0.90)),
        "prob_rounds_le_5": float((series <= 5).mean()),
        "prob_rounds_le_10": float((series <= 10).mean()),
        "prob_rounds_ge_20": float((series >= 20).mean()),
    }
