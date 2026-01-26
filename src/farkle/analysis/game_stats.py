# src/farkle/analysis/game_stats.py
"""Compute descriptive statistics for game lengths and victory margins.

Reads curated row-level parquet files (per-``n`` shards and the combined
superset) and aggregates the ``n_rounds`` column. Outputs both per-strategy
statistics and a small global summary grouped by ``n_players``.

The module also flags close margins and multi-target games, emitting pooled
artifacts under ``04_game_stats/pooled`` with per-game records plus aggregated
frequencies per strategy and player-count cohort.

The module also derives per-game ``margin_of_victory`` from seat-level scores
and writes ``04_game_stats/pooled/margin_stats.parquet`` with per-``(strategy,
n_players)`` summaries. Margin schema:

``summary_level``
    Literal "strategy" for compatibility with ``game_length.parquet``.
``strategy``
    Strategy string taken from ``P#_strategy`` columns.
``n_players``
    Player count inferred from the shard path.
``observations``
    Number of games with at least two valid seat scores.
``mean_margin`` / ``median_margin`` / ``std_margin``
    Descriptive statistics over ``margin_of_victory``.
``prob_margin_le_500`` / ``prob_margin_le_1000``
    Close-game shares for the given thresholds.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from pandas._libs.missing import NAType

from farkle.analysis import stage_logger
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.schema_helpers import n_players_from_schema
from farkle.utils.types import Compression
from farkle.utils.writer import ParquetShardWriter

StatValue: TypeAlias = float | int | str | NAType
ArrowColumnData: TypeAlias = np.ndarray | list[Any] | pa.Array | pa.ChunkedArray

LOGGER = logging.getLogger(__name__)


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
    rare_events_output = cfg.game_stats_output_path("rare_events.parquet")
    stamp_path = stage_done_path(stage_dir, "game_stats")
    game_length_stamp = stage_done_path(stage_dir, "game_stats.game_length")
    margin_stamp = stage_done_path(stage_dir, "game_stats.margin")
    rare_events_stamp = stage_done_path(stage_dir, "game_stats.rare_events")

    per_n_inputs = _discover_per_n_inputs(cfg)
    combined_path = cfg.curated_parquet
    input_paths: list[Path] = [p for _, p in per_n_inputs]
    if combined_path.exists():
        input_paths.append(combined_path)

    if not input_paths:
        stage_log.missing_input("no curated parquet files found", analysis_dir=str(cfg.analysis_dir))
        return

    outputs = [game_length_output, margin_output, rare_events_output]
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
    rare_events_up_to_date = not force and stage_is_up_to_date(
        rare_events_stamp,
        inputs=input_paths,
        outputs=[rare_events_output],
        config_sha=cfg.config_sha,
    )

    if game_length_up_to_date and margin_up_to_date and rare_events_up_to_date:
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
    if not game_length_up_to_date:
        strategy_stats = _per_strategy_stats(per_n_inputs)
        global_stats = _global_stats(combined_path) if combined_path.exists() else pd.DataFrame()

        combined = pd.concat([strategy_stats, global_stats], ignore_index=True)
        if combined.empty:
            stage_log.missing_input("no rows available to summarize")
            return
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

    if not margin_up_to_date:
        margin_stats = _per_strategy_margin_stats(
            per_n_inputs, thresholds=cfg.game_stats_margin_thresholds
        )
        if margin_stats.empty:
            stage_log.missing_input("no margins available to summarize")
            return
        margin_stats = _downcast_integer_stats(margin_stats, columns=("n_players", "observations"))

        margin_table = pa.Table.from_pandas(margin_stats, preserve_index=False)
        write_parquet_atomic(margin_table, margin_output, codec=cfg.parquet_codec)
        write_stage_done(
            margin_stamp,
            inputs=input_paths,
            outputs=[margin_output],
            config_sha=cfg.config_sha,
        )

    if not rare_events_up_to_date:
        rare_event_rows = _rare_event_flags(
            per_n_inputs,
            thresholds=cfg.game_stats_margin_thresholds,
            target_score=cfg.rare_event_target_score,
            output_path=rare_events_output,
            codec=cfg.parquet_codec,
        )
        if rare_event_rows == 0:
            raise RuntimeError("game-stats: no rare events available to summarize")
        write_stage_done(
            rare_events_stamp,
            inputs=input_paths,
            outputs=[rare_events_output],
            config_sha=cfg.config_sha,
        )
    write_stage_done(
        stamp_path,
        inputs=input_paths,
        outputs=outputs,
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

    long_frames: list[pd.DataFrame] = []
    for n_players, path in per_n_inputs:
        ds_in = ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        if not strategy_cols:
            LOGGER.warning(
                "Per-N parquet missing strategy columns",
                extra={"stage": "game_stats", "path": str(path)},
            )
            continue

        for col in strategy_cols:
            columns = ["n_rounds", col]
            scanner = ds_in.scanner(columns=columns, batch_size=65_536)
            for batch in scanner.to_batches():
                df = batch.to_pandas(categories=[col])
                if df.empty:
                    continue
                melted = df.melt(id_vars=["n_rounds"], value_vars=[col], value_name="strategy")
                melted = melted.dropna(subset=["strategy"])
                if melted.empty:
                    continue
                melted["strategy"] = melted["strategy"].astype("category")
                melted["n_players"] = n_players
                long_frames.append(melted[["strategy", "n_players", "n_rounds"]])

    if not long_frames:
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

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["n_rounds"] = pd.to_numeric(long_df["n_rounds"], errors="coerce")
    long_df = long_df.dropna(subset=["n_rounds", "strategy"])
    if long_df.empty:
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

    group_keys = ["strategy", "n_players"]
    grouped = long_df.groupby(group_keys, observed=True, sort=False)["n_rounds"]
    stats = grouped.agg(
        observations="count",
        mean_rounds="mean",
        median_rounds="median",
        std_rounds=lambda s: s.std(ddof=0),
        p10_rounds=lambda s: s.quantile(0.1),
        p50_rounds=lambda s: s.quantile(0.5),
        p90_rounds=lambda s: s.quantile(0.9),
    )
    prob_rounds_le_5 = (
        long_df["n_rounds"]
        .le(5)
        .groupby([long_df["strategy"], long_df["n_players"]], observed=True, sort=False)
        .mean()
        .rename("prob_rounds_le_5")
    )
    prob_rounds_le_10 = (
        long_df["n_rounds"]
        .le(10)
        .groupby([long_df["strategy"], long_df["n_players"]], observed=True, sort=False)
        .mean()
        .rename("prob_rounds_le_10")
    )
    prob_rounds_ge_20 = (
        long_df["n_rounds"]
        .ge(20)
        .groupby([long_df["strategy"], long_df["n_players"]], observed=True, sort=False)
        .mean()
        .rename("prob_rounds_ge_20")
    )

    stats = stats.join([prob_rounds_le_5, prob_rounds_le_10, prob_rounds_ge_20])
    stats = stats.reset_index()
    stats.insert(0, "summary_level", "strategy")

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


def _per_strategy_margin_stats(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
) -> pd.DataFrame:
    """Compute victory-margin statistics grouped by strategy and player count."""

    long_frames: list[pd.DataFrame] = []
    for n_players, path in per_n_inputs:
        ds_in = ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        score_cols = [name for name in ds_in.schema.names if name.startswith("P") and name.endswith("_score")]

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

        for col in strategy_cols:
            columns = [*score_cols, col]
            scanner = ds_in.scanner(columns=columns, batch_size=65_536)
            for batch in scanner.to_batches():
                df = batch.to_pandas(categories=[col])
                if df.empty:
                    continue
                margins = _compute_margins(df, score_cols)
                df = df.assign(margin_of_victory=margins)
                melted = df.melt(
                    id_vars=["margin_of_victory"],
                    value_vars=[col],
                    value_name="strategy",
                )
                melted = melted.dropna(subset=["strategy"])
                if melted.empty:
                    continue
                melted["strategy"] = melted["strategy"].astype("category")
                melted["n_players"] = n_players
                long_frames.append(melted[["strategy", "n_players", "margin_of_victory"]])

    if not long_frames:
        columns = [
            "summary_level",
            "strategy",
            "n_players",
            "observations",
            "mean_margin",
            "median_margin",
            "std_margin",
            *[f"prob_margin_le_{thr}" for thr in thresholds],
        ]
        return pd.DataFrame(columns=columns)

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["margin_of_victory"] = pd.to_numeric(long_df["margin_of_victory"], errors="coerce")
    long_df = long_df.dropna(subset=["margin_of_victory", "strategy"])
    if long_df.empty:
        columns = [
            "summary_level",
            "strategy",
            "n_players",
            "observations",
            "mean_margin",
            "median_margin",
            "std_margin",
            *[f"prob_margin_le_{thr}" for thr in thresholds],
        ]
        return pd.DataFrame(columns=columns)

    group_keys = ["strategy", "n_players"]
    grouped = long_df.groupby(group_keys, observed=True, sort=False)["margin_of_victory"]
    stats = grouped.agg(
        observations="count",
        mean_margin="mean",
        median_margin="median",
        std_margin=lambda s: s.std(ddof=0),
    )
    prob_frames = []
    for thr in thresholds:
        prob = (
            long_df["margin_of_victory"]
            .le(thr)
            .groupby([long_df["strategy"], long_df["n_players"]], observed=True, sort=False)
            .mean()
            .rename(f"prob_margin_le_{thr}")
        )
        prob_frames.append(prob)

    stats = stats.join(prob_frames)
    stats = stats.reset_index()
    stats.insert(0, "summary_level", "strategy")

    ordered_cols = [
        "summary_level",
        "strategy",
        "n_players",
        "observations",
        "mean_margin",
        "median_margin",
        "std_margin",
        *[f"prob_margin_le_{thr}" for thr in thresholds],
    ]
    return stats[ordered_cols]


def _rare_event_flags(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
    target_score: int,
    output_path: Path,
    codec: Compression,
) -> int:
    """Compute per-game rare events and aggregate counts."""

    flags = ["multi_reached_target", *[f"margin_le_{thr}" for thr in thresholds]]
    column_order = [
        "summary_level",
        "strategy",
        "n_players",
        "margin_of_victory",
        "multi_reached_target",
        "observations",
        *[f"margin_le_{thr}" for thr in thresholds],
    ]
    (
        strategy_sums,
        global_sums,
        rows_available,
        max_flag_count,
        max_observations,
    ) = _collect_rare_event_counts(per_n_inputs, thresholds=thresholds, target_score=target_score)

    if rows_available == 0:
        return 0

    flag_dtype, flag_arrow = _select_int_dtype(max_flag_count)
    obs_dtype, obs_arrow = _select_int_dtype(max_observations)
    player_dtype = np.int32
    fields: list[pa.Field] = [
        pa.field("summary_level", pa.string()),
        pa.field("strategy", pa.string()),
        pa.field("n_players", pa.int32()),
        pa.field("margin_of_victory", pa.float64()),
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
                name for name in ds_in.schema.names if name.startswith("P") and name.endswith("_score")
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
                margins = _compute_margins(df, score_cols)
                scores = df[score_cols].apply(pd.to_numeric, errors="coerce")
                multi_target = (scores >= target_score).sum(axis=1) >= 2
                flag_series: dict[str, pd.Series] = {"multi_reached_target": multi_target}
                for thr in thresholds:
                    flag_series[f"margin_le_{thr}"] = margins <= thr
                any_flag = flag_series["multi_reached_target"].copy()
                for thr in thresholds:
                    any_flag |= flag_series[f"margin_le_{thr}"]
                base = df.loc[any_flag, strategy_cols].copy()
                if base.empty:
                    continue
                base["margin_of_victory"] = margins[any_flag]
                base["multi_reached_target"] = flag_series["multi_reached_target"][any_flag]
                for thr in thresholds:
                    base[f"margin_le_{thr}"] = flag_series[f"margin_le_{thr}"][any_flag]
                base["n_players"] = n_players
                melted = base.melt(
                    id_vars=[
                        "margin_of_victory",
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
                    count = int(group.shape[0])
                    per_game_data: dict[str, ArrowColumnData] = {
                        "summary_level": np.full(count, "game", dtype=str),
                        "strategy": np.full(count, strategy, dtype=str),
                        "n_players": np.full(count, n_players, dtype=player_dtype),
                        "margin_of_victory": group["margin_of_victory"].to_numpy(dtype=float),
                        "multi_reached_target": group["multi_reached_target"].to_numpy(
                            dtype=flag_dtype
                        ),
                        "observations": np.ones(count, dtype=obs_dtype),
                    }
                    for thr in thresholds:
                        per_game_data[f"margin_le_{thr}"] = group[
                            f"margin_le_{thr}"
                        ].to_numpy(dtype=flag_dtype)

                    writer.write_batch(pa.Table.from_pydict(per_game_data, schema=schema))
                    rows_written += count

        strategy_rows: list[dict[str, object]] = []
        for (strategy, players), sums in strategy_sums.items():
            observations = sums["observations"]
            summary_row: dict[str, object] = {
                "summary_level": "strategy",
                "strategy": strategy,
                "n_players": players,
                "margin_of_victory": pd.NA,
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
                "margin_of_victory": pd.NA,
                "observations": observations,
            }
            for flag in flags:
                summary_row[flag] = sums[flag]
            global_rows.append(summary_row)
        global_summary = pd.DataFrame(global_rows, columns=column_order)

        summary_df = pd.concat([strategy_summary, global_summary], ignore_index=True)
        if not summary_df.empty:
            summary_df = _downcast_integer_stats(
                summary_df, columns=("n_players", "observations", *flags)
            )
            summary_table = pa.Table.from_pandas(summary_df, preserve_index=False, schema=schema)
            writer.write_batch(summary_table)
            rows_written += summary_table.num_rows
        return rows_written
    finally:
        writer.close(success=rows_written > 0)


def _collect_rare_event_counts(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
    target_score: int,
) -> tuple[dict[tuple[str, int], dict[str, int]], dict[int, dict[str, int]], int, int, int]:
    """Gather rare-event counts for downstream downcasting decisions."""
    flags = ["multi_reached_target", *[f"margin_le_{thr}" for thr in thresholds]]
    strategy_sums: dict[tuple[str, int], dict[str, int]] = {}
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
            margins = _compute_margins(df, score_cols)
            scores = df[score_cols].apply(pd.to_numeric, errors="coerce")
            multi_target = (scores >= target_score).sum(axis=1) >= 2
            flag_series: dict[str, pd.Series] = {"multi_reached_target": multi_target}
            for thr in thresholds:
                flag_series[f"margin_le_{thr}"] = margins <= thr
            base = df[strategy_cols].copy()
            base["multi_reached_target"] = flag_series["multi_reached_target"]
            for thr in thresholds:
                base[f"margin_le_{thr}"] = flag_series[f"margin_le_{thr}"]
            base["n_players"] = n_players
            melted = base.melt(
                id_vars=["multi_reached_target", "n_players", *[f"margin_le_{thr}" for thr in thresholds]],
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
                count = int(group.shape[0])
                rows_available += count
                strategy_key = (str(strategy), n_players)
                strategy_entry = strategy_sums.setdefault(
                    strategy_key,
                    {"observations": 0, **dict.fromkeys(flags, 0)},
                )
                strategy_entry["observations"] += count
                for flag in flags:
                    strategy_entry[flag] += int(group[flag].sum())

                global_entry = global_sums.setdefault(
                    n_players, {"observations": 0, **dict.fromkeys(flags, 0)}
                )
                global_entry["observations"] += count
                for flag in flags:
                    global_entry[flag] += int(group[flag].sum())

    max_flag_count = 0
    max_observations = 0
    for sums in list(strategy_sums.values()) + list(global_sums.values()):
        max_observations = max(max_observations, sums.get("observations", 0))
        for flag in flags:
            max_flag_count = max(max_flag_count, sums.get(flag, 0))
    return strategy_sums, global_sums, rows_available, max_flag_count, max_observations


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
    tbl = ds_in.to_table(columns=columns)
    df = tbl.to_pandas()

    if "seat_ranks" not in df.columns:
        LOGGER.warning(
            "Combined parquet missing seat_ranks; skipping global game-length stats",
            extra={"stage": "game_stats", "path": str(combined_path)},
        )
        return pd.DataFrame()

    df["n_players"] = df["seat_ranks"].apply(
        lambda ranks: len(ranks) if isinstance(ranks, list) else n_players
    )
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
            player_value = int(player_value)
        elif isinstance(player_value, (np.integer, int)):
            player_value = int(player_value)
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


def _compute_margins(df: pd.DataFrame, score_cols: Sequence[str]) -> pd.Series:
    """Derive per-game margin of victory from seat scores.

    For two-player games, this is ``|P1_score - P2_score|``. For more than two
    players, this is ``max(score) - min(score)`` based on available seat scores.
    Games with fewer than two valid scores return ``NaN`` margins.
    """

    scores = df.loc[:, list(score_cols)].apply(pd.to_numeric, errors="coerce")
    valid_counts = scores.notna().sum(axis=1)
    margins = scores.max(axis=1, skipna=True) - scores.min(axis=1, skipna=True)
    margins[valid_counts < 2] = np.nan
    return margins.astype(float)


def _summarize_margins(
    values: Iterable[int | float | np.integer | np.floating],
    thresholds: Sequence[int],
) -> dict[str, StatValue]:
    """Return descriptive statistics for per-game victory margins."""

    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    prob_keys = [f"prob_margin_le_{thr}" for thr in thresholds]
    if series.empty:
        base: dict[str, StatValue] = {
            "observations": 0,
            "mean_margin": float("nan"),
            "median_margin": float("nan"),
            "std_margin": float("nan"),
        }
        base.update({key: float("nan") for key in prob_keys})
        return base

    stats: dict[str, StatValue] = {
        "observations": int(series.size),
        "mean_margin": float(series.mean()),
        "median_margin": float(series.median()),
        "std_margin": float(series.std(ddof=0)),
    }
    stats.update({key: float((series <= thr).mean()) for key, thr in zip(prob_keys, thresholds, strict=True)})
    return stats


def _summarize_rounds(values: Iterable[int | float | np.integer | np.floating]) -> dict[str, StatValue]:
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

    count = int(series.size)
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
