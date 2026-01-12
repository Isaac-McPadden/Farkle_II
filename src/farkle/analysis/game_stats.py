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
from typing import TypeAlias

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

StatValue: TypeAlias = float | int | str | NAType

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

    per_n_inputs = _discover_per_n_inputs(cfg)
    combined_path = cfg.curated_parquet
    input_paths: list[Path] = [p for _, p in per_n_inputs]
    if combined_path.exists():
        input_paths.append(combined_path)

    if not input_paths:
        stage_log.missing_input("no curated parquet files found", analysis_dir=str(cfg.analysis_dir))
        return

    outputs = [game_length_output, margin_output, rare_events_output]
    if not force and stage_is_up_to_date(
        stamp_path, inputs=input_paths, outputs=outputs, config_sha=cfg.config_sha
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

    strategy_stats = _per_strategy_stats(per_n_inputs)
    global_stats = _global_stats(combined_path) if combined_path.exists() else pd.DataFrame()

    combined = pd.concat([strategy_stats, global_stats], ignore_index=True)
    if combined.empty:
        stage_log.missing_input("no rows available to summarize")
        return

    table = pa.Table.from_pandas(combined, preserve_index=False)
    write_parquet_atomic(table, game_length_output, codec=cfg.parquet_codec)

    margin_stats = _per_strategy_margin_stats(
        per_n_inputs, thresholds=cfg.game_stats_margin_thresholds
    )
    if margin_stats.empty:
        stage_log.missing_input("no margins available to summarize")
        return

    margin_table = pa.Table.from_pandas(margin_stats, preserve_index=False)
    write_parquet_atomic(margin_table, margin_output, codec=cfg.parquet_codec)

    rare_events = _rare_event_flags(
        per_n_inputs,
        thresholds=cfg.game_stats_margin_thresholds,
        target_score=cfg.rare_event_target_score,
    )
    if rare_events.empty:
        raise RuntimeError("game-stats: no rare events available to summarize")

    rare_events_table = pa.Table.from_pandas(rare_events, preserve_index=False)
    write_parquet_atomic(rare_events_table, rare_events_output, codec=cfg.parquet_codec)
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
            "rows": len(combined),
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

    rows: list[pd.Series] = []
    rounds_by_strategy: dict[tuple[str, int], list[float]] = {}
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
                df = batch.to_pandas()
                if df.empty:
                    continue
                rounds = pd.to_numeric(df["n_rounds"], errors="coerce")
                strategies = df[col]
                if strategies.isna().all():
                    continue
                for strategy in strategies.dropna().unique():
                    mask = strategies == strategy
                    matched_rounds = rounds[mask].dropna()
                    if matched_rounds.empty:
                        continue
                    rounds_by_strategy.setdefault((strategy, n_players), []).extend(
                        matched_rounds.tolist()
                    )

    for (strategy, players), rounds in rounds_by_strategy.items():
        stats = _summarize_rounds(rounds)
        stats.update(
            {
                "summary_level": "strategy",
                "strategy": strategy,
                "n_players": players,
            }
        )
        rows.append(pd.Series(stats))

    if not rows:
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

    return pd.DataFrame(rows)


def _per_strategy_margin_stats(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
) -> pd.DataFrame:
    """Compute victory-margin statistics grouped by strategy and player count."""

    rows: list[pd.Series] = []
    accumulators: dict[tuple[str, int], _MarginAccumulator] = {}
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
                df = batch.to_pandas()
                if df.empty:
                    continue
                margins = _compute_margins(df, score_cols)
                strategies = df[col]
                if strategies.isna().all():
                    continue
                grouped = margins.groupby(strategies)
                for strategy, grouped_margins in grouped:
                    if pd.isna(strategy):
                        continue
                    matched_margins = grouped_margins.dropna()
                    if matched_margins.empty:
                        continue
                    key = (strategy, n_players)
                    accumulator = accumulators.setdefault(
                        key, _MarginAccumulator(thresholds=thresholds)
                    )
                    accumulator.update(matched_margins)

    for (strategy, players), accumulator in accumulators.items():
        stats = accumulator.to_stats()
        stats.update(
            {
                "summary_level": "strategy",
                "strategy": strategy,
                "n_players": players,
            }
        )
        rows.append(pd.Series(stats))

    if not rows:
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

    return pd.DataFrame(rows)


class _MarginAccumulator:
    """Accumulate margin statistics incrementally."""

    def __init__(self, *, thresholds: Sequence[int]) -> None:
        self._thresholds = list(thresholds)
        self._values: list[float] = []
        self._count = 0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._threshold_counts = {thr: 0 for thr in self._thresholds}

    def update(self, margins: pd.Series) -> None:
        values = pd.to_numeric(margins, errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            return
        self._values.extend(values.tolist())
        self._count += int(values.size)
        self._sum += float(values.sum())
        self._sum_sq += float(np.square(values).sum())
        for thr in self._thresholds:
            self._threshold_counts[thr] += int((values <= thr).sum())

    def to_stats(self) -> dict[str, StatValue]:
        prob_keys = [f"prob_margin_le_{thr}" for thr in self._thresholds]
        if self._count == 0:
            base: dict[str, StatValue] = {
                "observations": 0,
                "mean_margin": float("nan"),
                "median_margin": float("nan"),
                "std_margin": float("nan"),
            }
            base.update({key: float("nan") for key in prob_keys})
            return base

        mean = self._sum / self._count
        variance = self._sum_sq / self._count - mean**2
        std = float(np.sqrt(max(variance, 0.0)))
        median = float(np.median(np.array(self._values, dtype=float)))
        stats: dict[str, StatValue] = {
            "observations": self._count,
            "mean_margin": float(mean),
            "median_margin": median,
            "std_margin": std,
        }
        stats.update(
            {
                key: float(self._threshold_counts[thr] / self._count)
                for key, thr in zip(prob_keys, self._thresholds, strict=True)
            }
        )
        return stats

def _rare_event_flags(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
    target_score: int,
) -> pd.DataFrame:
    """Compute per-game rare events and aggregate rates."""

    game_rows: list[dict[str, object]] = []
    flags = ["multi_reached_target", *[f"margin_le_{thr}" for thr in thresholds]]
    strategy_sums: dict[tuple[str, int], dict[str, int]] = {}
    global_sums: dict[int, dict[str, int]] = {}
    for n_players, path in per_n_inputs:
        ds_in = ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        score_cols = [name for name in ds_in.schema.names if name.startswith("P") and name.endswith("_score")]

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
            df = batch.to_pandas()
            if df.empty:
                continue
            margins = _compute_margins(df, score_cols)
            scores = df[score_cols].apply(pd.to_numeric, errors="coerce")
            multi_target = (scores >= target_score).sum(axis=1) >= 2

            event_df = pd.DataFrame(
                {
                    "margin_of_victory": margins,
                    "multi_reached_target": multi_target,
                }
            )

            for thr in thresholds:
                event_df[f"margin_le_{thr}"] = event_df["margin_of_victory"] <= thr

            for col in strategy_cols:
                strategies = df[col]
                if strategies.isna().all():
                    continue
                for strategy in strategies.dropna().unique():
                    mask = strategies == strategy
                    if not mask.any():
                        continue
                    matched_events = event_df.loc[mask].copy()
                    matched_events.insert(0, "summary_level", "game")
                    matched_events.insert(1, "strategy", strategy)
                    matched_events.insert(2, "n_players", n_players)
                    matched_events["observations"] = 1

                    records: list[dict[str, object]] = [
                        {str(key): value for key, value in record.items()}
                        for record in matched_events.to_dict(orient="records")
                    ]
                    game_rows.extend(records)

                    count = int(mask.sum())
                    strategy_key = (strategy, n_players)
                    strategy_entry = strategy_sums.setdefault(
                        strategy_key,
                        {"observations": 0, **{flag: 0 for flag in flags}},
                    )
                    strategy_entry["observations"] += count
                    for flag in flags:
                        strategy_entry[flag] += int(matched_events[flag].sum())

                    global_entry = global_sums.setdefault(
                        n_players, {"observations": 0, **{flag: 0 for flag in flags}}
                    )
                    global_entry["observations"] += count
                    for flag in flags:
                        global_entry[flag] += int(matched_events[flag].sum())

    if not game_rows:
        columns = [
            "summary_level",
            "strategy",
            "n_players",
            "margin_of_victory",
            "multi_reached_target",
            "observations",
            *[f"margin_le_{thr}" for thr in thresholds],
        ]
        return pd.DataFrame(columns=columns)

    game_df = pd.DataFrame(game_rows)

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
            summary_row[flag] = (
                float(sums[flag]) / observations if observations else float("nan")
            )
        strategy_rows.append(summary_row)
    strategy_summary = pd.DataFrame(strategy_rows)

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
            summary_row[flag] = (
                float(sums[flag]) / observations if observations else float("nan")
            )
        global_rows.append(summary_row)
    global_summary = pd.DataFrame(global_rows)

    combined = pd.concat([game_df, strategy_summary, global_summary], ignore_index=True)
    return combined[
        [
            "summary_level",
            "strategy",
            "n_players",
            "margin_of_victory",
            "multi_reached_target",
            "observations",
            *[f"margin_le_{thr}" for thr in thresholds],
        ]
    ]


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
        stats = _summarize_rounds(rounds)
        stats.update(
            {
                "summary_level": "n_players",
                "strategy": pd.NA,
                "n_players": int(players),
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
