"""RNG diagnostics for curated rows ordered by ``game_seed``.

Computes lagged autocorrelation for win indicators and game lengths at both
strategy and matchup-strategy levels. Outputs are written to
``00_rng/pooled/rng_diagnostics.parquet`` with approximate confidence intervals
and "expected ~0" reminders. When inputs or required columns are missing, the
module logs a skip instead of raising.
"""
from __future__ import annotations

import json
import logging
import re
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Iterator, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from farkle.analysis import stage_logger
from farkle.analysis.stage_state import stage_done_path
from farkle.config import AppConfig
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.parallel import apply_native_thread_limits, resolve_stage_parallel_policy
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)

_EXPECTED_NOTE = "Expected ~0 under IID seeds"
_SEAT_STRATEGY_RE = re.compile(r"^P(\d+)_strategy$")
_STREAM_BATCH_SIZE = 100_000
_STREAM_PROGRESS_EVERY = 25
_DEFAULT_MAX_MATCHUP_GROUPS = 100_000


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and np.isnan(value)


def run(cfg: AppConfig, *, lags: Sequence[int] | None = None, force: bool = False) -> None:
    """Compute lagged autocorrelation diagnostics for curated rows.

    Args:
        cfg: Application configuration for locating curated inputs and outputs.
        lags: Optional sequence of positive lags (defaults to ``(1,)``).
        force: Recompute even when the done-stamp matches inputs/outputs.
    """

    stage_log = stage_logger("rng_diagnostics", logger=LOGGER)
    stage_log.start()

    policy = resolve_stage_parallel_policy("rng_diagnostics", cfg.analysis)
    apply_native_thread_limits(policy)
    pa.set_cpu_count(policy.arrow_threads)
    pa.set_io_thread_count(policy.arrow_threads)
    LOGGER.info(
        "rng-diagnostics threading resolved",
        extra={
            "stage": "rng_diagnostics",
            "process_workers": policy.process_workers,
            "python_threads": policy.python_threads,
            "arrow_threads": policy.arrow_threads,
        },
    )

    interseed_ready, interseed_reason = cfg.interseed_ready()
    if not interseed_ready:
        stage_log.missing_input(interseed_reason)
        return

    data_file = cfg.curated_parquet
    out_file = cfg.rng_output_path("rng_diagnostics.parquet")
    stamp_path = stage_done_path(cfg.rng_stage_dir, "rng_diagnostics")

    lags = _normalize_lags(lags)
    if not lags:
        stage_log.missing_input("no valid lags provided")
        return

    if not data_file.exists():
        stage_log.missing_input("missing curated parquet", path=str(data_file))
        return

    if not force and _is_up_to_date(
        stamp_path, inputs=[data_file], outputs=[out_file], lags=lags, config_sha=cfg.config_sha
    ):
        LOGGER.info(
            "rng-diagnostics: up-to-date",
            extra={"stage": "rng_diagnostics", "path": str(out_file), "stamp": str(stamp_path)},
        )
        return

    dataset = ds.dataset(data_file)
    schema_names = set(dataset.schema.names)
    strat_cols = _seat_strategy_columns(cfg, dataset.schema.names)
    winner_col = _winner_column(schema_names)

    required = {"game_seed", "n_rounds"}
    if not required.issubset(schema_names) or winner_col is None:
        stage_log.missing_input(
            "curated parquet missing required columns",
            path=str(data_file),
            required_cols=sorted(required | {"winner_strategy", "winner_seat"}),
        )
        return

    if not strat_cols:
        stage_log.missing_input(
            "curated parquet missing seat strategy columns",
            path=str(data_file),
            required_cols=["P1_strategy"],
        )
        return

    columns = ["game_seed", "n_rounds", winner_col, *strat_cols]
    max_matchup_groups = cfg.analysis.rng_max_matchup_groups
    if max_matchup_groups is None:
        max_matchup_groups = _DEFAULT_MAX_MATCHUP_GROUPS
    if max_matchup_groups <= 0:
        max_matchup_groups = None
    prepared_batches = _iter_prepared_batches(
        dataset,
        columns=columns,
        winner_col=winner_col,
        strat_cols=strat_cols,
        batch_size=_STREAM_BATCH_SIZE,
        arrow_threads=policy.arrow_threads,
    )
    diagnostics, melted_rows = _collect_diagnostics_streaming_compact(
        prepared_batches,
        strat_cols=strat_cols,
        lags=lags,
        progress_every_batches=_STREAM_PROGRESS_EVERY,
        max_matchup_groups=max_matchup_groups,
    )
    if melted_rows == 0:
        stage_log.missing_input("no per-strategy rows after melting", path=str(data_file))
        return
    if diagnostics.empty:
        stage_log.missing_input("no diagnostics computed", path=str(data_file))
        return

    table_out = pa.Table.from_pandas(diagnostics, preserve_index=False)
    write_parquet_atomic(table_out, out_file, codec=cfg.parquet_codec)
    _write_stamp(
        stamp_path,
        inputs=[data_file],
        outputs=[out_file],
        lags=lags,
        config_sha=cfg.config_sha,
    )
    LOGGER.info(
        "rng-diagnostics: written",
        extra={"stage": "rng_diagnostics", "rows": len(diagnostics), "path": str(out_file)},
    )


def _iter_prepared_batches(
    dataset: ds.Dataset,
    *,
    columns: Sequence[str],
    winner_col: str,
    strat_cols: Sequence[str],
    batch_size: int,
    arrow_threads: int,
) -> Iterator[pd.DataFrame]:
    scanner = dataset.scanner(
        columns=list(columns),
        batch_size=batch_size,
        use_threads=arrow_threads > 1,
    )
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        df = batch.to_pandas(categories=list(strat_cols))
        if df.empty:
            continue
        df = df.sort_values("game_seed", kind="mergesort")
        df["matchup"] = _build_matchup_labels(df, strat_cols)
        df["n_players"] = df[strat_cols].notna().sum(axis=1).astype(int)
        df["winner_strategy"] = _winner_strategies(df, winner_col, strat_cols)
        yield df[["n_rounds", "matchup", "n_players", "winner_strategy", *strat_cols]]


def _collect_diagnostics_streaming_compact(
    data_batches: Iterable[pd.DataFrame],
    *,
    strat_cols: Sequence[str],
    lags: Sequence[int],
    progress_every_batches: int,
    max_matchup_groups: int | None,
) -> tuple[pd.DataFrame, int]:
    normalized_lags = tuple(int(lag) for lag in lags)
    strategy_states: dict[tuple[str, int], _GroupStreamAccumulator] = {}
    matchup_states: dict[tuple[str | None, str, int], _GroupStreamAccumulator] = {}
    processed_batches = 0
    melted_rows = 0
    skipped_matchup_groups = 0
    skipped_matchup_rows = 0

    for batch in data_batches:
        if batch.empty:
            continue
        processed_batches += 1

        for strat_col in strat_cols:
            if strat_col not in batch.columns:
                continue
            seat = batch[["matchup", "n_players", "winner_strategy", "n_rounds", strat_col]].rename(
                columns={strat_col: "strategy"}
            )
            seat = seat.dropna(subset=["strategy"]).copy()
            if seat.empty:
                continue
            seat["strategy"] = seat["strategy"].astype("string")
            seat["win_indicator"] = (
                seat["winner_strategy"].notna() & (seat["strategy"] == seat["winner_strategy"])
            ).astype(np.int8)
            seat = seat[["matchup", "strategy", "n_players", "win_indicator", "n_rounds"]]
            melted_rows += int(seat.shape[0])

            grouped_strategy = seat.groupby(["strategy", "n_players"], observed=True, sort=False)
            for (strategy, n_players), group in grouped_strategy:
                key = (str(strategy), int(cast(int, n_players)))
                state = strategy_states.setdefault(key, _GroupStreamAccumulator(normalized_lags))
                state.extend(group)

            grouped_matchup = seat.groupby(["matchup", "strategy", "n_players"], observed=True, sort=False)
            for (matchup, strategy, n_players), group in grouped_matchup:
                matchup_key = None if _is_missing_scalar(matchup) else str(matchup)
                matchup_state_key = (matchup_key, str(strategy), int(cast(int, n_players)))
                matchup_state = matchup_states.get(matchup_state_key)
                if matchup_state is None:
                    if max_matchup_groups is not None and len(matchup_states) >= max_matchup_groups:
                        skipped_matchup_groups += 1
                        skipped_matchup_rows += int(group.shape[0])
                        continue
                    matchup_state = matchup_states.setdefault(
                        matchup_state_key, _GroupStreamAccumulator(normalized_lags)
                    )
                matchup_state.extend(group)

        if progress_every_batches > 0 and processed_batches % progress_every_batches == 0:
            LOGGER.info(
                "rng-diagnostics progress",
                extra={
                    "stage": "rng_diagnostics",
                    "batches": processed_batches,
                    "melted_rows": melted_rows,
                    "strategy_groups": len(strategy_states),
                    "matchup_groups": len(matchup_states),
                    "matchup_groups_skipped": skipped_matchup_groups,
                    "matchup_rows_skipped": skipped_matchup_rows,
                },
            )

    if skipped_matchup_groups > 0:
        LOGGER.warning(
            "rng-diagnostics matchup grouping capped",
            extra={
                "stage": "rng_diagnostics",
                "max_matchup_groups": max_matchup_groups,
                "matchup_groups_tracked": len(matchup_states),
                "matchup_groups_skipped": skipped_matchup_groups,
                "matchup_rows_skipped": skipped_matchup_rows,
            },
        )

    rows: list[dict[str, object]] = []
    for (strategy, n_players), state in strategy_states.items():
        rows.extend(
            _rows_from_group_state(
                summary_level="strategy",
                strategy=strategy,
                n_players=n_players,
                lags=normalized_lags,
                group_state=state,
            )
        )
    for (matchup, strategy, n_players), state in matchup_states.items():
        rows.extend(
            _rows_from_group_state(
                summary_level="matchup_strategy",
                strategy=strategy,
                matchup=matchup,
                n_players=n_players,
                lags=normalized_lags,
                group_state=state,
            )
        )
    if not rows:
        return pd.DataFrame(), melted_rows

    diagnostics = pd.DataFrame(rows)
    diagnostics = diagnostics.sort_values(
        ["summary_level", "strategy", "n_players", "lag", "metric"],
        kind="mergesort",
    )
    diagnostics.reset_index(drop=True, inplace=True)
    return diagnostics, melted_rows


def _normalize_lags(lags: Sequence[int] | None) -> tuple[int, ...]:
    if lags is None:
        return (1,)
    valid = sorted({int(lag) for lag in lags if int(lag) > 0})
    return tuple(valid)


def _winner_column(names: set[str]) -> str | None:
    if "winner_strategy" in names:
        return "winner_strategy"
    if "winner_seat" in names:
        return "winner_seat"
    return None


def _seat_strategy_columns(cfg: AppConfig, schema_names: Sequence[str]) -> list[str]:
    schema_set = set(schema_names)
    configured_candidates: list[str] = []
    if cfg.sim.n_players_list:
        max_players = max(cfg.sim.n_players_list)
        configured_candidates = [f"P{seat}_strategy" for seat in range(1, max_players + 1)]

    fallback_candidates = [
        name for name in schema_names if _SEAT_STRATEGY_RE.match(name) and name != "winner_strategy"
    ]
    merged_candidates = set(configured_candidates) | set(fallback_candidates)
    present = [name for name in merged_candidates if name in schema_set and name != "winner_strategy"]

    return sorted(present, key=_seat_number_from_strategy_column)


def _build_matchup_labels(df: pd.DataFrame, strat_cols: Sequence[str]) -> pd.Series:
    valid_seat_cols = [col for col in strat_cols if col in df.columns and _SEAT_STRATEGY_RE.match(col)]
    if not valid_seat_cols:
        return pd.Series(index=df.index, dtype="string")

    seat_values = df[valid_seat_cols].astype("string")
    stacked = cast(pd.Series, seat_values.stack()).dropna()
    if stacked.empty:
        return pd.Series(index=df.index, dtype="string")

    matchups = cast(
        pd.Series,
        stacked.groupby(level=0, sort=False)
        .agg(list)
        .map(lambda participants: " | ".join(sorted(participants))),
    )
    return cast(pd.Series, matchups.reindex(df.index).astype("string"))


def _winner_strategies(
    df: pd.DataFrame, winner_col: str, strat_cols: Sequence[str]
) -> pd.Series:
    if winner_col == "winner_strategy":
        return df[winner_col]

    valid_cols = [col for col in strat_cols if _SEAT_STRATEGY_RE.match(col)]
    seat_to_col_position = {
        int(cast(re.Match[str], _SEAT_STRATEGY_RE.match(col)).group(1)): idx
        for idx, col in enumerate(valid_cols)
    }
    winners = pd.Series(pd.NA, index=df.index, dtype="string")

    if not valid_cols:
        return winners

    seat_indices = pd.to_numeric(
        df[winner_col].astype("string").str.extract(r"^P(\d+)$", expand=False),
        errors="coerce",
    ).astype("Int64")
    resolved_positions = seat_indices.map(seat_to_col_position)
    valid_mask = resolved_positions.notna()
    if not valid_mask.any():
        return winners

    strategy_values = df[valid_cols].astype("string").to_numpy(dtype=object)
    winner_rows = np.flatnonzero(valid_mask.to_numpy())
    winner_cols = resolved_positions[valid_mask].astype(int).to_numpy()
    winners.iloc[winner_rows] = strategy_values[winner_rows, winner_cols]
    return winners


def _seat_number_from_strategy_column(column_name: str) -> int:
    match = _SEAT_STRATEGY_RE.match(column_name)
    if match is None:
        raise ValueError(f"invalid seat strategy column: {column_name}")
    return int(match.group(1))


def _melt_strategies(df: pd.DataFrame, strat_cols: Sequence[str]) -> pd.DataFrame:
    id_vars = ["game_seed", "n_rounds", "matchup", "n_players", "winner_strategy"]
    melted = df[id_vars + list(strat_cols)].melt(
        id_vars=id_vars,
        value_vars=strat_cols,
        var_name="seat",
        value_name="strategy",
    )
    melted = melted.dropna(subset=["strategy"])
    melted["strategy"] = melted["strategy"].astype("category")
    melted["seat"] = melted["seat"].str.removesuffix("_strategy")
    melted["win_indicator"] = (melted["strategy"] == melted["winner_strategy"]).astype(int)
    return melted


@dataclass(slots=True)
class _LagCorrelationAccumulator:
    pair_count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: float, y: float) -> None:
        self.pair_count += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_x2 += x * x
        self.sum_y2 += y * y
        self.sum_xy += x * y

    def autocorr(self) -> float | None:
        if self.pair_count < 2:
            return None
        pairs = float(self.pair_count)
        numerator = pairs * self.sum_xy - self.sum_x * self.sum_y
        den_x = pairs * self.sum_x2 - self.sum_x * self.sum_x
        den_y = pairs * self.sum_y2 - self.sum_y * self.sum_y
        if den_x <= 0.0 or den_y <= 0.0:
            return None
        return float(numerator / (den_x * den_y) ** 0.5)


@dataclass(slots=True)
class _MetricStreamAccumulator:
    lags: tuple[int, ...]
    n_obs: int = 0
    buffers: dict[int, deque[float]] = field(default_factory=dict)
    states: dict[int, _LagCorrelationAccumulator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.buffers = {lag: deque(maxlen=lag) for lag in self.lags}
        self.states = {lag: _LagCorrelationAccumulator() for lag in self.lags}

    def extend(self, values: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        for value in numeric:
            if not np.isfinite(value):
                continue
            self._push(float(value))

    def _push(self, value: float) -> None:
        for lag in self.lags:
            buffer = self.buffers[lag]
            if len(buffer) == lag:
                self.states[lag].update(buffer[0], value)
            buffer.append(value)
        self.n_obs += 1

    def autocorr(self, lag: int) -> float | None:
        return self.states[lag].autocorr()


@dataclass(slots=True)
class _GroupStreamAccumulator:
    lags: tuple[int, ...]
    win_indicator: _MetricStreamAccumulator = field(init=False)
    n_rounds: _MetricStreamAccumulator = field(init=False)

    def __post_init__(self) -> None:
        self.win_indicator = _MetricStreamAccumulator(self.lags)
        self.n_rounds = _MetricStreamAccumulator(self.lags)

    def extend(self, frame: pd.DataFrame) -> None:
        self.win_indicator.extend(frame["win_indicator"])
        self.n_rounds.extend(frame["n_rounds"])


def _iter_melted_batches(
    dataset: ds.Dataset,
    *,
    columns: Sequence[str],
    winner_col: str,
    strat_cols: Sequence[str],
    batch_size: int,
    arrow_threads: int,
) -> Iterator[pd.DataFrame]:
    scanner = dataset.scanner(
        columns=list(columns),
        batch_size=batch_size,
        use_threads=arrow_threads > 1,
    )
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        df = batch.to_pandas(categories=list(strat_cols))
        if df.empty:
            continue
        df = df.sort_values("game_seed", kind="mergesort")
        df["matchup"] = _build_matchup_labels(df, strat_cols)
        df["n_players"] = df[strat_cols].notna().sum(axis=1).astype(int)
        df["winner_strategy"] = _winner_strategies(df, winner_col, strat_cols)
        melted = _melt_strategies(df, strat_cols)
        if melted.empty:
            continue
        yield melted[["game_seed", "strategy", "matchup", "n_players", "win_indicator", "n_rounds"]]


def _rows_from_group_state(
    *,
    summary_level: str,
    strategy: str,
    n_players: int,
    lags: Sequence[int],
    group_state: _GroupStreamAccumulator,
    matchup: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_states = {
        "win_indicator": group_state.win_indicator,
        "n_rounds": group_state.n_rounds,
    }
    for metric, metric_state in metric_states.items():
        n_obs = metric_state.n_obs
        for lag in lags:
            if n_obs <= lag:
                continue
            autocorr = metric_state.autocorr(lag)
            if autocorr is None or pd.isna(autocorr):
                continue
            stderr = 1.0 / n_obs**0.5
            rows.append(
                {
                    "summary_level": summary_level,
                    "strategy": strategy,
                    "matchup": matchup,
                    "n_players": n_players,
                    "observations": n_obs,
                    "lag": lag,
                    "metric": metric,
                    "autocorr": autocorr,
                    "ci_lower": autocorr - 1.96 * stderr,
                    "ci_upper": autocorr + 1.96 * stderr,
                    "note": _EXPECTED_NOTE,
                }
            )
    return rows


def _collect_diagnostics_streaming(
    data_batches: Iterable[pd.DataFrame],
    *,
    lags: Sequence[int],
    progress_every_batches: int,
    max_matchup_groups: int | None = None,
) -> tuple[pd.DataFrame, int]:
    normalized_lags = tuple(int(lag) for lag in lags)
    strategy_states: dict[tuple[str, int], _GroupStreamAccumulator] = {}
    matchup_states: dict[tuple[str | None, str, int], _GroupStreamAccumulator] = {}
    processed_batches = 0
    melted_rows = 0
    skipped_matchup_groups = 0
    skipped_matchup_rows = 0

    for batch in data_batches:
        if batch.empty:
            continue
        processed_batches += 1
        melted_rows += int(batch.shape[0])

        grouped_strategy = batch.groupby(["strategy", "n_players"], observed=True, sort=False)
        for (strategy, n_players), group in grouped_strategy:
            key = (str(strategy), int(cast(int, n_players)))
            state = strategy_states.setdefault(key, _GroupStreamAccumulator(normalized_lags))
            ordered = group.sort_values("game_seed", kind="mergesort")
            state.extend(ordered)

        grouped_matchup = batch.groupby(["matchup", "strategy", "n_players"], observed=True, sort=False)
        for (matchup, strategy, n_players), group in grouped_matchup:
            matchup_key = None if _is_missing_scalar(matchup) else str(matchup)
            matchup_state_key = (matchup_key, str(strategy), int(cast(int, n_players)))
            matchup_state = matchup_states.get(matchup_state_key)
            if matchup_state is None:
                if max_matchup_groups is not None and len(matchup_states) >= max_matchup_groups:
                    skipped_matchup_groups += 1
                    skipped_matchup_rows += int(group.shape[0])
                    continue
                matchup_state = matchup_states.setdefault(
                    matchup_state_key, _GroupStreamAccumulator(normalized_lags)
                )
            ordered = group.sort_values("game_seed", kind="mergesort")
            matchup_state.extend(ordered)

        if progress_every_batches > 0 and processed_batches % progress_every_batches == 0:
            LOGGER.info(
                "rng-diagnostics progress",
                extra={
                    "stage": "rng_diagnostics",
                    "batches": processed_batches,
                    "melted_rows": melted_rows,
                    "strategy_groups": len(strategy_states),
                    "matchup_groups": len(matchup_states),
                    "matchup_groups_skipped": skipped_matchup_groups,
                    "matchup_rows_skipped": skipped_matchup_rows,
                },
            )

    if skipped_matchup_groups > 0:
        LOGGER.warning(
            "rng-diagnostics matchup grouping capped",
            extra={
                "stage": "rng_diagnostics",
                "max_matchup_groups": max_matchup_groups,
                "matchup_groups_tracked": len(matchup_states),
                "matchup_groups_skipped": skipped_matchup_groups,
                "matchup_rows_skipped": skipped_matchup_rows,
            },
        )

    rows: list[dict[str, object]] = []
    for (strategy, n_players), state in strategy_states.items():
        rows.extend(
            _rows_from_group_state(
                summary_level="strategy",
                strategy=strategy,
                n_players=n_players,
                lags=normalized_lags,
                group_state=state,
            )
        )
    for (matchup, strategy, n_players), state in matchup_states.items():
        rows.extend(
            _rows_from_group_state(
                summary_level="matchup_strategy",
                strategy=strategy,
                matchup=matchup,
                n_players=n_players,
                lags=normalized_lags,
                group_state=state,
            )
        )
    if not rows:
        return pd.DataFrame(), melted_rows

    diagnostics = pd.DataFrame(rows)
    diagnostics = diagnostics.sort_values(
        ["summary_level", "strategy", "n_players", "lag", "metric"],
        kind="mergesort",
    )
    diagnostics.reset_index(drop=True, inplace=True)
    return diagnostics, melted_rows


def _collect_diagnostics(data: pd.DataFrame, *, lags: Iterable[int]) -> pd.DataFrame:
    rows: list[pd.Series] = []

    grouped_strategy = data.groupby(["strategy", "n_players"], observed=True, sort=False)
    strategy_diagnostics = (
        _group_diagnostics(
            group,
            lags=lags,
            summary_level="strategy",
            strategy=strategy_str,
            n_players=n_players_int,
        )
        for (strategy, n_players), group in grouped_strategy
        for strategy_str in (str(strategy),)
        for n_players_int in (int(cast(int, n_players)),)
    )
    # Each grouped call yields an iterable of pd.Series diagnostics.
    rows.extend(chain.from_iterable(strategy_diagnostics))

    grouped_matchup = data.groupby(
        ["matchup", "strategy", "n_players"], observed=True, sort=False
    )
    matchup_diagnostics = (
        _group_diagnostics(
            group,
            lags=lags,
            summary_level="matchup_strategy",
            strategy=strategy_str,
            matchup=matchup_str,
            n_players=n_players_int,
        )
        for (matchup, strategy, n_players), group in grouped_matchup
        for strategy_str in (str(strategy),)
        for n_players_int in (int(cast(int, n_players)),)
        for matchup_str in ((None if matchup is None else str(matchup)),)
    )
    rows.extend(chain.from_iterable(matchup_diagnostics))

    flattened = [row for row in rows if not row.empty]
    if not flattened:
        return pd.DataFrame()
    diagnostics = pd.DataFrame(flattened)
    diagnostics = diagnostics.sort_values(["summary_level", "strategy", "n_players", "lag", "metric"])
    return diagnostics


def _group_diagnostics(
    group: pd.DataFrame,
    *,
    lags: Iterable[int],
    summary_level: str,
    strategy: str,
    n_players: int,
    matchup: str | None = None,
) -> list[pd.Series]:
    rows: list[pd.Series] = []
    ordered = group.sort_values("game_seed")
    metrics = {
        "win_indicator": ordered["win_indicator"],
        "n_rounds": ordered["n_rounds"],
    }

    for lag in lags:
        for metric, series in metrics.items():
            cleaned = series.dropna()
            n_obs = len(cleaned)
            if n_obs <= lag:
                continue
            autocorr = cleaned.autocorr(lag=lag)
            if pd.isna(autocorr):
                continue
            stderr = 1.0 / n_obs**0.5
            ci_lower = autocorr - 1.96 * stderr
            ci_upper = autocorr + 1.96 * stderr
            rows.append(
                pd.Series(
                    {
                        "summary_level": summary_level,
                        "strategy": strategy,
                        "matchup": matchup,
                        "n_players": n_players,
                        "observations": n_obs,
                        "lag": lag,
                        "metric": metric,
                        "autocorr": autocorr,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "note": _EXPECTED_NOTE,
                    }
                )
            )
    return rows


def _stamp(path: Path) -> dict[str, float | int]:
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _write_stamp(
    stamp_path: Path,
    *,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    lags: Sequence[int],
    config_sha: str | None,
) -> None:
    payload = {
        "inputs": {str(p): _stamp(p) for p in inputs if p.exists()},
        "outputs": {str(p): _stamp(p) for p in outputs if p.exists()},
        "params": {"lags": list(lags)},
        "config_sha": config_sha,
    }
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(stamp_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2))


def _is_up_to_date(
    stamp_path: Path,
    *,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    lags: Sequence[int],
    config_sha: str | None,
) -> bool:
    if not (stamp_path.exists() and all(p.exists() for p in outputs)):
        return False
    try:
        meta = json.loads(stamp_path.read_text())
    except Exception:  # noqa: BLE001
        return False

    recorded_sha = meta.get("config_sha")
    if config_sha is not None and recorded_sha not in (None, config_sha):
        return False

    if meta.get("params", {}).get("lags") != list(lags):
        return False

    in_meta = meta.get("inputs", {})
    out_meta = meta.get("outputs", {})

    def _matches(paths: Iterable[Path], recorded: dict[str, dict[str, float | int]]) -> bool:
        for p in paths:
            data = recorded.get(str(p))
            if data is None:
                return False
            stat = p.stat()
            if data.get("mtime") != stat.st_mtime or data.get("size") != stat.st_size:
                return False
        return True

    return _matches(inputs, in_meta) and _matches(outputs, out_meta)


if __name__ == "__main__":  # pragma: no cover
    config = AppConfig()
    run(config)
