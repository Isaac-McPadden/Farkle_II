# src/farkle/analysis/rng_diagnostics.py
"""RNG diagnostics in canonical RNG-v2 tournament-player coordinate order.

Each seat exposure is identified by
``(root_seed, k, shuffle_index, game_index, seat_index)`` and the full stream is
ordered lexicographically by that coordinate. Seats are merged in ascending
zero-based ``seat_index`` (``P1 = 0``) within each game before observations are
filtered into strategy and matchup-strategy sequences. Lag correlations are
Pearson correlations of the resulting consecutive within-group metric values.

The zero-centered approximate reference bands are descriptive only and do not
establish independence. When inputs or required columns are missing, the module
logs a skip instead of raising.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, TypedDict, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from farkle.analysis import stage_logger
from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import (
    MethodContract,
    make_artifact_sidecar,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.parallel import apply_native_thread_limits, resolve_stage_parallel_policy
from farkle.utils.progress import ScheduledProgressLogger
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose
from farkle.utils.release_identity import is_v3_config
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done

LOGGER = logging.getLogger(__name__)

_EXPECTED_NOTE = (
    "Zero-centered approximate descriptive reference band only; values inside or "
    "outside the band do not establish or refute independence"
)
_BAND_METHOD = "zero_centered_1.96_over_sqrt_lagged_pairs_descriptive_reference_band"
_DIAGNOSTIC_METHOD_VERSION = 3
_GAME_COORDINATE_COLUMNS = ("root_seed", "k", "shuffle_index", "game_index")
_SEAT_COORDINATE_COLUMNS = (*_GAME_COORDINATE_COLUMNS, "seat_index")
_SEQUENCE_DEFINITION = "lexicographic_rng_v2_tournament_player_coordinate_then_within_group_filter"
_SEAT_STRATEGY_RE = re.compile(r"^P(\d+)_strategy$")
_STREAM_BATCH_SIZE = 100_000
_GLOBAL_MERGE_BATCH_SIZE = 50_000
_DEFAULT_MAX_MATCHUP_GROUPS = 100_000


@dataclass(frozen=True)
class RNGDiagnosticCapacityMetadata:
    """Capacity and lag support actually used by one diagnostic artifact."""

    effective_matchup_group_cap: int | None
    normalized_lags: tuple[int, ...]
    tracked_matchup_group_count: int
    skipped_matchup_group_count: int
    skipped_matchup_row_count: int


class RNGDiagnosticMethodParameters(TypedDict):
    """Authenticated RNG-diagnostic method parameters written to the sidecar."""

    method_version: int
    rng_scheme_version: int
    purpose_namespace: int
    global_order_columns: list[str]
    seat_order: list[int]
    sequence_definition: str
    reference_band_method: str
    claim: str
    effective_matchup_group_cap: int | None
    normalized_lags: list[int]
    tracked_matchup_group_count: int
    skipped_matchup_group_count: int
    skipped_matchup_row_count: int


def _is_missing_scalar(value: object) -> bool:
    """Return whether a grouped scalar should be treated as missing.

    Args:
        value: Scalar value taken from pandas grouping keys or cells.

    Returns:
        ``True`` when the value is ``None``, ``pd.NA``, or ``NaN``.
    """
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and np.isnan(value)


def run(cfg: AppConfig, *, lags: Sequence[int] | None = None, force: bool = False) -> None:
    """Compute lagged autocorrelation diagnostics for curated rows.

    Args:
        cfg: Application configuration for locating curated inputs and outputs.
        lags: Optional sequence of positive lags overriding the typed config field.
        force: Recompute even when the done-stamp matches inputs/outputs.
    """

    stage_log = stage_logger("rng_diagnostics", logger=LOGGER)
    stage_log.start()

    policy = resolve_stage_parallel_policy("rng_diagnostics", cfg.analysis, resources=cfg.resources)
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

    try:
        data_file = cfg.curated_parquet
    except KeyError as exc:
        stage_log.missing_input(str(exc))
        return
    out_file = cfg.rng_output_path("rng_diagnostics.parquet")
    stamp_path = stage_done_path(cfg.rng_stage_dir, "rng_diagnostics")

    lags = _normalize_lags(cfg.analysis.rng_diagnostic_lags if lags is None else lags)
    if not lags:
        stage_log.missing_input("no valid lags provided")
        return

    if not data_file.exists():
        stage_log.missing_input("missing curated parquet", path=str(data_file))
        return
    if is_v3_config(cfg):
        validate_artifact_sidecar(
            data_file,
            expected={
                "scope": ArtifactScope.CONCAT_KS.value,
                "operation": "concatenate",
            },
        )

    stage_config_sha = _rng_stage_config_sha(cfg, lags)
    if not force and stage_is_up_to_date(
        stamp_path,
        inputs=[data_file],
        outputs=[out_file],
        config_sha=cfg.config_sha,
        stage="rng_diagnostics",
        stage_config_sha=stage_config_sha,
        cache_key_version=cfg.stage_cache_key_version("rng_diagnostics"),
        sidecar_artifacts=[out_file],
    ):
        LOGGER.info(
            "rng-diagnostics: up-to-date",
            extra={"stage": "rng_diagnostics", "path": str(out_file), "stamp": str(stamp_path)},
        )
        return

    dataset = ds.dataset(data_file)
    total_source_rows = int(dataset.count_rows())
    schema_names = set(dataset.schema.names)
    strat_cols = _seat_strategy_columns(cfg, dataset.schema.names)
    winner_col = _winner_column(schema_names)

    required = {
        *_GAME_COORDINATE_COLUMNS,
        "rng_scheme_version",
        "rng_purpose_namespace",
        "n_rounds",
    }
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

    columns = [
        *_GAME_COORDINATE_COLUMNS,
        "rng_scheme_version",
        "rng_purpose_namespace",
        "n_rounds",
        winner_col,
        *strat_cols,
    ]
    max_matchup_groups = _effective_max_matchup_groups(cfg.analysis.rng_max_matchup_groups)
    prepared_batches = _iter_prepared_batches(
        dataset,
        columns=columns,
        winner_col=winner_col,
        strat_cols=strat_cols,
        batch_size=_STREAM_BATCH_SIZE,
        arrow_threads=policy.arrow_threads,
    )
    diagnostics, melted_rows, capacity = _collect_diagnostics_streaming_compact(
        prepared_batches,
        strat_cols=strat_cols,
        lags=lags,
        progress_logger=ScheduledProgressLogger(
            LOGGER,
            label="rng-diagnostics",
            schedule=cfg.analysis.progress_logging,
            unit="rows",
            total=total_source_rows,
        ),
        max_matchup_groups=max_matchup_groups,
    )
    if melted_rows == 0:
        stage_log.missing_input("no per-strategy rows after melting", path=str(data_file))
        return
    if diagnostics.empty:
        stage_log.missing_input("no diagnostics computed", path=str(data_file))
        return

    table_out = pa.Table.from_pandas(diagnostics, preserve_index=False)
    method_parameters = _rng_method_parameters(capacity, strat_cols=strat_cols)
    sidecar = make_artifact_sidecar(
        cfg,
        out_file,
        producer="rng_diagnostics",
        scope=ArtifactScope.DIAGNOSTICS,
        source_scope=ArtifactScope.CONCAT_KS,
        operation="semantic_coordinate_lag_correlation",
        uncertainty_method=_BAND_METHOD,
        support_count_role="ordered_group_observations",
        replication_unit="lagged_observation_pair",
        conditioning="diagnostic_only_no_independence_claim",
        method_contract=cast(
            MethodContract,
            {
                "kind": "diagnostic_band",
                "procedure": "semantic_coordinate_lag_correlation",
                "parameters": method_parameters,
            },
        ),
        consistency_columns=table_out.schema.names,
        source_artifacts=[data_file],
        grouping_keys=["summary_level", "strategy", "matchup", "n_players", "lag", "metric"],
        player_counts=cfg.sim.n_players_list,
        required_player_counts=cfg.sim.n_players_list,
        missing_cell_policy="not_applicable",
    )
    write_parquet_artifact_atomic(table_out, out_file, sidecar=sidecar, codec=cfg.parquet_codec)
    write_stage_done(
        stamp_path,
        inputs=[data_file],
        outputs=[out_file],
        cfg=cfg,
        config_sha=cfg.config_sha,
        stage="rng_diagnostics",
        stage_config_sha=stage_config_sha,
        cache_key_version=cfg.stage_cache_key_version("rng_diagnostics"),
        sidecar_artifacts=[out_file],
    )
    LOGGER.info(
        "rng-diagnostics: written",
        extra={"stage": "rng_diagnostics", "rows": len(diagnostics), "path": str(out_file)},
    )


def _rng_method_parameters(
    capacity: RNGDiagnosticCapacityMetadata,
    *,
    strat_cols: Sequence[str],
) -> RNGDiagnosticMethodParameters:
    """Build the typed authenticated method metadata for a diagnostic artifact."""

    return {
        "method_version": _DIAGNOSTIC_METHOD_VERSION,
        "rng_scheme_version": RNG_SCHEME_VERSION,
        "purpose_namespace": int(RandomPurpose.TOURNAMENT_PLAYER),
        "global_order_columns": list(_SEAT_COORDINATE_COLUMNS),
        "seat_order": [_rng_seat_index_from_strategy_column(column) for column in strat_cols],
        "sequence_definition": _SEQUENCE_DEFINITION,
        "reference_band_method": _BAND_METHOD,
        "claim": "descriptive_only_no_independence_claim",
        "effective_matchup_group_cap": capacity.effective_matchup_group_cap,
        "normalized_lags": list(capacity.normalized_lags),
        "tracked_matchup_group_count": capacity.tracked_matchup_group_count,
        "skipped_matchup_group_count": capacity.skipped_matchup_group_count,
        "skipped_matchup_row_count": capacity.skipped_matchup_row_count,
    }


def _iter_prepared_batches(
    dataset: ds.Dataset,
    *,
    columns: Sequence[str],
    winner_col: str,
    strat_cols: Sequence[str],
    batch_size: int,
    arrow_threads: int,
) -> Iterator[pd.DataFrame]:
    """Yield prepared game batches retaining every semantic ordering key.

    Args:
        dataset: Curated parquet dataset being scanned.
        columns: Source columns required for diagnostics.
        winner_col: Winner column name resolved from the schema.
        strat_cols: Seat strategy columns to inspect.
        batch_size: Arrow scanner batch size.
        arrow_threads: Number of Arrow threads available to the scanner.

    Yields:
        Data frames containing the full RNG-v2 game coordinate, normalized
        winner, matchup, and player-count columns. Per-batch sorting is only
        preparation for the later disk-backed global merge.
    """
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
        versions = pd.to_numeric(df["rng_scheme_version"], errors="coerce")
        if versions.isna().any() or not (versions == RNG_SCHEME_VERSION).all():
            raise ValueError(f"rng_diagnostics requires RNG scheme version {RNG_SCHEME_VERSION}")
        namespaces = pd.to_numeric(df["rng_purpose_namespace"], errors="coerce")
        if namespaces.isna().any() or not (namespaces == int(RandomPurpose.TOURNAMENT_GAME)).all():
            raise ValueError(
                "rng_diagnostics requires tournament-game outcome coordinates "
                f"(namespace {int(RandomPurpose.TOURNAMENT_GAME)})"
            )
        if df[list(_GAME_COORDINATE_COLUMNS)].isna().any(axis=None):
            raise ValueError("rng_diagnostics semantic game coordinates cannot be null")
        df = df.sort_values(list(_GAME_COORDINATE_COLUMNS), kind="mergesort")
        df["matchup"] = _build_matchup_labels(df, strat_cols)
        df["n_players"] = df[strat_cols].notna().sum(axis=1).astype(int)
        df["winner_strategy"] = _winner_strategies(df, winner_col, strat_cols)
        yield df[
            [
                *_GAME_COORDINATE_COLUMNS,
                "n_rounds",
                "matchup",
                "n_players",
                "winner_strategy",
                *strat_cols,
            ]
        ]


def _sql_text(value: object) -> str | None:
    """Normalize a nullable strategy or matchup value for a temporary SQL run."""

    if _is_missing_scalar(value):
        return None
    return str(value)


def _iter_globally_ordered_game_batches(
    data_batches: Iterable[pd.DataFrame],
    *,
    strat_cols: Sequence[str],
    output_batch_size: int = _GLOBAL_MERGE_BATCH_SIZE,
) -> Iterator[pd.DataFrame]:
    """Externally sort prepared games by the complete semantic coordinate.

    A disposable SQLite table provides a disk-backed global sort, so RAM stays
    bounded and Arrow batch boundaries cannot affect the emitted order. The
    semantic game coordinate is a primary key; duplicate coordinates fail
    because they cannot define one canonical seat-exposure sequence.
    """

    if output_batch_size < 1:
        raise ValueError("output_batch_size must be positive")
    normalized_strat_cols = tuple(strat_cols)
    if any(_SEAT_STRATEGY_RE.fullmatch(column) is None for column in normalized_strat_cols):
        raise ValueError("strat_cols must contain only canonical P<seat>_strategy columns")
    value_columns = (
        *_GAME_COORDINATE_COLUMNS,
        "n_rounds",
        "matchup",
        "n_players",
        "winner_strategy",
        *normalized_strat_cols,
    )
    numeric_columns = (*_GAME_COORDINATE_COLUMNS, "n_rounds", "n_players")
    placeholders = ", ".join("?" for _ in value_columns)
    quoted_columns = ", ".join(f'"{column}"' for column in value_columns)
    order_sql = ", ".join(f'"{column}"' for column in _GAME_COORDINATE_COLUMNS)
    column_definitions = [
        '"root_seed" INTEGER NOT NULL',
        '"k" INTEGER NOT NULL',
        '"shuffle_index" INTEGER NOT NULL',
        '"game_index" INTEGER NOT NULL',
        '"n_rounds" INTEGER NOT NULL',
        '"matchup" TEXT',
        '"n_players" INTEGER NOT NULL',
        '"winner_strategy" TEXT',
        *(f'"{column}" TEXT' for column in normalized_strat_cols),
        'PRIMARY KEY ("root_seed", "k", "shuffle_index", "game_index")',
    ]
    create_table_sql = (
        "CREATE TABLE games (\n"
        + ",\n".join(f"    {definition}" for definition in column_definitions)
        + "\n) WITHOUT ROWID"
    )

    with TemporaryDirectory(prefix="farkle_rng_diagnostics_") as temp_dir:
        database_path = Path(temp_dir) / "global_order.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(create_table_sql)
            insert_sql = f"INSERT INTO games ({quoted_columns}) VALUES ({placeholders})"
            for batch in data_batches:
                if batch.empty:
                    continue
                missing_required = [
                    column
                    for column in value_columns
                    if column not in batch.columns and column not in normalized_strat_cols
                ]
                if missing_required:
                    raise ValueError(
                        "rng_diagnostics prepared batch missing columns: "
                        + ", ".join(missing_required)
                    )
                working = batch.reindex(columns=value_columns)
                records: list[tuple[object, ...]] = []
                for values in working.itertuples(index=False, name=None):
                    raw = dict(zip(value_columns, values, strict=True))
                    if any(_is_missing_scalar(raw[column]) for column in numeric_columns):
                        raise ValueError(
                            "rng_diagnostics ordering and metric columns cannot be null"
                        )
                    records.append(
                        tuple(
                            (
                                int(raw[column])
                                if column in numeric_columns
                                else _sql_text(raw[column])
                            )
                            for column in value_columns
                        )
                    )
                try:
                    connection.executemany(insert_sql, records)
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        "rng_diagnostics requires unique semantic game coordinates"
                    ) from exc
                connection.commit()

            cursor = connection.execute(f"SELECT {quoted_columns} FROM games ORDER BY {order_sql}")
            while rows := cursor.fetchmany(output_batch_size):
                yield pd.DataFrame.from_records(rows, columns=value_columns)
        finally:
            connection.close()


def _merge_seats_in_semantic_order(
    games: pd.DataFrame,
    *,
    strat_cols: Sequence[str],
) -> pd.DataFrame:
    """Return seat exposures in complete RNG-v2 tournament-player order."""

    seat_frames: list[pd.DataFrame] = []
    base_columns = [
        *_GAME_COORDINATE_COLUMNS,
        "matchup",
        "n_players",
        "winner_strategy",
        "n_rounds",
    ]
    for strat_col in strat_cols:
        if strat_col not in games.columns:
            continue
        seat = games[[*base_columns, strat_col]].rename(columns={strat_col: "strategy"})
        seat = seat.dropna(subset=["strategy"]).copy()
        if seat.empty:
            continue
        seat["seat_index"] = _rng_seat_index_from_strategy_column(strat_col)
        seat_frames.append(seat)
    if not seat_frames:
        return pd.DataFrame(
            columns=[
                *_SEAT_COORDINATE_COLUMNS,
                "matchup",
                "strategy",
                "n_players",
                "win_indicator",
                "n_rounds",
            ]
        )

    merged = pd.concat(seat_frames, ignore_index=True)
    merged["strategy"] = merged["strategy"].astype("string")
    merged["winner_strategy"] = merged["winner_strategy"].astype("string")
    merged["win_indicator"] = (
        merged["winner_strategy"].notna() & (merged["strategy"] == merged["winner_strategy"])
    ).astype(np.int8)
    merged = merged.sort_values(list(_SEAT_COORDINATE_COLUMNS), kind="mergesort")
    return merged[
        [
            *_SEAT_COORDINATE_COLUMNS,
            "matchup",
            "strategy",
            "n_players",
            "win_indicator",
            "n_rounds",
        ]
    ]


def _collect_diagnostics_streaming_compact(
    data_batches: Iterable[pd.DataFrame],
    *,
    strat_cols: Sequence[str],
    lags: Sequence[int],
    progress_logger: ScheduledProgressLogger | None,
    max_matchup_groups: int | None,
) -> tuple[pd.DataFrame, int, RNGDiagnosticCapacityMetadata]:
    """Aggregate diagnostics after a global semantic-coordinate merge.

    Args:
        data_batches: Prepared game batches yielded by
            :func:`_iter_prepared_batches`; their input order is irrelevant.
        strat_cols: Seat strategy columns merged in ascending seat order.
        lags: Positive lags to evaluate.
        progress_logger: Optional scheduled progress logger.
        max_matchup_groups: Optional cap on tracked matchup-strategy groups.

    Returns:
        Diagnostics, melted seat-row count, and typed capacity/lag metadata.
    """
    normalized_lags = tuple(int(lag) for lag in lags)
    strategy_states: dict[tuple[str, int], _GroupStreamAccumulator] = {}
    matchup_states: dict[tuple[str | None, str, int], _GroupStreamAccumulator] = {}
    processed_batches = 0
    processed_rows = 0
    melted_rows = 0
    skipped_matchup_state_keys: set[tuple[str | None, str, int]] = set()
    skipped_matchup_rows = 0

    ordered_game_batches = _iter_globally_ordered_game_batches(
        data_batches,
        strat_cols=strat_cols,
    )
    for game_batch in ordered_game_batches:
        batch = _merge_seats_in_semantic_order(game_batch, strat_cols=strat_cols)
        if batch.empty:
            continue
        processed_batches += 1
        processed_rows += int(game_batch.shape[0])
        melted_rows += int(batch.shape[0])

        grouped_strategy = batch.groupby(["strategy", "n_players"], observed=True, sort=False)
        for (strategy, n_players), group in grouped_strategy:
            key = (str(strategy), int(cast(int, n_players)))
            state = strategy_states.setdefault(key, _GroupStreamAccumulator(normalized_lags))
            state.extend(group)

        grouped_matchup = batch.groupby(
            ["matchup", "strategy", "n_players"], observed=True, sort=False
        )
        for (matchup, strategy, n_players), group in grouped_matchup:
            matchup_key = None if _is_missing_scalar(matchup) else str(matchup)
            matchup_state_key = (matchup_key, str(strategy), int(cast(int, n_players)))
            matchup_state = matchup_states.get(matchup_state_key)
            if matchup_state is None:
                if max_matchup_groups is not None and len(matchup_states) >= max_matchup_groups:
                    skipped_matchup_state_keys.add(matchup_state_key)
                    skipped_matchup_rows += int(group.shape[0])
                    continue
                matchup_state = matchup_states.setdefault(
                    matchup_state_key, _GroupStreamAccumulator(normalized_lags)
                )
            matchup_state.extend(group)

        if progress_logger is not None:
            progress_logger.maybe_log(
                processed_rows,
                detail=(
                    f"{processed_batches:,} batches, {melted_rows:,} melted rows, "
                    f"{len(strategy_states):,} strategy groups, "
                    f"{len(matchup_states):,} matchup groups, "
                    f"{len(skipped_matchup_state_keys):,} matchup groups skipped"
                ),
                extra={
                    "stage": "rng_diagnostics",
                    "batches": processed_batches,
                    "rows": processed_rows,
                    "melted_rows": melted_rows,
                    "strategy_groups": len(strategy_states),
                    "matchup_groups": len(matchup_states),
                    "matchup_groups_skipped": len(skipped_matchup_state_keys),
                    "matchup_rows_skipped": skipped_matchup_rows,
                },
            )

    skipped_matchup_groups = len(skipped_matchup_state_keys)
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
    capacity = RNGDiagnosticCapacityMetadata(
        effective_matchup_group_cap=max_matchup_groups,
        normalized_lags=normalized_lags,
        tracked_matchup_group_count=len(matchup_states),
        skipped_matchup_group_count=skipped_matchup_groups,
        skipped_matchup_row_count=skipped_matchup_rows,
    )
    if not rows:
        return pd.DataFrame(), melted_rows, capacity

    diagnostics = pd.DataFrame(rows)
    diagnostics = diagnostics.sort_values(
        ["summary_level", "strategy", "matchup", "n_players", "lag", "metric"],
        kind="mergesort",
        na_position="first",
    )
    diagnostics.reset_index(drop=True, inplace=True)
    return diagnostics, melted_rows, capacity


def _normalize_lags(lags: Sequence[int] | None) -> tuple[int, ...]:
    """Normalize lag inputs to a sorted tuple of unique positive integers.

    Args:
        lags: Optional raw lag configuration from callers.

    Returns:
        Sorted positive lag tuple, defaulting to ``(1,)`` when unset.
    """
    if lags is None:
        return (1,)
    valid = sorted({int(lag) for lag in lags if int(lag) > 0})
    return tuple(valid)


def _effective_max_matchup_groups(configured_cap: int | None) -> int | None:
    """Resolve the public cap to the exact capacity used by the collector."""

    if configured_cap is None:
        return _DEFAULT_MAX_MATCHUP_GROUPS
    if configured_cap <= 0:
        return None
    return int(configured_cap)


def _winner_column(names: set[str]) -> str | None:
    """Pick the preferred winner column from a parquet schema name set.

    Args:
        names: Available column names from the curated parquet schema.

    Returns:
        ``winner_strategy``, ``winner_seat``, or ``None`` when neither exists.
    """
    if "winner_strategy" in names:
        return "winner_strategy"
    if "winner_seat" in names:
        return "winner_seat"
    return None


def _seat_strategy_columns(cfg: AppConfig, schema_names: Sequence[str]) -> list[str]:
    """Resolve and sort seat strategy columns available in the curated schema.

    Args:
        cfg: Application config used for configured player-count hints.
        schema_names: Column names present in the curated parquet schema.

    Returns:
        Sorted seat strategy column names that actually exist in the schema.
    """
    schema_set = set(schema_names)
    configured_candidates: list[str] = []
    if cfg.sim.n_players_list:
        max_players = max(cfg.sim.n_players_list)
        configured_candidates = [f"P{seat}_strategy" for seat in range(1, max_players + 1)]

    fallback_candidates = [
        name for name in schema_names if _SEAT_STRATEGY_RE.match(name) and name != "winner_strategy"
    ]
    merged_candidates = set(configured_candidates) | set(fallback_candidates)
    present = [
        name for name in merged_candidates if name in schema_set and name != "winner_strategy"
    ]

    return sorted(present, key=_seat_number_from_strategy_column)


def _build_matchup_labels(df: pd.DataFrame, strat_cols: Sequence[str]) -> pd.Series:
    """Build canonical matchup labels by sorting participating strategy names.

    Args:
        df: Batch containing seat strategy columns.
        strat_cols: Candidate seat strategy columns to include in the label.

    Returns:
        String series aligned to ``df.index`` with matchup labels per game row.
    """
    valid_seat_cols = [
        col for col in strat_cols if col in df.columns and _SEAT_STRATEGY_RE.match(col)
    ]
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


def _winner_strategies(df: pd.DataFrame, winner_col: str, strat_cols: Sequence[str]) -> pd.Series:
    """Resolve winner strategies even when the source stores winning seats.

    Args:
        df: Batch containing winner and seat strategy columns.
        winner_col: Winner column name, either strategy- or seat-based.
        strat_cols: Candidate seat strategy columns in seat order.

    Returns:
        String series containing the winning strategy for each row when resolvable.
    """
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
    """Extract the numeric seat index from a ``P<n>_strategy`` column name.

    Args:
        column_name: Seat strategy column name to parse.

    Returns:
        Parsed seat number.
    """
    match = _SEAT_STRATEGY_RE.match(column_name)
    if match is None:
        raise ValueError(f"invalid seat strategy column: {column_name}")
    return int(match.group(1))


def _rng_seat_index_from_strategy_column(column_name: str) -> int:
    """Return the zero-based RNG-v2 seat coordinate for a strategy column."""

    return _seat_number_from_strategy_column(column_name) - 1


def _melt_strategies(df: pd.DataFrame, strat_cols: Sequence[str]) -> pd.DataFrame:
    """Convert seat strategy columns into one seat-level row per strategy occurrence.

    Args:
        df: Prepared batch containing matchup, winner, and seat strategy columns.
        strat_cols: Seat strategy columns to melt.

    Returns:
        Seat-level frame with ``strategy`` and ``win_indicator`` columns.
    """
    id_vars = [
        *_GAME_COORDINATE_COLUMNS,
        "n_rounds",
        "matchup",
        "n_players",
        "winner_strategy",
    ]
    melted = df[id_vars + list(strat_cols)].melt(
        id_vars=id_vars,
        value_vars=strat_cols,
        var_name="seat",
        value_name="strategy",
    )
    melted = melted.dropna(subset=["strategy"])
    melted["strategy"] = melted["strategy"].astype("string")
    melted["winner_strategy"] = melted["winner_strategy"].astype("string")
    melted["seat_index"] = melted["seat"].map(_rng_seat_index_from_strategy_column)
    melted["win_indicator"] = (
        melted["winner_strategy"].notna() & (melted["strategy"] == melted["winner_strategy"])
    ).astype(int)
    return melted.sort_values(list(_SEAT_COORDINATE_COLUMNS), kind="mergesort")


@dataclass(slots=True)
class _LagCorrelationAccumulator:
    """Incrementally accumulate covariance terms for one lagged metric."""

    pair_count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: float, y: float) -> None:
        """Add one lagged pair to the accumulator.

        Args:
            x: Earlier observation in the lagged pair.
            y: Later observation in the lagged pair.
        """
        self.pair_count += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_x2 += x * x
        self.sum_y2 += y * y
        self.sum_xy += x * y

    def autocorr(self) -> float | None:
        """Compute the Pearson autocorrelation implied by the accumulated pairs.

        Returns:
            Correlation coefficient, or ``None`` when variance is insufficient.
        """
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
    """Maintain lagged state for one numeric metric streamed in time order."""

    lags: tuple[int, ...]
    n_obs: int = 0
    buffers: dict[int, deque[float]] = field(default_factory=dict)
    states: dict[int, _LagCorrelationAccumulator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.buffers = {lag: deque(maxlen=lag) for lag in self.lags}
        self.states = {lag: _LagCorrelationAccumulator() for lag in self.lags}

    def extend(self, values: pd.Series) -> None:
        """Add an ordered series of numeric observations to the stream state.

        Args:
            values: Metric series already ordered by the declared semantic coordinate.
        """
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        for value in numeric:
            if not np.isfinite(value):
                continue
            self._push(float(value))

    def _push(self, value: float) -> None:
        """Push one finite observation through all lag buffers.

        Args:
            value: Finite metric value to append to the stream.
        """
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
    """Track streamed lagged state for both win and round-count metrics."""

    lags: tuple[int, ...]
    win_indicator: _MetricStreamAccumulator = field(init=False)
    n_rounds: _MetricStreamAccumulator = field(init=False)

    def __post_init__(self) -> None:
        self.win_indicator = _MetricStreamAccumulator(self.lags)
        self.n_rounds = _MetricStreamAccumulator(self.lags)

    def extend(self, frame: pd.DataFrame) -> None:
        """Extend both metric streams from an ordered seat-level batch.

        Args:
            frame: Ordered frame containing ``win_indicator`` and ``n_rounds`` columns.
        """
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
    """Yield fully melted seat-level batches from the curated parquet dataset.

    Args:
        dataset: Curated parquet dataset being scanned.
        columns: Source columns required for diagnostics.
        winner_col: Winner column name resolved from the schema.
        strat_cols: Seat strategy columns to melt.
        batch_size: Arrow scanner batch size.
        arrow_threads: Number of Arrow threads available to the scanner.

    Yields:
        Seat-level frames sorted within the batch by the complete semantic
        coordinate. This compatibility helper is not the canonical global merge.
    """
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
        df = df.sort_values(list(_GAME_COORDINATE_COLUMNS), kind="mergesort")
        df["matchup"] = _build_matchup_labels(df, strat_cols)
        df["n_players"] = df[strat_cols].notna().sum(axis=1).astype(int)
        df["winner_strategy"] = _winner_strategies(df, winner_col, strat_cols)
        melted = _melt_strategies(df, strat_cols)
        if melted.empty:
            continue
        yield melted[
            [
                *_SEAT_COORDINATE_COLUMNS,
                "strategy",
                "matchup",
                "n_players",
                "win_indicator",
                "n_rounds",
            ]
        ]


def _rows_from_group_state(
    *,
    summary_level: str,
    strategy: str,
    n_players: int,
    lags: Sequence[int],
    group_state: _GroupStreamAccumulator,
    matchup: str | None = None,
) -> list[dict[str, object]]:
    """Materialize diagnostic rows from one accumulated strategy or matchup group.

    Args:
        summary_level: Output summary level label.
        strategy: Strategy identifier for the group.
        n_players: Player count for the group.
        lags: Lags to emit diagnostics for.
        group_state: Stream accumulator containing metric states.
        matchup: Optional matchup label for matchup-strategy rows.

    Returns:
        Output rows ready to append to the diagnostics frame.
    """
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
            lag_state = metric_state.states[lag]
            autocorr = lag_state.autocorr()
            if autocorr is None or pd.isna(autocorr):
                continue
            reference_half_width = 1.96 / lag_state.pair_count**0.5
            rows.append(
                {
                    "summary_level": summary_level,
                    "strategy": strategy,
                    "matchup": matchup,
                    "n_players": n_players,
                    "observations": n_obs,
                    "lagged_pairs": lag_state.pair_count,
                    "lag": lag,
                    "metric": metric,
                    "autocorr": autocorr,
                    "zero_centered_descriptive_reference_band_lower": -reference_half_width,
                    "zero_centered_descriptive_reference_band_upper": reference_half_width,
                    "sequence_order": ",".join(_SEAT_COORDINATE_COLUMNS),
                    "note": _EXPECTED_NOTE,
                }
            )
    return rows


def _collect_diagnostics_streaming(
    data_batches: Iterable[pd.DataFrame],
    *,
    lags: Sequence[int],
    progress_logger: ScheduledProgressLogger | None,
    max_matchup_groups: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Aggregate diagnostics from pre-melted batches using streaming group state.

    Args:
        data_batches: Melted seat-level batches sorted by ``game_seed``.
        lags: Positive lags to evaluate.
        progress_logger: Optional scheduled progress logger.
        max_matchup_groups: Optional cap on tracked matchup-strategy groups.

    Returns:
        Tuple of the diagnostics frame and total melted rows processed.
    """
    normalized_lags = tuple(int(lag) for lag in lags)
    strategy_states: dict[tuple[str, int], _GroupStreamAccumulator] = {}
    matchup_states: dict[tuple[str | None, str, int], _GroupStreamAccumulator] = {}
    processed_batches = 0
    processed_rows = 0
    melted_rows = 0
    skipped_matchup_groups = 0
    skipped_matchup_rows = 0

    for batch in data_batches:
        if batch.empty:
            continue
        processed_batches += 1
        processed_rows += int(batch.shape[0])
        melted_rows += int(batch.shape[0])

        grouped_strategy = batch.groupby(["strategy", "n_players"], observed=True, sort=False)
        for (strategy, n_players), group in grouped_strategy:
            key = (str(strategy), int(cast(int, n_players)))
            state = strategy_states.setdefault(key, _GroupStreamAccumulator(normalized_lags))
            ordered = group.sort_values(list(_SEAT_COORDINATE_COLUMNS), kind="mergesort")
            state.extend(ordered)

        grouped_matchup = batch.groupby(
            ["matchup", "strategy", "n_players"], observed=True, sort=False
        )
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
            ordered = group.sort_values(list(_SEAT_COORDINATE_COLUMNS), kind="mergesort")
            matchup_state.extend(ordered)

        if progress_logger is not None:
            progress_logger.maybe_log(
                processed_rows,
                detail=(
                    f"{processed_batches:,} batches, {melted_rows:,} melted rows, "
                    f"{len(strategy_states):,} strategy groups, "
                    f"{len(matchup_states):,} matchup groups, "
                    f"{skipped_matchup_groups:,} matchup groups skipped"
                ),
                extra={
                    "stage": "rng_diagnostics",
                    "batches": processed_batches,
                    "rows": processed_rows,
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
        ["summary_level", "strategy", "matchup", "n_players", "lag", "metric"],
        kind="mergesort",
        na_position="first",
    )
    diagnostics.reset_index(drop=True, inplace=True)
    return diagnostics, melted_rows


def _collect_diagnostics(data: pd.DataFrame, *, lags: Iterable[int]) -> pd.DataFrame:
    """Compute diagnostics eagerly from one fully materialized melted frame.

    Args:
        data: Melted seat-level diagnostics input frame.
        lags: Positive lags to evaluate.

    Returns:
        Diagnostics frame spanning strategy and matchup-strategy groupings.
    """
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

    grouped_matchup = data.groupby(["matchup", "strategy", "n_players"], observed=True, sort=False)
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
    diagnostics = diagnostics.sort_values(
        ["summary_level", "strategy", "matchup", "n_players", "lag", "metric"],
        na_position="first",
    )
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
    """Compute lagged diagnostics for one grouped, ordered seat-level frame.

    Args:
        group: Grouped seat-level frame for one strategy or matchup-strategy key.
        lags: Positive lags to evaluate.
        summary_level: Output summary level label.
        strategy: Strategy identifier for the group.
        n_players: Player count for the group.
        matchup: Optional matchup label for matchup-strategy rows.

    Returns:
        Diagnostic rows as series objects suitable for later concatenation.
    """
    rows: list[pd.Series] = []
    ordered = group.sort_values(list(_SEAT_COORDINATE_COLUMNS), kind="mergesort")
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
            lagged_pairs = n_obs - lag
            reference_half_width = 1.96 / lagged_pairs**0.5
            rows.append(
                pd.Series(
                    {
                        "summary_level": summary_level,
                        "strategy": strategy,
                        "matchup": matchup,
                        "n_players": n_players,
                        "observations": n_obs,
                        "lagged_pairs": lagged_pairs,
                        "lag": lag,
                        "metric": metric,
                        "autocorr": autocorr,
                        "zero_centered_descriptive_reference_band_lower": (-reference_half_width),
                        "zero_centered_descriptive_reference_band_upper": (reference_half_width),
                        "sequence_order": ",".join(_SEAT_COORDINATE_COLUMNS),
                        "note": _EXPECTED_NOTE,
                    }
                )
            )
    return rows


def _rng_stage_config_sha(cfg: AppConfig, lags: Sequence[int]) -> str:
    normalized_lags = _normalize_lags(lags)
    payload = {
        "base_stage_config_sha": cfg.stage_config_sha("rng_diagnostics"),
        "diagnostic_method_version": _DIAGNOSTIC_METHOD_VERSION,
        "rng_scheme_version": RNG_SCHEME_VERSION,
        "purpose_namespace": int(RandomPurpose.TOURNAMENT_PLAYER),
        "sequence_order": list(_SEAT_COORDINATE_COLUMNS),
        "sequence_definition": _SEQUENCE_DEFINITION,
        "reference_band_method": _BAND_METHOD,
        "effective_matchup_group_cap": _effective_max_matchup_groups(
            cfg.analysis.rng_max_matchup_groups
        ),
        "normalized_lags": [int(lag) for lag in normalized_lags],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    config = AppConfig()
    run(config)
