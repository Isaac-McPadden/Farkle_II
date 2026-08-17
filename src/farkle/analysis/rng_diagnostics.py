"""Bounded, deterministic RNG-v2 lag diagnostics.

The stage has four authenticated, resumable phases: count routing by source row
group, exact eligibility reduction by stable hash partition, eligible-observation
routing by source row group, and lag reduction by stable partition.  No phase
globally sorts or materializes the curated table.

Grouping semantics are deliberately non-redundant:

* ``strategy`` groups are seat exposures keyed by ``(strategy_id, k)`` and
  diagnose win indicators and game rounds.
* ``matchup`` groups are games keyed by the sorted participant-ID multiset and
  ``k`` and diagnose game rounds once per game.  A matchup is not expanded into
  one copy per participant.

All temporal sequences use the lexicographic RNG-v2 tournament-player
coordinate.  Matchup sequences omit seat because they contain one observation
per game.  Reference bands are descriptive and make no independence claim.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import sys
from collections.abc import Generator, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from farkle.analysis import stage_logger
from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import (
    MethodContract,
    make_artifact_sidecar,
    sha256_file,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import write_json_artifact_atomic
from farkle.utils.parallel import (
    ProcessTreeMemoryGuard,
    ResourceSafetyError,
    apply_native_thread_limits,
    resolve_stage_parallel_policy,
)
from farkle.utils.partitioned_stage import (
    PartitionedStageIdentity,
    PartitionedStageResult,
    PartitionedUnit,
    resolved_code_identity_sha256,
    run_partitioned_stage,
)
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose
from farkle.utils.release_identity import is_v3_config
from farkle.utils.stage_completion import (
    CompletionState,
    resolve_stage_state,
    stage_done_path,
    write_stage_done,
)
from farkle.utils.telemetry import (
    current_supervisor_recorder,
    current_supervisor_scope,
    report_worker_progress,
)
from farkle.utils.writer import ParquetShardWriter

LOGGER = logging.getLogger(__name__)

_EXPECTED_NOTE = (
    "Zero-centered approximate descriptive reference band only; values inside or "
    "outside the band do not establish or refute independence"
)
_BAND_METHOD = "zero_centered_1.96_over_sqrt_lagged_pairs_descriptive_reference_band"
_DIAGNOSTIC_METHOD_VERSION = 4
_PARTITION_SCHEMA_VERSION = 1
_GAME_COORDINATE_COLUMNS = ("root_seed", "k", "shuffle_index", "game_index")
_SEAT_COORDINATE_COLUMNS = (*_GAME_COORDINATE_COLUMNS, "seat_index")
_SEQUENCE_DEFINITION = "externally_partitioned_rng_v2_semantic_coordinate_then_group_filter"
_DEFAULT_MAX_MATCHUP_GROUPS = 100_000
_GROUP_STRATEGY = 0
_GROUP_MATCHUP = 1
_MISSING_STRATEGY = -1
_RUN_MERGE_FAN_IN = 32
_RUN_WRITE_ROWS = 8_192


@dataclass(frozen=True)
class RNGDiagnosticCapacityMetadata:
    """Exact capacity, exclusion, support, and completion metadata."""

    effective_matchup_group_cap: int | None
    normalized_lags: tuple[int, ...]
    partition_count: int
    minimum_usable_observations: int
    total_candidate_group_count: int
    candidate_strategy_group_count: int
    candidate_matchup_group_count: int
    eligible_group_count: int
    eligible_strategy_group_count: int
    eligible_matchup_group_count: int
    selected_group_count: int
    selected_strategy_group_count: int
    selected_matchup_group_count: int
    below_minimum_group_count: int
    deterministically_capped_group_count: int
    observation_count_distribution: tuple[dict[str, int | str], ...]
    usable_groups_per_lag: tuple[dict[str, int | str], ...]
    completeness_status: str

    @property
    def tracked_matchup_group_count(self) -> int:
        """Compatibility alias for authenticated release metadata."""

        return self.selected_matchup_group_count

    @property
    def skipped_matchup_group_count(self) -> int:
        """Compatibility alias covering explicit, classified exclusions."""

        return self.candidate_matchup_group_count - self.selected_matchup_group_count

    @property
    def skipped_matchup_row_count(self) -> int:
        """Retired encounter-order row-drop count (always zero in method v4)."""

        return 0


class RNGDiagnosticMethodParameters(TypedDict, total=False):
    """Authenticated method parameters shared by final diagnostic artifacts."""

    method_version: int
    rng_diagnostic_method_version: int
    rng_scheme_version: int
    purpose_namespace: int
    global_order_columns: list[str]
    matchup_order_columns: list[str]
    sequence_definition: str
    grouping_semantics: dict[str, str]
    reference_band_method: str
    claim: str
    effective_matchup_group_cap: int | None
    normalized_lags: list[int]
    partition_count: int
    minimum_usable_observations: int
    total_candidate_group_count: int
    eligible_group_count: int
    selected_group_count: int
    below_minimum_group_count: int
    deterministically_capped_group_count: int
    completeness_status: str
    tracked_matchup_group_count: int
    skipped_matchup_group_count: int
    skipped_matchup_row_count: int


@dataclass(frozen=True, slots=True)
class _BatchArrays:
    root_seed: np.ndarray
    k: np.ndarray
    shuffle_index: np.ndarray
    game_index: np.ndarray
    n_rounds: np.ndarray
    winner_strategy: np.ndarray
    seats: np.ndarray
    canonical_matchup: np.ndarray
    matchup_id: np.ndarray


@contextmanager
def _temporary_workspace(*, prefix: str) -> Iterator[Path]:
    """Clean a worker workspace without masking an active processing error."""

    directory = TemporaryDirectory(prefix=prefix)
    try:
        yield Path(directory.name)
    except BaseException:
        try:
            directory.cleanup()
        except BaseException:
            LOGGER.exception("RNG diagnostic temporary-workspace cleanup also failed")
        raise
    else:
        directory.cleanup()


@dataclass(frozen=True, slots=True)
class _CountRouteWriter:
    sources: tuple[tuple[int, str, int], ...]
    columns: tuple[str, ...]
    winner_col: str
    strat_cols: tuple[str, ...]
    partition_count: int
    batch_bytes: int
    expected_root_seed: int

    def __call__(self, unit: PartitionedUnit, path: Path) -> None:
        source_map = {ordinal: (source, row_group) for ordinal, source, row_group in self.sources}
        source, row_group = source_map[int(unit.key[0])]
        schema = _count_arrow_schema(len(self.strat_cols))
        with path.open("wb") as handle, ipc.new_file(handle, schema) as writer:
            batches = _iter_row_group_batches(
                Path(source),
                row_group,
                columns=self.columns,
                batch_bytes=self.batch_bytes,
                expansion=max(2, len(self.strat_cols) + 1),
            )
            with closing(batches):
                for batch in batches:
                    arrays = _extract_batch_arrays(
                        batch,
                        winner_col=self.winner_col,
                        strat_cols=self.strat_cols,
                        expected_root_seed=self.expected_root_seed,
                    )
                    records = _count_records(arrays)
                    partitions = _stable_partitions(records, self.partition_count)
                    for partition in range(self.partition_count):
                        writer.write_batch(
                            _count_records_to_batch(records[partitions == partition], schema)
                        )


@dataclass(frozen=True, slots=True)
class _EligibilityWriter:
    count_route_root: str
    count_route_units: int
    partition_count: int
    max_players: int
    minimum_observations: int
    root_seed: int
    batch_rows: int

    def __call__(self, unit: PartitionedUnit, path: Path) -> None:
        partition = int(unit.key[0])
        dtype = _count_dtype(self.max_players)
        with _temporary_workspace(prefix=f"farkle_rng_count_p{partition:03d}_") as temp:
            runs: list[Path] = []
            for source_unit in range(self.count_route_units):
                route_path = (
                    Path(self.count_route_root) / "units" / f"row-group-{source_unit:05d}.arrow"
                )
                with route_path.open("rb") as handle, ipc.open_file(handle) as reader:
                    for batch_index in range(
                        partition, reader.num_record_batches, self.partition_count
                    ):
                        batch = reader.get_batch(batch_index)
                        if batch.num_rows == 0:
                            continue
                        run = temp / f"run-{len(runs):06d}.bin"
                        _reduce_count_array(_count_batch_to_records(batch, dtype)).tofile(run)
                        runs.append(run)
            report_worker_progress(
                "rng_count_spill_complete",
                counters={
                    "spill_runs_created": len(runs),
                    "spill_bytes_written": _safe_file_bytes(runs),
                    "route_units_scanned": self.count_route_units,
                },
            )
            merged = _collapse_count_runs(runs, temp, dtype)
            _write_eligibility_partition(
                merged,
                path,
                dtype=dtype,
                partition=partition,
                minimum_observations=self.minimum_observations,
                root_seed=self.root_seed,
                batch_rows=self.batch_rows,
            )


@dataclass(frozen=True, slots=True)
class _StatsRouteWriter:
    sources: tuple[tuple[int, str, int], ...]
    selection: str
    columns: tuple[str, ...]
    winner_col: str
    strat_cols: tuple[str, ...]
    partition_count: int
    batch_bytes: int
    expected_root_seed: int
    selection_memory_bytes: int

    def __call__(self, unit: PartitionedUnit, path: Path) -> None:
        memberships = _load_selection_memberships(
            Path(self.selection),
            partition_count=self.partition_count,
            max_players=len(self.strat_cols),
            max_bytes=self.selection_memory_bytes,
        )
        source_map = {ordinal: (source, row_group) for ordinal, source, row_group in self.sources}
        source, row_group = source_map[int(unit.key[0])]
        schema = _observation_arrow_schema(len(self.strat_cols))
        with path.open("wb") as handle, ipc.new_file(handle, schema) as writer:
            batches = _iter_row_group_batches(
                Path(source),
                row_group,
                columns=self.columns,
                batch_bytes=self.batch_bytes,
                expansion=max(2, len(self.strat_cols) + 1),
            )
            with closing(batches):
                for batch in batches:
                    arrays = _extract_batch_arrays(
                        batch,
                        winner_col=self.winner_col,
                        strat_cols=self.strat_cols,
                        expected_root_seed=self.expected_root_seed,
                    )
                    records = _observation_records(arrays)
                    partitions = _stable_partitions(records, self.partition_count)
                    for partition in range(self.partition_count):
                        subset = records[partitions == partition]
                        if subset.size:
                            subset = subset[_membership_mask(subset, memberships[partition])]
                        writer.write_batch(_observation_records_to_batch(subset, schema))


@dataclass(frozen=True, slots=True)
class _StatsPartitionWriter:
    stats_route_root: str
    stats_route_units: int
    partition_count: int
    max_players: int
    lags: tuple[int, ...]
    batch_rows: int

    def __call__(self, unit: PartitionedUnit, path: Path) -> None:
        partition = int(unit.key[0])
        dtype = _observation_dtype(self.max_players)
        with _temporary_workspace(prefix=f"farkle_rng_stats_p{partition:03d}_") as temp:
            runs: list[Path] = []
            for source_unit in range(self.stats_route_units):
                route_path = (
                    Path(self.stats_route_root) / "units" / f"row-group-{source_unit:05d}.arrow"
                )
                with route_path.open("rb") as handle, ipc.open_file(handle) as reader:
                    for batch_index in range(
                        partition, reader.num_record_batches, self.partition_count
                    ):
                        batch = reader.get_batch(batch_index)
                        if batch.num_rows == 0:
                            continue
                        run = temp / f"run-{len(runs):06d}.bin"
                        records = _observation_batch_to_records(batch, dtype)
                        records[_observation_sort_order(records)].tofile(run)
                        runs.append(run)
            report_worker_progress(
                "rng_stats_spill_complete",
                counters={
                    "spill_runs_created": len(runs),
                    "spill_bytes_written": _safe_file_bytes(runs),
                    "route_units_scanned": self.stats_route_units,
                },
            )
            merged = _collapse_observation_runs(runs, temp, dtype)
            _write_stats_partition(
                merged,
                path,
                dtype=dtype,
                lags=self.lags,
                batch_rows=self.batch_rows,
            )


def run(cfg: AppConfig, *, lags: Sequence[int] | None = None, force: bool = False) -> None:
    """Compute bounded, resumable lag diagnostics from curated rows."""

    stage_log = stage_logger("rng_diagnostics", logger=LOGGER)
    stage_log.start()
    recorder = current_supervisor_recorder()
    scope = current_supervisor_scope()
    if scope is not None:
        scope.update(phase="rng_source_preparation", state="authenticating")
    policy = resolve_stage_parallel_policy("rng_diagnostics", cfg.analysis, resources=cfg.resources)
    apply_native_thread_limits(policy)
    pa.set_cpu_count(policy.arrow_threads)
    pa.set_io_thread_count(policy.arrow_threads)
    guard = ProcessTreeMemoryGuard(
        cfg.resources.aggregate_memory_hard_limit_mb,
        rss_warning_mb=cfg.resources.process_tree_warning_threshold_mb,
        minimum_system_available_memory_mb=cfg.resources.minimum_system_available_memory_mb,
        sample_interval_seconds=cfg.resources.rss_sample_interval_seconds,
    )
    guard.check_before_schedule(force=True)

    try:
        from farkle.analysis.combine import combined_partition_paths

        data_file = cfg.combined_manifest_path()
        data_partitions = combined_partition_paths(cfg)
    except KeyError as exc:
        stage_log.missing_input(str(exc))
        return
    out_file = cfg.rng_output_path("rng_diagnostics.parquet")
    summary_file = cfg.rng_output_path("rng_diagnostics_summary.json")
    selection_file = cfg.rng_output_path("rng_group_selection.parquet")
    selection_report_file = cfg.rng_output_path("rng_group_selection_summary.json")
    stamp_path = stage_done_path(cfg.rng_stage_dir, "rng_diagnostics")
    normalized_lags = _normalize_lags(cfg.analysis.rng_diagnostic_lags if lags is None else lags)
    if not normalized_lags:
        stage_log.missing_input("no valid lags provided")
        return
    if not data_file.exists():
        stage_log.missing_input("missing concat_ks partition manifest", path=str(data_file))
        return
    if is_v3_config(cfg):
        validate_artifact_sidecar(
            data_file,
            expected={"scope": ArtifactScope.CONCAT_KS.value, "operation": "concatenate"},
        )

    stage_config_sha = _rng_stage_config_sha(cfg, normalized_lags)
    state = resolve_stage_state(
        stamp_path,
        [data_file],
        [out_file, summary_file],
        cfg=cfg,
        stage="rng_diagnostics",
        sidecar_artifacts=[out_file, summary_file],
    )
    if not force and state in {CompletionState.COMPLETE_VALID, CompletionState.BLOCKED_BY_CAP}:
        LOGGER.info(
            "rng-diagnostics: current authenticated result reused", extra={"state": state.value}
        )
        if recorder is not None and scope is not None:
            summary: dict[str, object] = {
                "completion_status": state.value,
                "reconciled_from": "authenticated_stage_completion",
            }
            try:
                persisted = json.loads(summary_file.read_text(encoding="utf-8"))
                for key in (
                    "partition_count",
                    "normalized_lags",
                    "total_candidate_group_count",
                    "eligible_group_count",
                    "selected_group_count",
                    "tracked_matchup_group_count",
                    "skipped_matchup_group_count",
                    "skipped_matchup_row_count",
                ):
                    if key in persisted:
                        summary[key] = persisted[key]
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            recorder.record_completion_summary(
                f"{scope.scope}:rng_diagnostics",
                stage="rng_diagnostics",
                summary=summary,
            )
        return

    with pq.ParquetFile(data_partitions[0]) as parquet:
        schema_names = set(parquet.schema_arrow.names)
        strat_cols = tuple(_seat_strategy_columns(cfg, parquet.schema_arrow.names))
    winner_col = _winner_column(schema_names)
    required = {
        *_GAME_COORDINATE_COLUMNS,
        "rng_scheme_version",
        "rng_purpose_namespace",
        "n_rounds",
    }
    if winner_col is None or not strat_cols or not required.issubset(schema_names):
        stage_log.missing_input(
            "concat_ks partitions missing RNG diagnostic columns",
            path=str(data_file),
            required_cols=sorted(required | {"winner_strategy", "P1_strategy"}),
        )
        return

    columns = (
        *_GAME_COORDINATE_COLUMNS,
        "rng_scheme_version",
        "rng_purpose_namespace",
        "n_rounds",
        winner_col,
        *strat_cols,
    )
    partitions = int(cfg.analysis.rng_diagnostic_partitions)
    row_group_sources = tuple(
        (ordinal, str(path), row_group)
        for ordinal, (path, row_group) in enumerate(
            (path, row_group)
            for path in data_partitions
            for row_group in range(_parquet_num_row_groups(path))
        )
    )
    row_groups = len(row_group_sources)
    batch_bytes = int(
        cfg.resources.stage_batch_bytes.get(
            "rng_diagnostics", cfg.resources.stage_batch_bytes["analysis"]
        )
    )
    stage_root = cfg.rng_stage_dir / "checkpoints" / stage_config_sha[:20]
    source_sha = sha256_file(data_file)
    code_sha = _diagnostic_code_sha256(cfg)
    root_seed = int(cfg.sim.seed)
    minimum_observations = min(normalized_lags) + 2

    count_route_root = stage_root / "01_count_route"
    count_route = run_partitioned_stage(
        root=count_route_root,
        identity=_partition_identity(
            "rng_diagnostics_count_route",
            root_seed,
            (("curated_rows", source_sha),),
            stage_config_sha,
            code_sha,
        ),
        unit_source=lambda: _row_group_units(row_groups),
        writer=_CountRouteWriter(
            row_group_sources,
            tuple(columns),
            winner_col,
            strat_cols,
            partitions,
            batch_bytes,
            root_seed,
        ),
        resources=cfg.resources,
        requested_workers=cfg.analysis.n_jobs,
        mp_start_method=cfg.analysis.mp_start_method,
        force=force,
        memory_guard=guard,
        progress_total_units=row_groups,
        progress_phase="rng_count_route",
    )

    eligibility_root = stage_root / "02_eligibility"
    eligibility = run_partitioned_stage(
        root=eligibility_root,
        identity=_partition_identity(
            "rng_diagnostics_eligibility",
            root_seed,
            (("count_route_manifest", count_route.manifest_sha256),),
            stage_config_sha,
            code_sha,
        ),
        unit_source=lambda: _partition_units(partitions, "part", ".parquet"),
        writer=_EligibilityWriter(
            str(count_route_root),
            row_groups,
            partitions,
            len(strat_cols),
            minimum_observations,
            root_seed,
            max(256, batch_bytes // max(1, _count_dtype(len(strat_cols)).itemsize)),
        ),
        resources=cfg.resources,
        requested_workers=cfg.analysis.n_jobs,
        mp_start_method=cfg.analysis.mp_start_method,
        force=force,
        memory_guard=guard,
        progress_total_units=partitions,
        progress_phase="rng_eligibility_reduce",
        enable_worker_progress=True,
    )

    if scope is not None:
        scope.update(
            phase="rng_selection",
            state="working",
            progress={
                "eligibility_partitions": eligibility.required_units,
                "eligibility_reused": eligibility.reused_units,
                "eligibility_completed": eligibility.completed_units,
            },
        )
    selection_report = _write_or_reuse_selection(
        cfg,
        eligibility_root=eligibility_root,
        eligibility_manifest=eligibility.manifest_path,
        eligibility_manifest_sha=eligibility.manifest_sha256,
        partition_count=partitions,
        max_players=len(strat_cols),
        minimum_observations=minimum_observations,
        lags=normalized_lags,
        selection_file=selection_file,
        report_file=selection_report_file,
        stage_config_sha=stage_config_sha,
        source=data_file,
        force=force,
        guard=guard,
    )
    if scope is not None:
        scope.update(
            phase="rng_selection_complete",
            state="working",
            progress={
                "candidate_groups": int(selection_report["total_candidate_groups"]),
                "eligible_groups": int(selection_report["eligible_groups"]),
                "selected_groups": int(selection_report["selected_groups"]),
                "capped_groups": int(selection_report["deterministically_capped_groups"]),
                "skipped_groups": int(
                    selection_report["below_minimum_observation_groups"]
                ),
            },
        )
    selection_sha = sha256_file(selection_file)

    stats_route_root = stage_root / "03_stats_route"
    stats_route = run_partitioned_stage(
        root=stats_route_root,
        identity=_partition_identity(
            "rng_diagnostics_stats_route",
            root_seed,
            (("curated_rows", source_sha), ("selection", selection_sha)),
            stage_config_sha,
            code_sha,
        ),
        unit_source=lambda: _row_group_units(row_groups),
        writer=_StatsRouteWriter(
            row_group_sources,
            str(selection_file),
            tuple(columns),
            winner_col,
            strat_cols,
            partitions,
            batch_bytes,
            root_seed,
            max(batch_bytes * 4, 64 * 1024 * 1024),
        ),
        resources=cfg.resources,
        requested_workers=cfg.analysis.n_jobs,
        mp_start_method=cfg.analysis.mp_start_method,
        force=force,
        memory_guard=guard,
        progress_total_units=row_groups,
        progress_phase="rng_stats_route",
    )

    stats_root = stage_root / "04_stats"
    if scope is not None:
        scope.update(
            phase="rng_lag_reduce",
            state="working",
            progress={"lags": len(normalized_lags), "partitions": partitions},
        )
    stats = run_partitioned_stage(
        root=stats_root,
        identity=_partition_identity(
            "rng_diagnostics_stats",
            root_seed,
            (("stats_route_manifest", stats_route.manifest_sha256),),
            stage_config_sha,
            code_sha,
        ),
        unit_source=lambda: _partition_units(partitions, "part", ".parquet"),
        writer=_StatsPartitionWriter(
            str(stats_route_root),
            row_groups,
            partitions,
            len(strat_cols),
            normalized_lags,
            max(128, batch_bytes // 512),
        ),
        resources=cfg.resources,
        requested_workers=cfg.analysis.n_jobs,
        mp_start_method=cfg.analysis.mp_start_method,
        force=force,
        memory_guard=guard,
        progress_total_units=partitions,
        progress_phase="rng_lag_reduce",
        enable_worker_progress=True,
    )

    guard.check_before_schedule(force=True)
    if scope is not None:
        scope.update(
            phase="rng_output_construction",
            state="publishing",
            progress={
                "stats_partitions": stats.required_units,
                "lags": len(normalized_lags),
            },
        )
    capacity = _finalize_outputs(
        cfg,
        data_file=data_file,
        output=out_file,
        summary=summary_file,
        stats_root=stats_root,
        stats_result=stats,
        selection_report=selection_report,
        strat_cols=strat_cols,
        lags=normalized_lags,
        partition_count=partitions,
        minimum_observations=minimum_observations,
        peak_rss_mb=guard.peak_rss_bytes / (1024 * 1024),
    )
    status = "blocked_by_cap" if capacity.deterministically_capped_group_count else "success"
    guard.check_before_schedule(force=True)
    write_stage_done(
        stamp_path,
        inputs=[data_file],
        outputs=[out_file, summary_file],
        cfg=cfg,
        stage="rng_diagnostics",
        status=status,
        sidecar_artifacts=[out_file, summary_file],
    )
    if recorder is not None and scope is not None:
        recorder.record_completion_summary(
            f"{scope.scope}:rng_diagnostics",
            stage="rng_diagnostics",
            summary={
                "row_groups": row_groups,
                "partitions": partitions,
                "lags": len(normalized_lags),
                "candidate_groups": capacity.total_candidate_group_count,
                "eligible_groups": capacity.eligible_group_count,
                "selected_groups": capacity.selected_group_count,
                "tracked_matchup_groups": capacity.tracked_matchup_group_count,
                "skipped_matchup_groups": capacity.skipped_matchup_group_count,
                "skipped_matchup_rows": capacity.skipped_matchup_row_count,
                "completion_status": capacity.completeness_status,
                "reconciled_from": "authenticated_rng_outputs_and_partition_manifests",
            },
        )
    LOGGER.info(
        "rng-diagnostics: published",
        extra={
            "stage": "rng_diagnostics",
            "completion_status": capacity.completeness_status,
            "candidate_groups": capacity.total_candidate_group_count,
            "selected_groups": capacity.selected_group_count,
            "peak_sampled_rss_mb": guard.peak_rss_bytes / (1024 * 1024),
        },
    )


def _partition_identity(
    stage_name: str,
    root_seed: int,
    inputs: tuple[tuple[str, str], ...],
    config_sha: str,
    code_sha: str,
) -> PartitionedStageIdentity:
    return PartitionedStageIdentity(
        stage_name=stage_name,
        root_seed=root_seed,
        input_identities=tuple(sorted(inputs)),
        statistical_config_sha256=config_sha,
        code_identity_sha256=code_sha,
        schema_version=_PARTITION_SCHEMA_VERSION,
        method_version=_DIAGNOSTIC_METHOD_VERSION,
    )


def _diagnostic_code_sha256(cfg: AppConfig) -> str:
    return resolved_code_identity_sha256(cfg)


def _row_group_units(count: int) -> Iterator[PartitionedUnit]:
    for index in range(count):
        yield PartitionedUnit((index,), f"row-group-{index:05d}.arrow")


def _partition_units(count: int, prefix: str, suffix: str) -> Iterator[PartitionedUnit]:
    for index in range(count):
        yield PartitionedUnit((index,), f"{prefix}-{index:03d}{suffix}")


def _normalize_lags(lags: Sequence[int] | None) -> tuple[int, ...]:
    if lags is None:
        return (1,)
    return tuple(sorted({int(lag) for lag in lags if int(lag) > 0}))


def _effective_max_matchup_groups(configured_cap: int | None) -> int | None:
    if configured_cap is None:
        return _DEFAULT_MAX_MATCHUP_GROUPS
    return int(configured_cap) if configured_cap > 0 else None


def _winner_column(names: set[str]) -> str | None:
    if "winner_strategy" in names:
        return "winner_strategy"
    if "winner_seat" in names:
        return "winner_seat"
    return None


def _seat_strategy_columns(cfg: AppConfig, schema_names: Sequence[str]) -> list[str]:
    del cfg
    columns: list[tuple[int, str]] = []
    for name in schema_names:
        if name.startswith("P") and name.endswith("_strategy"):
            middle = name[1:-9]
            if middle.isdigit():
                columns.append((int(middle), name))
    return [name for _seat, name in sorted(columns)]


def _iter_row_group_batches(
    path: Path,
    row_group: int,
    *,
    columns: Sequence[str],
    batch_bytes: int,
    expansion: int,
) -> Generator[pa.RecordBatch, None, None]:
    """Yield projected batches whose Arrow bytes are bounded before expansion."""

    if batch_bytes < 1:
        raise ValueError("RNG diagnostic batch budget must be positive")
    projected_width = 8 * 4 + 8 + 4 * max(1, len(columns) - 5)
    batch_rows = max(1, min(65_536, batch_bytes // max(1, projected_width * expansion)))
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(
            batch_size=batch_rows,
            row_groups=[row_group],
            columns=list(columns),
            use_threads=False,
        ):
            if batch.nbytes <= batch_bytes:
                yield batch
                continue
            rows_per_slice = max(1, int(batch.num_rows * batch_bytes / batch.nbytes))
            for offset in range(0, batch.num_rows, rows_per_slice):
                piece = batch.slice(offset, rows_per_slice)
                if piece.nbytes > batch_bytes and piece.num_rows > 1:
                    raise ResourceSafetyError("projected Arrow slice exceeded byte budget")
                yield piece


def _parquet_num_row_groups(path: Path) -> int:
    with pq.ParquetFile(path) as parquet:
        return parquet.num_row_groups


def _column_numpy(batch: pa.RecordBatch, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    array = batch.column(batch.schema.get_field_index(name))
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=dtype)


def _extract_batch_arrays(
    batch: pa.RecordBatch,
    *,
    winner_col: str,
    strat_cols: Sequence[str],
    expected_root_seed: int,
) -> _BatchArrays:
    versions = _column_numpy(batch, "rng_scheme_version", np.dtype(np.int64))
    namespaces = _column_numpy(batch, "rng_purpose_namespace", np.dtype(np.int64))
    if np.any(versions != RNG_SCHEME_VERSION):
        raise ValueError(f"rng_diagnostics requires RNG scheme version {RNG_SCHEME_VERSION}")
    if np.any(namespaces != int(RandomPurpose.TOURNAMENT_GAME)):
        raise ValueError("rng_diagnostics requires tournament-game RNG coordinates")
    root = _column_numpy(batch, "root_seed", np.dtype(np.int64))
    if np.any(root != expected_root_seed):
        raise ValueError("rng_diagnostics input root_seed differs from the active root")
    k = _column_numpy(batch, "k", np.dtype(np.int16))
    shuffle = _column_numpy(batch, "shuffle_index", np.dtype(np.int64))
    game = _column_numpy(batch, "game_index", np.dtype(np.int64))
    rounds = _column_numpy(batch, "n_rounds", np.dtype(np.float64))
    if not np.all(np.isfinite(rounds)):
        raise ValueError("rng_diagnostics n_rounds cannot be null or non-finite")
    seats = np.full((batch.num_rows, len(strat_cols)), _MISSING_STRATEGY, dtype=np.int32)
    for index, column in enumerate(strat_cols):
        arrow = batch.column(batch.schema.get_field_index(column))
        casted = cast(pa.Array, pc.cast(arrow, pa.int32()))
        filled = cast(
            pa.Array,
            pc.fill_null(casted, pa.scalar(_MISSING_STRATEGY, type=pa.int32())),
        )
        seats[:, index] = np.asarray(
            filled.to_numpy(zero_copy_only=False),
            dtype=np.int32,
        )
    actual_k = np.count_nonzero(seats >= 0, axis=1).astype(np.int16)
    if np.any(actual_k != k):
        raise ValueError("rng_diagnostics k does not match non-null seat strategies")
    if winner_col == "winner_strategy":
        winner_arrow = batch.column(batch.schema.get_field_index(winner_col))
        winner_casted = cast(pa.Array, pc.cast(winner_arrow, pa.int32()))
        winner_filled = cast(
            pa.Array,
            pc.fill_null(
                winner_casted,
                pa.scalar(_MISSING_STRATEGY, type=pa.int32()),
            ),
        )
        winner = np.asarray(
            winner_filled.to_numpy(zero_copy_only=False),
            dtype=np.int32,
        )
    else:
        winner = _winner_from_seat_strings(batch, winner_col, seats)
    sorted_seats = np.where(seats < 0, np.iinfo(np.int32).max, seats)
    sorted_seats.sort(axis=1)
    sorted_seats[sorted_seats == np.iinfo(np.int32).max] = _MISSING_STRATEGY
    matchup_id = _matchup_ids(k, sorted_seats)
    return _BatchArrays(root, k, shuffle, game, rounds, winner, seats, sorted_seats, matchup_id)


def _winner_from_seat_strings(
    batch: pa.RecordBatch, winner_col: str, seats: np.ndarray
) -> np.ndarray:
    winner = np.full(batch.num_rows, _MISSING_STRATEGY, dtype=np.int32)
    values = batch.column(batch.schema.get_field_index(winner_col))
    extracted = cast(
        pa.StructArray,
        pc.extract_regex(pc.cast(values, pa.string()), r"^P(?P<seat>\d+)$"),
    )
    seat_casted = cast(pa.Array, pc.cast(extracted.field("seat"), pa.int32()))
    seat_filled = cast(pa.Array, pc.fill_null(seat_casted, pa.scalar(0, type=pa.int32())))
    seat_numbers = np.asarray(
        seat_filled.to_numpy(zero_copy_only=False),
        dtype=np.int32,
    )
    valid_rows = np.flatnonzero((seat_numbers >= 1) & (seat_numbers <= seats.shape[1]))
    if valid_rows.size:
        winner[valid_rows] = seats[valid_rows, seat_numbers[valid_rows] - 1]
    return winner


def _matchup_ids(k: np.ndarray, sorted_seats: np.ndarray) -> np.ndarray:
    composite = np.column_stack((k.astype(np.int32), sorted_seats))
    unique, inverse = np.unique(composite, axis=0, return_inverse=True)
    ids = np.empty(unique.shape[0], dtype=np.uint64)
    for index, row in enumerate(unique):
        digest = hashlib.blake2b(
            np.asarray(row, dtype="<i4").tobytes(),
            digest_size=8,
            person=b"farkle-m",
        ).digest()
        ids[index] = int.from_bytes(digest, "little", signed=False)
    return ids[inverse]


def _count_dtype(max_players: int) -> np.dtype[Any]:
    return np.dtype(
        [
            ("group_type", "u1"),
            ("k", "<i2"),
            ("group_id", "<u8"),
            *((f"p{index}", "<i4") for index in range(max_players)),
            ("count", "<u8"),
        ],
        align=False,
    )


def _observation_dtype(max_players: int) -> np.dtype[Any]:
    return np.dtype(
        [
            ("group_type", "u1"),
            ("k", "<i2"),
            ("group_id", "<u8"),
            *((f"p{index}", "<i4") for index in range(max_players)),
            ("root_seed", "<i8"),
            ("shuffle_index", "<i8"),
            ("game_index", "<i8"),
            ("seat_index", "<i2"),
            ("win_indicator", "i1"),
            ("n_rounds", "<f8"),
        ],
        align=False,
    )


def _key_fields(dtype: np.dtype[Any]) -> tuple[str, ...]:
    return tuple(name for name in dtype.names or () if name != "count")


def _count_key_fields(dtype: np.dtype[Any]) -> tuple[str, ...]:
    return tuple(name for name in dtype.names or () if name != "count")


def _count_records(arrays: _BatchArrays) -> np.ndarray:
    max_players = arrays.seats.shape[1]
    dtype = _count_dtype(max_players)
    active_rows, active_seats = np.nonzero(arrays.seats >= 0)
    total = active_rows.size + arrays.k.size
    records = np.empty(total, dtype=dtype)
    records["group_type"][: active_rows.size] = _GROUP_STRATEGY
    records["k"][: active_rows.size] = arrays.k[active_rows]
    records["group_id"][: active_rows.size] = arrays.seats[active_rows, active_seats].astype(
        np.uint64
    )
    records["group_type"][active_rows.size :] = _GROUP_MATCHUP
    records["k"][active_rows.size :] = arrays.k
    records["group_id"][active_rows.size :] = arrays.matchup_id
    for index in range(max_players):
        records[f"p{index}"][: active_rows.size] = _MISSING_STRATEGY
        records[f"p{index}"][active_rows.size :] = arrays.canonical_matchup[:, index]
    records["count"] = 1
    return _reduce_count_array(records)


def _count_sort_order(records: np.ndarray) -> np.ndarray:
    fields = _count_key_fields(records.dtype)
    return np.lexsort(tuple(records[name] for name in reversed(fields)))


def _reduce_count_array(records: np.ndarray) -> np.ndarray:
    if records.size == 0:
        return records
    ordered = records[_count_sort_order(records)]
    fields = _count_key_fields(ordered.dtype)
    starts = np.zeros(ordered.size, dtype=bool)
    starts[0] = True
    for name in fields:
        starts[1:] |= ordered[name][1:] != ordered[name][:-1]
    indices = np.flatnonzero(starts)
    reduced = ordered[indices].copy()
    reduced["count"] = np.add.reduceat(ordered["count"], indices)
    return reduced


def _splitmix64(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.uint64, copy=True)
    result += np.uint64(0x9E3779B97F4A7C15)
    result = (result ^ (result >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    result = (result ^ (result >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return result ^ (result >> np.uint64(31))


def _stable_partitions(records: np.ndarray, partition_count: int) -> np.ndarray:
    values = records["group_id"] ^ (records["k"].astype(np.uint64) << np.uint64(48))
    values ^= records["group_type"].astype(np.uint64) << np.uint64(63)
    return (_splitmix64(values) % np.uint64(partition_count)).astype(np.int16)


def _priority(records: np.ndarray) -> np.ndarray:
    values = records["group_id"] ^ (records["k"].astype(np.uint64) << np.uint64(48))
    values ^= records["group_type"].astype(np.uint64) << np.uint64(63)
    return _splitmix64(values ^ np.uint64(0xD1B54A32D192ED03))


def _count_arrow_schema(max_players: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("group_type", pa.uint8(), nullable=False),
            pa.field("k", pa.int16(), nullable=False),
            pa.field("group_id", pa.uint64(), nullable=False),
            *(pa.field(f"p{index}", pa.int32(), nullable=False) for index in range(max_players)),
            pa.field("count", pa.uint64(), nullable=False),
        ]
    )


def _count_records_to_batch(records: np.ndarray, schema: pa.Schema) -> pa.RecordBatch:
    arrays = [pa.array(records[name], type=schema.field(name).type) for name in schema.names]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def _count_batch_to_records(batch: pa.RecordBatch, dtype: np.dtype[Any]) -> np.ndarray:
    records = np.empty(batch.num_rows, dtype=dtype)
    for name in dtype.names or ():
        records[name] = np.asarray(batch.column(batch.schema.get_field_index(name)))
    return records


def _record_key(record: np.void, fields: Sequence[str]) -> tuple[int, ...]:
    return tuple(int(record[name]) for name in fields)


def _memmap_records(path: Path, dtype: np.dtype[Any]) -> np.memmap:
    size = path.stat().st_size // dtype.itemsize
    return np.memmap(path, dtype=dtype, mode="r", shape=(size,))


def _close_memmap(array: np.ndarray) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _safe_file_bytes(paths: Sequence[Path]) -> int:
    """Return best-effort operational spill volume without affecting computation."""

    total = 0
    for path in paths:
        try:
            total += int(path.stat().st_size)
        except OSError:
            continue
    return total


def _merge_count_files(inputs: Sequence[Path], output: Path, dtype: np.dtype[Any]) -> None:
    fields = _count_key_fields(dtype)
    arrays: list[np.memmap] = []
    try:
        arrays = [_memmap_records(path, dtype) for path in inputs]
        heap: list[tuple[tuple[int, ...], int, int]] = []
        for source, array in enumerate(arrays):
            if array.size:
                heapq.heappush(heap, (_record_key(array[0], fields), source, 0))
        buffer = np.empty(_RUN_WRITE_ROWS, dtype=dtype)
        used = 0
        with output.open("wb") as handle:
            current_key: tuple[int, ...] | None = None
            current: np.void | None = None
            current_count = 0
            while heap:
                key, source, index = heapq.heappop(heap)
                record = arrays[source][index]
                if current_key is not None and key != current_key:
                    assert current is not None
                    buffer[used] = current
                    buffer["count"][used] = current_count
                    used += 1
                    if used == buffer.size:
                        buffer.tofile(handle)
                        used = 0
                if key != current_key:
                    current_key = key
                    current = record.copy()
                    current_count = 0
                current_count += int(record["count"])
                next_index = index + 1
                if next_index < arrays[source].size:
                    heapq.heappush(
                        heap,
                        (_record_key(arrays[source][next_index], fields), source, next_index),
                    )
            if current_key is not None:
                assert current is not None
                buffer[used] = current
                buffer["count"][used] = current_count
                used += 1
            if used:
                buffer[:used].tofile(handle)
    finally:
        for array in arrays:
            _close_memmap(array)


def _collapse_count_runs(runs: Sequence[Path], temp: Path, dtype: np.dtype[Any]) -> Path:
    if not runs:
        empty = temp / "empty.bin"
        empty.write_bytes(b"")
        return empty
    current = list(runs)
    generation = 0
    while len(current) > 1:
        next_generation: list[Path] = []
        for start in range(0, len(current), _RUN_MERGE_FAN_IN):
            output = temp / f"merge-{generation:03d}-{len(next_generation):05d}.bin"
            _merge_count_files(current[start : start + _RUN_MERGE_FAN_IN], output, dtype)
            next_generation.append(output)
        report_worker_progress(
            "rng_count_merge_pass",
            counters={
                "merge_passes_completed": 1,
                "merge_runs_created": len(next_generation),
                "merge_bytes_written": _safe_file_bytes(next_generation),
            },
        )
        current = next_generation
        generation += 1
    return current[0]


def _eligibility_schema(max_players: int) -> pa.Schema:
    fields: list[pa.Field[Any]] = [
        pa.field("partition", pa.int16(), nullable=False),
        pa.field("group_type", pa.uint8(), nullable=False),
        pa.field("k", pa.int16(), nullable=False),
        pa.field("group_id", pa.uint64(), nullable=False),
    ]
    fields.extend(pa.field(f"p{index}", pa.int32(), nullable=False) for index in range(max_players))
    fields.extend(
        [
            pa.field("observations", pa.uint64(), nullable=False),
            pa.field("priority", pa.uint64(), nullable=False),
            pa.field("eligible", pa.bool_(), nullable=False),
        ]
    )
    return pa.schema(fields)


def _participant_fields(dtype: np.dtype[Any]) -> tuple[str, ...]:
    return tuple(name for name in dtype.names or () if name.startswith("p"))


def _validate_group_records(records: np.ndarray, dtype: np.dtype[Any]) -> None:
    """Validate the complete semantic key carried beside the compact digest."""

    if not records.size:
        return
    participant_fields = _participant_fields(dtype)
    participants = np.column_stack([records[name] for name in participant_fields])
    group_type = records["group_type"]
    strategy = group_type == _GROUP_STRATEGY
    matchup = group_type == _GROUP_MATCHUP
    if np.any(~(strategy | matchup)):
        raise ValueError("invalid RNG diagnostic group type")
    if np.any(participants[strategy] != _MISSING_STRATEGY):
        raise ValueError("strategy RNG diagnostic key contains matchup participants")
    if np.any(records["group_id"][strategy] > np.iinfo(np.int32).max):
        raise ValueError("strategy RNG diagnostic identifier is outside the source domain")
    if not np.any(matchup):
        return
    matchup_k = records["k"][matchup]
    matchup_participants = participants[matchup]
    if np.any((matchup_k < 1) | (matchup_k > len(participant_fields))):
        raise ValueError("matchup RNG diagnostic key has invalid player count")
    positions = np.arange(len(participant_fields))
    active = positions < matchup_k[:, None]
    if np.any(matchup_participants[active] < 0) or np.any(
        matchup_participants[~active] != _MISSING_STRATEGY
    ):
        raise ValueError("matchup RNG diagnostic key has inconsistent participant padding")
    adjacent_active = positions[1:] < matchup_k[:, None]
    if np.any((matchup_participants[:, 1:] < matchup_participants[:, :-1]) & adjacent_active):
        raise ValueError("matchup RNG diagnostic key is not canonically ordered")
    expected_ids = _matchup_ids(matchup_k, matchup_participants)
    if np.any(records["group_id"][matchup] != expected_ids):
        raise ValueError("matchup RNG diagnostic digest does not match its canonical key")


def _write_eligibility_partition(
    merged: Path,
    output: Path,
    *,
    dtype: np.dtype[Any],
    partition: int,
    minimum_observations: int,
    root_seed: int,
    batch_rows: int,
) -> None:
    del root_seed
    schema = _eligibility_schema(len([name for name in dtype.names or () if name.startswith("p")]))
    records = _memmap_records(merged, dtype) if merged.stat().st_size else np.empty(0, dtype=dtype)
    try:
        with pq.ParquetWriter(output, schema, compression="snappy", use_dictionary=True) as writer:
            for start in range(0, records.size, max(1, batch_rows)):
                chunk = np.asarray(records[start : start + batch_rows])
                _validate_group_records(chunk, dtype)
                arrays: list[pa.Array] = [
                    pa.array(np.full(chunk.size, partition, dtype=np.int16)),
                    pa.array(chunk["group_type"]),
                    pa.array(chunk["k"]),
                    pa.array(chunk["group_id"]),
                ]
                arrays.extend(
                    pa.array(chunk[name]) for name in dtype.names or () if name.startswith("p")
                )
                arrays.extend(
                    [
                        pa.array(chunk["count"]),
                        pa.array(_priority(chunk)),
                        pa.array(chunk["count"] >= minimum_observations),
                    ]
                )
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
    finally:
        _close_memmap(records)


def _selection_schema(max_players: int) -> pa.Schema:
    return _eligibility_schema(max_players)


def _priority_key_dtype(max_players: int) -> np.dtype[Any]:
    return np.dtype(
        [
            ("priority", "<u8"),
            ("group_type", "u1"),
            ("k", "<i2"),
            ("group_id", "<u8"),
            *((f"p{index}", "<i4") for index in range(max_players)),
        ]
    )


def _priority_sort_order(records: np.ndarray) -> np.ndarray:
    names = records.dtype.names or ()
    return np.lexsort(tuple(records[name] for name in reversed(names)))


def _priority_tuple(record: np.void) -> tuple[int, ...]:
    return tuple(int(record[name]) for name in record.dtype.names or ())


def _observation_histogram_bin(counts: np.ndarray, minimum: int) -> np.ndarray:
    result = np.zeros(counts.size, dtype=np.int16)
    below = counts < minimum
    result[below] = np.minimum(counts[below], np.iinfo(np.int16).max).astype(np.int16)
    if np.any(~below):
        result[~below] = minimum + np.floor(np.log2(counts[~below] - minimum + 1)).astype(np.int16)
    return result


def _write_or_reuse_selection(
    cfg: AppConfig,
    *,
    eligibility_root: Path,
    eligibility_manifest: Path,
    eligibility_manifest_sha: str,
    partition_count: int,
    max_players: int,
    minimum_observations: int,
    lags: tuple[int, ...],
    selection_file: Path,
    report_file: Path,
    stage_config_sha: str,
    source: Path,
    force: bool,
    guard: ProcessTreeMemoryGuard,
) -> dict[str, Any]:
    done = selection_file.with_name("rng_group_selection.done.json")
    selection_sha = hashlib.sha256(
        f"{stage_config_sha}:{eligibility_manifest_sha}".encode("ascii")
    ).hexdigest()
    if not force:
        state = resolve_stage_state(
            done,
            [eligibility_manifest],
            [selection_file, report_file],
            stage="rng_diagnostics_selection",
            stage_config_sha=selection_sha,
            cache_key_version=1,
            sidecar_artifacts=[selection_file, report_file],
        )
        if state is CompletionState.COMPLETE_VALID:
            return cast(dict[str, Any], json.loads(report_file.read_text(encoding="utf-8")))

    cap = _effective_max_matchup_groups(cfg.analysis.rng_max_matchup_groups)
    top_dtype = _priority_key_dtype(max_players)
    top = np.empty(0, dtype=top_dtype)
    totals = {"strategy": 0, "matchup": 0}
    eligible_totals = {"strategy": 0, "matchup": 0}
    histogram: dict[tuple[str, int], int] = {}
    for partition in range(partition_count):
        path = eligibility_root / "units" / f"part-{partition:03d}.parquet"
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=32_768, use_threads=False):
                guard.check_before_schedule()
                group_type = _column_numpy(batch, "group_type", np.dtype(np.uint8))
                observations = _column_numpy(batch, "observations", np.dtype(np.uint64))
                eligible = _column_numpy(batch, "eligible", np.dtype(bool))
                for value, label in (
                    (_GROUP_STRATEGY, "strategy"),
                    (_GROUP_MATCHUP, "matchup"),
                ):
                    mask = group_type == value
                    totals[label] += int(np.count_nonzero(mask))
                    eligible_totals[label] += int(np.count_nonzero(mask & eligible))
                    bins = _observation_histogram_bin(observations[mask], minimum_observations)
                    unique_bins, bin_counts = np.unique(bins, return_counts=True)
                    for bin_value, bin_count in zip(unique_bins, bin_counts, strict=True):
                        histogram[(label, int(bin_value))] = histogram.get(
                            (label, int(bin_value)), 0
                        ) + int(bin_count)
                if cap is not None:
                    mask = (group_type == _GROUP_MATCHUP) & eligible
                    if np.any(mask):
                        candidates = np.empty(int(np.count_nonzero(mask)), dtype=top_dtype)
                        for name in top_dtype.names or ():
                            candidates[name] = _column_numpy(
                                batch, name, top_dtype.fields[name][0]  # type: ignore[index]
                            )[mask]
                        top = np.concatenate((top, candidates))
                        if top.size > cap:
                            top = top[_priority_sort_order(top)[:cap]].copy()
    cutoff: tuple[int, ...] | None = None
    capped = 0
    if cap is not None and eligible_totals["matchup"] > cap:
        top = top[_priority_sort_order(top)]
        cutoff = _priority_tuple(top[-1])
        capped = eligible_totals["matchup"] - cap

    distribution = [
        {
            "summary_level": label,
            "bin": _histogram_label(code, minimum_observations),
            "groups": count,
        }
        for (label, code), count in sorted(histogram.items())
    ]
    report: dict[str, Any] = {
        "selection_schema_version": 1,
        "method_version": _DIAGNOSTIC_METHOD_VERSION,
        "partition_count": partition_count,
        "minimum_usable_observations": minimum_observations,
        "normalized_lags": list(lags),
        "effective_matchup_group_cap": cap,
        "total_candidate_groups": totals["strategy"] + totals["matchup"],
        "candidate_strategy_groups": totals["strategy"],
        "candidate_matchup_groups": totals["matchup"],
        "eligible_groups": eligible_totals["strategy"] + eligible_totals["matchup"],
        "eligible_strategy_groups": eligible_totals["strategy"],
        "eligible_matchup_groups": eligible_totals["matchup"],
        "selected_strategy_groups": eligible_totals["strategy"],
        "selected_matchup_groups": eligible_totals["matchup"] - capped,
        "below_minimum_observation_groups": (
            totals["strategy"]
            + totals["matchup"]
            - eligible_totals["strategy"]
            - eligible_totals["matchup"]
        ),
        "deterministically_capped_groups": capped,
        "exclusion_reasons": {
            "below_minimum_usable_observations": (
                totals["strategy"]
                + totals["matchup"]
                - eligible_totals["strategy"]
                - eligible_totals["matchup"]
            ),
            "deterministic_priority_cap": capped,
        },
        "observation_count_distribution": distribution,
        "priority_cutoff": list(cutoff) if cutoff is not None else None,
        "completeness_status": "blocked_by_cap" if capped else "planned_complete",
    }
    report["selected_groups"] = int(report["selected_strategy_groups"]) + int(
        report["selected_matchup_groups"]
    )

    selection_schema = _selection_schema(max_players)
    selection_parameters = {
        "method_version": _DIAGNOSTIC_METHOD_VERSION,
        "partition_count": partition_count,
        "minimum_usable_observations": minimum_observations,
        "stable_priority": "splitmix64_compact_digest_then_full_canonical_key",
        "effective_matchup_group_cap": cap,
    }
    sidecar = make_artifact_sidecar(
        cfg,
        selection_file,
        producer="rng_diagnostics",
        scope=ArtifactScope.DIAGNOSTICS,
        source_scope=ArtifactScope.CONCAT_KS,
        operation="rng_group_selection",
        support_count_role="candidate_group_observations",
        conditioning="structural_lag_eligibility_then_deterministic_priority_cap",
        method_contract=cast(
            MethodContract,
            {
                "kind": "operation",
                "procedure": "rng_group_selection",
                "parameters": selection_parameters,
            },
        ),
        consistency_columns=selection_schema.names,
        source_artifacts=[source],
        input_manifests=[eligibility_manifest],
        grouping_keys=[
            "group_type",
            "k",
            "group_id",
            *(f"p{index}" for index in range(max_players)),
        ],
        player_counts=cfg.sim.n_players_list,
        required_player_counts=cfg.sim.n_players_list,
    )
    with ParquetShardWriter(
        str(selection_file),
        schema=selection_schema,
        compression=cfg.parquet_codec,
        row_group_size=32_768,
        sidecar=sidecar,
    ) as writer:
        for partition in range(partition_count):
            path = eligibility_root / "units" / f"part-{partition:03d}.parquet"
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(batch_size=32_768, use_threads=False):
                    eligible = _column_numpy(batch, "eligible", np.dtype(bool))
                    group_type = _column_numpy(batch, "group_type", np.dtype(np.uint8))
                    keep = eligible.copy()
                    if cutoff is not None:
                        candidates = np.empty(batch.num_rows, dtype=top_dtype)
                        for name in top_dtype.names or ():
                            candidates[name] = _column_numpy(
                                batch, name, top_dtype.fields[name][0]  # type: ignore[index]
                            )
                        keep &= (group_type == _GROUP_STRATEGY) | _priority_at_or_below(
                            candidates, cutoff
                        )
                    if np.any(keep):
                        writer.write_batch(pa.Table.from_batches([batch.filter(pa.array(keep))]))
        if writer.rows_written == 0:
            writer.write_batch(pa.Table.from_batches([], schema=selection_schema))

    report_sidecar = make_artifact_sidecar(
        cfg,
        report_file,
        producer="rng_diagnostics",
        scope=ArtifactScope.DIAGNOSTICS,
        source_scope=ArtifactScope.CONCAT_KS,
        operation="rng_group_selection",
        support_count_role="candidate_group_observations",
        conditioning="structural_lag_eligibility_then_deterministic_priority_cap",
        method_contract=cast(
            MethodContract,
            {
                "kind": "operation",
                "procedure": "rng_group_selection",
                "parameters": selection_parameters,
            },
        ),
        source_artifacts=[source],
        input_manifests=[eligibility_manifest],
        player_counts=cfg.sim.n_players_list,
        required_player_counts=cfg.sim.n_players_list,
    )
    write_json_artifact_atomic(report, report_file, sidecar=report_sidecar)
    write_stage_done(
        done,
        inputs=[eligibility_manifest],
        outputs=[selection_file, report_file],
        stage="rng_diagnostics_selection",
        stage_config_sha=selection_sha,
        cache_key_version=1,
        sidecar_artifacts=[selection_file, report_file],
    )
    return report


def _histogram_label(code: int, minimum: int) -> str:
    if code < minimum:
        return str(code)
    power = code - minimum
    lower = minimum + (1 << power) - 1
    upper = minimum + (1 << (power + 1)) - 2
    return f"{lower}-{upper}"


def _priority_at_or_below(records: np.ndarray, cutoff: tuple[int, ...]) -> np.ndarray:
    """Vectorized lexicographic comparison against one semantic cutoff."""

    less = np.zeros(records.size, dtype=bool)
    equal = np.ones(records.size, dtype=bool)
    for name, value in zip(records.dtype.names or (), cutoff, strict=True):
        less |= equal & (records[name] < value)
        equal &= records[name] == value
    return less | equal


def _selection_key_dtype(max_players: int) -> np.dtype[Any]:
    return np.dtype(
        [
            ("group_type", "u1"),
            ("k", "<i2"),
            ("group_id", "<u8"),
            *((f"p{index}", "<i4") for index in range(max_players)),
        ]
    )


def _load_selection_memberships(
    path: Path, *, partition_count: int, max_players: int, max_bytes: int
) -> tuple[np.ndarray, ...]:
    dtype = _selection_key_dtype(max_players)
    chunks: list[list[np.ndarray]] = [[] for _ in range(partition_count)]
    total_bytes = 0
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(
            batch_size=32_768,
            columns=["partition", *(dtype.names or ())],
            use_threads=False,
        ):
            partition = _column_numpy(batch, "partition", np.dtype(np.int16))
            records = np.empty(batch.num_rows, dtype=dtype)
            for name in dtype.names or ():
                records[name] = _column_numpy(
                    batch, name, dtype.fields[name][0]  # type: ignore[index]
                )
            total_bytes += records.nbytes
            if total_bytes > max_bytes:
                raise ResourceSafetyError(
                    f"selected RNG group membership exceeds worker budget: {total_bytes} > {max_bytes}"
                )
            for value in np.unique(partition):
                chunks[int(value)].append(records[partition == value].copy())
    memberships: list[np.ndarray] = []
    for partition_chunks in chunks:
        combined = (
            np.concatenate(partition_chunks) if partition_chunks else np.empty(0, dtype=dtype)
        )
        if combined.size:
            combined.sort(order=list(dtype.names or ()), kind="stable")
        memberships.append(combined)
    return tuple(memberships)


def _observation_records(arrays: _BatchArrays) -> np.ndarray:
    max_players = arrays.seats.shape[1]
    dtype = _observation_dtype(max_players)
    active_rows, active_seats = np.nonzero(arrays.seats >= 0)
    strategy_count = active_rows.size
    total = strategy_count + arrays.k.size
    records = np.empty(total, dtype=dtype)
    records["group_type"][:strategy_count] = _GROUP_STRATEGY
    records["k"][:strategy_count] = arrays.k[active_rows]
    records["group_id"][:strategy_count] = arrays.seats[active_rows, active_seats].astype(np.uint64)
    records["group_type"][strategy_count:] = _GROUP_MATCHUP
    records["k"][strategy_count:] = arrays.k
    records["group_id"][strategy_count:] = arrays.matchup_id
    for index in range(max_players):
        records[f"p{index}"][:strategy_count] = _MISSING_STRATEGY
        records[f"p{index}"][strategy_count:] = arrays.canonical_matchup[:, index]
    records["root_seed"][:strategy_count] = arrays.root_seed[active_rows]
    records["root_seed"][strategy_count:] = arrays.root_seed
    records["shuffle_index"][:strategy_count] = arrays.shuffle_index[active_rows]
    records["shuffle_index"][strategy_count:] = arrays.shuffle_index
    records["game_index"][:strategy_count] = arrays.game_index[active_rows]
    records["game_index"][strategy_count:] = arrays.game_index
    records["seat_index"][:strategy_count] = active_seats.astype(np.int16)
    records["seat_index"][strategy_count:] = -1
    records["win_indicator"][:strategy_count] = (
        arrays.seats[active_rows, active_seats] == arrays.winner_strategy[active_rows]
    ).astype(np.int8)
    records["win_indicator"][strategy_count:] = -1
    records["n_rounds"][:strategy_count] = arrays.n_rounds[active_rows]
    records["n_rounds"][strategy_count:] = arrays.n_rounds
    return records


def _membership_mask(records: np.ndarray, membership: np.ndarray) -> np.ndarray:
    if records.size == 0 or membership.size == 0:
        return np.zeros(records.size, dtype=bool)
    dtype = membership.dtype
    keys = np.empty(records.size, dtype=dtype)
    for name in dtype.names or ():
        keys[name] = records[name]
    positions = np.searchsorted(membership, keys)
    valid = positions < membership.size
    result = np.zeros(records.size, dtype=bool)
    result[valid] = membership[positions[valid]] == keys[valid]
    return result


def _observation_arrow_schema(max_players: int) -> pa.Schema:
    dtype = _observation_dtype(max_players)
    type_map = {
        "group_type": pa.uint8(),
        "k": pa.int16(),
        "group_id": pa.uint64(),
        "root_seed": pa.int64(),
        "shuffle_index": pa.int64(),
        "game_index": pa.int64(),
        "seat_index": pa.int16(),
        "win_indicator": pa.int8(),
        "n_rounds": pa.float64(),
    }
    for name in dtype.names or ():
        if name.startswith("p"):
            type_map[name] = pa.int32()
    return pa.schema([pa.field(name, type_map[name], nullable=False) for name in dtype.names or ()])


def _observation_records_to_batch(records: np.ndarray, schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays(
        [pa.array(records[name], type=schema.field(name).type) for name in schema.names],
        schema=schema,
    )


def _observation_batch_to_records(batch: pa.RecordBatch, dtype: np.dtype[Any]) -> np.ndarray:
    records = np.empty(batch.num_rows, dtype=dtype)
    for name in dtype.names or ():
        records[name] = np.asarray(batch.column(batch.schema.get_field_index(name)))
    return records


def _observation_sort_fields(dtype: np.dtype[Any]) -> tuple[str, ...]:
    participant_fields = tuple(name for name in dtype.names or () if name.startswith("p"))
    return (
        "group_type",
        "k",
        "group_id",
        *participant_fields,
        "root_seed",
        "shuffle_index",
        "game_index",
        "seat_index",
    )


def _observation_sort_order(records: np.ndarray) -> np.ndarray:
    fields = _observation_sort_fields(records.dtype)
    return np.lexsort(tuple(records[name] for name in reversed(fields)))


def _merge_observation_files(inputs: Sequence[Path], output: Path, dtype: np.dtype[Any]) -> None:
    fields = _observation_sort_fields(dtype)
    arrays: list[np.memmap] = []
    try:
        arrays = [_memmap_records(path, dtype) for path in inputs]
        heap: list[tuple[tuple[int, ...], int, int]] = []
        for source, array in enumerate(arrays):
            if array.size:
                heapq.heappush(heap, (_record_key(array[0], fields), source, 0))
        buffer = np.empty(_RUN_WRITE_ROWS, dtype=dtype)
        used = 0
        previous: tuple[int, ...] | None = None
        with output.open("wb") as handle:
            while heap:
                key, source, index = heapq.heappop(heap)
                if key == previous:
                    raise ValueError("duplicate RNG diagnostic semantic observation coordinate")
                previous = key
                buffer[used] = arrays[source][index]
                used += 1
                if used == buffer.size:
                    buffer.tofile(handle)
                    used = 0
                next_index = index + 1
                if next_index < arrays[source].size:
                    heapq.heappush(
                        heap,
                        (_record_key(arrays[source][next_index], fields), source, next_index),
                    )
            if used:
                buffer[:used].tofile(handle)
    finally:
        for array in arrays:
            _close_memmap(array)


def _collapse_observation_runs(runs: Sequence[Path], temp: Path, dtype: np.dtype[Any]) -> Path:
    if not runs:
        empty = temp / "empty.bin"
        empty.write_bytes(b"")
        return empty
    current = list(runs)
    generation = 0
    while len(current) > 1:
        next_generation: list[Path] = []
        for start in range(0, len(current), _RUN_MERGE_FAN_IN):
            output = temp / f"merge-{generation:03d}-{len(next_generation):05d}.bin"
            _merge_observation_files(current[start : start + _RUN_MERGE_FAN_IN], output, dtype)
            next_generation.append(output)
        report_worker_progress(
            "rng_stats_merge_pass",
            counters={
                "merge_passes_completed": 1,
                "merge_runs_created": len(next_generation),
                "merge_bytes_written": _safe_file_bytes(next_generation),
            },
        )
        current = next_generation
        generation += 1
    return current[0]


@dataclass(slots=True)
class _OnlineMetric:
    lags: tuple[int, ...]
    ring: np.ndarray = field(init=False)
    n_obs: int = 0
    pair_count: np.ndarray = field(init=False)
    sum_x: np.ndarray = field(init=False)
    sum_y: np.ndarray = field(init=False)
    sum_x2: np.ndarray = field(init=False)
    sum_y2: np.ndarray = field(init=False)
    sum_xy: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.ring = np.empty(max(self.lags), dtype=np.float64)
        self.pair_count = np.zeros(len(self.lags), dtype=np.int64)
        self.sum_x = np.zeros(len(self.lags), dtype=np.float64)
        self.sum_y = np.zeros(len(self.lags), dtype=np.float64)
        self.sum_x2 = np.zeros(len(self.lags), dtype=np.float64)
        self.sum_y2 = np.zeros(len(self.lags), dtype=np.float64)
        self.sum_xy = np.zeros(len(self.lags), dtype=np.float64)

    def push(self, value: float) -> None:
        max_lag = self.ring.size
        for index, lag in enumerate(self.lags):
            if self.n_obs >= lag:
                earlier = self.ring[(self.n_obs - lag) % max_lag]
                self.pair_count[index] += 1
                self.sum_x[index] += earlier
                self.sum_y[index] += value
                self.sum_x2[index] += earlier * earlier
                self.sum_y2[index] += value * value
                self.sum_xy[index] += earlier * value
        self.ring[self.n_obs % max_lag] = value
        self.n_obs += 1

    def result(self, index: int) -> tuple[float | None, str]:
        pairs = int(self.pair_count[index])
        if pairs < 2:
            return None, "insufficient_pairs"
        count = float(pairs)
        numerator = count * self.sum_xy[index] - self.sum_x[index] * self.sum_y[index]
        den_x = count * self.sum_x2[index] - self.sum_x[index] ** 2
        den_y = count * self.sum_y2[index] - self.sum_y[index] ** 2
        if den_x <= 0.0 or den_y <= 0.0:
            return None, "zero_variance"
        return float(numerator / (den_x * den_y) ** 0.5), "estimated"


def _stats_schema() -> pa.Schema:
    fields: list[pa.Field[Any]] = [
        pa.field("summary_level", pa.string(), nullable=False),
        pa.field("strategy", pa.int32(), nullable=True),
        pa.field("matchup_id", pa.uint64(), nullable=True),
        pa.field("matchup", pa.string(), nullable=True),
        pa.field("participant_strategy_ids", pa.list_(pa.int32()), nullable=True),
        pa.field("n_players", pa.int16(), nullable=False),
        pa.field("observations", pa.int64(), nullable=False),
        pa.field("lagged_pairs", pa.int64(), nullable=False),
        pa.field("lag", pa.int32(), nullable=False),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("autocorr", pa.float64(), nullable=True),
        pa.field("estimability_status", pa.string(), nullable=False),
        pa.field("zero_centered_descriptive_reference_band_lower", pa.float64(), nullable=True),
        pa.field("zero_centered_descriptive_reference_band_upper", pa.float64(), nullable=True),
        pa.field("sequence_order", pa.string(), nullable=False),
        pa.field("note", pa.string(), nullable=False),
    ]
    return pa.schema(fields)


def _group_identity(record: np.void, dtype: np.dtype[Any]) -> tuple[int, ...]:
    return (
        int(record["group_type"]),
        int(record["k"]),
        int(record["group_id"]),
        *(int(record[name]) for name in dtype.names or () if name.startswith("p")),
    )


def _rows_for_online_group(
    identity: tuple[int, ...],
    *,
    lags: tuple[int, ...],
    rounds: _OnlineMetric,
    wins: _OnlineMetric | None,
) -> list[dict[str, Any]]:
    group_type, k, group_id, *raw_participants = identity
    participants = [value for value in raw_participants if value >= 0]
    summary = "strategy" if group_type == _GROUP_STRATEGY else "matchup"
    strategy = int(group_id) if group_type == _GROUP_STRATEGY else None
    matchup_id = int(group_id) if group_type == _GROUP_MATCHUP else None
    matchup = " | ".join(str(value) for value in participants) if participants else None
    metrics = [("n_rounds", rounds)]
    if wins is not None:
        metrics.insert(0, ("win_indicator", wins))
    rows: list[dict[str, Any]] = []
    sequence = (
        ",".join(_SEAT_COORDINATE_COLUMNS)
        if group_type == _GROUP_STRATEGY
        else ",".join(_GAME_COORDINATE_COLUMNS)
    )
    for metric, state in metrics:
        for index, lag in enumerate(lags):
            autocorr, status = state.result(index)
            pairs = int(state.pair_count[index])
            half_width = 1.96 / pairs**0.5 if pairs > 0 else None
            rows.append(
                {
                    "summary_level": summary,
                    "strategy": strategy,
                    "matchup_id": matchup_id,
                    "matchup": matchup,
                    "participant_strategy_ids": participants if participants else None,
                    "n_players": k,
                    "observations": state.n_obs,
                    "lagged_pairs": pairs,
                    "lag": lag,
                    "metric": metric,
                    "autocorr": autocorr,
                    "estimability_status": status,
                    "zero_centered_descriptive_reference_band_lower": (
                        -half_width if half_width is not None else None
                    ),
                    "zero_centered_descriptive_reference_band_upper": half_width,
                    "sequence_order": sequence,
                    "note": _EXPECTED_NOTE,
                }
            )
    return rows


def _write_stats_partition(
    merged: Path,
    output: Path,
    *,
    dtype: np.dtype[Any],
    lags: tuple[int, ...],
    batch_rows: int,
) -> None:
    schema = _stats_schema()
    records = _memmap_records(merged, dtype) if merged.stat().st_size else np.empty(0, dtype=dtype)
    try:
        buffered: list[dict[str, Any]] = []
        current_identity: tuple[int, ...] | None = None
        rounds: _OnlineMetric | None = None
        wins: _OnlineMetric | None = None
        with pq.ParquetWriter(output, schema, compression="snappy", use_dictionary=True) as writer:
            for record in records:
                identity = _group_identity(record, dtype)
                if identity != current_identity:
                    if current_identity is not None:
                        assert rounds is not None
                        buffered.extend(
                            _rows_for_online_group(
                                current_identity, lags=lags, rounds=rounds, wins=wins
                            )
                        )
                        if len(buffered) >= batch_rows:
                            writer.write_table(pa.Table.from_pylist(buffered, schema=schema))
                            buffered.clear()
                    current_identity = identity
                    rounds = _OnlineMetric(lags)
                    wins = _OnlineMetric(lags) if identity[0] == _GROUP_STRATEGY else None
                assert rounds is not None
                rounds.push(float(record["n_rounds"]))
                if wins is not None:
                    wins.push(float(record["win_indicator"]))
            if current_identity is not None:
                assert rounds is not None
                buffered.extend(
                    _rows_for_online_group(current_identity, lags=lags, rounds=rounds, wins=wins)
                )
            if buffered:
                writer.write_table(pa.Table.from_pylist(buffered, schema=schema))
    finally:
        _close_memmap(records)


def _finalize_outputs(
    cfg: AppConfig,
    *,
    data_file: Path,
    output: Path,
    summary: Path,
    stats_root: Path,
    stats_result: PartitionedStageResult,
    selection_report: dict[str, Any],
    strat_cols: Sequence[str],
    lags: tuple[int, ...],
    partition_count: int,
    minimum_observations: int,
    peak_rss_mb: float,
) -> RNGDiagnosticCapacityMetadata:
    usable: dict[tuple[str, str, int, str], int] = {}
    result_rows = 0
    for partition in range(partition_count):
        path = stats_root / "units" / f"part-{partition:03d}.parquet"
        with pq.ParquetFile(path) as parquet:
            result_rows += parquet.metadata.num_rows
            for batch in parquet.iter_batches(
                batch_size=32_768,
                columns=["summary_level", "metric", "lag", "estimability_status"],
                use_threads=False,
            ):
                table = pa.Table.from_batches([batch])
                for summary_level in ("strategy", "matchup"):
                    for metric in ("win_indicator", "n_rounds"):
                        for lag in lags:
                            for status in (
                                "estimated",
                                "insufficient_pairs",
                                "zero_variance",
                            ):
                                mask = pc.and_(
                                    pc.and_(
                                        pc.equal(table["summary_level"], pa.scalar(summary_level)),
                                        pc.equal(table["metric"], pa.scalar(metric)),
                                    ),
                                    pc.and_(
                                        pc.equal(table["lag"], pa.scalar(lag)),
                                        pc.equal(table["estimability_status"], pa.scalar(status)),
                                    ),
                                )
                                count = int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)
                                if count:
                                    usable[(summary_level, metric, lag, status)] = (
                                        usable.get((summary_level, metric, lag, status), 0) + count
                                    )
    status_rows: tuple[dict[str, int | str], ...] = tuple(
        {
            "summary_level": key[0],
            "metric": key[1],
            "lag": key[2],
            "estimability_status": key[3],
            "groups": value,
        }
        for key, value in sorted(usable.items())
    )
    usable_rows: tuple[dict[str, int | str], ...] = tuple(
        {
            "summary_level": key[0],
            "metric": key[1],
            "lag": key[2],
            "groups": value,
        }
        for key, value in sorted(usable.items())
        if key[3] == "estimated"
    )
    capped = int(selection_report["deterministically_capped_groups"])
    selected_groups = int(selection_report["selected_groups"])
    completeness_status = (
        "blocked_by_cap" if capped else ("not_estimable" if selected_groups == 0 else "complete")
    )
    capacity = RNGDiagnosticCapacityMetadata(
        effective_matchup_group_cap=cast(
            int | None, selection_report["effective_matchup_group_cap"]
        ),
        normalized_lags=lags,
        partition_count=partition_count,
        minimum_usable_observations=minimum_observations,
        total_candidate_group_count=int(selection_report["total_candidate_groups"]),
        candidate_strategy_group_count=int(selection_report["candidate_strategy_groups"]),
        candidate_matchup_group_count=int(selection_report["candidate_matchup_groups"]),
        eligible_group_count=int(selection_report["eligible_groups"]),
        eligible_strategy_group_count=int(selection_report["eligible_strategy_groups"]),
        eligible_matchup_group_count=int(selection_report["eligible_matchup_groups"]),
        selected_group_count=selected_groups,
        selected_strategy_group_count=int(selection_report["selected_strategy_groups"]),
        selected_matchup_group_count=int(selection_report["selected_matchup_groups"]),
        below_minimum_group_count=int(selection_report["below_minimum_observation_groups"]),
        deterministically_capped_group_count=capped,
        observation_count_distribution=tuple(selection_report["observation_count_distribution"]),
        usable_groups_per_lag=usable_rows,
        completeness_status=completeness_status,
    )
    parameters = _rng_method_parameters(capacity, strat_cols=strat_cols)
    sidecar = make_artifact_sidecar(
        cfg,
        output,
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
                "parameters": parameters,
            },
        ),
        consistency_columns=_stats_schema().names,
        source_artifacts=[data_file],
        grouping_keys=[
            "summary_level",
            "strategy",
            "matchup_id",
            "participant_strategy_ids",
            "n_players",
            "lag",
            "metric",
        ],
        player_counts=cfg.sim.n_players_list,
        required_player_counts=cfg.sim.n_players_list,
        input_manifests=[stats_result.manifest_path],
    )
    with ParquetShardWriter(
        str(output),
        schema=_stats_schema(),
        compression=cfg.parquet_codec,
        row_group_size=32_768,
        sidecar=sidecar,
    ) as writer:
        for partition in range(partition_count):
            path = stats_root / "units" / f"part-{partition:03d}.parquet"
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(batch_size=32_768, use_threads=False):
                    writer.write_batch(pa.Table.from_batches([batch]))
        if writer.rows_written == 0:
            writer.write_batch(pa.Table.from_batches([], schema=_stats_schema()))

    summary_payload = {
        "diagnostic_schema_version": 4,
        **asdict(capacity),
        "exclusion_reasons": selection_report["exclusion_reasons"],
        "estimability_status_distribution": status_rows,
        "result_rows": result_rows,
        "planned_partitions": partition_count,
        "validated_partitions": stats_result.required_units,
        "partition_manifest_sha256": stats_result.manifest_sha256,
        "peak_sampled_process_tree_rss_mb": peak_rss_mb,
        "aggregate_memory_hard_limit_mb": cfg.resources.aggregate_memory_hard_limit_mb,
        "scheduler_memory_budget_mb": cfg.resources.scheduler_memory_budget_mb,
        "process_tree_warning_threshold_mb": (cfg.resources.process_tree_warning_threshold_mb),
        "minimum_system_available_memory_mb": (cfg.resources.minimum_system_available_memory_mb),
        "parent_process_memory_mb": cfg.resources.parent_process_memory_mb,
    }
    summary_sidecar = make_artifact_sidecar(
        cfg,
        summary,
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
                "parameters": parameters,
            },
        ),
        source_artifacts=[data_file],
        player_counts=cfg.sim.n_players_list,
        required_player_counts=cfg.sim.n_players_list,
        input_manifests=[stats_result.manifest_path],
    )
    write_json_artifact_atomic(summary_payload, summary, sidecar=summary_sidecar)
    return capacity


def _rng_method_parameters(
    capacity: RNGDiagnosticCapacityMetadata,
    *,
    strat_cols: Sequence[str],
) -> RNGDiagnosticMethodParameters:
    del strat_cols
    return {
        "method_version": _DIAGNOSTIC_METHOD_VERSION,
        "rng_diagnostic_method_version": _DIAGNOSTIC_METHOD_VERSION,
        "rng_scheme_version": RNG_SCHEME_VERSION,
        "purpose_namespace": int(RandomPurpose.TOURNAMENT_PLAYER),
        "global_order_columns": list(_SEAT_COORDINATE_COLUMNS),
        "matchup_order_columns": list(_GAME_COORDINATE_COLUMNS),
        "sequence_definition": _SEQUENCE_DEFINITION,
        "grouping_semantics": {
            "strategy": "seat exposures keyed by strategy_id and k; win_indicator and n_rounds",
            "matchup": "one game per sorted participant-ID multiset and k; n_rounds only",
        },
        "reference_band_method": _BAND_METHOD,
        "claim": "descriptive_only_no_independence_claim",
        "effective_matchup_group_cap": capacity.effective_matchup_group_cap,
        "normalized_lags": list(capacity.normalized_lags),
        "partition_count": capacity.partition_count,
        "minimum_usable_observations": capacity.minimum_usable_observations,
        "total_candidate_group_count": capacity.total_candidate_group_count,
        "eligible_group_count": capacity.eligible_group_count,
        "selected_group_count": capacity.selected_group_count,
        "below_minimum_group_count": capacity.below_minimum_group_count,
        "deterministically_capped_group_count": capacity.deterministically_capped_group_count,
        "completeness_status": capacity.completeness_status,
        "tracked_matchup_group_count": capacity.tracked_matchup_group_count,
        "skipped_matchup_group_count": capacity.skipped_matchup_group_count,
        "skipped_matchup_row_count": capacity.skipped_matchup_row_count,
    }


def _rng_stage_config_sha(cfg: AppConfig, lags: Sequence[int]) -> str:
    payload = {
        "base_stage_config_sha": cfg.stage_config_sha("rng_diagnostics"),
        "diagnostic_method_version": _DIAGNOSTIC_METHOD_VERSION,
        "rng_scheme_version": RNG_SCHEME_VERSION,
        "purpose_namespace": int(RandomPurpose.TOURNAMENT_PLAYER),
        "strategy_sequence_order": list(_SEAT_COORDINATE_COLUMNS),
        "matchup_sequence_order": list(_GAME_COORDINATE_COLUMNS),
        "sequence_definition": _SEQUENCE_DEFINITION,
        "reference_band_method": _BAND_METHOD,
        "effective_matchup_group_cap": _effective_max_matchup_groups(
            cfg.analysis.rng_max_matchup_groups
        ),
        "partition_count": int(cfg.analysis.rng_diagnostic_partitions),
        "normalized_lags": list(_normalize_lags(lags)),
        "grouping_semantics_version": 2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    from farkle.utils.os_memory import supervise_module_if_needed

    direct_cfg = AppConfig()
    exit_code = supervise_module_if_needed(__name__, sys.argv[1:], direct_cfg.resources)
    if exit_code is not None:
        raise SystemExit(exit_code)
    run(direct_cfg)
