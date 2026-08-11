# src/farkle/analysis/ingest.py
"""Ingest raw simulation results into parquet shards for curation.

This entry point streams over experiment outputs, validates schemas, and
writes player-count-specific shards that feed the downstream combine and
metrics stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from farkle.config import AppConfig, ArtifactScope, load_app_config
from farkle.simulation.runner import simulation_is_complete
from farkle.utils.arrow_batches import iter_parquet_tables_by_bytes
from farkle.utils.artifact_contract import (
    ArtifactSidecar,
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
    write_artifact_with_sidecar_atomic,
)
from farkle.utils.authenticated_contract import (
    ManifestEntry,
    ManifestRootIdentity,
    compute_manifest_root,
    load_immutable_manifest_sidecar,
    validate_authenticated_artifact_metadata,
)
from farkle.utils.manifest import iter_manifest
from farkle.utils.parallel import (
    ProcessTreeMemoryGuard,
    apply_native_thread_limits,
    resolve_stage_parallel_policy,
)
from farkle.utils.partitioned_stage import (
    PartitionedStageIdentity,
    PartitionedUnit,
    resolved_code_identity_sha256,
    run_partitioned_stage,
)
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose
from farkle.utils.release_identity import CapturedV3Inputs, is_v3_config
from farkle.utils.schema_helpers import (
    OUTCOME_SCHEMA_VERSION,
    TOURNAMENT_METHOD_VERSION,
    raw_simulation_schema_for,
)
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.writer import ParquetShardWriter

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RowShard:
    """One manifest-authenticated tournament row shard."""

    path: Path
    expected_rows: int
    root_seed: int
    k: int
    shuffle_index: int
    deterministic_batch_id: int
    shuffle_seed: int
    byte_length: int
    data_sha256: str
    sidecar_sha256: str
    schema_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class _SimulationSourceSnapshot:
    """Authenticated simulation lifecycle and ordered shard inventory for one k."""

    n_players: int
    block: Path
    manifest_path: Path
    manifest_sha256: str
    manifest_sidecar_sha256: str
    manifest_root: ManifestRootIdentity
    completion_path: Path
    completion_sha256: str
    shards: tuple[_RowShard, ...]

    @property
    def identity_sha256(self) -> str:
        payload = {
            "n_players": self.n_players,
            "manifest_sha256": self.manifest_sha256,
            "manifest_sidecar_sha256": self.manifest_sidecar_sha256,
            "completion_sha256": self.completion_sha256,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def captured_inputs(self) -> CapturedV3Inputs:
        role = f"control:{self.completion_path.name}:0000"
        return CapturedV3Inputs(
            sources=(),
            manifests=(self.manifest_root,),
            source_paths=(),
            manifest_paths=(
                (
                    self.manifest_root.logical_role,
                    str(self.manifest_path),
                    str(sidecar_path(self.manifest_path)),
                ),
            ),
            controls=((role, str(self.completion_path), self.completion_sha256),),
        )


def _ingested_rows_sidecar(
    cfg: AppConfig,
    *,
    block: Path,
    n_players: int,
    source_manifest: Path,
    schema: pa.Schema,
) -> ArtifactSidecar:
    """Build the contract for a completed streamed ingest artifact."""
    output = cfg.ingested_rows_raw(n_players)
    completion = block / "simulation.done.json"
    return make_artifact_sidecar(
        cfg,
        output,
        producer="ingest",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation="ingest_simulation_rows",
        weighted_quantity="canonical_game_rows",
        support_count_role="raw_games",
        uncertainty_method="none",
        replication_unit="game",
        conditioning="unconditional",
        consistency_columns=schema.names,
        source_artifacts=[source_manifest, completion],
        grouping_keys=["root_seed", "k", "shuffle_index", "game_index"],
        player_counts=[n_players],
        required_player_counts=[n_players],
        missing_cell_policy="fail",
        seed_scope="single_root",
        input_manifests=[source_manifest],
    )


def _canonical_row_shards(
    block: Path,
    cfg: AppConfig,
    n_players: int,
) -> _SimulationSourceSnapshot:
    """Resolve one authenticated simulation lifecycle into an immutable snapshot."""

    row_dir = cfg.simulation_row_dir(n_players)
    if row_dir is None:
        raise FileNotFoundError(
            f"ingest requires sim.row_dir for {n_players}-player canonical rows"
        )
    manifest_path = row_dir / "manifest.jsonl"
    completion_path = block / "simulation.done.json"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError(
            "ingest requires a completed canonical row-shard directory with "
            f"manifest.jsonl: {row_dir}"
        )
    if is_v3_config(cfg):
        if not simulation_is_complete(cfg, n_players):
            raise ValueError(f"simulation lifecycle does not authenticate for {n_players} players")
        workload_path = block / "simulation_workload_plan.json"
        workload = json.loads(workload_path.read_text(encoding="utf-8"))
        completion = {
            "root_seed": cfg.sim.seed,
            "n_players": n_players,
            "rng_scheme_version": cfg.rng.scheme_version,
            "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
            "tournament_method_version": TOURNAMENT_METHOD_VERSION,
            "shuffle_index_start": 0,
            "shuffle_index_end": int(workload["required_shuffles"]) - 1,
            "shuffles_per_batch": int(workload["shuffles_per_batch"]),
        }
    else:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    try:
        start = int(completion["shuffle_index_start"])
        end = int(completion["shuffle_index_end"])
        shuffles_per_batch = int(completion["shuffles_per_batch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid simulation completion contract: {completion_path}") from exc
    if (
        int(completion.get("root_seed", -1)) != int(cfg.sim.seed)
        or int(completion.get("n_players", -1)) != n_players
        or int(completion.get("rng_scheme_version", -1)) != int(cfg.rng.scheme_version)
        or int(completion.get("outcome_schema_version", -1)) != OUTCOME_SCHEMA_VERSION
        or int(completion.get("tournament_method_version", -1)) != TOURNAMENT_METHOD_VERSION
        or start < 0
        or end < start
        or shuffles_per_batch < 1
    ):
        raise ValueError(f"simulation completion mismatch: {completion_path}")

    if not is_v3_config(cfg):
        raise ValueError("immutable ingest source snapshots require artifact-contract-v3")
    manifest_sidecar = load_immutable_manifest_sidecar(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    manifest_adjacent_sha256 = sha256_file(sidecar_path(manifest_path))
    if manifest_sha256 != manifest_sidecar.manifest_sha256:
        raise ValueError(f"row manifest bytes do not match its immutable sidecar: {manifest_path}")

    records_by_index: dict[int, _RowShard] = {}
    seen_paths: set[Path] = set()
    manifest_entries: list[ManifestEntry] = []
    observed_order: list[int] = []
    for record in iter_manifest(manifest_path):
        raw_name = record.get("path")
        if not isinstance(raw_name, str):
            raise ValueError(f"row manifest entry missing path: {manifest_path}")
        relative = Path(raw_name)
        if relative.is_absolute() or relative.name != raw_name or not raw_name.startswith("rows_"):
            raise ValueError(f"invalid row manifest path {raw_name!r}: {manifest_path}")
        try:
            shuffle_index = int(record["shuffle_index"])
            expected_rows = int(record["rows"])
            batch_id = int(record["deterministic_batch_id"])
            shuffle_seed = int(record["shuffle_seed"])
            byte_length = int(record["byte_length"])
            data_sha256 = str(record["data_sha256"])
            shard_sidecar_sha256 = str(record["sidecar_sha256"])
            schema_fingerprint_sha256 = str(record["schema_fingerprint_sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid row manifest coordinate: {manifest_path}") from exc
        shard_path = row_dir / relative
        if (
            shuffle_index in records_by_index
            or shard_path in seen_paths
            or expected_rows < 1
            or int(record.get("root_seed", -1)) != int(cfg.sim.seed)
            or int(record.get("n_players", -1)) != n_players
            or int(record.get("rng_scheme_version", -1)) != int(cfg.rng.scheme_version)
            or int(record.get("outcome_schema_version", -1)) != OUTCOME_SCHEMA_VERSION
            or int(record.get("tournament_method_version", -1)) != TOURNAMENT_METHOD_VERSION
            or batch_id != shuffle_index // shuffles_per_batch
        ):
            raise ValueError(f"row manifest support mismatch: {manifest_path}")
        metadata = validate_authenticated_artifact_metadata(
            shard_path,
            cfg=cfg,
            expected_sidecar_sha256=shard_sidecar_sha256,
        )
        if (
            metadata.artifact.content_sha256 != data_sha256
            or metadata.artifact.byte_length != byte_length
            or metadata.artifact.arrow_schema is None
            or schema_fingerprint_sha256 != metadata.artifact.arrow_schema.fingerprint_sha256
        ):
            raise ValueError(f"row manifest identity mismatch for {shard_path}")
        records_by_index[shuffle_index] = _RowShard(
            path=shard_path,
            expected_rows=expected_rows,
            root_seed=int(cfg.sim.seed),
            k=n_players,
            shuffle_index=shuffle_index,
            deterministic_batch_id=batch_id,
            shuffle_seed=shuffle_seed,
            byte_length=byte_length,
            data_sha256=data_sha256,
            sidecar_sha256=shard_sidecar_sha256,
            schema_fingerprint_sha256=schema_fingerprint_sha256,
        )
        seen_paths.add(shard_path)
        observed_order.append(shuffle_index)
        manifest_entries.append(
            ManifestEntry(
                coordinate=(shuffle_index,),
                canonical_relative_path=shard_path.resolve()
                .relative_to(cfg.results_root.resolve())
                .as_posix(),
                data_sha256=data_sha256,
                sidecar_sha256=shard_sidecar_sha256,
                schema_fingerprint_sha256=schema_fingerprint_sha256,
            )
        )

    expected_indices = set(range(start, end + 1))
    if set(records_by_index) != expected_indices:
        raise ValueError(
            f"row manifest does not cover completed shuffle support {start}..{end}: {manifest_path}"
        )
    if observed_order != list(range(start, end + 1)):
        raise ValueError(
            f"row manifest entries are not in canonical coordinate order: {manifest_path}"
        )
    if compute_manifest_root(manifest_entries) != manifest_sidecar.summary:
        raise ValueError(f"row manifest root identity mismatch: {manifest_path}")
    disk_paths = set(row_dir.glob("rows_*.parquet"))
    if disk_paths != seen_paths:
        raise ValueError(f"row manifest and shard directory disagree: {row_dir}")
    location = manifest_sidecar.location
    role_relative = location.relative_path.replace("/", ".").replace("\\", ".")
    manifest_role = (
        f"manifest.{location.stage_key}.{location.scope}."
        f"k_{location.player_count if location.player_count is not None else 'all'}.{role_relative}"
    )
    manifest_root = ManifestRootIdentity(
        logical_role=manifest_role,
        location=location,
        manifest_sha256=manifest_sha256,
        sidecar_sha256=manifest_adjacent_sha256,
        sidecar_contract_sha256=manifest_sidecar.sidecar_contract_sha256,
        summary=manifest_sidecar.summary,
    )
    return _SimulationSourceSnapshot(
        n_players=n_players,
        block=block,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        manifest_sidecar_sha256=manifest_adjacent_sha256,
        manifest_root=manifest_root,
        completion_path=completion_path,
        completion_sha256=sha256_file(completion_path),
        shards=tuple(records_by_index[index] for index in range(start, end + 1)),
    )


def _numeric(table: pa.Table, name: str) -> np.ndarray:
    """Extract a primitive Arrow column without creating Python row objects."""

    column = table.column(name).combine_chunks()
    if column.null_count:
        column = pc.fill_null(column, pa.scalar(0, type=column.type))
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)


def _first_error(path: Path, message: str, invalid: np.ndarray) -> None:
    """Raise the earliest invalid offset so diagnostics are worker-invariant."""

    offsets = np.flatnonzero(invalid)
    if offsets.size:
        raise ValueError(f"{message}: {path} (batch row {int(offsets[0])})")


def _validate_batch(table: pa.Table, shard: _RowShard, path: Path) -> None:
    """Vectorially validate canonical coordinates, outcomes, and seat invariants."""

    rows = table.num_rows
    required = (
        "root_seed",
        "k",
        "shuffle_index",
        "game_index",
        "deterministic_batch_id",
        "shuffle_seed",
        "termination_status",
        "hit_safety_limit",
        "outcome_schema_version",
        "rng_scheme_version",
        "rng_purpose_namespace",
        *(
            f"P{seat}_{field}"
            for seat in range(1, shard.k + 1)
            for field in ("strategy", "score", "hit_max_rounds")
        ),
    )
    for name in required:
        if table.column(name).null_count:
            raise ValueError(f"row shard contains null required field {name}: {path}")
    identity = np.column_stack(
        [
            _numeric(table, name)
            for name in (
                "root_seed",
                "k",
                "shuffle_index",
                "deterministic_batch_id",
                "shuffle_seed",
            )
        ]
    )
    expected_identity = np.asarray(
        (
            shard.root_seed,
            shard.k,
            shard.shuffle_index,
            shard.deterministic_batch_id,
            shard.shuffle_seed,
        )
    )
    _first_error(
        path,
        "row shard internal root/k/shuffle/batch identity mismatch",
        np.any(identity != expected_identity, axis=1),
    )
    _first_error(
        path,
        "row shard internal version/namespace mismatch",
        (_numeric(table, "rng_scheme_version") != int(RNG_SCHEME_VERSION))
        | (_numeric(table, "rng_purpose_namespace") != int(RandomPurpose.TOURNAMENT_GAME))
        | (_numeric(table, "outcome_schema_version") != OUTCOME_SCHEMA_VERSION),
    )
    strategies = np.column_stack(
        [_numeric(table, f"P{seat}_strategy") for seat in range(1, shard.k + 1)]
    )
    for seat in range(1, shard.k + 1):
        _first_error(
            path,
            f"simulation row P{seat}_strategy must be within [0, 2147483647]",
            (strategies[:, seat - 1] < 0) | (strategies[:, seat - 1] > np.iinfo(np.int32).max),
        )
    _first_error(
        path,
        "Simulation row must seat distinct strategies",
        np.any(np.diff(np.sort(strategies, axis=1), axis=1) == 0, axis=1),
    )
    statuses = np.asarray(
        table.column("termination_status").combine_chunks().to_numpy(zero_copy_only=False),
        dtype=object,
    )
    completed, safety = statuses == "completed", statuses == "safety_limit"
    _first_error(path, "Simulation row has invalid k or termination_status", ~(completed | safety))
    hit = np.asarray(
        table.column("hit_safety_limit").combine_chunks().to_numpy(zero_copy_only=False), dtype=bool
    )
    hits = np.column_stack(
        [
            np.asarray(
                table.column(f"P{seat}_hit_max_rounds")
                .combine_chunks()
                .to_numpy(zero_copy_only=False),
                dtype=bool,
            )
            for seat in range(1, shard.k + 1)
        ]
    )
    scores = np.column_stack([_numeric(table, f"P{seat}_score") for seat in range(1, shard.k + 1)])
    rank_null = np.column_stack(
        [
            np.asarray(pc.is_null(table.column(f"P{seat}_rank")).to_numpy(), dtype=bool)
            for seat in range(1, shard.k + 1)
        ]
    )
    ranks = np.column_stack([_numeric(table, f"P{seat}_rank") for seat in range(1, shard.k + 1)])
    winner_null = np.asarray(pc.is_null(table.column("winner_seat")).to_numpy(), dtype=bool)
    winner = np.asarray(
        table.column("winner_seat").combine_chunks().to_numpy(zero_copy_only=False), dtype=object
    )
    winner_strategy_null = np.asarray(
        pc.is_null(table.column("winner_strategy")).to_numpy(), dtype=bool
    )
    winner_strategy = _numeric(table, "winner_strategy")
    score_null = np.asarray(pc.is_null(table.column("winning_score")).to_numpy(), dtype=bool)
    winning_score = _numeric(table, "winning_score")
    margin_null = np.asarray(pc.is_null(table.column("victory_margin")).to_numpy(), dtype=bool)
    margin = _numeric(table, "victory_margin")
    loss_null = np.column_stack(
        [
            np.asarray(pc.is_null(table.column(f"P{seat}_loss_margin")).to_numpy(), dtype=bool)
            for seat in range(1, shard.k + 1)
        ]
    )
    losses = np.column_stack(
        [_numeric(table, f"P{seat}_loss_margin") for seat in range(1, shard.k + 1)]
    )
    order = np.argsort(-scores, axis=1, kind="stable")
    expected_ranks = np.empty_like(ranks)
    expected_ranks[np.arange(rows)[:, None], order] = np.arange(1, shard.k + 1)
    winner_index = np.argmax(ranks == 1, axis=1)
    expected_winner = np.asarray([f"P{index + 1}" for index in winner_index], dtype=object)
    expected_strategy = strategies[np.arange(rows), winner_index]
    best = scores[np.arange(rows), order[:, 0]]
    runner_up = (
        scores[np.arange(rows), order[:, 1]] if shard.k > 1 else np.zeros(rows, dtype=np.int64)
    )
    _first_error(
        path,
        "Completed simulation row must have exactly one winner matching its rank-1 seat",
        completed & (winner_null | (np.sum(ranks == 1, axis=1) != 1) | (winner != expected_winner)),
    )
    _first_error(
        path,
        "Completed simulation row ranks must be the permutation 1..k",
        completed
        & (
            np.any(rank_null, axis=1)
            | np.any(np.sort(ranks, axis=1) != np.arange(1, shard.k + 1), axis=1)
        ),
    )
    _first_error(
        path,
        "Completed simulation row ranks are inconsistent with final scores",
        completed & np.any(ranks != expected_ranks, axis=1),
    )
    _first_error(
        path,
        "Completed simulation row must identify the winning strategy",
        completed & (winner_strategy_null | (winner_strategy != expected_strategy)),
    )
    _first_error(
        path,
        "Completed simulation row must retain winner-conditioned fields",
        completed & (score_null | margin_null),
    )
    _first_error(path, "Completed simulation row cannot hit the safety limit", completed & hit)
    _first_error(
        path,
        "Completed simulation row cannot mark a seat at the safety limit",
        completed & np.any(hits, axis=1),
    )
    _first_error(
        path,
        "Completed simulation row has inconsistent winning_score",
        completed & ((winning_score != best) | (winning_score != scores.max(axis=1))),
    )
    _first_error(
        path,
        "Completed simulation row has inconsistent victory_margin",
        completed & (margin != winning_score - runner_up),
    )
    for seat in range(1, shard.k + 1):
        _first_error(
            path,
            f"Completed simulation row has inconsistent P{seat}_loss_margin",
            completed
            & (
                loss_null[:, seat - 1]
                | (losses[:, seat - 1] != winning_score - scores[:, seat - 1])
            ),
        )
    _first_error(path, "Safety-limit simulation row must set hit_safety_limit=true", safety & ~hit)
    _first_error(
        path,
        "Safety-limit simulation row must mark every seat at the safety limit",
        safety & ~np.all(hits, axis=1),
    )
    _first_error(
        path,
        "Safety-limit simulation row cannot claim a winner",
        safety & (~winner_null | ~winner_strategy_null | ~score_null | ~margin_null),
    )
    _first_error(
        path,
        "Safety-limit simulation row cannot assign participant ranks",
        safety & ~np.all(rank_null, axis=1),
    )
    _first_error(
        path,
        "Safety-limit simulation row cannot assign loss margins",
        safety & ~np.all(loss_null, axis=1),
    )
    list_ranks = table.column("seat_ranks").combine_chunks()
    list_lengths = pc.list_value_length(list_ranks)
    if list_lengths.null_count:
        raise ValueError(f"Simulation row has null seat_ranks: {path}")
    lengths = np.asarray(list_lengths.to_numpy(zero_copy_only=False), dtype=np.int64)
    _first_error(path, "Simulation row has inconsistent seat_ranks length", lengths != shard.k)
    expected_seats = np.asarray(
        [[f"P{seat + 1}" for seat in sequence] for sequence in order], dtype=object
    )
    for position in range(shard.k):
        actual = np.asarray(
            pc.list_element(list_ranks, position).to_numpy(zero_copy_only=False), dtype=object
        )
        expected_rank = expected_seats[:, position].astype(object)
        expected_rank[~completed] = None
        _first_error(path, "Simulation row has inconsistent seat_ranks", actual != expected_rank)


def _iter_shards_arrow(
    shards: tuple[_RowShard, ...],
    *,
    columns: tuple[str, ...],
    max_batch_bytes: int,
    max_batch_rows: int,
):
    """Projected stream with exact compact game-index bitmaps per input shard."""

    for shard in shards:
        path = shard.path
        parquet = pq.ParquetFile(path)
        expected = raw_simulation_schema_for(shard.k)
        unexpected = sorted(set(parquet.schema_arrow.names).difference(expected.names))
        missing = sorted(set(expected.names).difference(parquet.schema_arrow.names))
        if unexpected or missing:
            raise ValueError(
                f"row shard contains noncanonical columns {unexpected} and misses required columns {missing}: {path}"
            )
        if not parquet.schema_arrow.equals(expected, check_metadata=False):
            raise ValueError(
                f"row shard schema is not the exact canonical raw schema for k={shard.k}: {path}"
            )
        if parquet.metadata.num_rows != shard.expected_rows:
            raise ValueError(
                f"row manifest count mismatch for {path}: expected {shard.expected_rows}, found {parquet.metadata.num_rows}"
            )
        seen = bytearray((shard.expected_rows + 7) // 8)
        decoded_peak = 0

        def observe(size: int) -> None:
            nonlocal decoded_peak
            decoded_peak = max(decoded_peak, size)

        for _group, _batch, table in iter_parquet_tables_by_bytes(
            path,
            columns=columns,
            max_batch_bytes=max_batch_bytes,
            max_batch_rows=max_batch_rows,
            use_threads=False,
            on_decoded_batch=observe,
        ):
            _validate_batch(table, shard, path)
            indices = _numeric(table, "game_index")
            _first_error(
                path,
                f"row shard game_index support must be 0..{shard.expected_rows - 1}",
                (indices < 0) | (indices >= shard.expected_rows),
            )
            for index in indices:
                byte, bit = divmod(int(index), 8)
                if seen[byte] & (1 << bit):
                    raise ValueError(f"row shard contains duplicate or invalid game key: {path}")
                seen[byte] |= 1 << bit
            yield table, path
        if any(value != 255 for value in seen[:-1]) or (
            shard.expected_rows % 8 and seen[-1] != (1 << (shard.expected_rows % 8)) - 1
        ):
            raise ValueError(
                f"row shard game_index support must be 0..{shard.expected_rows - 1}: {path}"
            )
        LOGGER.debug(
            "Ingest shard decoded",
            extra={
                "stage": "ingest",
                "path": path.name,
                "decoded_peak_bytes": decoded_peak,
                "working_batch_bytes": max_batch_bytes,
            },
        )


@dataclass(frozen=True)
class _IngestUnitWriter:
    """Pickle-safe per-k writer used by the shared partition executor."""

    cfg: AppConfig
    sources: tuple[_SimulationSourceSnapshot, ...]

    def __call__(self, unit: PartitionedUnit, staged: Path) -> None:
        n_players = int(unit.key[0])
        snapshot = {source.n_players: source for source in self.sources}[n_players]
        schema = raw_simulation_schema_for(n_players)
        max_bytes = int(self.cfg.resources.stage_batch_bytes["ingest"])
        with ParquetShardWriter(
            out_path=str(staged),
            schema=schema,
            compression=self.cfg.parquet_codec,
            row_group_size=self.cfg.row_group_size,
        ) as writer:
            for table, _path in _iter_shards_arrow(
                snapshot.shards,
                columns=tuple(schema.names),
                max_batch_bytes=max_bytes,
                max_batch_rows=max(1, int(self.cfg.ingest.batch_rows)),
            ):
                if table.nbytes > max_bytes:
                    raise MemoryError("ingest working batch crossed its configured byte ceiling")
                writer.write_batch(table)
        if writer.rows_written < 1:
            raise ValueError(f"ingest produced zero rows for configured k={n_players}")


@dataclass(frozen=True)
class _IngestUnitPublisher:
    """Publish and validate the canonical output pair beside each completed k unit."""

    cfg: AppConfig
    sources: tuple[_SimulationSourceSnapshot, ...]
    sidecars: tuple[tuple[int, ArtifactSidecar], ...]

    def _source(self, unit: PartitionedUnit) -> _SimulationSourceSnapshot:
        n_players = int(unit.key[0])
        return {source.n_players: source for source in self.sources}[n_players]

    def sidecar(self, unit: PartitionedUnit, output: Path) -> ArtifactSidecar:
        snapshot = self._source(unit)
        n_players = snapshot.n_players
        if output != self.cfg.ingested_rows_raw(n_players):
            raise ValueError("ingest partition output does not match the canonical by-k path")
        return dict(self.sidecars)[n_players]

    def publish_manifest(self, unit: PartitionedUnit, output: Path) -> None:
        snapshot = self._source(unit)
        n_players, block = snapshot.n_players, snapshot.block
        manifest = self.cfg.ingest_manifest(n_players)
        rows = int(pq.ParquetFile(output).metadata.num_rows)
        record = {
            "path": output.name,
            "rows": rows,
            "n_players": n_players,
            "source_block": block.name,
            "root_seed": int(self.cfg.sim.seed),
            "coordinate_columns": [
                "root_seed",
                "k",
                "shuffle_index",
                "game_index",
                "deterministic_batch_id",
            ],
        }
        content = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        metadata = make_artifact_sidecar(
            self.cfg,
            manifest,
            producer="ingest",
            scope=ArtifactScope.BY_K,
            source_scope=ArtifactScope.BY_K,
            operation="ingest_simulation_rows_stream_manifest",
            weighted_quantity="coordinate_manifest",
            support_count_role="streamed_output_inventory",
            uncertainty_method="none",
            replication_unit="manifest_entry",
            conditioning="unconditional",
            source_artifacts=[output],
            player_counts=[n_players],
            required_player_counts=[n_players],
            missing_cell_policy="fail",
            seed_scope="single_root",
        )

        def write_manifest(staged: Path) -> None:
            staged.write_bytes(content)

        write_artifact_with_sidecar_atomic(manifest, metadata, write_manifest)

    def validate(self, unit: PartitionedUnit, output: Path) -> bool:
        snapshot = self._source(unit)
        n_players, block = snapshot.n_players, snapshot.block
        if output != self.cfg.ingested_rows_raw(n_players):
            return False
        if not pq.read_schema(output).equals(
            raw_simulation_schema_for(n_players), check_metadata=False
        ):
            return False
        validate_artifact_sidecar(
            output,
            expected={
                "scope": ArtifactScope.BY_K.value,
                "operation": "ingest_simulation_rows",
                "player_counts": [n_players],
            },
        )
        manifest = self.cfg.ingest_manifest(n_players)
        validate_artifact_sidecar(
            manifest,
            expected={"operation": "ingest_simulation_rows_stream_manifest"},
        )
        records = list(iter_manifest(manifest))
        loaded_output = validate_authenticated_artifact_metadata(output, cfg=self.cfg)
        captured = dict(self.sidecars)[n_players]._captured_v3_inputs
        return (
            len(records) == 1
            and records[0]
            == {
                "path": output.name,
                "rows": int(pq.ParquetFile(output).metadata.num_rows),
                "n_players": n_players,
                "source_block": block.name,
                "root_seed": int(self.cfg.sim.seed),
                "coordinate_columns": [
                    "root_seed",
                    "k",
                    "shuffle_index",
                    "game_index",
                    "deterministic_batch_id",
                ],
            }
            and captured is not None
            and loaded_output.manifest_roots == captured.manifests
            and all(
                loaded_output.stage_identity.immutable_design_identities.get(role) == digest
                for role, digest in captured.designs.items()
            )
        )


def _ingest_partition_identity(
    cfg: AppConfig,
    sources: tuple[_SimulationSourceSnapshot, ...],
) -> PartitionedStageIdentity:
    """Bind every configured simulation manifest to the reusable k units."""

    inputs: list[tuple[str, str]] = []
    for source in sources:
        inputs.append((f"k{source.n_players:03d}_simulation_rows", source.identity_sha256))
    return PartitionedStageIdentity(
        stage_name="ingest",
        root_seed=int(cfg.sim.seed),
        input_identities=tuple(sorted(inputs)),
        statistical_config_sha256=cfg.stage_config_sha("ingest"),
        code_identity_sha256=resolved_code_identity_sha256(cfg),
        schema_version=2,
        method_version=2,
    )


def _run_partitioned_ingest(cfg: AppConfig) -> None:
    player_counts = tuple(sorted(int(value) for value in cfg.sim.n_players_list))
    blocks = [cfg.results_root / f"{n_players}_players" for n_players in player_counts]
    missing = [str(block) for block in blocks if not block.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"ingest is incomplete; missing configured simulation blocks: {missing}"
        )
    stage_policy = resolve_stage_parallel_policy("ingest", cfg.ingest, resources=cfg.resources)
    apply_native_thread_limits(stage_policy)
    pa.set_cpu_count(stage_policy.arrow_threads)
    pa.set_io_thread_count(stage_policy.arrow_threads)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    outputs = [cfg.ingested_rows_raw(k) for k in player_counts]
    manifests = [cfg.ingest_manifest(k) for k in player_counts]
    sources = tuple(
        _canonical_row_shards(block, cfg, n_players)
        for block, n_players in zip(blocks, player_counts, strict=True)
    )
    upstream_inputs = [
        path for source in sources for path in (source.manifest_path, source.completion_path)
    ]
    identity = _ingest_partition_identity(cfg, sources)
    guard = ProcessTreeMemoryGuard(
        cfg.resources.aggregate_memory_hard_limit_mb,
        rss_warning_mb=cfg.resources.process_tree_warning_threshold_mb,
        minimum_system_available_memory_mb=cfg.resources.minimum_system_available_memory_mb,
        sample_interval_seconds=cfg.resources.rss_sample_interval_seconds,
    )
    captured_sidecars: list[tuple[int, ArtifactSidecar]] = []
    for source in sources:
        template = _ingested_rows_sidecar(
            cfg,
            block=source.block,
            n_players=source.n_players,
            source_manifest=source.manifest_path,
            schema=raw_simulation_schema_for(source.n_players),
        )
        captured_sidecars.append(
            (
                source.n_players,
                replace(template, _captured_v3_inputs=source.captured_inputs),
            )
        )
    publisher = _IngestUnitPublisher(cfg, sources, tuple(captured_sidecars))
    units = tuple(
        PartitionedUnit((k,), f"{k}p/{cfg.ingested_rows_raw(k).name}") for k in player_counts
    )
    result = run_partitioned_stage(
        root=cfg.ingest_stage_dir,
        identity=identity,
        unit_source=lambda: iter(units),
        writer=_IngestUnitWriter(cfg, sources),
        resources=cfg.resources,
        requested_workers=cfg.ingest.n_jobs,
        mp_start_method=cfg.analysis.mp_start_method,
        memory_guard=guard,
        output_prefix=ArtifactScope.BY_K.value,
        sidecar_factory=publisher.sidecar,
        post_publisher=publisher.publish_manifest,
        validator=publisher.validate,
    )
    if result.required_units != len(player_counts):
        raise RuntimeError("ingest final manifest does not cover every configured k")
    done = stage_done_path(cfg.ingest_stage_dir, "ingest")
    if stage_is_up_to_date(
        done,
        inputs=upstream_inputs,
        outputs=[*outputs, *manifests],
        cfg=cfg,
        stage="ingest",
        sidecar_artifacts=outputs,
    ):
        LOGGER.info("Ingest up-to-date", extra={"stage": "ingest", "path": str(done)})
        return
    guard.check_before_schedule(force=True)
    write_stage_done(
        done,
        inputs=upstream_inputs,
        outputs=[*outputs, *manifests],
        cfg=cfg,
        stage="ingest",
        sidecar_artifacts=outputs,
    )
    LOGGER.info(
        "Ingest finished",
        extra={
            "stage": "ingest",
            "blocks": len(blocks),
            "rows": sum(int(pq.ParquetFile(path).metadata.num_rows) for path in outputs),
            "reused_k": result.reused_units,
            "completed_k": result.completed_units,
            "peak_sampled_rss_mb": result.peak_sampled_rss_mb,
        },
    )


def run(cfg: AppConfig) -> None:
    """Ingest authenticated raw game results into canonical by-k Parquets."""

    _run_partitioned_ingest(cfg)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI wrapper
    """Parse command-line arguments and invoke :func:`run`."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/fast_config.yaml"), help="Path to YAML config"
    )
    args = parser.parse_args(argv)
    app_cfg = load_app_config(Path(args.config))
    from farkle.utils.os_memory import supervise_module_if_needed

    arguments = list(argv) if argv is not None else sys.argv[1:]
    exit_code = supervise_module_if_needed(__name__, arguments, app_cfg.resources)
    if exit_code is not None:
        raise SystemExit(exit_code)
    run(app_cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
