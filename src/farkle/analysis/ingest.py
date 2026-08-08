# src/farkle/analysis/ingest.py
"""Ingest raw simulation results into parquet shards for curation.

This entry point streams over experiment outputs, validates schemas, and
writes player-count-specific shards that feed the downstream combine and
metrics stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.config import AppConfig, ArtifactScope, load_app_config
from farkle.simulation.runner import simulation_is_complete
from farkle.simulation.simulation import validate_simulation_row
from farkle.utils.artifact_contract import (
    ArtifactSidecar,
    ensure_artifact_sidecar_atomic,
    make_artifact_sidecar,
)
from farkle.utils.manifest import iter_manifest
from farkle.utils.parallel import (
    ParallelNestingContext,
    ProcessTreeMemoryGuard,
    apply_native_thread_limits,
    normalize_n_jobs,
    resolve_mp_context,
    resolve_stage_parallel_policy,
)
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose
from farkle.utils.release_identity import is_v3_config
from farkle.utils.schema_helpers import (
    OUTCOME_SCHEMA_VERSION,
    TOURNAMENT_METHOD_VERSION,
    raw_simulation_schema_for,
)
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.streaming_loop import run_streaming_shard

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


def _ensure_ingested_rows_sidecar(
    cfg: AppConfig,
    *,
    block: Path,
    n_players: int,
    source_manifest: Path,
) -> None:
    """Bind an existing complete ingest artifact without rewriting its bytes."""

    output = cfg.ingested_rows_raw(n_players)
    sidecar = _ingested_rows_sidecar(
        cfg,
        block=block,
        n_players=n_players,
        source_manifest=source_manifest,
        schema=pq.read_schema(output),
    )
    ensure_artifact_sidecar_atomic(
        output,
        sidecar,
        expected={
            "scope": ArtifactScope.BY_K.value,
            "operation": "ingest_simulation_rows",
            "player_counts": [n_players],
        },
    )


def _canonical_row_shards(
    block: Path,
    cfg: AppConfig,
    n_players: int,
) -> tuple[Path, list[_RowShard]]:
    """Validate and return manifest-ordered canonical simulation row shards."""

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

    records_by_index: dict[int, _RowShard] = {}
    seen_paths: set[Path] = set()
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
        records_by_index[shuffle_index] = _RowShard(
            path=shard_path,
            expected_rows=expected_rows,
            root_seed=int(cfg.sim.seed),
            k=n_players,
            shuffle_index=shuffle_index,
            deterministic_batch_id=batch_id,
            shuffle_seed=shuffle_seed,
        )
        seen_paths.add(shard_path)

    expected_indices = set(range(start, end + 1))
    if set(records_by_index) != expected_indices:
        raise ValueError(
            f"row manifest does not cover completed shuffle support {start}..{end}: {manifest_path}"
        )
    disk_paths = set(row_dir.glob("rows_*.parquet"))
    if disk_paths != seen_paths:
        raise ValueError(f"row manifest and shard directory disagree: {row_dir}")
    return manifest_path, [records_by_index[index] for index in range(start, end + 1)]


def _iter_shards(shards: list[_RowShard], cols: tuple[str, ...]):
    """Yield validated shard row groups with the requested canonical columns."""

    for shard in shards:
        row_file = shard.path
        if not row_file.is_file():
            raise FileNotFoundError(f"row manifest references missing shard: {row_file}")
        parquet = pq.ParquetFile(row_file)
        raw_schema = raw_simulation_schema_for(shard.k)
        unexpected = sorted(set(parquet.schema_arrow.names).difference(raw_schema.names))
        missing = sorted(set(raw_schema.names).difference(parquet.schema_arrow.names))
        if unexpected or missing:
            raise ValueError(
                f"row shard contains noncanonical columns {unexpected} and misses "
                f"required columns {missing}: {row_file}"
            )
        if not parquet.schema_arrow.equals(raw_schema, check_metadata=False):
            raise ValueError(
                f"row shard schema is not the exact canonical raw schema for k={shard.k}: "
                f"{row_file}"
            )
        if parquet.metadata.num_rows != shard.expected_rows:
            raise ValueError(
                f"row manifest count mismatch for {row_file}: "
                f"expected {shard.expected_rows}, found {parquet.metadata.num_rows}"
            )
        present = [column for column in cols if column in parquet.schema_arrow.names]
        game_indices: set[int] = set()
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group)
            for row in table.to_pylist():
                identity = (
                    row.get("root_seed"),
                    row.get("k"),
                    row.get("shuffle_index"),
                    row.get("deterministic_batch_id"),
                    row.get("shuffle_seed"),
                )
                expected_identity = (
                    shard.root_seed,
                    shard.k,
                    shard.shuffle_index,
                    shard.deterministic_batch_id,
                    shard.shuffle_seed,
                )
                if identity != expected_identity:
                    raise ValueError(
                        f"row shard internal root/k/shuffle/batch identity mismatch: {row_file}"
                    )
                if (
                    row.get("rng_scheme_version") != int(RNG_SCHEME_VERSION)
                    or row.get("rng_purpose_namespace") != int(RandomPurpose.TOURNAMENT_GAME)
                    or row.get("outcome_schema_version") != OUTCOME_SCHEMA_VERSION
                ):
                    raise ValueError(f"row shard internal version/namespace mismatch: {row_file}")
                game_index = row.get("game_index")
                if (
                    isinstance(game_index, bool)
                    or not isinstance(game_index, (int, np.integer))
                    or int(game_index) in game_indices
                ):
                    raise ValueError(
                        f"row shard contains duplicate or invalid game key: {row_file}"
                    )
                game_indices.add(int(game_index))
                validate_simulation_row(row)
            yield table.select(present).to_pandas(), row_file
        if game_indices != set(range(shard.expected_rows)):
            raise ValueError(
                f"row shard game_index support must be 0..{shard.expected_rows - 1}: {row_file}"
            )


# Regex once, reuse
_SEAT_RE = re.compile(r"^P(\d+)_strategy$")


def _fix_winner(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and complete the canonical winner-related columns.

    Args:
        df: Raw results dataframe containing winner and seat strategy columns.

    Returns:
        A copy of the dataframe with standardized ``winner_seat``,
        ``winner_strategy``, and ``seat_ranks`` columns.
    """
    df = df.copy()

    if "winner" in df.columns:
        raise ValueError("retired winner column is not accepted; expected winner_seat")

    # strategy seat columns (P1_strategy, …)
    seat_cols = sorted(
        [c for c in df.columns if _SEAT_RE.match(c)],
        key=lambda c: int(_SEAT_RE.match(c).group(1)),  # type: ignore
    )

    # winner_strategy derived from seat strategy identifiers (add if missing)
    if "winner_strategy" not in df.columns and seat_cols:
        seat_idx = (
            df["winner_seat"].str.extract(r"P(?P<num>\d+)", expand=True)["num"].astype("Int64")
        )
        S = df[seat_cols].to_numpy(dtype=object)
        out = np.empty(len(df), dtype=object)
        rows = np.arange(len(df))
        has = seat_idx.notna()
        out[has.to_numpy()] = S[rows[has], (seat_idx[has] - 1).astype(int)]
        out[~has.to_numpy()] = None
        df["winner_strategy"] = out

    # seat_ranks: list[str] like ["P6","P2",...]
    if "seat_ranks" not in df.columns and seat_cols:
        rank_cols = [c.replace("_strategy", "_rank") for c in seat_cols]
        if all(col in df.columns for col in rank_cols):
            R = df[rank_cols].to_numpy(dtype=float)
            fill = R.shape[1] + 1
            np.nan_to_num(R, copy=False, nan=fill)
            seats = np.array([c.split("_", 1)[0] for c in seat_cols], dtype=object)
            order = np.argsort(R, axis=1)
            df["seat_ranks"] = [list(seats[idx]) for idx in order]
        elif "winner_seat" in df.columns:
            df["seat_ranks"] = df["winner_seat"].apply(lambda s: [s])

    return df


def _n_from_block(name: str) -> int | None:
    """Extract the player count from a ``<N>_players`` directory name.

    Args:
        name: Directory basename encoded with the player count.

    Returns:
        Parsed number of players, or ``None`` when the name does not follow
        the expected pattern.
    """
    m = re.match(r"^(\d+)_players$", name)
    return int(m.group(1)) if m else None


def _ingest_upstream_inputs(results_root: Path) -> list[Path]:
    """Return deterministic upstream files that should invalidate ingest freshness.

    Directory mtimes can stay unchanged when shard file contents are rewritten, so
    ingest freshness must key off concrete files beneath each ``*_players`` block.
    """

    blocks = sorted(
        (p for p in results_root.iterdir() if p.is_dir() and p.name.endswith("_players")),
        key=lambda p: (_n_from_block(p.name) or sys.maxsize, p.name),
    )
    inputs: list[Path] = []
    allowed_suffixes = {".parquet", ".csv", ".json", ".jsonl", ".txt"}
    for block in blocks:
        block_files = sorted(
            (p for p in block.rglob("*") if p.is_file() and p.suffix in allowed_suffixes),
            key=lambda p: p.relative_to(results_root).as_posix(),
        )
        inputs.extend(block_files)
    return inputs


def _process_block(block: Path, cfg: AppConfig, *, parent_process_workers: int = 1) -> int:
    """Process a single ``<N>_players`` block."""
    n = _n_from_block(block.name)
    if n is None:
        raise ValueError(f"invalid player-count block name: {block.name}")
    worker_policy = resolve_stage_parallel_policy(
        "ingest",
        cfg.ingest,
        ParallelNestingContext(
            active_process_executor=parent_process_workers > 1,
            parent_process_workers=max(1, int(parent_process_workers)),
        ),
        resources=cfg.resources,
    )
    apply_native_thread_limits(worker_policy)
    pa.set_cpu_count(worker_policy.arrow_threads)
    pa.set_io_thread_count(worker_policy.arrow_threads)
    LOGGER.info(
        "Ingest block discovered",
        extra={"stage": "ingest", "block": block.name, "path": str(block)},
    )

    raw_out = cfg.ingested_rows_raw(n)
    source_manifest, row_shards = _canonical_row_shards(block, cfg, n)

    canon = raw_simulation_schema_for(n)
    seat_cols = [c for c in canon.names if c.startswith("P")]
    wanted = tuple(
        dict.fromkeys(
            (
                *canon.names,
                *seat_cols,
            )
        )
    )

    total = 0

    def _iter_tables():
        """Yield canonicalized parquet tables from discovered shards.

        Returns:
            An iterator over :class:`pyarrow.Table` objects aligned to the
            expected schema for the current player count.
        """
        nonlocal total
        for shard_df, shard_path in _iter_shards(row_shards, tuple(wanted)):
            if shard_df.empty:
                LOGGER.debug(
                    "Empty shard skipped",
                    extra={"stage": "ingest", "path": shard_path.name},
                )
                continue
            LOGGER.debug(
                "Shard processed",
                extra={
                    "stage": "ingest",
                    "path": shard_path.name,
                    "rows": len(shard_df),
                },
            )

            shard_df = _fix_winner(shard_df)
            canon_names = canon.names
            extras = sorted(
                c for c in shard_df.columns if c not in canon_names and not c.startswith("P")
            )
            if extras:
                LOGGER.error(
                    "Schema mismatch",
                    extra={
                        "stage": "ingest",
                        "path": str(shard_path),
                        "unexpected_columns": extras,
                    },
                )
                raise RuntimeError("Schema mismatch")
            missing_columns = sorted(set(canon_names).difference(shard_df.columns))
            if missing_columns:
                raise ValueError(
                    f"row shard is missing required identity/outcome columns "
                    f"{missing_columns}: {shard_path}"
                )
            shard_df = shard_df[canon_names]
            table = pa.Table.from_pandas(shard_df, schema=canon, preserve_index=False)
            total += len(shard_df)
            yield table

    batches = _iter_tables()
    first = next(batches, None)
    if first is None:
        if raw_out.exists():
            raw_out.unlink()
        manifest_candidate = raw_out.with_suffix(".manifest.jsonl")
        if manifest_candidate.exists():
            manifest_candidate.unlink()
        LOGGER.info(
            "Ingest block produced zero rows",
            extra={"stage": "ingest", "n_players": n, "path": str(block)},
        )
        return 0

    def _all_batches():
        """Iterate over the first and remaining batches for streaming writes."""
        yield first
        yield from batches

    manifest_path = cfg.ingest_manifest(n)
    sidecar = _ingested_rows_sidecar(
        cfg,
        block=block,
        n_players=n,
        source_manifest=source_manifest,
        schema=canon,
    )
    run_streaming_shard(
        out_path=str(raw_out),
        manifest_path=str(manifest_path),
        schema=canon,
        batch_iter=_all_batches(),
        row_group_size=cfg.row_group_size,
        compression=cfg.parquet_codec,
        sidecar=sidecar,
        manifest_extra={
            "path": raw_out.name,
            "n_players": n,
            "source_block": block.name,
            "root_seed": cfg.sim.seed,
            "coordinate_columns": [
                "root_seed",
                "k",
                "shuffle_index",
                "game_index",
                "deterministic_batch_id",
            ],
        },
    )
    _ensure_ingested_rows_sidecar(
        cfg,
        block=block,
        n_players=n,
        source_manifest=source_manifest,
    )
    LOGGER.info(
        "Ingest block complete",
        extra={
            "stage": "ingest",
            "n_players": n,
            "rows": total,
            "path": str(raw_out),
            "manifest": str(manifest_path),
        },
    )
    return total


def run(cfg: AppConfig) -> None:
    """Ingest raw game results into curated parquet files and manifests.

    Args:
        cfg: Application configuration containing input/output paths and
            parallelism controls.
    """
    resolved_n_jobs = normalize_n_jobs(cfg.ingest.n_jobs)
    stage_policy = resolve_stage_parallel_policy("ingest", cfg.ingest, resources=cfg.resources)
    apply_native_thread_limits(stage_policy)
    pa.set_cpu_count(stage_policy.arrow_threads)
    pa.set_io_thread_count(stage_policy.arrow_threads)
    LOGGER.info(
        "Ingest started",
        extra={
            "stage": "ingest",
            "root": str(cfg.results_root),
            "data_dir": str(cfg.data_dir),
            "n_jobs": resolved_n_jobs,
            "process_workers": stage_policy.process_workers,
            "python_threads": stage_policy.python_threads,
            "arrow_threads": stage_policy.arrow_threads,
        },
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    blocks = sorted(
        (p for p in cfg.results_root.iterdir() if p.is_dir() and _n_from_block(p.name) is not None),
        key=lambda p: (_n_from_block(p.name) or sys.maxsize, p.name),
    )

    done = stage_done_path(cfg.ingest_stage_dir, "ingest")
    outputs = []
    manifests = []
    for block in blocks:
        n = _n_from_block(block.name)
        if n is None:  # pragma: no cover - filtered above
            continue
        outputs.append(cfg.ingested_rows_raw(n))
        manifests.append(cfg.ingest_manifest(n))
    upstream_inputs = _ingest_upstream_inputs(cfg.results_root)

    if stage_is_up_to_date(
        done,
        inputs=upstream_inputs,
        outputs=[*outputs, *manifests],
        cfg=cfg,
        stage="ingest",
        sidecar_artifacts=outputs,
    ):
        LOGGER.info(
            "Ingest up-to-date",
            extra={"stage": "ingest", "path": str(done)},
        )
        return

    if stage_is_up_to_date(
        done,
        inputs=upstream_inputs,
        outputs=[*outputs, *manifests],
        cfg=cfg,
        stage="ingest",
    ):
        for block in blocks:
            n = _n_from_block(block.name)
            if n is None:  # pragma: no cover - filtered above
                continue
            source_manifest, _row_shards = _canonical_row_shards(block, cfg, n)
            _ensure_ingested_rows_sidecar(
                cfg,
                block=block,
                n_players=n,
                source_manifest=source_manifest,
            )
        write_stage_done(
            done,
            inputs=upstream_inputs,
            outputs=[*outputs, *manifests],
            cfg=cfg,
            stage="ingest",
            sidecar_artifacts=outputs,
        )
        LOGGER.info("Ingest sidecars backfilled", extra={"stage": "ingest"})
        return

    mp_context = resolve_mp_context(cfg.analysis.mp_start_method)

    total_rows = 0
    memory_guard = ProcessTreeMemoryGuard(
        cfg.resources.rss_abort_mb,
        cfg.resources.rss_sample_interval_seconds,
    )
    memory_guard.check_before_schedule(force=True)
    if stage_policy.process_workers <= 1:
        for block in blocks:
            memory_guard.check_before_schedule()
            total_rows += _process_block(block, cfg, parent_process_workers=1)
    else:
        with ProcessPoolExecutor(
            max_workers=stage_policy.process_workers, mp_context=mp_context
        ) as executor:
            futures = []
            for block in blocks:
                memory_guard.check_before_schedule()
                futures.append(
                    executor.submit(
                        _process_block,
                        block,
                        cfg,
                        parent_process_workers=stage_policy.process_workers,
                    )
                )
            for f in futures:
                total_rows += f.result()

    LOGGER.info(
        "Ingest finished",
        extra={
            "stage": "ingest",
            "blocks": len(blocks),
            "rows": total_rows,
        },
    )
    write_stage_done(
        done,
        inputs=upstream_inputs,
        outputs=[*outputs, *manifests],
        cfg=cfg,
        stage="ingest",
        sidecar_artifacts=outputs,
    )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI wrapper
    """Parse command-line arguments and invoke :func:`run`."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/fast_config.yaml"), help="Path to YAML config"
    )
    args = parser.parse_args(argv)
    app_cfg = load_app_config(Path(args.config))
    run(app_cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
