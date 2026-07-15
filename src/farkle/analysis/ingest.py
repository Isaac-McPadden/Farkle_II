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
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.config import AppConfig, load_app_config
from farkle.utils.manifest import iter_manifest
from farkle.utils.parallel import (
    ParallelNestingContext,
    apply_native_thread_limits,
    normalize_n_jobs,
    resolve_mp_context,
    resolve_stage_parallel_policy,
)
from farkle.utils.schema_helpers import expected_schema_for
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.streaming_loop import run_streaming_shard

LOGGER = logging.getLogger(__name__)


def _canonical_row_shards(
    block: Path,
    cfg: AppConfig,
    n_players: int,
) -> tuple[Path, list[tuple[Path, int]]]:
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
        or start < 0
        or end < start
        or shuffles_per_batch < 1
    ):
        raise ValueError(f"simulation completion mismatch: {completion_path}")

    records_by_index: dict[int, tuple[Path, int]] = {}
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
            or batch_id != shuffle_index // shuffles_per_batch
        ):
            raise ValueError(f"row manifest support mismatch: {manifest_path}")
        records_by_index[shuffle_index] = (shard_path, expected_rows)
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


def _iter_shards(shards: list[tuple[Path, int]], cols: tuple[str, ...]):
    """
    Yield one ``(DataFrame, source_path)`` per shard, selecting only the
    *intersection* of *cols* with the shard’s schema.

    Parquet will raise if we request a column that doesn’t exist, so read the
    whole shard once and trim afterwards.  The ingest pipeline later pads any
    truly-missing columns back in :func:`_coerce_schema`, keeping the final
    table rectangular.
    """

    for row_file, expected_rows in shards:
        if not row_file.is_file():
            raise FileNotFoundError(f"row manifest references missing shard: {row_file}")
        parquet = pq.ParquetFile(row_file)
        unexpected = sorted(set(parquet.schema_arrow.names).difference(cols))
        if unexpected:
            raise ValueError(f"row shard contains noncanonical columns {unexpected}: {row_file}")
        if parquet.metadata.num_rows != expected_rows:
            raise ValueError(
                f"row manifest count mismatch for {row_file}: "
                f"expected {expected_rows}, found {parquet.metadata.num_rows}"
            )
        present = [column for column in cols if column in parquet.schema_arrow.names]
        for row_group in range(parquet.num_row_groups):
            yield parquet.read_row_group(row_group, columns=present).to_pandas(), row_file


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


def _coerce_strategy_ids(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure all strategy columns are normalized to integer IDs."""
    df = df.copy()
    strategy_cols = [c for c in df.columns if c == "winner_strategy" or _SEAT_RE.match(c)]
    for col in strategy_cols:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        fractional_mask = numeric.notna() & (numeric % 1 != 0)
        if fractional_mask.any():
            sample = series[fractional_mask].iloc[0]
            LOGGER.debug(
                "Non-integer strategy value detected",
                extra={
                    "stage": "ingest",
                    "column": col,
                    "sample_type": type(sample).__name__,
                },
            )
            raise RuntimeError("Non-integer strategy value detected in ingest")
        id_series = numeric.astype("Int64")
        missing_mask = id_series.isna() & series.notna()
        if missing_mask.any():
            sample = series[missing_mask].iloc[0]
            raise ValueError(
                f"retired nonnumeric strategy identifier in {col}: {sample!r}; "
                "regenerate simulation artifacts with numeric strategy IDs"
            )
        unresolved = id_series.isna() & series.notna()
        if unresolved.any():
            sample = series[unresolved].iloc[0]
            LOGGER.error(
                "Unmapped strategy identifiers detected",
                extra={
                    "stage": "ingest",
                    "column": col,
                    "sample": str(sample),
                },
            )
            raise RuntimeError("Unmapped strategy identifiers detected")
        df[col] = id_series.astype("Int64")
    return df


def _validate_strategy_dtypes(df: pd.DataFrame) -> None:
    """Validate that strategy columns conform to integer identifier dtype."""
    strategy_cols = [c for c in df.columns if c == "winner_strategy" or _SEAT_RE.match(c)]
    for col in strategy_cols:
        if not pd.api.types.is_integer_dtype(df[col].dtype):
            LOGGER.error(
                "Strategy column has unexpected dtype",
                extra={
                    "stage": "ingest",
                    "column": col,
                    "dtype": str(df[col].dtype),
                },
            )
            raise RuntimeError("Strategy column dtype mismatch")


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
    src_mtime = source_manifest.stat().st_mtime
    if raw_out.exists() and src_mtime and raw_out.stat().st_mtime >= src_mtime:
        LOGGER.info(
            "Ingest block up-to-date",
            extra={"stage": "ingest", "n_players": n, "path": str(raw_out)},
        )
        return 0

    canon = expected_schema_for(n)
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
            shard_df = _coerce_strategy_ids(shard_df)
            _validate_strategy_dtypes(shard_df)
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
            for name in canon_names:
                if name not in shard_df.columns:
                    shard_df[name] = pd.NA
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
    run_streaming_shard(
        out_path=str(raw_out),
        manifest_path=str(manifest_path),
        schema=canon,
        batch_iter=_all_batches(),
        row_group_size=cfg.row_group_size,
        compression=cfg.parquet_codec,
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
    stage_policy = resolve_stage_parallel_policy("ingest", cfg.ingest)
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
    ):
        LOGGER.info(
            "Ingest up-to-date",
            extra={"stage": "ingest", "path": str(done)},
        )
        return

    mp_context = resolve_mp_context(cfg.analysis.mp_start_method)

    total_rows = 0
    if stage_policy.process_workers <= 1:
        for block in blocks:
            total_rows += _process_block(block, cfg, parent_process_workers=1)
    else:
        with ProcessPoolExecutor(
            max_workers=stage_policy.process_workers, mp_context=mp_context
        ) as executor:
            futures = [
                executor.submit(
                    _process_block,
                    block,
                    cfg,
                    parent_process_workers=stage_policy.process_workers,
                )
                for block in blocks
            ]
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
