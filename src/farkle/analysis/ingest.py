# src/farkle/analysis/ingest.py
"""Ingest raw simulation results into parquet shards for curation.

This entry point streams over experiment outputs, validates schemas, and
writes player-count-specific shards that feed the downstream combine and
metrics stages.
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig, load_app_config
from farkle.utils.schema_helpers import expected_schema_for
from farkle.utils.streaming_loop import run_streaming_shard

LOGGER = logging.getLogger(__name__)


def _iter_shards(block: Path, cols: tuple[str, ...]):
    """
    Yield one ``(DataFrame, source_path)`` per shard, selecting only the
    *intersection* of *cols* with the shard’s schema.

    Parquet will raise if we request a column that doesn’t exist, so read the
    whole shard once and trim afterwards.  The ingest pipeline later pads any
    truly-missing columns back in :func:`_coerce_schema`, keeping the final
    table rectangular.
    """

    def _read_subset(path: Path, wanted: Iterable[str]) -> pd.DataFrame:
        """Read *path* and return only the columns present in *wanted*."""

        if path.suffix == ".csv":
            df = pd.read_csv(path, dtype_backend="pyarrow")
            wanted = list(wanted)
            present = [c for c in wanted if c in df.columns]
            if LOGGER.isEnabledFor(logging.DEBUG) and len(present) < len(wanted):
                missing = sorted(set(wanted) - set(present))
                LOGGER.debug(
                    "Shard missing requested columns",
                    extra={
                        "stage": "ingest",
                        "path": path.name,
                        "missing": missing,
                    },
                )
            return df[present]
        else:
            pf = pq.ParquetFile(path)
            wanted = list(wanted)
            present = [c for c in wanted if c in pf.schema_arrow.names]
            if LOGGER.isEnabledFor(logging.DEBUG) and len(present) < len(wanted):
                missing = sorted(set(wanted) - set(present))
                LOGGER.debug(
                    "Shard missing requested columns",
                    extra={
                        "stage": "ingest",
                        "path": path.name,
                        "missing": missing,
                    },
                )
            return pf.read(columns=present).to_pandas()

    # Newer runs emit a single `<Np_rows>.parquet` file directly under the
    # block directory. Prefer that if present. Sorting ensures deterministic
    # selection should multiple consolidated files coexist.
    row_files = sorted(block.glob("*p_rows.parquet"))
    row_file = row_files[0] if row_files else None
    if row_file is not None:
        # Stream by row-group to keep memory bounded
        pf = pq.ParquetFile(row_file)
        # Determine which of the wanted columns are present once, from schema
        wanted = list(cols)
        present = [c for c in wanted if c in pf.schema_arrow.names]
        missing = sorted(set(wanted) - set(present))
        if LOGGER.isEnabledFor(logging.DEBUG) and missing:
            LOGGER.debug(
                "Row file missing requested columns",
                extra={
                    "stage": "ingest",
                    "path": row_file.name,
                    "missing": missing,
                },
            )
        for i in range(pf.num_row_groups):
            tbl = pf.read_row_group(i, columns=present)
            # Convert this row-group only
            df = tbl.to_pandas()
            yield df, row_file
        return

    # Legacy layout with `<Np_rows>` directory containing shards
    row_dirs = [p for p in block.glob("*_rows") if p.is_dir()]
    if row_dirs:
        for shard_path in sorted(row_dirs[0].glob("*.parquet")):
            yield _read_subset(shard_path, cols), shard_path
    else:  # compact parquet or CSV
        for pqt in block.glob("*.parquet"):
            yield _read_subset(pqt, cols), pqt
        csv = block / "winners.csv"
        if csv.exists():
            yield _read_subset(csv, cols), csv


# Regex once, reuse
_SEAT_RE = re.compile(r"^P(\d+)_strategy$")


def _fix_winner(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize winner-related columns for compatibility with downstream code.

    Args:
        df: Raw results dataframe containing winner and seat strategy columns.

    Returns:
        A copy of the dataframe with standardized ``winner_seat``,
        ``winner_strategy``, and ``seat_ranks`` columns.
    """
    df = df.copy()

    # Rename legacy "winner" column to "winner_seat" (drop original)
    if "winner" in df.columns:
        if "winner_seat" in df.columns:
            df.drop(columns=["winner"], inplace=True)
        else:
            df.rename(columns={"winner": "winner_seat"}, inplace=True)

    # strategy seat columns (P1_strategy, …)
    seat_cols = sorted(
        [c for c in df.columns if _SEAT_RE.match(c)],
        key=lambda c: int(_SEAT_RE.match(c).group(1)),  # type: ignore
    )

    # winner_strategy as plain strings (add if missing)
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

    df = _normalize_strategy_columns(df)
    return df


def _normalize_strategy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all strategy columns are normalized to string dtype."""
    df = df.copy()
    strategy_cols = [
        c for c in df.columns if c == "winner_strategy" or _SEAT_RE.match(c)
    ]
    for col in strategy_cols:
        series = df[col]
        non_string_mask = series.notna() & ~series.apply(lambda v: isinstance(v, str))
        if non_string_mask.any():
            sample = series[non_string_mask].iloc[0]
            LOGGER.debug(
                "Non-string strategy value detected",
                extra={
                    "stage": "ingest",
                    "column": col,
                    "sample_type": type(sample).__name__,
                },
            )
        normalized = series.map(lambda v: str(v) if pd.notna(v) else pd.NA)
        df[col] = normalized.astype("string")
    return df


def _n_from_block(name: str) -> int:
    """Extract the player count from a ``<N>_players`` directory name.

    Args:
        name: Directory basename encoded with the player count.

    Returns:
        Parsed number of players, or ``0`` when the name does not follow the
        expected pattern.
    """
    m = re.match(r"^(\d+)_players$", name)
    return int(m.group(1)) if m else 0


def _migrate_legacy_raw(n: int, cfg: AppConfig) -> None:
    """Move legacy ingest outputs into the new ``00_ingest/<k>p`` layout."""

    new_raw = cfg.ingested_rows_raw(n)
    legacy_raw = cfg.combine_stage_dir / f"{n}p" / f"{n}p_ingested_rows.raw.parquet"
    if new_raw.exists() or not legacy_raw.exists() or new_raw == legacy_raw:
        return

    new_raw.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Migrating legacy ingest output",
        extra={
            "stage": "ingest",
            "n_players": n,
            "source": str(legacy_raw),
            "dest": str(new_raw),
        },
    )
    shutil.move(str(legacy_raw), new_raw)

    legacy_manifest = legacy_raw.with_suffix(".manifest.jsonl")
    new_manifest = cfg.ingest_manifest(n)
    if legacy_manifest.exists():
        new_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_manifest), new_manifest)


def _process_block(block: Path, cfg: AppConfig) -> int:
    """Process a single ``<N>_players`` block."""
    n = _n_from_block(block.name)
    LOGGER.info(
        "Ingest block discovered",
        extra={"stage": "ingest", "block": block.name, "path": str(block)},
    )

    _migrate_legacy_raw(n, cfg)
    raw_out = cfg.ingested_rows_raw(n)
    src_mtime = 0.0
    row_files = sorted(block.glob("*p_rows.parquet"))
    if row_files:
        src_mtime = row_files[0].stat().st_mtime
    else:
        row_dirs = [p for p in block.glob("*_rows") if p.is_dir()]
        if row_dirs:
            shards = list(row_dirs[0].glob("*.parquet"))
            if shards:
                src_mtime = max(s.stat().st_mtime for s in shards)
    if raw_out.exists() and src_mtime and raw_out.stat().st_mtime >= src_mtime:
        LOGGER.info(
            "Ingest block up-to-date",
            extra={"stage": "ingest", "n_players": n, "path": str(raw_out)},
        )
        return 0

    canon = expected_schema_for(n)
    seat_cols = [c for c in canon.names if c.startswith("P")]
    wanted = ("winner", "n_rounds", "winning_score", *seat_cols)

    total = 0

    def _iter_tables():
        """Yield canonicalized parquet tables from discovered shards.

        Returns:
            An iterator over :class:`pyarrow.Table` objects aligned to the
            expected schema for the current player count.
        """
        nonlocal total
        for shard_df, shard_path in _iter_shards(block, tuple(wanted)):
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
    LOGGER.info(
        "Ingest started",
        extra={
            "stage": "ingest",
            "root": str(cfg.results_root),
            "data_dir": str(cfg.data_dir),
            "n_jobs": cfg.n_jobs_ingest,
        },
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    blocks = sorted(
        (p for p in cfg.results_root.iterdir() if p.is_dir() and p.name.endswith("_players")),
        key=lambda p: (_n_from_block(p.name), p.name),
    )

    done = stage_done_path(cfg.ingest_stage_dir, "ingest")
    outputs = []
    manifests = []
    for block in blocks:
        n = _n_from_block(block.name)
        outputs.append(cfg.ingested_rows_raw(n))
        manifests.append(cfg.ingest_manifest(n))
    if stage_is_up_to_date(
        done,
        inputs=[cfg.results_root],
        outputs=[*outputs, *manifests],
        config_sha=getattr(cfg, "config_sha", None),
    ):
        LOGGER.info(
            "Ingest up-to-date",
            extra={"stage": "ingest", "path": str(done)},
        )
        return

    total_rows = 0
    if cfg.n_jobs_ingest <= 1:
        for block in blocks:
            total_rows += _process_block(block, cfg)
    else:
        with ProcessPoolExecutor(max_workers=cfg.n_jobs_ingest) as pool:
            futures = [pool.submit(_process_block, block, cfg) for block in blocks]
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
        inputs=[cfg.results_root],
        outputs=[*outputs, *manifests],
        config_sha=getattr(cfg, "config_sha", None),
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
