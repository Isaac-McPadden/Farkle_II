# src/farkle/ingest.py
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis_config import (
    PipelineCfg,
    expected_schema_for,
    n_players_from_schema,
)

log = logging.getLogger(__name__)


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
        """Read *path* and return only the columns present in *wanted*.

        Pandas' ``usecols`` parameter will raise if any requested column is
        missing.  To provide a more forgiving interface, load the file normally
        and then select the intersection of the desired and available columns.
        ``wanted`` is typically ``cfg.ingest_cols`` which may include seats not
        present in legacy results blocks (e.g., requesting ``P12_strategy`` when
        only two players were recorded).
        """

        if path.suffix == ".csv":
            df = pd.read_csv(path, dtype_backend="pyarrow")
        else:
            df = pd.read_parquet(path)  # read all columns first

        present = [c for c in wanted if c in df.columns]
        if len(present) < wanted.__sizeof__():  # debug noise only
            missing = set(wanted) - set(present)
            log.debug("%s missing cols: %s", path.name, sorted(missing))
        return df[present]

    # Newer runs emit a single `<Np_rows>.parquet` file directly under the
    # block directory. Prefer that if present. Sorting ensures deterministic
    # selection should multiple consolidated files coexist.
    row_files = sorted(block.glob("*p_rows.parquet"))
    row_file = row_files[0] if row_files else None
    if row_file is not None:
        yield _read_subset(row_file, cols), row_file
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
    """
    Enrich *df* with:
    • winner_strategy - the strategy string for the winning seat
    • winner_seat     - seat label (P1 … PN)
    • seat_ranks      - tuple of seat labels ordered by rank (winner first)

    Leaves all original columns intact so caller can `drop()` later.
    """
    if "winner_strategy" in df.columns:
        # Already processed — nothing to do.
        return df

    df = df.copy()  # cheap shallow copy
    seat_cols = sorted(
        [c for c in df.columns if _SEAT_RE.match(c)],
        key=lambda c: int(_SEAT_RE.match(c).group(1)),  # natural seat order  # type: ignore
    )

    # --- Step 1: find seat label (P#) that won --------------------------
    df["winner_seat"] = df["winner"]  # winner column still holds P#

    # --- Step 2: promote strategy string -------------------------------
    seat_idx = df["winner_seat"].str.extract(r"P(?P<num>\d+)")["num"].astype(int) - 1
    df["winner_strategy"] = df[seat_cols].to_numpy()[np.arange(len(df)), seat_idx.to_numpy()]

    # --- Step 3: capture finishing order -------------------------------
    rank_cols = [c.replace("_strategy", "_rank") for c in seat_cols]
    if all(col in df.columns for col in rank_cols):
        ranks = df[rank_cols].to_numpy()
        seat_labels = np.array([c[:-9] for c in seat_cols])
        df["seat_ranks"] = list(map(tuple, seat_labels[ranks.argsort(axis=1)]))

    return df


def _coerce_schema(tbl: pa.Table, target: pa.Schema | None = None) -> pa.Table:
    """Pad/cast *tbl* to the supplied *target* schema (or infer per-N if None)."""
    if target is None:
        target = expected_schema_for(n_players_from_schema(tbl.schema))
    arrays = []
    for field in target: # type: ignore
        if field.name in tbl.column_names:
            arrays.append(tbl[field.name].cast(field.type))
        else:
            arrays.append(pa.nulls(len(tbl), field.type))
    return pa.Table.from_arrays(arrays, names=target.names) # type: ignore



def run(cfg: PipelineCfg) -> None:
    """Consolidate raw results blocks into parquet files per player-count."""

    log.info("Ingest started: root=%s", cfg.results_dir)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[int, pq.ParquetWriter] = {}
    tmp_paths: dict[int, Path] = {}
    totals: dict[int, int] = {}

    try:
        for block in sorted(cfg.results_dir.glob(cfg.results_glob)):
            log.info("Reading block %s", block.name)
            expected_cols = set(cfg.ingest_cols)
            for shard_df, shard_path in _iter_shards(block, cfg.ingest_cols):
                if shard_df.empty:
                    log.debug("Shard %s is empty — skipped", shard_path.name)
                    continue

                if not set(shard_df.columns).issubset(expected_cols):
                    log.error("Schema mismatch in %s", shard_path)
                    raise RuntimeError("Shard DataFrame columns do not match expected columns")

                log.debug("Shard %s → %d rows", shard_path.name, len(shard_df))

                shard_df = _fix_winner(shard_df)

                table = pa.Table.from_pandas(shard_df, preserve_index=False)
                table = _coerce_schema(table)
                n_players = n_players_from_schema(table.schema)

                if n_players not in writers:
                    raw_path = cfg.ingested_rows_raw(n_players)
                    tmp_path = raw_path.with_suffix(raw_path.suffix + ".in-progress")
                    if tmp_path.exists():
                        tmp_path.unlink()
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    writers[n_players] = pq.ParquetWriter(
                        tmp_path, table.schema, compression=cfg.parquet_codec
                    )
                    tmp_paths[n_players] = tmp_path
                    totals[n_players] = 0

                writers[n_players].write_table(table, row_group_size=cfg.row_group_size)
                totals[n_players] += len(shard_df)
    finally:
        for w in writers.values():
            w.close()

    for n_players, tmp in tmp_paths.items():
        raw_path = cfg.ingested_rows_raw(n_players)
        if tmp.exists():
            tmp.replace(raw_path)
        log.info(
            "Ingest finished — %d rows written to %s", totals.get(n_players, 0), raw_path
        )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI wrapper
    cfg, _, _ = PipelineCfg.parse_cli(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
