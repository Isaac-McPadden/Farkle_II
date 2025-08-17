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
            df = pd.read_parquet(path)

        wanted = list(wanted)
        present = [c for c in wanted if c in df.columns]
        if log.isEnabledFor(logging.DEBUG) and len(present) < len(wanted):
            missing = sorted(set(wanted) - set(present))
            log.debug("%s missing cols: %s", path.name, missing)
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
    df = df.copy()

    # winner_seat mirrors winner (add if missing)
    if "winner_seat" not in df.columns and "winner" in df.columns:
        df["winner_seat"] = df["winner"]

    # strategy seat columns (P1_strategy, …)
    seat_cols = sorted(
        [c for c in df.columns if _SEAT_RE.match(c)],
        key=lambda c: int(_SEAT_RE.match(c).group(1)),  # type: ignore[arg-type]
    )

    # winner_strategy as plain strings (add if missing)
    if "winner_strategy" not in df.columns and seat_cols:
        seat_idx = (
            df["winner_seat"].str.extract(r"P(?P<num>\d+)", expand=True)["num"]
            .astype("Int64")  # robust to missing
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


def _n_from_block(name: str) -> int:
    m = re.match(r"^(\d+)_players$", name)
    return int(m.group(1)) if m else 0


def run(cfg: PipelineCfg) -> None:
    log.info("Ingest started: root=%s", cfg.results_dir)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[int, pq.ParquetWriter] = {}
    tmp_paths: dict[int, Path] = {}
    totals: dict[int, int] = {}

    blocks = sorted(
        (p for p in cfg.results_dir.iterdir()
         if p.is_dir() and p.name.endswith("_players")),
        key=lambda p: _n_from_block(p.name)
    )

    try:
        for block in blocks:
            log.info("Reading block %s", block.name)
            n = _n_from_block(block.name)

            # Per-block allow-list: base + all P1..Pn columns from canonical schema
            canon = expected_schema_for(n)
            seat_cols = [c for c in canon.names if c.startswith("P")]
            wanted = ("winner", "n_rounds", "winning_score", *seat_cols)

            for shard_df, shard_path in _iter_shards(block, tuple(wanted)):
                if shard_df.empty:
                    log.debug("Shard %s is empty — skipped", shard_path.name)
                    continue

                log.debug("Shard %s → %d rows", shard_path.name, len(shard_df))

                shard_df = _fix_winner(shard_df)         # fills winner_* + seat_ranks

                canon = expected_schema_for(n)                   # you already computed 'n' above
                canon_names = canon.names

                # 1) ensure every canonical column exists (fill with NA)
                for name in canon_names:
                    if name not in shard_df.columns:
                        shard_df[name] = pd.NA

                # 2) reorder to canonical order and build Arrow table in one go
                shard_df = shard_df[canon_names]
                table = pa.Table.from_pandas(shard_df, schema=canon, preserve_index=False)
                n_players = n  # you already know it from the block name

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
        log.info("Ingest finished — %d rows written to %s",
                 totals.get(n_players, 0), raw_path)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI wrapper
    cfg, _, _ = PipelineCfg.parse_cli(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
