"""Output helpers for simulation results.

The :mod:`farkle.simulation.runner` uses these functions to persist
aggregated results.  The helpers keep all file system interactions in a
single place which simplifies testing.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


def write_counter_csv(counter: Counter[str], path: Path) -> None:
    """Write ``counter`` to ``path`` as ``strategy,wins`` CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["strategy", "wins"])
        for strat, wins in counter.items():
            writer.writerow([strat, wins])


def write_parquet_rows(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    """Write a sequence of dictionaries to ``path`` as a parquet file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path)
