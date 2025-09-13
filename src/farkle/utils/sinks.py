"""Output sinks and helpers for simulation/analysis results."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# ------------------------
# CSV sink (stream-friendly)
# ------------------------

class CsvSink:
    """Append dictionaries to a CSV file.

    Creates parent dirs. Call open()/write_row[s]()/close().
    """
    def __init__(self, path: str | Path, header: Sequence[str], mode: str = "w"):
        self.path = Path(path)
        self.header = list(header)
        self.mode = mode
        self._writer: csv.DictWriter | None = None
        self._file = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, self.mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.header)
        if "w" in self.mode:
            self._writer.writeheader()

    def close(self) -> None:  # pragma: no cover - trivial
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

    def write_row(self, row: Mapping[str, object]) -> None:
        assert self._writer is not None, "CsvSink not opened"
        self._writer.writerow(row)

    def write_rows(self, rows: Iterable[Mapping[str, object]]) -> None:
        assert self._writer is not None, "CsvSink not opened"
        self._writer.writerows(rows)

# ------------------------
# Simple helpers for aggregates
# ------------------------

def write_counter_csv(counter: Counter[str], path: Path) -> None:
    """Write a Counter[str] to CSV as columns ``strategy,wins``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "wins"])
        for key, cnt in counter.items():
            w.writerow([key, cnt])

def write_parquet_rows(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    """Write an iterable of dicts to a parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Local import lets environments without pyarrow still run CSV-only flows.
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path)

__all__ = ["CsvSink", "write_counter_csv", "write_parquet_rows"]
