# src/farkle/utils/sinks.py
"""Output sinks and helpers for simulation/analysis results."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .writer import atomic_path

# ------------------------
# CSV sink (stream-friendly)
# ------------------------


class CsvSink:
    """Append dictionaries to a CSV file.

    Creates parent dirs. Use as a context manager or call open()/write_row[s]()/close().
    """

    def __init__(self, path: str | Path, header: Sequence[str], mode: str = "w"):
        self.path = Path(path)
        self.header = list(header)
        self.mode = mode
        self._writer: csv.DictWriter | None = None
        self._file = None  # type: ignore[assignment]
        self._atomic_cm = None

    # --- context manager hooks ---
    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close(exc_type, exc, tb)
        # return False to propagate exceptions (typical for I/O contexts)
        return False

    # --- lifecycle ---
    def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ctx = nullcontext(str(self.path))
        if "w" in self.mode and "a" not in self.mode:
            ctx = atomic_path(str(self.path))
        tmp_path = ctx.__enter__()
        try:
            self._file = open(tmp_path, self.mode, newline="", encoding="utf-8")  # noqa:SIM115
        except Exception:
            ctx.__exit__(*sys.exc_info())
            raise
        self._atomic_cm = ctx
        self._writer = csv.DictWriter(self._file, fieldnames=self.header)
        if "w" in self.mode:
            self._writer.writeheader()
        return self  # allow `with CsvSink(...) as sink:`

    def close(self, exc_type=None, exc=None, tb=None):  # pragma: no cover - trivial
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
        if self._atomic_cm is not None:
            self._atomic_cm.__exit__(exc_type, exc, tb)
            self._atomic_cm = None

    # --- writes ---
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
    with (
        atomic_path(str(path)) as tmp_path,
        Path(tmp_path).open("w", newline="", encoding="utf-8") as fh,
    ):
        w = csv.writer(fh)
        w.writerow(["strategy", "wins"])
        for key, cnt in counter.items():
            w.writerow([key, cnt])


__all__ = ["CsvSink", "write_counter_csv"]
