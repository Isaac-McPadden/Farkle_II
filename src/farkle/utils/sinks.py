"""Output *sink* abstractions.

Simulation and analysis pipelines often need to stream dictionaries of
metrics somewhere.  A *sink* is a minimal object with ``write_row`` and
``close`` methods.  Only a very small CSV implementation is provided
for now which covers the use cases in the tests.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class CsvSink:
    """Append dictionaries to a CSV file.

    Parameters
    ----------
    path:
        Destination file.  Parent directories are created automatically.
    header:
        Column names written as the first row when the sink is opened.
    mode:
        File open mode.  Defaults to ``"w"``.
    """

    def __init__(self, path: str | Path, header: Sequence[str], mode: str = "w"):
        self.path = Path(path)
        self.header = list(header)
        self.mode = mode
        self._fh: csv.DictWriter | None = None
        self._file = None

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------
    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, self.mode, newline="")
        self._fh = csv.DictWriter(self._file, fieldnames=self.header)
        if "w" in self.mode:
            self._fh.writeheader()

    def close(self) -> None:  # pragma: no cover - trivial
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
            self._fh = None

    # ------------------------------------------------------------------
    # Writing utilities
    # ------------------------------------------------------------------
    def write_row(self, row: Mapping[str, object]) -> None:
        assert self._fh is not None, "sink not opened"
        self._fh.writerow(row)

    def write_rows(self, rows: Iterable[Mapping[str, object]]) -> None:
        assert self._fh is not None, "sink not opened"
        self._fh.writerows(rows)


__all__ = ["CsvSink"]

