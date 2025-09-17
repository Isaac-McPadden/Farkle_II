from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .writer import atomic_path


def write_parquet_atomic(table: pa.Table, path: Union[Path, str], *, codec: str = "snappy") -> None:
    """Write *table* to *path* atomically using Parquet compression."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(final_path)) as tmp_path:
        pq.write_table(table, tmp_path, compression=codec)


def write_csv_atomic(df: pd.DataFrame, path: Union[Path, str]) -> None:
    """Write *df* to *path* atomically as UTF-8 CSV without index."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(final_path)) as tmp_path, \
        Path(tmp_path).open("w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=False)
