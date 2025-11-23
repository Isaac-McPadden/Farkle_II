from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence

# tests/helpers/golden_utils.py
"""Utilities for comparing generated artifacts to stored goldens."""


import pandas as pd


class GoldenMismatchError(AssertionError):
    """Raised when a golden artifact differs and regeneration is disabled."""


_MESSAGE = "Run tests with --update-goldens to refresh stored artifacts."


def _normalize_frame(df: pd.DataFrame, sort_by: Sequence[str] | None) -> pd.DataFrame:
    """Sort and reset a DataFrame to enable stable comparisons.

    Args:
        df: DataFrame being normalized.
        sort_by: Optional column names to sort by before comparison.

    Returns:
        Normalized DataFrame suitable for equality checks.
    """

    if sort_by:
        df = df.sort_values(list(sort_by)).reset_index(drop=True)
    return df


def assert_csv_golden(
    actual: Path,
    golden: Path,
    *,
    update: bool,
    sort_by: Sequence[str] | None = None,
) -> None:
    """Compare a CSV file against a golden, optionally refreshing it."""

    _assert_frame_golden(actual, golden, update=update, sort_by=sort_by, loader=pd.read_csv)


def assert_parquet_golden(
    actual: Path,
    golden: Path,
    *,
    update: bool,
    sort_by: Sequence[str] | None = None,
) -> None:
    """Compare a parquet-backed DataFrame to a CSV golden.

    Golden tables are stored in CSV format to keep the repository free of
    binary blobs, but the actual artifact under test can remain parquet.
    """

    _assert_frame_golden(actual, golden, update=update, sort_by=sort_by, loader=pd.read_parquet)


def copy_golden(actual: Path, golden: Path, *, update: bool) -> None:
    """Copy a non-tabular artifact when updates are allowed."""

    if update:
        golden.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actual, golden)
        return

    if not golden.exists():
        raise GoldenMismatchError(f"Missing golden {golden}. {_MESSAGE}")

    if actual.read_bytes() != golden.read_bytes():
        raise GoldenMismatchError(f"Golden mismatch for {golden}. {_MESSAGE}")


def _assert_frame_golden(
    actual: Path,
    golden: Path,
    *,
    update: bool,
    sort_by: Sequence[str] | None,
    loader,
) -> None:
    """Compare a tabular artifact to a stored golden copy.

    Args:
        actual: Path to the DataFrame produced by the test.
        golden: Path to the corresponding golden CSV artifact.
        update: Whether the golden should be rewritten from ``actual``.
        sort_by: Optional column names used to normalize ordering.
        loader: Callable that reads ``actual`` into a pandas DataFrame.

    Returns:
        None
    """

    actual_df = loader(actual)
    actual_df = _normalize_frame(actual_df, sort_by)

    if update:
        golden.parent.mkdir(parents=True, exist_ok=True)
        actual_df.to_csv(golden, index=False)
        return

    if not golden.exists():
        raise GoldenMismatchError(f"Missing golden {golden}. {_MESSAGE}")

    golden_df = pd.read_csv(golden)
    golden_df = _normalize_frame(golden_df, sort_by)
    golden_df = golden_df.astype(
        {col: dtype for col, dtype in actual_df.dtypes.items() if col in golden_df.columns}
    )
    try:
        pd.testing.assert_frame_equal(actual_df, golden_df, check_like=True)
    except AssertionError as exc:  # pragma: no cover - assertion branch exercised in tests
        raise GoldenMismatchError(f"CSV golden mismatch for {golden}. {_MESSAGE}") from exc


def assert_stamp_has_paths(
    stamp_path: Path, expected_inputs: Iterable[Path], expected_outputs: Iterable[Path]
) -> None:
    """Validate that the metrics stamp records the required inputs/outputs."""

    payload = json.loads(stamp_path.read_text())
    input_keys = set(payload.get("inputs", {}).keys())
    output_keys = set(payload.get("outputs", {}).keys())
    for path in expected_inputs:
        assert str(path) in input_keys, f"Missing input stamp for {path}"
    for path in expected_outputs:
        assert str(path) in output_keys, f"Missing output stamp for {path}"
