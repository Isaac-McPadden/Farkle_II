"""Shared conversion helpers for analysis pipelines.

These helpers centralize boundary conversions (JSON/Parquet/Pandas/Numpy) so
analysis stages can stay vectorized and avoid per-row Python loops.
"""

from __future__ import annotations

import math
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

TierMap: TypeAlias = dict[str, int]


_NUMERIC_RUNTIME_TYPES = (int, float, np.integer, np.floating)


def is_na(x: object) -> bool:
    """Return whether a scalar value should be treated as missing/NA."""

    if x is None or x is pd.NA:
        return True
    na_result = pd.isna(x)
    if isinstance(na_result, bool):
        return na_result
    raise TypeError(
        f"pd.isna returned non-bool for scalar check: type={type(x).__name__}, value={x!r}"
    )


def as_float(x: object) -> float:
    """Convert a Python/Numpy numeric scalar to ``float`` with explicit NA rejection."""

    if is_na(x):
        raise ValueError(f"cannot convert NA value to float: {x!r}")
    if isinstance(x, (bool, np.bool_)):
        return float(int(x))
    if isinstance(x, _NUMERIC_RUNTIME_TYPES):
        return float(x)
    raise TypeError(f"expected numeric scalar for float conversion, got {type(x).__name__}")


def as_int(x: object) -> int:
    """Convert a Python/Numpy numeric scalar to ``int`` with explicit checks."""

    if is_na(x):
        raise ValueError(f"cannot convert NA value to int: {x!r}")
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        numeric = float(x)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        raise ValueError(
            f"cannot convert non-integral float to int: type={type(x).__name__}, value={x!r}"
        )
    raise TypeError(f"expected numeric scalar for int conversion, got {type(x).__name__}")


def try_to_int(value: Any) -> int | None:
    """Best-effort integer conversion for boundary data.

    Returns ``None`` when conversion is not possible.
    """

    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001 - defensive for odd scalar types
        pass

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except (TypeError, ValueError):
            return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int:
    """Convert scalar groupby/config values to ``int`` for boundary coercion only."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        raise ValueError(
            f"cannot convert non-integral float to int: type={type(value).__name__}, value={value!r}"
        )
    raise ValueError(f"cannot convert value to int: type={type(value).__name__}, value={value!r}")


def to_stat_value(x: Any) -> int | float | str | None:
    """Convert analysis scalars into JSON-safe builtin values."""

    if x is None or x is pd.NA:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:  # noqa: BLE001 - defensive for odd scalar types
        pass

    if isinstance(x, np.generic):
        x = x.item()

    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, str):
        return x
    return str(x)


def tiers_to_map(tier_lists: list[list[str]]) -> TierMap:
    """Convert ordered tier groups into ``{strategy: tier_rank}``.

    Tier rank is deterministic and based on outer-list position (1-indexed).
    Duplicate strategies across tier groups are rejected.
    """

    tiers: TierMap = {}
    for tier_rank, members in enumerate(tier_lists, start=1):
        for strategy in members:
            strategy_key = str(strategy)
            if strategy_key in tiers:
                raise ValueError(f"duplicate strategy in tier lists: {strategy_key}")
            tiers[strategy_key] = tier_rank
    return tiers


__all__ = [
    "TierMap",
    "as_float",
    "as_int",
    "is_na",
    "tiers_to_map",
    "to_int",
    "to_stat_value",
    "try_to_int",
]
