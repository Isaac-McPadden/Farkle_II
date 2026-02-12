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


def to_int(value: Any) -> int | None:
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


__all__ = ["TierMap", "tiers_to_map", "to_int", "to_stat_value"]
