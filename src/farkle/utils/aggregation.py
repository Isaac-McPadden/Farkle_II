# src/farkle/utils/aggregation.py
"""Reusable aggregation configuration helpers for analysis stages."""

from __future__ import annotations

from typing import Literal

KAggregationMethod = Literal["equal-k", "declared-mapping"]


def normalize_k_aggregation_method(aggregation_method: str) -> KAggregationMethod:
    """Validate and return an exact canonical player-count method name."""

    normalized = aggregation_method.strip().lower()
    if normalized in {"equal-k", "declared-mapping"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Unknown aggregation scheme: {aggregation_method!r}")


__all__ = ["KAggregationMethod", "normalize_k_aggregation_method"]
