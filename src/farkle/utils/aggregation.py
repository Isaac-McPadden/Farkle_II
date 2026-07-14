# src/farkle/utils/aggregation.py
"""Reusable aggregation configuration helpers for analysis stages."""

from __future__ import annotations

from typing import Literal

KAggregationMethod = Literal["game-count", "equal-k", "config"]


def normalize_k_aggregation_method(aggregation_method: str) -> KAggregationMethod:
    """Normalize user-provided aggregation aliases into canonical values."""

    normalized = aggregation_method.strip().lower().replace("_", "-")
    if normalized in {"game-count", "gamecount", "count"}:
        return "game-count"
    if normalized in {"equal-k", "equalk", "equal"}:
        return "equal-k"
    if normalized in {"config", "config-provided", "custom"}:
        return "config"
    raise ValueError(f"Unknown aggregation scheme: {aggregation_method!r}")


__all__ = ["KAggregationMethod", "normalize_k_aggregation_method"]
