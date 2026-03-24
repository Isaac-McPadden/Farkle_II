# src/farkle/utils/pooling.py
"""Reusable pooling configuration helpers for analysis stages."""

from __future__ import annotations

from typing import Literal

PoolingScheme = Literal["game-count", "equal-k", "config"]


def normalize_pooling_scheme(pooling_scheme: str) -> PoolingScheme:
    """Normalize user-provided pooling aliases into canonical values."""

    normalized = pooling_scheme.strip().lower().replace("_", "-")
    if normalized in {"game-count", "gamecount", "count"}:
        return "game-count"
    if normalized in {"equal-k", "equalk", "equal"}:
        return "equal-k"
    if normalized in {"config", "config-provided", "custom"}:
        return "config"
    raise ValueError(f"Unknown pooling scheme: {pooling_scheme!r}")


__all__ = ["PoolingScheme", "normalize_pooling_scheme"]
