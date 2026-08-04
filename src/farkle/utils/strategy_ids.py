"""Canonical strategy-identifier types and strict boundary validation."""

from __future__ import annotations

from numbers import Integral
from typing import Final

import pandas as pd
import pyarrow as pa

STRATEGY_ID_ARROW_TYPE: Final[pa.DataType] = pa.int32()
STRATEGY_ID_PANDAS_DTYPE: Final[pd.Int32Dtype] = pd.Int32Dtype()
STRATEGY_ID_MIN: Final[int] = 0
STRATEGY_ID_MAX: Final[int] = (1 << 31) - 1


def canonical_strategy_ids(
    series: pd.Series,
    *,
    nullable: bool = False,
    context: str = "strategy identifier",
) -> pd.Series:
    """Return canonical ``Int32`` IDs without parsing strings or floats.

    Artifact-boundary coercion is deliberately limited to lossless conversion
    between integer physical widths. Text, floating-point, boolean, mixed
    object, null (unless explicitly allowed), negative, and out-of-range values
    fail instead of being converted or dropped.
    """

    if series.empty:
        return pd.Series(
            pd.array([], dtype=STRATEGY_ID_PANDAS_DTYPE),
            index=series.index,
            name=series.name,
        )
    if series.dtype == object:
        present_values = series.dropna().tolist()
        if all(
            isinstance(value, Integral) and not isinstance(value, bool) for value in present_values
        ):
            series = pd.Series(
                pd.array(series.tolist(), dtype=STRATEGY_ID_PANDAS_DTYPE),
                index=series.index,
                name=series.name,
            )
    if not pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError(
            f"{context} must use a canonical integer logical type; found {series.dtype}"
        )
    if not nullable and series.isna().any():
        raise ValueError(f"{context} must be non-null")
    present = series.dropna()
    if not present.empty:
        minimum = int(present.min())
        maximum = int(present.max())
        if minimum < STRATEGY_ID_MIN or maximum > STRATEGY_ID_MAX:
            raise ValueError(
                f"{context} must be within "
                f"[{STRATEGY_ID_MIN}, {STRATEGY_ID_MAX}]; found [{minimum}, {maximum}]"
            )
    return series.astype(STRATEGY_ID_PANDAS_DTYPE)


def canonical_strategy_id(value: object, *, context: str = "strategy identifier") -> int:
    """Validate and return one canonical non-null strategy identifier."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{context} must be an integer; found {value!r}")
    strategy_id = int(value)
    if not STRATEGY_ID_MIN <= strategy_id <= STRATEGY_ID_MAX:
        raise ValueError(
            f"{context} must be within "
            f"[{STRATEGY_ID_MIN}, {STRATEGY_ID_MAX}]; found {strategy_id}"
        )
    return strategy_id


def require_strategy_id_field(
    schema: pa.Schema,
    column: str,
    *,
    nullable: bool | None = None,
    context: str = "artifact",
) -> None:
    """Require the canonical Arrow physical type and declared nullability."""

    index = schema.get_field_index(column)
    if index < 0:
        raise ValueError(f"{context} is missing canonical strategy-ID column {column!r}")
    field = schema.field(index)
    if field.type != STRATEGY_ID_ARROW_TYPE or (
        nullable is not None and field.nullable != nullable
    ):
        nullability = "any" if nullable is None else str(nullable)
        raise ValueError(
            f"{context} column {column!r} must be "
            f"{STRATEGY_ID_ARROW_TYPE} nullable={nullability}; found "
            f"{field.type} nullable={field.nullable}"
        )


__all__ = [
    "STRATEGY_ID_ARROW_TYPE",
    "STRATEGY_ID_MAX",
    "STRATEGY_ID_MIN",
    "STRATEGY_ID_PANDAS_DTYPE",
    "canonical_strategy_id",
    "canonical_strategy_ids",
    "require_strategy_id_field",
]
