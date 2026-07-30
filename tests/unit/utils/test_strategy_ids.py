from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from farkle.utils.strategy_ids import (
    STRATEGY_ID_ARROW_TYPE,
    canonical_strategy_id,
    canonical_strategy_ids,
    require_strategy_id_field,
)


def test_canonical_strategy_ids_losslessly_narrow_integer_width() -> None:
    values = pd.Series([0, 7, 42], dtype="int64")

    result = canonical_strategy_ids(values)

    assert str(result.dtype) == "Int32"
    assert result.tolist() == [0, 7, 42]
    assert pa.int32() == STRATEGY_ID_ARROW_TYPE


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(["1", "2"], dtype="string"),
        pd.Series([1.0, 2.0], dtype="float64"),
        pd.Series([1, "2"], dtype="object"),
        pd.Series([True, False], dtype="bool"),
        pd.Series([-1, 2], dtype="int64"),
        pd.Series([0, 2**31], dtype="int64"),
        pd.Series([1, None], dtype="Int32"),
    ],
)
def test_canonical_strategy_ids_reject_malformed_or_null_boundaries(
    values: pd.Series,
) -> None:
    with pytest.raises(ValueError):
        canonical_strategy_ids(values)


def test_nullable_strategy_id_is_only_allowed_when_explicit() -> None:
    values = pd.Series([1, None], dtype="Int32")

    result = canonical_strategy_ids(values, nullable=True)

    assert result.tolist() == [1, pd.NA]


@pytest.mark.parametrize("value", [True, "1", 1.0, -1, 2**31])
def test_canonical_strategy_id_scalar_rejects_noncanonical_values(value: object) -> None:
    with pytest.raises(ValueError):
        canonical_strategy_id(value)


def test_canonical_arrow_strategy_field_requires_int32_and_nullability() -> None:
    require_strategy_id_field(
        pa.schema([pa.field("strategy", pa.int32(), nullable=False)]),
        "strategy",
        nullable=False,
    )

    with pytest.raises(ValueError, match="int32"):
        require_strategy_id_field(
            pa.schema([pa.field("strategy", pa.int64(), nullable=False)]),
            "strategy",
            nullable=False,
        )
    with pytest.raises(ValueError, match="nullable=False"):
        require_strategy_id_field(
            pa.schema([pa.field("strategy", pa.int32(), nullable=True)]),
            "strategy",
            nullable=False,
        )
