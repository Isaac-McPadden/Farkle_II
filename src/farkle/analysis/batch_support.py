"""Shared rectangular deterministic-batch support validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from farkle.utils.strategy_ids import canonical_strategy_ids

RECTANGULAR_SUPPORT_POLICY: Final[str] = (
    "require_declared_strategy_by_batch_cartesian_support; "
    "exclude_zero-exposure cells from univariate estimates and "
    "zero-containing vectors from joint resampling"
)


@dataclass(frozen=True)
class RectangularBatchSupport:
    """Validated support counts for one root/k batch table."""

    strategy_count: int
    batch_count: int
    declared_cells: int
    positive_exposure_cells: int
    excluded_zero_exposure_cells: int


def validate_rectangular_batch_support(
    frame: pd.DataFrame,
    *,
    context: str,
    strategy_column: str = "strategy",
    batch_column: str = "deterministic_batch_id",
    exposure_column: str = "raw_player_game_exposures",
) -> RectangularBatchSupport:
    """Require exact strategy-by-batch rows and classify declared zero cells.

    A missing cell is malformed rectangular support and fails. A present row
    with zero exposure is distinct: it remains part of the declared Cartesian
    support but is explicitly excluded from rate and MCSE calculations.
    """

    required = {strategy_column, batch_column, exposure_column}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"{context} lacks batch-support columns: {missing_columns}")
    if frame.empty:
        raise ValueError(f"{context} contains no declared strategy/batch support")

    canonical_strategy_ids(
        frame[strategy_column],
        context=f"{context} {strategy_column}",
    )
    batches = frame[batch_column]
    if (
        not pd.api.types.is_integer_dtype(batches.dtype)
        or pd.api.types.is_bool_dtype(batches.dtype)
        or batches.isna().any()
    ):
        raise ValueError(f"{context} {batch_column} must be a non-null integer")
    exposures = frame[exposure_column]
    if exposures.isna().any() or (exposures < 0).any():
        raise ValueError(f"{context} contains negative or null exposure support")

    keys = [batch_column, strategy_column]
    if frame.duplicated(keys).any():
        duplicate = frame.loc[frame.duplicated(keys, keep=False), keys].iloc[0].to_dict()
        raise ValueError(f"{context} contains duplicate strategy/batch cell {duplicate}")

    strategies = sorted(int(value) for value in frame[strategy_column].unique())
    batch_ids = sorted(int(value) for value in batches.unique())
    observed = set(
        zip(
            frame[batch_column].astype(int),
            frame[strategy_column].astype(int),
            strict=True,
        )
    )
    expected = {(batch_id, strategy) for batch_id in batch_ids for strategy in strategies}
    missing_cells = sorted(expected.difference(observed))
    if missing_cells:
        raise ValueError(
            f"{context} is missing declared rectangular strategy/batch cells: "
            f"{missing_cells[:20]}"
        )

    zero_cells = int(exposures.eq(0).sum())
    return RectangularBatchSupport(
        strategy_count=len(strategies),
        batch_count=len(batch_ids),
        declared_cells=len(frame),
        positive_exposure_cells=len(frame) - zero_cells,
        excluded_zero_exposure_cells=zero_cells,
    )


__all__ = [
    "RECTANGULAR_SUPPORT_POLICY",
    "RectangularBatchSupport",
    "validate_rectangular_batch_support",
]
