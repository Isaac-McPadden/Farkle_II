from __future__ import annotations

import pytest

from farkle.utils.aggregation import normalize_k_aggregation_method


@pytest.mark.parametrize("method", ["equal-k", "declared-mapping"])
def test_normalize_k_aggregation_method_accepts_exact_contract(method: str) -> None:
    assert normalize_k_aggregation_method(method) == method


@pytest.mark.parametrize("method", ["equal_k", "custom", "game-count", "mystery"])
def test_normalize_k_aggregation_method_rejects_aliases(method: str) -> None:
    with pytest.raises(ValueError, match="Unknown aggregation scheme"):
        normalize_k_aggregation_method(method)
