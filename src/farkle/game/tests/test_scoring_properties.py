from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from farkle.game.scoring import faces_to_counts_tuple, score_roll_cached


@pytest.mark.unit
@given(st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=6))
def test_score_roll_cached_is_permutation_invariant(roll: list[int]) -> None:
    reversed_roll = list(reversed(roll))
    score_a = score_roll_cached(tuple(roll))
    score_b = score_roll_cached(tuple(reversed_roll))
    assert score_a == score_b


@pytest.mark.unit
@given(st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=6))
def test_faces_to_counts_total_matches_roll_length(faces: list[int]) -> None:
    counts = faces_to_counts_tuple(faces)
    assert sum(counts) == len(faces)
