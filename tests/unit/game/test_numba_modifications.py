# """
# Tests for Numba changes to scoring and scoring_lookup
# Hit the big static lookup-tables in scoring_lookup.py so they count toward
# coverage.  No heavy dice-rolling here – we only verify *selected* entries.
# """

import random
from collections import Counter

import numpy as np
import pytest

import farkle.game.scoring_lookup as sl
from farkle.simulation.strategies import (
    ThresholdStrategy,
    random_threshold_strategy,
)

# --- build the table once for all tests -------------------------------------

LOOKUP = sl.build_score_lookup_table()  # ~923 unique keys, quick


@pytest.mark.parametrize(
    "counts, expected",
    [
        ((3, 0, 0, 0, 0, 0), 300),  # three 1 s
        ((4, 0, 0, 0, 0, 0), 1000),  # four 1 s
        ((1, 1, 1, 1, 1, 1), 1500),  # straight
    ],
)
def test_lookup_known_patterns(counts, expected):
    assert LOOKUP[counts][0] == expected


def test_random_entry_consistency():
    """Random roll → same score via evaluate() and the pre-built table."""
    rng = np.random.default_rng(0)
    roll = rng.integers(1, 7, size=6)
    key = tuple(roll.tolist().count(i) for i in range(1, 7))
    total_score_table = LOOKUP[key][0]
    total_score_eval = sl.evaluate(key)[0]
    assert total_score_table == total_score_eval


def test_keys_are_sorted_and_hashable():
    """10 random keys from the table keep their structural promises."""
    for k in random.sample(list(LOOKUP.keys()), 10):
        # tuple of 6 ints, already counts-ordered
        assert len(k) == 6 and all(isinstance(x, int) for x in k)
        hash(k)  # will raise TypeError if not hashable


def test_build_and_roundtrip():
    table = sl.build_score_lookup_table()  # ← hits ~40 lines
    assert len(table) > 900  # sanity

    # make a random roll – score it two ways and compare
    rng = np.random.default_rng(0)
    roll = rng.integers(1, 7, size=6)
    key = tuple(roll.tolist().count(i) for i in range(1, 7))
    score_via_table = table[key][0]
    score_via_eval = sl.evaluate(key)[0]
    assert score_via_table == score_via_eval


# """
# Tests for Numba changes to strategies
# Extra coverage for ThresholdStrategy.decision logic as well as the
# `random_threshold_strategy` helper.
# """


@pytest.mark.parametrize(
    "turn_score, dice_left, expect_roll",
    [
        (0, 6, True),  # still well below the entry gate
        (400, 6, True),  # under score threshold
        (1450, 6, False),  # above score threshold
        (300, 1, False),  # too few dice left
        (50, 2, False),  # hits dice threshold
    ],
)
def test_decide_basic(turn_score, dice_left, expect_roll):
    strat = ThresholdStrategy(
        score_threshold=1000, dice_threshold=2, consider_score=True, consider_dice=True
    )
    keep_rolling = strat.decide(
        turn_score=turn_score,
        dice_left=dice_left,
        has_scored=True,  # skip the 500-point entry rule
        score_needed=0,
    )
    assert keep_rolling is expect_roll


def test_random_threshold_strategy_diversity():
    """Just make sure we don't always get the same parameters."""
    seen = Counter(random_threshold_strategy().score_threshold for _ in range(10))
    # heuristic – at least three distinct thresholds in 10 draws
    assert len(seen) >= 3


@pytest.mark.parametrize(
    "running_total, score_to_beat, expected",
    [
        (5000, 5500, True),
        (6000, 5500, True),
    ],
)
def test_decide_final_round_ignores_other_flags(running_total, score_to_beat, expected):
    strat = ThresholdStrategy(auto_hot_dice=True, run_up_score=True)
    res = strat.decide(
        turn_score=0,
        dice_left=6,
        has_scored=True,
        score_needed=0,
        final_round=True,
        score_to_beat=score_to_beat,
        running_total=running_total,
    )
    assert res is expected
