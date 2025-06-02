"""Table-driven scoring rules for Farkle Mk II.

This module holds **only** the rule functions and the evaluation
pipeline.  All higher-level helpers (smart-5 discard logic, player
turns, etc.) should import :func:`build_score_lookup_table` rather than duplicating
rule logic.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations_with_replacement
from typing import Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A rule takes a Counter of face-counts (1 → n₁, 2 → n₂, …) and returns
#    (points_awarded, dice_used).
# If the rule **does not** trigger it must return (None, 0).
Rule = Callable[[Counter[int]], Tuple[int | None, int]]

# ---------------------------------------------------------------------------
# Individual rule functions (ordered from most-specific to most-generic)
# ---------------------------------------------------------------------------

def straight(counts: Counter[int]) -> Tuple[int | None, int]:
    """1-6 straight worth 1500, uses all six dice."""
    return (1500, 6) if len(counts) == 6 else (None, 0)


def three_pairs(counts: Counter[int]) -> Tuple[int | None, int]:
    """Exactly three pairs worth 1500, uses all six dice."""
    return (1500, 6) if len(counts) == 3 and all(v == 2 for v in counts.values()) else (None, 0)


def two_triplets(counts: Counter[int]) -> Tuple[int | None, int]:
    """Two distinct triples worth 2500, uses all six dice."""
    return (2500, 6) if len(counts) == 2 and set(counts.values()) == {3} else (None, 0)


def four_kind_plus_pair(counts: Counter[int]) -> Tuple[int | None, int]:
    """4-of-a-kind + separate pair worth 1500, uses all six dice."""
    return (
        (1500, 6)
        if len(counts) == 2 and 4 in counts.values() and 2 in counts.values()
        else (None, 0)
    )


def n_of_a_kind(counts: Counter[int]) -> Tuple[int | None, int]:
    """Triplets and larger (same face).

    • Three 1s  →  300
    • Three Xs  →  X * 100  (X = 2-6)
    • Four-of-a-kind → 1000
    • Five-of-a-kind → 2000
    • Six-of-a-kind  → 3000
    (These follow the house rules used in the original Farkle notebook.)
    """
    # Find *any* face with ≥3 counts.  If multiple qualify, the first one
    # triggers; caller is expected to remove the used dice and call again
    # if they want cumulative scoring.
    for face, n in counts.items():
        if n >= 3:
            # Calculate points based on how many dice are in the set.
            if n == 3:
                pts = 300 if face == 1 else face * 100
            elif n == 4:
                pts = 1000
            elif n == 5:
                pts = 2000
            else:  # n == 6
                pts = 3000
            return pts, n
    return None, 0


def singles(counts: Counter[int]) -> Tuple[int | None, int]:
    """Score any leftover single 1s and 5s (others are 0)."""
    n1 = counts.get(1, 0)
    n5 = counts.get(5, 0)
    if n1 or n5:
        return n1 * 100 + n5 * 50, n1 + n5
    return None, 0


# ---------------------------------------------------------------------------
# Ordered rule chain - evaluated top-to-bottom until the *first* rule fires.
# Once a special 6-dice pattern is detected we **stop** (per common house
# rules).  If none match, we fall back to n-of-a-kind and singles.
# ---------------------------------------------------------------------------

SCORING_CHAIN: List[Rule] = [
    straight,
    three_pairs,
    two_triplets,
    four_kind_plus_pair,
]

# The generic rules are handled in a second pass so we can accumulate points
# from multiple sets (e.g. triple-4 plus single-1).

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(counts: Counter[int]) -> tuple[int, int]:
    """Return (total_points, total_dice_used) for the given roll counts.

    The caller is expected to make a *copy* of the Counter if they still
    need the original because this function will mutate it while peeling
    off scoring combinations.
    """
    # Pass 1: check special 6-dice patterns in priority order.
    for rule in SCORING_CHAIN:
        pts, used = rule(counts)
        if pts is not None:
            return pts, used

    total_score = total_used = 0

    # Pass 2: peel off n-of-a-kind combinations until none remain.
    while True:
        pts, used = n_of_a_kind(counts)
        if pts is None:
            break
        total_score += pts
        total_used += used
        # Remove the dice that produced points so singles logic sees leftovers.
        for face,  count in list(counts.items()):
            if  count >= 3:
                del counts[face]
                break

    # Pass 3: single 1s and 5s.
    pts, used = singles(counts)
    if pts:
        total_score += pts
        total_used += used

    return total_score, total_used


# ---------------------------------------------------------------------------
# Convenience helper for raw rolls
# ---------------------------------------------------------------------------

def score_roll(roll: list[int]) -> tuple[int, int]:
    """Thin wrapper so callers can pass a raw list of ints (dice)."""
    return evaluate(Counter(roll))


"""
Fast lookup table for any 1-6-die roll.

Key  : tuple(count_1, count_2, count_3, count_4, count_5, count_6)
Value: (score, used_dice)  # reroll = n_total - used_dice
"""
# LOOKUP: Dict[Tuple[int, int, int, int, int, int], Tuple[int, int]] = {}

def build_score_lookup_table():
    LOOKUP: Dict[Tuple[int, int, int, int, int, int], Tuple[int, int]] = {}
    faces = range(1, 7)
    for n in range(1, 7):                                # 1-die … 6-die rolls
        for multiset in combinations_with_replacement(faces, n):
            counts = Counter(multiset)
            key = tuple(counts.get(f, 0) for f in faces)
            assert len(key) == 6, "Counter length not 6 somehow"

            score, used = evaluate(counts.copy())
            if score == 0:                               # fall back to singles
                ones, fives = key[0], key[4]
                score = ones * 100 + fives * 50
                used  = ones + fives
            try:
                LOOKUP[key] = (score, used)
            except Exception as e:
                print(f"Error {e} on key, value assignment: {key}, score: {score}, used: {used}")
                print(f"\nKey {key} value set to (-1, -1)")
                LOOKUP[key] = (-1, -1)
    return LOOKUP
