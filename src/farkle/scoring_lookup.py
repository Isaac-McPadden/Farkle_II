# src/farkle/scoring_lookup.py  – Numba- & cache-ready
"""Evaluate Farkle dice rolls using Numba accelerated routines.

This module exposes the key functions :func:`_evaluate_nb`,
:func:`evaluate`, :func:`score_roll`, and
:func:`build_score_lookup_table` for scoring rolls or generating a
lookup table.  Inputs are 6-tuples of counts whose elements are
non‐negative and sum to at most six.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations_with_replacement
from typing import Sequence

import numba as nb
import numpy as np

from farkle.types import Int64Array1D, SixFaceCounts

# ---------------------------------------------------------------------------
# 0.  Low-level helpers  (all *nopython*-safe)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _straight(ctr: Int64Array1D) -> Tuple[int, int]:
    """Return the straight bonus if every face appears once.

    Args:
        ctr: Array of counts for faces one through six.

    Returns:
        (1500, 6) when ctr equals [1, 1, 1, 1, 1, 1].
        Otherwise (0, 0).
    """
    return (1500, 6) if np.all(ctr == 1) else (0, 0)


@nb.njit(cache=True)
def _three_pairs(ctr: Int64Array1D) -> Tuple[int, int]:
    """Detect the *three pairs* pattern.

    Args:
        ctr: Array of counts for faces one through six.

    Returns:
        A tuple (1500, 6) when ctr contains exactly three
        different pairs, otherwise (0, 0).
    """
    pairs = (ctr == 2).sum()
    return (1500, 6) if pairs == 3 else (0, 0)


@nb.njit(cache=True)
def _two_triplets(ctr: Int64Array1D) -> Tuple[int, int]:
    """Detect the *two triplets* pattern.

    Args:
        ctr: Array of counts for faces one through six.

    Returns:
        (2500, 6) when ctr contains two distinct three-of-a-kind
        groups, otherwise (0, 0).
    """
    trips = (ctr == 3).sum()
    return (2500, 6) if trips == 2 else (0, 0)


@nb.njit(cache=True)
def _four_kind_plus_pair(ctr: Int64Array1D) -> Tuple[int, int]:
    """Check for four of a kind together with a separate pair.

    Args:
        ctr: Array of counts for faces one through six.

    Returns:
        (1500, 6) if ctr contains a four-of-a-kind and a
        different pair, otherwise (0, 0).
    """
    return (1500, 6) if 4 in ctr and 2 in ctr else (0, 0)


@nb.njit(cache=True)
def _apply_sets(ctr: Int64Arr1D) -> tuple[int, int]:
    """Score and consume any n-of-a-kind groups.

    Parameters
    ----------
    ctr:
        Array of counts for faces one through six. Modified in-place.

    Returns
    -------
    tuple[int, int]
        Additional (score, used) from all qualifying sets.
    """
    score_add = 0
    used_add = 0
    for face in range(6):
        n = ctr[face]
        if n >= 3:
            if n == 3:
                pts = 300 if face == 0 else (face + 1) * 100
            elif n == 4:
                pts = 1000
            elif n == 5:
                pts = 2000
            else:  # n == 6
                pts = 3000
            score_add += pts
            used_add += n
            ctr[face] = 0
    return score_add, used_add


# ---------------------------------------------------------------------------
# 1.  Master evaluator (Numba core + pure-Python wrapper)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _evaluate_nb(
    c1: int,
    c2: int,
    c3: int,
    c4: int,
    c5: int,
    c6: int,
) -> tuple[int, int, int, int]:
    """Score a roll purely within Numba.

    Args:
        c1, c2, c3, c4, c5, c6: Number of dice showing each face value
            from 1 through 6.

    Returns:
        A tuple (score, used, single_fives, single_ones) where
        score is the total points, used is the number of dice that
        contribute to that score and the last two elements report how many
        lone fives and ones remain.
    """
    # ---- convert inputs to fixed array -----------------------------------
    ctr = np.array([c1, c2, c3, c4, c5, c6], dtype=np.int64)
    score = used = 0

    # ---- special 6-dice patterns ----------------------------------------
    pts, ud = _straight(ctr)
    if pts:
        return pts, ud, 0, 0
    pts, ud = _three_pairs(ctr)
    if pts:
        return pts, ud, 0, 0
    pts, ud = _two_triplets(ctr)
    if pts:
        return pts, ud, 0, 0
    pts, ud = _four_kind_plus_pair(ctr)
    if pts:
        return pts, ud, 0, 0

    # ---- n-of-a-kind sets -------------------------------------------------
    pts, ud = _apply_sets(ctr)
    score += pts
    used += ud

    # ---- leftover singles (only 1s and 5s score) -------------------------
    lone_ones = ctr[0]
    lone_fives = ctr[4]
    score += lone_ones * 100 + lone_fives * 50
    used += lone_ones + lone_fives
    return score, used, lone_fives, lone_ones


# ---------------------------------------------------------------------------
# 2.  Thin Python shims (hashable → JIT core → cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def evaluate(counts: SixFaceCounts) -> tuple[int, int, int, int]:
    """Score a counts tuple via the JIT compiled core.

    The function is intentionally defensive – invalid input should raise a
    :class:`ValueError` rather than yielding nonsensical results.

    Args:
        counts: A 6-tuple giving the number of dice showing each face. The
            tuple must consist of six non-negative integers whose sum does not
            exceed six.

    Returns:
        (score, used, single_fives, single_ones) in the same format as
        :func:`_evaluate_nb`.
    """
    if len(counts) != 6:
        raise ValueError("counts must contain exactly six values")
    if not all(isinstance(c, int) for c in counts):
        raise TypeError(f"non-integers in {counts!r}")
    if any(c < 0 for c in counts):
        raise ValueError(f"negative count in {counts!r}")
    if sum(counts) > 6:
        raise ValueError(f"more than six dice specified: {counts!r}")
    return _evaluate_nb(*counts)


def score_roll(roll: Sequence[int]) -> tuple[int, int]:
    """Score a roll given as a list of faces.

    Args:
        roll: Sequence of integers in ``[1, 6]`` representing the dice.
            The sequence may contain at most six values.

    Returns:
        A ``(score, used_dice)`` tuple describing the total points scored
        and how many dice contributed to the score.

    Raises:
        ValueError: If ``roll`` contains values outside ``[1, 6]`` or has
        more than six elements.
    """
    if len(roll) > 6:
        raise ValueError("roll cannot contain more than six dice")
    if any(d < 1 or d > 6 for d in roll):
        raise ValueError(f"invalid die face in {roll!r}")

    key = (
        roll.count(1),
        roll.count(2),
        roll.count(3),
        roll.count(4),
        roll.count(5),
        roll.count(6),
    )
    pts, used, *_ = evaluate(key)
    return pts, used


# ---------------------------------------------------------------------------
# 3.  Pre-compute full lookup table (still ~56 k combos → 923 uniques)
# ---------------------------------------------------------------------------


def build_score_lookup_table() -> dict[SixFaceCounts, tuple[int, int, SixFaceCounts, int, int]]:
    """
    Fast O(1) table for any ≤6-dice roll.
    Loads pre-computed scores for every multiset of up to six dice.

    Inputs:
        None

    Returns:
        A dictionary mapping (c1, c2, c3, c4, c5, c6) tuples to
        (score, used, counts, single_fives, single_ones).  The
        first element is the total points for that combination and
        counts repeats the key so callers can keep a reference.
    """
    look: dict[
        tuple[int, int, int, int, int, int],
        tuple[int, int, tuple[int, int, int, int, int, int], int, int],
    ] = {}
    faces = range(1, 7)

    for n in range(1, 7):
        for multiset in combinations_with_replacement(faces, n):
            key = (
                multiset.count(1),
                multiset.count(2),
                multiset.count(3),
                multiset.count(4),
                multiset.count(5),
                multiset.count(6),
            )
            if key in look:  # skip duplicates (e.g. (2,2,2,5,5) permutes)
                continue
            score, used, sf, so = evaluate(key)
            look[key] = (score, used, key, sf, so)
    return look
