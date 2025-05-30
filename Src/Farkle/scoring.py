from __future__ import annotations
"""scoring.py
================
Pure scoring utilities for the Farkle simulation suite.

The module purposefully keeps *all* logic side‑effect–free so that it can
be unit‑tested in isolation and reused by any front‑end (command‑line,
web, RL agents, etc.).

Functions
---------
`default_score` – score an arbitrary dice roll under canonical Farkle
rules, with an optional *Smart‑5* heuristic that discards low‑value lone
fives when it is in the player’s interest.

Type aliases
------------
`DiceRoll` – ``list[int]`` convenience alias for type‑checking clarity.
"""

from collections import Counter
from typing import List, Tuple

__all__: list[str] = [
    "DiceRoll",
    "default_score",
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DiceRoll = List[int]
"""A list of integers 1‑6 representing a single dice roll."""

# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

def default_score(
    dice_roll: DiceRoll,
    *,
    smart: bool = False,
    score_threshold: int = 300,
) -> Tuple[int, int, int]:
    """Evaluate a dice roll under Farkle rules.

    Parameters
    ----------
    dice_roll
        Sequence of integers 1–6 from a single throw of *n* dice.
    smart
        Whether to apply the Smart‑5 heuristic.  If *True* and the roll
        contains **one or two** single‑die fives, **no other scoring
        dice**, and the provisional turn score is below *score_threshold*,
        then lone fives (except one) are discarded: 50 points are
        subtracted for each discarded five, one die is freed for each,
        and the roll never becomes a bust.
    score_threshold
        The player’s turn‑score threshold (usually the strategy’s
        ``score_threshold``).  Smart‑5 only triggers when the provisional
        score is strictly less than this value.

    Returns
    -------
    score_pts, dice_used, dice_to_reroll
        *score_pts* is the integer score of the roll **after** Smart‑5
        adjustments, *dice_used* is how many dice contributed to that
        score, and *dice_to_reroll* is how many dice remain for the next
        throw (0 implies *hot dice*).
    """
    counts: Counter[int] = Counter(dice_roll)
    score: int = 0
    used: int = 0

    # ----- Special combos -------------------------------------------------
    if len(counts) == 6:  # straight 1‑6
        return 1500, 6, 0
    if len(counts) == 3 and all(v == 2 for v in counts.values()):  # three pairs
        return 1500, 6, 0
    if len(counts) == 2 and set(counts.values()) == {3, 3}:  # two triplets
        return 2500, 6, 0
    if len(counts) == 2 and 4 in counts.values() and 2 in counts.values():  # 4‑kind + pair
        return 1500, 6, 0

    single_fives: int = 0  # track single‑die fives for Smart‑5

    # ----- Triplets & singles --------------------------------------------
    for num, cnt in counts.items():
        if cnt >= 3:
            if cnt == 3:
                score += 300 if num == 1 else num * 100
            elif cnt == 4:
                score += 1000
            elif cnt == 5:
                score += 2000
            elif cnt == 6:
                score += 3000
            used += cnt
        elif num == 1:
            score += 100 * cnt
            used += cnt
        elif num == 5:
            score += 50 * cnt
            used += cnt
            single_fives += cnt  # only singles (cnt 1 or 2) fall here

    reroll: int = len(dice_roll) - used

    # ----- Smart‑5 adjustment --------------------------------------------
    if smart and 1 <= single_fives <= 2 and score < score_threshold:
        # Any *other* scoring dice cancel Smart‑5
        other_scoring = any(
            (n == 1) or (c >= 3 and n != 5)
            for n, c in counts.items()
        )
        if not other_scoring:
            discard = single_fives - 1 if single_fives > 1 else 0
            if discard:
                score -= 50 * discard
                used -= discard
                reroll += discard

    return score, used, reroll
