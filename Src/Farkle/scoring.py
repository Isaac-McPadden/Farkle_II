from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from farkle.scoring_lookup import build_score_lookup_table

"""scoring.py
================
Pure scoring utilities for the Farkle simulation suite.

The module purposefully keeps *all* logic side-effect-free so that it can
be unit-tested in isolation and reused by any front-end (command-line,
web, RL agents, etc.).

Functions
---------
`default_score` - score an arbitrary dice roll under canonical Farkle
rules, with an optional *Smart-5* heuristic that discards low-value lone
fives when it is in the player's interest.

Type aliases
------------
`DiceRoll` - ``list[int]`` convenience alias for type-checking clarity.
"""



__all__: list[str] = [
    "DiceRoll",
    "default_score",
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DiceRoll = List[int]
"""A list of integers 1-6 representing a single dice roll."""



def compute_raw_score(
    dice_roll: DiceRoll,
) -> Tuple[int, int, Counter[int], int, int]:
    """
    Given a single roll (up to 6 dice), compute:
      - raw_score:   total points from this roll under the Farkle rules
      - raw_used:    how many dice “scored” (i.e. removed from play this roll)
      - counts:      Counter of face→count, for use by Smart-1/Smart-5 logic
      - single_fives: # of solitary 5's not part of any higher combo
      - single_ones:  # of solitary 1's not part of any higher combo

    Scoring hierarchy (highest-priority first):
      1) Straight 1-2-3-4-5-6          = 1,500   (uses all 6 dice)
      2) Three pairs                   = 1,500   (uses all 6 dice)
      3) Two triplets                  = 2,500   (uses all 6 dice)
      4) Four of a kind + a pair       = 1,500   (uses all 6 dice)
      5) Six of a kind                 = 3,000   (uses all 6 dice)
      6) Five of a kind                = 2,000   + (leftover die can still be a 1 or a 5)
      7) Four of a kind (alone)        = 1,000   + (score leftover 1's or 5's)
      8) Three of a kind (single)      = 300 if face=1 else face*100
                                         + (score leftover 1's or 5's)
      9) Single 1's (not in any triplet or above) = 100 each
     10) Single 5's (not in any triplet or above) = 50 each
    """
    counts     = Counter(dice_roll)
    raw_score  = 0
    raw_used   = 0

    # ------------- 1) Straight 1-6? -------------
    # If there are exactly 6 distinct faces, each must be 1-6 once:
    if len(counts) == 6:
        # Must be exactly {1:1, 2:1, 3:1, 4:1, 5:1, 6:1}
        return 1500, 6, counts, 0, 0

    # ------------- 2) Three pairs? -------------
    # Exactly three faces each with count==2
    if len(counts) == 3 and all(v == 2 for v in counts.values()):
        return 1500, 6, counts, 0, 0

    # ------------- 3) Two triplets? -------------
    # Exactly two faces each with count>=3. (In practice, each will be exactly 3.)
    if len(counts) == 2 and set(counts.values()) == {3}:
        return 2500, 6, counts, 0, 0

    # ------------- 4) Four-of-a-kind + pair? -------------
    # Look for one face with count>=4 AND another (different) face with count>=2.
    if len(counts) == 2 and 4 in counts.values() and 2 in counts.values():
        return 1500, 6, counts, 0, 0

    # At this point we know we do NOT have any of the “6-dice”-only combos above.
    # We'll now handle “n-of-a-kind” for n=6,5,4,3, in descending order, then fall back
    # to scoring any leftover single 1's or 5's.

    # Track how many stand-alone 1's or 5's are left once we peel off triplets/quads/etc.
    single_fives = 0
    single_ones  = 0

    # ------------- 5) Six-of-a-kind? -------------
    for face, count in counts.items():  # noqa: B007
        if count == 6:
            return 3000, 6, counts, 0, 0

    # ------------- 6) Five-of-a-kind? -------------
    # Score 2,000 for the five, then we must “look at” the leftover single die (if it's a 1 or a 5).
    for face, count in counts.items():
        if count == 5:
            raw_score = 2000
            raw_used  = 5
            # Remove those five from a temp Counter to see what's left:
            temp_counts = counts.copy()
            temp_counts[face] -= 5
            if temp_counts[face] == 0:
                del temp_counts[face]

            # If the leftover die is a 1 or a 5, score it now:
            if temp_counts.get(1, 0) == 1:
                raw_score += 100
                raw_used  += 1
            elif temp_counts.get(5, 0) == 1:
                raw_score += 50
                raw_used  += 1

            # Count how many single 5's / single 1's remain for later Smart logic:
            single_fives = temp_counts.get(5, 0)
            single_ones  = temp_counts.get(1, 0)
            return raw_score, raw_used, counts, single_fives, single_ones

    # ------------- 7) Four-of-a-kind (alone)? -------------
    for face, count in counts.items():
        if count == 4:
            # Score 1,000 for the four-of-a-kind
            raw_score = 1000
            raw_used  = 4

            # Remove those four from a temp Counter to score leftovers:
            temp_counts = counts.copy()
            temp_counts[face] -= 4
            if temp_counts[face] == 0:
                del temp_counts[face]

            # Any leftover dice (2 of them) can only contribute single 1's or 5's:
            #   - Each 1 → +100
            #   - Each 5 → +50
            single_ones  = temp_counts.get(1, 0)
            single_fives = temp_counts.get(5, 0)
            raw_score += 100 * single_ones + 50 * single_fives
            raw_used  += single_ones + single_fives

            return raw_score, raw_used, counts, single_fives, single_ones

    # ------------- 8) Single three-of-a-kind? -------------
    # (We already excluded two triplets, four-pair+pair, etc. above.)
    for face, count in counts.items():
        if count >= 3:
            # Score the triplet:
            raw_score = 300 if face == 1 else face * 100
            raw_used = 3

            # Remove the three from counts so we can score leftover dice:
            temp_counts = counts.copy()
            temp_counts[face] -= 3
            if temp_counts[face] == 0:
                del temp_counts[face]

            # Now any remaining (up to 3) dice can only be single 1's or 5's:
            single_ones  = temp_counts.get(1, 0)
            single_fives = temp_counts.get(5, 0)
            raw_score += 100 * single_ones + 50 * single_fives
            raw_used  += single_ones + single_fives

            return raw_score, raw_used, counts, single_fives, single_ones

    # ------------- 9) Fallback: only single 1's or 5's -------------
    # At this point, no straights, no three-pairs, no two-triplets,
    # no four+pair, no six, no five, no four, no three. So the only scoring
    # dice left are single 1's or single 5's.
    single_ones  = counts.get(1, 0)
    single_fives = counts.get(5, 0)

    raw_score = 100 * single_ones + 50 * single_fives
    raw_used  = single_ones + single_fives

    return raw_score, raw_used, counts, single_fives, single_ones



lookup_table = build_score_lookup_table()

def score_roll_cached(
    roll: list[int],
    lookup: dict = lookup_table,
) -> tuple[int, int, Counter[int], int, int]:
    """
    Return (score, used, counts, single_fives, single_ones) in O(1).
    """
    key = (
        roll.count(1), roll.count(2), roll.count(3),
        roll.count(4), roll.count(5), roll.count(6)
    )
    score, used, base_counts, sfives, sones = lookup[key]
    return score, used, base_counts.copy(), sfives, sones


def decide_smart_discards(
    counts: Counter[int],
    single_fives: int,
    single_ones: int,
    raw_score: int,
    raw_used: int,
    dice_roll_len: int,  # noqa: ARG001
    turn_score_pre: int,
    score_threshold: int,
    smart_five: bool,
    smart_one: bool,
) -> Tuple[int, int]:
    """
    Decide how many 5's and then 1's to discard, but only commit them if
    doing so leaves (turn_score_pre + new_score) < threshold.

    Steps:
      1) Compute discard_fives purely from smart_five rules.
      2) Compute intermediate_score_after5 = raw_score - 50*discard_fives.
         If turn_score_pre + intermediate_score_after5 >= threshold, restore
         discard_fives = 0 (i.e. do not toss any 5's).
      3) Compute discard_ones purely from smart_one rules, applied to a Counter
         that reflects removing the (possibly zero) discard_fives. 
      4) Compute final_score_after_all = intermediate_score_after5 - 100*discard_ones.
         If turn_score_pre + final_score_after_all >= threshold, restore discard_ones = 0.

    Return (discard_fives, discard_ones).
    """

    # --- 1) Decide how many 5's we would toss, ignoring threshold for the moment ---
    discard_fives = 0
    if smart_five and single_fives >= 1:
        # Is there any “other scoring die” besides lone 5's?
        other_scoring_for_five = any(
            (count >= 3 and face != 5) or (face == 1 and count >= 1)
            for face, count in counts.items()
        )
        if other_scoring_for_five:
            # Discard all lone 5's (even if single_fives == 1)
            discard_fives = single_fives
        else:
            # Only scoring dice are these single 5's. If ≥ 2, keep exactly 1, discard rest
            if single_fives >= 2:
                discard_fives = single_fives - 1
            # If single_fives == 1 and no other scoring, we do not discard it.

    # Compute intermediate score & used if we DID discard those 5's
    score_after_5 = raw_score - 50 * discard_fives
    used_after_5  = raw_used  - discard_fives

    # --- 2) If throwing away those 5's would STILL leave us ≥ threshold, restore them ---
    if turn_score_pre + score_after_5 >= score_threshold:
        discard_fives = 0
        score_after_5 = raw_score
        used_after_5  = raw_used

    # Remove those (possibly zero) 5's from counts so Smart-1 sees the right picture
    temp_counts = counts.copy()
    if discard_fives > 0:
        temp_counts[5] -= discard_fives
        if temp_counts[5] <= 0:
            del temp_counts[5]

    # Re-count how many lone 1's remain (no need to re-compute single_fives now)
    # single_ones still represents how many 1's were in the original roll.
    # But if we removed some 5's, single_ones is unchanged.
    # (We only need single_ones to see if ≥2 remain.)

    # --- 3) Decide how many 1's we would toss, ignoring threshold again ---
    discard_ones = 0
    if smart_one and single_ones >= 1:  # noqa: SIM102
        # Only attempt Smart-1 if the raw roll (or after Smart-5) 
        # left us below threshold:
        if turn_score_pre + score_after_5 < score_threshold:
            # “Other scoring” w.r.t the remaining dice (after removing 5's):
            other_scoring_for_one = any(
                (count >= 3 and face != 1) or (face == 5 and count >= 1) # failsafe if sf=F, so=T was allowed
                for face, count in temp_counts.items()
            )
            if not other_scoring_for_one and single_ones == 2:
                # If ≥ 2 lonely 1's remain, discard all but one.
                discard_ones = 1
            if not other_scoring_for_one and single_ones == 1:
                discard_ones = 0
            if other_scoring_for_one and single_ones == 2:
                # If ≥ 2 lonely 1's remain, discard all but one.
                discard_ones = 2
            if other_scoring_for_one and single_ones == 1:
                discard_ones = 1

    # Compute final score if we commit to discarding those 1's
    score_after_all = score_after_5 - 100 * discard_ones
    used_after_5 -= discard_ones

    # --- 4) If discarding those 1's would STILL leave us ≥ threshold, restore them ---
    if turn_score_pre + score_after_all >= score_threshold:
        discard_ones   = 0
        score_after_all = score_after_5

    return discard_fives, discard_ones



def apply_discards(
    raw_score:     int,
    raw_used:      int,
    discard_fives: int,
    discard_ones:  int,
    dice_roll_len: int,
) -> Tuple[int, int, int]:
    """
    Given:
      - raw_score, raw_used   (from compute_raw_score)
      - discard_fives, discard_ones   (from decide_smart_discards)
      - dice_roll_len = len(dice_roll)

    Return:
      - final_score   = raw_score - 50*discard_fives - 100*discard_ones
      - final_used    = raw_used  - discard_fives - discard_ones
      - final_reroll  = dice_roll_len - final_used
    """
    final_score  = raw_score   - 50 * discard_fives - 100 * discard_ones
    final_used   = raw_used    - discard_fives   - discard_ones
    final_reroll = dice_roll_len - final_used
    return final_score, final_used, final_reroll


# scoring.py

def default_score(
    dice_roll:        DiceRoll,
    *,
    turn_score_pre:   int,
    smart_five:       bool = False,
    smart_one:        bool = False,
    score_threshold:  int  = 300,
) -> Tuple[int, int, int]:
    """
    Master function that:
      1) computes raw score/used/counts
      2) decides how many 5's/1's to discard (but only commits if it keeps you below threshold)
      3) applies those discards
      4) returns (score, used, reroll)
    """
    # 1) Raw scoring
    raw_score, raw_used, counts, single_fives, single_ones = score_roll_cached(dice_roll)

    # 2) Figure out discards
    discard_fives, discard_ones = decide_smart_discards(
        counts          = counts,
        single_fives    = single_fives,
        single_ones     = single_ones,
        raw_score       = raw_score,
        raw_used        = raw_used,
        dice_roll_len   = len(dice_roll),
        turn_score_pre  = turn_score_pre,
        score_threshold = score_threshold,
        smart_five      = smart_five,
        smart_one       = smart_one,
    )

    # 3) Convert those discards into final numbers
    final_score, final_used, final_reroll = apply_discards(
        raw_score       = raw_score,
        raw_used        = raw_used,
        discard_fives   = discard_fives,
        discard_ones    = discard_ones,
        dice_roll_len   = len(dice_roll),
    )

    return final_score, final_used, final_reroll


# ---------------------------------------------------------------------------
# Old Core routine
# ---------------------------------------------------------------------------

# def default_score(
#     dice_roll: DiceRoll,
#     *,
#     smart: bool = False,
#     score_threshold: int = 300,
# ) -> Tuple[int, int, int]:
#     """Evaluate a dice roll under Farkle rules.

#     Parameters
#     ----------
#     dice_roll
#         Sequence of integers 1-6 from a single throw of *n* dice.
#     smart
#         Whether to apply the Smart-5 heuristic.  If *True* and the roll
#         contains **one or two** single-die fives, **no other scoring
#         dice**, and the provisional turn score is below *score_threshold*,
#         then lone fives (except one) are discarded: 50 points are
#         subtracted for each discarded five, one die is freed for each,
#         and the roll never becomes a bust.
#     score_threshold
#         The player's turn-score threshold (usually the strategy's
#         ``score_threshold``).  Smart-5 only triggers when the provisional
#         score is strictly less than this value.

#     Returns
#     -------
#     score_pts, dice_used, dice_to_reroll
#         *score_pts* is the integer score of the roll **after** Smart-5
#         adjustments, *dice_used* is how many dice contributed to that
#         score, and *dice_to_reroll* is how many dice remain for the next
#         throw (0 implies *hot dice*).
#     """
#     counts: Counter[int] = Counter(dice_roll)
#     score: int = 0
#     used: int = 0

#     # ----- Special combos -------------------------------------------------
#     if len(counts) == 6:  # straight 1-6
#         return 1500, 6, 0
#     if len(counts) == 3 and all(v == 2 for v in counts.values()):  # three pairs
#         return 1500, 6, 0
#     if len(counts) == 2 and set(counts.values()) == {3}:  # two triplets
#         return 2500, 6, 0
#     if len(counts) == 2 and 4 in counts.values() and 2 in counts.values():  # 4-kind + pair
#         return 1500, 6, 0

#     single_fives: int = 0  # track single-die fives for Smart-5

#     # ----- Triplets & singles --------------------------------------------
#     for num, count in counts.items():
#         if count >= 3:
#             if count == 3:
#                 score += 300 if num == 1 else num * 100
#             elif count == 4:
#                 score += 1000
#             elif count == 5:
#                 score += 2000
#             elif count == 6:
#                 score += 3000
#             used += count
#         elif num == 1:
#             score += 100 * count
#             used += count
#         elif num == 5:
#             score += 50 * count
#             used += count
#             single_fives += count  # only singles (count 1 or 2) fall here

#     reroll: int = len(dice_roll) - used

#     # ----- Smart-5 adjustment --------------------------------------------
#     if smart and 1 <= single_fives <= 2 and score < score_threshold:
#         # Any *other* scoring dice cancel Smart-5
#         other_scoring = any(
#             (n == 1) or (c >= 3 and n != 5)
#             for n, c in counts.items()
#         )
#         if not other_scoring:
#             discard = single_fives - 1 if single_fives > 1 else 0
#             if discard:
#                 score -= 50 * discard
#                 used -= discard
#                 reroll += discard

#     return score, used, reroll