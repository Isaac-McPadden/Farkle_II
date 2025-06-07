from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

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

# ---------------------------------------------------------------------------
# Utility – lossless Counter → list[int]
# ---------------------------------------------------------------------------

def counter_to_roll(counts: Counter[int]) -> list[int]:
    """
    Convert a Counter({face: n, ...}) back to a *sorted* dice-roll list.

    Counter({5: 2, 2: 3})  ➔  [2, 2, 2, 5, 5]
    """
    roll: list[int] = []
    for face, n in counts.items():
        roll.extend([face] * n)
    roll.sort()
    return roll


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


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def expand_sorted(counts_dict: dict[int, int]) -> List[int]:
    """Return a *sorted list* of dice faces expanded from a ``counts`` mapping.

    Example
    -------
    >>> expand_sorted({2: 3, 5: 2})
    [2, 2, 2, 5, 5]
    """
    result: List[int] = []
    for face in sorted(counts_dict):
        result.extend([face] * counts_dict[face])
    return result


def generate_sequences(counter_dict: dict[int, int], *, smart_one: bool = False) -> List[List[int]]:
    """Enumerate **every** legal *post‑discard* kept‑dice arrangement.

    Discards happen *one at a time* in the mandated order – all 5’s first,
    followed by 1’s **iff** ``smart_one`` is enabled – and each intermediate
    state is recorded.  The original (no‑discard) roll is always first.

    No attempt is made to filter non‑scoring rolls here; that’s deferred to
    :pyfunc:`score_lister`.
    """
    counts = counter_dict.copy()
    sequences: List[List[int]] = [expand_sorted(counts)]  # original roll first

    # Remove single 5’s one by one
    while counts.get(5, 0) > 0:
        counts[5] -= 1
        if counts[5] == 0:
            del counts[5]
        sequences.append(expand_sorted(counts))

    # Optional removal of single 1’s after *all* 5’s are gone
    if smart_one:
        while counts.get(1, 0) > 0:
            counts[1] -= 1
            if counts[1] == 0:
                del counts[1]
            sequences.append(expand_sorted(counts))

    return sequences


def score_lister(dice_rolls: List[List[int]]) -> List[Tuple[List[int], int, int, int, Counter[int], int, int]]:
    """Annotate each roll with its scoring metadata, **skipping** non‑scoring rolls.

    Returns
    -------
    List of tuples:
        (roll, roll_len, raw_score, raw_used, counts, single_fives, single_ones)
    """
    scored: List[Tuple[List[int], int, int, int, Counter[int], int, int]] = []
    for roll in dice_rolls:
        raw_score, raw_used, counts, single_fives, single_ones = score_roll_cached(roll)
        if raw_score == 0:
            # A non‑scoring roll cannot be kept – skip entirely
            continue
        scored.append((roll, len(roll), raw_score, raw_used, counts, single_fives, single_ones))
    return scored


# ────────────────────────────────────────────────────────────────────────────
# Smart‑discard V2 – public API
# ────────────────────────────────────────────────────────────────────────────

def decide_smart_discards(
    *,
    counts: Counter[int],
    single_fives: int,
    single_ones: int,
    raw_score: int,  # noqa: ARG001
    raw_used: int,
    dice_roll_len: int,
    turn_score_pre: int,
    score_threshold: int,
    dice_threshold: int,
    smart_five: bool,
    smart_one: bool,
    consider_score: bool = True,
    consider_dice: bool = True,
    require_both: bool = False,
    prefer_score: bool = True,
) -> Tuple[int, int]:
    """Return *(discard_fives, discard_ones)* as dictated by the Smart strategy.

    The routine examines *every* post-discard kept-dice configuration produced
    by :pyfunc:`generate_sequences`, filters out those that would *force* the
    player to bank under the configured threshold policy, and finally selects
    the candidate that maximises the chosen metric:

    * ``prefer_score = True``  → sort by ``(score_after, dice_left_after)``.
    * ``prefer_score = False`` → sort by ``(dice_left_after, score_after)``.

    If **no** candidate satisfies the “keep rolling” condition, the function
    defaults to *(0, 0)* - i.e. keep everything.
    """

    # ───────‑ early exits ───────
    if not smart_five:
        return 0, 0                                    # feature disabled
    if raw_used == dice_roll_len:
        return 0, 0                                    # hot‑dice – all dice scored
    if single_fives == 0 and single_ones == 0:
        return 0, 0                                    # nothing to discard

    # Helper: decide whether the player *must* bank after this candidate.
    def _must_bank(score_after: int, dice_left_after: int) -> bool:
        score_hit = (score_after >= score_threshold) if consider_score else False
        dice_hit  = (dice_left_after <= dice_threshold) if consider_dice else False
        if consider_score and consider_dice and require_both:
            return score_hit and dice_hit              # AND logic
        return score_hit or dice_hit                  # OR logic (default)

    # Enumerate and score every discard combination.
    sequences = generate_sequences(dict(counts), smart_one=smart_one)
    candidates = score_lister(sequences)

    best_key: Optional[Tuple[int, int]] = None        # sort key per prefer_score
    best_single_fives = single_fives                  # initialise with original
    best_single_ones  = single_ones

    for (_roll, _roll_len, cand_score, cand_used,
         _cnt, cand_sf, cand_so) in candidates:

        score_after = turn_score_pre + cand_score
        dice_left_after = dice_roll_len - cand_used   # dice available *next* roll

        if _must_bank(score_after, dice_left_after):
            # This candidate would send us to the bank – not a smart discard
            continue

        key: Tuple[int, int]
        if prefer_score:  # noqa: SIM108
            key = (score_after, dice_left_after)
        else:
            key = (dice_left_after, score_after)

        if best_key is None or key > best_key:
            best_key = key
            best_single_fives = cand_sf
            best_single_ones  = cand_so

    if best_key is None:
        # All candidates would bank – fall back to keeping everything
        return 0, 0

    # Number of *lone* dice we actually discarded
    discard_fives = single_fives - best_single_fives
    discard_ones  = single_ones  - best_single_ones
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
    consider_score:   bool = True,
    consider_dice:    bool = True,
    require_both:     bool = False,
    score_threshold:  int  = 300,
    dice_threshold:   int  = 3,
    prefer_score:     bool = True,
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
        dice_threshold  = dice_threshold,
        smart_five      = smart_five,
        smart_one       = smart_one,
        consider_score  = consider_score,
        consider_dice   = consider_dice,
        require_both    = require_both,
        prefer_score    = prefer_score
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

def _compute_raw_score(
    dice_roll: DiceRoll,
) -> Tuple[int, int, Counter[int], int, int]:
    """
    Legacy, present for regression testing
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