# src/farkle/scoring.py
"""Score Farkle dice rolls and apply Smart discard logic.

This module wraps the Numba-accelerated helpers in
:mod:`farkle.scoring_lookup` to compute raw scores and exposes
:func:`default_score` for applying the Smart-5 and Smart-1 heuristics.
Tuple based logic for compatibility with Numba
"""

from __future__ import annotations

import functools
from typing import Callable, NamedTuple, Sequence, Tuple, Union, cast

# Numba is only used in the low-level helpers;
# no caller needs to install it explicitly.
import numba as nb
import numpy as np

from farkle.game.scoring_lookup import build_score_lookup_table
from farkle.game.scoring_lookup import evaluate as _eval_nb  # fast JIT core
from farkle.simulation.strategies import FavorDiceOrScore
from farkle.utils.types import DiceRoll, FacesSequence, Int64Array1D, SixFaceCounts

SCORE_TABLE = build_score_lookup_table()

# --------------------------------------------------------------------------- #
# 0.  Public type alias
# --------------------------------------------------------------------------- #


class ScoreCandidate(NamedTuple):
    """Container for potential scoring outcomes after discards."""

    faces: list[int]
    dice_len: int
    score: int
    used: int
    counts: SixFaceCounts
    single_fives: int
    single_ones: int


# --------------------------------------------------------------------------- #
# 1.  Tiny helpers – all immutable / hash-friendly
# --------------------------------------------------------------------------- #
@nb.njit(cache=True)
def _faces_to_counts_nb(faces: np.ndarray) -> SixFaceCounts:
    """Count occurrences of each face value.

    Inputs
    ------
    faces (np.ndarray):
        1-D array of dice faces.

    Returns
    -------
    SixFaceCounts:
        Tuple of counts for faces one through six.
    """
    out = np.zeros(6, dtype=np.int64)
    for v in faces:
        out[v - 1] += 1
    return (int(out[0]), int(out[1]), int(out[2]), int(out[3]), int(out[4]), int(out[5]))


def faces_to_counts_tuple(faces: Sequence[int]) -> SixFaceCounts:
    """Convert a sequence of faces to a counts tuple.

    Inputs
    ------
    faces (Sequence[int]):
        Iterable of dice faces.

    Returns
    -------
    SixFaceCounts:
        Six-element tuple of counts for faces one through six.

    Raises
    ------
    ValueError:
        If any face value is outside the ``1``–``6`` range.
    """
    if not all(1 <= f <= 6 for f in faces):
        raise ValueError("dice faces must be between 1 and 6")
    return _faces_to_counts_nb(np.asarray(faces, dtype=np.int64))


# --------------------------------------------------------------------------- #
# 2.  Fast single-roll scorer (cached)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=32_768)
def _score_by_counts(key: SixFaceCounts) -> Tuple[int, int, SixFaceCounts, int, int]:
    """Score a roll represented by face counts.

    Performance tip
    -----------------
    Pre-build the global lookup once at import time:

        >>> from farkle.scoring_lookup import build_score_lookup_table
        >>> SCORE_TABLE = build_score_lookup_table()

    All subsequent calls become an O(1) dict lookup

    Inputs
    ------
    key (SixFaceCounts):
        Tuple of counts for faces one through six.

    Returns
    -------
    tuple[int, int, SixFaceCounts, int, int]:
        (score, used, counts_key, single_fives, single_ones).
    """
    try:
        return SCORE_TABLE[key]
    except KeyError:
        score, used, sfives, sones = _eval_nb(key)  # JIT kernel
        return score, used, key, sfives, sones


def score_roll_cached(roll: Sequence[int]) -> Tuple[int, int, SixFaceCounts, int, int]:
    """Score a roll, caching by its face counts.

    Inputs
    ------
    roll (Sequence[int]):
        Iterable of dice faces.

    Returns
    -------
    tuple[int, int, SixFaceCounts, int, int]:
        (score, used, counts_key, single_fives, single_ones).
    """
    key = faces_to_counts_tuple(roll)
    return _score_by_counts(key)


# --------------------------------------------------------------------------- #
# 3.  Expand counts → sorted faces  (needed for Smart discards)
# --------------------------------------------------------------------------- #
Int64Array1DNP = np.ndarray  # local alias to keep the signature tidy inside numba


# 1) **FAST kernel** – stays in Numba, but now returns an ndarray
@nb.njit(cache=True)
def _expand_sorted_nb(c1: int, c2: int, c3: int, c4: int, c5: int, c6: int) -> Int64Array1D:
    """Expand counts into a sorted NumPy array.

    Inputs
    ------
    c1, c2, c3, c4, c5, c6 (int):
        Counts for faces one through six.

    Returns
    -------
    Int64Array1D:
        Array of face values in ascending order.
    """
    n_tot = c1 + c2 + c3 + c4 + c5 + c6
    out = np.empty(n_tot, dtype=np.int64)
    pos = 0
    for face, n in enumerate((c1, c2, c3, c4, c5, c6), 1):
        for _ in range(n):
            out[pos] = face
            pos += 1
    return out


# 2) **Thin wrapper** – cheap Python, converts to an immutable tuple
def _expand_sorted(counts: SixFaceCounts) -> FacesSequence:
    """Return sorted faces as an immutable tuple.

    Inputs
    ------
    counts (SixFaceCounts):
        Counts for faces one through six.

    Returns
    -------
    FacesSequence:
        Tuple of face values in ascending order.
    """
    return tuple(_expand_sorted_nb(*counts))


# --------------------------------------------------------------------------- #
# 4.  Enumerate every post-discard state  (cached)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4096)
def generate_sequences(
    counts: SixFaceCounts, *, smart_one: bool = False
) -> tuple[FacesSequence, ...]:
    """Enumerate all post-discard face sequences.
    Only ever run by functions with smart_five == True.

    Inputs
    ------
    counts (SixFaceCounts):
        Counts representing the roll.
    smart_one (bool, optional):
        Include sequences discarding single ones.

    Returns
    -------
    tuple[FacesSequence, ...]:
        All possible remaining dice as sorted tuples.
    """
    base_counts = list(counts)
    seqs: list[FacesSequence] = []
    seen: set[FacesSequence] = set()

    max_fives = base_counts[4]
    max_ones = base_counts[0] if smart_one else 0

    ones_range = range(max_ones + 1) if smart_one else range(1)

    for drop_fives in range(max_fives + 1):
        for drop_ones in ones_range:
            new_counts = list(base_counts)
            new_counts[4] -= drop_fives
            new_counts[0] -= drop_ones
            if new_counts[4] < 0 or new_counts[0] < 0:
                continue
            seq = _expand_sorted(cast(SixFaceCounts, tuple(new_counts)))
            if seq not in seen:
                seen.add(seq)
                seqs.append(seq)

    return tuple(seqs)


# --------------------------------------------------------------------------- #
# 5.  Score a batch of rolls (tuples of faces) – cached
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=16_384)
def score_lister(
    dice_rolls: tuple[FacesSequence, ...],
) -> tuple[ScoreCandidate, ...]:
    """Score multiple sorted rolls for smart five
    and smart one decision logic.  Prevents non-scoring
    rolls from occurring.

    Inputs
    ------
    dice_rolls (tuple[FacesSequence, ...]):
        Rolls represented as tuples of sorted faces.

    Returns
    -------
    tuple[ScoreCandidate, ...]:
        Candidate scoring states. Non-scoring rolls are skipped.
    """
    out: list[ScoreCandidate] = []
    for faces in dice_rolls:
        counts_key = faces_to_counts_tuple(faces)
        score, used, _, sf, so = _score_by_counts(counts_key)
        if score == 0:
            continue
        out.append(
            ScoreCandidate(
                faces=list(faces),
                dice_len=len(faces),
                score=score,
                used=used,
                counts=counts_key,
                single_fives=sf,
                single_ones=so,
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- #
# 6.  Smart-discard logic (unchanged externally, tuple-friendly inside)
# --------------------------------------------------------------------------- #


def _must_bank(
    score_after: int,
    dice_left_after: int,
    *,
    score_threshold: int,
    dice_threshold: int,
    consider_score: bool,
    consider_dice: bool,
    require_both: bool,
) -> bool:
    """Return True if thresholds force a bank."""
    hit_score = (score_after >= score_threshold) if consider_score else False
    hit_dice = (dice_left_after <= dice_threshold) if consider_dice else False
    return (
        (hit_score and hit_dice)
        if (consider_score and consider_dice and require_both)
        else (hit_score or hit_dice)
    )


def _select_candidate(
    candidates: tuple[
        tuple[list[int], int, int, int, SixFaceCounts, int, int],
        ...,
    ],
    *,
    turn_score_pre: int,
    dice_roll_len: int,
    counts: SixFaceCounts,
    single_fives: int,
    single_ones: int,
    favor_dice_or_score: Union[FavorDiceOrScore, bool] = FavorDiceOrScore.SCORE,
    must_bank: Callable[[int, int], bool],
) -> tuple[int, int] | None:
    """Pick the best discard option from scored candidates."""
    best_key: tuple[int, int] | None = None
    best_sf, best_so = single_fives, single_ones

    for _roll, _len, cand_score, cand_used, cand_cnt, cand_sf, cand_so in candidates:
        # Skip options that discard more single 5s or 1s than exist in the
        # original roll. This prevents breaking up scoring sets which would
        # otherwise inflate the post-discard single counts and lead to
        # negative discard totals.
        drop_5 = counts[4] - cand_cnt[4]
        drop_1 = counts[0] - cand_cnt[0]
        if drop_5 > single_fives or drop_1 > single_ones:
            continue

        score_after = (
            turn_score_pre + cand_score
        )  # score entering turn plus candidate score under evaluation
        dice_left_after = (
            dice_roll_len - cand_used
        )  # Dice remaining after candidate under evaluation is scored
        if must_bank(
            score_after, dice_left_after
        ):  # If must_bank returns true, smart rules don't apply to this candidate
            continue

        _prefer_score = (  # convert FavorDiceOrScore attribute into usable boolean
            favor_dice_or_score is True
            or favor_dice_or_score is FavorDiceOrScore.SCORE  # bool vs enum normalization
        )
        key = (  # favor_dice_or_score prioritization encoded in tuple position
            (score_after, dice_left_after)
            if _prefer_score
            else (
                dice_left_after,
                score_after,
            )
        )
        if best_key is None or key > best_key:  # compare candidate key to current best key
            best_key = key  # Holding current best key (dice-score or score-dice outcome)
            best_sf, best_so = (
                cand_sf,
                cand_so,
            )  # if a new best was found, record the candidate single five and single one counts

    if best_key is None:  # every path banks → keep everything
        return None
    return (
        best_sf,
        best_so,
    )  # discards will be applied by subtracting these from original single fives and ones


def _decide_smart_discards_impl(
    *,
    counts: SixFaceCounts,
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
    favor_dice_or_score: Union[FavorDiceOrScore, bool] = FavorDiceOrScore.SCORE,
) -> Tuple[int, int]:
    """Determine how many single 5s and 1s to throw back.

    Note that ``smart_one=True`` only has an effect when ``smart_five`` is
    also ``True``. Smart-1 discards are ignored otherwise.

    This is the implementation that gets run by decide_smart_discards which exists
    to cache outcomes for a small speed boost.

    Inputs
    ------
    counts (SixFaceCounts):
        Face counts for the roll.
    single_fives (int):
        Number of single fives available.
    single_ones (int):
        Number of single ones available.
    raw_score (int):
        Score before discarding.
    raw_used (int):
        Dice used before discarding.
    dice_roll_len (int):
        Number of dice rolled.
    turn_score_pre (int):
        Score already accumulated this turn.
    score_threshold (int):
        Minimum score before banking.
    dice_threshold (int):
        Maximum dice left before banking.
    smart_five (bool):
        Enable Smart-5 heuristic.
    smart_one (bool):
        Enable Smart-1 heuristic. Always False if ``smart_five`` is False.
    consider_score (bool, optional):
        Whether to check score_threshold.
    consider_dice (bool, optional):
        Whether to check dice_threshold.
    require_both (bool, optional):
        Bank only if both thresholds are hit.
    favor_dice_or_score (bool, optional):
        Break ties in favor of higher score.

    Returns
    -------
    tuple[int, int]:
        (discard_fives, discard_ones).
    """
    if not smart_five or raw_used == dice_roll_len or (single_fives == 0 and single_ones == 0):
        return 0, 0

    def must_bank(score_after: int, dice_left_after: int) -> bool:
        return _must_bank(
            score_after,
            dice_left_after,
            score_threshold=score_threshold,
            dice_threshold=dice_threshold,
            consider_score=consider_score,
            consider_dice=consider_dice,
            require_both=require_both,
        )

    candidates = score_lister(generate_sequences(counts, smart_one=smart_one))
    best = _select_candidate(
        candidates,
        turn_score_pre=turn_score_pre,
        dice_roll_len=dice_roll_len,
        counts=counts,
        single_fives=single_fives,
        single_ones=single_ones,
        favor_dice_or_score=favor_dice_or_score,
        must_bank=must_bank,
    )

    if best is None:
        return 0, 0

    best_sf, best_so = best
    # Subtract the remaining singles from the originals to determine how many
    # dice to throw back. The candidate filtering above guarantees these values
    # never exceed the available singles, so the result is always non-negative.
    return single_fives - best_sf, single_ones - best_so


@functools.lru_cache(maxsize=131_072)
def decide_smart_discards(
    *,
    counts: SixFaceCounts,
    single_fives: int,
    single_ones: int,
    raw_score: int,
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
    favor_dice_or_score: Union[FavorDiceOrScore, bool] = FavorDiceOrScore.SCORE,
) -> tuple[int, int]:
    """Caches results of the implementation function, _decide_smart_discards_impl"""
    return _decide_smart_discards_impl(
        counts=counts,
        single_fives=single_fives,
        single_ones=single_ones,
        raw_score=raw_score,
        raw_used=raw_used,
        dice_roll_len=dice_roll_len,
        turn_score_pre=turn_score_pre,
        score_threshold=score_threshold,
        dice_threshold=dice_threshold,
        smart_five=smart_five,
        smart_one=smart_one,
        consider_score=consider_score,
        consider_dice=consider_dice,
        require_both=require_both,
        favor_dice_or_score=favor_dice_or_score,
    )


def apply_discards(
    raw_score: int,
    raw_used: int,
    discard_fives: int,
    discard_ones: int,
    dice_roll_len: int,
) -> Tuple[int, int, int]:
    """Apply the discard decision to the raw score.

    Inputs
    ------
    raw_score (int):
        Score before discards.
    raw_used (int):
        Dice used before discards.
    discard_fives (int):
        Number of single fives discarded.
    discard_ones (int):
        Number of single ones discarded.
    dice_roll_len (int):
        Total dice rolled.

    Returns
    -------
    tuple[int, int, int]:
        (final_score, final_used, dice_to_reroll).
    """
    final_score = raw_score - 50 * discard_fives - 100 * discard_ones
    final_used = raw_used - discard_fives - discard_ones
    final_reroll = dice_roll_len - final_used
    return final_score, final_used, final_reroll


# --------------------------------------------------------------------------- #
# 7.  High-level public API
# --------------------------------------------------------------------------- #
def default_score(
    dice_roll: DiceRoll,
    *,
    turn_score_pre: int,
    smart_five: bool = False,
    smart_one: bool = False,
    consider_score: bool = True,
    consider_dice: bool = True,
    require_both: bool = False,
    score_threshold: int = 300,
    dice_threshold: int = 3,
    favor_dice_or_score: Union[FavorDiceOrScore, bool] = FavorDiceOrScore.SCORE,
    return_discards: bool = False,  # reports if 5's or 1's were discarded at all
) -> Tuple[int, int, int] | Tuple[int, int, int, int, int]:
    """Score a roll and apply Smart discard heuristics.

    ``smart_one=True`` only has an effect when ``smart_five`` is ``True``.
    Otherwise the Smart-1 logic is skipped.

    Inputs
    ------
    dice_roll (DiceRoll):
        List of faces in the roll.
    turn_score_pre (int):
        Score already accumulated this turn.
    smart_five (bool, optional):
        Enable Smart-5 discard.
    smart_one (bool, optional):
        Enable Smart-1 discard. Ignored unless ``smart_five`` is True.
    consider_score (bool, optional):
        Whether to respect score_threshold.
    consider_dice (bool, optional):
        Whether to respect dice_threshold.
    require_both (bool, optional):
        Bank only if both thresholds are hit.
    score_threshold (int, optional):
        Minimum score before banking.
    dice_threshold (int, optional):
        Maximum dice left before banking.
    favor_dice_or_score (bool, optional):
        Break ties in favor of higher score.
    return_discards (bool, optional):
        If True, include the number of discarded 5s and 1s in the result.

    Returns
    -------
    tuple[int, int, int]:
        Returned when ``return_discards`` is ``False`` as
        ``(final_score, final_used, final_reroll)``.
    tuple[int, int, int, int, int]:
        Returned when ``return_discards`` is ``True`` as
        ``(final_score, final_used, final_reroll, discarded_fives, discarded_ones)``.
    """
    raw_score, raw_used, counts_key, sfives, sones = score_roll_cached(tuple(dice_roll))

    d5, d1 = decide_smart_discards(
        counts=counts_key,
        single_fives=sfives,
        single_ones=sones,
        raw_score=raw_score,
        raw_used=raw_used,
        dice_roll_len=len(dice_roll),
        turn_score_pre=turn_score_pre,
        score_threshold=score_threshold,
        dice_threshold=dice_threshold,
        smart_five=smart_five,
        smart_one=smart_one,
        consider_score=consider_score,
        consider_dice=consider_dice,
        require_both=require_both,
        favor_dice_or_score=favor_dice_or_score,
    )

    final_scoring_info = apply_discards(raw_score, raw_used, d5, d1, len(dice_roll))
    return (*final_scoring_info, d5, d1) if return_discards else final_scoring_info
