# src/farkle/scoring.py   –  100 % tuple-based, Numba-compatible
from __future__ import annotations

import functools
from typing import Callable, Sequence, Tuple, cast, NamedTuple

# Numba is only used in the low-level helpers;
# no caller needs to install it explicitly.
import numba as nb
import numpy as np

from farkle.scoring_lookup import evaluate as _eval_nb  # fast JIT core
from farkle.types import Counts6, FacesT, Int64Arr1D, DiceRoll

# --------------------------------------------------------------------------- #
# 0.  Public type alias
# --------------------------------------------------------------------------- #


class ScoreCandidate(NamedTuple):
    """Container for potential scoring outcomes after discards."""

    faces: list[int]
    dice_len: int
    score: int
    used: int
    counts: Counts6
    single_fives: int
    single_ones: int


# --------------------------------------------------------------------------- #
# 1.  Tiny helpers – all immutable / hash-friendly
# --------------------------------------------------------------------------- #
@nb.njit(cache=True)
def _faces_to_counts_nb(faces: np.ndarray) -> Counts6:
    """Count occurrences of each face value.

    Inputs
    ------
    faces (np.ndarray):
        1-D array of dice faces.

    Returns
    -------
    Counts6:
        Tuple of counts for faces one through six.
    """
    out = np.zeros(6, dtype=np.int64)
    for v in faces:
        out[v - 1] += 1
    return (int(out[0]), int(out[1]), int(out[2]), int(out[3]), int(out[4]), int(out[5]))


def faces_to_counts_tuple(faces: Sequence[int]) -> Counts6:
    """Convert a sequence of faces to a counts tuple.

    Inputs
    ------
    faces (Sequence[int]):
        Iterable of dice faces.

    Returns
    -------
    Counts6:
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
def _score_by_counts(key: Counts6) -> Tuple[int, int, Counts6, int, int]:
    """Score a roll represented by face counts.

    Inputs
    ------
    key (Counts6):
        Tuple of counts for faces one through six.

    Returns
    -------
    tuple[int, int, Counts6, int, int]:
        (score, used, counts_key, single_fives, single_ones).
    """
    score, used, sfives, sones = _eval_nb(key)  # JIT kernel
    return score, used, key, sfives, sones


def score_roll_cached(roll: Sequence[int]) -> Tuple[int, int, Counts6, int, int]:
    """Score a roll, caching by its face counts.

    Inputs
    ------
    roll (Sequence[int]):
        Iterable of dice faces.

    Returns
    -------
    tuple[int, int, Counts6, int, int]:
        (score, used, counts_key, single_fives, single_ones).
    """
    key = faces_to_counts_tuple(roll)
    return _score_by_counts(key)


# --------------------------------------------------------------------------- #
# 3.  Expand counts → sorted faces  (needed for Smart discards)
# --------------------------------------------------------------------------- #
Int64Arr1DNP = np.ndarray  # local alias to keep the signature tidy


# 1) **FAST kernel** – stays in Numba, but now returns an ndarray
@nb.njit(cache=True)
def _expand_sorted_nb(c1: int, c2: int, c3: int, c4: int, c5: int, c6: int) -> Int64Arr1D:
    """Expand counts into a sorted NumPy array.

    Inputs
    ------
    c1, c2, c3, c4, c5, c6 (int):
        Counts for faces one through six.

    Returns
    -------
    Int64Arr1D:
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
def _expand_sorted(counts: Counts6) -> FacesT:
    """Return sorted faces as an immutable tuple.

    Inputs
    ------
    counts (Counts6):
        Counts for faces one through six.

    Returns
    -------
    FacesT:
        Tuple of face values in ascending order.
    """
    return tuple(_expand_sorted_nb(*counts))


# --------------------------------------------------------------------------- #
# 4.  Enumerate every post-discard state  (cached)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4096)
def generate_sequences(counts: Counts6, *, smart_one: bool = False) -> tuple[FacesT, ...]:
    """Enumerate all post-discard face sequences.

    Inputs
    ------
    counts (Counts6):
        Counts representing the roll.
    smart_one (bool, optional):
        Include sequences discarding single ones.

    Returns
    -------
    tuple[FacesT, ...]:
        All possible remaining dice as sorted tuples.
    """
    c = list(counts)  # mutable copy
    seqs: list[FacesT] = [_expand_sorted(counts)]

    while c[4]:  # index 4 == face-value 5
        c[4] -= 1
        seqs.append(_expand_sorted(cast(Counts6, tuple(c))))

    if smart_one:
        while c[0]:  # index 0 == face-value 1
            c[0] -= 1
            seqs.append(_expand_sorted(cast(Counts6, tuple(c))))

    return tuple(seqs)


# --------------------------------------------------------------------------- #
# 5.  Score a batch of rolls (tuples of faces) – cached
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4096)
def score_lister(
    dice_rolls: tuple[FacesT, ...],
) -> tuple[ScoreCandidate, ...]:
    """Score multiple sorted rolls.

    Inputs
    ------
    dice_rolls (tuple[FacesT, ...]):
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
        tuple[list[int], int, int, int, Counts6, int, int],
        ...,
    ],
    *,
    turn_score_pre: int,
    dice_roll_len: int,
    single_fives: int,
    single_ones: int,
    prefer_score: bool,
    must_bank: Callable[[int, int], bool],
) -> tuple[int, int] | None:
    """Pick the best discard option from scored candidates."""
    best_key: tuple[int, int] | None = None
    best_sf, best_so = single_fives, single_ones

    for _roll, _len, cand_score, cand_used, _cnt, cand_sf, cand_so in candidates:
        score_after = turn_score_pre + cand_score
        dice_left_after = dice_roll_len - cand_used
        if must_bank(score_after, dice_left_after):
            continue

        key = (
            (score_after, dice_left_after)
            if prefer_score
            else (
                dice_left_after,
                score_after,
            )
        )
        if best_key is None or key > best_key:
            best_key = key
            best_sf, best_so = cand_sf, cand_so

    if best_key is None:  # every path banks → keep everything
        return None
    return best_sf, best_so


def decide_smart_discards(
    *,
    counts: Counts6,
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
    """Determine how many single 5s and 1s to throw back.

    Note that ``smart_one=True`` only has an effect when ``smart_five`` is
    also ``True``. Smart‑1 discards are ignored otherwise.

    Inputs
    ------
    counts (Counts6):
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
    prefer_score (bool, optional):
        Break ties in favour of higher score.

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
        single_fives=single_fives,
        single_ones=single_ones,
        prefer_score=prefer_score,
        must_bank=must_bank,
    )

    if best is None:
        return 0, 0

    best_sf, best_so = best
    return single_fives - best_sf, single_ones - best_so


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
    prefer_score: bool = True,
    return_discards: bool = False,  # reports if 5's or 1's were discarded at all
) -> Tuple[int, int, int] | Tuple[int, int, int, int, int]:
    """Score a roll and apply Smart discard heuristics.

    ``smart_one=True`` only has an effect when ``smart_five`` is ``True``.
    Otherwise the Smart‑1 logic is skipped.

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
    prefer_score (bool, optional):
        Break ties in favour of higher score.
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
        prefer_score=prefer_score,
    )

    final_scoring_info = apply_discards(raw_score, raw_used, d5, d1, len(dice_roll))
    return (*final_scoring_info, d5, d1) if return_discards else final_scoring_info


# --------------------------------------------------------------------------- #
# 8.  Legacy shim (kept for unit-test parity)
# --------------------------------------------------------------------------- #
def _compute_raw_score(dice_roll: DiceRoll):
    """Legacy scoring helper for backwards-compatibility tests.

    Inputs
    ------
    dice_roll (DiceRoll):
        Sequence of faces to score.

    Returns
    -------
    tuple[int, int, Counts6, int, int]:
        Raw scoring tuple from :func:`score_roll_cached`.
    """
    return score_roll_cached(tuple(dice_roll))
