# src/farkle/scoring.py   –  100 % tuple-based, Numba-compatible
from __future__ import annotations

import functools
from typing import List, Sequence, Tuple, cast

# Numba is only used in the low-level helpers;
# no caller needs to install it explicitly.
import numba as nb
import numpy as np

from farkle.scoring_lookup import evaluate as _eval_nb  # fast JIT core
from farkle.types import Counts6, FacesT, Int64Arr1D

# --------------------------------------------------------------------------- #
# 0.  Public type alias
# --------------------------------------------------------------------------- #
DiceRoll = List[int]                 # a raw roll as a list of faces

# --------------------------------------------------------------------------- #
# 1.  Tiny helpers – all immutable / hash-friendly
# --------------------------------------------------------------------------- #
@nb.njit(cache=True)
def _faces_to_counts_nb(faces: np.ndarray) -> Counts6:
    """Vectorised `bincount` → 6-tuple (ones … sixes)."""
    out = np.zeros(6, dtype=np.int64)
    for v in faces:              # Numba for-loops are fine
        out[v - 1] += 1
    return (int(out[0]), int(out[1]), int(out[2]),
            int(out[3]), int(out[4]), int(out[5]))


def faces_to_counts_tuple(faces: Sequence[int]) -> Counts6:
    """Pure-Python wrapper so we can feed lru_cache keys."""
    return _faces_to_counts_nb(np.asarray(faces, dtype=np.int64))


# --------------------------------------------------------------------------- #
# 2.  Fast single-roll scorer (cached)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=32_768)
def _score_by_counts(key: Counts6) -> Tuple[int, int, Counts6, int, int]:
    """
    Return (score, used, counts_key, single_fives, single_ones)
    """
    score, used, sfives, sones = _eval_nb(key)        # JIT kernel
    return score, used, key, sfives, sones


def score_roll_cached(roll: Sequence[int]) -> Tuple[int, int, Counts6, int, int]:
    """
    Public helper – accepts *either* list[int] or tuple[int,…] of faces.
    """
    key = faces_to_counts_tuple(roll)
    return _score_by_counts(key)


# --------------------------------------------------------------------------- #
# 3.  Expand counts → sorted faces  (needed for Smart discards)
# --------------------------------------------------------------------------- #
Int64Arr1DNP = np.ndarray  # local alias to keep the signature tidy
# 1) **FAST kernel** – stays in Numba, but now returns an ndarray
@nb.njit(cache=True)
def _expand_sorted_nb(c1: int, c2: int, c3: int,
                      c4: int, c5: int, c6: int) -> Int64Arr1D:
    """Return the faces in ascending order as a 1-D int64 array."""
    n_tot = c1 + c2 + c3 + c4 + c5 + c6
    out   = np.empty(n_tot, dtype=np.int64)
    pos   = 0
    for face, n in enumerate((c1, c2, c3, c4, c5, c6), 1):
        for _ in range(n):
            out[pos] = face
            pos += 1
    return out

# 2) **Thin wrapper** – cheap Python, converts to an immutable tuple
def _expand_sorted(counts: Counts6) -> FacesT:
    return tuple(_expand_sorted_nb(*counts))


# --------------------------------------------------------------------------- #
# 4.  Enumerate every post-discard state  (cached)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4096)
def generate_sequences(counts: Counts6, *, smart_one: bool = False) -> tuple[FacesT, ...]:
    c = list(counts)                        # mutable copy
    seqs: list[FacesT] = [_expand_sorted(counts)]

    while c[4]:                             # index 4 == face-value 5
        c[4] -= 1
        seqs.append(_expand_sorted(cast(Counts6, tuple(c))))   # Ruff happy

    if smart_one:
        while c[0]:                         # index 0 == face-value 1
            c[0] -= 1
            seqs.append(_expand_sorted(cast(Counts6, tuple(c))))

    return tuple(seqs)


# --------------------------------------------------------------------------- #
# 5.  Score a batch of rolls (tuples of faces) – cached
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4096)
def score_lister(
    dice_rolls: tuple[FacesT, ...],
) -> tuple[
        tuple[list[int],     # original roll (sorted)
              int,           # dice_len
              int,           # cand_score
              int,           # cand_used
              Counts6,       # counts
              int,           # cand_sf
              int],          # cand_so
        ...
   ]:
    """
    Return tuples:
        (roll_faces, roll_len, score, used, counts_key, lone_5s, lone_1s)
    skipping any non-scoring roll.
    """
    out = []
    for faces in dice_rolls:
        counts_key = faces_to_counts_tuple(faces)
        score, used, _, sf, so = _score_by_counts(counts_key)
        if score == 0:
            continue
        out.append((list(faces), len(faces), score, used, counts_key, sf, so))
    return tuple(out)


# --------------------------------------------------------------------------- #
# 6.  Smart-discard logic (unchanged externally, tuple-friendly inside)
# --------------------------------------------------------------------------- #
def decide_smart_discards(
    *,
    counts: Counts6,
    single_fives: int,
    single_ones:  int,
    raw_score:    int,  # noqa: ARG001
    raw_used:     int,
    dice_roll_len: int,
    turn_score_pre: int,
    score_threshold: int,
    dice_threshold:  int,
    smart_five: bool,
    smart_one:  bool,
    consider_score: bool = True,
    consider_dice:  bool = True,
    require_both:   bool = False,
    prefer_score:   bool = True,
) -> Tuple[int, int]:

    if not smart_five or raw_used == dice_roll_len or (single_fives == 0 and single_ones == 0):
        return 0, 0

    def _must_bank(score_after: int, dice_left_after: int) -> bool:
        hit_score = (score_after >= score_threshold) if consider_score else False
        hit_dice  = (dice_left_after <= dice_threshold) if consider_dice else False
        return (hit_score and hit_dice) if (consider_score and consider_dice and require_both) else (hit_score or hit_dice)

    candidates = score_lister(generate_sequences(counts, smart_one=smart_one))

    best_key: Tuple[int, int] | None = None
    best_sf, best_so = single_fives, single_ones

    for (_roll, _len, cand_score, cand_used,
         _cnt, cand_sf, cand_so) in candidates:

        score_after     = turn_score_pre + cand_score
        dice_left_after = dice_roll_len - cand_used
        if _must_bank(score_after, dice_left_after):
            continue

        key = (score_after, dice_left_after) if prefer_score else (dice_left_after, score_after)
        if best_key is None or key > best_key:
            best_key = key
            best_sf, best_so = cand_sf, cand_so

    if best_key is None:                 # every path banks → keep everything
        return 0, 0

    return single_fives - best_sf, single_ones - best_so


def apply_discards(
    raw_score:     int,
    raw_used:      int,
    discard_fives: int,
    discard_ones:  int,
    dice_roll_len: int,
) -> Tuple[int, int, int]:
    final_score  = raw_score - 50 * discard_fives - 100 * discard_ones
    final_used   = raw_used  - discard_fives - discard_ones
    final_reroll = dice_roll_len - final_used
    return final_score, final_used, final_reroll


# --------------------------------------------------------------------------- #
# 7.  High-level public API
# --------------------------------------------------------------------------- #
def default_score(
    dice_roll:       DiceRoll,
    *,
    turn_score_pre:  int,
    smart_five:      bool = False,
    smart_one:       bool = False,
    consider_score:  bool = True,
    consider_dice:   bool = True,
    require_both:    bool = False,
    score_threshold: int  = 300,
    dice_threshold:  int  = 3,
    prefer_score:    bool = True,
) -> Tuple[int, int, int]:
    """
    Evaluate a roll *and* apply Smart-discard heuristics in one go.
      Returns (final_score, final_used, final_reroll).
    """
    raw_score, raw_used, counts_key, sfives, sones = score_roll_cached(tuple(dice_roll))

    d5, d1 = decide_smart_discards(
        counts          = counts_key,
        single_fives    = sfives,
        single_ones     = sones,
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
        prefer_score    = prefer_score,
    )

    return apply_discards(
        raw_score, raw_used, d5, d1, len(dice_roll)
    )


# --------------------------------------------------------------------------- #
# 8.  Legacy shim (kept for unit-test parity)
# --------------------------------------------------------------------------- #
def _compute_raw_score(dice_roll: DiceRoll):
    """
    Thin wrapper maintained for backwards-compat tests.
    """
    return score_roll_cached(tuple(dice_roll))