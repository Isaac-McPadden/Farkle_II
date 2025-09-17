# tests/unit/test_scoring.py
from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import List, Tuple

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from farkle.game.scoring import (
    apply_discards,
    decide_smart_discards,
    default_score,
    generate_sequences,
    score_lister,
    score_roll_cached,
)
from farkle.game.scoring_lookup import build_score_lookup_table, evaluate, score_roll

# ────────────────────────────────────────────────────────────────────────────
# Tiny helpers – tuple world
# ────────────────────────────────────────────────────────────────────────────

def faces_to_counts_tuple(faces: List[int]) -> Tuple[int, int, int, int, int, int]:
    """Convert a raw dice list to the 6-tuple (ones … sixes)."""
    counts_tup = tuple(faces.count(face) for face in range(1, 7))
    assert len(counts_tup) == 6
    return counts_tup


def dict_to_counts_tuple(d: dict[int, int]) -> Tuple[int, int, int, int, int, int]:
    """Convert a Counter-like dict to the 6-tuple representation."""
    counts_tup = tuple(d.get(face, 0) for face in range(1, 7))
    assert len(counts_tup) == 6
    return counts_tup


def counts_to_list(counts: Tuple[int, int, int, int, int, int]) -> List[int]:
    """Expand the 6-tuple back to a sorted dice roll list."""
    out: List[int] = []
    for face, n in enumerate(counts, 1):
        out.extend([face] * n)
    return out

# ────────────────────────────────────────────────────────────────────────────
# CSV loader for golden test cases
# ────────────────────────────────────────────────────────────────────────────

def _make_csv_loader(filename: str):
    repo_root = Path(__file__).resolve().parents[2]  # one above 'tests/'
    data_path = repo_root / "data" / filename
    if not data_path.exists():
        raise FileNotFoundError(f"Missing test-data file {data_path!s}")

    def _loader():
        with data_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader, 1):
                yield idx, row

    return _loader


def load_score_cases():
    loader = _make_csv_loader("test_farkle_scores_data.csv")
    cases = []
    for idx, row in loader():
        try:
            roll = ast.literal_eval(row["Dice_Roll"])
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Row {idx}: bad Dice_Roll {row['Dice_Roll']!r}") from exc

        vals = [int(row[f]) for f in
                ("Score", "Used_Dice", "Reroll_Dice", "Single_Fives", "Single_Ones")]
        param_id = f"row{idx:03d}:{roll}"
        cases.append(pytest.param(roll, *vals, id=param_id))
    return cases


# ────────────────────────────────────────────────────────────────────────────
# Convenience wrapper for decide_smart_discards (same as before)
# ────────────────────────────────────────────────────────────────────────────

def _call(
    roll,
    *,
    turn_pre = 0,
    score_thr = 300,
    dice_thr = 2,
    smart_five = True,
    smart_one = False,
    consider_score = True,
    consider_dice = True,
    require_both = False,
    favor_dice_or_score = True,
):
    raw_s, raw_u, counts, sf, so = score_roll_cached(roll)
    return decide_smart_discards(
        counts = counts,
        single_fives = sf,
        single_ones = so,
        raw_score = raw_s,
        raw_used = raw_u,
        dice_roll_len = len(roll),
        turn_score_pre = turn_pre,
        score_threshold = score_thr,
        dice_threshold = dice_thr,
        smart_five = smart_five,
        smart_one = smart_one,
        consider_score = consider_score,
        consider_dice = consider_dice,
        require_both = require_both,
        favor_dice_or_score = favor_dice_or_score,
    )


# ────────────────────────────────────────────────────────────────────────────
# 1) early-exit guards
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "kw, expected",
    [
        ({"smart_five": False, "roll": [5, 2, 3, 4, 6]}, (0, 0)),
        ({"roll": [2, 2, 2, 3, 3, 3]}, (0, 0)), # hot dice
        ({"roll": [2, 2, 2, 3, 4, 6]}, (0, 0)), # no single 1/5
    ],
)
def test_early_exits(kw, expected):
    assert _call(**kw) == expected


# ────────────────────────────────────────────────────────────────────────────
# 2) require_both logic branch
# ────────────────────────────────────────────────────────────────────────────

def test_require_both_logic():
    roll = [5, 5, 2, 3, 4]
    keep = _call(roll, turn_pre=290, score_thr=300, dice_thr=2, require_both=True)
    bank = _call(roll, turn_pre=290, score_thr=300, dice_thr=2, require_both=False)
    assert keep == (0, 0) and bank == (0, 0)


# ────────────────────────────────────────────────────────────────────────────
# 3) favor-score vs favor-dice
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "favor_dice_or_score, expected",
    [(True, (0, 0)), (False, (1, 0))],
)
def test_favor_score_vs_dice(favor_dice_or_score, expected):
    roll = [5, 5, 2, 3, 4]
    out = _call(roll, favor_dice_or_score=favor_dice_or_score, dice_thr=2, score_thr=300)
    assert out == expected


# ────────────────────────────────────────────────────────────────────────────
# 4) Golden CSV – score_roll_cached
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "roll, exp_score, exp_used, exp_reroll, exp_sfives, exp_sones",
    load_score_cases(),
)
def test_score_roll_cached(roll, exp_score, exp_used, exp_reroll, exp_sfives, exp_sones):
    score, used, _counts, sf, so = score_roll_cached(roll)
    assert (score, used, sf, so, len(roll) - used) == (
        exp_score, exp_used, exp_sfives, exp_sones, exp_reroll
    )


# ────────────────────────────────────────────────────────────────────────────
# 5) generate_sequences  &  score_lister  (tuple API)
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "counts_dict, smart_one, expected",
    [
        ({2: 3, 5: 2}, False,
         [[2, 2, 2, 5, 5], [2, 2, 2, 5], [2, 2, 2]]),
        ({1: 2, 2: 1, 3: 1, 5: 2}, True,
         [[1, 1, 2, 3, 5, 5],
          [1, 1, 2, 3, 5],
          [1, 1, 2, 3],
          [1, 2, 3],
          [2, 3]]),
    ],
)
def test_generate_sequences(counts_dict, smart_one, expected):
    counts_key = dict_to_counts_tuple(counts_dict)
    out = [list(seq) for seq in generate_sequences(counts_key, smart_one=smart_one)]
    assert out == expected


def test_score_lister_filters_busts():
    rolls = [[2, 3, 4, 6], [5, 5]]  # first is bust
    tup_rolls = tuple(tuple(r) for r in rolls)
    listed = score_lister(tup_rolls)
    assert len(listed) == 1
    cand = listed[0]
    assert (cand.score, cand.used, cand.single_fives, cand.single_ones) == (
        100,
        2,
        2,
        0,
    )


# ────────────────────────────────────────────────────────────────────────────
# 6) apply_discards
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw_score, raw_used, discard5, discard1, dice_len, expected",
    [
        (250, 3, 0, 0, 5, (250, 3, 2)),
        (250, 3, 2, 0, 5, (150, 1, 4)),
        (300, 4, 0, 2, 6, (100, 2, 4)),
        (500, 6, 2, 1, 6, (300, 3, 3)),
    ],
)
def test_apply_discards(raw_score, raw_used, discard5, discard1, dice_len, expected):
    assert apply_discards(raw_score, raw_used, discard5, discard1, dice_len) == expected


# ────────────────────────────────────────────────────────────────────────────
# 7) default_score invariants (property-based)
# ────────────────────────────────────────────────────────────────────────────

@given(st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=6))
def test_default_score_invariants(roll):
    score, used, reroll = default_score( # type: ignore default score update is backwards compatible
        dice_roll = roll,
        turn_score_pre = 0,
        smart_five = False,
        smart_one = False,
    )
    assert used + reroll == len(roll)
    if score == 0:
        assert used == 0


# ────────────────────────────────────────────────────────────────────────────
# 8) hot-dice sanity: smart discard must never drop scoring dice
# ────────────────────────────────────────────────────────────────────────────

table = build_score_lookup_table()
_HOT_DICE_ROLLS = [
    counts_to_list(k) for k, info in table.items() if info[1] == 6
]
_HOT_ROLL_STRAT = st.sampled_from(_HOT_DICE_ROLLS)


@given(roll=_HOT_ROLL_STRAT)
def test_hot_dice_discard_never_trims_six(roll):
    _score, used, reroll = default_score(roll, turn_score_pre=0) # type: ignore default score update is backwards compatible
    assert used == 6 and reroll == 0


# ────────────────────────────────────────────────────────────────────────────
# 9) quick wrapper vs. low-level evaluate agreement
# ────────────────────────────────────────────────────────────────────────────

def test_score_roll_wrapper_agrees_with_evaluate():
    roll = [1, 5, 5, 2, 2, 2]  # 350 pts
    counts_key = faces_to_counts_tuple(roll)
    assert score_roll(roll) == evaluate(counts_key)[:2]


# ────────────────────────────────────────────────────────────────────────────
# 10) concrete default_score branch cases
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "dice_roll, turn_pre, threshold, smart5, smart1, expected",
    [
        ([1, 2, 3, 4, 5, 6], 0, 300, False, False, (1500, 6, 0)),
        ([5, 5, 1, 2], 0, 300, True, False, (200, 3, 1)),
        ([5, 5, 1, 1, 2], 0, 300, True, True, (100, 1, 4)),
        ([5], 300, 300, True, True, (50, 1, 0)),
    ],
)
def test_default_score_cases(dice_roll, turn_pre, threshold, smart5, smart1, expected):
    out = default_score(
        dice_roll = dice_roll,
        turn_score_pre = turn_pre,
        smart_five = smart5,
        smart_one = smart1,
        score_threshold = threshold,
        dice_threshold = 3,
    )
    assert out == expected


@pytest.mark.parametrize(
    "faces",
    [
        [0, 1, 2],
        [1, 2, 7],
        [-1, 3, 4],
    ],
)
def test_faces_to_counts_tuple_invalid_faces(faces):
    import farkle.game.scoring as scoring

    with pytest.raises(ValueError):
        scoring.faces_to_counts_tuple(faces)
