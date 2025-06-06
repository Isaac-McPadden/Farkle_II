from __future__ import annotations

import ast
import csv
from collections import Counter  # noqa: F401 Counter objects loaded from csv
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from farkle.scoring import (
    _compute_raw_score,
    apply_discards,
    decide_smart_discards,
    # decide_smart_discards,
    default_score,
    expand_sorted,
    generate_sequences,
    score_lister,
    score_roll_cached,
)
from farkle.scoring_lookup import build_score_lookup_table, evaluate, score_roll

# ---------------------------------------------------------------------------
# Location of the test data CSV
# tests/unit/  ->  tests/data/
# ---------------------------------------------------------------------------

def _make_csv_loader(filename: str):
    """
    Returns a generator that yields pytest.param tuples for each CSV row.

    The CSV file is expected to live in 'tests/data/<filename>'.
    This helper will raise FileNotFoundError immediately if the CSV is missing.
    """
    # 1) Build an absolute path to tests/data/filename
    repo_root = Path(__file__).resolve().parents[1]  # one above 'tests/'
    data_path = repo_root / "data" / filename

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path!r} - "
            "make sure your CSV lives in tests/data/ and is spelled correctly."
        )

    def _loader():
        with data_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader, start=1):
                yield idx, row

    return _loader



def load_score_cases():
    """
    Load test_farkle_scores_data.csv → yields pytest.param tuples of the form:
    (roll: List[int], Score: int, Used: int, Reroll: int, Single_Fives: int, Single_Ones: int, id=“row###:<roll>”)
    """
    loader = _make_csv_loader("test_farkle_scores_data.csv")
    cases = []
    for idx, row in loader():
        try:
            roll = ast.literal_eval(row["Dice_Roll"])
        except Exception as e:
            raise ValueError(f"Row {idx}: cannot parse Dice_Roll={row['Dice_Roll']!r}") from e

        try:
            score       = int(row["Score"])
            used        = int(row["Used_Dice"])
            reroll      = int(row["Reroll_Dice"])
            single_fives= int(row["Single_Fives"])
            single_ones = int(row["Single_Ones"])
        except Exception as e:
            raise ValueError(f"Row {idx}: invalid numeric field - {e}") from e

        param_id = f"row{idx:03d}:{roll}"
        cases.append(
            pytest.param(roll, score, used, reroll, single_fives, single_ones, id=param_id)
        )
    return cases



# --------------------------------------------------------------------------- #
# Helper: fill in the *rarely-changed* kwargs so every test only specifies
# what is strictly relevant to the branch it targets.
# --------------------------------------------------------------------------- #
def _call(
    roll,
    *,
    turn_pre       = 0,
    score_thr      = 300,
    dice_thr       = 2,
    smart_five     = True,
    smart_one      = False,
    consider_score = True,
    consider_dice  = True,
    require_both   = False,
    prefer_score   = True,
):
    raw_s, raw_u, cnts, sf, so = score_roll_cached(roll)
    return decide_smart_discards(
        counts          = cnts,
        single_fives    = sf,
        single_ones     = so,
        raw_score       = raw_s,          # kept for API parity
        raw_used        = raw_u,
        dice_roll_len   = len(roll),
        turn_score_pre  = turn_pre,
        score_threshold = score_thr,
        dice_threshold  = dice_thr,
        smart_five      = smart_five,
        smart_one       = smart_one,
        consider_score  = consider_score,
        consider_dice   = consider_dice,
        require_both    = require_both,
        prefer_score    = prefer_score,
    )

# --------------------------------------------------------------------------- #
# 1) early-exit guards  (three independent paths)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kw, expected",
    [
        # a) smart-5 disabled
        ({"smart_five": False, "roll": [5, 2, 3, 4, 6]},
         (0, 0)),
        # b) hot dice  (all six score → raw_used == len)
        ({"roll": [2,2,2,3,3,3]},          # 200 + 300
         (0, 0)),
        # c) no single 1/5 present
        ({"roll": [2,2,2,3,4,6]},          # triples + 3 loose bust dice
         (0, 0)),
    ],
)
def test_early_exits(kw, expected):
    assert _call(**kw) == expected


# --------------------------------------------------------------------------- #
# 2) _must_bank filter — branch on `require_both`
#    *With AND-logic the candidate passes; OR-logic kills it*
# --------------------------------------------------------------------------- #
def test_require_both_logic():
    roll = [5, 5, 2, 3, 4]          # 100 pts, uses 2 dice, 3 left
    # (a) AND-logic → keep rolling ⇒ no discard
    keep = _call(roll, turn_pre=290, score_thr=300, dice_thr=2,
                 require_both=True)
    # (b) OR-logic  → every candidate must bank ⇒ default (0,0)
    bank = _call(roll, turn_pre=290, score_thr=300, dice_thr=2,
                 require_both=False)
    assert keep == (0, 0) and bank == (0, 0)   # different paths, same result
    # sanity check that *a* really went through AND logic:
    # OR-logic would have discarded one 5 (see next test)


# --------------------------------------------------------------------------- #
# 3) prefer-score  vs  prefer-dice
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "prefer_score, expected",
    [
        (True,  (0, 0)),   # choose higher score → keep both 5s
        (False, (1, 0)),   # choose more dice   → throw one 5 back
    ],
)
def test_prefer_score_vs_dice(prefer_score, expected):
    roll = [5, 5, 2, 3, 4]
    out = _call(roll, prefer_score=prefer_score,
                dice_thr=2, score_thr=300)  # low thresholds keep options open
    assert out == expected


# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "roll, exp_score, exp_used, exp_reroll, exp_sfives, exp_sones",
    list(load_score_cases()),
)
def test_compute_raw_score(
    roll, exp_score, exp_used, exp_reroll, exp_sfives, exp_sones
):
    score, used, _counts, single_fives, single_ones = _compute_raw_score(roll)

    assert score == exp_score
    assert used == exp_used
    assert single_fives == exp_sfives
    assert single_ones == exp_sones
    assert len(roll) - used == exp_reroll

@pytest.mark.parametrize(
    "roll, exp_score, exp_used, exp_reroll, exp_sfives, exp_sones",
    list(load_score_cases()),
)
def test_score_roll_cached(
    roll, exp_score, exp_used, exp_reroll, exp_sfives, exp_sones
):
    score, used, _counts, single_fives, single_ones = score_roll_cached(roll)

    assert score == exp_score
    assert used == exp_used
    assert single_fives == exp_sfives
    assert single_ones == exp_sones
    assert len(roll) - used == exp_reroll
    


# ───────────────────────── expand_sorted ─────────────────────────
@pytest.mark.parametrize(
    "counts, expected",
    [
        ({5: 2, 2: 3},          [2, 2, 2, 5, 5]),
        ({1: 2, 3: 1, 5: 1},    [1, 1, 3, 5]),
        ({},                    []),
    ],
)
def test_expand_sorted(counts, expected):
    assert expand_sorted(counts) == expected


# ─────────────────────── generate_sequences ──────────────────────
@pytest.mark.parametrize(
    "counts, smart_one, expected",
    [
        # Example from the spec
        ({2: 3, 5: 2}, False,
         [[2, 2, 2, 5, 5],
          [2, 2, 2, 5],
          [2, 2, 2]]),

        ({1: 2, 2: 1, 3: 1, 5: 2}, True,
         [[1, 1, 2, 3, 5, 5],
          [1, 1, 2, 3, 5],
          [1, 1, 2, 3],
          [1, 2, 3],
          [2, 3]]),
    ],
)
def test_generate_sequences(counts, smart_one, expected):
    assert generate_sequences(counts, smart_one=smart_one) == expected


# ───────────────────────── score_lister ──────────────────────────
def test_score_lister_filters_busts():
    rolls = [[2, 3, 4, 6], [5, 5]]          # first is a bust, second scores 100
    listed = score_lister(rolls)
    # Only one surviving entry and its score is 100
    assert len(listed) == 1
    *_, raw_score, raw_used, counts, sf, so = listed[0]
    assert raw_score == 100 and raw_used == 2
    assert sf == 2 and so == 0
    

def test_compute_raw_score_cleans_up_zero_keys():
    # 4-of-a-kind plus a lonely 1 → after scoring the quad, face 2 count is 0
    roll = [2, 2, 2, 2, 1]
    score, used, *_ = _compute_raw_score(roll.copy())
    assert (score, used) == (1100, 5)          # 1000 + 100

    # The temporary Counter inside the function was cleaned,
    # but the original 'counts' must be untouched.
    assert roll == [2, 2, 2, 2, 1]
    
def test_score_roll_wrapper_agrees_with_evaluate():
    roll = [1, 5, 5, 2, 2, 2]          # 200 + 100 + 50 = 350
    assert score_roll(roll) == tuple(evaluate(Counter(roll))[:2])
    
@pytest.mark.parametrize(
    "raw_score,raw_used,discard5,discard1,dice_len,expected",
    [
        # A) no discards
        (250, 3, 0, 0, 5, (250, 3, 2)),
        # B) fives only
        (250, 3, 2, 0, 5, (150, 1, 4)),
        # C) ones only
        (300, 4, 0, 2, 6, (100, 2, 4)),
        # D) both together
        (500, 6, 2, 1, 6, (300, 3, 3)),
    ],
)
def test_apply_discards(raw_score, raw_used, discard5, discard1, dice_len, expected):
    assert apply_discards(raw_score, raw_used, discard5, discard1, dice_len) == expected


@given(st.lists(st.integers(min_value=1, max_value=6),
                min_size=1, max_size=6))
def test_default_score_invariants(roll):
    # Pass required keyword arg; keep smart-flags off for a neutral baseline
    score, used, reroll = default_score(
        dice_roll      = roll,
        turn_score_pre = 0,          # <- new
        smart_five     = False,
        smart_one      = False,
        # score_threshold left at default 300
    )

    # --- invariants -------------------------------------------------
    # 1) dice are partitioned correctly
    assert used + reroll == len(roll)
    assert 0 <= used <= len(roll)

    # 2) a bust never reports used dice
    if score == 0:
        assert used == 0

        
# ------------------------------------------------------------------
# Build every 6-die roll in which *all six dice score*.
# ------------------------------------------------------------------
table = build_score_lookup_table()

def counts_to_list(counts: tuple[int, int, int, int, int, int]) -> list[int]:
    result: list[int] = []
    for idx, how_many in enumerate(counts):
        value = idx + 1
        # Add “value” exactly how_many times
        result.extend([value] * how_many)
    return result

_HOT_DICE_ROLLS = [
    counts_to_list(multiset)   # expand!
    for multiset, info in table.items()
    if info[1] == 6
]

_HOT_ROLL_STRAT = st.sampled_from(_HOT_DICE_ROLLS)
# ------------------------------------------------------------------

@given(roll=_HOT_ROLL_STRAT)
def test_hot_dice_discard_never_trims_six(roll):
    """
    When every die in the roll already scores, smart-discard logic must keep
    them all:  used == 6  and reroll == 0.
    """
    score, used, reroll = default_score(roll, turn_score_pre=0)
    assert used == 6
    assert reroll == 0
    

@pytest.mark.parametrize(
    "dice_roll, turn_pre, threshold, smart5, smart1, expected",
    [
        # 1) plain roll, straight 1-6 → 1 500 pts, all six dice used
        ([1, 2, 3, 4, 5, 6], 0, 300, False, False,
         (1500, 6, 0)),

        # 2) Smart-5 only: every discard would force a bank → keep all dice
        ([5, 5, 1, 2],      0, 300, True,  False,
         (200, 3, 1)),

        # 3) Smart-5 + Smart-1: only roll that avoids banking is
        #    “discard both 5s and one 1” → leaves 4 dice to roll
        ([5, 5, 1, 1, 2],   0, 300, True,  True,
         (100, 1, 4)),

        # 4) Already at threshold, so the lone 5 is kept (no discard)
        ([5],               300, 300, True, True,
         (50, 1, 0)),
    ],
)
def test_default_score_cases(
    dice_roll, turn_pre, threshold, smart5, smart1, expected
):
    out = default_score(
        dice_roll       = dice_roll,
        turn_score_pre  = turn_pre,
        smart_five      = smart5,
        smart_one       = smart1,
        score_threshold = threshold,
        dice_threshold  = 3,      # default
    )
    assert out == expected