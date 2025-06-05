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
    default_score,
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
    data_path = repo_root / "tests" / "data" / filename

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



def load_discard_cases():
    """
    Load test_decide_smart_discards.csv → yields a tuple:
       (
         [counts: dict], single_fives: int, single_ones: int,
         raw_score: int, raw_used: int, dice_len: int,
         turn_pre: int, threshold: int,
         smart_five: bool, smart_one: bool,
         expected: (int, int)
       )
    and a parallel list of IDs [ "case01", "case02", ... ].
    """
    loader = _make_csv_loader("test_decide_smart_discards.csv")
    cases = []
    ids   = []
    for idx, row in loader():
        try:
            counts = ast.literal_eval(row["counts"])
        except Exception as e:
            raise ValueError(f"Row {idx}: cannot parse counts={row['counts']!r}") from e

        try:
            single_fives    = int(row["single_fives"])
            single_ones     = int(row["single_ones"])
            raw_score       = int(row["raw_score"])
            raw_used        = int(row["raw_used"])
            dice_len        = int(row["dice_len"])
            turn_pre        = int(row["turn_score_pre"])
            score_threshold = int(row["score_threshold"])
            smart_five_flag = row["smart_five"].strip().lower() == "true"
            smart_one_flag  = row["smart_one"].strip().lower() == "true"
            expected        = ast.literal_eval(row["expected"])
        except Exception as e:
            raise ValueError(f"Row {idx}: invalid field - {e}") from e

        cases.append(
            (
                counts,
                single_fives,
                single_ones,
                raw_score,
                raw_used,
                dice_len,
                turn_pre,
                score_threshold,
                smart_five_flag,
                smart_one_flag,
                expected,
            )
        )
        ids.append(f"case{idx:02d}")

    return cases, ids


# DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "test_farkle_scores_data.csv"
# if not DATA_PATH.exists():  # fail fast with a helpful hint
#     raise FileNotFoundError(
#         f"Could not find {DATA_PATH} - is the CSV in tests/data/?"
#     )

# # ---------------------------------------------------------------------------
# # Turn each CSV row into a pytest.param
# # ---------------------------------------------------------------------------
# def _load_cases():
#     with DATA_PATH.open(newline="", encoding="utf-8") as fh:
#         reader = csv.DictReader(fh)
#         for idx, row in enumerate(reader, start=1):
#             try:
#                 roll = ast.literal_eval(row["Dice_Roll"])
#             except Exception as e:  # bad “[1, 5]” string?
#                 raise ValueError(
#                     f"Row {idx}: cannot parse Dice_Roll={row['Dice_Roll']!r}"
#                 ) from e

#             try:
#                 yield pytest.param(
#                     roll,
#                     int(row["Score"]),
#                     int(row["Used_Dice"]),
#                     int(row["Reroll_Dice"]),
#                     int(row["Single_Fives"]),
#                     int(row["Single_Ones"]),
#                     id=f"row{idx}:{roll}",
#                 )
#             except ValueError as e:  # e.g. blank numeric field
#                 raise ValueError(f"Row {idx}: {e}") from e


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
    

# DISCARD_PATH = Path(__file__).resolve().parents[1] / "data" / "test_decide_smart_discards.csv"
    
# def _load_discard_cases(path_obj):
#     csv_path = path_obj
#     cases = []
#     ids = []                 # nice test-case names for pytest -vv
#     with csv_path.open(newline="") as f:
#         reader = csv.DictReader(f)
#         for i, row in enumerate(reader, 1):
#             cases.append((
#                 ast.literal_eval(row["counts"]),      # Counter({...})
#                 int(row["single_fives"]),
#                 int(row["single_ones"]),
#                 int(row["raw_score"]),
#                 int(row["raw_used"]),
#                 int(row["dice_len"]),
#                 int(row["turn_score_pre"]),
#                 int(row["score_threshold"]),
#                 row["smart_five"] == "True",
#                 row["smart_one"]  == "True",
#                 ast.literal_eval(row["expected"]),    # (discard_fives, discard_ones)
#             ))
#             ids.append(f"case{i:02d}")
#     return cases, ids


# CASES, CASE_IDS = _load_discard_cases(DISCARD_PATH)

CASES, CASE_IDS = load_discard_cases()

# ----------------------------------------------------------------------
# The parametrized test itself
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "counts,single_fives,single_ones,raw_score,raw_used,dice_len,"
    "turn_pre,threshold,smart_five,smart_one,expected",
    CASES,
    ids=CASE_IDS,
)
def test_decide_smart_discards(counts, single_fives, single_ones,
                               raw_score, raw_used, dice_len,
                               turn_pre, threshold, smart_five,
                               smart_one, expected):
    assert decide_smart_discards(
        counts,
        single_fives,
        single_ones,
        raw_score,
        raw_used,
        dice_len,
        turn_pre,
        threshold,
        smart_five,
        smart_one,
    ) == expected    
    

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
    

@pytest.mark.parametrize(
    "dice_roll,turn_pre,thr,smart5,smart1,expected",
    [
        # 1)  plain scoring, no smarts
        ([1, 2, 3, 4, 5, 6], 0, 300, False, False, (1500, 6, 0)),
        # 2)  smart-5 only, two lonely 5s + a 1   -> discard both 5s
        ([5, 5, 1],          0, 300, True,  False, (100, 1, 2)),
        # 3)  smart-5 + smart-1, 2 fives & 2 ones  -> discard 2×5, then 1×1
        ([5, 5, 1, 1],       0, 300, True,  True,  (100, 1, 3)),
        # 4)  rollback case: lone 5, but turn already ≥ thr so keep it
        ([5],               300, 300, True, True,  (50, 1, 0)),
    ],
)
def test_default_score(dice_roll, turn_pre, thr, smart5, smart1, expected):
    assert default_score(
        dice_roll       = dice_roll,
        turn_score_pre  = turn_pre,
        smart_five      = smart5,
        smart_one       = smart1,
        score_threshold = thr,
    ) == expected
    


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
