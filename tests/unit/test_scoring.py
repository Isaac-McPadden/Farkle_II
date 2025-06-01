# tests/unit/test_scoring.py
from __future__ import annotations

import ast
import csv
from pathlib import Path

import pytest

from farkle.scoring import compute_raw_score

# ---------------------------------------------------------------------------
# Location of the test data CSV
# tests/unit/  ->  tests/data/
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "test_farkle_scores_data.csv"
if not DATA_PATH.exists():  # fail fast with a helpful hint
    raise FileNotFoundError(
        f"Could not find {DATA_PATH} – is the CSV in tests/data/?"
    )

# ---------------------------------------------------------------------------
# Turn each CSV row into a pytest.param
# ---------------------------------------------------------------------------
def _load_cases():
    with DATA_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, start=1):
            try:
                roll = ast.literal_eval(row["Dice_Roll"])
            except Exception as e:  # bad “[1, 5]” string?
                raise ValueError(
                    f"Row {idx}: cannot parse Dice_Roll={row['Dice_Roll']!r}"
                ) from e

            try:
                yield pytest.param(
                    roll,
                    int(row["Score"]),
                    int(row["Used_Dice"]),
                    int(row["Reroll_Dice"]),
                    int(row["Single_Fives"]),
                    int(row["Single_Ones"]),
                    id=f"row{idx}:{roll}",
                )
            except ValueError as e:  # e.g. blank numeric field
                raise ValueError(f"Row {idx}: {e}") from e


# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "roll, exp_score, exp_used, exp_reroll, exp_sfives, exp_sones",
    list(_load_cases()),
)
def test_compute_raw_score(
    roll, exp_score, exp_used, exp_reroll, exp_sfives, exp_sones
):
    score, used, _counts, single_fives, single_ones = compute_raw_score(roll)

    assert score == exp_score
    assert used == exp_used
    assert single_fives == exp_sfives
    assert single_ones == exp_sones
    assert len(roll) - used == exp_reroll
