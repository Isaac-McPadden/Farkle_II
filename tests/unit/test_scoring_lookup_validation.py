import pytest

from farkle.scoring_lookup import score_roll, evaluate


def test_score_roll_rejects_invalid_faces():
    with pytest.raises(ValueError):
        score_roll([1, 2, 7])


def test_score_roll_rejects_too_many_dice():
    with pytest.raises(ValueError):
        score_roll([1, 2, 3, 4, 5, 6, 1])


def test_evaluate_rejects_sum_over_six():
    with pytest.raises(ValueError):
        evaluate((2, 2, 2, 1, 0, 0))


def test_evaluate_rejects_negative_counts():
    with pytest.raises(ValueError):
        evaluate((1, -1, 0, 0, 0, 0))
