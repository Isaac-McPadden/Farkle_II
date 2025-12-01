import pickle
import random
from collections import Counter

import pandas as pd
import pytest

from farkle.simulation.strategies import (
    FavorDiceOrScore,
    STOP_AT_THRESHOLDS,
    StopAtStrategy,
    ThresholdStrategy,
    _sample_favor_score,
    build_stop_at_strategy,
    load_farkle_results,
    parse_strategy,
    parse_strategy_for_df,
    random_threshold_strategy,
)


@pytest.mark.parametrize(
    "tscore,dleft,has500,cs,cd,rb,keep_rolling",
    [
        # opening-roll shortcut
        (200, 6, False, True, True, True, True),
        # score-only
        (250, 3, True, True, False, False, True),
        (350, 3, True, True, False, False, False),
        (250, 4, True, True, False, False, True),
        (350, 4, True, True, False, False, False),
        # dice-only
        (0, 4, True, False, True, False, True),
        (0, 3, True, False, True, False, False),
        (500, 4, True, False, True, False, True),
        (500, 3, True, False, True, False, False),
        # both + require_both thresholds met
        (400, 4, True, True, True, True, True),  # Enough dice but too many points
        (200, 2, True, True, True, True, True),  # Low enough points but not enough dice
        (200, 4, True, True, True, True, True),  # Low enough points and enough dice available
        (400, 2, True, True, True, True, False),  # # Too many points and not enough dice available
        # both + OR logic
        (400, 4, True, True, True, False, False),  # Enough dice but too many points
        (200, 2, True, True, True, False, False),  # Low enough points but not enough dice
        (200, 4, True, True, True, False, True),  # Low enough points and enough dice available
        (400, 2, True, True, True, False, False),  # # Too many points and not enough dice available
    ],
)
def test_decide(tscore, dleft, has500, cs, cd, rb, keep_rolling):
    strat = ThresholdStrategy(
        score_threshold=300, dice_threshold=3, consider_score=cs, consider_dice=cd, require_both=rb
    )
    assert (
        strat.decide(turn_score=tscore, dice_left=dleft, has_scored=has500, score_needed=1_000)
        is keep_rolling
    )


def test_smart1_requires_smart5():
    with pytest.raises(ValueError):
        ThresholdStrategy(smart_five=False, smart_one=True)


def test_require_both_guard():
    with pytest.raises(ValueError):
        ThresholdStrategy(consider_score=True, consider_dice=False, require_both=True)


def test_random_strategy_factory():
    rng = random.Random(123)
    for _ in range(200):
        ts = random_threshold_strategy(rng)
        # smart_one requires smart_five
        assert not ts.smart_one or ts.smart_five
        # require_both only legal when both flags set
        assert not ts.require_both or (ts.consider_score and ts.consider_dice)
        # NEW: favor_dice_or_score must follow the truth-table above
        if ts.consider_score and not ts.consider_dice:
            assert ts.favor_dice_or_score is FavorDiceOrScore.SCORE
        elif ts.consider_dice and not ts.consider_score:
            assert ts.favor_dice_or_score is FavorDiceOrScore.DICE


def test_build_stop_at_strategy_and_naming():
    strat = build_stop_at_strategy(350)
    assert isinstance(strat, StopAtStrategy)
    assert str(strat) == "stop_at_350"
    assert strat.consider_score and not strat.consider_dice
    assert not strat.auto_hot_dice

    heuristic = build_stop_at_strategy(500, heuristic=True, inactive_dice_threshold=-2)
    assert isinstance(heuristic, StopAtStrategy)
    assert heuristic.smart_five and heuristic.smart_one
    assert heuristic.auto_hot_dice
    assert heuristic.dice_threshold == -2
    assert str(heuristic) == "stop_at_500_heuristic"

    with pytest.raises(ValueError):
        build_stop_at_strategy(999)


# ────────────────────────────────────────────────────────────────────────────
# 1) A handful of “known-good” strings, together with the expected fields.
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "input_str, expected",
    [
        # ────────────────────────────────────────────────────────────────────
        # Basic case: all flags True, favor_dice_or_score=True, auto_hot=True, run_up=True
        #   • score_threshold = 300
        #   • dice_threshold  = 2
        #   • consider_score  = True  (“S”)
        #   • consider_dice   = True  (“D”)
        #   • smart_five      = True  (“*F”)
        #   • smart_one       = True  (“*O”)
        #   • favor_dice_or_score    = True  (“P”)
        #   • require_both    = True  (“AND”)
        #   • auto_hot_dice   = True  (“H”)
        #   • run_up_score    = True  (“R”)
        (
            "Strat(300,2)[SD][FOFS][AND][HR]",
            {
                "score_threshold": 300,
                "dice_threshold": 2,
                "consider_score": True,
                "consider_dice": True,
                "smart_five": True,
                "smart_one": True,
                "favor_dice_or_score": FavorDiceOrScore.SCORE,
                "require_both": True,
                "auto_hot_dice": True,
                "run_up_score": True,
            },
        ),
        # ────────────────────────────────────────────────────────────────────
        # 2) Variation: smart_one=False, favor_dice_or_score=False, auto_hot=False, run_up=False
        #   • score_threshold = 500
        #   • dice_threshold  = 1
        #   • consider_score  = True  (“S”)
        #   • consider_dice   = False (“-”)
        #   • smart_five      = True  (“*F”)
        #   • smart_one       = False (“-”)
        #   • favor_dice_or_score    = False (“-”)
        #   • require_both    = False (“OR”)
        #   • auto_hot_dice   = False (“-”)
        #   • run_up_score    = False (“-”)
        (
            "Strat(500,1)[S-][F-FD][OR][--]",
            {
                "score_threshold": 500,
                "dice_threshold": 1,
                "consider_score": True,
                "consider_dice": False,
                "smart_five": True,
                "smart_one": False,
                "favor_dice_or_score": FavorDiceOrScore.DICE,
                "require_both": False,
                "auto_hot_dice": False,
                "run_up_score": False,
            },
        ),
        # ────────────────────────────────────────────────────────────────────
        # 3) Neither consider_score nor consider_dice (cs=False, cd=False), favor_dice_or_score=True
        #   • score_threshold = 250
        #   • dice_threshold  = 3
        #   • consider_score  = False (“-”)
        #   • consider_dice   = False (“-”)
        #   • smart_five      = False (“-F”)
        #   • smart_one       = False (“--”)
        #   • favor_dice_or_score    = True  (“P”)
        #   • require_both    = False (“OR”)
        #   • auto_hot_dice   = True  (“H”)
        #   • run_up_score    = False (“-”)
        (
            "Strat(250,3)[--][--FS][OR][H-]",
            {
                "score_threshold": 250,
                "dice_threshold": 3,
                "consider_score": False,
                "consider_dice": False,
                "smart_five": False,
                "smart_one": False,
                "favor_dice_or_score": FavorDiceOrScore.SCORE,
                "require_both": False,
                "auto_hot_dice": True,
                "run_up_score": False,
            },
        ),
        # ────────────────────────────────────────────────────────────────────
        # 4) cs=False, cd=True => favor_dice_or_score must be False
        #   • score_threshold = 700
        #   • dice_threshold  = 4
        #   • consider_score  = False (“-”)
        #   • consider_dice   = True  (“D”)
        #   • smart_five      = False (“-F”)
        #   • smart_one       = False (“--”)
        #   • favor_dice_or_score    = False (“-”)
        #   • require_both    = False (“OR”)
        #   • auto_hot_dice   = False (“-”)
        #   • run_up_score    = True  (“R”)
        (
            "Strat(700,4)[-D][--FD][OR][-R]",
            {
                "score_threshold": 700,
                "dice_threshold": 4,
                "consider_score": False,
                "consider_dice": True,
                "smart_five": False,
                "smart_one": False,
                "favor_dice_or_score": FavorDiceOrScore.DICE,
                "require_both": False,
                "auto_hot_dice": False,
                "run_up_score": True,
            },
        ),
    ],
)
def test_parse_strategy_valid(input_str, expected):
    strat = parse_strategy(input_str)

    # Verify numeric thresholds
    assert strat.score_threshold == expected["score_threshold"]
    assert strat.dice_threshold == expected["dice_threshold"]

    # Verify all boolean flags
    assert strat.consider_score == expected["consider_score"]
    assert strat.consider_dice == expected["consider_dice"]
    assert strat.smart_five == expected["smart_five"]
    assert strat.smart_one == expected["smart_one"]
    assert strat.favor_dice_or_score == expected["favor_dice_or_score"]
    assert strat.require_both == expected["require_both"]
    assert strat.auto_hot_dice == expected["auto_hot_dice"]
    assert strat.run_up_score == expected["run_up_score"]

    # Finally, round-trip: str(strat) must match the same “fields” (though formatting may differ)
    # We only check that parse_strategy(__str__()) does not error and produces consistent fields.
    reparsed = parse_strategy(str(strat))
    assert reparsed.score_threshold == strat.score_threshold
    assert reparsed.dice_threshold == strat.dice_threshold
    assert reparsed.consider_score == strat.consider_score
    assert reparsed.consider_dice == strat.consider_dice
    assert reparsed.smart_five == strat.smart_five
    assert reparsed.smart_one == strat.smart_one
    assert reparsed.favor_dice_or_score == strat.favor_dice_or_score
    assert reparsed.require_both == strat.require_both
    assert reparsed.auto_hot_dice == strat.auto_hot_dice
    assert reparsed.run_up_score == strat.run_up_score


def test_parse_strategy_round_trip_str():
    s = "Strat(500,1)[S-][--FD][OR][-R]"
    assert str(parse_strategy(s)) == s


def test_entry_gate_requires_rolling():
    strat = ThresholdStrategy(score_threshold=300, dice_threshold=2)
    assert strat.decide(
        turn_score=400,
        dice_left=3,
        has_scored=False,
        score_needed=1000,
    )


# ────────────────────────────────────────────────────────────────────────────
# 2) A collection of malformed strings that should all raise ValueError
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_str",
    [
        "",  # completely empty
        "Strat()",  # missing thresholds
        "Strat(300)[SD][*F*OP][AND][HR]",  # missing comma
        "Strat(300,2)[XY][*F*OP][AND][HR]",  # invalid consider_score/consider_dice tokens
        "Strat(300,2)[SD][ZZ][AND][HR]",  # invalid smart_five/smart_one block
        "Strat(300,2)[SD][*F*OP][MAYBE][HR]",  # invalid require_both token
        "Strat(300,2)[SD][*F*OP][AND][XZ]",  # invalid auto_hot/run_up tokens
        # Missing the “ps” slot in the second bracket:
        "Strat(300,2)[SD][*F*O ][AND][HR]",
        # Too short or too long inside second bracket:
        "Strat(300,2)[SD][*FO P][AND][HR]",
        "Strat(300,2)[SD][*F*OOP][AND][HR]",
        # Wrong numeric types:
        "Strat(abc,2)[SD][*F*OP][AND][HR]",
        "Strat(300,xyz)[SD][*F*OP][AND][HR]",
        # Non‐matching entire pattern (a stray character at end):
        "Strat(300,2)[SD][*F*OP][AND][HR]!",
    ],
)
def test_parse_strategy_invalid(bad_str):
    with pytest.raises(ValueError):
        parse_strategy(bad_str)


def test_parse_strategy_for_df():  # noqa: ARG001
    strat_case_1 = "Strat(300,2)[SD][FOFS][AND][HR]"
    strat_case_2 = "Strat(300,2)[--][--FD][OR][--]"
    strat_dict_1 = parse_strategy_for_df(strat_case_1)
    strat_dict_2 = parse_strategy_for_df(strat_case_2)
    assert strat_dict_1 == {
        "score_threshold": 300,
        "dice_threshold": 2,
        "smart_five": True,
        "smart_one": True,
        "consider_score": True,
        "consider_dice": True,
        "require_both": True,
        "auto_hot_dice": True,
        "run_up_score": True,
        "favor_dice_or_score": FavorDiceOrScore.SCORE,
    }
    assert strat_dict_2 == {
        "score_threshold": 300,
        "dice_threshold": 2,
        "smart_five": False,
        "smart_one": False,
        "consider_score": False,
        "consider_dice": False,
        "require_both": False,
        "auto_hot_dice": False,
        "run_up_score": False,
        "favor_dice_or_score": FavorDiceOrScore.DICE,
    }


def test_sample_favor_score_deterministic():
    rng = random.Random(0)
    assert _sample_favor_score(True, False, rng) is FavorDiceOrScore.SCORE
    rng = random.Random(0)
    assert _sample_favor_score(False, True, rng) is FavorDiceOrScore.DICE
    rng = random.Random(0)
    expected_tt = rng.choice([FavorDiceOrScore.SCORE, FavorDiceOrScore.DICE])
    rng = random.Random(0)
    assert _sample_favor_score(True, True, rng) is expected_tt
    rng = random.Random(0)
    expected_ff = rng.choice([FavorDiceOrScore.SCORE, FavorDiceOrScore.DICE])
    rng = random.Random(0)
    assert _sample_favor_score(False, False, rng) is expected_ff


def test_load_farkle_results(tmp_path):
    counter = Counter(
        {
            "Strat(300,2)[SD][FOFS][AND][HR]": 5,
            "Strat(250,3)[--][--FD][OR][--]": 3,
        }
    )
    pkl = tmp_path / "results.pkl"
    pkl.write_bytes(pickle.dumps(counter))

    df = load_farkle_results(pkl)

    row1 = {"strategy": "Strat(300,2)[SD][FOFS][AND][HR]", "wins": 5}
    row1.update(parse_strategy_for_df(row1["strategy"]))
    row2 = {"strategy": "Strat(250,3)[--][--FD][OR][--]", "wins": 3}
    row2.update(parse_strategy_for_df(row2["strategy"]))
    expected = pd.DataFrame([row1, row2])
    expected = expected[df.columns]
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)


def test_load_farkle_results_unordered(tmp_path):
    counter = Counter(
        {
            "Strat(300,2)[SD][FOFS][AND][HR]": 5,
        }
    )
    pkl = tmp_path / "results.pkl"
    pkl.write_bytes(pickle.dumps(counter))

    df = load_farkle_results(pkl, ordered=False)
    assert df.columns.tolist() == [
        "strategy",
        "wins",
        "score_threshold",
        "dice_threshold",
        "smart_five",
        "smart_one",
        "consider_score",
        "consider_dice",
        "require_both",
        "auto_hot_dice",
        "run_up_score",
        "favor_dice_or_score",
    ]
