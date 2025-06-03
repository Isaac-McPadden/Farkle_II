import pytest

from farkle.strategies import ThresholdStrategy


@pytest.mark.parametrize(
    "tscore,dleft,has500,cs,cd,rb,exp",
    [
        # opening-roll shortcut
        (200, 6, False, True, True, True,  True),
        # score-only
        (250, 3, True,  True, False, True,  True),
        (350, 3, True,  True, False, True,  False),
        # dice-only
        (0,   4, True,  False, True, True,  True),
        (0,   2, True,  False, True, True,  False),
        # both + require_both
        (200, 4, True,  True, True,  True,  True),
        (200, 2, True,  True, True,  True,  False),
        # both + OR logic
        (200, 2, True,  True, True,  False, True),
        (350, 2, True,  True, True,  False, False),
    ],
)
def test_decide(tscore,dleft,has500,cs,cd,rb,exp):
    strat = ThresholdStrategy(
        score_threshold=300, dice_threshold=3,
        consider_score=cs, consider_dice=cd, require_both=rb
    )
    assert strat.decide(
        turn_score=tscore, dice_left=dleft,
        has_scored=has500, score_needed=1_000
    ) is exp
