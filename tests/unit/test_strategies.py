import pytest

from farkle.strategies import ThresholdStrategy


@pytest.mark.parametrize(
    "tscore,dleft,has500,cs,cd,rb,keep_rolling",
    [
        # opening-roll shortcut
        (200, 6, False, True, True, True,  True),
        # score-only
        (250, 3, True,  True, False, False,  True),
        (350, 3, True,  True, False, False,  False),
        (250, 4, True,  True, False, False,  True),
        (350, 4, True,  True, False, False,  False),        
        # dice-only
        (0,   4, True,  False, True, False,  True),
        (0,   3, True,  False, True, False,  False),
        (500, 4, True,  False, True, False,  True),
        (500, 3, True,  False, True, False,  False),
        # both + require_both thresholds met
        (400, 4, True,  True, True,  True,  True),  # Enough dice but too many points
        (200, 2, True,  True, True,  True,  True),  # Low enough points but not enough dice
        (200, 4, True,  True, True,  True,  True),   # Low enough points and enough dice available
        (400, 2, True,  True, True,  True,  False),  # # Too many points and not enough dice available
        # both + OR logic
        (400, 4, True,  True, True,  False,  False),  # Enough dice but too many points
        (200, 2, True,  True, True,  False,  False),  # Low enough points but not enough dice
        (200, 4, True,  True, True,  False,  True),   # Low enough points and enough dice available
        (400, 2, True,  True, True,  False,  False),  # # Too many points and not enough dice available
    ],
)
def test_decide(tscore,dleft,has500,cs,cd,rb,keep_rolling):
    strat = ThresholdStrategy(
        score_threshold=300, dice_threshold=3,
        consider_score=cs, consider_dice=cd, require_both=rb
    )
    assert strat.decide(
        turn_score=tscore, dice_left=dleft,
        has_scored=has500, score_needed=1_000
    ) is keep_rolling
