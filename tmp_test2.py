import sys
import types
from pathlib import Path

fake = types.ModuleType("tomllib")


def load(fh):
    data = fh.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return {"project": {"version": "0.0.0"}}


fake.load = load
sys.modules["tomllib"] = fake
sys.path.insert(0, str(Path("src").resolve()))

from farkle.game.scoring import decide_smart_discards, default_score

turn_score = 220
roll = [1, 5]

result = default_score(
    roll,
    turn_score_pre=turn_score,
    smart_five=True,
    smart_one=True,
    consider_score=True,
    consider_dice=False,
    require_both=False,
    score_threshold=300,
    dice_threshold=0,
    favor_dice_or_score=True,
    return_discards=True,
)
print("default_score", result)

# direct call (clearing cache first)
decide_smart_discards.cache_clear()
res = decide_smart_discards(
    counts=(1, 0, 0, 0, 1, 0),
    single_fives=1,
    single_ones=1,
    raw_score=150,
    raw_used=2,
    dice_roll_len=2,
    turn_score_pre=turn_score,
    score_threshold=300,
    dice_threshold=0,
    smart_five=True,
    smart_one=True,
    consider_score=True,
    consider_dice=False,
    require_both=False,
    favor_dice_or_score=True,
)
print("decide", res)
