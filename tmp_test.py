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
sys.modules['tomllib'] = fake
sys.path.insert(0, str(Path('src').resolve()))

from farkle.game.scoring import default_score

turn_score = 220
roll = [1, 5]
res = default_score(
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
print(res)
