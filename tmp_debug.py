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

from farkle.game.scoring import (
    _decide_smart_discards_impl,
    generate_sequences,
    score_lister,
    score_roll_cached,
)
from farkle.simulation.strategies import FavorDiceOrScore

roll = [1, 5]
raw_score, raw_used, counts, sfives, sones = score_roll_cached(roll)
print("raw", raw_score, raw_used, counts, sfives, sones)
seqs = generate_sequences(counts, smart_one=True)
print("seqs", seqs)
cands = score_lister(tuple(seqs))
print("cand count", len(cands))
for cand in cands:
    print("cand", cand)

turn_score_pre = 220
n = _decide_smart_discards_impl(
    counts=counts,
    single_fives=sfives,
    single_ones=sones,
    raw_score=raw_score,
    raw_used=raw_used,
    dice_roll_len=len(roll),
    turn_score_pre=turn_score_pre,
    score_threshold=300,
    dice_threshold=0,
    smart_five=True,
    smart_one=True,
    consider_score=True,
    consider_dice=False,
    require_both=False,
    favor_dice_or_score=FavorDiceOrScore.SCORE,
)
print("result", n)
