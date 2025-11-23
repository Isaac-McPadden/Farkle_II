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
    _must_bank,
    _select_candidate,
    generate_sequences,
    score_lister,
    score_roll_cached,
)
from farkle.simulation.strategies import FavorDiceOrScore

roll = [1, 5]
turn_score_pre = 220
score_threshold = 300
single_fives, single_ones = None, None
raw_score, raw_used, counts, sfives, sones = score_roll_cached(roll)
print("raw", raw_score, raw_used, counts, sfives, sones)

seqs = generate_sequences(counts, smart_one=True)
print("seqs", seqs)

cands = score_lister(tuple(seqs))
print("cand count", len(cands))
for cand in cands:
    print("cand", cand)


def must_bank(score_after, dice_left_after):
    return _must_bank(
        score_after,
        dice_left_after,
        score_threshold=score_threshold,
        dice_threshold=0,
        consider_score=True,
        consider_dice=False,
        require_both=False,
    )


best = _select_candidate(
    cands,
    turn_score_pre=turn_score_pre,
    dice_roll_len=len(roll),
    counts=counts,
    single_fives=sfives,
    single_ones=sones,
    favor_dice_or_score=FavorDiceOrScore.SCORE,
    must_bank=must_bank,
)
print("best", best)
if best:
    best_sf, best_so = best
    print("diff", sfives - best_sf, sones - best_so)
