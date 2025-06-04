import numpy as np

from farkle.engine import FarkleGame, FarklePlayer
from farkle.strategies import ThresholdStrategy


def fixed_rng(seq):
    """Return a numpy Generator that cycles through *seq* forever."""
    arr = np.array(seq)
    class _G(np.random.Generator):
        def integers(self, low, high, size):  # noqa: ARG002
            idxs = np.arange(size) % len(arr)
            return arr[idxs]
    return _G(np.random.PCG64())


def test_take_turn_success(monkeypatch):
    seq = [6,6,6,2,3,4]  # 3×6 = 600 pts, uses 3 dice, reroll 3 → decide() false ⇒ stop 
    # must finish turn above 500 pts
    monkeypatch.setattr(np.random, "default_rng", lambda *_: fixed_rng(seq))
    strat = ThresholdStrategy(score_threshold=0, dice_threshold=6)  # stop after first roll
    p = FarklePlayer("P", strat, rng=np.random.default_rng())
    p.take_turn(target_score=10000)
    assert p.score == 600
    assert p.n_rolls == 1
    assert p.has_scored
    assert p.highest_turn == 600


def stop_after_one():
    return ThresholdStrategy(score_threshold=10000, dice_threshold=6)

def test_game_play_deterministic():
    # make every roll be six 1-s  ⇒  (3000, 6, 0)
    # Create one fixed_rng per player, so each sees “six 1’s” on their first roll:
    rng1 = fixed_rng([1,1,1,1,1,1])
    rng2 = fixed_rng([1,1,1,1,1,1])
    players = [
        FarklePlayer("P1", stop_after_one(), rng=rng1),
        FarklePlayer("P2", stop_after_one(), rng=rng2),
    ]
    gm = FarkleGame(players, target_score=1500).play()
    assert gm.winner in {"P1", "P2"}
    # Six 1's count as six-of-a-kind = 3000 points, so first turn already ≥1500
    assert gm.winning_score == 6000
    # both players still only rolled once
    total_rolls = []
    for stats in gm.per_player.values():
        total_rolls.append(stats["rolls"])
        
    assert total_rolls == [1, 2]
    assert gm.n_rounds == 1