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
    seq = [1,1,1,2,3,4]  # 3×1 = 300 pts, uses 3 dice, reroll 3 → decide() false ⇒ stop
    monkeypatch.setattr(np.random, "default_rng", lambda *_: fixed_rng(seq))
    strat = ThresholdStrategy(score_threshold=0, dice_threshold=6)  # stop after first roll
    p = FarklePlayer("P", strat, rng=np.random.default_rng())
    p.take_turn(target_score=10000)
    assert p.score == 300
    assert p.n_rolls == 1
    assert p.has_scored
    assert p.highest_turn == 300


def stop_after_one():
    return ThresholdStrategy(score_threshold=10000, dice_threshold=6)

def test_game_play_deterministic(monkeypatch):
    # make every roll be six 1-s  ⇒  (1500, 6, 0)
    monkeypatch.setattr(np.random, "default_rng",
                        lambda *_: fixed_rng([1,1,1,1,1,1]))
    players = [FarklePlayer(f"P{i+1}", stop_after_one()) for i in range(2)]
    gm = FarkleGame(players, target_score=1500).play()
    assert gm.winner in {"P1", "P2"}
    assert gm.winning_score == 1500
    # both players should have exactly one turn
    for stats in gm.per_player.values():
        assert stats["rolls"] == 1
