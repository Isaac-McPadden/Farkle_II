import itertools

import numpy as np
import pytest

from farkle.engine import ROLL_LIMIT, FarkleGame, FarklePlayer
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
    seq = [6, 6, 6, 2, 3, 4]  # 3×6 = 600 pts, uses 3 dice, reroll 3 → decide() false ⇒ stop 
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
    return ThresholdStrategy(score_threshold=100, dice_threshold=6, run_up_score=False)

def test_game_play_deterministic():
    # make every roll be six 1-s  ⇒  (3000, 6, 0)
    # Create one fixed_rng per player, so each sees “six 1’s” on their first roll:
    rng1 = fixed_rng([1, 1, 1, 1, 1, 1])
    rng2 = fixed_rng([1, 1, 1, 1, 1, 1])
    players = [
        FarklePlayer("P1", stop_after_one(), rng=rng1),
        FarklePlayer("P2", stop_after_one(), rng=rng2),
    ]
    gm = FarkleGame(players, target_score=1500).play()
    winner = max(gm.players, key=lambda n: gm.players[n].score)
    assert winner in {"P1", "P2"}
    # Six 1's count as six-of-a-kind = 3000 points, so first turn already ≥1500
    assert gm.players[winner].score == 6000
    # both players still only had one round
    total_rolls = [gm.players[name].rolls for name in sorted(gm.players)]
        
    assert total_rolls == [1, 2]
    assert gm.game.n_rounds == 1


class _SeqGen(np.random.Generator):
    """
    A deterministic RNG that cycles through *seq* forever.

    It satisfies the numpy.random.Generator interface, so
    static type-checkers are happy and the production code
    needs no changes.
    """
    
    def __init__(self, seq):
        # Initialise the parent class with *any* BitGenerator
        super().__init__(np.random.PCG64())
        self._cycle = itertools.cycle(seq)

    # Override only the method the engine actually calls.
    def integers(
        self,
        low: int,  # noqa: ARG002
        high: int | None = None,  # noqa: ARG002
        size: int | tuple[int, ...] | None = None,
        dtype=int,
        endpoint: bool = False,  # noqa: ARG002
    ):
        # The engine always passes (1, 7, size=6) style args,
        # so we ignore *low*, *high*, *endpoint*.
        if size is None:
            size = 1
        n = int(np.prod(size))
        arr = np.fromiter((next(self._cycle) for _ in range(n)), dtype=dtype, count=n)
        return arr.reshape(size)


def fixed_rng_2(seq):
    """Return a numpy.random.Generator that cycles through *seq*."""
    return _SeqGen(seq)
    
    
def test_auto_hot_dice_forces_roll():
    seq = [1, 1, 1, 5, 5, 5, 2, 2, 3, 3, 4, 6]  # 6 scoring + 6 busting dice
    rng = fixed_rng_2(seq)

    strat_hot = ThresholdStrategy(score_threshold=0, dice_threshold=6, auto_hot_dice=True)
    strat_cold = ThresholdStrategy(score_threshold=0, dice_threshold=6, auto_hot_dice=False)

    p_hot = FarklePlayer("H", strat_hot, rng=rng)
    p_cold = FarklePlayer("C", strat_cold, rng=rng)

    # target_score argument comes from take_turn signature
    p_hot.take_turn(target_score=10_000)
    p_cold.take_turn(target_score=10_000)

    assert p_hot.score == 0  # rolled twice, busts on 2,2,3,3,4,6
    assert p_cold.score == 2500  # banked after first roll
    assert p_hot.n_hot_dice == 1  # records auto_hot_dice option taken
    assert p_hot.n_rolls == 2  # records correct roll count: 2 rolls, busts on second roll
    assert p_cold.n_rolls == 1  # records correct roll count: declines auto_hot_dice, rolls once and banks


class SeqGen2(np.random.Generator):
    """Deterministic RNG that cycles through *seq* forever."""
    
    def __init__(self, seq):
        super().__init__(np.random.PCG64())
        self._seq = list(seq)
        self._i = 0

    def integers(self, low, high=None, size=None, **kwargs):  # noqa: ARG002
        if size is None:
            size = 1
        n = int(np.prod(size))
        out = [self._seq[(self._i + k) % len(self._seq)] for k in range(n)]
        self._i = (self._i + n) % len(self._seq)
        return np.array(out).reshape(size)


def test_final_round_override():
    # ---------- Player 1 triggers the final round ----------
    p1_rng = SeqGen2([1, 1, 1, 2, 3, 4])  # 300 pts
    strat1 = ThresholdStrategy(score_threshold=0, dice_threshold=6)
    p1 = FarklePlayer("P1", strat1, rng=p1_rng)
    p1.score = 9_900  # so first turn pushes ≥10 000
    p1.has_scored = True

    # ---------- Player 2 would normally STOP after 1 roll ----------
    p2_rng = SeqGen2([1, 1, 2, 3, 4, 6, 2, 2, 3, 4])  # rolls: 200 pts (would bank) then 0 pts → bust
    strat2 = ThresholdStrategy(
        score_threshold=150, dice_threshold=6, consider_score=True, consider_dice=False
    )
    p2 = FarklePlayer("P2", strat2, rng=p2_rng)
    p2.score = 8_500
    p2.has_scored = True

    # ---------- Run one game ----------
    g = FarkleGame([p1, p2])
    g.play()

    # P2 should have rolled *twice*: override forced a 2nd roll
    assert p2.n_rolls == 2
    # …and still finished behind the trigger score because of the bust
    assert p2.score == 8_500


def test_turn_roll_fuse():
    # RNG that always returns hot dice: 1,1,1,2,2,2 …
    class HotGen(np.random.Generator):
        def __init__(self): 
            super().__init__(np.random.PCG64())
            
        def integers(self, *a, size=None, **k):  # noqa: ARG002, D401
            return np.array([1, 1, 1, 2, 2, 2][:size or 1])
        
    p = FarklePlayer("X", ThresholdStrategy(auto_hot_dice=True),
                     rng=HotGen())
    with pytest.raises(RuntimeError):
        p.take_turn(target_score=10_000)


class AlwaysRoll(ThresholdStrategy):
    """Strategy that always opts to roll again."""

    def decide(self, **kwargs):  # noqa: D401, ARG002
        return True


def test_final_round_stop_when_ahead_run_up_false():
    seq = [1, 1, 1, 1, 1, 1, 2, 3, 4, 6, 2, 3]
    rng = SeqGen2(seq)

    strat = ThresholdStrategy(
        score_threshold=0,
        dice_threshold=6,
        consider_score=False,
        consider_dice=False,
    )
    p = FarklePlayer("P", strat, rng=rng)
    p.score = 9_000
    p.has_scored = True

    p.take_turn(target_score=10_000, final_round=True, score_to_beat=9_500)

    assert p.n_rolls == 1  # stopped automatically when ahead
    assert p.score == 12_000


def test_final_round_continue_when_ahead_run_up_true():
    seq = [1, 1, 1, 1, 1, 1, 2, 3, 4, 6, 2, 3]
    rng = SeqGen2(seq)

    strat = AlwaysRoll(
        score_threshold=0,
        dice_threshold=6,
        consider_score=False,
        consider_dice=False,
        run_up_score=True,
    )
    p = FarklePlayer("P", strat, rng=rng)
    p.score = 9_000
    p.has_scored = True

    p.take_turn(target_score=10_000, final_round=True, score_to_beat=9_500)

    assert p.n_rolls == 2  # second roll allowed despite being ahead
    assert p.score == 9_000  # bust on the second roll wipes the turn


def test_take_turn_roll_limit(monkeypatch):
    """Player exceeds ``ROLL_LIMIT`` when every roll scores."""

    def scoring_roll(self, n):
        self.n_rolls += 1
        return [1] * n

    monkeypatch.setattr(FarklePlayer, "_roll", scoring_roll)

    p = FarklePlayer("P", AlwaysRoll())

    with pytest.raises(RuntimeError):
        p.take_turn(target_score=10_000)

    assert p.n_rolls == ROLL_LIMIT + 1


def test_game_stops_at_default_max_rounds(monkeypatch):
    """Ensure the game loop honours the default 200-round cap."""

    def bust_roll(self, n):
        self.n_rolls += 1
        return [2, 3, 4, 6, 2, 3][:n]

    monkeypatch.setattr(FarklePlayer, "_roll", bust_roll)

    players = [FarklePlayer("A", AlwaysRoll()), FarklePlayer("B", AlwaysRoll())]
    gm = FarkleGame(players, target_score=10_000).play()

    assert gm.game.n_rounds == 200

