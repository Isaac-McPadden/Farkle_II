from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from farkle.scoring import DiceRoll, default_score
from farkle.strategies import ThresholdStrategy

"""engine.py
============
Player and single-game engine for Farkle simulations.

High-level flow
---------------
* ``FarkleGame.play`` drives a game loop until someone reaches
  ``target_score``.  When that happens the *final round* rule is applied:
  every other player gets one last turn.
* ``FarklePlayer.take_turn`` handles the intra-turn loop of rolling,
  scoring, and consulting its strategy.

The module keeps no global state; randomness lives inside each
``FarklePlayer`` via its dedicated ``random.Random`` instance (passed in
from the outer simulation layer).
"""



__all__: list[str] = [
    "FarklePlayer",
    "GameMetrics",
    "FarkleGame",
]



# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

@dataclass
class FarklePlayer:
    """A single player in a Farkle game."""

    name: str
    strategy: ThresholdStrategy
    score: int = 0
    has_scored: bool = False  # entered game (≥500) flag
    rng:  np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    # stats
    n_farkles: int = 0
    n_rolls: int = 0
    highest_turn: int = 0

    # ----------------------------- helpers -----------------------------
    def _roll(self, n: int) -> DiceRoll:
        """Return *n* pseudo-random dice using the player's RNG."""
        self.n_rolls += 1
        return list(self.rng.integers(1, 7, size=n))

    # ------------------------------ turn --------------------------------
    def take_turn(self, target_score: int) -> None:
        """Simulate one complete turn for the player."""
        dice = 6
        turn_score = 0

        while dice > 0:
            # 1) Roll `dice` number of dice
            roll = self._roll(dice)

            # 2) Compute points from this roll, after applying Smart‐5/Smart‐1 discards
            pts, used, reroll = default_score(
                dice_roll        = roll,
                turn_score_pre   = turn_score,
                smart_five       = self.strategy.smart_five,
                smart_one        = self.strategy.smart_one,
                score_threshold  = self.strategy.score_threshold,
            )

            # 3) If pts == 0, that's a Farkle: bust, lose all points this turn, and end turn
            if pts == 0:  # farkle bust
                self.n_farkles += 1
                turn_score = 0
                break

            # 4) Otherwise, accumulate the points from this roll
            turn_score += pts

            # 5) “Hot dice” logic: if all dice in `roll` scored (used == len(roll)),
            #    and `reroll == 0`, that means you get to roll all 6 dice again.
            #    Otherwise, you roll `reroll` dice next.
            dice = 6 if (used == len(roll) and reroll == 0) else reroll

            # 6) Check the strategy’s decide() method to see if we should keep rolling.
            #    Pass in:
            #      - current turn_score
            #      - dice_left = dice (from step 5)
            #      - has_scored = whether we’ve ever scored ≥500 on a previous turn
            #      - score_needed = how many more points (from banked + turn_score) we need to reach target_score
            if not self.strategy.decide(
                turn_score   = turn_score,
                dice_left    = dice,
                has_scored   = self.has_scored,
                score_needed = max(0, target_score - (self.score + turn_score)),
            ):
                break

        # 7) After leaving the loop, check if this is our “entry turn” (we need ≥500 to get on the board)
        if not self.has_scored and turn_score >= 500:
            self.has_scored = True

        # 8) If we already had a “first‐500” in a previous turn, or just got it, bank the points
        if self.has_scored:
            self.score        += turn_score
            self.highest_turn = max(self.highest_turn, turn_score)


# ---------------------------------------------------------------------------
# Game-level structures
# ---------------------------------------------------------------------------

@dataclass
class GameMetrics:
    """Aggregate stats returned after a single game."""

    winner: str
    winning_score: int
    n_rounds: int
    per_player: Dict[str, Dict[str, Any]]


class FarkleGame:
    """Driver for a *single* Farkle game."""

    def __init__(self, players: Sequence[FarklePlayer], *, target_score: int = 10_000) -> None:
        self.players: List[FarklePlayer] = list(players)
        self.target_score: int = target_score

    # ---------------------------- gameplay -----------------------------
    def play(self, max_rounds: int = 100) -> GameMetrics:
        """Run the game and return a *GameMetrics* summary."""
        final_round = False
        trigger: Optional[FarklePlayer] = None
        rounds = 0
        while rounds < max_rounds:
            for p in self.players:
                p.take_turn(self.target_score)
                if p.score >= self.target_score and not final_round:
                    final_round = True
                    trigger = p
            rounds += 1
            if final_round:
                for p in self.players:
                    if p is not trigger:
                        p.take_turn(self.target_score)
                break
        winner = max(self.players, key=lambda pl: pl.score)
        per_player: Dict[str, Dict[str, Any]] = {
            p.name: {
                "score": p.score,
                "farkles": p.n_farkles,
                "rolls": p.n_rolls,
                "highest_turn": p.highest_turn,
                "strategy": str(p.strategy),
            }
            for p in self.players
        }
        return GameMetrics(winner.name, winner.score, rounds, per_player)
