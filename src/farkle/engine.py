from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

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
    def take_turn(
        self,
        target_score: int,
        *,
        final_round: bool = False,
        score_to_beat: int = 0,
    ) -> None:
        """Simulate one complete turn for the player."""
        dice = 6
        turn_score = 0
        rolls_this_turn = 0

        while dice > 0:
            if rolls_this_turn > 1000:
                raise RuntimeError("Turn exceeded 1000 rolls - aborting.")
            # 1) Roll `dice` number of dice
            roll = self._roll(dice)
            rolls_this_turn += 1

            # 2) Compute points from this roll, after applying Smart‐5/Smart‐1 discards
            pts, used, reroll = default_score(
                dice_roll        = roll,
                turn_score_pre   = turn_score,
                smart_five       = self.strategy.smart_five,
                smart_one        = self.strategy.smart_one,
                score_threshold  = self.strategy.score_threshold,
                dice_threshold   = self.strategy.dice_threshold,
                consider_score   = self.strategy.consider_score,
                consider_dice    = self.strategy.consider_dice,
                require_both     = self.strategy.require_both,
                prefer_score     = self.strategy.prefer_score
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
            
            # 5.5) Hot dice short circuit
            if self.strategy.auto_hot_dice and dice == 6:
                continue                       # force another throw
            
            # 6-A) If we’re in the final round and have already won, stop.
            running_total = self.score + turn_score
            if final_round and running_total > score_to_beat:
                if self.strategy.run_up_score:         # NEW opt-in flag
                    pass                               # fall through to decide()
                else:
                    break

            # 6) Check the strategy’s decide() method to see if we should keep rolling.
            #    Pass in:
            #      - current turn_score
            #      - dice_left = dice (from step 5)
            #      - has_scored = whether we’ve ever scored ≥500 on a previous turn
            #      - score_needed = how many more points (from banked + turn_score) we need to reach target_score
            keep_rolling = self.strategy.decide(
                turn_score   = turn_score,
                dice_left    = dice,
                has_scored   = self.has_scored,
                score_needed = max(0, target_score - running_total),
                final_round  = final_round,
                score_to_beat= score_to_beat,
                running_total= running_total,
            )
        
            # 6-C) Override the strategy if we’re still behind in the final round.
            if final_round and running_total <= score_to_beat:
                keep_rolling = True

            if not keep_rolling:
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
    def play(self, max_rounds: int = 50) -> GameMetrics:
        """Run the game and return a *GameMetrics* summary."""
        final_round = False
        score_to_beat = self.target_score   # updated when someone triggers
        rounds = 0
        while rounds < max_rounds and final_round == False:
            rounds += 1
            for p in self.players:
                p.take_turn(
                    self.target_score,  # This is that vestigial stat 
                    final_round = final_round,
                    score_to_beat = score_to_beat,
                )
                # First trigger starts the final round
                if not final_round and p.score >= self.target_score:
                    final_round   = True
                    score_to_beat = p.score
                    triggering_player = p
                    final_players = [player for player in self.players if player != triggering_player]
                
                if final_round:
                    for player in final_players:  # All other players have chance to beat the first tentative winner
                                                  # It's not fair, but it's the rules
                        player.take_turn(target_score=self.target_score,
                                         final_round=True,
                                         score_to_beat=score_to_beat)    
                # During the final round update the bar whenever someone
                #     jumps ahead (so later players know what they must beat).
                        if player.score > score_to_beat:
                            score_to_beat = p.score
                        
                        # whole table has now had exactly one shot
                        
                if final_round:
                    break
            if final_round:
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
