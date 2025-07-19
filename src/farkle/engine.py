from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from farkle.scoring import DiceRoll, default_score
from farkle.strategies import ThresholdStrategy

"""engine.py
============
Player and single-game engine for Farkle simulations.

High-level flow
---------------
* FarkleGame.play drives a game loop until someone reaches
  target_score.  When that happens the *final round* rule is applied:
  every other player gets one last turn.
* FarklePlayer.take_turn handles the intra-turn loop of rolling,
  scoring, and consulting its strategy.

The module keeps no global state; randomness lives inside each
FarklePlayer via its dedicated random.Random instance (passed in
from the outer simulation layer).
"""


__all__ = [
    "FarklePlayer",
    "PlayerStats",
    "GameStats",
    "GameMetrics",
    "FarkleGame",
]


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FarklePlayer:
    """A single player in a Farkle game."""

    name: str
    strategy: ThresholdStrategy
    score: int = 0
    has_scored: bool = False  # entered game (≥500) flag
    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    
    # counters
    n_farkles: int = 0
    n_rolls: int = 0
    highest_turn: int = 0
    smart_five_uses: int = 0
    n_smart_five_dice: int = 0
    smart_one_uses: int = 0
    n_smart_one_dice: int = 0
    n_hot_dice: int = 0
    strategy_str: str = field(init=False, repr=False)
    
    def __post_init__(self):
        object.__setattr__(self, "strategy_str", str(self.strategy))
        
    # ----------------------------- helpers -----------------------------
    def _roll(self, n: int) -> DiceRoll:
        """Produce n dice using this player's RNG.

        Inputs
        ------
        n
            Number of dice to roll.

        Returns
        -------
        DiceRoll
            List of integers in the range 1-6.
        """
        self.n_rolls += 1
        # ask for **one** uint32 with the classic 3-arg signature
        # Use the RNG directly so deterministic test generators work.
        return self.rng.integers(1, 7, size=n).tolist()

    # ------------------------------ turn --------------------------------
    def take_turn(
        self,
        target_score: int,
        *,
        final_round: bool = False,
        score_to_beat: int = 0,
    ) -> None:
        """Simulates a full turn for this player.

        Inputs
        ------
        target_score
            Score that ends the game and triggers the final round.
        final_round
            Whether this turn occurs during the closing round.
        score_to_beat
            Current leading score in the final round.

        Returns
        -------
        None
            The player's internal state is updated in place.
        """
        dice = 6
        turn_score = 0
        rolls_this_turn = 0

        while dice > 0:
            if rolls_this_turn > 1000:
                raise RuntimeError("Turn exceeded 1000 rolls - aborting.")
            # 1) Roll dice number of dice
            roll = self._roll(dice)
            rolls_this_turn += 1

            # 2) Compute points from this roll, after applying Smart‐5/Smart‐1 discards
            pts, used, reroll, d5, d1 = default_score( # type: ignore
                dice_roll = roll,
                turn_score_pre = turn_score,
                smart_five = self.strategy.smart_five,
                smart_one = self.strategy.smart_one,
                score_threshold = self.strategy.score_threshold,
                dice_threshold = self.strategy.dice_threshold,
                consider_score = self.strategy.consider_score,
                consider_dice = self.strategy.consider_dice,
                require_both = self.strategy.require_both,
                prefer_score = self.strategy.prefer_score,
                return_discards = True
            )

            # 3) If pts == 0, that's a Farkle: bust, lose all points this turn, and end turn
            if pts == 0:  # farkle bust
                self.n_farkles += 1
                turn_score = 0
                break

            # 4) Otherwise, accumulate the points from this roll
            turn_score += pts
            
            # 4.5) Update smart_five and smart_one tracking
            if d5:
                self.smart_five_uses += 1
                self.n_smart_five_dice += d5
            if d1:
                self.smart_one_uses += 1
                self.n_smart_one_dice += d1
            
            # 5) “Hot dice” logic: if all dice in roll scored (used == len(roll)),
            #    and reroll == 0, that means you get to roll all 6 dice again.
            #    Otherwise, you roll reroll dice next.
            dice = 6 if (used == len(roll) and reroll == 0) else reroll
            
            # 5.5) Hot dice short circuit
            if self.strategy.auto_hot_dice and dice == 6:
                self.n_hot_dice += 1
                continue  # force another throw
            
            # 6-A) If we’re in the final round and have already won, stop.
            running_total = self.score + turn_score
            score_needed = max(0, target_score - running_total)
            if final_round and running_total > score_to_beat:
                if self.strategy.run_up_score:  # NEW opt-in flag
                    pass  # fall through to decide()
                else:
                    break

            # 6) Check the strategy’s decide() method to see if we should keep rolling.
            #    Pass in:
            #      - current turn_score
            #      - dice_left = dice (from step 5)
            #      - has_scored = whether we’ve ever scored ≥500 on a previous turn
            #      - score_needed = how many more points (from banked + turn_score) we need to reach target_score
            keep_rolling = self.strategy.decide(
                turn_score = turn_score,
                dice_left = dice,
                has_scored = self.has_scored,
                score_needed = score_needed,
                final_round = final_round,
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
            self.score += turn_score
            self.highest_turn = max(self.highest_turn, turn_score)


# ---------------------------------------------------------------------------
# Game-level structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GameStats:
    """Summary statistics describing a completed game.

    Inputs
    ------
    winner
        Name of the winning player.
    winning_score
        Score achieved by the winner.
    n_rounds
        Number of rounds played.
    per_player
        Stats for each player keyed by name.
    winner_data
        Stats for the winner only.

    Returns
    -------
    GameMetrics
        New dataclass instance populated with results.
    """
    n_players: int
    table_seed: int
    n_rounds: int
    total_rolls: int
    total_farkles: int
    margin: int
    # per player and winner pulled into separate class, PlayerStats


@dataclass(slots=True)
class GameMetrics:
    players: Dict[str, PlayerStats]
    game: GameStats

    # legacy – keep old call‑sites alive until migrated
    @property
    def per_player(self):
        return {n: asdict(ps) for n, ps in self.players.items()}

    # ------------------------------------------------------------------
    # Compatability helpers for the previous GameMetrics API
    # ------------------------------------------------------------------
    @property
    def winner(self) -> str:
        return max(self.players.items(), key=lambda p: p[1].score)[0]

    @property
    def winning_score(self) -> int:
        return self.players[self.winner].score

    @property
    def n_rounds(self) -> int:
        return self.game.n_rounds


@dataclass(slots=True)
class PlayerStats:
    score: int
    farkles: int
    rolls: int
    highest_turn: int
    strategy: str
    rank: int # 1 = winner
    loss_margin: int # 0 for winner, >0 otherwise
    smart_five_uses: int = 0
    n_smart_five_dice: int = 0
    smart_one_uses: int = 0
    n_smart_one_dice: int = 0
    hot_dice: int = 0


class FarkleGame:
    """Driver for a *single* Farkle game."""

    def __init__(self, players: Sequence[FarklePlayer], *, target_score: int = 10_000, table_seed: int = 0) -> None:
        self.players: List[FarklePlayer] = list(players)
        self.target_score: int = target_score
        self.table_seed = table_seed

    # ---------------------------- gameplay -----------------------------
    def play(self, max_rounds: int = 50) -> GameMetrics:
        """Execute the game loop and return final statistics.

        Inputs
        ------
        max_rounds
            Safety cap on the number of rounds played.

        Returns
        -------
        GameMetrics
            Dataclass summarising the winner and per-player stats.
        """
        final_round = False
        score_to_beat = self.target_score  # updated when someone triggers
        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            for p in self.players:
                p.take_turn(
                    self.target_score,  # This is that vestigial stat 
                    final_round = final_round,
                    score_to_beat = score_to_beat,
                )
                # First trigger starts the final round
                if not final_round and p.score >= self.target_score:
                    final_round = True
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
                            score_to_beat = player.score
                        
                        # whole table has now had exactly one shot
                        
                if final_round:
                    break
            if final_round:
                break
        
        
        sorted_pl = sorted(self.players, key=lambda pl: pl.score, reverse=True)
        winner = sorted_pl[0]
        runner = sorted_pl[1] if len(sorted_pl) > 1 else None

        ranks = {pl.name: rk for rk, pl in enumerate(sorted_pl, start=1)}
        players_block: Dict[str, PlayerStats] = {}
        for pl in sorted_pl:
            players_block[pl.name] = PlayerStats(
                score = pl.score,
                farkles = pl.n_farkles,
                rolls = pl.n_rolls,
                highest_turn = pl.highest_turn,
                strategy = str(pl.strategy),
                rank = ranks[pl.name],
                loss_margin = winner.score - pl.score,
                smart_five_uses = pl.smart_five_uses,
                n_smart_five_dice = pl.n_smart_five_dice,
                smart_one_uses = pl.smart_one_uses,
                n_smart_one_dice = pl.n_smart_one_dice,
                hot_dice = pl.n_hot_dice,
            )

        game_block = GameStats(
            n_players = len(self.players),
            table_seed = self.table_seed,
            n_rounds = rounds,
            total_rolls = sum(pl.n_rolls for pl in self.players),
            total_farkles = sum(pl.n_farkles for pl in self.players),
            margin = winner.score - (runner.score if runner else 0),
        )

        return GameMetrics(players_block, game_block)
    
        # Legacy metrics engine left here just in case
        # from typing import Any
        #         
        # winner = max(self.players, key=lambda pl: pl.score)
        # winner_data: Dict = {
        #     winner.name: {
        #         "score": p.score,
        #         "farkles": p.n_farkles,
        #         "rolls": p.n_rolls,
        #         "highest_turn": p.highest_turn,
        #         "strategy": str(p.strategy),
        #     }
        # }    
        # per_player: Dict[str, Dict[str, Any]] = {
        #     p.name: {
        #         "score": p.score,
        #         "farkles": p.n_farkles,
        #         "rolls": p.n_rolls,
        #         "highest_turn": p.highest_turn,
        #         "strategy": str(p.strategy),
        #     }
        #     for p in self.players
        # }
        #
        # return GameMetrics(winner.name, winner.score, rounds, per_player, winner_data)
