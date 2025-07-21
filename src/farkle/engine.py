from __future__ import annotations

import warnings
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

# Maximum number of rolls permitted in a single turn before aborting
ROLL_LIMIT: int = 1000


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

    def _score_roll(self, roll: DiceRoll, turn_score: int) -> tuple[int, int, bool]:
        """Return points, dice to reroll and whether we farkled."""
        pts, used, reroll, d5, d1 = default_score(  # type: ignore[misc]
            dice_roll=roll,
            turn_score_pre=turn_score,
            smart_five=self.strategy.smart_five,
            smart_one=self.strategy.smart_one,
            score_threshold=self.strategy.score_threshold,
            dice_threshold=self.strategy.dice_threshold,
            consider_score=self.strategy.consider_score,
            consider_dice=self.strategy.consider_dice,
            require_both=self.strategy.require_both,
            prefer_score=self.strategy.prefer_score,
            return_discards=True,
        )

        if pts == 0:
            self.n_farkles += 1
            return 0, 0, True

        if d5:
            self.smart_five_uses += 1
            self.n_smart_five_dice += d5
        if d1:
            self.smart_one_uses += 1
            self.n_smart_one_dice += d1

        next_dice = 6 if (used == len(roll) and reroll == 0) else reroll
        return pts, next_dice, False

    def _apply_hot_dice(self, dice: int) -> bool:
        """Return True if auto-hot-dice forces another roll."""
        if self.strategy.auto_hot_dice and dice == 6:
            self.n_hot_dice += 1
            return True
        return False

    def _should_continue(
        self,
        *,
        turn_score: int,
        dice_left: int,
        final_round: bool,
        score_to_beat: int,
        target_score: int,
    ) -> bool:
        running_total = self.score + turn_score
        score_needed = max(0, target_score - running_total)

        if final_round and running_total > score_to_beat:
            if not self.strategy.run_up_score:
                return False

        keep_rolling = self.strategy.decide(
            turn_score=turn_score,
            dice_left=dice_left,
            has_scored=self.has_scored,
            score_needed=score_needed,
            final_round=final_round,
            score_to_beat=score_to_beat,
            running_total=running_total,
        )

        if final_round and running_total <= score_to_beat:
            keep_rolling = True

        return keep_rolling

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
        Raises
        ------
        RuntimeError
            If the number of rolls in this turn exceeds ``ROLL_LIMIT``.
        """
        dice = 6
        turn_score = 0
        rolls_this_turn = 0

        while dice > 0:
            if rolls_this_turn > ROLL_LIMIT:
                raise RuntimeError(
                    f"Turn exceeded {ROLL_LIMIT} rolls - aborting."
                )
            roll = self._roll(dice)
            rolls_this_turn += 1
            pts, dice, bust = self._score_roll(roll, turn_score)
            if bust:
                turn_score = 0
                break

            turn_score += pts

            if self._apply_hot_dice(dice):
                continue

            if not self._should_continue(
                turn_score=turn_score,
                dice_left=dice,
                final_round=final_round,
                score_to_beat=score_to_beat,
                target_score=target_score,
            ):
                break

        # After leaving the loop, check if this is our “entry turn"
        # (we need ≥500 to get on the board)
        if not self.has_scored and turn_score >= 500:
            self.has_scored = True

        # If we already had a "first‐500" in a previous turn, or just got it, bank the points
        if self.has_scored:
            self.score += turn_score
            self.highest_turn = max(self.highest_turn, turn_score)


# ---------------------------------------------------------------------------
# Game-level structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GameStats:
    """Aggregate results of one game.

    Attributes
    ----------
    n_players
        Number of participants.
    table_seed
        Seed used for the table RNG.
    n_rounds
        Rounds played before a winner emerged.
    total_rolls
        Combined dice rolls for all players.
    total_farkles
        Total number of Farkles rolled.
    margin
        Points separating first and second place.
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
    """Per-game statistics keyed by player name.

    Attributes
    ----------
    players
        Mapping from player name to :class:`PlayerStats`.
    game
        :class:`GameStats` summarising the overall session.
    """

    players: Dict[str, PlayerStats]
    game: GameStats

    @property
    def players_dict(self) -> Dict[str, Dict[str, int]]:
        """Return per-player statistics as plain dictionaries.

        This property mirrors the legacy :attr:`per_player` attribute which
        originally exposed player results before :class:`PlayerStats` existed.
        """
        return {n: asdict(ps) for n, ps in self.players.items()}

    # legacy – keep old call‑sites alive until migrated
    @property
    def per_player(self):
        """Deprecated alias for :attr:`players_dict`.

        The original :class:`GameMetrics` exposed player results via
        ``per_player`` before :class:`PlayerStats` was introduced.  Existing
        code may still rely on that attribute, so it now forwards to
        :attr:`players_dict` and emits a :class:`DeprecationWarning`.
        """
        warnings.warn(
            "GameMetrics.per_player is deprecated; use players_dict instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.players_dict

    # ------------------------------------------------------------------
    # Compatibility helpers for the previous GameMetrics API
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
    """Statistics for a single player.

    Attributes
    ----------
    score
        Final score for the game.
    farkles
        Number of times the player farkled.
    rolls
        Dice rolls taken across all turns.
    highest_turn
        Highest single-turn score.
    strategy
        String representation of the strategy used.
    rank
        Finishing position (1 for the winner).
    loss_margin
        Point difference from the winner (``0`` if they won).
    smart_five_uses, n_smart_five_dice
        Counts for Smart‑5 heuristic usage and dice removed.
    smart_one_uses, n_smart_one_dice
        Counts for Smart‑1 heuristic usage and dice removed.
    hot_dice
        Number of hot-dice rerolls.
    """

    score: int
    farkles: int
    rolls: int
    highest_turn: int
    strategy: str
    rank: int  # 1 = winner
    loss_margin: int  # 0 for winner, >0 otherwise
    smart_five_uses: int = 0
    n_smart_five_dice: int = 0
    smart_one_uses: int = 0
    n_smart_one_dice: int = 0
    hot_dice: int = 0


class FarkleGame:
    """Driver for a *single* Farkle game."""

    def __init__(
        self,
        players: Sequence[FarklePlayer],
        *,
        target_score: int = 10_000,
        table_seed: int = 0,
    ) -> None:
        """Create a new game instance.

        Inputs
        ------
        players
            Participants in turn order.
        target_score
            Score that triggers the final round.
        table_seed
            Seed for any table-level randomness.
        """
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
            for player in self.players:
                player.take_turn(
                    self.target_score,  # This is that vestigial stat
                    final_round = final_round,
                    score_to_beat = score_to_beat,
                )
                # First trigger starts the final round
                if not final_round and player.score >= self.target_score:
                    final_round = True
                    score_to_beat = player.score
                    final_players = [pl for pl in self.players if pl != player]
                    score_to_beat = self._run_final_round(final_players, score_to_beat)

                    break
            if final_round:
                break

        sorted_players = sorted(self.players, key=lambda pl: pl.score, reverse=True)
        winner = sorted_players[0]
        runner = sorted_players[1] if len(sorted_pl) > 1 else None
        ranks = {player.name: rk for rk, player in enumerate(sorted_players, start=1)}
        
        players_block: Dict[str, PlayerStats] = {}
        for player in sorted_players:
            players_block[player.name] = PlayerStats(
                score = player.score,
                farkles = player.n_farkles,
                rolls = player.n_rolls,
                highest_turn = player.highest_turn,
                strategy = str(player.strategy),
                rank = ranks[player.name],
                loss_margin = winner.score - player.score,
                smart_five_uses = player.smart_five_uses,
                n_smart_five_dice = player.n_smart_five_dice,
                smart_one_uses = player.smart_one_uses,
                n_smart_one_dice = player.n_smart_one_dice,
                hot_dice = player.n_hot_dice,
            )

        game_block = GameStats(
            n_players = len(self.players),
            table_seed = self.table_seed,
            n_rounds = rounds,
            total_rolls = sum(player.n_rolls for player in self.players),
            total_farkles = sum(player.n_farkles for player in self.players),
            margin = winner.score - (runner.score if runner else 0),
        )
        
        return GameMetrics(players_block, game_block)

      
    def _run_final_round(self, final_players: Sequence[FarklePlayer], score_to_beat: int) -> int:
        """Give each remaining player one last turn.

        Parameters
        ----------
        final_players
            Players who get a final chance to beat the trigger score.
        score_to_beat
            The current leading score at the start of the round.

        Returns
        -------
        int
            Updated score to beat after all final turns.
        """
        for player in final_players:
            # All other players have exactly one chance to overtake
            player.take_turn(
                target_score=self.target_score,
                final_round=True,
                score_to_beat=score_to_beat,
            )

            # Update bar so later players know what they must beat
            if player.score > score_to_beat:
                score_to_beat = player.score

        return score_to_beat
