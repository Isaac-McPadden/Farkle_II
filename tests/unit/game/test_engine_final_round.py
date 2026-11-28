from __future__ import annotations

"""Tests covering final-round behavior for the game engine."""

from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np
import pytest

from farkle.game import engine
from farkle.game.engine import FarkleGame, FarklePlayer
from farkle.simulation.strategies import ThresholdStrategy


@dataclass
class ScriptedRNG:
    """Deterministic RNG that returns predefined rolls in order."""

    script: Sequence[Sequence[int]]
    pos: int = 0

    def integers(self, low: int, high: int, size: int) -> np.ndarray:  # noqa: ARG002
        """Return the next scripted die roll sequence.

        Args:
            low: Lower bound for faces (unused in the scripted path).
            high: Upper bound for faces (unused in the scripted path).
            size: Number of dice expected for the roll.

        Returns:
            A NumPy array containing the next set of predetermined faces.
        """

        try:
            faces: Sequence[int] = self.script[self.pos]
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("Ran out of scripted dice rolls") from exc
        self.pos += 1
        if len(faces) != size:
            raise ValueError(f"Expected {size} dice but script provided {len(faces)}")
        return np.asarray(faces, dtype=int)


@dataclass(slots=True)
class _QuietStrategy(ThresholdStrategy):
    """Minimal strategy that always banks after scoring."""

    consider_score: bool = False
    consider_dice: bool = False
    smart_five: bool = False
    smart_one: bool = False
    auto_hot_dice: bool = False
    run_up_score: bool = False

    def decide(self, *_, **__) -> bool:  # type: ignore[override]
        """Always bank after scoring.

        Returns:
            False to indicate no additional reroll should occur.
        """

        return False


@pytest.mark.unit
def test_reaching_target_exactly_triggers_final_round_and_respects_score_to_beat():
    """Ensure reaching the target starts the final round and requires a higher score.

    Returns:
        None
    """

    # Player 1 triggers the final round by reaching the target exactly.
    p1 = FarklePlayer(
        name="opener",
        strategy=_QuietStrategy(),
        rng=cast(np.random.Generator, ScriptedRNG([[5, 5, 5, 2, 3, 4]])),
    )

    # Player 2 farkles immediately during the final round.
    p2 = FarklePlayer(
        name="bust",
        strategy=_QuietStrategy(),
        rng=cast(np.random.Generator, ScriptedRNG([[2, 3, 4, 6, 2, 4]])),
    )

    # Player 3 leaps past the score_to_beat with a single roll and should bank.
    p3 = FarklePlayer(
        name="closer",
        strategy=_QuietStrategy(),
        rng=cast(np.random.Generator, ScriptedRNG([[1, 1, 1, 2, 2, 2]])),
    )

    game = FarkleGame([p1, p2, p3], target_score=500)
    metrics = game.play(max_rounds=5)

    assert metrics.winner == "closer"
    assert metrics.players["opener"].score == 500
    assert metrics.players["closer"].score == 2500
    assert metrics.players["bust"].score == 0
    assert metrics.game.n_rounds == 1
    assert metrics.game.total_rolls == 3
    assert metrics.game.total_farkles == 1
    assert metrics.game.margin == 2000
    assert [metrics.players[name].n_turns for name in ("opener", "bust", "closer")] == [1, 1, 1]


@pytest.mark.parametrize("trigger_seat", [0, 1, 2])
def test_turn_counts_follow_final_round_trigger_position(trigger_seat: int) -> None:
    """Earlier seats receive a normal turn plus one closing turn after a later trigger."""

    players: list[FarklePlayer] = []
    for seat in range(3):
        if seat == trigger_seat:
            script = [[1, 1, 1, 2, 2, 2]]
        elif seat < trigger_seat:
            script = [
                [5, 5, 5, 2, 3, 4],
                [2, 3, 4, 6, 2, 4],
            ]
        else:
            script = [[2, 3, 4, 6, 2, 4]]
        players.append(
            FarklePlayer(
                name=f"P{seat + 1}",
                strategy=_QuietStrategy(),
                rng=cast(np.random.Generator, ScriptedRNG(script)),
            )
        )

    metrics = FarkleGame(players, target_score=2_000).play(max_rounds=5)

    assert [metrics.players[f"P{seat + 1}"].n_turns for seat in range(3)] == [
        1 if seat >= trigger_seat else 2 for seat in range(3)
    ]


BUST = (2, 2, 3, 3, 4, 6)
FIVE_HUNDRED = (5, 5, 5, 2, 3, 4)
SIX_HUNDRED = (6, 6, 6, 2, 3, 4)
THOUSAND = (3, 3, 3, 3, 2, 6)
STRAIGHT = (1, 2, 3, 4, 5, 6)


@pytest.mark.parametrize(
    "trigger, first, second, last",
    [(1, 2, 3, 4), (2, 3, 4, 1), (3, 4, 1, 2), (4, 1, 2, 3)],
)
def test_challenges_follow_seats_and_raise_the_lead(trigger, first, second, last):
    # One shared script makes each challenge outcome identify its seat. The last
    # challenger ties the second one's 1,000 and then busts on the two remaining dice.
    dice = ScriptedRNG(
        [BUST] * (trigger - 1) + [FIVE_HUNDRED, SIX_HUNDRED, THOUSAND, THOUSAND, (2, 3)]
    )
    players = [
        FarklePlayer(f"P{seat}", _QuietStrategy(), cast(np.random.Generator, dice))
        for seat in range(1, 5)
    ]
    game = FarkleGame(players, target_score=500).play(max_rounds=1)

    assert dice.pos == len(dice.script)
    assert [game.players[f"P{s}"].score for s in (trigger, first, second, last)] == [
        500,
        600,
        1000,
        0,
    ]
    assert game.winner == f"P{second}"
    assert game.game.margin == 400
    assert [p.n_turns for p in players] == [1 + int(s < trigger) for s in range(1, 5)]
    assert game.players[f"P{last}"].farkles == 1 + int(last < trigger)


def scripted_player(
    strategy: ThresholdStrategy, *rolls: Sequence[int], name: str = "P"
) -> FarklePlayer:
    return FarklePlayer(name, strategy, cast(np.random.Generator, ScriptedRNG(rolls)))


@pytest.mark.parametrize(
    "run_up, auto_hot, final_score, rolls, automatic",
    [
        (False, False, 1500, 1, 0),
        (False, True, 1500, 1, 0),
        (True, False, 1500, 1, 0),
        (True, True, 0, 2, 1),
    ],
)
def test_bankable_final_lead_on_hot_dice(run_up, auto_hot, final_score, rolls, automatic):
    strategy = ThresholdStrategy(
        score_threshold=500, consider_dice=False, run_up_score=run_up, auto_hot_dice=auto_hot
    )
    player = scripted_player(strategy, STRAIGHT, BUST)
    assert strategy.decide(
        turn_score=1500,
        dice_left=6,
        has_scored=False,
        score_needed=0,
        final_round=True,
        score_to_beat=500,
        running_total=1500,
    ) is (rolls == 2)
    player.take_turn(500, final_round=True, score_to_beat=500)
    assert (player.score, player.n_rolls, player.n_hot_dice) == (final_score, rolls, automatic)
    assert player.n_farkles == rolls - 1


@pytest.mark.parametrize("run_up, expected", [(False, 1500), (True, 0)])
def test_final_lead_can_risk_a_voluntary_hot_roll(run_up, expected):
    strategy = ThresholdStrategy(
        score_threshold=2000, consider_dice=False, run_up_score=run_up, auto_hot_dice=False
    )
    player = scripted_player(strategy, STRAIGHT, BUST)
    player.take_turn(500, final_round=True, score_to_beat=500)
    assert player.score == expected
    assert player.n_hot_dice == 0


def test_equality_continues_until_a_strict_lead():
    player = scripted_player(_QuietStrategy(), FIVE_HUNDRED, (5, 2, 3))
    player.take_turn(500, final_round=True, score_to_beat=500)
    assert (player.score, player.n_rolls, player.highest_turn) == (550, 2, 550)


@pytest.mark.parametrize("auto_hot, automatic", [(False, 0), (True, 1)])
def test_hot_dice_while_tied_requires_another_roll(auto_hot, automatic):
    player = scripted_player(ThresholdStrategy(auto_hot_dice=auto_hot), STRAIGHT, SIX_HUNDRED)
    player.take_turn(1500, final_round=True, score_to_beat=1500)
    assert (player.score, player.n_rolls, player.n_hot_dice) == (2100, 2, automatic)


@pytest.mark.parametrize("challenge", [False, True])
def test_entry_precedes_banking_and_later_small_banks_are_legal(challenge):
    # Exercise entry even when a caller supplies a score to beat below 500.
    player = scripted_player(_QuietStrategy(), (4, 4, 4, 2, 3, 6), (1, 2, 3), (5, 2, 2, 3, 3, 6))
    player.take_turn(300, final_round=challenge, score_to_beat=300)
    assert (player.has_scored, player.score, player.n_rolls) == (True, 500, 2)
    player.take_turn(10_000)
    assert (player.score, player.highest_turn, player.n_turns, player.n_rolls) == (550, 500, 2, 3)


def test_ordinary_target_crossing_then_farkle_leaves_game_open():
    player = scripted_player(
        ThresholdStrategy(score_threshold=700, consider_dice=False), SIX_HUNDRED, (2, 3, 4)
    )
    opponent = scripted_player(_QuietStrategy(), BUST, name="opponent")
    game = FarkleGame([player, opponent], target_score=500).play(max_rounds=1)
    assert (player.score, player.has_scored, player.n_rolls, player.n_farkles) == (0, False, 2, 1)
    assert game.game.hit_safety_limit and game.winner is None and game.players["P"].rank is None


def test_final_challenge_at_the_roll_limit(monkeypatch):
    monkeypatch.setattr(engine, "ROLL_LIMIT", 1)
    leader = scripted_player(_QuietStrategy(), FIVE_HUNDRED)
    tied = scripted_player(_QuietStrategy(), FIVE_HUNDRED, name="challenger")
    with pytest.raises(RuntimeError, match="exceeded 1 rolls"):
        FarkleGame([leader, tied], target_score=500).play(max_rounds=1)
    assert (leader.score, leader.n_rolls) == (500, 1)
    assert (tied.score, tied.has_scored, tied.n_turns, tied.n_rolls) == (0, False, 1, 1)


@pytest.mark.parametrize(
    "auto_hot, require_both, first_roll, next_bust, bank, rolls, automatic",
    [
        # On the straight, score passes but dice reset to six (above the threshold).
        (False, False, STRAIGHT, BUST, 1500, 1, 0),
        (False, True, STRAIGHT, BUST, 0, 2, 0),
        (True, False, STRAIGHT, BUST, 0, 2, 1),
        (True, True, STRAIGHT, BUST, 0, 2, 1),
        # Three remaining dice meet the threshold, but 600 is below the score threshold.
        (True, False, SIX_HUNDRED, (2, 3, 4), 600, 1, 0),
        (True, True, SIX_HUNDRED, (2, 3, 4), 0, 2, 0),
    ],
)
def test_ordinary_criteria_after_scoring(
    auto_hot, require_both, first_roll, next_bust, bank, rolls, automatic
):
    strategy = ThresholdStrategy(
        score_threshold=1000, dice_threshold=3, require_both=require_both, auto_hot_dice=auto_hot
    )
    player = scripted_player(strategy, first_roll, next_bust)
    player.take_turn(10_000)
    assert (player.score, player.n_rolls, player.n_hot_dice) == (bank, rolls, automatic)
