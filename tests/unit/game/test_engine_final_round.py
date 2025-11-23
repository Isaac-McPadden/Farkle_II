from __future__ import annotations

"""Tests covering final-round behavior for the game engine."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pytest

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
def test_final_round_respects_score_to_beat_and_reruns():
    """Ensure the final round replays until the leading score is surpassed.

    Returns:
        None
    """

    # Player 1 triggers the final round by crossing the target on the opening turn.
    p1 = FarklePlayer(
        name="opener",
        strategy=_QuietStrategy(),
        rng=ScriptedRNG([[5, 5, 5, 2, 3, 4]]),
    )

    # Player 2 farkles immediately during the final round.
    p2 = FarklePlayer(name="bust", strategy=_QuietStrategy(), rng=ScriptedRNG([[2, 3, 4, 6, 2, 4]]))

    # Player 3 leaps past the score_to_beat with a single roll and should bank.
    p3 = FarklePlayer(
        name="closer",
        strategy=_QuietStrategy(),
        rng=ScriptedRNG([[1, 1, 1, 2, 2, 2]]),
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
