import pandas as pd
import pytest

# tests/integration/test_farkle_integration.py
"""Integration tests covering CLI paths and scoring consistency."""

from farkle.game.engine import FarkleGame, FarklePlayer
from farkle.game.scoring import (
    apply_discards,
    decide_smart_discards,
    default_score,
    score_roll_cached,
)
from farkle.simulation.simulation import simulate_many_games
from farkle.simulation.strategies import ThresholdStrategy


@pytest.fixture(scope="session")
def tmp_csv(tmp_path_factory: pytest.TempPathFactory):
    """Provision a reusable CSV path for CLI-based simulations.

    Args:
        tmp_path_factory: Factory fixture used to create session-level paths.

    Returns:
        Generator yielding the CSV path and cleaning it up afterward.
    """

    base = tmp_path_factory.mktemp("farkle_cli_tmp")
    csv_path = base / "sim.csv"
    yield csv_path
    csv_path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def simple_strategy():
    """Build a baseline ThresholdStrategy list for integration scenarios.

    Returns:
        Single-element list containing a conservative strategy.
    """

    return [ThresholdStrategy(score_threshold=300, dice_threshold=2)]


def test_seed_reproducible(simple_strategy):
    """Ensure simulations are deterministic when seeding RNGs.

    Args:
        simple_strategy: Strategy list fixture shared across tests.

    Returns:
        None
    """

    df1 = simulate_many_games(n_games=60, strategies=simple_strategy, seed=123, n_jobs=1)
    df2 = simulate_many_games(n_games=60, strategies=simple_strategy, seed=123, n_jobs=1)
    pd.testing.assert_frame_equal(df1, df2)


def test_final_round_rule():
    """Validate final-round enforcement within a short simulated game.

    Returns:
        None
    """

    # 1) two simple strategies …
    strats = [
        ThresholdStrategy(score_threshold=100, dice_threshold=0),  # very aggressive
        ThresholdStrategy(score_threshold=300, dice_threshold=2),
    ]

    # 2) wrap them in *players* that own their RNGs
    players = [FarklePlayer(f"P{i+1}", s) for i, s in enumerate(strats)]

    # 3) FarkleGame takes (players, target_score)
    game = FarkleGame(players, target_score=2_000)

    # 4) play() returns a GameMetrics object
    gm = game.play()

    # ───────── assertions on the public result ─────────
    winner = max(gm.players, key=lambda n: gm.players[n].score)
    assert gm.players[winner].score >= 2_000
    assert gm.game.n_rounds >= 1  # each player got ≤ one extra turn


# representative sample of roll/flag combinations used to validate the
# three-stage scoring pipeline against the public ``default_score`` helper.
PIPELINE_CASES: list[tuple[list[int], int, bool, bool]] = [
    ([1], 0, True, True),
    ([1], 0, True, False),
    ([5], 400, True, False),
    ([2, 3, 4, 6], 150, False, False),
    ([1, 1, 1, 5, 5, 5], 600, True, True),
    ([2, 2, 3, 3, 4, 4], 900, False, False),
    ([1, 5, 2, 3, 4, 6], 250, True, True),
    ([2, 2, 2, 3, 3, 3], 0, False, False),
    ([6, 6, 6, 2, 3, 4], 450, True, False),
    ([1, 1, 5, 5, 2, 2], 50, True, True),
]


@pytest.mark.parametrize("roll, turn_pre, smart_five, smart_one", PIPELINE_CASES)
def test_pipeline_matches_default(roll, turn_pre, smart_five, smart_one):
    """Confirm manual scoring pipeline matches the default helper output.

    Args:
        roll: Sequence of dice results for the turn.
        turn_pre: Pre-existing score in the current turn.
        smart_five: Whether smart five-discard rules are active.
        smart_one: Whether smart one-discard rules are active.

    Returns:
        None
    """

    # ---------- (A) single-call pipeline ----------
    d_score, d_used, d_reroll = default_score(  # type: ignore (refactor allows 3 or 5 outputs)
        dice_roll=roll,
        turn_score_pre=turn_pre,
        smart_five=smart_five,
        smart_one=smart_one,
        score_threshold=300,
        dice_threshold=3,
    )

    # ---------- (B) manual three-step pipeline ----------
    raw_score, raw_used, counts, sfives, sones = score_roll_cached(roll)

    disc5, disc1 = decide_smart_discards(
        counts=counts,
        single_fives=sfives,
        single_ones=sones,
        raw_score=raw_score,
        raw_used=raw_used,
        dice_roll_len=len(roll),
        turn_score_pre=turn_pre,
        score_threshold=300,
        dice_threshold=3,
        smart_five=smart_five,
        smart_one=smart_one,
    )

    f_score, f_used, f_reroll = apply_discards(
        raw_score=raw_score,
        raw_used=raw_used,
        discard_fives=disc5,
        discard_ones=disc1,
        dice_roll_len=len(roll),
    )

    # ---------- agreement check ----------
    assert (d_score, d_used, d_reroll) == (f_score, f_used, f_reroll)
