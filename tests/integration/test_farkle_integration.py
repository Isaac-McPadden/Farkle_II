import contextlib
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from farkle.cli import main as cli_main
from farkle.game.engine import FarkleGame, FarklePlayer
from farkle.game.scoring import (
    apply_discards,
    decide_smart_discards,
    default_score,
    score_roll_cached,
)
from farkle.simulation.simulation import simulate_many_games
from farkle.simulation.strategies import ThresholdStrategy

TMP = Path(__file__).with_suffix("") / "_tmp"
TMP.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def tmp_csv():
    """Unique CSV path per test-session that we can freely overwrite."""
    csv_path = TMP / "sim.csv"
    # clean between test runs
    if csv_path.exists():
        with contextlib.suppress(PermissionError):
            csv_path.unlink()
    yield csv_path
    with contextlib.suppress(PermissionError):
        csv_path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def simple_strategy():
    return [ThresholdStrategy(score_threshold=300, dice_threshold=2)]


def test_seed_reproducible(simple_strategy):
    df1 = simulate_many_games(n_games=60, strategies=simple_strategy, seed=123, n_jobs=1)
    df2 = simulate_many_games(n_games=60, strategies=simple_strategy, seed=123, n_jobs=1)
    pd.testing.assert_frame_equal(df1, df2)


def test_cli_smoke(tmp_path: Path, capinfo, monkeypatch: pytest.MonkeyPatch):
    """Smoke-test the unified CLI by exercising the ``run`` command."""

    captured: dict[str, object] = {}

    def fake_run_tournament(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_tournament", fake_run_tournament)

    cfg = {
        "global_seed": 9,
        "n_players": 2,
        "num_shuffles": 1,
        "checkpoint_path": str(tmp_path / "checkpoint.pkl"),
        "row_output_directory": str(tmp_path / "rows"),
    }

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cli_main.main(["--config", str(cfg_path), "run"])

    assert captured["n_players"] == 2
    assert captured["num_shuffles"] == 1
    assert captured["global_seed"] == 9
    chk = captured["checkpoint_path"]
    assert isinstance(chk, (str, os.PathLike))
    assert Path(chk) == tmp_path / "checkpoint.pkl"

    rowdir = captured["row_output_directory"]
    assert isinstance(rowdir, (str, os.PathLike))
    assert Path(rowdir) == tmp_path / "rows"
    if capinfo.text:
        assert "CLI arguments parsed" in capinfo.text


def test_final_round_rule():
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
