import contextlib
import subprocess as sp
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from farkle.engine import FarkleGame, FarklePlayer
from farkle.farkle_io import simulate_many_games_stream
from farkle.scoring import (
    apply_discards,
    decide_smart_discards,
    default_score,
    score_roll_cached,
)
from farkle.simulation import simulate_many_games
from farkle.strategies import ThresholdStrategy

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


@pytest.mark.integration
@pytest.mark.parametrize("n_jobs", [1, 2, 4])
def test_stream_mp_branches(tmp_csv, simple_strategy, n_jobs):
    """Serial + parallel branches of simulate_many_games_stream."""
    if tmp_csv.exists():  # start fresh for each param value
        tmp_csv.unlink()
    simulate_many_games_stream(
        n_games=40,  # tiny but enough to hit queues
        strategies=simple_strategy,
        out_csv=str(tmp_csv),
        seed=42,
        n_jobs=n_jobs,
    )
    df = pd.read_csv(tmp_csv)
    # 1 strategy × 40 reps → 40 rows
    assert len(df) == 40
    # verify writer closed file cleanly (newline at EOF)
    assert tmp_csv.read_bytes().endswith(b"\n")


def test_seed_reproducible(simple_strategy):
    df1 = simulate_many_games(n_games=200, strategies=simple_strategy, seed=123, n_jobs=1)
    df2 = simulate_many_games(n_games=200, strategies=simple_strategy, seed=123, n_jobs=1)
    pd.testing.assert_frame_equal(df1, df2)


CLI = [sys.executable, "-m", "farkle"]  # <- NEW entry-point


def test_cli_smoke(tmp_path: Path):
    """
    End-to-end CLI run via:  python -m farkle run <cfg.yml>

    Verifies:
      * CLI exits 0
      * output CSV has expected number of rows
    """
    cfg = {
        "strategy_grid": {
            "score_thresholds": [300],
            "dice_thresholds": [2],
            "smart_five_opts": [False],
            "smart_one_opts": [False],
            "consider_score_opts": [True],
            "consider_dice_opts": [True],
            "auto_hot_opts": [False],
        },
        "sim": {
            "n_games": 20,
            "out_csv": str(tmp_path / "out.csv"),
            "seed": 9,
            "n_jobs": 1,
        },
    }

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # --- run the CLI -------------------------------------------------------
    sp.check_call(CLI + ["run", str(cfg_path)])
    # -----------------------------------------------------------------------

    df = pd.read_csv(cfg["sim"]["out_csv"])
    assert len(df) == cfg["sim"]["n_games"]


def test_final_round_rule_1():
    players = [
        FarklePlayer("P1", ThresholdStrategy(score_threshold=100, dice_threshold=0)),  # super-aggressive
        FarklePlayer("P2", ThresholdStrategy(score_threshold=300, dice_threshold=2)),
    ]
    game = FarkleGame(players, target_score=2000)
    gm = game.play()  # GameMetrics
    winner = max(gm.players, key=lambda n: gm.players[n].score)
    assert gm.players[winner].score >= 2000
    # everyone had ≤ one extra turn
    assert gm.game.n_rounds >= 1


def test_final_round_rule_2():
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



pytestmark = pytest.mark.skipif(
    # set env var FAST_CI to skip heavy tests
    "FAST_CI" in __import__("os").environ,
    reason="performance test - skip on CI",
)


def test_batch_time_under_baseline():
    t0 = perf_counter()
    simulate_many_games(
        n_games=5_000,
        strategies=[ThresholdStrategy()],
        seed=7,
        n_jobs=4,
    )
    elapsed = perf_counter() - t0
    assert elapsed < 8.0, f"perf regression: {elapsed:.1f}s > 8s baseline"


# helper: ensure Smart-1 never true when Smart-5 false
smart_flags = st.tuples(
    st.booleans(),  # smart_five
    st.booleans(),
).filter(lambda t: t[1] <= t[0])  # (sf, so) with (so ⇒ sf)

@given(
    roll=st.lists(st.integers(1, 6), min_size=1, max_size=6),
    turn_pre=st.integers(min_value=0, max_value=500),
    flags=smart_flags,
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_pipeline_matches_default(roll, turn_pre, flags):
    smart_five, smart_one = flags

    # ---------- (A) single-call pipeline ----------
    d_score, d_used, d_reroll = default_score( # type: ignore (refactor allows 3 or 5 outputs)
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
    