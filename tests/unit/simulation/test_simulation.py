import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from farkle.game.engine import FarkleGame, GameMetrics, GameStats, PlayerStats
from farkle.simulation.simulation import (
    _play_game,
    experiment_size,
    generate_strategy_grid,
    simulate_many_games,
    simulate_many_games_from_seeds,
    simulate_one_game,
)
from farkle.simulation.strategies import ThresholdStrategy


def test_default_grid_size():
    strategies, meta = generate_strategy_grid()
    assert len(strategies) == 8160
    assert len(meta) == 8160
    for object in strategies:
        assert isinstance(object, ThresholdStrategy)
    assert isinstance(meta, DataFrame)


def test_default_size():
    assert experiment_size() == 8160


def test_size_and_grid_match():
    strats, _ = generate_strategy_grid()
    size = experiment_size()
    assert size == len(strats)


def test_play_helpers_consistency():
    # one always-stop, one always-roll (score_threshold huge)
    s1 = ThresholdStrategy(score_threshold=0, dice_threshold=6)
    s2 = ThresholdStrategy(score_threshold=10_000, dice_threshold=6)

    stats_dict = _play_game(seed=123, strategies=[s1, s2], target_score=1_000)
    gm = simulate_one_game(strategies=[s1, s2], target_score=1_000, seed=123)
    winner = max(gm.players, key=lambda n: gm.players[n].score)
    assert stats_dict["winner"] == winner
    # simulate_many_games should aggregate identical winners when n_games=1
    df = simulate_many_games(n_games=1, strategies=[s1, s2], target_score=1_000, seed=999, n_jobs=1)
    assert len(df) == 1
    assert df.iloc[0]["winner"] in {"P1", "P2"}


def test_custom_grid_size():
    # Only auto_hot_dice == True → half the default grid (1 275)
    strategies, meta = generate_strategy_grid(auto_hot_dice_opts=[True])
    assert len(strategies) == 4080
    assert len(meta) == 4080


def test_limited_consider_opts_grid_and_size():
    """Grid size and experiment_size with restricted consider_* options."""
    opts = {"consider_score_opts": [False], "consider_dice_opts": [False]}
    strategies, _ = generate_strategy_grid(**opts)
    # hand-computed: base combinations (1020) × 2 ps values
    assert len(strategies) == 2040
    assert experiment_size(**opts) == 2040 # type: ignore


def test_consider_true_true_options():
    opts = {"consider_score_opts": [True], "consider_dice_opts": [True]}
    strategies, _ = generate_strategy_grid(**opts)
    assert len(strategies) == 4080  # 1020 base × 4 variants
    assert experiment_size(**opts) == 4080 # type: ignore


def test_parallel_simulation():
    """simulate_many_games uses multiprocessing when n_jobs > 1."""
    s1 = ThresholdStrategy(score_threshold=0, dice_threshold=6)
    s2 = ThresholdStrategy(score_threshold=0, dice_threshold=6)
    df = simulate_many_games(
        n_games=2,
        strategies=[s1, s2],
        target_score=200,
        seed=42,
        n_jobs=2,
    )
    assert len(df) == 2
    assert set(df["winner"]) <= {"P1", "P2"}


def test_play_game_checks_single_winner(monkeypatch):
    """_play_game should error if multiple players have rank==1."""

    def fake_play(_self):
        ps = PlayerStats(
            score=100,
            farkles=0,
            rolls=1,
            highest_turn=100,
            strategy="S",
            rank=1,
            loss_margin=0,
        )
        players = {"P1": ps, "P2": ps}
        gm = GameMetrics(
            players,
            GameStats(
                n_players=2,
                table_seed=0,
                n_rounds=1,
                total_rolls=0,
                total_farkles=0,
                margin=0,
            ),
        )
        return gm

    monkeypatch.setattr(FarkleGame, "play", fake_play, raising=True)

    strat = [ThresholdStrategy(score_threshold=0, dice_threshold=6)] * 2
    with pytest.raises(ValueError):
        _play_game(seed=123, strategies=strat)


def test_experiment_size_subset_options():
    cs_opts = [True]
    cd_opts = [True, False]
    strats, _ = generate_strategy_grid(consider_score_opts=cs_opts, consider_dice_opts=cd_opts)
    size = experiment_size(consider_score_opts=cs_opts, consider_dice_opts=cd_opts)
    assert size == len(strats)

    
def test_simulate_many_games_from_seeds_matches():
    strats = [ThresholdStrategy(score_threshold=100, dice_threshold=0)]
    rng_seed = 42
    n_games = 5
    rng = np.random.default_rng(rng_seed)
    seeds = rng.integers(0, 2**32 - 1, size=n_games).tolist()
    df1 = simulate_many_games_from_seeds(seeds=seeds, strategies=strats, n_jobs=1)
    df2 = simulate_many_games(n_games=n_games, strategies=strats, seed=rng_seed, n_jobs=1)
    pd.testing.assert_frame_equal(df1, df2)


def test_simulate_many_games_deterministic_counts():
    strategies = [
        ThresholdStrategy(score_threshold=0, dice_threshold=6),
        ThresholdStrategy(score_threshold=500, dice_threshold=3),
        ThresholdStrategy(score_threshold=1000, dice_threshold=2),
    ]
    df = simulate_many_games(
        n_games=10,
        strategies=strategies,
        target_score=5000,
        seed=123,
        n_jobs=1,
    )
    counts = df["winner"].value_counts().to_dict()
    assert counts == {"P3": 5, "P1": 3, "P2": 2}
