from pandas import DataFrame

from farkle.simulation import (
    _play_game,
    experiment_size,
    generate_strategy_grid,
    simulate_many_games,
    simulate_one_game,
)
from farkle.strategies import ThresholdStrategy


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
    assert stats_dict["winner"] == gm.winner
    # simulate_many_games should aggregate identical winners when n_games=1
    df = simulate_many_games(n_games=1, strategies=[s1, s2],
                             target_score=1_000, seed=999, n_jobs=1)
    assert len(df) == 1
    assert df.iloc[0]["winner"] in {"P1", "P2"}
    
def test_custom_grid_size():
    # Only auto_hot_dice == True → half the default grid (1 275)
    strategies, meta = generate_strategy_grid(auto_hot_opts=[True])
    assert len(strategies) == 4080
    assert len(meta) == 4080