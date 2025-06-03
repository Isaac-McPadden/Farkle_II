from farkle.simulation import generate_strategy_grid

def test_default_grid_size():
    strategies, meta = generate_strategy_grid()
    assert len(strategies) == 1_275
    assert len(meta) == 1_275