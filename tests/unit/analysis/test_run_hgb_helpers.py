import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from sklearn.ensemble import HistGradientBoostingRegressor

from farkle.analysis import run_hgb
from farkle.analysis.run_hgb import plot_partial_dependence
from farkle.simulation.strategies import ThresholdStrategy


def test_plot_partial_dependence(tmp_path):
    X = pd.DataFrame({"a": range(5), "b": range(5, 10)}).astype(float)
    y = pd.Series(range(5))
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(X, y)

    out_file = plot_partial_dependence(model, X, "a", tmp_path)
    assert out_file.exists()


def test_select_partial_dependence_features_filters_constants():
    features = pd.DataFrame(
        {"a": [1, 1, 1], "b": [0.0, 0.5, 1.0], "c": [np.nan, np.nan, np.nan]}
    )

    kept, skipped = run_hgb._select_partial_dependence_features(features, tolerance=0.0)

    assert kept == ["b"]
    assert set(skipped) == {"a", "c"}


def test_parse_strategy_features_handles_invalid_and_types():
    valid = str(ThresholdStrategy(100, 2, True, True))
    strategies = pd.Series([valid, "invalid", None])

    frame = run_hgb._parse_strategy_features(strategies)

    assert "score_threshold" in frame.columns
    assert set(frame.index) == {valid}
    assert frame.loc[valid, "dice_threshold"] == pytest.approx(2.0)


def test_metric_helpers_handle_edge_cases():
    assert run_hgb._mae(np.array([]), np.array([])) == 0.0
    assert run_hgb._r2(np.array([]), np.array([])) == 0.0

    y_true = np.array([1.0, 1.0, 1.0])
    assert run_hgb._r2(y_true, y_true) == 0.0

    y_pred = np.array([0.0, 1.0, 2.0])
    assert run_hgb._mae(y_true, y_pred) == pytest.approx(np.mean(np.abs(y_true - y_pred)))


def test_load_seed_targets_collects_seed_metadata(tmp_path):
    seeds_dir = tmp_path / "pooled"
    seeds_dir.mkdir()

    for seed in (1, 2):
        frame = pd.DataFrame({"strategy": ["A", "B"], "mu": [1.0 * seed, 2.0 * seed]})
        frame.to_parquet(seeds_dir / f"ratings_k_weighted_seed{seed}.parquet")

    combined = run_hgb._load_seed_targets(seeds_dir)

    assert set(combined["seed"]) == {1, 2}
    assert combined.shape[0] == 4
