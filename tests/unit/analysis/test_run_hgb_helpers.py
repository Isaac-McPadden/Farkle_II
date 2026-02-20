import warnings
import builtins
import sys
import types
from typing import Any

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

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        kept, skipped = run_hgb._select_partial_dependence_features(features, tolerance=0.0)

    assert len(record) == 0
    assert kept == ["b"]
    assert set(skipped) == {"a", "c"}


def test_parse_strategy_features_handles_invalid_and_types():
    valid = str(ThresholdStrategy(100, 2, True, True))
    strategies = pd.Series([valid, "invalid", None])

    frame = run_hgb._parse_strategy_features(strategies)

    expected_columns = [name for name, _dtype in run_hgb.FEATURE_SPECS]
    assert list(frame.columns) == expected_columns
    assert set(frame.index) == {valid, "invalid"}
    assert frame.loc[valid, "dice_threshold"] == pytest.approx(2.0)
    assert pd.isna(frame.loc["invalid", "score_threshold"])
    assert pd.isna(frame.loc["invalid", "dice_threshold"])
    assert frame.loc["invalid", ["consider_score", "consider_dice", "smart_five", "smart_one", "favor_score", "require_both", "auto_hot_dice", "run_up_score"]].eq(0.0).all()
    assert str(frame.dtypes["score_threshold"]) == "float32"


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


def test_load_seed_targets_returns_empty_when_missing_or_nonmatching(tmp_path):
    seeds_dir = tmp_path / "pooled"
    seeds_dir.mkdir()
    pd.DataFrame({"strategy": ["A"], "mu": [1.0]}).to_parquet(
        seeds_dir / "ratings_k_weighted.parquet"
    )

    combined = run_hgb._load_seed_targets(seeds_dir)

    assert combined.empty
    assert list(combined.columns) == ["strategy", "mu", "seed"]


def test_select_partial_dependence_features_empty_input_is_noop():
    features = pd.DataFrame(columns=["a", "b"])

    kept, skipped = run_hgb._select_partial_dependence_features(features)

    assert kept == []
    assert skipped == ["a", "b"]


def test_parse_strategy_features_empty_series_returns_expected_schema():
    frame = run_hgb._parse_strategy_features(pd.Series(dtype=object))

    expected_columns = [name for name, _dtype in run_hgb.FEATURE_SPECS]
    assert list(frame.columns) == expected_columns
    assert frame.empty


def test_run_grouped_cv_logs_skip_reasons_and_success(monkeypatch):
    feature_cols = [name for name, _dtype in run_hgb.FEATURE_SPECS]
    subset = pd.DataFrame(
        {
            "strategy": ["s1", "s2"],
            **{name: [0.0, 1.0] for name in feature_cols},
        }
    )
    logged: list[tuple[str, dict[str, Any]]] = []

    def _log(message: str, *_args: object, **kwargs: object) -> None:
        extra = kwargs.get("extra", {})
        logged.append((message, extra if isinstance(extra, dict) else {}))

    monkeypatch.setattr(run_hgb.LOGGER, "info", _log)

    model_selection = types.ModuleType("model_selection")
    model_selection.__spec__ = None

    class PlaceholderSplitter:
        def __init__(self, n_splits: int):
            _ = n_splits

        def split(self, X, y, groups):
            _ = X, y, groups
            return iter(())

    model_selection.GroupKFold = PlaceholderSplitter  # type: ignore[attr-defined]  # intentional monkeypatch of dynamic module attribute in test
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", model_selection)
    monkeypatch.setattr(sys.modules["sklearn"], "__path__", [], raising=False)

    # no per-seed ratings
    run_hgb._run_grouped_cv(
        players=2,
        subset=subset,
        feature_cols=feature_cols,
        seed_targets=pd.DataFrame(columns=["strategy", "mu", "seed"]),
        random_state=0,
    )
    assert logged[-1][0] == "Grouped CV skipped: no per-seed ratings"

    # insufficient overlap
    run_hgb._run_grouped_cv(
        players=2,
        subset=subset,
        feature_cols=feature_cols,
        seed_targets=pd.DataFrame({"strategy": ["other"], "mu": [1.0], "seed": [1]}),
        random_state=0,
    )
    assert logged[-1][0] == "Grouped CV skipped: insufficient overlap"

    # <2 unique seeds
    run_hgb._run_grouped_cv(
        players=2,
        subset=subset,
        feature_cols=feature_cols,
        seed_targets=pd.DataFrame(
            {
                "strategy": ["s1", "s2"],
                "mu": [1.0, 2.0],
                "seed": [1, 1],
            }
        ),
        random_state=0,
    )
    assert logged[-1][0] == "Grouped CV skipped: <2 unique seeds"

    # insufficient splits
    original_min = builtins.min

    def fake_min(left: int, right: int):
        if {left, right} == {2, 5}:
            return 1
        return original_min(left, right)

    monkeypatch.setattr(builtins, "min", fake_min)
    run_hgb._run_grouped_cv(
        players=2,
        subset=subset,
        feature_cols=feature_cols,
        seed_targets=pd.DataFrame(
            {
                "strategy": ["s1", "s2"],
                "mu": [1.0, 2.0],
                "seed": [1, 2],
            }
        ),
        random_state=0,
    )
    assert logged[-1][0] == "Grouped CV skipped: insufficient splits"

    class EmptyFoldSplitter:
        def __init__(self, n_splits: int):
            _ = n_splits

        def split(self, X, y, groups):
            _ = X, y, groups
            yield np.array([], dtype=int), np.array([], dtype=int)

    monkeypatch.setattr(builtins, "min", original_min)
    monkeypatch.setattr(model_selection, "GroupKFold", EmptyFoldSplitter)
    run_hgb._run_grouped_cv(
        players=2,
        subset=subset,
        feature_cols=feature_cols,
        seed_targets=pd.DataFrame(
            {
                "strategy": ["s1", "s2"],
                "mu": [1.0, 2.0],
                "seed": [1, 2],
            }
        ),
        random_state=0,
    )
    assert logged[-1][0] == "Grouped CV skipped: empty folds"

    class DummyModel:
        def __init__(self, random_state: int):
            _ = random_state

        def fit(self, _X, _y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=np.float32)

    class Splitter:
        def __init__(self, n_splits: int):
            _ = n_splits

        def split(self, X, y, groups):
            _ = y, groups
            yield np.array([0], dtype=int), np.array([1], dtype=int)
            yield np.array([1], dtype=int), np.array([0], dtype=int)

    monkeypatch.setattr(model_selection, "GroupKFold", Splitter)
    monkeypatch.setattr(
        "sklearn.ensemble.HistGradientBoostingRegressor",
        lambda random_state=0: DummyModel(random_state),
    )
    run_hgb._run_grouped_cv(
        players=2,
        subset=subset,
        feature_cols=feature_cols,
        seed_targets=pd.DataFrame(
            {
                "strategy": ["s1", "s2", "s1", "s2"],
                "mu": [1.0, 2.0, 1.5, 2.5],
                "seed": [1, 1, 2, 2],
            }
        ),
        random_state=0,
    )
    assert logged[-1][0] == "Grouped CV metrics"
    assert logged[-1][1]["splits"] == 2
