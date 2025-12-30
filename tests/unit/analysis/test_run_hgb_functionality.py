import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from farkle.analysis import run_hgb
from farkle.simulation.strategies import FavorDiceOrScore, ThresholdStrategy


def _strategy_literal(**kwargs) -> str:
    strat = ThresholdStrategy(**kwargs)
    return str(strat)


def _setup_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    strategies = [
        _strategy_literal(
            score_threshold=300,
            dice_threshold=2,
            smart_five=True,
            smart_one=True,
            consider_score=True,
            consider_dice=True,
            require_both=True,
            auto_hot_dice=True,
            run_up_score=False,
            favor_dice_or_score=FavorDiceOrScore.SCORE,
        ),
        _strategy_literal(
            score_threshold=450,
            dice_threshold=1,
            smart_five=True,
            smart_one=False,
            consider_score=True,
            consider_dice=False,
            require_both=False,
            auto_hot_dice=False,
            run_up_score=True,
            favor_dice_or_score=FavorDiceOrScore.SCORE,
        ),
    ]
    metrics = pd.DataFrame(
        {
            "strategy": strategies,
            "n_players": [2, 2],
            "games": [10, 10],
            "win_rate": [0.5, 0.5],
        }
    )
    metrics.to_parquet(data_dir / "metrics.parquet", index=False)
    ratings = pd.DataFrame(
        {
            "strategy": strategies,
            "mu": [0.0, 0.0],
            "sigma": [1.0, 1.0],
        }
    )
    ratings.to_parquet(data_dir / "ratings_pooled.parquet", index=False)
    return data_dir


def _perm_result(values: np.ndarray | list[float]) -> SimpleNamespace:
    arr = np.asarray(values, dtype=float)
    return SimpleNamespace(
        importances_mean=arr,
        importances_std=np.zeros_like(arr),
    )


def test_run_hgb_custom_output_path(tmp_path):
    data_dir = _setup_data(tmp_path)
    out_file = data_dir / "custom.json"
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(run_hgb, "_run_grouped_cv", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            run_hgb,
            "permutation_importance",
            lambda model, X, y, n_repeats=5, random_state=None: _perm_result(  # noqa: ARG005
                np.zeros(X.shape[1])
            ),
        )
        run_hgb.run_hgb(output_path=out_file, root=data_dir)
    finally:
        os.chdir(cwd)
        monkeypatch.undo()
    assert out_file.exists()
    parquet_path = (
        data_dir
        / f"{2}p"
        / run_hgb.IMPORTANCE_TEMPLATE.format(players=2)
    )
    assert parquet_path.exists()
    assert not (data_dir / "hgb_importance.json").exists()


def test_run_hgb_importance_length_check(tmp_path, monkeypatch):
    data_dir = _setup_data(tmp_path)

    def fake_perm_importance(_model, _X, _y, n_repeats=5, random_state=None):
        _ = n_repeats, random_state
        return _perm_result([0.1, 0.2])

    monkeypatch.setattr(run_hgb, "permutation_importance", fake_perm_importance)
    monkeypatch.setattr(run_hgb, "_run_grouped_cv", lambda *args, **kwargs: None)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValueError, match="Mismatch between number of features"):
            run_hgb.run_hgb(output_path=data_dir / "out.json", root=data_dir)
        assert not (data_dir / "out.tmp").exists()
    finally:
        os.chdir(cwd)


def test_win_rate_is_regression_target(tmp_path, monkeypatch):
    data_dir = _setup_data(tmp_path)
    metrics = pd.DataFrame(
        {
            "strategy": pd.read_parquet(data_dir / "metrics.parquet")["strategy"],
            "n_players": [2, 2],
            "games": [10, 10],
            "win_rate": [0.25, 0.75],
        }
    )
    metrics.to_parquet(data_dir / "metrics.parquet", index=False)

    captured = {}

    class DummyModel:
        def fit(self, _X, y):
            captured["y"] = list(y)
            return self

    monkeypatch.setattr(
        run_hgb,
        "HistGradientBoostingRegressor",
        lambda random_state=None: DummyModel(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        run_hgb,
        "permutation_importance",
        lambda model, X, y, n_repeats=5, random_state=None: _perm_result(  # noqa: ARG005
            np.zeros(X.shape[1])
        ),
    )
    monkeypatch.setattr(
        run_hgb,
        "plot_partial_dependence",
        lambda model, X, column, out_dir: Path(out_dir) / f"pd_{column}.png",  # noqa: ARG005
    )
    monkeypatch.setattr(run_hgb, "_run_grouped_cv", lambda *args, **kwargs: None)
    run_hgb.run_hgb(output_path=data_dir / "out.json", root=data_dir)
    assert captured["y"] == [0.25, 0.75]


def test_partial_dependence_warning_and_limit(tmp_path, monkeypatch, caplog):
    data_dir = _setup_data(tmp_path)
    monkeypatch.setattr(run_hgb, "MAX_PD_PLOTS", 5)
    num_cols = run_hgb.MAX_PD_PLOTS + 5
    strategies = pd.read_parquet(data_dir / "metrics.parquet")["strategy"].tolist()
    metrics = pd.DataFrame(
        {
            "strategy": strategies,
            "n_players": [2, 2],
            "games": [10, 10],
            "win_rate": [0.5, 0.6],
            **{f"extra_{i}": [i, i + 1] for i in range(num_cols)},
        }
    )
    metrics.to_parquet(data_dir / "metrics.parquet", index=False)

    class DummyModel:
        def fit(self, _X, _y):
            return self

    monkeypatch.setattr(
        run_hgb,
        "HistGradientBoostingRegressor",
        lambda random_state=None: DummyModel(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        run_hgb,
        "permutation_importance",
        lambda model, X, y, n_repeats=5, random_state=None: _perm_result(  # noqa: ARG005
            np.zeros(X.shape[1])
        ),
    )
    monkeypatch.setattr(run_hgb, "_run_grouped_cv", lambda *args, **kwargs: None)
    plotted: list[str] = []

    def fake_plot(model, X, column, out_dir):  # noqa: ARG001
        plotted.append(column)
        p = Path(out_dir) / f"pd_{column}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy")
        return p

    monkeypatch.setattr(run_hgb, "plot_partial_dependence", fake_plot)

    warnings: list[tuple[str, dict]] = []

    def _record_warning(message, *args, **kwargs):  # noqa: ANN001, ARG002
        warnings.append((message, kwargs))

    monkeypatch.setattr(run_hgb.LOGGER, "warning", _record_warning)
    run_hgb.run_hgb(output_path=data_dir / "out.json", root=data_dir)

    assert warnings, "expected Too many features warning"
    assert "Too many features for partial dependence" in warnings[0][0]
    expected_plots = [
        "score_threshold",
        "dice_threshold",
        "consider_dice",
        "smart_one",
        "require_both",
    ][: run_hgb.MAX_PD_PLOTS]
    assert plotted == expected_plots


def test_partial_dependence_skips_constant_features(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    strategies = [
        _strategy_literal(
            score_threshold=300,
            dice_threshold=2,
            smart_five=True,
            smart_one=True,
            consider_score=True,
            consider_dice=True,
            require_both=True,
            auto_hot_dice=True,
            run_up_score=False,
            favor_dice_or_score=FavorDiceOrScore.SCORE,
        ),
        _strategy_literal(
            score_threshold=450,
            dice_threshold=1,
            smart_five=True,
            smart_one=False,
            consider_score=True,
            consider_dice=True,
            require_both=False,
            auto_hot_dice=False,
            run_up_score=True,
            favor_dice_or_score=FavorDiceOrScore.SCORE,
        ),
    ]
    metrics = pd.DataFrame(
        {
            "strategy": strategies,
            "n_players": [2, 2],
            "games": [10, 10],
            "win_rate": [0.5, 0.5],
        }
    )
    metrics.to_parquet(data_dir / "metrics.parquet", index=False)
    ratings = pd.DataFrame(
        {
            "strategy": strategies,
            "mu": [0.0, 0.0],
            "sigma": [1.0, 1.0],
        }
    )
    ratings.to_parquet(data_dir / "ratings_pooled.parquet", index=False)

    class DummyModel:
        def fit(self, _X, _y):
            return self

    monkeypatch.setattr(
        run_hgb,
        "HistGradientBoostingRegressor",
        lambda random_state=None: DummyModel(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        run_hgb,
        "permutation_importance",
        lambda model, X, y, n_repeats=5, random_state=None: _perm_result(  # noqa: ARG005
            np.zeros(X.shape[1])
        ),
    )
    monkeypatch.setattr(run_hgb, "_run_grouped_cv", lambda *args, **kwargs: None)
    plotted: list[str] = []

    def fake_plot(model, X, column, out_dir):  # noqa: ARG001
        plotted.append(column)
        p = Path(out_dir) / f"pd_{column}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy")
        return p

    monkeypatch.setattr(run_hgb, "plot_partial_dependence", fake_plot)

    run_hgb.run_hgb(output_path=data_dir / "out.json", root=data_dir)

    assert "consider_score" not in plotted
    assert "consider_dice" not in plotted
    skipped_logs = [r for r in caplog.records if "Skipping near-constant features" in r.message]
    assert skipped_logs, "expected skip log for near-constant features"


def test_run_hgb_default_output(tmp_path, monkeypatch):
    data_dir = _setup_data(tmp_path)

    class DummyModel:
        def fit(self, _X, _y):
            return self

    monkeypatch.setattr(
        run_hgb,
        "HistGradientBoostingRegressor",
        lambda random_state=None: DummyModel(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        run_hgb,
        "permutation_importance",
        lambda model, X, y, n_repeats=5, random_state=None: _perm_result(  # noqa: ARG005
            np.zeros(X.shape[1])
        ),
    )
    monkeypatch.setattr(
        run_hgb,
        "plot_partial_dependence",
        lambda model, X, column, out_dir: Path(out_dir) / f"pd_{column}.png",  # noqa: ARG005
    )
    monkeypatch.setattr(run_hgb, "_run_grouped_cv", lambda *args, **kwargs: None)

    run_hgb.run_hgb(root=data_dir)
    assert (data_dir / "pooled" / "hgb_importance.json").exists()
    parquet_path = (
        data_dir
        / f"{2}p"
        / run_hgb.IMPORTANCE_TEMPLATE.format(players=2)
    )
    assert parquet_path.exists()
