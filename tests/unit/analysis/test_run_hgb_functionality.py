import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    ratings.to_parquet(data_dir / "ratings_k_weighted.parquet", index=False)
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
    monkeypatch = pytest.MonkeyPatch()
    os.chdir(tmp_path)
    try:
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

    def fake_plot(model: Any, X: Any, column: str, out_dir: Path | str) -> Path:
        plotted.append(column)
        p = Path(out_dir) / f"pd_{column}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy")
        return p

    monkeypatch.setattr(run_hgb, "plot_partial_dependence", fake_plot)

    warnings: list[tuple[str, dict]] = []

    def _record_warning(message: str, *_args: object, **kwargs: object) -> None:
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
    ratings.to_parquet(data_dir / "ratings_k_weighted.parquet", index=False)

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

    def fake_plot(model: Any, X: Any, column: str, out_dir: Path | str) -> Path:
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


@pytest.mark.parametrize("seed", [0, 17])
def test_run_hgb_passes_seed_to_model_and_permutation(tmp_path, monkeypatch, seed):
    data_dir = _setup_data(tmp_path)
    model_random_states: list[int | None] = []
    perm_calls: list[tuple[int, int | None]] = []

    class DummyModel:
        def fit(self, _X, _y):
            return self

    def fake_model(*, random_state=None):
        model_random_states.append(random_state)
        return DummyModel()

    def fake_perm(model, X, y, n_repeats=5, random_state=None):
        _ = model, y
        perm_calls.append((n_repeats, random_state))
        return _perm_result(np.zeros(X.shape[1]))

    monkeypatch.setattr(run_hgb, "HistGradientBoostingRegressor", fake_model)
    monkeypatch.setattr(run_hgb, "permutation_importance", fake_perm)
    monkeypatch.setattr(run_hgb, "_run_grouped_cv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_hgb,
        "plot_partial_dependence",
        lambda model, X, column, out_dir: Path(out_dir) / f"pd_{column}.png",  # noqa: ARG005
    )

    run_hgb.run_hgb(seed=seed, root=data_dir)

    assert model_random_states == [seed]
    assert perm_calls == [(10, seed)]


def test_run_hgb_reads_manifest_and_writes_deterministic_output(tmp_path, monkeypatch):
    data_dir = _setup_data(tmp_path)
    manifest_path = data_dir / "strategy_manifest.parquet"
    manifest = pd.DataFrame({"strategy": ["dummy"], "source": ["manifest"]})
    manifest.to_parquet(manifest_path, index=False)

    captured_manifest = {}
    original_parse = run_hgb._parse_strategy_features

    def fake_parse(strategies, *, manifest=None):
        captured_manifest["manifest"] = manifest
        return original_parse(strategies, manifest=manifest)

    class DummyModel:
        def fit(self, _X, _y):
            return self

    monkeypatch.setattr(run_hgb, "_parse_strategy_features", fake_parse)
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
    monkeypatch.setattr(
        run_hgb,
        "plot_partial_dependence",
        lambda model, X, column, out_dir: Path(out_dir) / f"pd_{column}.png",  # noqa: ARG005
    )

    run_hgb.run_hgb(root=data_dir, manifest_path=manifest_path)

    assert captured_manifest["manifest"].equals(manifest)
    expected_output = data_dir / "pooled" / "hgb_importance.json"
    assert expected_output.exists()
    payload = json.loads(expected_output.read_text())
    assert "2p" in payload


def test_run_hgb_missing_win_rate_column_raises(tmp_path):
    data_dir = _setup_data(tmp_path)
    metrics = pd.read_parquet(data_dir / "metrics.parquet").drop(columns=["win_rate"])
    metrics.to_parquet(data_dir / "metrics.parquet", index=False)

    with pytest.raises(ValueError, match="missing win_rate"):
        run_hgb.run_hgb(root=data_dir)


def test_run_hgb_skips_when_no_features_or_join_rows(tmp_path, monkeypatch):
    data_dir = _setup_data(tmp_path)

    monkeypatch.setattr(
        run_hgb,
        "_parse_strategy_features",
        lambda strategies, *, manifest=None: pd.DataFrame(index=pd.Index([], name="strategy")),  # noqa: ARG005
    )
    run_hgb.run_hgb(root=data_dir)
    assert not (data_dir / "pooled" / "hgb_importance.json").exists()

    def wrong_index(strategies, *, manifest=None):
        _ = strategies, manifest
        cols = [name for name, _dtype in run_hgb.FEATURE_SPECS]
        return pd.DataFrame([dict.fromkeys(cols, 1.0)], index=["other-strategy"])

    monkeypatch.setattr(run_hgb, "_parse_strategy_features", wrong_index)
    run_hgb.run_hgb(root=data_dir)
    assert not (data_dir / "pooled" / "hgb_importance.json").exists()


def test_run_hgb_propagates_model_fit_failure(tmp_path, monkeypatch):
    data_dir = _setup_data(tmp_path)

    class BoomModel:
        def fit(self, _X, _y):
            raise RuntimeError("fit failed")

    monkeypatch.setattr(
        run_hgb,
        "HistGradientBoostingRegressor",
        lambda random_state=None: BoomModel(),  # noqa: ARG005
    )

    with pytest.raises(RuntimeError, match="fit failed"):
        run_hgb.run_hgb(output_path=data_dir / "out.json", root=data_dir)
    assert not (data_dir / "out.tmp").exists()


def test_run_hgb_schema_mismatch_feature_columns_raises(tmp_path, monkeypatch):
    data_dir = _setup_data(tmp_path)

    def bad_features(_strategies, *, manifest=None):
        _ = manifest
        return pd.DataFrame(
            {
                "score_threshold": [300.0, 450.0],
                "dice_threshold": [2.0, 1.0],
            },
            index=pd.read_parquet(data_dir / "metrics.parquet")["strategy"],
        )

    monkeypatch.setattr(run_hgb, "_parse_strategy_features", bad_features)

    with pytest.raises(KeyError):
        run_hgb.run_hgb(root=data_dir)



def test_run_hgb_handles_empty_and_insufficient_player_buckets(tmp_path, monkeypatch):
    data_dir = _setup_data(tmp_path)
    metrics = pd.read_parquet(data_dir / "metrics.parquet")
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame(
                {
                    "strategy": [metrics.loc[0, "strategy"], None],
                    "n_players": [3, 4],
                    "games": [10, 10],
                    "win_rate": [0.7, 0.1],
                }
            ),
        ],
        ignore_index=True,
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
            np.arange(X.shape[1], dtype=float)
        ),
    )
    monkeypatch.setattr(run_hgb, "_run_grouped_cv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_hgb,
        "plot_partial_dependence",
        lambda model, X, column, out_dir: Path(out_dir) / f"pd_{column}.png",  # noqa: ARG005
    )

    run_hgb.run_hgb(root=data_dir)

    three_p = pd.read_parquet(data_dir / "3p" / run_hgb.IMPORTANCE_TEMPLATE.format(players=3))
    assert three_p.empty
    assert list(three_p.columns) == ["feature", "importance_mean", "importance_std", "players"]

    four_p = pd.read_parquet(data_dir / "4p" / run_hgb.IMPORTANCE_TEMPLATE.format(players=4))
    assert four_p.empty


def test_run_hgb_players_only_schema_and_manifest_fallback_and_pooled_artifacts(tmp_path, monkeypatch):
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
            smart_five=False,
            smart_one=False,
            consider_score=False,
            consider_dice=True,
            require_both=False,
            auto_hot_dice=False,
            run_up_score=True,
            favor_dice_or_score=FavorDiceOrScore.DICE,
        ),
    ]
    metrics = pd.DataFrame(
        {
            "strategy": [strategies[0], strategies[1], strategies[0], strategies[1]],
            "players": [2, 2, 4, 4],
            "games": [10, 10, 10, 10],
            "win_rate": [0.2, 0.8, 0.6, 0.4],
        }
    )
    metrics.to_parquet(data_dir / "metrics.parquet", index=False)
    pd.DataFrame({"strategy": strategies, "mu": [0.0, 0.1]}).to_parquet(
        data_dir / "ratings_k_weighted.parquet", index=False
    )

    captured_manifest = {}
    original_parse = run_hgb._parse_strategy_features

    def fake_parse(strategies_in, *, manifest=None):
        captured_manifest["manifest"] = manifest
        return original_parse(strategies_in, manifest=manifest)

    class DummyModel:
        def fit(self, _X, _y):
            return self

    monkeypatch.setattr(run_hgb, "_parse_strategy_features", fake_parse)
    monkeypatch.setattr(
        run_hgb,
        "HistGradientBoostingRegressor",
        lambda random_state=None: DummyModel(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        run_hgb,
        "permutation_importance",
        lambda model, X, y, n_repeats=10, random_state=None: _perm_result(  # noqa: ARG005
            np.linspace(0.0, 1.0, X.shape[1])
        ),
    )
    monkeypatch.setattr(run_hgb, "_run_grouped_cv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_hgb,
        "plot_partial_dependence",
        lambda model, X, column, out_dir: Path(out_dir) / f"pd_{column}.png",  # noqa: ARG005
    )

    run_hgb.run_hgb(root=data_dir, manifest_path=data_dir / "missing_manifest.parquet")

    assert captured_manifest["manifest"] is None

    pooled_long = pd.read_parquet(data_dir / "pooled" / run_hgb.LONG_IMPORTANCE_NAME)
    pooled_overall = pd.read_parquet(data_dir / "pooled" / run_hgb.OVERALL_IMPORTANCE_NAME)
    assert set(pooled_long["players"]) == {2, 4}
    assert set(pooled_overall["players"]) == {"overall"}

    payload = json.loads((data_dir / "pooled" / "hgb_importance.json").read_text())
    assert list(payload) == ["2p", "4p", "overall"]
    assert set(payload["overall"]) == {name for name, _dtype in run_hgb.FEATURE_SPECS}
