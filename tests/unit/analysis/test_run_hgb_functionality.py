import logging
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from farkle.analysis import run_hgb, run_trueskill


def _setup_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    metrics = pd.DataFrame({"strategy": ["A", "B"], "feat": [1, 2]})
    metrics.to_parquet(data_dir / "metrics.parquet")
    ratings = {
        "A": run_trueskill.RatingStats(0.0, 1.0),
        "B": run_trueskill.RatingStats(0.0, 1.0),
    }
    with open(data_dir / "ratings_pooled.pkl", "wb") as fh:
        pickle.dump(ratings, fh)
    return data_dir


def test_run_hgb_custom_output_path(tmp_path):
    data_dir = _setup_data(tmp_path)
    out_file = data_dir / "custom.json"
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_hgb.run_hgb(output_path=out_file, root=data_dir)
    finally:
        os.chdir(cwd)
    assert out_file.exists()
    assert not (data_dir / "hgb_importance.json").exists()


def test_run_hgb_importance_length_check(tmp_path, monkeypatch):
    data_dir = _setup_data(tmp_path)

    def fake_perm_importance(_model, _X, _y, n_repeats=5, random_state=None):
        _ = n_repeats, random_state
        return {"importances_mean": np.array([0.1, 0.2])}

    monkeypatch.setattr(run_hgb, "permutation_importance", fake_perm_importance)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValueError, match="Mismatch between number of features"):
            run_hgb.run_hgb(output_path=data_dir / "out.json", root=data_dir)
        assert not (data_dir / "out.tmp").exists()
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize(
    "rating_val,mu",
    [
        (SimpleNamespace(mu=1.0), 1.0),
        ({"mu": 2.0}, 2.0),
        ([3.0, 0.5], 3.0),
    ],
)
def test_get_mu_various_types(tmp_path, monkeypatch, rating_val, mu):
    data_dir = _setup_data(tmp_path)
    ratings = {"A": rating_val, "B": rating_val}
    with open(data_dir / "ratings_pooled.pkl", "wb") as fh:
        pickle.dump(ratings, fh)

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
        lambda model, X, y, n_repeats=5, random_state=None: {  # noqa: ARG005
            "importances_mean": np.zeros(X.shape[1])
        },
    )
    monkeypatch.setattr(
        run_hgb,
        "plot_partial_dependence",
        lambda model, X, column, out_dir: Path(out_dir) / f"pd_{column}.png",  # noqa: ARG005
    )
    run_hgb.run_hgb(output_path=data_dir / "out.json", root=data_dir)
    assert captured["y"] == [mu, mu]


def test_get_mu_unsupported_type(tmp_path):
    data_dir = _setup_data(tmp_path)
    ratings = {"A": object(), "B": object()}
    with open(data_dir / "ratings_pooled.pkl", "wb") as fh:
        pickle.dump(ratings, fh)

    with pytest.raises(TypeError):
        run_hgb.run_hgb(output_path=data_dir / "out.json", root=data_dir)


def test_partial_dependence_warning_and_limit(tmp_path, monkeypatch, caplog):
    data_dir = _setup_data(tmp_path)
    num_cols = run_hgb.MAX_PD_PLOTS + 5
    metrics = pd.DataFrame(
        {
            "strategy": ["A", "B"],
            **{f"feat{i}": [i, i + 1] for i in range(num_cols)},
        }
    )
    metrics.to_parquet(data_dir / "metrics.parquet")

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
        lambda model, X, y, n_repeats=5, random_state=None: {  # noqa: ARG005
            "importances_mean": np.zeros(X.shape[1])
        },
    )
    plotted = []

    def fake_plot(model, X, column, out_dir):  # noqa: ARG001
        plotted.append(column)
        p = Path(out_dir) / f"pd_{column}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy")
        return p

    monkeypatch.setattr(run_hgb, "plot_partial_dependence", fake_plot)

    with caplog.at_level(logging.WARNING, logger=run_hgb.logger.name):
        run_hgb.run_hgb(output_path=data_dir / "out.json", root=data_dir)

    assert any("More than" in m and "only plotting the first" in m for m in caplog.messages)
    assert plotted == [f"feat{i}" for i in range(run_hgb.MAX_PD_PLOTS)]


def test_main_default_output(tmp_path, monkeypatch):
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
        lambda model, X, y, n_repeats=5, random_state=None: {  # noqa: ARG005
            "importances_mean": np.zeros(X.shape[1])
        },
    )
    monkeypatch.setattr(
        run_hgb,
        "plot_partial_dependence",
        lambda model, X, column, out_dir: Path(out_dir) / f"pd_{column}.png",  # noqa: ARG005
    )

    run_hgb.main(["--root", str(data_dir)])
    assert (data_dir / "hgb_importance.json").exists()
