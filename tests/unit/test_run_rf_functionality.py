import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farkle import run_rf, run_trueskill


def _setup_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    metrics = pd.DataFrame({"strategy": ["A"], "feat": [1]})
    metrics.to_parquet(data_dir / "metrics.parquet")
    ratings = {"A": run_trueskill.RatingStats(0.0, 1.0)}
    with open(data_dir / "ratings_pooled.pkl", "wb") as fh:
        pickle.dump(ratings, fh)
    return data_dir


def test_run_rf_custom_output_path(tmp_path):
    data_dir = _setup_data(tmp_path)
    out_file = data_dir / "custom.json"
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_rf.run_rf(output_path=out_file)
    finally:
        os.chdir(cwd)
    assert out_file.exists()
    assert not (data_dir / "rf_importance.json").exists()


def test_run_rf_importance_length_check(tmp_path, monkeypatch):
    data_dir = _setup_data(tmp_path)

    def fake_perm_importance(_model, _X, _y, n_repeats=5, random_state=None):
        _ = n_repeats, random_state
        return {"importances_mean": np.array([0.1, 0.2])}

    monkeypatch.setattr(run_rf, "permutation_importance", fake_perm_importance)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValueError, match="Mismatch between number of features"):
            run_rf.run_rf(output_path=data_dir / "out.json")
    finally:
        os.chdir(cwd)
