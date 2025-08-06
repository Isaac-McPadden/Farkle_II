import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pipeline = pytest.importorskip("pipeline")


def _write_fixture(root: Path) -> None:
    """Create a minimal results block under *root* for two players."""
    block = root / "2_players"
    block.mkdir()
    np.save(block / "keepers_2.npy", np.array(["A", "B"]))
    df = pd.DataFrame(
        {
            "winner": ["P1", "P2"],
            "n_rounds": [5, 6],
            "winning_score": [1000, 1100],
            "P1_strategy": ["A", "A"],
            "P2_strategy": ["B", "B"],
            "P1_rank": [1, 2],
            "P2_rank": [2, 1],
        }
    )
    df.to_csv(block / "winners.csv", index=False)


def test_pipeline_all_creates_outputs(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        pipeline.main(["all", "--root", str(tmp_path)])
    finally:
        os.chdir(cwd)

    analysis = tmp_path / "analysis"
    assert (analysis / "data" / "game_rows.parquet").exists()
    assert not (analysis / "data" / "game_rows.raw.parquet").exists()
    assert (analysis / "metrics.parquet").exists()
    assert (analysis / "seat_advantage.csv").exists()

    # analytics artefacts
    assert (tmp_path / "ratings_pooled.pkl").exists()
    assert (tmp_path / "hgb_importance.json").exists()
    figs = tmp_path / "notebooks" / "figs"
    assert any(figs.glob("pd_*.png"))


def test_pipeline_ingest_only(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        pipeline.main(["ingest", "--root", str(tmp_path)])
    finally:
        os.chdir(cwd)

    analysis = tmp_path / "analysis"
    raw = analysis / "data" / "game_rows.raw.parquet"
    curated = analysis / "data" / "game_rows.parquet"
    assert raw.exists()
    assert not curated.exists()
    assert not (analysis / "metrics.parquet").exists()
    assert not (tmp_path / "hgb_importance.json").exists()


def test_pipeline_missing_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_fixture(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        def _boom(cfg):  # simulate analytics dependency failure
            raise RuntimeError("missing dependency")

        monkeypatch.setattr("farkle.analytics.run_all", _boom)
        with pytest.raises(RuntimeError):
            pipeline.main(["all", "--root", str(tmp_path)])
    finally:
        os.chdir(cwd)
