from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
pytest.importorskip("pyarrow")

import farkle.simulation.runner as runner


def test_runner_passes_metric_flags(tmp_path, monkeypatch, tmp_artifacts_with_legacy):
    calls: dict[str, object] = {}

    def fake_run_tournament(**kwargs):  # noqa: ANN001 - signature mirrors target
        calls.update(kwargs)
        ckpt_path: Path = kwargs["checkpoint_path"]
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_artifacts_with_legacy["checkpoint"], ckpt_path)
        metrics_dst = ckpt_path.with_name("5p_metrics.parquet")
        shutil.copy2(tmp_artifacts_with_legacy["metrics"], metrics_dst)

    monkeypatch.setattr(runner.tournament_mod, "run_tournament", fake_run_tournament)

    cfg = runner.AppConfig(
        io=runner.IOConfig(results_dir=tmp_path / "out"),
        sim=runner.SimConfig(
            jobs=None,
            seed=11,
            n_games=4,
            n_players=2,
            collect_metrics=True,
            row_dir=Path("rows"),
        ),
    )

    total_games = runner.run_tournament(cfg)

    assert calls["collect_metrics"] is True
    assert calls["row_output_directory"] == tmp_path / "out" / "rows"
    assert calls["checkpoint_path"] == tmp_path / "out" / "checkpoint.pkl"
    expected_games = runner.TournamentConfig(n_players=2).games_per_shuffle
    assert total_games == expected_games

    csv_path = tmp_path / "out" / "win_counts.csv"
    assert csv_path.exists()
    assert "alpha" in csv_path.read_text()

    metrics_path = tmp_path / "out" / "5p_metrics.parquet"
    assert metrics_path.exists()
    try:
        metrics_df = pd.read_parquet(metrics_path)
    except Exception:
        metrics_df = pd.read_csv(metrics_path)
    assert metrics_df.iloc[0].strategy == "alpha"

