from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import farkle.simulation.runner as runner


def test_runner_passes_metric_flags(tmp_path, monkeypatch):
    calls: dict[str, object] = {}

    def fake_run_tournament(**kwargs):  # noqa: ANN001 - signature mirrors target
        calls.update(kwargs)
        ckpt_path: Path = kwargs["checkpoint_path"]
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"win_totals": Counter({"alpha": 3})}
        ckpt_path.write_bytes(pickle.dumps(payload))

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
