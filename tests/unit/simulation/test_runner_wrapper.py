from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from farkle.config import AppConfig, IOConfig, SimConfig
import farkle.simulation.runner as runner


def _patch_tournament(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    *,
    after_checkpoint: Callable[[Path, dict[str, object]], None] | None = None,
) -> dict[str, object]:
    calls: dict[str, object] = {}

    def fake_run_tournament(**kwargs: object) -> None:  # noqa: ANN001 - mirrors target
        calls.update(kwargs)
        ckpt_path: Path = kwargs["checkpoint_path"]  # type: ignore[index]
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_bytes(pickle.dumps(payload))
        if after_checkpoint is not None:
            after_checkpoint(ckpt_path, kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runner.tournament_mod, "run_tournament", fake_run_tournament)
    return calls


def test_runner_passes_metric_flags(tmp_path, monkeypatch, tmp_artifacts_with_legacy):
    payload = pickle.loads(tmp_artifacts_with_legacy["checkpoint"].read_bytes())

    calls = _patch_tournament(monkeypatch, payload)

    cfg = AppConfig(
        io=IOConfig(results_dir=tmp_path / "out", append_seed=False),
        sim=SimConfig(
            n_players_list=[2],
            seed=11,
            num_shuffles=1,
            expanded_metrics=True,
            row_dir=Path("rows"),
            recompute_num_shuffles=False,
        ),
    )

    total_games = runner.run_tournament(cfg)

    assert calls["collect_metrics"] is True
    assert calls["row_output_directory"] == tmp_path / "out" / "2_players" / "rows"
    check_absolute = calls["row_output_directory"]
    assert isinstance(check_absolute, (str, os.PathLike))
    assert Path(check_absolute).is_absolute()
    assert calls["checkpoint_path"] == tmp_path / "out" / "2_players" / "2p_checkpoint.pkl"
    expected_games = calls["config"].games_per_shuffle
    assert total_games == expected_games

    summary_path = tmp_path / "out" / "2_players" / "2p_checkpoint.parquet"
    assert summary_path.exists()
    summary_df = pd.read_parquet(summary_path)
    assert "alpha" in summary_df["strategy"].tolist()

    metrics_path = tmp_path / "out" / "2_players" / "2p_metrics.parquet"
    assert metrics_path.exists()
    try:
        metrics_df = pd.read_parquet(metrics_path)
    except Exception:
        metrics_df = pd.read_csv(metrics_path)
    assert metrics_df.iloc[0].strategy == "alpha"


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        pytest.param({"win_totals": ["invalid"]}, list, id="list_payload"),
        pytest.param({"win_totals": 3.14}, float, id="wrong_value_type"),
    ],
)
def test_runner_rejects_malformed_checkpoints(tmp_path, monkeypatch, payload, expected_type):
    _patch_tournament(monkeypatch, payload)

    cfg = AppConfig(
        io=IOConfig(results_dir=tmp_path / "out", append_seed=False),
        sim=SimConfig(n_players_list=[2], recompute_num_shuffles=False, num_shuffles=1),
    )

    with pytest.raises(TypeError) as excinfo:
        runner.run_tournament(cfg)

    assert str(excinfo.value) == f"Unexpected win_totals payload type: {expected_type!r}"


def test_runner_writes_normalized_counters(tmp_path, monkeypatch):
    payload = {"win_totals": {"alpha": "2", 9: 3}}
    calls = _patch_tournament(monkeypatch, payload)

    games_per_shuffle = runner.TournamentConfig(n_players=3).games_per_shuffle
    cfg = AppConfig(
        io=IOConfig(results_dir=tmp_path / "out", append_seed=False),
        sim=SimConfig(
            n_players_list=[3],
            num_shuffles=2,
            recompute_num_shuffles=False,
        ),
    )

    total_games = runner.run_tournament(cfg)

    summary_path = tmp_path / "out" / "3_players" / "3p_checkpoint.parquet"
    assert summary_path.exists()
    summary_df = pd.read_parquet(summary_path)
    counters = dict(zip(summary_df["strategy"], summary_df["wins"]))
    assert counters == {"alpha": 2, "9": 3}

    expected_total = 2 * games_per_shuffle
    assert total_games == expected_total
