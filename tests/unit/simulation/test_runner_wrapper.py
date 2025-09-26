from __future__ import annotations

import pickle
import shutil
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

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

    def copy_metrics(ckpt_path: Path, _kwargs: dict[str, object]) -> None:
        metrics_dst = ckpt_path.with_name("5p_metrics.parquet")
        shutil.copy2(tmp_artifacts_with_legacy["metrics"], metrics_dst)

    calls = _patch_tournament(monkeypatch, payload, after_checkpoint=copy_metrics)

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
    assert Path(calls["row_output_directory"]).is_absolute()
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


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        pytest.param(["invalid"], list, id="list_payload"),
        pytest.param({"win_totals": 3.14}, float, id="wrong_value_type"),
    ],
)
def test_runner_rejects_malformed_checkpoints(tmp_path, monkeypatch, payload, expected_type):
    _patch_tournament(monkeypatch, payload)

    def fail_sink(*_args: object, **_kwargs: object) -> None:
        pytest.fail("sinks.write_counter_csv should not be invoked for bad payloads")

    monkeypatch.setattr(runner.sinks, "write_counter_csv", fail_sink)

    cfg = runner.AppConfig(io=runner.IOConfig(results_dir=tmp_path / "out"))

    with pytest.raises(TypeError) as excinfo:
        runner.run_tournament(cfg)

    assert excinfo.value.args[0] == "Unexpected win_totals payload type"
    assert excinfo.value.args[1] is expected_type


def test_runner_writes_normalized_counters(tmp_path, monkeypatch):
    payload = {"win_totals": {"alpha": "2", 9: 3}}
    _patch_tournament(monkeypatch, payload)

    recorded: dict[str, object] = {}

    def capture_sink(counter: Counter[str], path: Path) -> None:
        recorded["counter"] = counter
        recorded["path"] = path

    monkeypatch.setattr(runner.sinks, "write_counter_csv", capture_sink)

    games_per_shuffle = runner.TournamentConfig(n_players=3).games_per_shuffle
    cfg = runner.AppConfig(
        io=runner.IOConfig(results_dir=tmp_path / "out"),
        sim=runner.SimConfig(n_players=3, n_games=games_per_shuffle + 1),
    )

    total_games = runner.run_tournament(cfg)

    assert recorded["path"] == tmp_path / "out" / "win_counts.csv"
    assert isinstance(recorded["counter"], Counter)
    assert recorded["counter"] == Counter({"alpha": 2, "9": 3})

    expected_total = 2 * games_per_shuffle
    assert total_games == expected_total

