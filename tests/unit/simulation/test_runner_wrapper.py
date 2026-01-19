from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable, cast

import pandas as pd
import pytest

# pytestmark = pytest.mark.xfail(
#     reason=(
#         "Runner output normalization is flaky under randomized seeds; "
#         "see https://github.com/Isaac-McPadden/Farkle_II/issues/201"
#     ),
#     strict=False,
# )

pytest.importorskip("pyarrow")

import farkle.simulation.runner as runner
from farkle.config import AppConfig, IOConfig, SimConfig


def _patch_tournament(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    *,
    after_checkpoint: Callable[[Path, dict[str, object]], None] | None = None,
) -> dict[str, object]:
    calls: dict[str, object] = {}

    def fake_run_tournament(**kwargs: object) -> None:  # noqa: ANN001 - mirrors target
        calls.update(kwargs)
        ckpt_path = cast(Path, kwargs["checkpoint_path"])
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_bytes(pickle.dumps(payload))
        if after_checkpoint is not None:
            after_checkpoint(ckpt_path, kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runner.tournament_mod, "run_tournament", fake_run_tournament)
    return calls


def test_runner_passes_metric_flags(tmp_path, monkeypatch, sim_artifacts):
    payload = pickle.loads(sim_artifacts["checkpoint"].read_bytes())

    calls = _patch_tournament(monkeypatch, payload)

    cfg = AppConfig(
        io=IOConfig(results_dir=tmp_path / "out"),
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
    assert calls["row_output_directory"] == tmp_path / "out" / "2_players" / "2p_rows"
    check_absolute = calls["row_output_directory"]
    assert isinstance(check_absolute, (str, os.PathLike))
    assert Path(check_absolute).is_absolute()
    assert calls["checkpoint_path"] == tmp_path / "out" / "2_players" / "2p_checkpoint.pkl"
    config = cast(runner.TournamentConfig, calls["config"])
    expected_games = config.games_per_shuffle
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
        io=IOConfig(results_dir=tmp_path / "out"),
        sim=SimConfig(n_players_list=[2], recompute_num_shuffles=False, num_shuffles=1),
    )

    with pytest.raises(TypeError) as excinfo:
        runner.run_tournament(cfg)

    assert str(excinfo.value) == f"Unexpected win_totals payload type: {expected_type!r}"


def test_runner_writes_normalized_counters(tmp_path, monkeypatch):
    payload = {"win_totals": {"alpha": "2", 9: 3}}
    calls = _patch_tournament(monkeypatch, payload)

    tournament = runner.TournamentConfig(n_players=3)
    assert tournament.games_per_shuffle > 0
    cfg = AppConfig(
        io=IOConfig(results_dir=tmp_path / "out"),
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
    counters = dict(zip(summary_df["strategy"], summary_df["wins"], strict=False))
    assert counters == {"alpha": 2, "9": 3}

    config = cast(runner.TournamentConfig, calls["config"])
    expected_total = config.games_per_shuffle * cfg.sim.num_shuffles
    assert total_games == expected_total


def test_resolve_row_output_dir_formats_and_prefixes(tmp_path):
    cfg = AppConfig(
        io=IOConfig(results_dir=tmp_path / "results"),
        sim=SimConfig(n_players_list=[2], row_dir=Path("custom")),
    )

    formatted = runner._resolve_row_output_dir(cfg, 2)
    assert formatted == tmp_path / "results" / "2_players" / "2p_custom"

    cfg.sim.row_dir = Path("{n}_rows")
    formatted = runner._resolve_row_output_dir(cfg, 3)
    assert formatted == tmp_path / "results" / "3_players" / "3_rows"


def test_filter_player_counts_reports_invalid(monkeypatch):
    cfg = AppConfig(
        io=IOConfig(results_dir=Path("ignored")),
        sim=SimConfig(n_players_list=[3]),
    )
    monkeypatch.setattr(runner, "experiment_size", lambda **_: 4)

    valid, invalid, grid_size, source = runner._filter_player_counts(cfg, [2, 3])
    assert valid == [2]
    assert invalid == [3]
    assert grid_size == 4
    assert source == "experiment_size"


def test_compute_num_shuffles_prefers_overrides(monkeypatch):
    per_cfg = SimConfig(num_shuffles=7, recompute_num_shuffles=False)
    cfg = AppConfig(io=IOConfig(results_dir=Path("ignored")), sim=SimConfig(per_n={2: per_cfg}))

    n_shuffles = runner._compute_num_shuffles_from_config(cfg, n_strategies=4, n_players=2)
    assert n_shuffles == 7


def test_compute_num_shuffles_recomputes(monkeypatch):
    cfg = AppConfig(io=IOConfig(results_dir=Path("ignored")), sim=SimConfig())
    cfg.sim.recompute_num_shuffles = True
    cfg.sim.num_shuffles = 1

    # Keep the calculation deterministic
    monkeypatch.setattr(
        runner, "games_for_power_from_design", lambda n_strategies, k_players, method, design: 9
    )

    n_shuffles = runner._compute_num_shuffles_from_config(cfg, n_strategies=6, n_players=2)
    assert n_shuffles == 9
