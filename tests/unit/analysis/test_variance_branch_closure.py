from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farkle.analysis import variance
from farkle.analysis.stage_state import stage_done_path, write_stage_done
from farkle.config import AppConfig, IOConfig, SimConfig
from farkle.utils.analysis_shared import is_na


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    config = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=7, n_players_list=[2]),
    )
    config.analysis.outputs = {}
    return config


def _write_metrics(cfg: AppConfig, rows: list[dict[str, object]]) -> Path:
    path = cfg.metrics_input_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_seed_summary(cfg: AppConfig, *, players: int, seed: int, rows: list[dict[str, object]]) -> Path:
    path = cfg.seed_summaries_dir(players) / f"strategy_summary_{players}p_seed{seed}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{**row, "players": players, "seed": seed} for row in rows]
    pd.DataFrame(payload).to_parquet(path, index=False)
    return path


def _seed_rows(strategy_id: str, *, win_rate: float, score: float, turns: float) -> dict[str, object]:
    return {
        "strategy_id": strategy_id,
        "win_rate": win_rate,
        "score_mean": score,
        "turns_mean": turns,
    }


def _variance_paths(cfg: AppConfig) -> tuple[Path, Path, Path]:
    return (
        cfg.variance_output_path(variance.VARIANCE_OUTPUT),
        cfg.variance_output_path(variance.SUMMARY_OUTPUT),
        cfg.variance_output_path(variance.COMPONENTS_OUTPUT),
    )


def _variance_stamps(cfg: AppConfig) -> tuple[Path, Path, Path, Path]:
    return (
        stage_done_path(cfg.variance_stage_dir, "variance.detail"),
        stage_done_path(cfg.variance_stage_dir, "variance.summary"),
        stage_done_path(cfg.variance_stage_dir, "variance.components"),
        stage_done_path(cfg.variance_stage_dir, "variance"),
    )


def test_run_early_returns(cfg: AppConfig) -> None:
    variance_path, summary_path, components_path = _variance_paths(cfg)
    detail_stamp, summary_stamp, components_stamp, master_stamp = _variance_stamps(cfg)

    variance.run(cfg)
    assert not variance_path.exists()
    assert not summary_path.exists()
    assert not components_path.exists()
    assert not detail_stamp.exists()

    _write_metrics(cfg, [{"strategy_id": "A", "players": 2, "win_rate": 0.5}])
    variance.run(cfg)
    assert not summary_stamp.exists()

    empty_seed = cfg.seed_summaries_dir(2) / "strategy_summary_2p_seed1.parquet"
    empty_seed.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["strategy_id", "players", "seed", "win_rate"]).to_parquet(empty_seed, index=False)
    variance.run(cfg)
    assert not components_stamp.exists()
    assert not master_stamp.exists()


def test_run_up_to_date_skip(cfg: AppConfig) -> None:
    metrics_path = _write_metrics(cfg, [{"strategy_id": "A", "players": 2, "win_rate": 0.52}])
    seed_path = _write_seed_summary(
        cfg,
        players=2,
        seed=1,
        rows=[_seed_rows("A", win_rate=0.5, score=100.0, turns=8.0)],
    )
    _write_seed_summary(
        cfg,
        players=2,
        seed=2,
        rows=[_seed_rows("A", win_rate=0.54, score=102.0, turns=8.2)],
    )

    variance_path, summary_path, components_path = _variance_paths(cfg)
    for out in (variance_path, summary_path, components_path):
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": [1]}).to_parquet(out, index=False)

    detail_stamp, summary_stamp, components_stamp, _ = _variance_stamps(cfg)
    inputs = [metrics_path, seed_path]
    write_stage_done(detail_stamp, inputs=inputs, outputs=[variance_path], config_sha=cfg.config_sha)
    write_stage_done(summary_stamp, inputs=inputs, outputs=[summary_path], config_sha=cfg.config_sha)
    write_stage_done(components_stamp, inputs=inputs, outputs=[components_path], config_sha=cfg.config_sha)

    mtimes = [p.stat().st_mtime_ns for p in (variance_path, summary_path, components_path)]
    variance.run(cfg)
    assert mtimes == [p.stat().st_mtime_ns for p in (variance_path, summary_path, components_path)]


def test_run_no_overlap_and_insufficient_seed_pruning(cfg: AppConfig) -> None:
    _write_metrics(cfg, [{"strategy_id": "A", "players": 2, "win_rate": 0.52}])
    _write_seed_summary(
        cfg,
        players=2,
        seed=1,
        rows=[_seed_rows("B", win_rate=0.4, score=100.0, turns=8.0)],
    )
    _write_seed_summary(
        cfg,
        players=2,
        seed=2,
        rows=[_seed_rows("B", win_rate=0.6, score=101.0, turns=8.1)],
    )
    variance.run(cfg)
    assert not cfg.variance_output_path(variance.VARIANCE_OUTPUT).exists()

    cfg2 = AppConfig(io=IOConfig(results_dir_prefix=cfg.io.results_dir_prefix.parent / "results2"), sim=SimConfig(seed=7, n_players_list=[2]))
    cfg2.analysis.outputs = {}
    _write_metrics(cfg2, [{"strategy_id": "A", "players": 2, "win_rate": 0.52}])
    _write_seed_summary(
        cfg2,
        players=2,
        seed=1,
        rows=[_seed_rows("A", win_rate=0.4, score=100.0, turns=8.0)],
    )
    variance.run(cfg2)
    assert not cfg2.variance_output_path(variance.VARIANCE_OUTPUT).exists()
    assert not stage_done_path(cfg2.variance_stage_dir, "variance").exists()


@pytest.mark.parametrize(
    ("remove_stamp", "expected_updated"),
    [
        ("variance.detail", variance.VARIANCE_OUTPUT),
        ("variance.summary", variance.SUMMARY_OUTPUT),
        ("variance.components", variance.COMPONENTS_OUTPUT),
    ],
)
def test_run_partial_up_to_date_updates_only_stale_output(
    cfg: AppConfig, remove_stamp: str, expected_updated: str
) -> None:
    _write_metrics(
        cfg,
        [
            {"strategy_id": "A", "players": 2, "win_rate": 0.52},
            {"strategy_id": "B", "players": 2, "win_rate": 0.48},
        ],
    )
    for seed, (a_rate, b_rate) in enumerate([(0.50, 0.48), (0.54, 0.46)], start=1):
        _write_seed_summary(
            cfg,
            players=2,
            seed=seed,
            rows=[
                _seed_rows("A", win_rate=a_rate, score=100 + seed, turns=8 + seed / 10),
                _seed_rows("B", win_rate=b_rate, score=95 + seed, turns=7 + seed / 10),
            ],
        )

    variance.run(cfg)

    variance_path, summary_path, components_path = _variance_paths(cfg)
    detail_stamp, summary_stamp, components_stamp, master_stamp = _variance_stamps(cfg)
    stamp_map = {
        "variance.detail": detail_stamp,
        "variance.summary": summary_stamp,
        "variance.components": components_stamp,
    }
    old_output_times = {
        variance.VARIANCE_OUTPUT: variance_path.stat().st_mtime_ns,
        variance.SUMMARY_OUTPUT: summary_path.stat().st_mtime_ns,
        variance.COMPONENTS_OUTPUT: components_path.stat().st_mtime_ns,
    }
    old_stamp_times = {name: path.stat().st_mtime_ns for name, path in stamp_map.items()}
    old_master = master_stamp.stat().st_mtime_ns

    stamp_map[remove_stamp].unlink()
    variance.run(cfg)

    new_output_times = {
        variance.VARIANCE_OUTPUT: variance_path.stat().st_mtime_ns,
        variance.SUMMARY_OUTPUT: summary_path.stat().st_mtime_ns,
        variance.COMPONENTS_OUTPUT: components_path.stat().st_mtime_ns,
    }
    for output_name, old_time in old_output_times.items():
        if output_name == expected_updated:
            assert new_output_times[output_name] > old_time
        else:
            assert new_output_times[output_name] == old_time

    for name, stamp_path in stamp_map.items():
        if name == remove_stamp:
            assert stamp_path.stat().st_mtime_ns > old_stamp_times[name]
        else:
            assert stamp_path.stat().st_mtime_ns == old_stamp_times[name]
    assert master_stamp.stat().st_mtime_ns > old_master


def test_helper_edge_cases_and_column_order(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    empty_metrics_path = tmp_path / "metrics_empty.parquet"
    pd.DataFrame(columns=["strategy_id", "players", "win_rate"]).to_parquet(empty_metrics_path, index=False)
    loaded_empty = variance._load_metrics(empty_metrics_path)
    assert list(loaded_empty.columns) == ["strategy_id", "players", "win_rate"]
    assert loaded_empty.empty

    bad_metrics_path = tmp_path / "metrics_bad.parquet"
    pd.DataFrame([{"strategy_id": "A", "players": 2}]).to_parquet(bad_metrics_path, index=False)
    with pytest.raises(ValueError, match="metrics parquet missing required columns"):
        variance._load_metrics(bad_metrics_path)

    seed_frame = pd.DataFrame(
        {
            "strategy_id": ["A", "A", "B", "B"],
            "players": [2, 2, 2, 2],
            "seed": [1, 2, 1, 2],
            "win_rate": [np.nan, np.nan, 0.4, 0.6],
            "score_mean": [100.0, np.nan, 90.0, np.nan],
            "turns_mean": [8.0, np.nan, 7.0, np.nan],
        }
    )
    computed = variance._compute_variance(seed_frame)
    assert set(computed["strategy_id"]) == {"B"}

    empty_components = variance._compute_variance_components(pd.DataFrame())
    assert list(empty_components.columns) == [
        "strategy_id",
        "players",
        "component",
        "n_seeds",
        "mean",
        "variance",
        "std_dev",
        "se_mean",
        "ci_lower",
        "ci_upper",
    ]
    assert empty_components.empty

    with caplog.at_level("INFO"):
        components = variance._compute_variance_components(seed_frame, min_seeds=2)
    assert any("Skipping variance components due to insufficient seeds" in rec.message for rec in caplog.records)
    assert set(components["component"]) == {"win_rate"}

    merged_empty = variance._merge_metrics(pd.DataFrame(), pd.DataFrame())
    assert merged_empty.empty
    assert list(merged_empty.columns) == [
        "strategy_id",
        "players",
        "win_rate",
        "mean_seed_win_rate",
        "variance_win_rate",
        "std_win_rate",
        "se_win_rate",
        "signal_to_noise",
        "n_seeds",
    ]

    variance_frame = pd.DataFrame(
        {
            "strategy_id": ["A", "B"],
            "players": [2, 2],
            "n_seeds": [2, 2],
            "mean_seed_win_rate": [0.60, 0.55],
            "variance_win_rate": [0.02, np.nan],
            "std_win_rate": [0.141421356, 0.0],
            "se_win_rate": [0.1, 0.0],
        }
    )
    metrics_empty = pd.DataFrame(columns=["strategy_id", "players", "win_rate"])
    merged_right = variance._merge_metrics(metrics_empty, variance_frame)
    assert merged_right.loc[0, "win_rate_mean"] == pytest.approx(0.60)
    assert merged_right.loc[0, "signal_to_noise"] == pytest.approx(1.0)
    assert is_na(merged_right.loc[1, "signal_to_noise"])

    desired_order = [
        "strategy_id",
        "players",
        "win_rate",
        "mean_seed_win_rate",
        "variance_win_rate",
        "std_win_rate",
        "se_win_rate",
        "signal_to_noise",
        "n_seeds",
    ]
    assert merged_right.columns.tolist()[: len(desired_order)] == desired_order

    empty_summary = variance._summarize_variance(pd.DataFrame())
    assert empty_summary.empty
    assert list(empty_summary.columns) == ["n_players", "mean_variance", "median_variance"]

    nan_summary = variance._summarize_variance(
        pd.DataFrame(
            {
                "players": [2, 3],
                "variance_win_rate": [np.nan, np.nan],
            }
        )
    )
    assert nan_summary.empty
