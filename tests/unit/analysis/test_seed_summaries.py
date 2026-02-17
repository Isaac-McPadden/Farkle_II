from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from farkle.analysis import seed_summaries
from farkle.config import AppConfig
from farkle.utils.stats import wilson_ci


def _make_cfg(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.sim.seed = 18
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    cfg.analysis.outputs = {}
    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _write_metrics(cfg: AppConfig, frame: pd.DataFrame) -> None:
    path = cfg.metrics_output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_seed_summaries_writes_expected_columns(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    metrics = pd.DataFrame(
        [
            {
                "strategy": "1",
                "n_players": 2,
                "games": 10,
                "wins": 6,
                "mean_n_rounds": 17.0,
                "mean_farkles": 1.2,
            },
            {
                "strategy": "1",
                "n_players": 2,
                "games": 5,
                "wins": 3,
                "mean_n_rounds": 18.0,
                "mean_farkles": 1.0,
            },
            {
                "strategy": "2",
                "n_players": 2,
                "games": 15,
                "wins": 4,
                "mean_n_rounds": 20.0,
                "mean_farkles": 1.4,
            },
        ]
    )
    _write_metrics(cfg, metrics)

    seed_summaries.run(cfg)

    path = cfg.seed_summaries_dir(2) / "strategy_summary_2p_seed18.parquet"
    assert path.exists()
    summary = pd.read_parquet(path)
    expected_cols = [
        "strategy_id",
        "players",
        "seed",
        "games",
        "wins",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "farkles_mean",
        "turns_mean",
    ]
    assert summary.columns.tolist() == expected_cols

    s1 = summary.loc[summary["strategy_id"] == 1].iloc[0]
    assert s1["games"] == 15
    assert s1["wins"] == 9
    assert s1["win_rate"] == pytest.approx(9 / 15)
    ci_lo, ci_hi = wilson_ci(9, 15)
    assert s1["ci_lo"] == pytest.approx(ci_lo)
    assert s1["ci_hi"] == pytest.approx(ci_hi)
    assert s1["turns_mean"] == pytest.approx((17.0 * 10 + 18.0 * 5) / 15)
    assert s1["farkles_mean"] == pytest.approx((1.2 * 10 + 1.0 * 5) / 15)


def test_seed_summaries_respects_input_seed_column(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    metrics = pd.DataFrame(
        [
            {"strategy": "1", "n_players": 2, "games": 10, "wins": 5, "seed": 99},
            {"strategy": "1", "n_players": 2, "games": 12, "wins": 6, "seed": 101},
        ]
    )
    _write_metrics(cfg, metrics)

    seed_summaries.run(cfg)

    for seed in (99, 101):
        path = cfg.seed_summaries_dir(2) / f"strategy_summary_2p_seed{seed}.parquet"
        assert path.exists()
        summary = pd.read_parquet(path)
        assert summary["seed"].unique().tolist() == [seed]


def test_seed_summaries_skips_when_unchanged(tmp_path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path)
    metrics = pd.DataFrame(
        [
            {"strategy": "1", "n_players": 2, "games": 10, "wins": 5},
        ]
    )
    _write_metrics(cfg, metrics)
    seed_summaries.run(cfg)

    def _fail(*_args, **_kwargs) -> None:  # pragma: no cover - executed when logic regresses
        raise AssertionError("should not rewrite identical summary")

    monkeypatch.setattr(seed_summaries, "write_parquet_atomic", _fail)
    seed_summaries.run(cfg)


def test_seed_summaries_force_rewrites(tmp_path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path)
    metrics = pd.DataFrame(
        [
            {"strategy": "1", "n_players": 2, "games": 10, "wins": 5},
            {"strategy": "2", "n_players": 3, "games": 12, "wins": 6},
        ]
    )
    _write_metrics(cfg, metrics)
    seed_summaries.run(cfg)

    calls: list[Path] = []

    def _record(table, path, codec="snappy"):  # noqa: ARG001
        calls.append(Path(path))

    monkeypatch.setattr(seed_summaries, "write_parquet_atomic", _record)
    seed_summaries.run(cfg, force=True)

    assert {p.name for p in calls} == {
        "strategy_summary_2p_seed18.parquet",
        "strategy_summary_3p_seed18.parquet",
        "seed_18_summary_long.parquet",
        "seed_18_summary_weighted.parquet",
    }


def test_seed_summaries_syncs_to_meta_dir(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.io.meta_analysis_dir = Path("shared_meta")

    metrics = pd.DataFrame(
        [
            {"strategy": "1", "n_players": 2, "games": 10, "wins": 5},
        ]
    )
    _write_metrics(cfg, metrics)

    seed_summaries.run(cfg)

    seed_long_stage_path = cfg.seed_summaries_stage_dir / "seed_18_summary_long.parquet"
    seed_weighted_stage_path = cfg.seed_summaries_stage_dir / "seed_18_summary_weighted.parquet"
    assert seed_long_stage_path.exists()
    assert seed_weighted_stage_path.exists()

    meta_path = cfg.meta_analysis_dir / "strategy_summary_2p_seed18.parquet"
    meta_long_path = cfg.meta_analysis_dir / "seed_18_summary_long.parquet"
    meta_weighted_path = cfg.meta_analysis_dir / "seed_18_summary_weighted.parquet"
    assert meta_path.exists()
    assert meta_long_path.exists()
    assert meta_weighted_path.exists()

    summary = pd.read_parquet(meta_path)
    assert summary["seed"].unique().tolist() == [18]
    assert summary["players"].unique().tolist() == [2]


def test_seed_summaries_rebuilds_missing_meta_mirror(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.io.meta_analysis_dir = Path("shared_meta")
    metrics = pd.DataFrame(
        [
            {"strategy": "1", "n_players": 2, "games": 10, "wins": 5},
        ]
    )
    _write_metrics(cfg, metrics)

    seed_summaries.run(cfg)

    meta_long_path = cfg.meta_analysis_dir / "seed_18_summary_long.parquet"
    assert meta_long_path.exists()


def test_seed_summaries_rebuilds_missing_seed_file_only(tmp_path: Path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path)
    metrics = pd.DataFrame(
        [
            {"strategy": "1", "n_players": 2, "games": 10, "wins": 6, "seed": 18},
            {"strategy": "1", "n_players": 2, "games": 8, "wins": 5, "seed": 19},
        ]
    )
    _write_metrics(cfg, metrics)
    seed_summaries.run(cfg)

    missing_path = cfg.seed_summaries_dir(2) / "strategy_summary_2p_seed19.parquet"
    assert missing_path.exists()
    missing_path.unlink()

    calls: list[Path] = []
    original_writer = seed_summaries.write_parquet_atomic

    def _record_writer(table, path, codec="snappy"):  # noqa: ARG001
        calls.append(Path(path))
        original_writer(table, path)

    monkeypatch.setattr(seed_summaries, "write_parquet_atomic", _record_writer)
    seed_summaries.run(cfg)

    assert missing_path.exists()
    assert {p.name for p in calls} == {"strategy_summary_2p_seed19.parquet"}


def test_seed_summaries_handles_mixed_schema_across_seeds_and_orders_rows(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    metrics = pd.DataFrame(
        [
            {
                "strategy": "2",
                "n_players": 2,
                "seed": 19,
                "games": 3,
                "wins": 2,
                "mean_n_rounds": float("nan"),
                "mean_farkles": 2.0,
            },
            {
                "strategy": "1",
                "n_players": 3,
                "seed": 18,
                "games": 5,
                "wins": 3,
                "mean_n_rounds": 20.0,
                "mean_farkles": float("nan"),
            },
            {
                "strategy": "1",
                "n_players": 2,
                "seed": 19,
                "games": 7,
                "wins": 4,
                "mean_n_rounds": float("nan"),
                "mean_farkles": 1.0,
            },
            {
                "strategy": "2",
                "n_players": 3,
                "seed": 18,
                "games": 4,
                "wins": 2,
                "mean_n_rounds": 16.0,
                "mean_farkles": float("nan"),
            },
        ]
    )
    _write_metrics(cfg, metrics)

    seed_summaries.run(cfg)

    seed18_long = pd.read_parquet(cfg.seed_summaries_stage_dir / "seed_18_summary_long.parquet")
    assert seed18_long["players"].tolist() == [3, 3]
    assert seed18_long["strategy_id"].tolist() == [1, 2]
    assert seed18_long["turns_mean"].tolist() == pytest.approx([20.0, 16.0])
    assert seed18_long["farkles_mean"].isna().all()

    seed19_long = pd.read_parquet(cfg.seed_summaries_stage_dir / "seed_19_summary_long.parquet")
    assert seed19_long["players"].tolist() == [2, 2]
    assert seed19_long["strategy_id"].tolist() == [1, 2]
    assert seed19_long["farkles_mean"].tolist() == pytest.approx([1.0, 2.0])
    assert seed19_long["turns_mean"].isna().all()


def test_seed_summaries_zero_game_seed_logs_pooling_warning_and_persists_outputs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.analysis.pooling_weights = "config"
    cfg.analysis.pooling_weights_by_k = {2: 1.0}
    metrics = pd.DataFrame(
        [
            {
                "strategy": "10",
                "n_players": 2,
                "seed": 31,
                "games": 0,
                "wins": 0,
                "mean_n_rounds": float("nan"),
            },
            {
                "strategy": "11",
                "n_players": 3,
                "seed": 31,
                "games": 0,
                "wins": 0,
                "mean_n_rounds": float("nan"),
            },
        ]
    )
    _write_metrics(cfg, metrics)

    with caplog.at_level(logging.WARNING):
        seed_summaries.run(cfg)

    warning_records = [
        rec for rec in caplog.records if "Missing pooling weights for player counts" in rec.message
    ]
    assert len(warning_records) == 1
    assert warning_records[0].missing == [3]

    seed2 = pd.read_parquet(cfg.seed_summaries_dir(2) / "strategy_summary_2p_seed31.parquet")
    assert seed2["games"].tolist() == [0]
    assert seed2["win_rate"].tolist() == pytest.approx([0.0])
    assert seed2["ci_lo"].tolist() == pytest.approx([0.0])
    assert seed2["ci_hi"].tolist() == pytest.approx([1.0])

    weighted = pd.read_parquet(cfg.seed_summaries_stage_dir / "seed_31_summary_weighted.parquet")
    assert weighted["strategy_id"].tolist() == [10, 11]
    assert weighted["pooling_weight_sum"].tolist() == pytest.approx([0.0, 0.0])


def test_load_metrics_frame_validates_inputs(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    metrics = pd.DataFrame(
        [
            {"strategy": "1", "n_players": 2, "games": 10, "wins": 5},
            {"strategy": "2", "n_players": 2, "games": 0, "wins": 0},
        ]
    )
    _write_metrics(cfg, metrics)

    frame, metrics_path = seed_summaries._load_metrics_frame(cfg)

    assert metrics_path.exists()
    assert frame["seed"].unique().tolist() == [18]
    assert frame["games"].tolist() == [10, 0]
    assert frame.dtypes["games"].kind in {"i", "u"}


def test_load_metrics_frame_raises_on_invalid_inputs(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    _write_metrics(
        cfg,
        pd.DataFrame([
            {"strategy": "1", "n_players": 2, "wins": 1},
        ]),
    )

    with pytest.raises(ValueError, match="missing required columns"):
        seed_summaries._load_metrics_frame(cfg)

    _write_metrics(
        cfg,
        pd.DataFrame([
            {"strategy": "1", "n_players": 2, "wins": 1, "games": -1},
        ]),
    )

    with pytest.raises(ValueError, match="negative game counts"):
        seed_summaries._load_metrics_frame(cfg)
