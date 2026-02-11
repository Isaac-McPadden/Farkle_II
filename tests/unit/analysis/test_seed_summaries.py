from __future__ import annotations

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
    meta_long_path.unlink()

    seed_summaries.run(cfg)

    assert meta_long_path.exists()


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
