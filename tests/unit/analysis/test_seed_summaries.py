from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from farkle.analysis import seed_summaries
from farkle.config import AppConfig
from farkle.utils.analysis_shared import is_na
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


def test_seed_summaries_meta_sync_order_is_deterministic_under_parallel_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.io.meta_analysis_dir = Path("shared_meta")
    metrics = pd.DataFrame(
        [
            {"strategy": "1", "n_players": 3, "games": 4, "wins": 2, "seed": 20},
            {"strategy": "1", "n_players": 2, "games": 8, "wins": 5, "seed": 19},
            {"strategy": "2", "n_players": 3, "games": 6, "wins": 3, "seed": 19},
            {"strategy": "2", "n_players": 2, "games": 5, "wins": 3, "seed": 20},
        ]
    )
    _write_metrics(cfg, metrics)

    sync_order: list[str] = []
    original_sync = seed_summaries._sync_meta_summary

    def _record_sync(local_cfg, summary, analysis_path):
        sync_order.append(analysis_path.name)
        return original_sync(local_cfg, summary, analysis_path)

    monkeypatch.setattr(seed_summaries, "_sync_meta_summary", _record_sync)

    seed_summaries.run(cfg)

    assert sync_order == [
        "strategy_summary_2p_seed19.parquet",
        "strategy_summary_3p_seed19.parquet",
        "seed_19_summary_long.parquet",
        "seed_19_summary_weighted.parquet",
        "strategy_summary_2p_seed20.parquet",
        "strategy_summary_3p_seed20.parquet",
        "seed_20_summary_long.parquet",
        "seed_20_summary_weighted.parquet",
    ]


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
    assert cast(Any, warning_records[0]).missing == [3]

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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("game-count", "game-count"),
        ("GameCount", "game-count"),
        ("count", "game-count"),
        ("equal_k", "equal-k"),
        ("equalk", "equal-k"),
        ("custom", "config"),
    ],
)
def test_normalize_pooling_scheme_aliases(raw: str, expected: str) -> None:
    assert seed_summaries._normalize_pooling_scheme(raw) == expected


def test_normalize_pooling_scheme_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unknown pooling scheme"):
        seed_summaries._normalize_pooling_scheme("mystery")


def test_pooling_weights_for_seed_summary_schemes() -> None:
    frame = pd.DataFrame(
        {
            "players": [2, 2, 3],
            "games": [4, 6, 10],
        }
    )
    game_count = seed_summaries._pooling_weights_for_seed_summary(
        frame, pooling_scheme="game-count", weights_by_k={}
    )
    assert game_count.tolist() == pytest.approx([4.0, 6.0, 10.0])

    equal_k = seed_summaries._pooling_weights_for_seed_summary(
        frame, pooling_scheme="equal-k", weights_by_k={}
    )
    assert equal_k.tolist() == pytest.approx([0.4, 0.6, 1.0])

    config = seed_summaries._pooling_weights_for_seed_summary(
        frame,
        pooling_scheme="config",
        weights_by_k={2: 0.5, 3: 0.25},
    )
    assert config.tolist() == pytest.approx([0.2, 0.3, 0.25])


def test_pooling_weights_for_seed_summary_rejects_unknown_scheme() -> None:
    frame = pd.DataFrame({"players": [2], "games": [1]})
    with pytest.raises(ValueError, match="Unknown pooling scheme"):
        seed_summaries._pooling_weights_for_seed_summary(
            frame, pooling_scheme="unknown", weights_by_k={}
        )


def test_mean_output_name_override_and_default_passthrough() -> None:
    assert seed_summaries._mean_output_name("mean_n_rounds") == "turns_mean"
    assert seed_summaries._mean_output_name("mean_custom_metric") == "custom_metric_mean"


def test_cast_int32_if_safe_handles_bounds() -> None:
    safe = pd.Series([0, np.iinfo(np.int32).max])
    assert seed_summaries._cast_int32_if_safe(safe).dtype == np.int32

    overflow = pd.Series([0, np.iinfo(np.int32).max + 1], dtype=np.int64)
    assert seed_summaries._cast_int32_if_safe(overflow).dtype == np.int64


def test_weighted_means_helpers_ignore_nan_and_zero_weights() -> None:
    frame = pd.DataFrame(
        {
            "strategy_id": [1, 1, 2, 2],
            "games": [5, 0, 0, 0],
            "mean_score": [2.0, 8.0, np.nan, 3.0],
        }
    )
    by_strategy = seed_summaries._weighted_means_by_strategy(frame, ["mean_score"])
    assert by_strategy.loc[1, "score_mean"] == pytest.approx(2.0)
    assert is_na(by_strategy.loc[2, "score_mean"])

    weighted = seed_summaries._weighted_means_with_weights(
        frame,
        ["mean_score"],
        pd.Series([1.0, 0.0, 0.0, 0.0]),
    )
    assert weighted.loc[1, "pooling_weight_sum"] == pytest.approx(1.0)
    assert weighted.loc[1, "mean_score"] == pytest.approx(2.0)
    assert weighted.loc[2, "pooling_weight_sum"] == pytest.approx(0.0)
    assert is_na(weighted.loc[2, "mean_score"])


def test_load_metrics_frame_raises_for_missing_file_null_seed_and_non_numeric_strategy(
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        seed_summaries._load_metrics_frame(cfg)

    _write_metrics(
        cfg,
        pd.DataFrame(
            [
                {"strategy": "1", "n_players": 2, "games": 1, "wins": 1, "seed": np.nan},
            ]
        ),
    )
    with pytest.raises(ValueError, match="null seed"):
        seed_summaries._load_metrics_frame(cfg)

    _write_metrics(
        cfg,
        pd.DataFrame(
            [
                {"strategy": "not-an-int", "n_players": 2, "games": 1, "wins": 1, "seed": 5},
            ]
        ),
    )
    with pytest.raises(ValueError, match="non-numeric strategy_id"):
        seed_summaries._load_metrics_frame(cfg)


def test_existing_summary_matches_handles_missing_corrupt_column_and_value_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.parquet"
    new_df = pd.DataFrame({"strategy_id": [1], "seed": [1], "games": [2], "wins": [1]})
    assert seed_summaries._existing_summary_matches(path, new_df) is False

    path.write_text("not parquet")
    assert seed_summaries._existing_summary_matches(path, new_df) is False

    pd.DataFrame({"strategy_id": [1], "seed": [1], "games": [2]}).to_parquet(path, index=False)
    assert seed_summaries._existing_summary_matches(path, new_df) is False

    pd.DataFrame({"strategy_id": [1], "seed": [1], "games": [2], "wins": [0]}).to_parquet(
        path, index=False
    )
    assert seed_summaries._existing_summary_matches(path, new_df) is False


def test_sync_meta_summary_short_circuits_or_skips_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path)
    summary = pd.DataFrame({"strategy_id": [1], "seed": [18], "games": [1], "wins": [1]})
    analysis_path = cfg.seed_summaries_stage_dir / "seed_18_summary_long.parquet"

    assert seed_summaries._sync_meta_summary(cfg, summary, analysis_path) is None

    cfg.io.meta_analysis_dir = cfg.analysis_dir
    same_path = cfg.analysis_dir / "seed_18_summary_long.parquet"
    assert seed_summaries._sync_meta_summary(cfg, summary, same_path) is None

    cfg.io.meta_analysis_dir = Path("meta")
    writes: list[Path] = []

    def _record_write(df: pd.DataFrame, path: Path) -> None:
        writes.append(path)

    monkeypatch.setattr(seed_summaries, "_write_summary", _record_write)
    mirrored = cfg.meta_analysis_dir / analysis_path.name
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(mirrored, index=False)
    result = seed_summaries._sync_meta_summary(cfg, summary, analysis_path)
    assert result == mirrored
    assert writes == []


def test_run_returns_early_for_missing_and_empty_metrics(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    seed_summaries.run(cfg)
    done = seed_summaries.stage_done_path(cfg.seed_summaries_stage_dir, "seed_summaries")
    assert not done.exists()

    _write_metrics(cfg, pd.DataFrame(columns=["strategy", "n_players", "games", "wins"]))
    seed_summaries.run(cfg)
    assert not done.exists()


def test_run_raises_for_config_pooling_without_weights(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.analysis.pooling_weights = "config"
    cfg.analysis.pooling_weights_by_k = {}
    _write_metrics(cfg, pd.DataFrame([{"strategy": "1", "n_players": 2, "games": 1, "wins": 1}]))

    with pytest.raises(ValueError, match="pooling_weights_by_k must be set"):
        seed_summaries.run(cfg)


def test_run_does_not_write_stage_done_when_expected_outputs_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path)
    metrics_path = cfg.metrics_output_path()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"strategy": "1", "n_players": 2, "games": 1, "wins": 1}]).to_parquet(
        metrics_path, index=False
    )

    weird_frame = pd.DataFrame(
        [{"strategy_id": 1, "players": np.nan, "seed": 10, "games": 1, "wins": 1}]
    )
    monkeypatch.setattr(seed_summaries, "_load_metrics_frame", lambda _cfg: (weird_frame, metrics_path))
    marker: list[Path] = []

    def _record_stage_done(*_args, **_kwargs) -> None:
        marker.append(Path("called"))

    monkeypatch.setattr(seed_summaries, "write_stage_done", _record_stage_done)
    seed_summaries.run(cfg)
    assert marker == []


def test_run_rewrites_only_when_long_or_weighted_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path)
    _write_metrics(
        cfg,
        pd.DataFrame([{"strategy": "1", "n_players": 2, "games": 10, "wins": 5}]),
    )
    seed_summaries.run(cfg)

    monkeypatch.setattr(seed_summaries, "stage_is_up_to_date", lambda *args, **kwargs: False)
    written: list[str] = []

    def _record_write(_df: pd.DataFrame, path: Path) -> None:
        written.append(path.name)

    monkeypatch.setattr(seed_summaries, "_write_summary", _record_write)

    def _only_weighted_diff(path: Path, _df: pd.DataFrame) -> bool:
        return path.name != "seed_18_summary_weighted.parquet"

    monkeypatch.setattr(seed_summaries, "_existing_summary_matches", _only_weighted_diff)
    seed_summaries.run(cfg)
    assert written == ["seed_18_summary_weighted.parquet"]

    written.clear()

    def _only_long_diff(path: Path, _df: pd.DataFrame) -> bool:
        return path.name != "seed_18_summary_long.parquet"

    monkeypatch.setattr(seed_summaries, "_existing_summary_matches", _only_long_diff)
    seed_summaries.run(cfg)
    assert written == ["seed_18_summary_long.parquet"]
