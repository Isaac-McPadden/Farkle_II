from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd
import pytest

from farkle.analysis import meta
from farkle.config import AppConfig


def _make_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.io.results_dir = tmp_path / "results_meta"
    cfg.io.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.analysis.outputs = {}
    cfg.sim.seed = 42
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_estimate_rate_and_variance_validates_inputs():
    rate, var = meta._estimate_rate_and_variance(0, 0, None)
    assert math.isnan(rate)
    assert var == math.inf

    with pytest.raises(ValueError):
        meta._estimate_rate_and_variance(5, 2, None)

    rate, var = meta._estimate_rate_and_variance(10, 10, 1.0)
    assert 0.0 <= rate <= 1.0
    assert var >= meta.MIN_VARIANCE


def test_pool_winrates_prefers_fixed_effects_when_I2_low() -> None:
    df_seed1 = pd.DataFrame(
        [
            {
                "strategy_id": "S1",
                "players": 2,
                "seed": 1,
                "games": 50,
                "wins": 25,
                "win_rate": 0.50,
            },
            {
                "strategy_id": "S2",
                "players": 2,
                "seed": 1,
                "games": 50,
                "wins": 20,
                "win_rate": 0.40,
            },
        ]
    )
    df_seed2 = pd.DataFrame(
        [
            {
                "strategy_id": "S1",
                "players": 2,
                "seed": 2,
                "games": 50,
                "wins": 26,
                "win_rate": 0.52,
            },
            {
                "strategy_id": "S2",
                "players": 2,
                "seed": 2,
                "games": 50,
                "wins": 18,
                "win_rate": 0.36,
            },
        ]
    )

    result = meta.pool_winrates([df_seed1, df_seed2], use_random_if_I2_gt=80.0)
    assert result.method == "fixed"
    assert result.pooled.shape[0] == 2

    s1 = result.pooled.loc[result.pooled["strategy_id"] == "S1"].iloc[0]
    weight_a = 1.0 / (0.5 * 0.5 / 50.0)
    weight_b = 1.0 / (0.52 * 0.48 / 50.0)
    expected = (weight_a * 0.50 + weight_b * 0.52) / (weight_a + weight_b)
    assert s1["win_rate"] == pytest.approx(expected)
    expected_se = math.sqrt(1.0 / (weight_a + weight_b))
    assert s1["se"] == pytest.approx(expected_se)
    assert pytest.approx(0.0) == result.I2


def test_meta_run_writes_pooled_outputs(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)

    df_seed1 = pd.DataFrame(
        [
            {
                "strategy_id": "Keep",
                "players": 2,
                "seed": 1,
                "games": 30,
                "wins": 18,
                "win_rate": 0.60,
            },
            {
                "strategy_id": "Drop",
                "players": 2,
                "seed": 1,
                "games": 30,
                "wins": 10,
                "win_rate": 0.33,
            },
        ]
    )
    df_seed2 = pd.DataFrame(
        [
            {
                "strategy_id": "Keep",
                "players": 2,
                "seed": 2,
                "games": 40,
                "wins": 20,
                "win_rate": 0.50,
            },
            # "Drop" missing here to trigger the presence rule.
        ]
    )
    df_seed1.to_parquet(cfg.analysis_dir / "strategy_summary_2p_seed1.parquet", index=False)
    df_seed2.to_parquet(cfg.analysis_dir / "strategy_summary_2p_seed2.parquet", index=False)

    meta.run(cfg, use_random_if_I2_gt=90.0)

    parquet_name = "strategy_summary_2p_meta.parquet"
    json_name = "meta_2p.json"
    parquet_path = cfg.meta_output_path(2, parquet_name)
    json_path = cfg.meta_output_path(2, json_name)
    assert parquet_path.exists()
    assert json_path.exists()

    pooled = pd.read_parquet(cfg.meta_input_path(2, parquet_name))
    assert pooled.columns.tolist() == meta.POOLED_COLUMNS
    assert pooled["strategy_id"].tolist() == ["Keep"]

    stats = json.loads(cfg.meta_input_path(2, json_name).read_text())
    assert stats["method"] == "fixed"
    assert stats["I2"] <= 90.0
    assert math.isfinite(stats["Q"])
    assert math.isfinite(stats["tau2"])


def test_meta_skips_when_single_seed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg = _make_cfg(tmp_path)
    caplog.set_level(logging.INFO)

    df_seed1 = pd.DataFrame(
        [
            {
                "strategy_id": "Solo",
                "players": 2,
                "seed": 1,
                "games": 20,
                "wins": 10,
                "win_rate": 0.50,
            }
        ]
    )
    df_seed1.to_parquet(cfg.analysis_dir / "strategy_summary_2p_seed1.parquet", index=False)

    meta.run(cfg, use_random_if_I2_gt=90.0)

    parquet_path = cfg.meta_output_path(2, "strategy_summary_2p_meta.parquet")
    assert not parquet_path.exists()
    assert any("requires multiple seeds" in rec.message for rec in caplog.records)


def test_meta_limits_other_seeds_and_respects_override(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.analysis.meta_max_other_seeds = 1

    frames_by_seed: dict[int, pd.DataFrame] = {}

    def _write(seed: int, wins: int) -> None:
        df = pd.DataFrame(
            [
                {
                    "strategy_id": "Only",
                    "players": 2,
                    "seed": seed,
                    "games": 10,
                    "wins": wins,
                    "win_rate": wins / 10.0,
                }
            ]
        )
        frames_by_seed[seed] = df
        df.to_parquet(cfg.analysis_dir / f"strategy_summary_2p_seed{seed}.parquet", index=False)

    _write(42, 6)
    _write(7, 5)
    _write(99, 2)

    meta.run(cfg, use_random_if_I2_gt=90.0)

    pooled = pd.read_parquet(cfg.meta_input_path(2, "strategy_summary_2p_meta.parquet"))
    assert pooled["n_seeds"].iloc[0] == 2
    expected_win_rate = meta.pool_winrates(
        [frames_by_seed[42], frames_by_seed[99]], use_random_if_I2_gt=90.0
    ).pooled["win_rate"].iloc[0]
    assert pooled["win_rate"].iloc[0] == pytest.approx(expected_win_rate)

    cfg.analysis.meta_comparison_seed = 7
    meta.run(cfg, force=True, use_random_if_I2_gt=90.0)

    pooled_override = pd.read_parquet(cfg.meta_input_path(2, "strategy_summary_2p_meta.parquet"))
    assert pooled_override["n_seeds"].iloc[0] == 2
    expected_override = meta.pool_winrates(
        [frames_by_seed[42], frames_by_seed[7]], use_random_if_I2_gt=90.0
    ).pooled["win_rate"].iloc[0]
    assert pooled_override["win_rate"].iloc[0] == pytest.approx(expected_override)


def test_apply_strategy_presence_filters_and_reports_missing():
    frames = [
        pd.DataFrame({"strategy_id": ["A", "B"], "seed": [1, 1]}),
        pd.DataFrame({"strategy_id": ["A", "C"], "seed": [2, 2]}),
    ]

    filtered, missing = meta._apply_strategy_presence(frames)

    assert all(df["strategy_id"].tolist() == ["A"] for df in filtered)
    assert missing == {"B": [2], "C": [1]}
