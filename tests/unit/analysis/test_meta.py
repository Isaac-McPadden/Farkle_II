from __future__ import annotations

import json
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

    parquet_path = cfg.analysis_dir / "strategy_summary_2p_meta.parquet"
    json_path = cfg.analysis_dir / "meta_2p.json"
    assert parquet_path.exists()
    assert json_path.exists()

    pooled = pd.read_parquet(parquet_path)
    assert pooled.columns.tolist() == meta.POOLED_COLUMNS
    assert pooled["strategy_id"].tolist() == ["Keep"]

    stats = json.loads(json_path.read_text())
    assert stats["method"] == "fixed"
    assert stats["I2"] <= 90.0
    assert math.isfinite(stats["Q"])
    assert math.isfinite(stats["tau2"])
