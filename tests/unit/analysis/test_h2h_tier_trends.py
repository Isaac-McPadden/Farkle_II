from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from farkle.analysis import h2h_tier_trends
from farkle.analysis.stage_state import stage_done_path
from farkle.config import AppConfig


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.sim.n_players_list = [2, "3", "bad", -1]
    cfg.analysis.outputs = {}
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _write_meta(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_resolve_s_tiers_path_prefers_post_h2h_then_head2head_then_analysis(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    post_path = cfg.post_h2h_stage_dir / "h2h_s_tiers.json"
    h2h_path = cfg.head2head_stage_dir / "h2h_s_tiers.json"
    analysis_path = cfg.analysis_dir / "h2h_s_tiers.json"

    analysis_path.write_text('{"1": "S"}')
    assert h2h_tier_trends._resolve_s_tiers_path(cfg) == analysis_path

    h2h_path.parent.mkdir(parents=True, exist_ok=True)
    h2h_path.write_text('{"1": "A"}')
    assert h2h_tier_trends._resolve_s_tiers_path(cfg) == h2h_path

    post_path.parent.mkdir(parents=True, exist_ok=True)
    post_path.write_text('{"1": "S+"}')
    assert h2h_tier_trends._resolve_s_tiers_path(cfg) == post_path


def test_load_s_tiers_handles_invalid_json_shapes_and_mixed_values(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert h2h_tier_trends._load_s_tiers(bad) == {}

    not_dict = tmp_path / "list.json"
    not_dict.write_text('["a", "b"]')
    assert h2h_tier_trends._load_s_tiers(not_dict) == {}

    mixed = tmp_path / "mixed.json"
    mixed.write_text('{"1": "S", "2": 2, "_meta": "skip", "3": "A"}')
    assert h2h_tier_trends._load_s_tiers(mixed) == {"1": "S", "3": "A"}


def test_collect_meta_paths_uses_config_then_glob_fallback(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    k2 = cfg.meta_input_path(2, h2h_tier_trends.META_TEMPLATE.format(players=2))
    _write_meta(k2, [{"strategy_id": "1", "win_rate": 0.6, "se": 0.1, "ci_lo": 0.5, "ci_hi": 0.7}])

    paths = h2h_tier_trends._collect_meta_paths(cfg)
    assert paths == [k2]

    k2.unlink()
    cfg.sim.n_players_list = [2]
    meta_dir = cfg.meta_stage_dir
    k3 = meta_dir / "3p" / "strategy_summary_3p_meta.parquet"
    k4 = meta_dir / "4p" / "strategy_summary_4p_meta.parquet"
    _write_meta(k4, [{"strategy_id": "1", "players": 4, "win_rate": 0.4, "se": 0.1, "ci_lo": 0.2, "ci_hi": 0.6}])
    _write_meta(k3, [{"strategy_id": "1", "players": 3, "win_rate": 0.5, "se": 0.1, "ci_lo": 0.3, "ci_hi": 0.7}])

    assert h2h_tier_trends._collect_meta_paths(cfg) == [k3, k4]


def test_pooled_across_k_zero_weight_and_heterogeneity_metrics() -> None:
    frame = pd.DataFrame(
        [
            {"strategy_id": "nanw", "players": 2, "win_rate": 0.3, "se": math.inf},
            {"strategy_id": "nanw", "players": 3, "win_rate": 0.4, "se": math.inf},
            {"strategy_id": "het", "players": 2, "win_rate": 0.2, "se": 0.1},
            {"strategy_id": "het", "players": 3, "win_rate": 0.8, "se": 0.1},
        ]
    )

    pooled = h2h_tier_trends._pooled_across_k(frame).set_index("strategy_id")

    assert math.isnan(float(pooled.loc["nanw", "pooled_win_rate"]))
    assert math.isnan(float(pooled.loc["nanw", "pooled_se"]))

    assert pooled.loc["het", "Q"] == 18.0
    assert pooled.loc["het", "I2"] == ((18.0 - 1.0) / 18.0) * 100.0


def test_run_handles_missing_input_and_up_to_date_and_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)

    h2h_tier_trends.run(cfg)
    assert not (cfg.stage_dir("h2h_tier_trends") / h2h_tier_trends.OUTPUT_PARQUET).exists()

    s_path = cfg.post_h2h_stage_dir / "h2h_s_tiers.json"
    s_path.parent.mkdir(parents=True, exist_ok=True)
    s_path.write_text('{"1": "S"}')

    meta_path = cfg.meta_input_path(2, h2h_tier_trends.META_TEMPLATE.format(players=2))
    _write_meta(
        meta_path,
        [
            {
                "strategy_id": "1",
                "players": 2,
                "win_rate": 0.6,
                "se": 0.1,
                "ci_lo": 0.4,
                "ci_hi": 0.8,
            }
        ],
    )

    monkeypatch.setattr(h2h_tier_trends, "stage_is_up_to_date", lambda *a, **k: True)
    original_write = h2h_tier_trends.write_parquet_atomic

    def _fail(*_args, **_kwargs) -> None:  # pragma: no cover
        raise AssertionError("write should be skipped when up-to-date")

    monkeypatch.setattr(h2h_tier_trends, "write_parquet_atomic", _fail)
    h2h_tier_trends.run(cfg)

    monkeypatch.setattr(h2h_tier_trends, "stage_is_up_to_date", lambda *a, **k: False)
    monkeypatch.setattr(h2h_tier_trends, "write_parquet_atomic", original_write)
    h2h_tier_trends.run(cfg)

    out_path = cfg.stage_dir("h2h_tier_trends") / h2h_tier_trends.OUTPUT_PARQUET
    done_path = stage_done_path(cfg.stage_dir("h2h_tier_trends"), "h2h_tier_trends")
    assert out_path.exists()
    assert done_path.exists()

    out = pd.read_parquet(out_path)
    assert out.columns.tolist()[:3] == ["strategy_id", "s_tier", "players"]
    assert out.loc[0, "delta_vs_baseline"] == pytest.approx(0.1)
