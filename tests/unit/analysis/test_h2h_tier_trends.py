from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from farkle.analysis import h2h_tier_trends
from farkle.analysis.stage_state import stage_done_path
from farkle.config import AppConfig
from farkle.utils.analysis_shared import is_na


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    bad_n_players_list: Any = [2, "3", "bad", -1]
    cfg.sim.n_players_list = bad_n_players_list
    cfg.analysis.outputs = {}
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _write_meta(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_resolve_s_tier_inputs_uses_seed_paths_contract(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    a_path = tmp_path / "a" / "h2h_s_tiers.json"
    b_path = tmp_path / "b" / "h2h_s_tiers.json"
    a_path.parent.mkdir(parents=True, exist_ok=True)
    b_path.parent.mkdir(parents=True, exist_ok=True)
    a_path.write_text('{"1": "S"}')
    b_path.write_text('{"1": "S+"}')

    resolved = h2h_tier_trends._resolve_s_tier_inputs(
        cfg,
        seed_s_tier_paths=[a_path, b_path],
        interseed_s_tier_path=None,
    )
    assert [item.path for item in resolved] == [a_path, b_path]


def test_resolve_s_tier_inputs_uses_interseed_contract(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    interseed_path = tmp_path / "interseed" / "combined.json"
    interseed_path.parent.mkdir(parents=True, exist_ok=True)
    interseed_path.write_text('{"1": "S"}')

    resolved = h2h_tier_trends._resolve_s_tier_inputs(
        cfg,
        seed_s_tier_paths=None,
        interseed_s_tier_path=interseed_path,
    )
    assert len(resolved) == 1
    assert resolved[0].path == interseed_path


def test_resolve_s_tier_inputs_rejects_missing_paths_with_stage_hint(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    missing = tmp_path / "missing" / "h2h_s_tiers.json"

    with pytest.raises(h2h_tier_trends.MissingTierInputsError) as excinfo:
        h2h_tier_trends._resolve_s_tier_inputs(
            cfg,
            seed_s_tier_paths=[missing],
            interseed_s_tier_path=None,
        )

    message = str(excinfo.value)
    assert "Expected producer stage" in message
    assert "11_post_h2h/h2h_s_tiers.json" in message
    assert str(missing) in message


def test_load_s_tiers_from_sources_merges_by_majority_vote(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    a.write_text('{"1": "S", "2": "S-"}')
    b.write_text('{"1": "S+", "2": "S-"}')
    c.write_text('{"1": "S", "2": "S"}')

    merged = h2h_tier_trends._load_s_tiers_from_sources(
        [
            h2h_tier_trends.SSource(path=a, source_label="seed[0]"),
            h2h_tier_trends.SSource(path=b, source_label="seed[1]"),
            h2h_tier_trends.SSource(path=c, source_label="seed[2]"),
        ]
    )
    assert merged == {"1": "S", "2": "S-"}


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


def test_load_s_tiers_rejects_meta_only_payload(tmp_path: Path) -> None:
    meta_only = tmp_path / "meta_only.json"
    meta_only.write_text('{"_meta": {"status": "failed", "reason": "insufficient_signal"}}')
    assert h2h_tier_trends._load_s_tiers(meta_only) == {}


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


def test_collect_meta_paths_returns_empty_when_meta_stage_inactive(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    cfg.sim.n_players_list = [2]
    meta_path = cfg.meta_input_path(2, h2h_tier_trends.META_TEMPLATE.format(players=2))
    if meta_path.exists():
        meta_path.unlink()

    assert cfg.stage_dir_if_active("meta") is not None
    monkeypatch.setattr(
        cfg,
        "stage_dir_if_active",
        lambda stage: None if stage == "meta" else cfg.stage_dir(stage),
    )

    assert h2h_tier_trends._collect_meta_paths(cfg) == []


def test_load_meta_frames_handles_mixed_inputs_and_dropna(tmp_path: Path) -> None:
    empty_path = tmp_path / "2p" / "strategy_summary_2p_meta.parquet"
    _write_meta(empty_path, [])

    missing_strategy_path = tmp_path / "3p" / "strategy_summary_3p_meta.parquet"
    _write_meta(
        missing_strategy_path,
        [{"players": 3, "win_rate": 0.2, "se": 0.1, "ci_lo": 0.1, "ci_hi": 0.3}],
    )

    no_players_path = tmp_path / "4p" / "strategy_summary_4p_meta.parquet"
    _write_meta(
        no_players_path,
        [
            {"strategy_id": "ok", "win_rate": 0.6, "se": 0.2, "ci_lo": None, "ci_hi": None},
            {"strategy_id": "drop", "win_rate": None, "se": 0.2, "ci_lo": 0.1, "ci_hi": 0.2},
        ],
    )

    valid_partial_path = tmp_path / "5p" / "strategy_summary_5p_meta.parquet"
    _write_meta(
        valid_partial_path,
        [
            {"strategy_id": "v1", "players": 5, "win_rate": 0.3, "se": 0.1, "ci_lo": 0.2, "ci_hi": None},
            {"strategy_id": "v2", "players": 6, "win_rate": None, "se": 0.1, "ci_lo": 0.3, "ci_hi": 0.5},
        ],
    )

    loaded = h2h_tier_trends._load_meta_frames(
        [empty_path, missing_strategy_path, no_players_path, valid_partial_path]
    )

    assert loaded.columns.tolist() == ["strategy_id", "players", "win_rate", "se", "ci_lo", "ci_hi"]
    assert loaded["strategy_id"].tolist() == ["ok", "v1"]
    assert loaded["players"].tolist() == [4, 5]
    assert loaded["ci_hi"].isna().tolist() == [True, True]


@pytest.mark.parametrize(
    ("dirname", "expected"),
    [
        ("3p", 3),
        ("no_players_dir", 0),
    ],
)
def test_players_from_path(dirname: str, expected: int) -> None:
    path = Path("/tmp") / dirname / "strategy_summary_meta.parquet"
    assert h2h_tier_trends._players_from_path(path) == expected


def test_clean_variances_handles_nan_and_nonpositive_values() -> None:
    bad_variances: Any = [None, 0.0, -1.0, 0.2]
    cleaned = h2h_tier_trends._clean_variances(bad_variances)
    assert np.isfinite(cleaned).all()
    assert (cleaned >= h2h_tier_trends.MIN_VARIANCE).all()
    assert cleaned[-1] == pytest.approx(0.2)


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

    assert is_na(pooled.loc["nanw", "pooled_win_rate"])
    assert is_na(pooled.loc["nanw", "pooled_se"])

    assert pooled.loc["het", "Q"] == 18.0
    assert pooled.loc["het", "I2"] == ((18.0 - 1.0) / 18.0) * 100.0


def test_pooled_across_k_i2_falls_back_to_zero_when_df_or_q_zero() -> None:
    frame = pd.DataFrame(
        [
            {"strategy_id": "single", "players": 2, "win_rate": 0.6, "se": 0.1},
            {"strategy_id": "flat", "players": 2, "win_rate": 0.4, "se": 0.1},
            {"strategy_id": "flat", "players": 3, "win_rate": 0.4, "se": 0.1},
        ]
    )
    pooled = h2h_tier_trends._pooled_across_k(frame).set_index("strategy_id")

    assert pooled.loc["single", "I2"] == 0.0
    assert pooled.loc["flat", "Q"] == pytest.approx(0.0)
    assert pooled.loc["flat", "I2"] == 0.0


@pytest.mark.parametrize(
    ("s_tiers", "meta_paths", "meta_frame", "expected_reason"),
    [
        ({}, [Path("unused.parquet")], pd.DataFrame([{"strategy_id": "1", "players": 2, "win_rate": 0.5}]), "h2h_s_tiers.json empty"),
        ({"1": "S"}, [], pd.DataFrame([{"strategy_id": "1", "players": 2, "win_rate": 0.5}]), "meta pooled summaries missing"),
        (
            {"1": "S"},
            [Path("meta.parquet")],
            pd.DataFrame(columns=["strategy_id", "players", "win_rate", "se", "ci_lo", "ci_hi"]),
            "meta pooled summaries empty",
        ),
        (
            {"9": "S"},
            [Path("meta.parquet")],
            pd.DataFrame(
                [{"strategy_id": "1", "players": 2, "win_rate": 0.5, "se": 0.1, "ci_lo": 0.4, "ci_hi": 0.6}]
            ),
            "no overlapping strategies between S tiers and meta",
        ),
    ],
)
def test_run_early_return_branches(
    tmp_path: Path,
    monkeypatch,
    s_tiers: dict[str, str],
    meta_paths: list[Path],
    meta_frame: pd.DataFrame,
    expected_reason: str,
) -> None:
    cfg = _cfg(tmp_path)
    s_path = cfg.post_h2h_stage_dir / "h2h_s_tiers.json"
    s_path.parent.mkdir(parents=True, exist_ok=True)
    s_path.write_text('{"1": "S"}')

    reasons: list[str] = []

    class _FakeStageLog:
        def start(self) -> None:
            return

        def missing_input(self, reason: str, **_kwargs: object) -> None:
            reasons.append(reason)

    monkeypatch.setattr(h2h_tier_trends, "stage_logger", lambda *_args, **_kwargs: _FakeStageLog())
    monkeypatch.setattr(h2h_tier_trends, "_load_s_tiers", lambda _path: s_tiers)
    monkeypatch.setattr(h2h_tier_trends, "_collect_meta_paths", lambda _cfg: meta_paths)
    monkeypatch.setattr(h2h_tier_trends, "_load_meta_frames", lambda _paths: meta_frame)
    monkeypatch.setattr(
        h2h_tier_trends,
        "write_parquet_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not write")),
    )

    h2h_tier_trends.run(cfg)

    assert reasons == [expected_reason]


def test_run_force_true_bypasses_up_to_date_short_circuit(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)

    s_path = cfg.post_h2h_stage_dir / "h2h_s_tiers.json"
    s_path.parent.mkdir(parents=True, exist_ok=True)
    s_path.write_text('{"1": "S"}')
    meta_path = cfg.meta_input_path(2, h2h_tier_trends.META_TEMPLATE.format(players=2))
    _write_meta(
        meta_path,
        [{"strategy_id": "1", "players": 2, "win_rate": 0.6, "se": 0.1, "ci_lo": 0.4, "ci_hi": 0.8}],
    )

    monkeypatch.setattr(h2h_tier_trends, "stage_is_up_to_date", lambda *a, **k: True)

    calls: list[Path] = []
    original_write = h2h_tier_trends.write_parquet_atomic

    def _record_write(table, out_path: Path, codec: Any) -> None:
        calls.append(out_path)
        original_write(table, out_path, codec=codec)

    monkeypatch.setattr(h2h_tier_trends, "write_parquet_atomic", _record_write)

    h2h_tier_trends.run(cfg, force=True)

    assert len(calls) == 1
    assert calls[0] == cfg.stage_dir("h2h_tier_trends") / h2h_tier_trends.OUTPUT_PARQUET


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
