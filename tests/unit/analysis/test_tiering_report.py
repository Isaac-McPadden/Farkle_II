from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from farkle.analysis import tiering_report
from farkle.config import AppConfig


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.analysis.outputs = {}
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_weighted_winrate_with_grouping_weights():
    df = pd.DataFrame(
        [
            {"strategy": "s1", "n_players": 2, "games": 10, "wins": 6, "win_rate": 0.6},
            {"strategy": "s1", "n_players": 2, "games": 5, "wins": 1, "win_rate": 0.2},
            {"strategy": "s1", "n_players": 3, "games": 0, "wins": 0, "win_rate": 0.5},
            {"strategy": "s2", "n_players": 2, "games": 1, "wins": 1, "win_rate": 1.0},
            {"strategy": "s2", "n_players": 3, "games": 10, "wins": 5, "win_rate": 0.5},
            {"strategy": "s2", "n_players": 3, "games": 10, "wins": 7, "win_rate": 0.7},
        ]
    )

    collapsed, per_k = tiering_report._weighted_winrate(df, {2: 0.6, 3: 0.4})

    expected_per_k = pd.DataFrame(
        [
            {"strategy": "s1", "n_players": 2, "games": 15.0, "win_rate": (0.6 * 10 + 0.2 * 5) / 15, "w_k": 0.6},
            {"strategy": "s1", "n_players": 3, "games": 1.0, "win_rate": 0.5, "w_k": 0.4},
            {"strategy": "s2", "n_players": 2, "games": 1.0, "win_rate": 1.0, "w_k": 0.6},
            {
                "strategy": "s2",
                "n_players": 3,
                "games": 20.0,
                "win_rate": (0.5 * 10 + 0.7 * 10) / 20,
                "w_k": 0.4,
            },
        ]
    )
    expected_per_k["weighted"] = expected_per_k["win_rate"] * expected_per_k["w_k"]

    pd.testing.assert_frame_equal(
        per_k.reset_index(drop=True),
        expected_per_k,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    expected_collapsed = pd.Series(
        {
            "s2": expected_per_k.loc[expected_per_k["strategy"] == "s2", "weighted"].sum(),
            "s1": expected_per_k.loc[expected_per_k["strategy"] == "s1", "weighted"].sum(),
        },
        name="weighted",
    )
    expected_collapsed.index.name = "strategy"
    pd.testing.assert_series_equal(
        collapsed,
        expected_collapsed,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_weighted_winrate_grouping_defaults_for_missing_and_extra_weights():
    df = pd.DataFrame(
        [
            {"strategy": "s1", "n_players": 2, "games": 4, "win_rate": 0.5},
            {"strategy": "s1", "n_players": 3, "games": 6, "win_rate": 0.25},
            {"strategy": "s2", "n_players": 2, "games": 8, "win_rate": 0.75},
            {"strategy": "s2", "n_players": 3, "games": 2, "win_rate": 0.5},
        ]
    )

    collapsed, per_k = tiering_report._weighted_winrate(df, {2: 1.0, 4: 0.5})

    assert (per_k.loc[per_k["n_players"] == 3, "w_k"] == 0.0).all()
    assert (per_k.loc[per_k["n_players"] == 2, "w_k"] == 1.0).all()

    expected = pd.Series({"s2": 0.75, "s1": 0.5}, name="weighted")
    expected.index.name = "strategy"
    pd.testing.assert_series_equal(
        collapsed,
        expected,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_run_skips_when_tiers_or_metrics_missing(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(tiering_report, "load_tier_payload", lambda _path: {})

    called = {"load_metrics": False}

    def _load_metrics(*_args, **_kwargs):
        called["load_metrics"] = True
        return pd.DataFrame()

    monkeypatch.setattr(tiering_report, "_load_isolated_metrics", _load_metrics)
    tiering_report.run(cfg)
    assert called["load_metrics"] is False

    payload = {"trueskill": {"tiers": {"1": 1}}}
    monkeypatch.setattr(tiering_report, "load_tier_payload", lambda _path: payload)
    monkeypatch.setattr(tiering_report, "_load_isolated_metrics", lambda *_a, **_k: pd.DataFrame())
    tiering_report.run(cfg)
    assert not (cfg.tiering_stage_dir / "tiering_report.csv").exists()


def test_run_prefers_trueskill_then_falls_back_to_other_payload_sections(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _cfg(tmp_path)

    monkeypatch.setattr(tiering_report, "_prepare_inputs", lambda _cfg: tiering_report.TieringInputs([1], [2], None, 1.645, None))
    df = pd.DataFrame(
        [
            {"strategy": 1, "n_players": 2, "games": 10, "wins": 6, "win_rate": 0.6, "seed": 1},
        ]
    )
    monkeypatch.setattr(tiering_report, "_load_isolated_metrics", lambda *_a, **_k: df.copy())
    monkeypatch.setattr(
        tiering_report,
        "tiering_ingredients_from_df",
        lambda *_a, **_k: {
            "mdd": 0.05,
            "tau2_sxk": 0.0,
            "components": type("C", (), {"tau2_seed": 0.0, "R": 1, "K": 1})(),
        },
    )
    monkeypatch.setattr(
        tiering_report,
        "_weighted_winrate",
        lambda *_a, **_k: (pd.Series({1: 0.6}), pd.DataFrame([{"strategy": 1, "n_players": 2, "games": 10.0, "win_rate": 0.6}])),
    )
    monkeypatch.setattr(
        tiering_report,
        "_build_frequentist_tiers",
        lambda *_a, **_k: pd.DataFrame([{"strategy": 1, "win_rate": 0.6, "mdd_tier": 1}]),
    )

    captured: list[dict[int, int]] = []
    monkeypatch.setattr(tiering_report, "_build_report", lambda _df, tiers: captured.append(dict(tiers)) or pd.DataFrame([{"strategy": 1, "mdd_tier": 1, "trueskill_tier": 1, "delta_tier": 0, "in_mdd_top": True, "in_ts_top": True, "win_rate": 0.6}]))
    monkeypatch.setattr(tiering_report, "_write_outputs", lambda *_a, **_k: None)
    monkeypatch.setattr(tiering_report, "_write_consolidated_tiers", lambda *_a, **_k: None)
    monkeypatch.setattr(tiering_report, "_write_frequentist_scores", lambda *_a, **_k: None)

    monkeypatch.setattr(tiering_report, "load_tier_payload", lambda _p: {"trueskill": {"tiers": {"1": 1}}, "frequentist": {"tiers": {"1": 5}}})
    tiering_report.run(cfg)

    monkeypatch.setattr(tiering_report, "load_tier_payload", lambda _p: {"frequentist": {"tiers": {"1": 3}}})
    tiering_report.run(cfg)

    assert captured == [{1: 1}, {1: 3}]


def test_write_outputs_serializes_expected_columns_and_sorted_rows(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    report = pd.DataFrame(
        [
            {"strategy": 2, "win_rate": 0.55, "mdd_tier": 2, "trueskill_tier": 1, "delta_tier": 1, "in_mdd_top": False, "in_ts_top": True},
            {"strategy": 1, "win_rate": 0.80, "mdd_tier": 1, "trueskill_tier": 1, "delta_tier": 0, "in_mdd_top": True, "in_ts_top": True},
            {"strategy": 3, "win_rate": 0.60, "mdd_tier": 2, "trueskill_tier": 3, "delta_tier": -1, "in_mdd_top": False, "in_ts_top": False},
        ]
    )
    tier_data = {
        "mdd": 0.05,
        "tau2_sxk": 0.01,
        "components": type("C", (), {"tau2_seed": 0.02, "R": 3, "K": 2})(),
    }
    inputs = tiering_report.TieringInputs([1], [2, 3], None, 1.645, 0.01)

    tiering_report._write_outputs(cfg, report, tier_data, inputs)

    out_csv = cfg.tiering_stage_dir / "tiering_report.csv"
    out_json = cfg.tiering_stage_dir / "tiering_report.json"
    assert out_csv.exists()
    assert out_json.exists()

    written = pd.read_csv(out_csv)
    assert written.columns.tolist() == report.columns.tolist()
    assert written["strategy"].tolist() == [1, 3, 2]

    summary = json.loads(out_json.read_text())
    assert summary["total_strategies"] == 3
    assert summary["disagreements"] == 2
    assert summary["trueskill_z_star"] == 1.645
