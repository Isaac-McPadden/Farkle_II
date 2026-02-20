from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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


def test_build_frequentist_tiers_handles_ties_and_no_significant_differences() -> None:
    tied = pd.Series([0.72, 0.72, 0.70], index=[30, 10, 20], name="weighted")
    tied_out = tiering_report._build_frequentist_tiers(tied, mdd=0.01)

    assert tied_out["strategy"].tolist() == [30, 10, 20]
    assert tied_out["mdd_tier"].tolist() == [1, 1, 2]

    # Large MDD means all pairwise differences are practically insignificant.
    no_diff = pd.Series([0.80, 0.77, 0.74], index=[1, 2, 3], name="weighted")
    no_diff_out = tiering_report._build_frequentist_tiers(no_diff, mdd=0.25)
    assert no_diff_out["mdd_tier"].tolist() == [1, 1, 1]


def test_build_report_stable_with_tier_ties_and_missing_ts_entries() -> None:
    freq_df = pd.DataFrame(
        [
            {"strategy": 5, "win_rate": 0.70, "mdd_tier": 1},
            {"strategy": 1, "win_rate": 0.70, "mdd_tier": 1},
            {"strategy": 9, "win_rate": 0.60, "mdd_tier": 2},
        ]
    )

    report = tiering_report._build_report(freq_df, ts_tiers={5: 1, 9: 2})

    # Keeps deterministic row ordering and fills missing TS tiers deterministically.
    id_col = "strategy" if "strategy" in report.columns else "index"
    assert report[id_col].tolist() == [1, 5, 9]
    assert report["trueskill_tier"].tolist() == [3, 1, 2]
    assert report["delta_tier"].tolist() == [-2, 0, 0]
    assert report["in_mdd_top"].tolist() == [True, True, False]
    assert report["in_ts_top"].tolist() == [False, True, False]


def test_write_frequentist_scores_writes_parquet_for_full_and_partial_inputs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    full_tiers = pd.DataFrame(
        [
            {"strategy": 1, "win_rate": 0.7, "mdd_tier": 1},
            {"strategy": 2, "win_rate": 0.5, "mdd_tier": 2},
        ]
    )
    winrates = pd.Series({1: 0.7, 2: 0.5}, name="weighted")
    full_per_k = pd.DataFrame(
        [
            {"strategy": 1, "n_players": 2, "games": 10.0, "win_rate": 0.7},
            {"strategy": 2, "n_players": 2, "games": 12.0, "win_rate": 0.5},
        ]
    )

    tiering_report._write_frequentist_scores(
        cfg,
        full_tiers,
        winrates,
        full_per_k,
        weights_by_k={2: 1.0},
    )

    score_path = cfg.tiering_stage_dir / "frequentist_scores_k_weighted.parquet"
    provenance_path = cfg.tiering_stage_dir / "tiering_k_weighted_provenance.json"
    written = pd.read_parquet(score_path).sort_values(["players", "strategy"]).reset_index(drop=True)
    assert written.columns.tolist() == ["strategy", "players", "win_rate", "tier", "mdd_tier"]
    assert written["players"].tolist() == [0, 0, 2, 2]
    assert written["tier"].tolist() == [1, 2, 1, 2]

    provenance = json.loads(provenance_path.read_text())
    assert provenance["weight_source"] == "config:tiering_weights_by_k"
    assert provenance["normalized_weights_by_k"] == {"2": 1.0}

    # Re-run with a partial per-k frame (missing optional columns like games) and no explicit weights.
    partial_per_k = pd.DataFrame(
        [
            {"strategy": 1, "n_players": 2, "win_rate": 0.7},
            {"strategy": 2, "n_players": 3, "win_rate": 0.5},
        ]
    )
    tiering_report._write_frequentist_scores(
        cfg,
        full_tiers,
        winrates,
        partial_per_k,
        weights_by_k=None,
    )
    uniform_provenance = json.loads(provenance_path.read_text())
    assert uniform_provenance["weight_source"] == "uniform_by_k"
    assert uniform_provenance["normalized_weights_by_k"] == {"2": 0.5, "3": 0.5}
    assert "effective_sample_sizes_games_by_k" in uniform_provenance


def test_tiering_artifact_preserves_existing_stage_output_over_legacy(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    stage = cfg.tiering_stage_dir / "tiers.json"
    legacy = cfg.analysis_dir / "tiers.json"
    stage.write_text('{"source": "stage"}')
    legacy.write_text('{"source": "legacy"}')

    resolved = tiering_report._tiering_artifact(cfg, "tiers.json")

    # Existing stage artifact should not be replaced by legacy output.
    assert resolved == stage
    assert json.loads(stage.read_text())["source"] == "stage"


def test_load_isolated_metrics_logs_warning_for_missing_and_partial_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _cfg(tmp_path)
    inputs = tiering_report.TieringInputs(seeds=[1, 2], player_counts=[2, 3], weights_by_k=None, z_star=1.645, min_gap=None)

    seed1_root = tmp_path / "results" / "run_seed_1"
    seed1_root.mkdir(parents=True)

    def _results_dir(_cfg: AppConfig, seed: int) -> Path:
        if seed == 1:
            return seed1_root
        raise FileNotFoundError("missing")

    monkeypatch.setattr(tiering_report, "_results_dir_for_seed", _results_dir)
    monkeypatch.setattr(tiering_report, "prepare_seed_config", lambda cfg, **_k: cfg)

    p2 = tmp_path / "seed1_2p.parquet"
    pd.DataFrame([{"strategy": 1, "n_players": 2, "games": 10, "wins": 6, "win_rate": 0.6}]).to_parquet(p2)

    def _build(_cfg: AppConfig, k: int) -> Path:
        if k == 2:
            return p2
        raise FileNotFoundError("missing 3p")

    monkeypatch.setattr(tiering_report, "build_isolated_metrics", _build)

    with caplog.at_level("WARNING"):
        out = tiering_report._load_isolated_metrics(cfg, inputs)

    assert out.shape[0] == 1
    assert set(out.columns) >= {"strategy", "n_players", "games", "wins", "win_rate", "seed"}
    assert "Skipping seed: results directory not found" in caplog.text
    assert "Missing isolated metrics" in caplog.text


def test_tiering_report_helper_branch_coverage_and_run_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)

    # _prepare_inputs: normalization + integer key coercion + seed/z fallback.
    cfg.sim.seed = 77
    cfg.analysis.tiering_seeds = []
    cfg.sim.n_players_list = [4, 2, 4]
    cfg.analysis.tiering_weights_by_k = {"2": 2.0, 4: 1.0}
    cfg.analysis.tiering_z_star = 0.0
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    prepared = tiering_report._prepare_inputs(cfg)
    assert prepared.seeds == [77]
    assert prepared.player_counts == [2, 4]
    assert prepared.z_star == pytest.approx(1.645)
    assert prepared.weights_by_k is not None
    assert prepared.weights_by_k == {2: pytest.approx(2 / 3), 4: pytest.approx(1 / 3)}

    # _prepare_inputs: zero-sum weights should disable weighted pooling.
    cfg.analysis.tiering_weights_by_k = {2: 0.0, 4: 0.0}
    prepared_zero = tiering_report._prepare_inputs(cfg)
    assert prepared_zero.weights_by_k is None

    # _results_dir_for_seed: existing and missing branches.
    existing = tmp_path / "existing_seed"
    existing.mkdir(parents=True)
    monkeypatch.setattr(tiering_report, "base_results_dir", lambda _cfg: tmp_path / "base")
    monkeypatch.setattr(tiering_report, "resolve_results_dir", lambda _base, _seed: existing)
    assert tiering_report._results_dir_for_seed(cfg, 77) == existing

    missing = tmp_path / "missing_seed"
    monkeypatch.setattr(tiering_report, "resolve_results_dir", lambda _base, _seed: missing)
    with pytest.raises(FileNotFoundError):
        tiering_report._results_dir_for_seed(cfg, 77)

    # direct key/id coercion helpers.
    integral = tiering_report._coerce_strategy_ids(pd.Series([1.0, 2.0]))
    assert str(integral.dtype) == "int64"
    non_integral = tiering_report._coerce_strategy_ids(pd.Series([1.25, 2.0]))
    assert pd.api.types.is_float_dtype(non_integral)

    assert tiering_report._normalize_mapping_key(4) == 4
    assert tiering_report._normalize_mapping_key("5") == 5
    assert tiering_report._normalize_mapping_key("x") == "x"
    marker = object()
    assert tiering_report._normalize_mapping_key(marker) == str(marker)

    coerced_tiers = tiering_report._coerce_tier_keys({"1": 1, 2: 2, "x": 9, marker: 8})
    assert coerced_tiers == {1: 1, 2: 2}

    # _tiering_artifact migration branch: legacy moved into stage directory.
    legacy_path = cfg.analysis_dir / "legacy_only.json"
    legacy_payload = {"source": "legacy"}
    legacy_path.write_text(json.dumps(legacy_payload))
    migrated_path = tiering_report._tiering_artifact(cfg, "legacy_only.json")
    assert migrated_path == cfg.tiering_stage_dir / "legacy_only.json"
    assert json.loads(migrated_path.read_text()) == legacy_payload

    # _write_consolidated_tiers payload content and trueskill passthrough/None behavior.
    captured_payloads: list[dict[str, object]] = []

    def _capture_write(_path: Path, *, trueskill, frequentist):
        captured_payloads.append({"trueskill": trueskill, "frequentist": frequentist})

    monkeypatch.setattr(tiering_report, "write_tier_payload", _capture_write)
    freq_df = pd.DataFrame([{"strategy": 1, "win_rate": 0.7, "mdd_tier": 1}])
    tiering_report._write_consolidated_tiers(
        cfg,
        ts_payload={},
        freq_tiers=freq_df,
        mdd=0.05,
        weights_by_k=None,
    )
    tiering_report._write_consolidated_tiers(
        cfg,
        ts_payload={"tiers": {"1": 1}},
        freq_tiers=freq_df,
        mdd=0.05,
        weights_by_k={2: 1.0},
    )
    assert captured_payloads[0]["trueskill"] is None
    assert captured_payloads[0]["frequentist"] == {"tiers": {1: 1}, "mdd": 0.05}
    assert captured_payloads[1]["trueskill"] == {"tiers": {"1": 1}}
    assert captured_payloads[1]["frequentist"] == {
        "tiers": {1: 1},
        "mdd": 0.05,
        "weights_by_k": {2: 1.0},
    }

    # run() orchestration happy-path: verify transformed args passed through.
    inputs = tiering_report.TieringInputs([11], [2], {2: 1.0}, 1.96, 0.02)
    monkeypatch.setattr(tiering_report, "_prepare_inputs", lambda _cfg: inputs)
    monkeypatch.setattr(
        tiering_report,
        "load_tier_payload",
        lambda _path: {"trueskill": {"tiers": {"1": 1, "x": 99}}},
    )
    monkeypatch.setattr(
        tiering_report,
        "_load_isolated_metrics",
        lambda _cfg, _inputs: pd.DataFrame(
            [{"strategy": "1", "seed": 11, "n_players": 2, "games": 5, "wins": 3, "win_rate": 0.6}]
        ),
    )
    monkeypatch.setattr(
        tiering_report,
        "tiering_ingredients_from_df",
        lambda *_a, **_k: {
            "mdd": 0.07,
            "tau2_sxk": 0.0,
            "components": type("C", (), {"tau2_seed": 0.0, "R": 1, "K": 1})(),
        },
    )
    monkeypatch.setattr(
        tiering_report,
        "_weighted_winrate",
        lambda *_a, **_k: (
            pd.Series({1: 0.6}, name="weighted"),
            pd.DataFrame([{"strategy": 1, "n_players": 2, "games": 5.0, "win_rate": 0.6}]),
        ),
    )

    run_calls: dict[str, object] = {}

    def _capture_build_frequentist(winrates: pd.Series, mdd: float) -> pd.DataFrame:
        run_calls["build_freq"] = {"index": list(winrates.index), "mdd": mdd}
        return pd.DataFrame([{"strategy": 1, "win_rate": 0.6, "mdd_tier": 1}])

    monkeypatch.setattr(tiering_report, "_build_frequentist_tiers", _capture_build_frequentist)
    def _capture_build_report(_freq: pd.DataFrame, ts: dict[int, int]) -> pd.DataFrame:
        run_calls["build_report"] = dict(ts)
        return pd.DataFrame(
            [
                {
                    "strategy": 1,
                    "win_rate": 0.6,
                    "mdd_tier": 1,
                    "trueskill_tier": 1,
                    "delta_tier": 0,
                    "in_mdd_top": True,
                    "in_ts_top": True,
                }
            ]
        )

    monkeypatch.setattr(tiering_report, "_build_report", _capture_build_report)
    monkeypatch.setattr(
        tiering_report,
        "_write_outputs",
        lambda _cfg, _report, tier_data, _inputs: run_calls.setdefault("write_outputs", tier_data["mdd"]),
    )
    monkeypatch.setattr(
        tiering_report,
        "_write_consolidated_tiers",
        lambda _cfg, **kwargs: run_calls.setdefault("write_consolidated", kwargs),
    )
    monkeypatch.setattr(
        tiering_report,
        "_write_frequentist_scores",
        lambda _cfg, _tiers, _win, _per_k, **kwargs: run_calls.setdefault("write_scores", kwargs),
    )

    tiering_report.run(cfg)

    assert run_calls["build_freq"] == {"index": [1], "mdd": 0.07}
    assert run_calls["build_report"] == {1: 1}
    assert run_calls["write_outputs"] == 0.07
    consolidated = run_calls["write_consolidated"]
    assert isinstance(consolidated, dict)
    assert consolidated["ts_payload"] == {"tiers": {"1": 1, "x": 99}}
    assert consolidated["mdd"] == 0.07
    assert consolidated["weights_by_k"] == {2: 1.0}
    pd.testing.assert_frame_equal(
        consolidated["freq_tiers"],
        pd.DataFrame([{"strategy": 1, "win_rate": 0.6, "mdd_tier": 1}]),
    )
    assert run_calls["write_scores"] == {"weights_by_k": {2: 1.0}}
