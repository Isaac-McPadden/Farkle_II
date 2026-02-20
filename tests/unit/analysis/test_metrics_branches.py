from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.helpers import metrics_samples as sample_data
from tests.helpers.config_factory import make_test_app_config

from farkle.analysis import metrics
from farkle.config import AnalysisConfig, AppConfig, IOConfig, SimConfig


def test_run_raises_when_combined_parquet_missing(tmp_path):
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))

    with pytest.raises(FileNotFoundError, match="metrics: missing combined parquet"):
        metrics.run(cfg)


def test_run_raises_when_no_isolated_metrics_generated(tmp_path, monkeypatch):
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))
    cfg.curated_parquet.parent.mkdir(parents=True, exist_ok=True)
    cfg.curated_parquet.write_text("placeholder")

    monkeypatch.setattr(metrics, "check_pre_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(metrics, "stage_is_up_to_date", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(metrics, "_ensure_isolated_metrics", lambda *_args, **_kwargs: ([], []))

    with pytest.raises(RuntimeError, match="metrics: no isolated metric files generated"):
        metrics.run(cfg)


def test_run_skips_symmetry_when_no_two_player_config(tmp_path, monkeypatch):
    cfg = sample_data.stage_sample_run(tmp_path, refresh_inputs=True)
    cfg.sim.n_players_list = [3]

    stamp_calls: list[tuple[Path, list[Path], list[Path]]] = []
    done_calls: list[tuple[Path, list[Path], list[Path]]] = []

    monkeypatch.setattr(metrics, "check_pre_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(metrics, "stage_is_up_to_date", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        metrics,
        "_ensure_isolated_metrics",
        lambda *_args, **_kwargs: ([cfg.metrics_isolated_path(3)], [cfg.results_root / "3_players" / "3p_metrics.parquet"]),
    )
    monkeypatch.setattr(
        metrics,
        "_collect_metrics_frames",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "strategy": [1],
                "n_players": [3],
                "games": [10],
                "wins": [5],
                "win_rate": [0.5],
                "win_prob": [0.5],
                "expected_score": [12.0],
            }
        ),
    )
    monkeypatch.setattr(metrics, "_compute_weighted_metrics", lambda *_args, **_kwargs: pd.DataFrame({"strategy": [1]}))
    monkeypatch.setattr(metrics, "compute_seat_advantage", lambda *_args, **_kwargs: pd.DataFrame({"seat": [1], "games_with_seat": [1], "wins": [1], "win_rate": [1.0]}))
    monkeypatch.setattr(metrics, "compute_seat_metrics", lambda *_args, **_kwargs: pd.DataFrame({"seat": [1]}))

    def _fail_symmetry(*_args, **_kwargs):
        raise AssertionError("symmetry should be skipped")

    monkeypatch.setattr(metrics, "compute_symmetry_checks", _fail_symmetry)
    monkeypatch.setattr(metrics, "write_parquet_atomic", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(metrics, "write_csv_atomic", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        metrics,
        "_write_stamp",
        lambda path, *, inputs, outputs: stamp_calls.append((path, list(inputs), list(outputs))),
    )
    monkeypatch.setattr(
        metrics,
        "write_stage_done",
        lambda path, *, inputs, outputs, config_sha=None: done_calls.append((path, list(inputs), list(outputs))),
    )

    metrics.run(cfg)

    symmetry_stamp = cfg.metrics_output_path("metrics.symmetry.stamp.json")
    assert any(path == symmetry_stamp and inputs == [] and outputs == [] for path, inputs, outputs in stamp_calls)

    symmetry_done = cfg.metrics_stage_dir / "metrics_symmetry.done.json"
    assert any(path == symmetry_done and inputs == [] and outputs == [] for path, inputs, outputs in done_calls)


def test_run_stage_up_to_date_permutations(tmp_path, monkeypatch):
    cfg = sample_data.stage_sample_run(tmp_path, refresh_inputs=True)

    for n in sorted(set(cfg.sim.n_players_list)):
        cfg.metrics_isolated_path(n).parent.mkdir(parents=True, exist_ok=True)
        cfg.metrics_isolated_path(n).write_text("isolated")

    calls: list[str] = []

    def fake_stage_up_to_date(done_path: Path, *_args, **_kwargs) -> bool:
        mapping = {
            "metrics.done.json": False,
            "metrics_isolated.done.json": True,
            "metrics_core.done.json": True,
            "metrics_weighted.done.json": False,
            "metrics_seat_advantage.done.json": True,
            "metrics_seat_metrics.done.json": False,
            "metrics_symmetry.done.json": True,
        }
        return mapping[done_path.name]

    def fake_read_parquet(path: Path, *_args, **_kwargs) -> pd.DataFrame:
        calls.append(f"read_parquet:{Path(path).name}")
        name = Path(path).name
        if name == "metrics.parquet":
            return pd.DataFrame(
                {
                    "strategy": ["A"],
                    "n_players": [2],
                    "games": [10],
                    "wins": [6],
                    "win_rate": [0.6],
                    "win_prob": [0.6],
                    "expected_score": [15.0],
                }
            )
        if name == "symmetry_checks.parquet":
            return pd.DataFrame({"strategy": ["A"], "n_players": [2], "observations": [1]})
        raise AssertionError(f"unexpected read_parquet path: {path}")

    monkeypatch.setattr(metrics, "check_pre_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(metrics, "stage_is_up_to_date", fake_stage_up_to_date)
    monkeypatch.setattr(metrics.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(metrics.pd, "read_csv", lambda path, *_args, **_kwargs: calls.append(f"read_csv:{Path(path).name}") or pd.DataFrame({"seat": [1], "games_with_seat": [1], "wins": [1], "win_rate": [1.0]}))
    monkeypatch.setattr(metrics, "_compute_weighted_metrics", lambda *_args, **_kwargs: calls.append("compute_weighted") or pd.DataFrame({"strategy": ["A"], "games": [10], "wins": [6]}))
    monkeypatch.setattr(metrics, "compute_seat_metrics", lambda *_args, **_kwargs: calls.append("compute_seat_metrics") or pd.DataFrame({"seat": [1]}))
    monkeypatch.setattr(metrics, "compute_symmetry_checks", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("symmetry compute should not run")))
    monkeypatch.setattr(metrics, "write_parquet_atomic", lambda *_args, **_kwargs: calls.append("write_parquet"))
    monkeypatch.setattr(metrics, "write_csv_atomic", lambda *_args, **_kwargs: calls.append("write_csv"))
    monkeypatch.setattr(metrics, "_write_stamp", lambda *_args, **_kwargs: calls.append("write_stamp"))
    monkeypatch.setattr(metrics, "write_stage_done", lambda *_args, **_kwargs: calls.append("write_done"))

    metrics.run(cfg)

    assert "read_parquet:metrics.parquet" in calls
    assert "compute_weighted" in calls
    assert "read_csv:seat_advantage.csv" in calls
    assert "compute_seat_metrics" in calls
    assert "read_parquet:symmetry_checks.parquet" in calls


def test_normalize_pooling_scheme_aliases_and_invalid():
    assert metrics._normalize_pooling_scheme(" game_count ") == "game-count"
    assert metrics._normalize_pooling_scheme("EQUAL") == "equal-k"
    assert metrics._normalize_pooling_scheme("config-provided") == "config"

    with pytest.raises(ValueError, match="Unknown pooling scheme"):
        metrics._normalize_pooling_scheme("mystery")


def test_pooling_weights_for_metrics_variants_and_warning(caplog):
    frame = pd.DataFrame(
        {
            "n_players": [2, 2, 3],
            "games": [10, 30, 5],
        }
    )

    game_count = metrics._pooling_weights_for_metrics(
        frame, pooling_scheme="game-count", weights_by_k={}
    )
    assert game_count.tolist() == [10.0, 30.0, 5.0]

    equal_k = metrics._pooling_weights_for_metrics(
        frame, pooling_scheme="equal-k", weights_by_k={}
    )
    assert equal_k.tolist() == [0.25, 0.75, 1.0]

    with caplog.at_level("WARNING"):
        config_weights = metrics._pooling_weights_for_metrics(
            frame, pooling_scheme="config", weights_by_k={2: 2.0}
        )
    assert "Missing pooling weights" in caplog.text
    assert config_weights.tolist() == [0.5, 1.5, 0.0]

    with pytest.raises(ValueError, match="Unknown pooling scheme"):
        metrics._pooling_weights_for_metrics(frame, pooling_scheme="invalid", weights_by_k={})


def test_compute_weighted_metrics_empty_missing_config_and_zero_weight_rows():
    cfg_config = make_test_app_config()
    cfg_config.analysis.pooling_weights = "config"
    cfg_config.analysis.pooling_weights_by_k = None
    with pytest.raises(ValueError, match="pooling_weights_by_k must be set"):
        metrics._compute_weighted_metrics(
            pd.DataFrame(
                {
                    "strategy": ["A"],
                    "n_players": [2],
                    "games": [1],
                    "wins": [1],
                    "win_rate": [1.0],
                    "win_prob": [1.0],
                    "expected_score": [1.0],
                }
            ),
            cfg_config,
        )

    out_empty = metrics._compute_weighted_metrics(
        pd.DataFrame(),
        make_test_app_config(analysis=AnalysisConfig(pooling_weights="equal-k", pooling_weights_by_k={})),
    )
    assert out_empty.empty

    frame = pd.DataFrame(
        {
            "strategy": ["A", "B"],
            "n_players": [2, 2],
            "games": [0, 10],
            "wins": [0, 5],
            "win_rate": [0.0, 0.5],
            "win_prob": [0.0, 0.5],
            "expected_score": [0.0, 10.0],
        }
    )
    out = metrics._compute_weighted_metrics(
        frame,
        make_test_app_config(analysis=AnalysisConfig(pooling_weights="game-count", pooling_weights_by_k={})),
    )
    assert out["strategy"].tolist() == ["B"]


def test_weighted_mean_masks_non_finite_and_returns_nan_for_no_valid_samples():
    values = pd.Series([1.0, np.nan, np.inf, 3.0])
    weights = np.array([1.0, 1.0, 2.0, np.inf])
    result = metrics._weighted_mean(values, weights)
    assert result == pytest.approx(1.0)

    no_valid = metrics._weighted_mean(pd.Series([np.nan]), np.array([1.0]))
    assert np.isnan(no_valid)


def test_pooled_value_columns_excludes_non_numeric_and_counters():
    frame = pd.DataFrame(
        {
            "strategy": ["A"],
            "n_players": [2],
            "games": [10],
            "wins": [5],
            "seed": [1],
            "value_count": [4],
            "expected_score": [15.5],
            "win_rate": [0.5],
            "label": ["text"],
            "custom_metric": pd.Series([7], dtype="Int64"),
        }
    )

    cols = metrics._pooled_value_columns(frame)
    assert "expected_score" in cols
    assert "win_rate" in cols
    assert "custom_metric" in cols
    assert "games" not in cols
    assert "value_count" not in cols
    assert "label" not in cols


def test_downcast_metric_counters_handles_fractional_and_bounds():
    frame = pd.DataFrame(
        {
            "games": pd.Series([1.0, 2.0], dtype="float64"),
            "wins": pd.Series([1.5, 2.0], dtype="float64"),
            "n_players": pd.Series([2.0, 3.0], dtype="float64"),
            "seed": pd.Series([2**31, 1], dtype="int64"),
            "turn_count": pd.Series([1.0, 2.0], dtype="float64"),
            "n_events": pd.Series([1.0, 2.0], dtype="float64"),
        }
    )

    out = metrics._downcast_metric_counters(frame)

    assert str(out["games"].dtype).startswith("int")
    assert out["wins"].dtype == np.float64
    assert str(out["n_players"].dtype).startswith("int")
    assert out["seed"].dtype == np.int64
    assert str(out["turn_count"].dtype).startswith("int")
    assert str(out["n_events"].dtype).startswith("int")


def test_write_stamp_includes_only_existing_paths(tmp_path):
    stamp_path = tmp_path / "stamps" / "metrics.stamp.json"
    existing_input = tmp_path / "input.parquet"
    missing_input = tmp_path / "missing.parquet"
    existing_output = tmp_path / "output.parquet"
    missing_output = tmp_path / "missing_out.parquet"

    existing_input.write_text("in")
    existing_output.write_text("out")

    metrics._write_stamp(
        stamp_path,
        inputs=[existing_input, missing_input],
        outputs=[existing_output, missing_output],
    )

    payload = json.loads(stamp_path.read_text())
    assert list(payload["inputs"]) == [str(existing_input)]
    assert list(payload["outputs"]) == [str(existing_output)]


def test_ensure_isolated_metrics_fallback_and_missing_raw_warning(tmp_path, monkeypatch, caplog):
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2, 3, 4]))

    raw2 = cfg.results_root / "2_players" / "2p_metrics.parquet"
    raw3 = cfg.results_root / "3_players" / "3p_metrics.parquet"
    raw2.parent.mkdir(parents=True, exist_ok=True)
    raw3.parent.mkdir(parents=True, exist_ok=True)
    raw2.write_text("raw2")
    raw3.write_text("raw3")

    preferred2 = cfg.metrics_isolated_path(2)
    preferred3 = cfg.metrics_isolated_path(3)
    legacy3 = cfg.legacy_metrics_isolated_path(3)
    preferred2.parent.mkdir(parents=True, exist_ok=True)
    preferred3.parent.mkdir(parents=True, exist_ok=True)
    legacy3.parent.mkdir(parents=True, exist_ok=True)
    preferred2.write_text("preferred")
    legacy3.write_text("legacy")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(metrics, "build_isolated_metrics", _raise)

    with caplog.at_level("WARNING"):
        iso_paths, raw_inputs = metrics._ensure_isolated_metrics(cfg, [2, 3, 4])

    assert iso_paths == [preferred2, legacy3]
    assert raw_inputs == [
        cfg.results_root / "2_players" / "2p_metrics.parquet",
        cfg.results_root / "3_players" / "3p_metrics.parquet",
        cfg.results_root / "4_players" / "4p_metrics.parquet",
    ]
    assert "Expanded metrics missing" in caplog.text
