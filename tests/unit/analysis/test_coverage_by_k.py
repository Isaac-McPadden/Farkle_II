from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.config_factory import make_test_app_config

from farkle.analysis import coverage_by_k
from farkle.analysis.stage_state import stage_done_path, write_stage_done
from farkle.config import AnalysisConfig, AppConfig, IOConfig, SimConfig


def _cfg(tmp_path: Path, **kwargs: object) -> AppConfig:
    sim_cfg: Any = kwargs.pop("sim", SimConfig(n_players_list=[2, 3], seed=7, seed_list=[7, 8]))
    analysis_cfg: Any = kwargs.pop("analysis", AnalysisConfig())
    io_cfg: Any = kwargs.pop("io", IOConfig(results_dir_prefix=tmp_path))
    return make_test_app_config(
        results_dir_prefix=tmp_path,
        sim=sim_cfg,
        analysis=analysis_cfg,
        io=io_cfg,
        **kwargs,
    )


def test_pandas_scalar_to_int_cases() -> None:
    assert coverage_by_k._pandas_scalar_to_int(pa.scalar(11)) == 11
    assert coverage_by_k._pandas_scalar_to_int(pd.NA) is None
    assert coverage_by_k._pandas_scalar_to_int(3.0) == 3
    assert coverage_by_k._pandas_scalar_to_int(3.25) == 3
    assert coverage_by_k._pandas_scalar_to_int("12") == 12
    assert coverage_by_k._pandas_scalar_to_int(" 13.0 ") == 13
    assert coverage_by_k._pandas_scalar_to_int(object()) is None
    assert coverage_by_k._pandas_scalar_to_int(complex(2, 1)) is None


@pytest.mark.parametrize("setting", [None, "", 0, False])
def test_optional_csv_path_falsey_returns_none(tmp_path: Path, setting: object) -> None:
    cfg = _cfg(tmp_path, analysis=AnalysisConfig(outputs={"coverage_by_k_csv": setting}))
    assert coverage_by_k._optional_csv_path(cfg, tmp_path / "stage") is None


def test_optional_csv_path_relative_and_absolute_and_non_string(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    absolute = tmp_path / "out.csv"

    cfg_rel = _cfg(
        tmp_path,
        analysis=AnalysisConfig(outputs={"coverage_by_k_csv": "nested/coverage.csv"}),
    )
    assert coverage_by_k._optional_csv_path(cfg_rel, stage_dir) == stage_dir / "nested/coverage.csv"

    cfg_abs = _cfg(
        tmp_path,
        analysis=AnalysisConfig(outputs={"coverage_by_k_csv": str(absolute)}),
    )
    assert coverage_by_k._optional_csv_path(cfg_abs, stage_dir) == absolute

    cfg_bool = _cfg(tmp_path, analysis=AnalysisConfig(outputs={"coverage_by_k_csv": True}))
    assert coverage_by_k._optional_csv_path(cfg_bool, stage_dir) == stage_dir / "coverage_by_k.csv"


def test_player_counts_from_config_filters_invalid_and_non_positive(tmp_path: Path) -> None:
    bad_n_players_list: Any = [3, "x", 0, -2, 2.0, 3, 5]
    cfg = _cfg(tmp_path, sim=SimConfig(n_players_list=bad_n_players_list))
    assert coverage_by_k._player_counts_from_config(cfg) == [2, 3, 5]


def test_resolve_isolated_metrics_path_branches(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    preferred = cfg.metrics_isolated_path(2)
    preferred.parent.mkdir(parents=True, exist_ok=True)
    preferred.write_bytes(b"ok")
    assert coverage_by_k._resolve_isolated_metrics_path(cfg, 2) == preferred

    preferred_legacy = cfg.metrics_isolated_path(3)
    legacy = cfg.legacy_metrics_isolated_path(3)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"legacy")
    assert not preferred_legacy.exists()
    assert coverage_by_k._resolve_isolated_metrics_path(cfg, 3) == legacy

    preferred_missing = cfg.metrics_isolated_path(4)
    preferred_missing.parent.mkdir(parents=True, exist_ok=True)
    assert coverage_by_k._resolve_isolated_metrics_path(cfg, 4) == preferred_missing

    missing_parent_path = tmp_path / "missing" / "5p_isolated_metrics.parquet"
    monkey_cfg = _cfg(tmp_path)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(monkey_cfg, "metrics_isolated_path", lambda _k: missing_parent_path)
    monkeypatch.setattr(
        monkey_cfg,
        "legacy_metrics_isolated_path",
        lambda _k: tmp_path / "legacy_missing" / "5p_isolated_metrics.parquet",
    )
    try:
        assert coverage_by_k._resolve_isolated_metrics_path(monkey_cfg, 5) is None
    finally:
        monkeypatch.undo()


def test_map_isolated_paths_uses_resolved_lookup_and_unresolved_fallback(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    k2 = cfg.metrics_isolated_path(2)
    k2.parent.mkdir(parents=True, exist_ok=True)
    k2.write_bytes(b"k2")

    rel_k2 = k2.parent / ".." / k2.parent.name / k2.name

    mapping = coverage_by_k._map_isolated_paths(cfg, [2, 3], [rel_k2])

    assert mapping[2] == rel_k2
    assert mapping[3] == cfg.metrics_isolated_path(3)


def test_coverage_inputs_includes_metrics_existing_isolated_and_deduped_ordered(tmp_path: Path) -> None:
    bad_n_players_list: Any = [3, "bad", 3, 0, -1, 2, 2]
    cfg = _cfg(tmp_path, sim=SimConfig(n_players_list=bad_n_players_list))
    metrics_path = cfg.metrics_input_path()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_bytes(b"metrics")

    iso2 = cfg.metrics_isolated_path(2)
    iso2.parent.mkdir(parents=True, exist_ok=True)
    iso2.write_bytes(b"k2")

    # k=3 resolves to preferred path but file does not exist, so it should be excluded.
    cfg.metrics_isolated_path(3).parent.mkdir(parents=True, exist_ok=True)

    inputs = coverage_by_k._coverage_inputs(cfg, metrics_path)
    assert inputs == [metrics_path, iso2]


def test_stream_metrics_counts_raises_for_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "metrics_missing.parquet"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        coverage_by_k._stream_metrics_counts(path, default_seed=99)


def test_stream_metrics_counts_alt_columns_seed_fallback_and_missing_merge(tmp_path: Path) -> None:
    path = tmp_path / "metrics_alt.parquet"
    pd.DataFrame(
        {
            "players": [2, 2, 3, 3],
            "strategy_id": [1, 2, 3, 4],
            "total_games_strat": [5, 7, 6, 9],
            "missing_before_pad": [1, 4, None, 2],
        }
    ).to_parquet(path, index=False)

    counts = coverage_by_k._stream_metrics_counts(path, default_seed=123)
    counts = counts.sort_values(["seed", "k"]).reset_index(drop=True)

    assert counts.to_dict(orient="records") == [
        {"seed": 123, "k": 2, "games": 12, "strategies": 2, "missing_before_pad": 4},
        {"seed": 123, "k": 3, "games": 15, "strategies": 2, "missing_before_pad": 2},
    ]


def test_stream_metrics_counts_canonical_filters_invalid_and_seed_fallback(tmp_path: Path) -> None:
    path = tmp_path / "metrics_canonical.parquet"
    pq.write_table(
        pa.table(
            {
                "n_players": ["2", "2", "x", "2", "3"],
                "strategy": ["1", "2", "3", "bad", "1"],
                "games": [5, 7, 9, 4, 10],
                "seed": ["bad", "9", "9", "9", None],
            }
        ),
        path,
    )

    counts = coverage_by_k._stream_metrics_counts(path, default_seed=77)
    assert counts.to_dict(orient="records") == [
        {"seed": 9, "k": 2, "games": 7, "strategies": 1, "missing_before_pad": None},
        {"seed": 77, "k": 2, "games": 5, "strategies": 1, "missing_before_pad": None},
        {"seed": 77, "k": 3, "games": 10, "strategies": 1, "missing_before_pad": None},
    ]


def test_stream_metrics_counts_without_optional_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "metrics_no_missing_col.parquet"
    pd.DataFrame(
        {
            "n_players": [2, 2, 2],
            "strategy": [1, 1, 2],
            "games": [1, 3, 6],
            "seed": [4, 4, 4],
        }
    ).to_parquet(path, index=False)

    counts = coverage_by_k._stream_metrics_counts(path, default_seed=42)
    assert counts.to_dict(orient="records") == [
        {"seed": 4, "k": 2, "games": 10, "strategies": 2, "missing_before_pad": None}
    ]


def test_expected_strategies_by_k_prefers_isolated_fallbacks_on_unreadable_and_uses_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, sim=SimConfig(n_players_list=[2, 3, 4]))

    iso2 = cfg.metrics_isolated_path(2)
    iso2.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"strategy": [1, 2, 3]})), iso2)

    iso3 = cfg.metrics_isolated_path(3)
    iso3.parent.mkdir(parents=True, exist_ok=True)
    iso3.write_bytes(b"not parquet")

    calls: list[int] = []

    real_parquet_file = pq.ParquetFile

    def fake_parquet_file(path: Path) -> pq.ParquetFile:
        if Path(path) == iso3:
            raise OSError("cannot read")
        return real_parquet_file(path)

    def fake_grid(**_kwargs: object) -> tuple[None, dict[str, pd.Series]]:
        calls.append(1)
        return None, {"strategy_id": pd.Series([10, 11, 12, 13])}

    monkeypatch.setattr(coverage_by_k.pq, "ParquetFile", fake_parquet_file)
    monkeypatch.setattr(coverage_by_k, "generate_strategy_grid", fake_grid)

    out = coverage_by_k._expected_strategies_by_k(cfg, [2, 3, 4], [iso2, iso3])
    assert out == {2: 3, 3: 4, 4: 4}
    assert len(calls) == 1


def test_build_coverage_handles_empty_counts_and_rollups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, sim=SimConfig(n_players_list=[2, 3], seed=5, seed_list=[5, 6]))

    monkeypatch.setattr(
        coverage_by_k,
        "_stream_metrics_counts",
        lambda *_args, **_kwargs: pd.DataFrame(
            columns=["seed", "k", "games", "strategies", "missing_before_pad"]
        ),
    )
    monkeypatch.setattr(coverage_by_k, "_expected_strategies_by_k", lambda *_args, **_kwargs: {2: 4, 3: 4})

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        out = coverage_by_k._build_coverage(cfg, tmp_path / "metrics.parquet", [])

    assert len(captured) == 0
    assert len(out) == 4
    assert str(out["games"].dtype) == "Int64"
    assert str(out["strategies"].dtype) == "Int64"
    assert set(out["games"]) == {0}
    assert set(out["missing_strategies"]) == {4}
    assert out.groupby("k")["games_per_k"].first().to_dict() == {2: 0, 3: 0}
    assert out.groupby("k")["seeds_present"].first().to_dict() == {2: 2, 3: 2}


def test_build_coverage_non_empty_counts_pads_missing_and_rolls_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, sim=SimConfig(n_players_list=[2, 3], seed=7, seed_list=[7, 8]))

    counts = pd.DataFrame(
        [
            {"seed": 7, "k": 2, "games": 12, "strategies": 2, "missing_before_pad": pd.NA},
            {"seed": 8, "k": 2, "games": 8, "strategies": 1, "missing_before_pad": 5},
        ]
    )
    monkeypatch.setattr(coverage_by_k, "_stream_metrics_counts", lambda *_args, **_kwargs: counts)
    monkeypatch.setattr(coverage_by_k, "_expected_strategies_by_k", lambda *_args, **_kwargs: {2: 5, 3: 4})

    out = coverage_by_k._build_coverage(cfg, tmp_path / "metrics.parquet", [])

    row_8_2 = out[(out["seed"] == 8) & (out["k"] == 2)].iloc[0]
    row_7_3 = out[(out["seed"] == 7) & (out["k"] == 3)].iloc[0]

    assert row_8_2["missing_before_pad"] == 5
    assert row_8_2["missing_strategies"] == 5
    assert row_7_3["strategies"] == 0
    assert row_7_3["missing_strategies"] == 4

    by_k_games = out.groupby("k")["games_per_k"].first().to_dict()
    assert by_k_games == {2: 20, 3: 0}


def test_build_coverage_coerces_non_numeric_counts_without_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, sim=SimConfig(n_players_list=[2], seed=11, seed_list=[11]))

    counts = pd.DataFrame(
        [{"seed": 11, "k": 2, "games": "bad", "strategies": None, "missing_before_pad": pd.NA}]
    )
    monkeypatch.setattr(coverage_by_k, "_stream_metrics_counts", lambda *_args, **_kwargs: counts)
    monkeypatch.setattr(coverage_by_k, "_expected_strategies_by_k", lambda *_args, **_kwargs: {2: 3})

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        out = coverage_by_k._build_coverage(cfg, tmp_path / "metrics.parquet", [])

    assert len(captured) == 0
    row = out.iloc[0]
    assert row["games"] == 0
    assert row["strategies"] == 0
    assert row["estimated_games"] == 0.0
    assert row["missing_strategies"] == 3


def test_build_coverage_infers_k_grid_and_synthesizes_missing_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, sim=SimConfig(n_players_list=[], seed=17, seed_list=[]))
    counts = pd.DataFrame(
        [
            {"seed": 17, "k": 3, "games": 9, "strategies": 2},
            {"seed": 19, "k": 2, "games": 8, "strategies": 2},
        ]
    )

    monkeypatch.setattr(coverage_by_k, "_stream_metrics_counts", lambda *_args, **_kwargs: counts)
    monkeypatch.setattr(coverage_by_k, "_expected_strategies_by_k", lambda *_args, **_kwargs: {2: 3, 3: 4})

    out = coverage_by_k._build_coverage(cfg, tmp_path / "metrics.parquet", [])

    assert out[["k", "seed"]].to_dict(orient="records") == [
        {"k": 2, "seed": 17},
        {"k": 2, "seed": 19},
        {"k": 3, "seed": 17},
        {"k": 3, "seed": 19},
    ]
    assert str(out["missing_before_pad"].dtype) == "Int64"
    assert str(out["games"].dtype) == "Int64"
    assert str(out["strategies"].dtype) == "Int64"
    assert out.groupby("k")["games_per_k"].first().to_dict() == {2: 8, 3: 9}


def test_build_coverage_empty_counts_uses_seed_fallback_when_seed_list_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, sim=SimConfig(n_players_list=[2], seed=31, seed_list=[]))
    monkeypatch.setattr(
        coverage_by_k,
        "_stream_metrics_counts",
        lambda *_args, **_kwargs: pd.DataFrame(columns=["seed", "k", "games", "strategies"]),
    )
    monkeypatch.setattr(coverage_by_k, "_expected_strategies_by_k", lambda *_args, **_kwargs: {2: 2})

    out = coverage_by_k._build_coverage(cfg, tmp_path / "metrics.parquet", [])
    assert out[["seed", "k"]].to_dict(orient="records") == [{"seed": 31, "k": 2}]
    assert out.iloc[0]["seeds_present"] == 1


def test_log_imbalance_warnings_paths(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        coverage_by_k._log_imbalance_warnings(pd.DataFrame())
    assert not caplog.records

    coverage = pd.DataFrame(
        [
            {"k": 2, "seed": 7, "strategies": 4, "games": 100, "missing_strategies": 2},
            {"k": 2, "seed": 8, "strategies": 2, "games": 80, "missing_strategies": 1},
        ]
    )
    caplog.clear()
    with caplog.at_level("WARNING"):
        coverage_by_k._log_imbalance_warnings(coverage)

    messages = [record.message for record in caplog.records]
    assert "Coverage: strategy counts differ across seeds" in messages
    assert "Coverage: game counts differ across seeds" in messages
    assert "Coverage: missing strategies detected" in messages



def test_log_imbalance_warnings_treats_padding_as_informational(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coverage = pd.DataFrame(
        [
            {
                "k": 2,
                "seed": 7,
                "strategies": 4,
                "expected_strategies": 4,
                "games": 100,
                "missing_before_pad": 3,
                "missing_strategies": 3,
            },
            {
                "k": 2,
                "seed": 8,
                "strategies": 4,
                "expected_strategies": 4,
                "games": 100,
                "missing_before_pad": 3,
                "missing_strategies": 3,
            },
        ]
    )

    with caplog.at_level("INFO"):
        coverage_by_k._log_imbalance_warnings(coverage)

    messages = [record.message for record in caplog.records]
    assert "Coverage: missing strategies detected" not in messages
    assert "Coverage: reconciled padded strategies" in messages

def test_run_missing_metrics_input_skips(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    coverage_by_k.run(cfg)
    assert not (cfg.stage_dir("coverage_by_k") / "coverage_by_k.parquet").exists()


def test_run_up_to_date_short_circuits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    metrics_path = cfg.metrics_input_path()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"n_players": [2], "strategy": [1], "games": [1], "seed": [7]}).to_parquet(
        metrics_path,
        index=False,
    )

    monkeypatch.setattr(coverage_by_k, "stage_is_up_to_date", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        coverage_by_k,
        "_build_coverage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should skip")),
    )

    coverage_by_k.run(cfg)


def test_run_empty_coverage_skips_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    metrics_path = cfg.metrics_input_path()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"n_players": [2], "strategy": [1], "games": [1], "seed": [7]}).to_parquet(
        metrics_path,
        index=False,
    )

    monkeypatch.setattr(coverage_by_k, "_build_coverage", lambda *_args, **_kwargs: pd.DataFrame())

    stage_dir = cfg.stage_dir("coverage_by_k")
    done = stage_done_path(stage_dir, "coverage_by_k")

    coverage_by_k.run(cfg)

    assert not done.exists()
    assert not (stage_dir / "coverage_by_k.parquet").exists()


def test_run_success_writes_outputs_and_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, analysis=AnalysisConfig(outputs={"coverage_by_k_csv": True}))
    metrics_path = cfg.metrics_input_path()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"n_players": [2], "strategy": [1], "games": [1], "seed": [7]}).to_parquet(
        metrics_path,
        index=False,
    )

    coverage = pd.DataFrame(
        {
            "k": [2],
            "seed": [7],
            "games": [10],
            "estimated_games": [5.0],
            "games_per_k": [10],
            "estimated_games_per_k": [5.0],
            "strategies": [2],
            "strategies_per_k": [2],
            "expected_strategies": [2],
            "missing_before_pad": [pd.NA],
            "missing_strategies": [0],
            "padded_strategies": [0],
            "seeds_present": [1],
        }
    )
    monkeypatch.setattr(coverage_by_k, "_build_coverage", lambda *_args, **_kwargs: coverage)

    stage_dir = cfg.stage_dir("coverage_by_k")
    done = stage_done_path(stage_dir, "coverage_by_k")

    coverage_by_k.run(cfg)

    parquet_path = stage_dir / "coverage_by_k.parquet"
    csv_path = stage_dir / "coverage_by_k.csv"
    assert parquet_path.exists()
    assert csv_path.exists()
    assert done.exists()

    out = pd.read_parquet(parquet_path)
    assert out[["k", "seed", "games"]].to_dict(orient="records") == [
        {"k": 2, "seed": 7, "games": 10}
    ]


def test_run_force_recomputes_even_if_up_to_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    metrics_path = cfg.metrics_input_path()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"n_players": [2], "strategy": [1], "games": [1], "seed": [7]}).to_parquet(
        metrics_path,
        index=False,
    )

    coverage = pd.DataFrame(
        {
            "k": [2],
            "seed": [7],
            "games": [1],
            "estimated_games": [0.5],
            "games_per_k": [1],
            "estimated_games_per_k": [0.5],
            "strategies": [1],
            "strategies_per_k": [1],
            "expected_strategies": [1],
            "missing_before_pad": [pd.NA],
            "missing_strategies": [0],
            "padded_strategies": [0],
            "seeds_present": [1],
        }
    )
    called = {"build": 0}

    monkeypatch.setattr(coverage_by_k, "stage_is_up_to_date", lambda *_args, **_kwargs: True)

    def _build(*_args: object, **_kwargs: object) -> pd.DataFrame:
        called["build"] += 1
        return coverage

    monkeypatch.setattr(coverage_by_k, "_build_coverage", _build)
    coverage_by_k.run(cfg, force=True)
    assert called["build"] == 1


def test_run_without_csv_output_writes_parquet_and_done_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, analysis=AnalysisConfig(outputs={"coverage_by_k_csv": False}))
    metrics_path = cfg.metrics_input_path()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"n_players": [2], "strategy": [1], "games": [1], "seed": [7]}).to_parquet(
        metrics_path,
        index=False,
    )

    coverage = pd.DataFrame(
        {
            "k": [2],
            "seed": [7],
            "games": [3],
            "estimated_games": [1.5],
            "games_per_k": [3],
            "estimated_games_per_k": [1.5],
            "strategies": [1],
            "strategies_per_k": [1],
            "expected_strategies": [1],
            "missing_before_pad": [pd.NA],
            "missing_strategies": [0],
            "padded_strategies": [0],
            "seeds_present": [1],
        }
    )
    monkeypatch.setattr(coverage_by_k, "_build_coverage", lambda *_args, **_kwargs: coverage)

    coverage_by_k.run(cfg)

    stage_dir = cfg.stage_dir("coverage_by_k")
    assert (stage_dir / "coverage_by_k.parquet").exists()
    assert not (stage_dir / "coverage_by_k.csv").exists()
    assert stage_done_path(stage_dir, "coverage_by_k").exists()


def test_run_up_to_date_with_real_done_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    metrics_path = cfg.metrics_input_path()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"n_players": [2], "strategy": [1], "games": [1], "seed": [7]}).to_parquet(
        metrics_path,
        index=False,
    )

    stage_dir = cfg.stage_dir("coverage_by_k")
    output_parquet = stage_dir / "coverage_by_k.parquet"
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"k": [2], "seed": [7], "games": [1]}).to_parquet(output_parquet, index=False)
    done = stage_done_path(stage_dir, "coverage_by_k")
    write_stage_done(done, inputs=[metrics_path], outputs=[output_parquet], config_sha=cfg.config_sha)

    monkeypatch.setattr(
        coverage_by_k,
        "_build_coverage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should skip real")),
    )

    coverage_by_k.run(cfg)
