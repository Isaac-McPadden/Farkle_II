from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis import coverage_by_k
from farkle.analysis.stage_state import stage_done_path, write_stage_done
from farkle.config import AnalysisConfig, AppConfig, IOConfig, SimConfig


def _cfg(tmp_path: Path, **kwargs: object) -> AppConfig:
    sim = kwargs.pop("sim", SimConfig(n_players_list=[2, 3], seed=7, seed_list=[7, 8]))
    analysis = kwargs.pop("analysis", AnalysisConfig())
    io = kwargs.pop("io", IOConfig(results_dir_prefix=tmp_path))
    return AppConfig(io=io, sim=sim, analysis=analysis, **kwargs)


def test_pandas_scalar_to_int_cases() -> None:
    assert coverage_by_k._pandas_scalar_to_int(pa.scalar(11)) == 11
    assert coverage_by_k._pandas_scalar_to_int(pd.NA) is None
    assert coverage_by_k._pandas_scalar_to_int(object()) is None
    assert coverage_by_k._pandas_scalar_to_int(3.0) == 3
    assert coverage_by_k._pandas_scalar_to_int(3.25) is None


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
    cfg = _cfg(tmp_path, sim=SimConfig(n_players_list=[3, "x", 0, -2, 2.0, 3, 5]))
    assert coverage_by_k._player_counts_from_config(cfg) == [2, 3, 5]


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
