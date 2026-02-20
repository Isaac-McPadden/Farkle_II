import logging
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.diagnostic_fixtures import build_curated_fixture

from farkle.analysis import rng_diagnostics
from farkle.analysis.stage_registry import resolve_interseed_stage_layout
from farkle.utils.types import Compression


def test_collect_diagnostics_empty_input():
    empty = pd.DataFrame(columns=["strategy", "n_players", "win_indicator", "n_rounds", "game_seed", "matchup"])
    result = rng_diagnostics._collect_diagnostics(empty, lags=(1, 2))

    assert result.empty


def test_collect_diagnostics_all_modes_deterministic_statistics():
    melted = pd.DataFrame(
        {
            "strategy": ["A"] * 6,
            "n_players": [2] * 6,
            "win_indicator": [1, 0, 1, 0, 1, 0],
            "n_rounds": [2, 4, 6, 8, 10, 12],
            "game_seed": [1, 2, 3, 4, 5, 6],
            "matchup": ["A | B"] * 6,
        }
    )

    result = rng_diagnostics._collect_diagnostics(melted, lags=(1, 2))

    assert set(result["summary_level"].unique()) == {"strategy", "matchup_strategy"}
    assert set(result["metric"].unique()) == {"win_indicator", "n_rounds"}
    assert set(result["lag"].unique()) == {1, 2}

    stderr = 1.0 / (6**0.5)
    for summary_level in ("strategy", "matchup_strategy"):
        for lag, expected in ((1, -1.0), (2, 1.0)):
            row = result[
                (result["summary_level"] == summary_level)
                & (result["strategy"] == "A")
                & (result["metric"] == "win_indicator")
                & (result["lag"] == lag)
            ].iloc[0]
            assert row["autocorr"] == pytest.approx(expected, abs=1e-12)
            assert row["ci_lower"] == pytest.approx(expected - 1.96 * stderr, abs=1e-12)
            assert row["ci_upper"] == pytest.approx(expected + 1.96 * stderr, abs=1e-12)

        rounds = result[
            (result["summary_level"] == summary_level)
            & (result["strategy"] == "A")
            & (result["metric"] == "n_rounds")
        ]
        assert all(value == pytest.approx(1.0, abs=1e-12) for value in rounds["autocorr"].tolist())


def test_run_skips_when_missing_columns(tmp_path):
    cfg, _, _ = build_curated_fixture(tmp_path)
    cfg.io.interseed_input_dir = tmp_path / "interseed"
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))
    curated = cfg.curated_parquet
    curated.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pydict({"only": [1, 2, 3]}),
        curated,
        compression=cast(Compression, cfg.parquet_codec),
    )

    rng_diagnostics.run(cfg, lags=(1,))

    assert not cfg.rng_output_path("rng_diagnostics.parquet").exists()


@pytest.mark.parametrize("winner_mode", ["winner_strategy", "winner_seat"])
def test_run_emits_artifact_for_winner_modes(tmp_path, winner_mode):
    cfg, _, _ = build_curated_fixture(tmp_path)
    cfg.io.interseed_input_dir = tmp_path / "interseed"
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))

    curated = cfg.curated_parquet
    curated.parent.mkdir(parents=True, exist_ok=True)

    base = {
        "game_seed": [1, 2, 3, 4, 5, 6],
        "n_rounds": [2, 4, 6, 8, 10, 12],
        "P1_strategy": ["A"] * 6,
        "P2_strategy": ["B"] * 6,
    }
    if winner_mode == "winner_strategy":
        base["winner_strategy"] = ["A", "B", "A", "B", "A", "B"]
    else:
        base["winner_seat"] = ["P1", "P2", "P1", "P2", "P1", "P2"]

    pq.write_table(
        pa.Table.from_pydict(base),
        curated,
        compression=cast(Compression, cfg.parquet_codec),
    )

    rng_diagnostics.run(cfg, lags=(1, 2), force=True)

    out = cfg.rng_output_path("rng_diagnostics.parquet")
    assert out.exists()
    diagnostics = pq.read_table(out).to_pandas()
    assert set(diagnostics["summary_level"].unique()) == {"strategy", "matchup_strategy"}
    assert set(diagnostics["metric"].unique()) == {"win_indicator", "n_rounds"}


def test_run_missing_curated_file_logs_and_emits_no_artifact(tmp_path, caplog):
    cfg, _, _ = build_curated_fixture(tmp_path)
    cfg.io.interseed_input_dir = tmp_path / "interseed"
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))
    curated = cfg.curated_parquet
    if curated.exists():
        curated.unlink()

    caplog.set_level(logging.INFO)
    rng_diagnostics.run(cfg, lags=(1,), force=True)

    assert not cfg.rng_output_path("rng_diagnostics.parquet").exists()
    assert any(
        getattr(rec, "stage", None) == "rng_diagnostics"
        and getattr(rec, "reason", None) == "missing curated parquet"
        for rec in caplog.records
    )


def test_run_interseed_not_ready_short_circuits(tmp_path, caplog):
    cfg, _, _ = build_curated_fixture(tmp_path)
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))
    cfg.io.interseed_input_dir = None

    caplog.set_level(logging.INFO)
    rng_diagnostics.run(cfg, lags=(1,), force=True)

    assert not cfg.rng_output_path("rng_diagnostics.parquet").exists()
    assert any(
        getattr(rec, "stage", None) == "rng_diagnostics"
        and "interseed inputs missing" in str(getattr(rec, "reason", ""))
        for rec in caplog.records
    )


def test_run_invalid_lags_short_circuits(tmp_path, caplog):
    cfg, _, _ = build_curated_fixture(tmp_path)
    cfg.io.interseed_input_dir = tmp_path / "interseed"
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))

    caplog.set_level(logging.INFO)
    rng_diagnostics.run(cfg, lags=(0, -5), force=True)

    assert not cfg.rng_output_path("rng_diagnostics.parquet").exists()
    assert any(
        getattr(rec, "stage", None) == "rng_diagnostics"
        and getattr(rec, "reason", None) == "no valid lags provided"
        for rec in caplog.records
    )


def test_run_missing_seat_strategy_columns_logs_and_skips(tmp_path, caplog):
    cfg, _, _ = build_curated_fixture(tmp_path)
    cfg.io.interseed_input_dir = tmp_path / "interseed"
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))
    curated = cfg.curated_parquet
    curated.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pydict(
            {
                "game_seed": [1, 2],
                "n_rounds": [7, 8],
                "winner_strategy": ["A", "B"],
            }
        ),
        curated,
    )

    caplog.set_level(logging.INFO)
    rng_diagnostics.run(cfg, lags=(1,), force=True)

    assert not cfg.rng_output_path("rng_diagnostics.parquet").exists()
    assert any(
        getattr(rec, "stage", None) == "rng_diagnostics"
        and getattr(rec, "reason", None) == "curated parquet missing seat strategy columns"
        for rec in caplog.records
    )


def test_run_up_to_date_short_circuit_skips_dataset_read(tmp_path, monkeypatch, caplog):
    cfg, _, _ = build_curated_fixture(tmp_path)
    cfg.io.interseed_input_dir = tmp_path / "interseed"
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))

    curated = cfg.curated_parquet
    curated.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pydict(
            {
                "game_seed": [1, 2, 3, 4, 5, 6],
                "n_rounds": [2, 4, 6, 8, 10, 12],
                "winner_strategy": ["A", "B", "A", "B", "A", "B"],
                "P1_strategy": ["A"] * 6,
                "P2_strategy": ["B"] * 6,
            }
        ),
        curated,
    )

    rng_diagnostics.run(cfg, lags=(1, 2), force=True)
    assert cfg.rng_output_path("rng_diagnostics.parquet").exists()

    def _raise_if_called(_: Path):
        raise AssertionError("dataset read should not run when done stamp is up-to-date")

    monkeypatch.setattr(rng_diagnostics.ds, "dataset", _raise_if_called)
    caplog.set_level(logging.INFO)
    rng_diagnostics.run(cfg, lags=(1, 2), force=False)

    assert any("rng-diagnostics: up-to-date" in rec.getMessage() for rec in caplog.records)


def test_run_malformed_schema_logs_and_emits_no_artifact(tmp_path, caplog):
    cfg, _, _ = build_curated_fixture(tmp_path)
    cfg.io.interseed_input_dir = tmp_path / "interseed"
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))

    curated = cfg.curated_parquet
    curated.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict({"game_seed": [1, 2], "only": [0, 1]}), curated)

    caplog.set_level(logging.INFO)
    rng_diagnostics.run(cfg, lags=(1,), force=True)

    assert not cfg.rng_output_path("rng_diagnostics.parquet").exists()
    assert any(
        getattr(rec, "stage", None) == "rng_diagnostics"
        and getattr(rec, "reason", None) == "curated parquet missing required columns"
        for rec in caplog.records
    )


def test_run_insufficient_sample_size_logs_and_emits_no_artifact(tmp_path, caplog):
    cfg, _, _ = build_curated_fixture(tmp_path)
    cfg.io.interseed_input_dir = tmp_path / "interseed"
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))

    curated = cfg.curated_parquet
    curated.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pydict(
            {
                "game_seed": [1, 2],
                "n_rounds": [10, 11],
                "winner_strategy": ["A", "B"],
                "P1_strategy": ["A", "A"],
                "P2_strategy": ["B", "B"],
            }
        ),
        curated,
    )

    caplog.set_level(logging.INFO)
    rng_diagnostics.run(cfg, lags=(3,), force=True)

    assert not cfg.rng_output_path("rng_diagnostics.parquet").exists()
    assert any(
        getattr(rec, "stage", None) == "rng_diagnostics"
        and getattr(rec, "reason", None) == "no diagnostics computed"
        for rec in caplog.records
    )


def test_collect_diagnostics_deterministic_sort(tmp_path):
    cfg, combined, _ = build_curated_fixture(tmp_path)
    table = pq.read_table(combined)
    df = table.to_pandas()
    df["matchup"] = df[["P1_strategy", "P2_strategy"]].astype(str).agg(" vs ".join, axis=1)
    df["n_players"] = 2
    df["strategy"] = df["P1_strategy"]
    df["win_indicator"] = (df["winner_strategy"] == df["P1_strategy"]).astype(int)
    df = df[["strategy", "n_players", "win_indicator", "n_rounds", "game_seed", "matchup"]].copy()

    diag = rng_diagnostics._collect_diagnostics(df, lags=(1,))

    assert set(diag["summary_level"].unique()) == {"strategy", "matchup_strategy"}
    assert diag.iloc[0]["lag"] == 1


def test_normalize_lags_and_winner_resolution():
    assert rng_diagnostics._normalize_lags([3, 1, -1, 1]) == (1, 3)

    df = pd.DataFrame(
        {
            "winner_seat": ["P2", "P1", "P9", "PX", pd.NA],
            "P1_strategy": ["X", "Y", "A", "B", "C"],
            "P2_strategy": ["Z", "Z", "D", "E", "F"],
        }
    )
    resolved = rng_diagnostics._winner_strategies(
        df,
        winner_col="winner_seat",
        strat_cols=["P1_strategy", "P2_strategy"],
    )

    assert resolved.tolist() == ["Z", "Y", pd.NA, pd.NA, pd.NA]


def test_winner_column_and_winner_strategy_edge_cases():
    assert rng_diagnostics._winner_column({"winner_strategy", "winner_seat"}) == "winner_strategy"
    assert rng_diagnostics._winner_column({"winner_seat"}) == "winner_seat"
    assert rng_diagnostics._winner_column({"P1_strategy"}) is None

    direct = pd.DataFrame({"winner_strategy": ["A", "B"]})
    assert rng_diagnostics._winner_strategies(direct, "winner_strategy", ["P1_strategy"]).tolist() == [
        "A",
        "B",
    ]

    seat_df = pd.DataFrame({"winner_seat": ["P1", "P2"], "other": [1, 2]})
    none_cols = rng_diagnostics._winner_strategies(seat_df, "winner_seat", ["other"])
    assert none_cols.tolist() == [pd.NA, pd.NA]

    unresolved = pd.DataFrame(
        {
            "winner_seat": ["P7", "bad"],
            "P1_strategy": ["A", "C"],
            "P2_strategy": ["B", "D"],
        }
    )
    unresolved_winners = rng_diagnostics._winner_strategies(
        unresolved,
        "winner_seat",
        ["P1_strategy", "P2_strategy"],
    )
    assert unresolved_winners.tolist() == [pd.NA, pd.NA]


def test_seat_strategy_columns_excludes_winner_strategy_and_sorts(tmp_path):
    cfg, _, _ = build_curated_fixture(tmp_path)

    cols = rng_diagnostics._seat_strategy_columns(
        cfg,
        ["winner_strategy", "P10_strategy", "P2_strategy", "P1_strategy", "other"],
    )

    assert cols == ["P1_strategy", "P2_strategy", "P10_strategy"]


def test_build_matchup_labels_handles_mixed_player_counts_and_nulls():
    df = pd.DataFrame(
        {
            "game_seed": [1001, 1002, 1003],
            "P1_strategy": ["Alpha", "Bravo", "Delta"],
            "P2_strategy": ["Charlie", pd.NA, "Echo"],
            "P3_strategy": [pd.NA, pd.NA, "Foxtrot"],
        }
    )

    labels = rng_diagnostics._build_matchup_labels(df, ["P1_strategy", "P2_strategy", "P3_strategy"])

    assert labels.tolist() == [
        "Alpha | Charlie",
        "Bravo",
        "Delta | Echo | Foxtrot",
    ]


def test_build_matchup_labels_empty_paths_and_strategy_parser_error():
    no_valid_cols = pd.DataFrame({"x": [1, 2]})
    labels = rng_diagnostics._build_matchup_labels(no_valid_cols, ["x", "y"])
    assert labels.tolist() == [pd.NA, pd.NA]

    all_nulls = pd.DataFrame({"P1_strategy": [pd.NA, pd.NA]})
    null_labels = rng_diagnostics._build_matchup_labels(all_nulls, ["P1_strategy"])
    assert null_labels.tolist() == [pd.NA, pd.NA]

    with pytest.raises(ValueError, match="invalid seat strategy column"):
        rng_diagnostics._seat_number_from_strategy_column("seat_1")


def test_group_diagnostics_edge_cases_and_stamp_lifecycle(tmp_path):
    short_group = pd.DataFrame(
        {
            "game_seed": [1, 2],
            "win_indicator": [1, 0],
            "n_rounds": [5, 6],
        }
    )
    assert not rng_diagnostics._group_diagnostics(
        short_group,
        lags=(3,),
        summary_level="strategy",
        strategy="A",
        n_players=2,
    )

    constant_group = pd.DataFrame(
        {
            "game_seed": [3, 2, 1],
            "win_indicator": [1, 1, 1],
            "n_rounds": [10, 10, 10],
        }
    )
    assert not rng_diagnostics._group_diagnostics(
        constant_group,
        lags=(1,),
        summary_level="strategy",
        strategy="B",
        n_players=2,
    )

    input_path = tmp_path / "in.txt"
    output_path = tmp_path / "out.txt"
    stamp_path = tmp_path / "stamp.json"
    input_path.write_text("input")
    output_path.write_text("output")

    stamp = rng_diagnostics._stamp(input_path)
    assert set(stamp) == {"mtime", "size"}
    assert stamp["size"] == input_path.stat().st_size

    rng_diagnostics._write_stamp(
        stamp_path,
        inputs=[input_path],
        outputs=[output_path],
        lags=(1, 2),
        config_sha="abc",
    )
    assert rng_diagnostics._is_up_to_date(
        stamp_path,
        inputs=[input_path],
        outputs=[output_path],
        lags=(1, 2),
        config_sha="abc",
    )

    assert not rng_diagnostics._is_up_to_date(
        stamp_path,
        inputs=[input_path],
        outputs=[output_path],
        lags=(2, 3),
        config_sha="abc",
    )
    assert not rng_diagnostics._is_up_to_date(
        stamp_path,
        inputs=[input_path],
        outputs=[output_path],
        lags=(1, 2),
        config_sha="def",
    )

    input_path.write_text("input-changed")
    assert not rng_diagnostics._is_up_to_date(
        stamp_path,
        inputs=[input_path],
        outputs=[output_path],
        lags=(1, 2),
        config_sha="abc",
    )

    input_path.write_text("input")
    rng_diagnostics._write_stamp(
        stamp_path,
        inputs=[input_path],
        outputs=[output_path],
        lags=(1, 2),
        config_sha="abc",
    )
    output_path.unlink()
    assert not rng_diagnostics._is_up_to_date(
        stamp_path,
        inputs=[input_path],
        outputs=[output_path],
        lags=(1, 2),
        config_sha="abc",
    )

    output_path.write_text("output")
    stamp_path.write_text("not-json")
    assert not rng_diagnostics._is_up_to_date(
        stamp_path,
        inputs=[input_path],
        outputs=[output_path],
        lags=(1, 2),
        config_sha="abc",
    )
