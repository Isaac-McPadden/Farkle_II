import json
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis import metrics
from farkle.analysis.schema import expected_schema_for`r`nfrom farkle.config import AppConfig


def _sample_combined_table() -> pa.Table:
    schema = expected_schema_for(3)
    row_specs = [
        {
            "winner_seat": "P1",
            "winner_strategy": "alpha",
            "seat_ranks": ["P1", "P2", "P3"],
            "winning_score": 100,
            "n_rounds": 10,
            "seat_strategies": {1: "alpha", 2: "beta", 3: "gamma"},
            "seat_rank_values": {1: 1, 2: 2, 3: 3},
        },
        {
            "winner_seat": "P2",
            "winner_strategy": "beta",
            "seat_ranks": ["P2", "P1", "P3"],
            "winning_score": 110,
            "n_rounds": 11,
            "seat_strategies": {1: "alpha", 2: "beta", 3: "gamma"},
            "seat_rank_values": {1: 2, 2: 1, 3: 3},
        },
        {
            "winner_seat": "P1",
            "winner_strategy": "alpha",
            "seat_ranks": ["P1", "P2"],
            "winning_score": 95,
            "n_rounds": 9,
            "seat_strategies": {1: "alpha", 2: "cautious"},
            "seat_rank_values": {1: 1, 2: 2},
        },
    ]

    rows: list[dict[str, object]] = []
    for idx, spec in enumerate(row_specs):
        row: dict[str, object] = {
            "winner_seat": spec["winner_seat"],
            "winner_strategy": spec["winner_strategy"],
            "seat_ranks": spec["seat_ranks"],
            "winning_score": spec["winning_score"],
            "n_rounds": spec["n_rounds"],
        }
        strategies = spec["seat_strategies"]
        rank_values = spec["seat_rank_values"]
        for field in schema:
            if field.name in row:
                continue
            if not field.name.startswith("P"):
                continue
            seat_part, _, suffix = field.name.partition("_")
            seat_idx = int(seat_part[1:])
            present = seat_idx in strategies
            if suffix == "strategy":
                row[field.name] = strategies.get(seat_idx)
            elif suffix == "rank":
                row[field.name] = rank_values.get(seat_idx, 0)
            else:
                if not present:
                    row[field.name] = 0
                elif suffix in {"score", "loss_margin"}:
                    row[field.name] = 100 * seat_idx + idx
                else:
                    row[field.name] = seat_idx + idx
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=schema)


def _prepare_metrics_inputs(tmp_path: Path) -> tuple[AppConfig, Path]:
    cfg = AppConfig()
    cfg.io.results_dir = tmp_path / "results"
    combined_dir = cfg.analysis_dir / "data" / "all_n_players_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    table = _sample_combined_table()
    data_file = combined_dir / "all_ingested_rows.parquet"
    pq.write_table(table, data_file)

    manifests = {2: 1, 3: 2}
    for n_players, row_count in manifests.items():
        manifest = cfg.manifest_for(n_players)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"row_count": row_count}))

    return cfg, data_file


def test_metrics_run_creates_outputs_and_stamp(tmp_path: Path) -> None:
    cfg, data_file = _prepare_metrics_inputs(tmp_path)

    metrics.run(cfg)

    analysis_dir = cfg.analysis_dir
    metrics_path = analysis_dir / cfg.analysis.metrics_filename
    seat_csv = analysis_dir / "seat_advantage.csv"
    seat_parquet = analysis_dir / "seat_advantage.parquet"
    stamp = analysis_dir / (cfg.analysis.metrics_filename + cfg.analysis.done_suffix)

    assert metrics_path.exists()
    assert seat_csv.exists()
    assert seat_parquet.exists()
    assert stamp.exists()

    metrics_df = (
        pq.read_table(metrics_path).to_pandas().sort_values("strategy").reset_index(drop=True)
    )
    expected_rows = [
        {
            "strategy": "alpha",
            "games": 3,
            "wins": 2,
            "win_rate": 2 / 3,
            "expected_score": 65.0,
            "mean_score": 97.5,
            "mean_rounds": 9.5,
        },
        {
            "strategy": "beta",
            "games": 2,
            "wins": 1,
            "win_rate": 0.5,
            "expected_score": 55.0,
            "mean_score": 110.0,
            "mean_rounds": 11.0,
        },
        {
            "strategy": "cautious",
            "games": 1,
            "wins": 0,
            "win_rate": 0.0,
            "expected_score": 0.0,
            "mean_score": None,
            "mean_rounds": None,
        },
        {
            "strategy": "gamma",
            "games": 2,
            "wins": 0,
            "win_rate": 0.0,
            "expected_score": 0.0,
            "mean_score": None,
            "mean_rounds": None,
        },
    ]

    records = metrics_df.to_dict("records")
    assert [rec["strategy"] for rec in records] == [row["strategy"] for row in expected_rows]
    for record, expected in zip(records, expected_rows, strict=True):
        assert int(record["games"]) == expected["games"]
        assert int(record["wins"]) == expected["wins"]
        assert record["win_rate"] == pytest.approx(expected["win_rate"])
        assert record["expected_score"] == pytest.approx(expected["expected_score"])
        if expected["mean_score"] is None:
            assert math.isnan(record["mean_score"])
            assert math.isnan(record["mean_rounds"])
        else:
            assert record["mean_score"] == pytest.approx(expected["mean_score"])
            assert record["mean_rounds"] == pytest.approx(expected["mean_rounds"])

    seat_csv_df = pd.read_csv(seat_csv)
    assert len(seat_csv_df) == 12
    seat1 = seat_csv_df[seat_csv_df["seat"] == 1].iloc[0]
    assert seat1["wins"] == 2
    assert seat1["games_with_seat"] == 3
    assert seat1["win_rate"] == pytest.approx(2 / 3)

    seat4 = seat_csv_df[seat_csv_df["seat"] == 4].iloc[0]
    assert seat4["games_with_seat"] == 0
    assert seat4["win_rate"] == pytest.approx(0.0)

    seat_parquet_df = pq.read_table(seat_parquet).to_pandas()
    pd.testing.assert_frame_equal(
        seat_csv_df.sort_values("seat").reset_index(drop=True),
        seat_parquet_df.sort_values("seat").reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
    )

    stamp_data = json.loads(stamp.read_text())
    assert set(stamp_data["inputs"].keys()) == {str(data_file)}
    input_meta = stamp_data["inputs"][str(data_file)]
    assert input_meta["size"] == data_file.stat().st_size
    assert input_meta["mtime"] == pytest.approx(data_file.stat().st_mtime)

    for output_path in (metrics_path, seat_csv, seat_parquet):
        meta = stamp_data["outputs"][str(output_path)]
        assert meta["size"] == output_path.stat().st_size
        assert meta["mtime"] == pytest.approx(output_path.stat().st_mtime)


def test_metrics_run_short_circuits_when_outputs_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg, data_file = _prepare_metrics_inputs(tmp_path)
    analysis_dir = cfg.analysis_dir

    metrics_path = analysis_dir / cfg.analysis.metrics_filename
    seat_csv = analysis_dir / "seat_advantage.csv"
    seat_parquet = analysis_dir / "seat_advantage.parquet"
    for path in (metrics_path, seat_csv, seat_parquet):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"placeholder for {path.name}")

    stamp = analysis_dir / (cfg.analysis.metrics_filename + cfg.analysis.done_suffix)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(
        json.dumps(
            {
                "inputs": {
                    str(data_file): {
                        "mtime": data_file.stat().st_mtime,
                        "size": data_file.stat().st_size,
                    }
                },
                "outputs": {
                    str(metrics_path): {
                        "mtime": metrics_path.stat().st_mtime,
                        "size": metrics_path.stat().st_size,
                    },
                    str(seat_csv): {
                        "mtime": seat_csv.stat().st_mtime,
                        "size": seat_csv.stat().st_size,
                    },
                    str(seat_parquet): {
                        "mtime": seat_parquet.stat().st_mtime,
                        "size": seat_parquet.stat().st_size,
                    },
                },
            },
            indent=2,
        )
    )

    def _fail(*_args, **_kwargs) -> None:
        raise AssertionError("should not write when outputs up-to-date")

    monkeypatch.setattr(metrics, "_write_parquet", _fail)
    monkeypatch.setattr(metrics, "write_csv_atomic", _fail)
    monkeypatch.setattr(metrics, "write_parquet_atomic", _fail)

    caplog.set_level("INFO")
    metrics.run(cfg)

    assert any("Metrics: outputs up-to-date" in rec.getMessage() for rec in caplog.records)

