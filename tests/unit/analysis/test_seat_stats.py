import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis import seat_stats
from farkle.config import AppConfig


def _write_parquet(path: Path, data: dict[str, list]) -> None:
    table = pa.Table.from_pydict(data)
    pq.write_table(table, path)


def test_compute_seat_metrics_returns_empty_when_no_columns(tmp_path: Path) -> None:
    source = tmp_path / "combined.parquet"
    _write_parquet(source, {"unrelated": [1, 2, 3]})

    cfg = seat_stats.SeatMetricConfig(seat_range=(1, 2))
    df = seat_stats.compute_seat_metrics(source, cfg)

    assert df.empty
    assert list(df.columns) == [
        "strategy",
        "seat",
        "n_players",
        "games",
        "wins",
        "win_rate",
        "mean_score",
        "mean_farkles",
        "mean_rounds",
    ]


def test_compute_seat_metrics_aggregates_expected_columns(tmp_path: Path) -> None:
    source = tmp_path / "combined.parquet"
    _write_parquet(
        source,
        {
            "winner_seat": ["P1", "P2"],
            "seat_ranks": [["P1", "P2"], ["P2", "P1"]],
            "P1_strategy": ["Aggro", "Aggro"],
            "P1_score": [100, 200],
            "P1_farkles": [1, 2],
            "P1_rounds": [10, 12],
            "P2_strategy": ["Control", "Control"],
            "P2_score": [150, 110],
            "P2_farkles": [3, 1],
            "P2_rounds": [11, 9],
        },
    )

    cfg = seat_stats.SeatMetricConfig(seat_range=(1, 2))
    df = seat_stats.compute_seat_metrics(source, cfg)

    aggro = df[df["strategy"] == "Aggro"].iloc[0]
    control = df[df["strategy"] == "Control"].iloc[0]

    assert aggro["wins"] == 1
    assert aggro["mean_score"] == pytest.approx(150.0)
    assert control["n_players"] == 2


def test_compute_symmetry_checks_missing_columns_returns_empty(tmp_path: Path, caplog) -> None:
    source = tmp_path / "combined.parquet"
    _write_parquet(source, {"winner_strategy": ["A"]})

    cfg = seat_stats.SeatMetricConfig(seat_range=(1, 2))
    df = seat_stats.compute_symmetry_checks(source, cfg)

    assert df.empty
    matching = [rec for rec in caplog.records if "required columns missing" in rec.message]
    assert matching
    record = matching[0]
    assert str(source) == record.__dict__["curated_path"]
    assert "available_columns_sample" in record.__dict__


def test_compute_symmetry_checks_no_warning_when_columns_present(tmp_path: Path, caplog) -> None:
    source = tmp_path / "combined.parquet"
    _write_parquet(
        source,
        {
            "P1_strategy": ["Aggro"],
            "P2_strategy": ["Aggro"],
            "P1_farkles": [1],
            "P2_farkles": [1],
            "P1_rounds": [10],
            "P2_rounds": [10],
            "n_players": [2],
        },
    )

    cfg = seat_stats.SeatMetricConfig(seat_range=(1, 2))
    df = seat_stats.compute_symmetry_checks(source, cfg)

    assert not df.empty
    assert not any("required columns missing" in rec.message for rec in caplog.records)


def test_compute_seat_advantage_builds_deltas(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path
    combined = tmp_path / "combined.parquet"
    _write_parquet(
        combined,
        {
            "winner_seat": ["P1", "P1", "P2"],
            "seat_ranks": [["P1", "P2"], ["P1", "P2"], ["P2", "P1"]],
        },
    )

    manifest1 = cfg.manifest_for(1)
    manifest1.parent.mkdir(parents=True, exist_ok=True)
    manifest1.write_text(json.dumps({"row_count": 3}))

    manifest2 = cfg.manifest_for(2)
    manifest2.parent.mkdir(parents=True, exist_ok=True)
    manifest2.write_text(json.dumps({"row_count": 1}))

    seat_cfg = seat_stats.SeatMetricConfig(seat_range=(1, 2))
    df = seat_stats.compute_seat_advantage(cfg, combined, seat_cfg)

    assert set(df["seat"]) == {1, 2}
    seat1_rate = df.loc[df["seat"] == 1, "win_rate"].item()
    seat2_rate = df.loc[df["seat"] == 2, "win_rate"].item()
    assert seat1_rate == pytest.approx(2 / 4)  # 3 manifest rows across 1-4 players
    assert seat2_rate == pytest.approx(1 / 1)
    assert df.loc[df["seat"] == 2, "win_rate_delta_prev"].item() == pytest.approx(
        seat2_rate - seat1_rate
    )
