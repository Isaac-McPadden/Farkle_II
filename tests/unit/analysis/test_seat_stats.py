import json
from pathlib import Path
from typing import Literal

import pandas as pd
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


@pytest.mark.parametrize(
    ("data", "expected_n_players", "expected_rounds"),
    [
        (
            {
                "winner_seat": ["P1", "P2"],
                "n_rounds": [7, 9],
                "P1_strategy": ["Aggro", "Aggro"],
                "P1_score": [25, 30],
                "P1_farkles": [1, 2],
                "P2_strategy": ["Control", "Control"],
                "P2_score": [20, 35],
                "P2_farkles": [2, 3],
            },
            2,
            8.0,
        ),
        (
            {
                "winner_seat": ["P1", "P2"],
                "seat_ranks": [["P1", "P2"], ["P2", "P1"]],
                "n_rounds": [11, 13],
                "P1_strategy": ["Aggro", "Aggro"],
                "P1_score": [80, 120],
                "P1_farkles": [1, 0],
                "P2_strategy": ["Control", "Control"],
                "P2_score": [75, 140],
                "P2_farkles": [2, 1],
            },
            2,
            12.0,
        ),
    ],
)
def test_compute_seat_metrics_fallbacks_for_players_and_rounds(
    tmp_path: Path, data: dict[str, list], expected_n_players: int, expected_rounds: float
) -> None:
    source = tmp_path / "combined.parquet"
    _write_parquet(source, data)

    cfg = seat_stats.SeatMetricConfig(seat_range=(1, 2))
    df = seat_stats.compute_seat_metrics(source, cfg)

    aggro = df[df["strategy"] == "Aggro"].iloc[0]
    assert aggro["n_players"] == expected_n_players
    assert aggro["mean_rounds"] == pytest.approx(expected_rounds)


def test_compute_seat_metrics_all_null_strategy_records_drop_to_empty(tmp_path: Path) -> None:
    source = tmp_path / "combined.parquet"
    _write_parquet(
        source,
        {
            "winner_seat": ["P1", "P2"],
            "P1_strategy": [None, None],
            "P1_score": [10, 20],
            "P1_farkles": [1, 1],
            "P1_rounds": [6, 6],
        },
    )

    cfg = seat_stats.SeatMetricConfig(seat_range=(1, 1))
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


def test_compute_seat_metrics_missing_score_farkles_rounds_columns(tmp_path: Path) -> None:
    source = tmp_path / "combined.parquet"
    _write_parquet(
        source,
        {
            "winner_seat": ["P1", "P2", "P1"],
            "seat_ranks": [["P1", "P2"], ["P2", "P1"], ["P1", "P2"]],
            "P1_strategy": ["Aggro", "Aggro", "Aggro"],
        },
    )

    cfg = seat_stats.SeatMetricConfig(seat_range=(1, 1))
    df = seat_stats.compute_seat_metrics(source, cfg)

    row = df.iloc[0]
    assert row["games"] == 3
    assert row["wins"] == 2
    assert pd.isna(row["mean_score"])
    assert pd.isna(row["mean_farkles"])
    assert pd.isna(row["mean_rounds"])


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


@pytest.mark.parametrize(
    ("data", "expected_players"),
    [
        (
            {
                "P1_strategy": ["Aggro", "Aggro"],
                "P2_strategy": ["Aggro", "Aggro"],
                "P1_farkles": [1, 3],
                "P2_farkles": [2, 2],
                "P1_rounds": [10, 12],
                "P2_rounds": [11, 11],
                "seat_ranks": [["P1", "P2"], ["P2", "P1"]],
            },
            2,
        ),
        (
            {
                "P1_strategy": ["Aggro", "Aggro"],
                "P2_strategy": ["Aggro", "Aggro"],
                "P1_farkles": [1, 3],
                "P2_farkles": [2, 2],
                "P1_rounds": [10, 12],
                "P2_rounds": [11, 11],
            },
            2,
        ),
    ],
)
def test_compute_symmetry_checks_derived_or_default_players(
    tmp_path: Path, data: dict[str, list], expected_players: int
) -> None:
    source = tmp_path / "combined.parquet"
    _write_parquet(source, data)

    cfg = seat_stats.SeatMetricConfig(seat_range=(1, 2), symmetry_tolerance=0.5)
    df = seat_stats.compute_symmetry_checks(source, cfg)

    row = df.iloc[0]
    assert row["n_players"] == expected_players
    assert row["observations"] == 2
    assert row["farkle_diff"] == pytest.approx(0.0)
    assert row["rounds_diff"] == pytest.approx(0.0)
    assert not row["farkle_flagged"]
    assert not row["rounds_flagged"]


def test_compute_symmetry_checks_returns_empty_when_no_symmetric_rows(tmp_path: Path) -> None:
    source = tmp_path / "combined.parquet"
    _write_parquet(
        source,
        {
            "P1_strategy": ["Aggro", "Control"],
            "P2_strategy": ["Control", "Aggro"],
            "P1_farkles": [1, 2],
            "P2_farkles": [2, 1],
            "P1_rounds": [10, 11],
            "P2_rounds": [10, 11],
            "n_players": [2, 2],
        },
    )

    cfg = seat_stats.SeatMetricConfig(seat_range=(1, 2))
    df = seat_stats.compute_symmetry_checks(source, cfg)

    assert df.empty


def test_compute_symmetry_checks_coerces_non_int_group_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "combined.parquet"
    _write_parquet(
        source,
        {
            "P1_strategy": ["Aggro", "Aggro"],
            "P2_strategy": ["Aggro", "Aggro"],
            "P1_farkles": [7, 7],
            "P2_farkles": [5, 5],
            "P1_rounds": [9, 9],
            "P2_rounds": [8, 8],
            "n_players": [2, 2],
        },
    )

    orig_astype = seat_stats.pd.Series.astype

    def _astype_preserve_strings(
        self,
        dtype,
        copy: bool = True,
        errors: Literal["ignore", "raise"] = "raise",
    ):
        if self.name == "n_players" and dtype in {int, "int"}:
            return self
        return orig_astype(self, dtype=dtype, copy=copy, errors=errors)

    monkeypatch.setattr(seat_stats.pd.Series, "astype", _astype_preserve_strings)

    cfg = seat_stats.SeatMetricConfig(seat_range=(1, 2), symmetry_tolerance=0.5)
    df = seat_stats.compute_symmetry_checks(source, cfg)

    row = df.iloc[0]
    assert row["n_players"] == 2
    assert row["farkle_flagged"]
    assert row["rounds_flagged"]


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


def test_compute_seat_advantage_handles_missing_and_corrupt_manifests(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path
    combined = tmp_path / "combined.parquet"
    _write_parquet(combined, {"winner_seat": ["P1", "P2", "P2"]})

    manifest1 = cfg.manifest_for(1)
    manifest1.parent.mkdir(parents=True, exist_ok=True)
    manifest1.write_text(json.dumps({"row_count": 5}))

    # Present but corrupt, should contribute 0 via exception branch.
    manifest2 = cfg.manifest_for(2)
    manifest2.parent.mkdir(parents=True, exist_ok=True)
    manifest2.write_text("{ this is not json }")

    seat_cfg = seat_stats.SeatMetricConfig(seat_range=(1, 2))
    df = seat_stats.compute_seat_advantage(cfg, combined, seat_cfg)

    seat1 = df[df["seat"] == 1].iloc[0]
    seat2 = df[df["seat"] == 2].iloc[0]
    assert seat1["games_with_seat"] == 5
    assert seat2["games_with_seat"] == 0
    assert seat2["wins"] == 2
    assert seat2["win_rate"] == 0.0
