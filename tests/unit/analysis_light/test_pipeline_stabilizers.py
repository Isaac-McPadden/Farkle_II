# tests/unit/analysis_light/test_pipeline_stabilizers.py
import datetime as _dt
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pandas as pd
import pyarrow.parquet as pq
import pytest


class _CfgProto(Protocol):
    results_dir: Path
    analysis_dir: Path
    metrics_name: str
    parquet_codec: str
    curated_parquet: Path
    def ingested_rows_raw(self, n: int) -> Path: ...
    def ingested_rows_curated(self, n: int) -> Path: ...
    def manifest_for(self, n: int) -> Path: ...


HAS_UTC = hasattr(_dt, "UTC")
pytestmark = pytest.mark.skipif(
    not HAS_UTC,
    reason="analysis pipeline stabilizers require datetime.UTC (Python 3.11+)",
)

if TYPE_CHECKING or HAS_UTC:
    from farkle.analysis import combine, curate, ingest, metrics
    from farkle.analysis.analysis_config import PipelineCfg
else:  # pragma: no cover - tests skipped when UTC unavailable
    combine = curate = ingest = metrics = None  # type: ignore[assignment]
    PipelineCfg = object  # type: ignore[assignment]


STRATEGY_MAP = {"P1": "Aggro", "P2": "Balanced", "P3": "Cautious"}


def test_ingest_golden_dataset(tmp_results_dir, caplog, golden_dataset):
    cfg = PipelineCfg(results_dir=tmp_results_dir)
    cfg_proto = cast(_CfgProto, cfg)
    golden_dataset.copy_into(cfg_proto.results_dir)

    caplog.set_level(logging.INFO, logger="farkle.analysis.ingest")
    ingest.run(cfg)

    raw_file = cfg_proto.ingested_rows_raw(3)
    assert raw_file.exists()
    table = pq.read_table(raw_file)
    columns = set(table.column_names)
    assert {"winner_seat", "winner_strategy", "seat_ranks"}.issubset(columns)
    assert table.num_rows == len(golden_dataset.dataframe)
    df = table.to_pandas()
    observed_strategies = (
        df[["winner_seat", "winner_strategy"]]
        .drop_duplicates()
        .set_index("winner_seat")
        ["winner_strategy"]
        .to_dict()
    )
    expected_strategies = (
        golden_dataset.dataframe[["winner"]]
        .drop_duplicates()
        .assign(winner_strategy=lambda frame: frame["winner"].map(STRATEGY_MAP))
        .set_index("winner")
        ["winner_strategy"]
        .to_dict()
    )
    assert observed_strategies == expected_strategies
    assert all(set(ranks) == {"P1", "P2", "P3"} for ranks in df["seat_ranks"])
    expected = golden_dataset.dataframe["winner"].value_counts().to_dict()
    assert df["winner_seat"].value_counts().to_dict() == expected

    messages = [rec.message for rec in caplog.records]
    assert any("Ingest finished" in msg for msg in messages)


def test_curate_golden_dataset(tmp_results_dir, caplog, golden_dataset):
    cfg = PipelineCfg(results_dir=tmp_results_dir)
    cfg_proto = cast(_CfgProto, cfg)
    golden_dataset.copy_into(cfg_proto.results_dir)
    raw_path = cfg_proto.ingested_rows_raw(3)
    ingest.run(cfg)
    assert raw_path.exists()

    caplog.set_level(logging.INFO, logger="farkle.analysis.curate")
    curate.run(cfg)

    curated = cfg_proto.ingested_rows_curated(3)
    manifest = cfg_proto.manifest_for(3)
    assert curated.exists()
    assert manifest.exists()
    assert not raw_path.exists()

    table = pq.read_table(curated)
    assert table.num_rows == len(golden_dataset.dataframe)
    meta = json.loads(manifest.read_text())
    assert meta["row_count"] == len(golden_dataset.dataframe)
    assert meta["schema_hash"]
    assert meta.get("compression") == cfg_proto.parquet_codec
    assert "created_at" in meta

    messages = [rec.message for rec in caplog.records]
    assert any("Curate finished" in msg for msg in messages)


def test_metrics_golden_dataset(tmp_results_dir, caplog, golden_dataset):
    cfg = PipelineCfg(results_dir=tmp_results_dir)
    cfg_proto = cast(_CfgProto, cfg)
    golden_dataset.copy_into(cfg_proto.results_dir)
    ingest.run(cfg)
    curate.run(cfg)
    combine.run(cfg)

    caplog.set_level(logging.INFO, logger="farkle.analysis.metrics")
    metrics.run(cfg)

    metrics_path = cfg_proto.analysis_dir / cfg_proto.metrics_name
    seat_csv = cfg_proto.analysis_dir / "seat_advantage.csv"
    seat_parquet = cfg_proto.analysis_dir / "seat_advantage.parquet"
    stamp_path = cfg_proto.analysis_dir / "metrics.done.json"

    assert metrics_path.exists()
    assert seat_csv.exists()
    assert seat_parquet.exists()
    assert stamp_path.exists()

    metrics_df = pq.read_table(metrics_path).to_pandas()
    stamp = json.loads(stamp_path.read_text())
    expected_input = str(cfg_proto.curated_parquet)
    assert expected_input in stamp.get("inputs", {})
    for expected_output in (metrics_path, seat_csv, seat_parquet):
        assert str(expected_output) in stamp.get("outputs", {})
    strategy_series = golden_dataset.dataframe["winner"].map(STRATEGY_MAP)
    expected_wins = strategy_series.value_counts()
    total_games = len(golden_dataset.dataframe)
    metrics_by_strategy = metrics_df.set_index("strategy")
    games_by_strategy = metrics_by_strategy["games"].to_dict()
    wins_by_strategy = metrics_by_strategy["wins"].to_dict()
    win_rate = metrics_by_strategy["win_rate"].to_dict()
    expected_scores = (
        golden_dataset.dataframe.assign(winner_strategy=strategy_series)
        .groupby("winner_strategy")
        .agg({"winning_score": "sum", "n_rounds": "sum"})
        .rename(columns={"winning_score": "score_sum", "n_rounds": "round_sum"})
    )
    for strategy, wins in expected_wins.to_dict().items():
        assert wins_by_strategy[strategy] == wins
        assert games_by_strategy[strategy] == total_games
        assert win_rate[strategy] == pytest.approx(wins / total_games)
        score_sum = expected_scores.loc[strategy, "score_sum"]
        round_sum = expected_scores.loc[strategy, "round_sum"]
        observed = metrics_by_strategy.loc[strategy]
        assert observed["expected_score"] == pytest.approx(score_sum / total_games)
        assert observed["mean_score"] == pytest.approx(score_sum / wins)
        assert observed["mean_rounds"] == pytest.approx(round_sum / wins)

    seat_df = pd.read_csv(seat_csv)
    seat_df["seat"] = seat_df["seat"].apply(lambda s: f"P{int(s)}")
    seats = seat_df.set_index("seat")["wins"].to_dict()
    expected_seat_wins = golden_dataset.dataframe["winner"].value_counts().to_dict()
    observed = {seat: seats[seat] for seat in expected_seat_wins}
    assert observed == expected_seat_wins
    games_by_seat = seat_df.set_index("seat")["games_with_seat"].to_dict()
    expected_games_per_seat = {
        f"P{seat}": (total_games if seat <= 3 else 0) for seat in range(1, 13)
    }
    assert games_by_seat == expected_games_per_seat
    seat_from_parquet = (
        pq.read_table(seat_parquet)
        .to_pandas()
        .assign(seat=lambda frame: frame["seat"].apply(lambda s: f"P{int(s)}"))
    )
    pd.testing.assert_frame_equal(
        seat_df.sort_values("seat").reset_index(drop=True),
        seat_from_parquet.sort_values("seat").reset_index(drop=True),
        check_dtype=False,
    )

    messages = [rec.message for rec in caplog.records]
    assert any("Metrics leaderboard computed" in msg for msg in messages)
    assert any("Metrics stage complete" in msg for msg in messages)
