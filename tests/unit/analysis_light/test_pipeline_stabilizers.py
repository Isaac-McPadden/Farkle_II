import logging

import json
import logging

import pandas as pd
import pyarrow.parquet as pq

from farkle.analysis import combine, curate, ingest, metrics
from farkle.analysis.analysis_config import PipelineCfg


def test_ingest_golden_dataset(tmp_path, caplog, golden_dataset):
    cfg = PipelineCfg(results_dir=tmp_path)
    golden_dataset.copy_into(cfg.results_dir)

    caplog.set_level(logging.INFO, logger="farkle.analysis.ingest")
    ingest.run(cfg)

    raw_file = cfg.ingested_rows_raw(3)
    assert raw_file.exists()
    table = pq.read_table(raw_file)
    assert table.num_rows == len(golden_dataset.dataframe)
    df = table.to_pandas()
    expected = golden_dataset.dataframe["winner"].value_counts().to_dict()
    assert df["winner_seat"].value_counts().to_dict() == expected

    messages = [rec.message for rec in caplog.records]
    assert any("Ingest finished" in msg for msg in messages)


def test_curate_golden_dataset(tmp_path, caplog, golden_dataset):
    cfg = PipelineCfg(results_dir=tmp_path)
    golden_dataset.copy_into(cfg.results_dir)
    ingest.run(cfg)

    caplog.set_level(logging.INFO, logger="farkle.analysis.curate")
    curate.run(cfg)

    curated = cfg.ingested_rows_curated(3)
    manifest = cfg.manifest_for(3)
    assert curated.exists()
    assert manifest.exists()

    table = pq.read_table(curated)
    assert table.num_rows == len(golden_dataset.dataframe)
    meta = json.loads(manifest.read_text())
    assert meta["row_count"] == len(golden_dataset.dataframe)

    messages = [rec.message for rec in caplog.records]
    assert any("Curate finished" in msg for msg in messages)


def test_metrics_golden_dataset(tmp_path, caplog, golden_dataset):
    cfg = PipelineCfg(results_dir=tmp_path)
    golden_dataset.copy_into(cfg.results_dir)
    ingest.run(cfg)
    curate.run(cfg)
    combine.run(cfg)

    caplog.set_level(logging.INFO, logger="farkle.analysis.metrics")
    metrics.run(cfg)

    metrics_path = cfg.analysis_dir / cfg.metrics_name
    seat_csv = cfg.analysis_dir / "seat_advantage.csv"
    seat_parquet = cfg.analysis_dir / "seat_advantage.parquet"

    assert metrics_path.exists()
    assert seat_csv.exists()
    assert seat_parquet.exists()

    metrics_df = pq.read_table(metrics_path).to_pandas()
    expected_wins = golden_dataset.dataframe["winner"].map(
        {"P1": "Aggro", "P2": "Balanced", "P3": "Cautious"}
    ).value_counts()
    wins_by_strategy = metrics_df.set_index("strategy")["wins"].to_dict()
    assert wins_by_strategy == expected_wins.to_dict()
    assert set(metrics_df["games"]) == {len(golden_dataset.dataframe)}

    seat_df = pd.read_csv(seat_csv)
    seat_df["seat"] = seat_df["seat"].apply(lambda s: f"P{int(s)}")
    seats = seat_df.set_index("seat")["wins"].to_dict()
    expected_seat_wins = golden_dataset.dataframe["winner"].value_counts().to_dict()
    observed = {seat: seats[seat] for seat in expected_seat_wins}
    assert observed == expected_seat_wins

    messages = [rec.message for rec in caplog.records]
    assert any("Metrics leaderboard computed" in msg for msg in messages)
    assert any("Metrics stage complete" in msg for msg in messages)
