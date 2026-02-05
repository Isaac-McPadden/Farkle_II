"""Synthetic analysis fixtures for metrics and diagnostics tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.config import AppConfig, IOConfig, MetricsConfig, SimConfig


def build_curated_fixture(tmp_path: Path) -> tuple[AppConfig, Path, Path]:
    """Create a compact curated parquet with deterministic seat outcomes.

    The fixture mirrors the layout produced by the pipeline so metrics, game stats,
    and diagnostics can reuse it without bespoke setup code. Manifests are written
    with the expected row counts so seat-advantage denominators are well defined.
    """

    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path),
        sim=SimConfig(n_players_list=[2]),
        metrics=MetricsConfig(seat_range=(1, 2)),
    )

    combined_dir = cfg.combine_pooled_dir()
    combined_dir.mkdir(parents=True, exist_ok=True)
    per_n_dir = cfg.data_dir / "2p"
    per_n_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.DataFrame(
        [
            {
                "game_seed": 1,
                "seat_ranks": ["P1", "P2"],
                "winner_seat": "P1",
                "winner_strategy": 1,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 120,
                "P2_score": 105,
                "P1_rounds": 6,
                "P2_rounds": 6,
                "n_rounds": 6,
            },
            {
                "game_seed": 2,
                "seat_ranks": ["P1", "P2"],
                "winner_seat": "P1",
                "winner_strategy": 1,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 90,
                "P2_score": 40,
                "P1_rounds": 9,
                "P2_rounds": 9,
                "n_rounds": 9,
            },
            {
                "game_seed": 3,
                "seat_ranks": ["P2", "P1"],
                "winner_seat": "P2",
                "winner_strategy": 2,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 150,
                "P2_score": 200,
                "P1_rounds": 12,
                "P2_rounds": 12,
                "n_rounds": 12,
            },
        ]
    )

    table = pa.Table.from_pandas(rows, preserve_index=False)

    combined_path = combined_dir / "all_ingested_rows.parquet"
    pq.write_table(table, combined_path)
    per_n_path = per_n_dir / "2p_rows.parquet"
    pq.write_table(table, per_n_path)

    manifest = per_n_dir / "manifest.jsonl"
    manifest.write_text(json.dumps({"row_count": len(rows)}))

    return cfg, combined_path, per_n_path
