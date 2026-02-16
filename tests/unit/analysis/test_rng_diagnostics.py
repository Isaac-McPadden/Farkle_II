from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tests.helpers.diagnostic_fixtures import build_curated_fixture

from farkle.analysis import rng_diagnostics
from farkle.analysis.stage_registry import resolve_interseed_stage_layout
from farkle.utils.types import Compression


def test_collect_diagnostics_empty_input():
    empty = pd.DataFrame(columns=["strategy", "n_players", "win_indicator", "n_rounds", "game_seed", "matchup"])
    result = rng_diagnostics._collect_diagnostics(empty, lags=(1, 2))

    assert result.empty


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
