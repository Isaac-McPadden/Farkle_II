from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


@dataclass(frozen=True)
class GoldenDataset:
    parquet: Path
    dataframe: pd.DataFrame

    def copy_into(self, results_root: Path) -> Path:
        block = results_root / "3_players"
        block.mkdir(parents=True, exist_ok=True)
        target = block / "3p_rows.parquet"
        target.write_bytes(self.parquet.read_bytes())
        return target


def _build_golden_df() -> pd.DataFrame:
    winners = ["P1"] * 20 + ["P2"] * 15 + ["P3"] * 15
    rounds = [6 + (i % 4) for i in range(50)]
    base_scores = 950 + np.arange(50) * 7
    strategies = {
        "P1": "Aggro",
        "P2": "Balanced",
        "P3": "Cautious",
    }
    rows = []
    for idx, seat in enumerate(winners):
        row = {
            "winner": seat,
            "n_rounds": rounds[idx],
            "winning_score": int(base_scores[idx] + (idx % 3) * 10),
            "P1_strategy": strategies["P1"],
            "P2_strategy": strategies["P2"],
            "P3_strategy": strategies["P3"],
        }
        ranks = {"P1": 2, "P2": 3, "P3": 4}
        ranks[seat] = 1
        for seat_label, rank in ranks.items():
            row[f"{seat_label}_rank"] = rank
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def golden_dataset(tmp_path_factory: pytest.TempPathFactory) -> GoldenDataset:
    base = tmp_path_factory.mktemp("golden_results")
    df = _build_golden_df()
    parquet = base / "3p_rows.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, parquet)
    return GoldenDataset(parquet=parquet, dataframe=df)
