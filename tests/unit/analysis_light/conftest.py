from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, SupportsFloat, cast

import numpy as np
import pandas as pd
import pytest

from farkle.config import AppConfig, IOConfig, SimConfig

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


@dataclass(frozen=True)
class GoldenDataset:
    parquet: Path
    dataframe: pd.DataFrame
    strategy_map: Mapping[str, int] | None = None

    def copy_into(self, results_root: Path) -> Path:
        block = results_root / "3_players"
        block.mkdir(parents=True, exist_ok=True)
        target = block / "3p_rows.parquet"
        target.write_bytes(self.parquet.read_bytes())
        return target

    def write_metrics(
        self,
        results_root: Path,
        *,
        player_count: int = 3,
        strategy_map: Mapping[str, int] | None = None,
    ) -> Path:
        mapping = dict(strategy_map or self.strategy_map or {})
        strategies = self.dataframe["winner"].map(mapping)
        total_games = len(self.dataframe)

        records = []
        for strategy, wins in strategies.value_counts().items():
            mask = strategies == strategy
            scores = self.dataframe.loc[mask, "winning_score"]
            rounds = self.dataframe.loc[mask, "n_rounds"]
            records.append(
                {
                    "strategy": strategy,
                    "wins": float(wins),
                    "total_games_strat": float(total_games),
                    "win_rate": float(wins / total_games),
                    "sum_winning_score": float(scores.sum()),
                    "sq_sum_winning_score": float((scores**2).sum()),
                    "mean_winning_score": float(scores.mean()),
                    "var_winning_score": float(cast(SupportsFloat, scores.var(ddof=0))),
                    "sum_n_rounds": float(rounds.sum()),
                    "sq_sum_n_rounds": float((rounds**2).sum()),
                    "mean_n_rounds": float(rounds.mean()),
                    "var_n_rounds": float(cast(SupportsFloat, rounds.var(ddof=0))),
                    "sum_winner_hit_max_rounds": 0.0,
                }
            )

        metrics_df = pd.DataFrame(records)
        metrics_path = results_root / f"{player_count}_players" / f"{player_count}p_metrics.parquet"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_parquet(metrics_path, index=False)
        return metrics_path


def _build_golden_df() -> pd.DataFrame:
    winners = ["P1"] * 20 + ["P2"] * 15 + ["P3"] * 15
    rounds = [6 + (i % 4) for i in range(50)]
    base_scores = 950 + np.arange(50) * 7
    strategies = {"P1": 101, "P2": 202, "P3": 303}
    rows = []
    for idx, seat in enumerate(winners):
        score_base = int(base_scores[idx])
        row = {
            "winner": seat,
            "winner_seat": seat,
            "winner_strategy": strategies[seat],
            "game_seed": idx,
            "seat_ranks": [seat, *(s for s in ("P1", "P2", "P3") if s != seat)],
            "n_rounds": rounds[idx],
            "winning_score": int(score_base + (idx % 3) * 10),
            "P1_strategy": strategies["P1"],
            "P2_strategy": strategies["P2"],
            "P3_strategy": strategies["P3"],
        }
        ranks = {"P1": 2, "P2": 3, "P3": 4}
        ranks[seat] = 1
        for seat_label, rank in ranks.items():
            row[f"{seat_label}_rank"] = rank
            row[f"{seat_label}_score"] = score_base + (rank * 3)
            row[f"{seat_label}_farkles"] = 1 if seat_label != seat else 0
            row[f"{seat_label}_rolls"] = rounds[idx] + rank
            row[f"{seat_label}_highest_turn"] = max(1, rounds[idx] - rank)
            row[f"{seat_label}_loss_margin"] = 10 * rank
            row[f"{seat_label}_smart_five_uses"] = rank
            row[f"{seat_label}_n_smart_five_dice"] = rank + 1
            row[f"{seat_label}_smart_one_uses"] = rank - 1
            row[f"{seat_label}_n_smart_one_dice"] = rank
            row[f"{seat_label}_hot_dice"] = 1 if rank == 1 else 0
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def golden_dataset(tmp_path_factory: pytest.TempPathFactory) -> GoldenDataset:
    base = tmp_path_factory.mktemp("golden_results")
    df = _build_golden_df()
    parquet = base / "3p_rows.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, parquet)

    strategy_map = {"P1": 101, "P2": 202, "P3": 303}
    return GoldenDataset(parquet=parquet, dataframe=df, strategy_map=strategy_map)


@pytest.fixture
def patched_strategy_grid(monkeypatch: pytest.MonkeyPatch) -> Sequence[object]:
    class _StubStrategy:
        def __init__(self, label: str) -> None:
            self._label = label

        def __str__(self) -> str:  # pragma: no cover - trivial
            return self._label

    strategies = [_StubStrategy(name) for name in ("101", "202", "303")]

    def _grid(**_: object):
        return strategies, pd.DataFrame({"strategy_id": [101, 202, 303]})

    monkeypatch.setattr(
        "farkle.analysis.isolated_metrics.generate_strategy_grid", _grid, raising=False
    )
    monkeypatch.setattr("farkle.analysis.isolated_metrics._STRATEGY_CACHE", {}, raising=False)
    return strategies


@pytest.fixture
def analysis_config(tmp_results_dir: Path) -> Callable[..., AppConfig]:
    def _factory(**overrides: Any) -> AppConfig:
        sim_cfg = cast(SimConfig | None, overrides.pop("sim", None))
        if sim_cfg is None:
            sim_cfg = SimConfig(n_players_list=[3], expanded_metrics=True)
        io_cfg = IOConfig(results_dir_prefix=tmp_results_dir)
        return AppConfig(io=io_cfg, sim=sim_cfg, **overrides)

    return _factory
