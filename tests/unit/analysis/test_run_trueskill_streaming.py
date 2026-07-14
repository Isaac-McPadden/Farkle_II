from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import trueskill

import farkle.analysis.run_trueskill as rt


class _DummyRating:
    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu = mu
        self.sigma = sigma


class _DummyEnv:
    def __init__(self) -> None:
        self.rate_calls: list[list[int]] = []

    def create_rating(self, mu: float = 0.0, sigma: float = 1.0) -> _DummyRating:
        return _DummyRating(mu, sigma)

    def rate(self, teams: list[list[_DummyRating]], ranks: list[int]) -> list[list[_DummyRating]]:
        self.rate_calls.append(list(ranks))
        result: list[list[_DummyRating]] = []
        for team, rank in zip(teams, ranks, strict=True):
            base = team[0]
            result.append([_DummyRating(float(rank), base.sigma)])
        return result


@pytest.fixture()
def sample_parquet(tmp_path: Path) -> Path:
    table = pa.table(
        {
            "winner_seat": ["P1", "P2", "P1", "P2", "P1"],
            "P1_strategy": ["A", "B", "C", "D", "E"],
            "P2_strategy": ["B", "C", "D", "E", "F"],
        }
    )
    path = tmp_path / "games.parquet"
    pq.write_table(table, path, row_group_size=3)
    return path


def test_stream_batches_respects_resume_offsets(sample_parquet: Path) -> None:
    batches = list(
        rt._stream_batches(
            sample_parquet,
            ["winner_seat", "P1_strategy", "P2_strategy"],
            batch_rows=2,
        )
    )
    assert [(rg, bi, batch.num_rows) for rg, bi, batch in batches] == [
        (0, 0, 2),
        (0, 1, 1),
        (1, 0, 2),
    ]
    assert batches[0][2].column("P1_strategy").to_pylist() == ["A", "B"]

    resumed = list(
        rt._stream_batches(
            sample_parquet,
            ["winner_seat", "P1_strategy"],
            start_row_group=0,
            start_batch_idx=1,
            batch_rows=2,
        )
    )
    assert [(rg, bi) for rg, bi, _ in resumed] == [(0, 1), (1, 0)]
    assert resumed[0][2].column("P1_strategy").to_pylist() == ["C"]

    later_groups = list(
        rt._stream_batches(
            sample_parquet,
            ["winner_seat"],
            start_row_group=1,
            batch_rows=2,
        )
    )
    assert [(rg, bi, batch.num_rows) for rg, bi, batch in later_groups] == [(1, 0, 2)]


def test_players_and_ranks_precedence(tmp_path: Path) -> None:
    seat_ranks = pa.array(
        [
            ["P3", "P2", "P1"],
            ["P2", "P1"],
            None,
        ],
        type=pa.list_(pa.string()),
    )
    table = pa.table(
        {
            "seat_ranks": seat_ranks,
            "winner": ["P3", "P2", "P3"],
            "P1_strategy": ["alpha", "delta", "zeta"],
            "P2_strategy": ["beta", "epsilon", "eta"],
            "P3_strategy": ["gamma", None, "theta"],
            "P1_rank": pa.array([1, None, None], type=pa.int64()),
            "P2_rank": pa.array([1, None, None], type=pa.int64()),
            "P3_rank": pa.array([2, None, None], type=pa.int64()),
        }
    )
    path = tmp_path / "precedence.parquet"
    pq.write_table(table, path, row_group_size=3)

    batch = next(
        rt._stream_batches(
            path,
            list(table.schema.names),
            batch_rows=10,
        )
    )[2]
    rows = list(rt._players_and_ranks_from_batch(batch, 3))
    assert rows == [
        (["alpha", "beta", "gamma"], [0, 0, 1]),
        (["epsilon", "delta"], [0, 1]),
    ]


def test_rate_block_worker_resumes_from_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    data_dir = root / "by_k" / "2p"
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = data_dir
    block_dir = tmp_path / "results" / "2_players"
    block_dir.mkdir(parents=True, exist_ok=True)
    np.save(block_dir / "keepers_2.npy", np.array(["A", "C"]))

    table = pa.table(
        {
            "winner_seat": ["P1", "P2"],
            "P1_strategy": ["A", "A"],
            "P2_strategy": ["B", "C"],
            "P1_rank": pa.array([0, 0], type=pa.int64()),
            "P2_rank": pa.array([1, 2], type=pa.int64()),
        }
    )
    row_file = data_dir / "2p_ingested_rows.parquet"
    pq.write_table(table, row_file, row_group_size=1)

    ratings_ck = checkpoint_dir / "ratings_2.checkpoint.parquet"
    rt._save_ratings_parquet(
        ratings_ck, {"A": trueskill.TrueSkill().create_rating(mu=25.0, sigma=8.0)}
    )
    ck_path = checkpoint_dir / "ratings_2.ckpt.json"
    rt._save_block_ckpt(
        ck_path,
        rt._BlockCkpt(
            row_file=str(row_file),
            row_group=1,
            batch_index=0,
            games_done=7,
            ratings_path=str(ratings_ck),
        ),
    )

    player_count, games = rt._rate_block_worker(
        str(block_dir),
        str(root),
        "",
        batch_rows=1,
        resume=True,
        checkpoint_every_batches=1,
        row_data_dir=str(root),
        curated_rows_name="2p_ingested_rows.parquet",
    )
    assert player_count == "2"
    assert games == 8

    ratings = rt._load_ratings_parquet(data_dir / "ratings_2.parquet")
    assert set(ratings) == {"A", "C"}
    assert "B" not in ratings
    assert not (data_dir / "ratings_2.ckpt.json").exists()
    assert not (data_dir / "ratings_2.checkpoint.parquet").exists()


@pytest.mark.parametrize(
    ("loader", "filename"),
    [
        (rt._load_block_ckpt, "block.ckpt.json"),
    ],
)
def test_load_ckpts_handle_missing_and_invalid(
    tmp_path: Path, loader: Callable[[Path], Any], filename: str
) -> None:
    path = tmp_path / filename
    assert loader(path) is None

    path.write_text("{not json")
    assert loader(path) is None


def test_rate_block_worker_rejects_missing_canonical_coordinates(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    block = tmp_path / "results" / "2_players"
    block.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="explicit canonical curated-row"):
        rt._rate_block_worker(
            str(block),
            str(root),
            "",
            batch_rows=1,
            resume=False,
            checkpoint_every_batches=1,
        )
