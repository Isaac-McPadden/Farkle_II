import json
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
        (["theta", "zeta", "eta"], [0, 1, 1]),
    ]


def test_rate_single_pass_resumes_from_checkpoint(tmp_path: Path) -> None:
    env = trueskill.TrueSkill()
    table = pa.table(
        {
            "winner_seat": ["P1", "P2"],
            "P1_strategy": ["A", "B"],
            "P2_strategy": ["B", "C"],
            "P1_rank": pa.array([0, 1], type=pa.int64()),
            "P2_rank": pa.array([1, 0], type=pa.int64()),
        }
    )
    source = tmp_path / "combined.parquet"
    pq.write_table(table, source, row_group_size=1)

    ratings_ck = tmp_path / "ratings.checkpoint.parquet"
    rt._save_ratings_parquet(ratings_ck, {"B": env.create_rating(mu=33.0, sigma=7.0)})
    ck_path = tmp_path / "ratings.ckpt.json"
    rt._save_ckpt(
        ck_path,
        rt._TSCheckpoint(
            source=str(source),
            row_group=1,
            batch_index=0,
            games_done=3,
            ratings_path=str(ratings_ck),
        ),
    )

    stats, games = rt._rate_single_pass(
        source=source,
        env=env,
        resume=True,
        checkpoint_path=ck_path,
        ratings_ckpt_path=ratings_ck,
        batch_rows=1,
        checkpoint_every_batches=1,
    )
    assert games == 4
    assert set(stats) == {"B", "C"}
    assert "A" not in stats

    updated = json.loads(ck_path.read_text())
    assert updated["row_group"] == 1
    assert updated["batch_index"] == 1
    assert updated["games_done"] == games

    interim = rt._load_ratings_parquet(ratings_ck)
    assert set(interim) == {"B", "C"}


def test_rate_block_worker_resumes_from_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    data_dir = root / "2p"
    data_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = root / "data" / "2p"
    legacy_dir.mkdir(parents=True, exist_ok=True)
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

    ratings_ck = legacy_dir / "ratings_2.checkpoint.parquet"
    rt._save_ratings_parquet(
        ratings_ck, {"A": trueskill.TrueSkill().create_rating(mu=25.0, sigma=8.0)}
    )
    ck_path = legacy_dir / "ratings_2.ckpt.json"
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
        (rt._load_ckpt, "ratings.ckpt.json"),
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


def test_rate_block_worker_missing_curated_and_fallback(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    block = tmp_path / "results" / "2_players"
    block.mkdir(parents=True, exist_ok=True)

    player_count, games = rt._rate_block_worker(
        str(block),
        str(root),
        "",
        batch_rows=1,
        resume=False,
        checkpoint_every_batches=1,
    )

    assert player_count == "2"
    assert games == 0


def test_coerce_and_seed_ratings() -> None:
    mapping: dict[str, Any] = {
        "A": rt.RatingStats(1.0, 2.0),
        "B": {"mu": 3.0, "sigma": 4.0},
        "C": (5.0, 6.0),
        "D": [7.0, 8.0, 9.0],
        "E": 10.0,
    }
    coerced = rt._coerce_ratings(mapping)
    assert coerced["A"] == rt.RatingStats(1.0, 2.0)
    assert coerced["B"] == rt.RatingStats(3.0, 4.0)
    assert coerced["C"] == rt.RatingStats(5.0, 6.0)
    assert coerced["D"] == rt.RatingStats(7.0, 8.0)
    assert coerced["E"] == rt.RatingStats(0.0, 0.0)

    env = _DummyEnv()
    ratings = {"A": _DummyRating(0.5, 0.1)}
    rt._ensure_seed_ratings(ratings, ["A", "B"], env)  # type: ignore[arg-type]
    assert {"A", "B"} <= set(ratings)
    assert isinstance(ratings["B"], _DummyRating)


def test_rate_stream_applies_keeper_filter(tmp_path: Path) -> None:
    env = _DummyEnv()
    table = pa.table(
        {
            "winner": ["P1"],
            "P1_strategy": ["A"],
            "P2_strategy": ["B"],
            "P3_strategy": ["C"],
            "P1_rank": pa.array([0], type=pa.int64()),
            "P2_rank": pa.array([2], type=pa.int64()),
            "P3_rank": pa.array([5], type=pa.int64()),
        }
    )
    path = tmp_path / "stream.parquet"
    pq.write_table(table, path)
    ratings, games = rt._rate_stream(path, 3, ["A", "C"], env, batch_size=1)  # type: ignore[arg-type]
    assert games == 1
    assert env.rate_calls == [[0, 1]]
    assert set(ratings) == {"A", "C"}
    assert ratings["A"].mu == 0.0
    assert ratings["C"].mu == 1.0
