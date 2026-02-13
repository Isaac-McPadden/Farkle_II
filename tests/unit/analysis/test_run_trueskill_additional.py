from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import trueskill

import farkle.analysis.run_trueskill as rt
from farkle.analysis.stage_registry import resolve_interseed_stage_layout
from farkle.config import AppConfig, IOConfig, SimConfig
from farkle.utils.types import Compression, normalize_compression


def test_find_combined_parquet_none_cases(tmp_path: Path) -> None:
    assert rt._find_combined_parquet(None) is None
    base = tmp_path / "empty"
    base.mkdir()
    assert rt._find_combined_parquet(base) is None


def test_ratings_to_table_handles_tuple_and_scalar() -> None:
    table = rt._ratings_to_table(
        {
            "tuple": (10.5, 3.2),
            "scalar": 27.0,
            "stats": rt.RatingStats(5.0, 1.0),
        }
    )
    data = table.to_pydict()
    rows = {
        s: (mu, sigma)
        for s, mu, sigma in zip(data["strategy"], data["mu"], data["sigma"], strict=True)
    }
    assert rows["tuple"] == pytest.approx((10.5, 3.2))
    assert rows["scalar"][1] == pytest.approx(0.0)
    assert rows["stats"] == pytest.approx((5.0, 1.0))


def test_players_and_ranks_from_batch_branching() -> None:
    seat_ranks = pa.array(
        [
            None,
            ["P2", "P3"],
            ["P1", "P3"],
            None,
            None,
            None,
            None,
            None,
        ],
        type=pa.list_(pa.string()),
    )
    winner_seat = pa.array([None, None, None, None, "bad", "P1", "P2", "P1"], type=pa.string())
    p1_strategy = pa.array(["A0", None, "A2", "A3", "A4", None, "A6", "A7"], type=pa.string())
    p2_strategy = pa.array(["B0", "B1", None, None, "B4", "B5", "B6", None], type=pa.string())
    p3_strategy = pa.array([None, "C1", None, None, None, None, "C6", None], type=pa.string())
    p1_rank = pa.array([1, 1, None, None, None, None, None, None], type=pa.int64())
    p2_rank = pa.array([2, None, None, None, None, None, None, None], type=pa.int64())
    p3_rank = pa.array([None, None, None, None, None, None, None, None], type=pa.int64())

    batch = pa.table(
        {
            "seat_ranks": seat_ranks,
            "winner_seat": winner_seat,
            "P1_strategy": p1_strategy,
            "P2_strategy": p2_strategy,
            "P3_strategy": p3_strategy,
            "P1_rank": p1_rank,
            "P2_rank": p2_rank,
            "P3_rank": p3_rank,
        }
    )

    results = list(rt._players_and_ranks_from_batch(batch, 3))
    assert results == [
        (["A0", "B0"], [0, 1]),
        (["B1", "C1"], [0, 1]),
        (["B6", "A6", "C6"], [0, 1, 1]),
    ]


def test_players_and_ranks_from_batch_without_rank_columns() -> None:
    seat_ranks = pa.array([["P1", "P2"], ["P2", "P1"], []], type=pa.list_(pa.string()))
    batch = pa.table(
        {
            "seat_ranks": seat_ranks,
            "P1_strategy": pa.array(["A", "C", "E"], type=pa.string()),
            "P2_strategy": pa.array(["B", "D", "F"], type=pa.string()),
        }
    )
    results = list(rt._players_and_ranks_from_batch(batch, 2))
    assert results == [(["A", "B"], [0, 1]), (["D", "C"], [0, 1])]


def test_players_and_ranks_from_batch_missing_strategy() -> None:
    batch = pa.table(
        {
            "P1_strategy": pa.array([None], type=pa.string()),
            "P2_strategy": pa.array(["B"], type=pa.string()),
            "P1_rank": pa.array([1], type=pa.int64()),
            "P2_rank": pa.array([2], type=pa.int64()),
            "winner_seat": pa.array([None], type=pa.string()),
        }
    )
    assert list(rt._players_and_ranks_from_batch(batch, 2)) == []


def test_rate_single_pass_resume_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "combined.parquet"
    df = pd.DataFrame(
        {
            "P1_strategy": ["A", "B"],
            "P1_rank": [1, 2],
            "P2_strategy": ["B", "A"],
            "P2_rank": [2, 1],
            "winner_seat": ["P1", "P2"],
        }
    )
    df.to_parquet(source)

    env = trueskill.TrueSkill()
    ck_path = tmp_path / "ts.ckpt.json"
    ratings_ckpt = tmp_path / "ts.checkpoint.parquet"
    rt._save_ratings_parquet(ratings_ckpt, {"A": rt.RatingStats(31.0, 3.0)})
    ck = rt._TSCheckpoint(
        source=str(source),
        row_group=0,
        batch_index=0,
        games_done=5,
        ratings_path=str(ratings_ckpt),
    )
    rt._save_ckpt(ck_path, ck)

    stats, games = rt._rate_single_pass(
        source,
        env=env,
        resume=True,
        checkpoint_path=ck_path,
        ratings_ckpt_path=ratings_ckpt,
        batch_rows=1,
        checkpoint_every_batches=1,
    )

    assert games == 7
    assert {"A", "B"} <= set(stats)
    ck_data = json.loads(ck_path.read_text())
    assert ck_data["games_done"] == games
    assert Path(ck_data["ratings_path"]).exists()


def test_rate_single_pass_without_resume(tmp_path: Path) -> None:
    source = tmp_path / "single.parquet"
    df = pd.DataFrame(
        {
            "P1_strategy": ["A"],
            "P1_rank": [1],
            "P2_strategy": ["B"],
            "P2_rank": [2],
            "winner_seat": ["P1"],
        }
    )
    df.to_parquet(source)

    env = trueskill.TrueSkill()
    stats, games = rt._rate_single_pass(
        source,
        env=env,
        resume=False,
        checkpoint_path=tmp_path / "unused_ck.json",
        ratings_ckpt_path=tmp_path / "unused_ratings.parquet",
        batch_rows=10,
        checkpoint_every_batches=10,
    )
    assert games == 1
    assert set(stats) == {"A", "B"}


def test_rate_single_pass_resume_without_ratings_file(tmp_path: Path) -> None:
    source = tmp_path / "missing_ck.parquet"
    df = pd.DataFrame(
        {
            "P1_strategy": ["A"],
            "P1_rank": [1],
            "P2_strategy": ["B"],
            "P2_rank": [2],
            "winner_seat": ["P1"],
        }
    )
    df.to_parquet(source)

    ck_path = tmp_path / "resume.ckpt.json"
    missing_ratings = tmp_path / "resume.checkpoint.parquet"
    ck = rt._TSCheckpoint(
        source=str(source),
        row_group=0,
        batch_index=0,
        games_done=2,
        ratings_path=str(missing_ratings),
    )
    rt._save_ckpt(ck_path, ck)

    env = trueskill.TrueSkill()
    stats, games = rt._rate_single_pass(
        source,
        env=env,
        resume=True,
        checkpoint_path=ck_path,
        ratings_ckpt_path=missing_ratings,
        batch_rows=5,
        checkpoint_every_batches=5,
    )
    assert games == 3
    assert {"A", "B"} <= set(stats)


def test_rate_single_pass_resume_without_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "no_ck.parquet"
    df = pd.DataFrame(
        {
            "P1_strategy": ["A"],
            "P1_rank": [1],
            "P2_strategy": ["B"],
            "P2_rank": [2],
            "winner_seat": ["P1"],
        }
    )
    df.to_parquet(source)

    env = trueskill.TrueSkill()
    stats, games = rt._rate_single_pass(
        source,
        env=env,
        resume=True,
        checkpoint_path=tmp_path / "absent.ckpt.json",
        ratings_ckpt_path=tmp_path / "absent.checkpoint.parquet",
        batch_rows=5,
        checkpoint_every_batches=10,
    )
    assert games == 1
    assert set(stats) == {"A", "B"}


def test_iter_players_and_ranks_seat_ranks_branch(tmp_path: Path) -> None:
    table = pa.table(
        {
            "seat_ranks": pa.array([["P1", "P2"], ["P1", "P2"], []], type=pa.list_(pa.string())),
            "P1_strategy": pa.array(["A", "AA", "Z"], type=pa.string()),
            "P2_strategy": pa.array(["B", None, "Y"], type=pa.string()),
        }
    )
    row_file = tmp_path / "seat_ranks.parquet"
    pq.write_table(table, row_file)

    result = list(rt._iter_players_and_ranks(row_file, 2, batch_size=1))
    assert result == [(["A", "B"], [0, 1])]


def test_iter_players_and_ranks_fallback_cases(tmp_path: Path) -> None:
    table = pa.table(
        {
            "P1_strategy": pa.array(["A0", None, None, "A2", "A3", None, "A5"], type=pa.string()),
            "P2_strategy": pa.array(["B0", "B1", "B2", "B3", "B4", "B5", "B6"], type=pa.string()),
            "P1_rank": pa.array([1, 1, 1, None, None, None, None], type=pa.int64()),
            "P2_rank": pa.array([2, None, 2, None, None, None, None], type=pa.int64()),
            "winner_seat": pa.array(["P1", "P2", "P2", None, "bad", "P1", "P2"], type=pa.string()),
        }
    )
    row_file = tmp_path / "fallback.parquet"
    pq.write_table(table, row_file)

    result = list(rt._iter_players_and_ranks(row_file, 2, batch_size=1))
    assert result == [(["A0", "B0"], [0, 1]), (["B6", "A5"], [0, 1])]


def test_rate_stream_honors_keepers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    row_file = tmp_path / "rows.parquet"
    row_file.touch()

    emits = [
        (["A", "B"], [0, 1]),
        (["B", "C"], [1, 2]),
        (["A", "C"], [0, 3]),
    ]

    def fake_iter(path: Path, n: int, batch_size: int):
        assert path == row_file
        yield from emits

    monkeypatch.setattr(rt, "_iter_players_and_ranks", fake_iter)
    env = trueskill.TrueSkill()

    stats, games = rt._rate_stream(row_file, 2, ["A", "C"], env, batch_size=10)
    assert games == 1
    assert {"A", "C"} <= set(stats)


def test_rate_stream_without_keeper_filter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    row_file = tmp_path / "rows2.parquet"
    row_file.touch()

    def fake_iter(path: Path, n: int, batch_size: int):
        yield ["A", "B"], [0, 1]

    monkeypatch.setattr(rt, "_iter_players_and_ranks", fake_iter)
    env = trueskill.TrueSkill()
    stats, games = rt._rate_stream(row_file, 2, [], env, batch_size=10)
    assert games == 1
    assert set(stats) == {"A", "B"}


def test_rate_block_worker_up_to_date_guard(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    block = tmp_path / "results" / "2_players"
    per_player_dir = root / "2p"
    per_player_dir.mkdir(parents=True, exist_ok=True)
    block.mkdir(parents=True, exist_ok=True)

    row_file = per_player_dir / "2p_ingested_rows.parquet"
    df = pd.DataFrame(
        {
            "P1_strategy": ["A", "B", "C"],
            "P1_rank": [1, 2, 3],
            "P2_strategy": ["D", "E", "F"],
            "P2_rank": [2, 3, 4],
        }
    )
    df.to_parquet(row_file)

    parquet_path = per_player_dir / "ratings_2.parquet"
    rt._save_ratings_parquet(parquet_path, {"A": rt.RatingStats(10.0, 1.0)})
    newer = row_file.stat().st_mtime + 100.0
    os.utime(parquet_path, (newer, newer))

    result = rt._rate_block_worker(str(block), str(root), "", 100_000, resume=False)
    assert result == ("2", 3)


def test_rate_block_worker_metadata_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "analysis"
    block = tmp_path / "results" / "2_players"
    per_player_dir = root / "2p"
    per_player_dir.mkdir(parents=True, exist_ok=True)
    block.mkdir(parents=True, exist_ok=True)

    row_file = per_player_dir / "2p_ingested_rows.parquet"
    df = pd.DataFrame({"P1_strategy": ["A"], "P1_rank": [1], "P2_strategy": ["B"], "P2_rank": [2]})
    df.to_parquet(row_file)

    parquet_path = per_player_dir / "ratings_2.parquet"
    rt._save_ratings_parquet(parquet_path, {"A": rt.RatingStats(10.0, 1.0)})
    os.utime(parquet_path, (row_file.stat().st_mtime + 10, row_file.stat().st_mtime + 10))

    def boom(path):
        raise OSError("meta")

    monkeypatch.setattr(rt.pq, "read_metadata", boom)
    result = rt._rate_block_worker(str(block), str(root), "", 100_000, resume=False)
    assert result == ("2", 0)


def test_rate_block_worker_without_resume_processes(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    block = tmp_path / "results" / "2_players"
    per_player_dir = root / "2p"
    per_player_dir.mkdir(parents=True, exist_ok=True)
    block.mkdir(parents=True, exist_ok=True)

    row_file = per_player_dir / "2p_ingested_rows.parquet"
    df = pd.DataFrame(
        [
            {"P1_strategy": "A", "P1_rank": 1, "P2_strategy": "B", "P2_rank": 2},
            {"P1_strategy": "B", "P1_rank": 2, "P2_strategy": "A", "P2_rank": 1},
        ]
    )
    df.to_parquet(row_file)

    result = rt._rate_block_worker(
        str(block), str(root), "", batch_rows=10, resume=False, checkpoint_every_batches=10
    )
    assert result == ("2", 2)


def test_rate_block_worker_resume_with_keepers(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    block = tmp_path / "results" / "2_players"
    per_player_dir = root / "2p"
    per_player_dir.mkdir(parents=True, exist_ok=True)
    block.mkdir(parents=True, exist_ok=True)

    keepers = np.array(["A", "B"])
    np.save(block / "keepers_2.npy", keepers)

    row_file = per_player_dir / "2p_ingested_rows.parquet"
    df = pd.DataFrame(
        [
            {"P1_strategy": "C", "P1_rank": 1, "P2_strategy": "D", "P2_rank": 2},
            {"P1_strategy": "A", "P1_rank": 1, "P2_strategy": "B", "P2_rank": 4},
        ]
    )
    df.to_parquet(row_file)

    parquet_path = per_player_dir / "ratings_2.parquet"
    ck_path = per_player_dir / "ratings_2.ckpt.json"
    rk_path = per_player_dir / "ratings_2.checkpoint.parquet"
    rt._save_ratings_parquet(rk_path, {"A": rt.RatingStats(21.0, 2.0)})
    block_ck = rt._BlockCkpt(
        row_file=str(row_file),
        row_group=0,
        batch_index=0,
        games_done=4,
        ratings_path=str(rk_path),
    )
    rt._save_block_ckpt(ck_path, block_ck)

    result = rt._rate_block_worker(
        str(block), str(root), "", batch_rows=1, resume=True, checkpoint_every_batches=1
    )

    assert result == ("2", 5)
    assert parquet_path.exists()
    assert (per_player_dir / "ratings_2.json").exists()
    assert not ck_path.exists()
    assert not rk_path.exists()


def test_rate_block_worker_preserves_checkpoint_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "analysis"
    block = tmp_path / "results" / "2_players"
    per_player_dir = root / "2p"
    per_player_dir.mkdir(parents=True, exist_ok=True)
    block.mkdir(parents=True, exist_ok=True)

    row_file = per_player_dir / "2p_ingested_rows.parquet"
    df = pd.DataFrame(
        [
            {"P1_strategy": "A", "P1_rank": 1, "P2_strategy": "B", "P2_rank": 2},
            {"P1_strategy": "B", "P1_rank": 2, "P2_strategy": "A", "P2_rank": 1},
        ]
    )
    df.to_parquet(row_file)

    ck_path = per_player_dir / "ratings_2.ckpt.json"
    rk_path = per_player_dir / "ratings_2.checkpoint.parquet"

    original_save = rt._save_ratings_parquet

    def fail_on_final(path: Path, ratings: Mapping[str, rt.RatingStats]) -> None:
        if Path(path) == per_player_dir / "ratings_2.parquet":
            raise RuntimeError("boom")
        original_save(path, ratings)

    monkeypatch.setattr(rt, "_save_ratings_parquet", fail_on_final)

    with pytest.raises(RuntimeError):
        rt._rate_block_worker(
            str(block), str(root), "", batch_rows=1, resume=True, checkpoint_every_batches=1
        )

    assert ck_path.exists()
    assert rk_path.exists()


def test_rate_block_worker_resume_missing_checkpoint_ratings_file(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    block = tmp_path / "results" / "2_players"
    per_player_dir = root / "2p"
    per_player_dir.mkdir(parents=True, exist_ok=True)
    block.mkdir(parents=True, exist_ok=True)

    row_file = per_player_dir / "2p_ingested_rows.parquet"
    df = pd.DataFrame(
        [
            {"P1_strategy": "A", "P1_rank": 1, "P2_strategy": "B", "P2_rank": 2},
        ]
    )
    df.to_parquet(row_file)

    ck_path = per_player_dir / "ratings_2.ckpt.json"
    missing_rk = per_player_dir / "ratings_2.checkpoint.parquet"
    ck = rt._BlockCkpt(
        row_file=str(row_file),
        row_group=0,
        batch_index=0,
        games_done=0,
        ratings_path=str(missing_rk),
    )
    rt._save_block_ckpt(ck_path, ck)

    result = rt._rate_block_worker(
        str(block), str(root), "", batch_rows=10, resume=True, checkpoint_every_batches=10
    )
    assert result[0] == "2"
    assert result[1] >= 1


def test_run_trueskill_handles_worker_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = tmp_path / "results"
    block2 = data_root / "2_players"
    block3 = data_root / "3_players"
    block2.mkdir(parents=True, exist_ok=True)
    block3.mkdir(parents=True, exist_ok=True)
    (data_root / "manifest.yaml").write_text("seed: 0\n")

    analysis_root = tmp_path / "analysis"

    per_block = {
        "2": ({"A": rt.RatingStats(10.0, 1.0)}, 8),
        "3": ({"C": rt.RatingStats(15.0, 2.0)}, 6),
    }

    def immediate_write(
        table: pa.Table, path: Path | str, *, codec: Compression = "snappy"
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression=normalize_compression(codec))

    class _Future:
        def __init__(self, fn, args, kwargs, fail: bool):
            self._fn = fn
            self._args = args
            self._kwargs = kwargs
            self._fail = fail
            if not fail:
                self._result = fn(*args, **kwargs)

        def result(self):
            if self._fail:
                raise RuntimeError("boom")
            return self._result

    class _Executor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def submit(self, fn, *args, **kwargs):
            block_name = Path(args[0]).name.split("_")[0]
            fail = block_name == "3"
            return _Future(fn, args, kwargs, fail)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    exceptions: list[dict] = []

    def fake_exception(msg, *args, extra=None, **kwargs):
        exceptions.append(extra or {})

    def fake_rate_block_worker(
        block_dir: str,
        root_dir: str,
        suffix: str,
        batch_rows: int,
        *,
        resume: bool,
        checkpoint_every_batches: int,
        env_kwargs: dict | None,
        row_data_dir: str | None = None,
        curated_rows_name: str | None = None,
    ) -> tuple[str, int]:
        player_count = Path(block_dir).name.split("_")[0]
        stats, games = per_block[player_count]
        per_player_dir = Path(root_dir) / f"{player_count}p"
        per_player_dir.mkdir(parents=True, exist_ok=True)
        rt._save_ratings_parquet(
            per_player_dir / f"ratings_{player_count}{suffix}.parquet", stats
        )
        return player_count, games

    monkeypatch.setattr(rt, "write_parquet_atomic", immediate_write)
    monkeypatch.setattr(rt.cf, "ProcessPoolExecutor", _Executor)
    monkeypatch.setattr(rt.cf, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(rt, "_rate_block_worker", fake_rate_block_worker)
    monkeypatch.setattr(rt.LOGGER, "exception", fake_exception)

    rt.run_trueskill(root=analysis_root, dataroot=data_root, workers=3)

    assert exceptions and exceptions[0]["block"] == "3_players"
    assert (analysis_root / "pooled" / "ratings_pooled.parquet").exists()


def test_run_trueskill_reads_rows_from_data_dir(tmp_path: Path) -> None:
    data_root = tmp_path / "results_seed_21"
    block = data_root / "2_players"
    block.mkdir(parents=True, exist_ok=True)
    (data_root / "manifest.yaml").write_text("seed: 21\n")

    analysis_root = tmp_path / "analysis" / "03_trueskill"
    row_data_dir = tmp_path / "analysis" / "01_combine" / "data"
    per_n_dir = row_data_dir / "2p"
    per_n_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {"P1_strategy": "Alpha", "P1_rank": 1, "P2_strategy": "Beta", "P2_rank": 2},
            {"P1_strategy": "Beta", "P1_rank": 2, "P2_strategy": "Alpha", "P2_rank": 1},
        ]
    )
    curated_name = "custom_rows.parquet"
    df.to_parquet(per_n_dir / curated_name)

    rt.run_trueskill(
        root=analysis_root,
        dataroot=data_root,
        row_data_dir=row_data_dir,
        curated_rows_name=curated_name,
        workers=1,
        batch_rows=10,
    )

    ratings_path = analysis_root / "2p" / "ratings_2.parquet"
    assert ratings_path.exists()


def test_run_trueskill_rebuilds_outdated_pooled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = tmp_path / "results"
    block = data_root / "2_players"
    block.mkdir(parents=True, exist_ok=True)
    (data_root / "manifest.yaml").write_text("seed: 0\n")

    analysis_root = tmp_path / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    per_block = {"2": ({"A": rt.RatingStats(12.0, 1.5)}, 10)}

    def immediate_write(
        table: pa.Table, path: Path | str, *, codec: Compression = "snappy"
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression=normalize_compression(codec))

    def fake_rate_block_worker(
        block_dir: str,
        root_dir: str,
        suffix: str,
        batch_rows: int,
        *,
        resume: bool,
        checkpoint_every_batches: int,
        env_kwargs: dict | None,
        row_data_dir: str | None = None,
        curated_rows_name: str | None = None,
    ) -> tuple[str, int]:
        stats, games = per_block["2"]
        per_player_dir = Path(root_dir) / "2p"
        per_player_dir.mkdir(parents=True, exist_ok=True)
        rt._save_ratings_parquet(per_player_dir / f"ratings_2{suffix}.parquet", stats)
        return "2", games

    tier_calls: list[tuple[dict[str, float], dict[str, float]]] = []

    def fake_build_tiers(
        means: dict[str, float], stdevs: dict[str, float], **_: object
    ) -> dict[str, int]:
        tier_calls.append((means, stdevs))
        return dict.fromkeys(means, 1)

    pooled_path = analysis_root / "pooled" / "ratings_pooled.parquet"
    pooled_path.parent.mkdir(parents=True, exist_ok=True)
    pooled_path.touch()
    old_time = pooled_path.stat().st_mtime - 200.0
    os.utime(pooled_path, (old_time, old_time))

    monkeypatch.setattr(rt, "write_parquet_atomic", immediate_write)
    monkeypatch.setattr(rt, "_rate_block_worker", fake_rate_block_worker)
    monkeypatch.setattr(rt, "build_tiers", fake_build_tiers)
    monkeypatch.setattr(rt.os, "cpu_count", lambda: 2)

    rt.run_trueskill(root=analysis_root, dataroot=data_root, workers=1)

    assert tier_calls
    assert pooled_path.stat().st_mtime > old_time
    assert (analysis_root / "pooled" / "ratings_pooled.json").exists()
    assert (analysis_root / "tiers.json").exists()


def test_run_trueskill_all_seeds_resolves_per_seed_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pair_root = tmp_path / "results_seed_pair_12_13"
    seed12_root = pair_root / "results_seed_12"
    seed13_root = pair_root / "results_seed_13"
    seed12_analysis = seed12_root / "analysis"
    seed13_analysis = seed13_root / "analysis"
    expected_seed12_row_data_dir = seed12_analysis / "01_curate"
    expected_seed13_row_data_dir = seed13_analysis / "01_curate"
    expected_seed12_row_data_dir.mkdir(parents=True, exist_ok=True)
    expected_seed13_row_data_dir.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig(
        io=IOConfig(
            results_dir_prefix=seed12_root,
            interseed_input_dir=seed12_analysis,
            interseed_input_layout={"curate": "01_curate"},
        ),
        sim=SimConfig(seed=12, n_players_list=[2], seed_list=[12, 13], seed_pair=(12, 13)),
    )
    cfg.analysis.tiering_seeds = [12, 13]
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))

    seen_seed_inputs: list[tuple[int, Path | None, Path | None]] = []

    def fake_run_trueskill(
        output_seed: int = 0,
        root: Path | None = None,
        dataroot: Path | None = None,
        row_data_dir: Path | None = None,
        curated_rows_name: str | None = None,
        workers: int | None = None,
        batch_rows: int = 100_000,
        resume_per_n: bool = True,
        checkpoint_every_batches: int = 500,
        env_kwargs: Mapping[str, float] | rt.TrueSkillInitKwargs | None = None,
        pooled_weights_by_k: Mapping[int, float] | None = None,
        tiering_z: float | None = None,
        tiering_min_gap: float | None = None,
    ) -> None:
        _ = (
            dataroot,
            curated_rows_name,
            workers,
            batch_rows,
            resume_per_n,
            checkpoint_every_batches,
            env_kwargs,
            pooled_weights_by_k,
            tiering_z,
            tiering_min_gap,
        )
        seen_seed_inputs.append(
            (
                int(output_seed),
                Path(dataroot) if dataroot is not None else None,
                Path(row_data_dir) if row_data_dir is not None else None,
            )
        )
        assert root is not None
        root = Path(root)
        rt._save_ratings_parquet(
            root / "2p" / f"ratings_2_seed{output_seed}.parquet",
            {"alpha": rt.RatingStats(25.0, 8.0)},
        )
        rt._save_ratings_parquet(
            root / "pooled" / f"ratings_pooled_seed{output_seed}.parquet",
            {"alpha": rt.RatingStats(25.0, 8.0)},
        )

    monkeypatch.setattr(rt, "run_trueskill", fake_run_trueskill)
    rt.run_trueskill_all_seeds(cfg)

    assert seen_seed_inputs == [
        (12, seed12_root, expected_seed12_row_data_dir),
        (13, seed13_root, expected_seed13_row_data_dir),
    ]
