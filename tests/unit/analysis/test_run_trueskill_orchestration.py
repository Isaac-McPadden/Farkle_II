import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import farkle.analysis.run_trueskill as rt
from farkle.utils.types import Compression, normalize_compression


def test_run_trueskill_pooling_and_short_circuit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "results"
    block2 = data_root / "2_players"
    block3 = data_root / "3_players"
    block2.mkdir(parents=True, exist_ok=True)
    block3.mkdir(parents=True, exist_ok=True)

    analysis_root = tmp_path / "analysis"

    per_block: Dict[str, Tuple[dict[str, rt.RatingStats], int]] = {
        "2": (
            {
                "A": rt.RatingStats(20.0, 2.0),
                "B": rt.RatingStats(15.0, 1.5),
            },
            5,
        ),
        "3": (
            {
                "A": rt.RatingStats(40.0, 4.0),
                "C": rt.RatingStats(30.0, 3.0),
            },
            15,
        ),
    }

    def immediate_write(
        table: pa.Table, path: Path | str, *, codec: Compression = "snappy"
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression=normalize_compression(codec))

    created_executors: list["_FakeExecutor"] = []

    class _FakeFuture:
        def __init__(self, fn, args, kwargs):
            try:
                self._result = fn(*args, **kwargs)
                self._error = None
            except Exception as exc:  # pragma: no cover - defensive
                self._result = None
                self._error = exc

        def result(self):
            if self._error is not None:  # pragma: no cover - defensive
                raise self._error
            return self._result

    class _FakeExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers
            self.submissions = []
            created_executors.append(self)

        def submit(self, fn, *args, **kwargs):
            self.submissions.append((fn, args, kwargs))
            return _FakeFuture(fn, args, kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_as_completed(futures):
        return list(futures)

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
        per_player = Path(root_dir) / f"{player_count}p"
        per_player.mkdir(parents=True, exist_ok=True)
        out_path = per_player / f"ratings_{player_count}{suffix}.parquet"
        if not out_path.exists():
            rt._save_ratings_parquet(out_path, stats)
        return player_count, games

    tier_calls: list[tuple[dict[str, float], dict[str, float]]] = []

    def fake_build_tiers(
        means: dict[str, float], stdevs: dict[str, float], **_: object
    ) -> dict[str, int]:
        tier_calls.append((means, stdevs))
        return {name: idx + 1 for idx, name in enumerate(sorted(means))}

    monkeypatch.setattr(rt, "write_parquet_atomic", immediate_write)
    monkeypatch.setattr(rt.cf, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(rt.cf, "as_completed", fake_as_completed)
    monkeypatch.setattr(rt, "_rate_block_worker", fake_rate_block_worker)
    monkeypatch.setattr(rt, "build_tiers", fake_build_tiers)
    monkeypatch.setattr(rt.os, "cpu_count", lambda: 2)

    rt.run_trueskill(root=analysis_root, dataroot=data_root, workers=8)

    assert created_executors and created_executors[0].max_workers == 2

    pooled_path = analysis_root / "pooled" / "ratings_pooled.parquet"
    json_path = analysis_root / "pooled" / "ratings_pooled.json"
    tiers_path = analysis_root / "tiers.json"
    assert pooled_path.exists() and json_path.exists() and tiers_path.exists()

    pooled = rt._load_ratings_parquet(pooled_path)
    assert pooled.keys() == {"A", "B", "C"}
    assert pooled["A"].mu == pytest.approx(1000.0 / 35.0)
    assert pooled["A"].sigma == pytest.approx(math.sqrt(16.0 / 35.0))
    assert pooled["B"].mu == pytest.approx(15.0)
    assert pooled["B"].sigma == pytest.approx(math.sqrt(9.0 / 20.0))
    assert pooled["C"].mu == pytest.approx(30.0)
    assert pooled["C"].sigma == pytest.approx(math.sqrt(3.0 / 5.0))

    pooled_json = json.loads(json_path.read_text())
    for key, stats in pooled.items():
        assert pooled_json[key]["mu"] == pytest.approx(stats.mu)
        assert pooled_json[key]["sigma"] == pytest.approx(stats.sigma)

    assert tier_calls and tiers_path.read_text()

    before = pooled_path.stat().st_mtime + 10.0
    os.utime(pooled_path, (before, before))
    tier_calls.clear()

    rt.run_trueskill(root=analysis_root, dataroot=data_root, workers=8)

    assert not tier_calls
    assert pooled_path.stat().st_mtime == before


def test_run_trueskill_skips_zero_game_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "results"
    block2 = data_root / "2_players"
    block3 = data_root / "3_players"
    block2.mkdir(parents=True, exist_ok=True)
    block3.mkdir(parents=True, exist_ok=True)

    analysis_root = tmp_path / "analysis"

    per_block = {
        "2": ("A", rt.RatingStats(10.0, 2.0), 12),
        "3": ("B", rt.RatingStats(50.0, 5.0), 0),
    }

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
        player_count = Path(block_dir).name.split("_")[0]
        name, stats, games = per_block[player_count]
        per_player = Path(root_dir) / f"{player_count}p"
        per_player.mkdir(parents=True, exist_ok=True)
        rt._save_ratings_parquet(
            per_player / f"ratings_{player_count}{suffix}.parquet", {name: stats}
        )
        return player_count, games

    monkeypatch.setattr(rt, "write_parquet_atomic", immediate_write)
    monkeypatch.setattr(rt, "_rate_block_worker", fake_rate_block_worker)

    rt.run_trueskill(root=analysis_root, dataroot=data_root, workers=1)

    pooled = rt._load_ratings_parquet(analysis_root / "pooled" / "ratings_pooled.parquet")
    assert set(pooled) == {"A"}
    assert pooled["A"].mu == pytest.approx(10.0)
