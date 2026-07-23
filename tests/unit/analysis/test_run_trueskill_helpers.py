"""Unit tests for helpers in :mod:`farkle.analysis.run_trueskill`."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import farkle.analysis.run_trueskill as rt


def test_rating_artifact_paths_are_canonical_only(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    paths = rt._rating_artifact_paths(root, "2", "_seed3")
    assert paths["parquet"] == root / "by_k" / "2p" / "ratings_2_seed3.parquet"
    assert set(paths) == {"parquet", "json", "ckpt", "checkpoint", "dir"}


def test_iter_rating_parquets_deduplicates_filters_and_sorts_stably(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "by_k" / "2p").mkdir(parents=True)
    (root / "by_k" / "2p" / "ratings_2_seed0.parquet").touch()
    (root / "by_k" / "3p").mkdir()
    (root / "by_k" / "3p" / "ratings_3_seed0.parquet").touch()
    (root / "ratings_k_weighted_seed0.parquet").touch()
    (root / "data" / "2p").mkdir(parents=True)
    (root / "data" / "2p" / "ratings_2_seed0.parquet").touch()

    legacy_root = tmp_path / "legacy"
    (legacy_root / "2p").mkdir(parents=True)
    (legacy_root / "2p" / "ratings_2_seed0.parquet").touch()
    (legacy_root / "1p").mkdir(parents=True)
    (legacy_root / "1p" / "ratings_1_seed0.parquet").touch()

    results = rt._iter_rating_parquets(root, "_seed0")
    assert len(results) == 2
    assert [rt._player_count_from_stem(path.stem) for path in results] == [2, 3]

    resolved = [path.resolve().as_posix() for path in results]
    assert resolved == sorted(
        resolved,
        key=lambda p: (
            rt._player_count_from_stem(Path(p).stem) or float("inf"),
            p,
        ),
    )


def test_player_count_from_stem_handles_variants():
    assert rt._player_count_from_stem("ratings_2_seed1") == 2
    assert rt._player_count_from_stem("ratings_3p_seed1") is None
    assert rt._player_count_from_stem("invalid") is None


def _write_results_block(block: Path, winners: list[str]) -> None:
    block.mkdir(parents=True, exist_ok=True)
    prefix = block.name.split("_")[0]
    np.save(block / f"keepers_{prefix}.npy", np.array(["A", "B"]))

    rows = []
    for winner in winners:
        if winner == "A":
            rows.append(
                {
                    "winner_seat": "P1",
                    "winner_strategy": "A",
                    "P1_strategy": "A",
                    "P2_strategy": "B",
                    "P1_rank": 1,
                    "P2_rank": 2,
                }
            )
        else:
            rows.append(
                {
                    "winner_seat": "P2",
                    "winner_strategy": "B",
                    "P1_strategy": "A",
                    "P2_strategy": "B",
                    "P1_rank": 2,
                    "P2_rank": 1,
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(block / f"{prefix}p_rows.parquet")


def test_run_trueskill_writes_only_root_k_ratings(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    res_dir = data_root / "results"
    res_dir.mkdir(parents=True)

    _write_results_block(res_dir / "2_players", ["A", "B", "A"])
    (res_dir / "manifest.yaml").write_text("seed: 0\n")
    row_data_dir = data_root / "curate"
    canonical_rows = row_data_dir / "by_k" / "2p" / "game_rows.parquet"
    canonical_rows.parent.mkdir(parents=True, exist_ok=True)
    pd.read_parquet(res_dir / "2_players" / "2p_rows.parquet").to_parquet(canonical_rows)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rt.run_trueskill(
            root=data_root,
            row_data_dir=row_data_dir,
            curated_rows_name="game_rows.parquet",
            workers=1,
            batch_rows=2,
            cell_freshness_sha256="a" * 64,
        )
    finally:
        os.chdir(cwd)

    ratings_dir = data_root
    ratings_2 = rt._load_ratings_parquet(ratings_dir / "by_k" / "2p" / "ratings_2_seed0.parquet")
    assert set(ratings_2)
    assert not (data_root / "across_k").exists()


def test_run_trueskill_with_seed_suffix(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    res_dir = data_root / "results"
    res_dir.mkdir(parents=True)

    _write_results_block(res_dir / "2_players", ["A", "B"])
    (res_dir / "manifest.yaml").write_text("seed: 0\n")
    row_data_dir = data_root / "curate"
    canonical_rows = row_data_dir / "by_k" / "2p" / "game_rows.parquet"
    canonical_rows.parent.mkdir(parents=True, exist_ok=True)
    pd.read_parquet(res_dir / "2_players" / "2p_rows.parquet").to_parquet(canonical_rows)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rt.run_trueskill(
            output_seed=3,
            root=data_root,
            row_data_dir=row_data_dir,
            curated_rows_name="game_rows.parquet",
            workers=1,
            batch_rows=2,
            cell_freshness_sha256="a" * 64,
        )
    finally:
        os.chdir(cwd)

    assert (data_root / "by_k" / "2p" / "ratings_2_seed3.parquet").exists()
    assert not (data_root / "across_k").exists()
