"""Unit tests for helpers in :mod:`farkle.analysis.run_trueskill`."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import trueskill

import farkle.analysis.run_trueskill as rt


def test_read_manifest_seed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("seed: 42\n")
    assert rt._read_manifest_seed(path) == 42
    assert rt._read_manifest_seed(tmp_path / "missing.yaml") == 0


def test_find_combined_parquet(tmp_path: Path) -> None:
    base = tmp_path / "root1"
    combined = base / "analysis" / "02_combine" / "pooled" / "all_ingested_rows.parquet"
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.touch()
    assert rt._find_combined_parquet(base) == combined

    base2 = tmp_path / "root2"
    direct = base2 / "data" / "all_n_players_combined" / "all_ingested_rows.parquet"
    direct.parent.mkdir(parents=True, exist_ok=True)
    direct.touch()
    assert rt._find_combined_parquet(base2) == direct

    base3 = tmp_path / "root3"
    fallback = base3 / "all_ingested_rows.parquet"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.touch()
    assert rt._find_combined_parquet(base3) == fallback


def test_rating_artifact_paths_and_ensure_new_location(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir(parents=True)

    # create legacy file that should be migrated
    legacy_file = legacy_root / "data" / "2p" / "ratings_2_seed3.parquet"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.touch()

    paths = rt._rating_artifact_paths(root, "2", "_seed3", legacy_root=legacy_root)
    migrated = rt._ensure_new_location(paths["parquet"], *paths["legacy_parquet"])

    assert migrated == paths["parquet"]
    assert migrated.exists()


def test_iter_rating_parquets_deduplicates_and_filters(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "2p").mkdir()
    (root / "2p" / "ratings_2_seed0.parquet").touch()
    (root / "ratings_k_weighted_seed0.parquet").touch()
    (root / "data" / "2p").mkdir(parents=True)
    (root / "data" / "2p" / "ratings_2_seed0.parquet").touch()

    results = rt._iter_rating_parquets(root, "_seed0")
    assert len(results) == 2
    assert all(path.stem.startswith("ratings_2") for path in results)


def test_player_count_from_stem_handles_variants():
    assert rt._player_count_from_stem("ratings_2_seed1") == 2
    assert rt._player_count_from_stem("ratings_3p_seed1") == 3
    assert rt._player_count_from_stem("invalid") is None


def test_update_ratings_adds_missing_strategies() -> None:
    env = trueskill.TrueSkill()
    games = [["A", "B"], ["B", "A"], ["C", "A"]]
    keepers = ["A", "B"]

    ratings = rt._update_ratings(games, keepers, env)
    assert {"A", "B", "C"} <= set(ratings)
    assert ratings["C"] == rt.RatingStats(rt.DEFAULT_RATING.mu, rt.DEFAULT_RATING.sigma)


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


def test_run_trueskill_skips_empty_blocks(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    res_dir = data_root / "results"
    res_dir.mkdir(parents=True)

    _write_results_block(res_dir / "2_players", ["A", "B", "A"])
    (res_dir / "3_players").mkdir()
    (res_dir / "manifest.yaml").write_text("seed: 0\n")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rt.run_trueskill(root=data_root, workers=1, batch_rows=2)
    finally:
        os.chdir(cwd)

    ratings_dir = data_root
    ratings_2 = rt._load_ratings_parquet(ratings_dir / "2p" / "ratings_2_seed0.parquet")
    ratings3_path = ratings_dir / "3p" / "ratings_3_seed0.parquet"
    ratings_3 = rt._load_ratings_parquet(ratings3_path) if ratings3_path.exists() else {}
    pooled = rt._load_ratings_parquet(data_root / "pooled" / "ratings_k_weighted_seed0.parquet")

    assert set(ratings_2)
    assert ratings_3 == {}
    assert set(pooled) == set(ratings_2)


def test_run_trueskill_with_seed_suffix(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    res_dir = data_root / "results"
    res_dir.mkdir(parents=True)

    _write_results_block(res_dir / "2_players", ["A", "B"])
    (res_dir / "manifest.yaml").write_text("seed: 0\n")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rt.run_trueskill(output_seed=3, root=data_root, workers=1, batch_rows=2)
    finally:
        os.chdir(cwd)

    assert (data_root / "2p" / "ratings_2_seed3.parquet").exists()
    assert (data_root / "pooled" / "ratings_k_weighted_seed3.parquet").exists()
