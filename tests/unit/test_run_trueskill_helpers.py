import os
import pickle

import numpy as np
import pandas as pd
import pytest
import trueskill

import farkle.run_trueskill as rt


def test_read_manifest_seed(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text("seed: 42\n")
    assert rt._read_manifest_seed(path) == 42
    assert rt._read_manifest_seed(tmp_path / "missing.yaml") == 0


def test_read_row_shards(tmp_path):
    row_dir = tmp_path / "rows"
    row_dir.mkdir()
    pd.DataFrame({"w": [1]}).to_parquet(row_dir / "a.parquet")
    pd.DataFrame({"w": [2]}).to_parquet(row_dir / "b.parquet")

    df = rt._read_row_shards(row_dir)
    assert sorted(df["w"]) == [1, 2]


def test_read_winners_csv(tmp_path):
    block = tmp_path
    pd.DataFrame({"winner": ["A", "B"]}).to_csv(block / "winners.csv", index=False)

    df = rt._read_winners_csv(block)
    assert list(df["winner"]) == ["A", "B"]


def test_read_loose_parquets(tmp_path):
    block = tmp_path / "block"
    block.mkdir()
    pd.DataFrame({"w": ["X"]}).to_parquet(block / "a.parquet")
    pd.DataFrame({"w": ["Y"]}).to_parquet(block / "b.parquet")

    df = rt._read_loose_parquets(block)
    assert sorted(df["w"]) == ["X", "Y"]
    assert rt._read_loose_parquets(tmp_path / "empty") is None


def test_load_ranked_games_parquet(tmp_path):
    block = tmp_path / "b_players"
    row_dir = block / "1_rows"
    row_dir.mkdir(parents=True)
    pd.DataFrame({"winner_strategy": ["A"]}).to_parquet(row_dir / "a.parquet")
    pd.DataFrame({"winner_strategy": ["B", "A"]}).to_parquet(row_dir / "b.parquet")

    games = rt._load_ranked_games(block)
    assert sorted(games) == [["A"], ["A"], ["B"]]  # ← list-of-lists


def test_load_ranked_games_rank_based(tmp_path):
    block = tmp_path / "r_players"
    row_dir = block / "1_rows"
    row_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "P1_strategy": ["A"],
            "P1_rank": [1],
            "P2_strategy": ["B"],
            "P2_rank": [2],
        }
    )
    df.to_parquet(row_dir / "rows.parquet")

    games = rt._load_ranked_games(block)
    assert games == [["A", "B"]]  # full ranking


def test_load_ranked_games_csv(tmp_path):
    block = tmp_path / "c_players"
    block.mkdir()
    pd.DataFrame({"winner": ["X", "Y"]}).to_csv(block / "winners.csv", index=False)

    games = rt._load_ranked_games(block)
    assert games == [["X"], ["Y"]]  # list-of-lists


def test_load_ranked_games_multi_row_dirs(tmp_path):
    block   = tmp_path / "m_players"
    row1    = block / "1_rows"
    row2    = block / "2_rows"
    row1.mkdir(parents=True)
    row2.mkdir()
    pd.DataFrame({"winner_strategy": ["A"]}).to_parquet(row1 / "a.parquet")
    pd.DataFrame({"winner_strategy": ["B"]}).to_parquet(row2 / "b.parquet")

    games = rt._load_ranked_games(block)
    assert sorted(games) == [["A"], ["B"]]


def test_update_ratings_ranked():
    env = trueskill.TrueSkill()

    # original winner stream:  A, B, A, C, A
    games = [
        ["A", "B"],  # A beats B
        ["B", "A"],  # B beats A
        ["A", "C"],  # A beats C
        ["C", "A"],  # C beats A
        ["A", "B"],  # A beats B again
    ]

    keepers = ["A", "B"]  # only rate these two
    result = rt._update_ratings(games, keepers, env)

    # build the expected ratings by replaying the same tables
    ratings = {k: env.create_rating() for k in keepers}
    for g in games:
        p = [s for s in g if s in keepers]
        if len(p) < 2:
            continue
        new = env.rate([[ratings[p[0]]], [ratings[p[1]]]], ranks=[0, 1])
        ratings[p[0]], ratings[p[1]] = new[0][0], new[1][0]

    expected = {k: rt.RatingStats(r.mu, r.sigma) for k, r in ratings.items()}
    assert result == expected


def test_load_ranked_games_empty(tmp_path):
    block = tmp_path / "empty_players"
    block.mkdir()
    assert rt._load_ranked_games(block) == []


def test_load_ranked_games_missing_strategy(tmp_path):
    block = tmp_path / "bad_players"
    row_dir = block / "2_rows"
    row_dir.mkdir(parents=True)
    df = pd.DataFrame({"P1_rank": [1], "P2_rank": [2], "P2_strategy": ["B"]})
    df.to_parquet(row_dir / "rows.parquet")
    with pytest.raises(KeyError):
        rt._load_ranked_games(block)


def test_run_trueskill_incomplete_block(tmp_path):
    data_root = tmp_path / "data"
    res_dir = data_root / "results"
    res_dir.mkdir(parents=True)

    block_good = res_dir / "2_players"
    block_good.mkdir()
    np.save(block_good / "keepers_2.npy", np.array(["A", "B"]))
    pd.DataFrame({"winner_strategy": ["A", "B"]}).to_csv(block_good / "winners.csv", index=False)

    block_empty = res_dir / "3_players"
    block_empty.mkdir()

    (res_dir / "manifest.yaml").write_text("seed: 0\n")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rt.run_trueskill()
    finally:
        os.chdir(cwd)

    with open(data_root / "ratings_2.pkl", "rb") as fh:
        good = pickle.load(fh)
    with open(data_root / "ratings_3.pkl", "rb") as fh:
        empty = pickle.load(fh)
    with open(data_root / "ratings_pooled.pkl", "rb") as fh:
        pooled = pickle.load(fh)

    assert good
    assert empty == {}
    assert set(pooled) == set(good)


def test_run_trueskill_with_suffix(tmp_path):
    data_root = tmp_path / "data"
    res_dir = data_root / "results"
    res_dir.mkdir(parents=True)

    block = res_dir / "2_players"
    block.mkdir()
    np.save(block / "keepers_2.npy", np.array(["A", "B"]))
    pd.DataFrame({"winner_strategy": ["A", "B"]}).to_csv(block / "winners.csv", index=False)

    (res_dir / "manifest.yaml").write_text("seed: 0\n")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rt.run_trueskill(output_seed=1)
    finally:
        os.chdir(cwd)

    assert (data_root / "ratings_2_seed1.pkl").exists()
    assert (data_root / "ratings_pooled_seed1.pkl").exists()
