import pandas as pd
import trueskill

import farkle.run_trueskill as rt


def test_read_manifest_seed(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text("seed: 42\n")
    assert rt._read_manifest_seed(path) == 42
    assert rt._read_manifest_seed(tmp_path / "missing.yaml") == 0


def test_load_ranked_games_parquet(tmp_path):
    block   = tmp_path / "b_players"
    row_dir = block / "1_rows"
    row_dir.mkdir(parents=True)
    pd.DataFrame({"winner_strategy": ["A"]}).to_parquet(row_dir / "a.parquet")
    pd.DataFrame({"winner_strategy": ["B", "A"]}).to_parquet(row_dir / "b.parquet")

    games = rt._load_ranked_games(block)
    assert sorted(games) == [["A"], ["A"], ["B"]]  # ← list-of-lists


def test_load_ranked_games_rank_based(tmp_path):
    block   = tmp_path / "r_players"
    row_dir = block / "1_rows"
    row_dir.mkdir(parents=True)
    df = pd.DataFrame({
        "P1_strategy": ["A"],
        "P1_rank":     [1],
        "P2_strategy": ["B"],
        "P2_rank":     [2],
    })
    df.to_parquet(row_dir / "rows.parquet")

    games = rt._load_ranked_games(block)
    assert games == [["A", "B"]]  # full ranking


def test_load_ranked_games_csv(tmp_path):
    block = tmp_path / "c_players"
    block.mkdir()
    pd.DataFrame({"winner": ["X", "Y"]}).to_csv(block / "winners.csv", index=False)

    games = rt._load_ranked_games(block)
    assert games == [["X"], ["Y"]]  # list-of-lists


def test_update_ratings_ranked():
    env = trueskill.TrueSkill()

    # original winner stream:  A, B, A, C, A
    games = [
        ["A", "B"],   # A beats B
        ["B", "A"],   # B beats A
        ["A", "C"],   # A beats C
        ["C", "A"],   # C beats A
        ["A", "B"],   # A beats B again
    ]

    keepers = ["A", "B"]                     # only rate these two
    result = rt._update_ratings(games, keepers, env)

    # build the expected ratings by replaying the same tables
    ratings = {k: env.create_rating() for k in keepers}
    for g in games:
      p = [s for s in g if s in keepers]
      if len(p) < 2:
          continue
      new = env.rate([[ratings[p[0]]], [ratings[p[1]]]], ranks=[0, 1])
      ratings[p[0]], ratings[p[1]] = new[0][0], new[1][0]

    expected = {k: (r.mu, r.sigma) for k, r in ratings.items()}
    assert result == expected