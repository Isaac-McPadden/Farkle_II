import pandas as pd
import trueskill

import farkle.run_trueskill as rt


def test_read_manifest_seed(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text("seed: 42\n")
    assert rt._read_manifest_seed(path) == 42
    assert rt._read_manifest_seed(tmp_path / "missing.yaml") == 0


def test_load_winners_parquet(tmp_path):
    block = tmp_path / "b_players"
    row_dir = block / "1_rows"
    row_dir.mkdir(parents=True)
    pd.DataFrame({"winner_strategy": ["A"]}).to_parquet(row_dir / "a.parquet")
    pd.DataFrame({"winner_strategy": ["B", "A"]}).to_parquet(row_dir / "b.parquet")
    winners = rt._load_winners(block)
    assert sorted(winners) == ["A", "A", "B"]


def test_load_winners_csv(tmp_path):
    block = tmp_path / "c_players"
    block.mkdir()
    pd.DataFrame({"winner": ["X", "Y"]}).to_csv(block / "winners.csv", index=False)
    assert rt._load_winners(block) == ["X", "Y"]


def test_update_ratings_simple():
    env = trueskill.TrueSkill()
    winners = ["A", "B", "A", "C", "A"]
    keepers = ["A", "B"]

    result = rt._update_ratings(winners, keepers, env)

    ratings = {k: env.create_rating() for k in keepers}
    dummy = env.create_rating()
    for w in winners:
        if w not in ratings:
            continue
        ratings[w], dummy = trueskill.rate_1vs1(ratings[w], dummy, env=env)
    expected = {k: (r.mu, r.sigma) for k, r in ratings.items()}

    assert result == expected