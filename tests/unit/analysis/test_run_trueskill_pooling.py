import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from farkle.analysis import run_trueskill


def test_pooled_ratings_are_weighted_mean(tmp_path):
    data_root = tmp_path / "data"
    res_root = data_root / "results"

    # --- block with A beating B -------------------------------------------------
    block2 = res_root / "2_players"
    block2.mkdir(parents=True)
    df2 = pd.DataFrame(
        {
            "P1_strategy": ["A"] * 3,
            "P1_rank": [1] * 3,
            "P2_strategy": ["B"] * 3,
            "P2_rank": [2] * 3,
        }
    )
    df2.to_parquet(block2 / "2p_rows.parquet")
    np.save(block2 / "keepers_2.npy", np.array(["A", "B"]))

    # --- block with B beating A (extra player ignored) -------------------------
    block3 = res_root / "3_players"
    block3.mkdir(parents=True)
    df3 = pd.DataFrame(
        {
            "P1_strategy": ["B"] * 6,
            "P1_rank": [1] * 6,
            "P2_strategy": ["A"] * 6,
            "P2_rank": [2] * 6,
            "P3_strategy": ["C"] * 6,
            "P3_rank": [3] * 6,
        }
    )
    df3.to_parquet(block3 / "3p_rows.parquet")
    np.save(block3 / "keepers_3.npy", np.array(["A", "B"]))

    (res_root / "manifest.yaml").write_text(yaml.safe_dump({"seed": 0}))

    # copy source tree so imports work under the tmp cwd
    root = Path(__file__).resolve().parents[3]
    shutil.copytree(root / "src", tmp_path / "src", dirs_exist_ok=True)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_trueskill.run_trueskill(root=data_root)
    finally:
        os.chdir(cwd)

    with open(tmp_path / "data" / "ratings_2.pkl", "rb") as fh:
        r2 = pickle.load(fh)
    with open(tmp_path / "data" / "ratings_3.pkl", "rb") as fh:
        r3 = pickle.load(fh)
    with open(tmp_path / "data" / "ratings_pooled.pkl", "rb") as fh:
        pooled = pickle.load(fh)

    env = run_trueskill.trueskill.TrueSkill()
    g2 = [["A", "B"]] * 3
    g3 = [["B", "A", "C"]] * 6
    expected2 = run_trueskill._update_ratings(g2, ["A", "B"], env)
    expected3 = run_trueskill._update_ratings(g3, ["A", "B"], env)

    w2, w3 = len(g2), len(g3)
    expected_pooled = {}
    for k in ("A", "B", "C"):
        e2 = expected2.get(k, run_trueskill.RatingStats(run_trueskill.DEFAULT_RATING.mu, run_trueskill.DEFAULT_RATING.sigma))
        e3 = expected3.get(k, run_trueskill.RatingStats(run_trueskill.DEFAULT_RATING.mu, run_trueskill.DEFAULT_RATING.sigma))
        expected_pooled[k] = run_trueskill.RatingStats(
            (e2.mu * w2 + e3.mu * w3) / (w2 + w3),
            (e2.sigma * w2 + e3.sigma * w3) / (w2 + w3),
        )

    assert r2 == expected2
    assert r3 == expected3
    assert pooled == expected_pooled
