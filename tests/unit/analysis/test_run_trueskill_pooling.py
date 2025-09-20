import math
import os
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

    r2 = run_trueskill._load_ratings_parquet(tmp_path / "data" / "ratings_2.parquet")
    r3 = run_trueskill._load_ratings_parquet(tmp_path / "data" / "ratings_3.parquet")
    pooled = run_trueskill._load_ratings_parquet(tmp_path / "data" / "ratings_pooled.parquet")

    env = run_trueskill.trueskill.TrueSkill()
    g2 = [["A", "B"]] * 3
    g3 = [["B", "A", "C"]] * 6
    expected2 = run_trueskill._update_ratings(g2, ["A", "B"], env)
    expected3 = run_trueskill._update_ratings(g3, ["A", "B"], env)

    w2, w3 = len(g2), len(g3)

    pooled_expect = {}
    for key in pooled:
        s_mu, s_tau = 0.0, 0.0
        stats2 = expected2.get(key)
        if stats2 is not None:
            tau2 = 1.0 / (stats2.sigma**2) if stats2.sigma > 0 else 0.0
            s_mu += tau2 * stats2.mu * w2
            s_tau += tau2 * w2
        stats3 = expected3.get(key)
        if stats3 is not None:
            tau3 = 1.0 / (stats3.sigma**2) if stats3.sigma > 0 else 0.0
            s_mu += tau3 * stats3.mu * w3
            s_tau += tau3 * w3
        if s_tau > 0:
            pooled_expect[key] = run_trueskill.RatingStats(s_mu / s_tau, math.sqrt(1.0 / s_tau))

    assert r2 == expected2
    assert r3 == {k: expected3[k] for k in ("A", "B")}
    for key, expected_stats in pooled_expect.items():
        actual = pooled[key]
        assert math.isclose(actual.mu, expected_stats.mu, rel_tol=1e-6)
        assert math.isclose(actual.sigma, expected_stats.sigma, rel_tol=1e-6)
