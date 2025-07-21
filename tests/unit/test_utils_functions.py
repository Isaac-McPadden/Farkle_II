import numpy as np
import pandas as pd
import pytest

from farkle.utils import bh_correct, bonferroni_pairs, build_tiers


def test_build_tiers_empty():
    assert build_tiers(means={}, stdevs={}) == {}


def test_build_tiers_overlapping():
    mu = {"A": 100.0, "B": 99.0}
    sigma = {"A": 0.5, "B": 0.5}
    tiers = build_tiers(means=mu, stdevs=sigma, z=2)
    assert tiers == {"A": 1, "B": 1}


def test_build_tiers_multiple_tiers():
    mu = {"A": 100.0, "B": 95.0, "C": 93.0}
    sigma = {"A": 1.0, "B": 1.0, "C": 1.0}
    tiers = build_tiers(means=mu, stdevs=sigma, z=2)
    assert tiers["A"] == 1
    assert tiers["B"] == 2
    assert tiers["C"] == 2


def test_build_tiers_key_mismatch():
    mu = {"A": 100.0}
    sigma = {"A": 1.0, "B": 1.0}
    with pytest.raises(ValueError):
        build_tiers(mu, sigma)


def test_bh_correct_typical():
    pvals = np.array([0.01, 0.02, 0.03, 0.2])
    mask = bh_correct(pvals, alpha=0.05)
    assert mask.tolist() == [True, True, True, False]


def test_bh_correct_none_pass():
    pvals = np.array([0.2, 0.8, 0.4])
    mask = bh_correct(pvals, alpha=0.05)
    assert mask.tolist() == [False, False, False]


def test_bh_correct_all_high_pvals():
    pvals = np.array([0.9, 0.85, 0.95, 0.99])
    mask = bh_correct(pvals, alpha=0.05)
    assert mask.tolist() == [False, False, False, False]


def test_bonferroni_pairs_basic_determinism():
    strats = ["S1", "S2", "S3"]
    df1 = bonferroni_pairs(strats, games_needed=2, seed=42)
    df2 = bonferroni_pairs(strats, games_needed=2, seed=42)
    assert df1.equals(df2)
    assert set(df1.columns) == {"a", "b", "seed"}
    assert len(df1) == 6
    counts = df1.groupby(["a", "b"]).size()
    assert all(v == 2 for v in counts)
    assert pd.api.types.is_integer_dtype(df1["seed"])


def test_bonferroni_pairs_not_enough_strats():
    assert bonferroni_pairs([], games_needed=2, seed=0).empty
    assert bonferroni_pairs(["A"], games_needed=2, seed=0).empty


def test_bonferroni_pairs_zero_games():
    df = bonferroni_pairs(["A", "B"], games_needed=0, seed=1)
    assert df.empty


def test_bonferroni_pairs_negative_games():
    with pytest.raises(ValueError):
        bonferroni_pairs(["A", "B"], games_needed=-1, seed=1)
