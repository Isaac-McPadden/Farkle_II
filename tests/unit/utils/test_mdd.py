import warnings
from typing import cast

import pandas as pd
import pytest

from farkle.utils.mdd import (
    _ensure_winrate,
    compute_mdd_for_tiers,
    estimate_tau2_seed,
    estimate_tau2_sxk,
    prepare_cell_means,
    tiering_ingredients_from_df,
)


def test_ensure_winrate_prefers_existing_column():
    df = pd.DataFrame({"wins": [1], "games": [2], "winrate": [0.25]})
    result = _ensure_winrate(df, wins_col="wins", games_col="games", winrate_col="winrate")
    assert result.tolist() == [0.25]


def test_ensure_winrate_requires_inputs():
    df = pd.DataFrame({"wins": [1], "games": [2]})
    with pytest.raises(ValueError):
        _ensure_winrate(df, wins_col=None, games_col=None, winrate_col=None)


def test_prepare_cell_means_and_tau2_seed():
    df = pd.DataFrame(
        {
            "strategy": ["A", "A", "A", "B"],
            "player_count": [2, 2, 3, 2],
            "seed": [1, 1, 2, 1],
            "wins": [1, 2, 0, 1],
            "games": [2, 2, 2, 2],
        }
    )

    cell = prepare_cell_means(df)
    assert set(cell.columns) == {"strategy", "k", "seed", "winrate", "games"}
    games = cast(float, cell.loc[(cell.strategy == "A") & (cell.k == 2), "games"].iloc[0])
    assert games == 4.0

    comps = estimate_tau2_seed(cell, robust=False)
    assert comps.tau2_seed >= 0
    assert comps.R == 2
    assert comps.K == 2


def test_estimate_tau2_sxk_with_weights():
    cell = pd.DataFrame(
        {
            "strategy": ["A", "A", "A", "A"],
            "k": [2, 3, 2, 3],
            "seed": [1, 1, 2, 2],
            "winrate": [0.5, 0.6, 0.4, 0.55],
            "games": [10, 10, 10, 10],
        }
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        tau2 = estimate_tau2_sxk(
            cell,
            tau2_seed=0.01,
            weights_by_k={2: 1.0, 3: 2.0},
            robust=False,
        )
    assert not any(issubclass(w.category, FutureWarning) for w in record)
    assert tau2 >= 0


def test_compute_mdd_for_tiers_validates_inputs():
    with pytest.raises(ValueError):
        compute_mdd_for_tiers(tau2_seed=0.1, tau2_sxk=0.0, binom_by_k=pd.Series(dtype=float), R=2)

    binom_by_k = pd.Series({2: 0.1, 3: 0.2})
    with pytest.raises(ValueError):
        compute_mdd_for_tiers(
            tau2_seed=0.1,
            tau2_sxk=0.0,
            binom_by_k=binom_by_k,
            weights_by_k={4: 1.0},
            R=2,
        )


def test_tiering_ingredients_from_df_round_trip():
    df = pd.DataFrame(
        {
            "strategy": ["S1", "S1", "S2", "S2"],
            "player_count": [2, 2, 2, 2],
            "seed": [1, 2, 1, 2],
            "wins": [3, 4, 5, 6],
            "games": [5, 5, 5, 5],
        }
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = tiering_ingredients_from_df(df, weights_by_k={2: 1.0}, z_star=1.0)
    assert not any(issubclass(w.category, FutureWarning) for w in record)
    assert set(result.keys()) == {"cell", "components", "tau2_sxk", "mdd"}
    mdd = cast(float, result["mdd"])
    assert mdd > 0


def test_compute_mdd_for_tiers_reindexes_string_k_index() -> None:
    binom_by_k = pd.Series([0.1, 0.2], index=["2", "3"])

    mdd = compute_mdd_for_tiers(
        tau2_seed=0.1,
        tau2_sxk=0.05,
        binom_by_k=binom_by_k,
        weights_by_k={2: 1.0, 3: 1.0},
        R=4,
        z_star=2.0,
    )

    assert mdd > 0



def test_ensure_winrate_without_jeffreys_fills_nan_to_zero() -> None:
    df = pd.DataFrame({"wins": [0], "games": [0]})

    result = _ensure_winrate(
        df,
        wins_col="wins",
        games_col="games",
        winrate_col=None,
        use_jeffreys=False,
    )

    assert result.tolist() == [0.0]


def test_prepare_cell_means_defaults_games_when_column_missing() -> None:
    df = pd.DataFrame(
        {
            "strategy": ["A", "A"],
            "player_count": [2, 2],
            "seed": [1, 2],
            "winrate": [0.2, 0.4],
        }
    )

    cell = prepare_cell_means(df, wins_col=None, games_col=None, winrate_col="winrate")

    assert set(cell["games"].tolist()) == {1.0}


def test_estimate_tau2_sxk_uses_equal_weights_by_default() -> None:
    cell = pd.DataFrame(
        {
            "strategy": ["A", "A", "A", "A"],
            "k": [2, 3, 2, 3],
            "seed": [1, 1, 2, 2],
            "winrate": [0.5, 0.6, 0.4, 0.55],
            "games": [10, 10, 10, 10],
        }
    )

    tau2 = estimate_tau2_sxk(cell, tau2_seed=0.01, weights_by_k=None, robust=True)

    assert tau2 >= 0
