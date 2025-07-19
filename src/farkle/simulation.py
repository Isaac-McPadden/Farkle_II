from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from farkle.engine import FarkleGame, FarklePlayer
from farkle.strategies import ThresholdStrategy

"""simulation.py
================
High-level utilities for *batch* and *grid* simulations.

This is the entry point most users will reach for:

* ``generate_strategy_grid`` - produce the canonical 800-strategy (or
  custom) grid and its accompanying ``DataFrame``.
* ``simulate_many_games`` - run *N* games, optionally in parallel, and
  return tidy metrics.
* ``aggregate_metrics`` - summarize a DataFrame of game results.
"""

__all__: list[str] = [
    "generate_strategy_grid",
    "experiment_size",
    "simulate_one_game",
    "simulate_many_games",
    "aggregate_metrics",
]


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def generate_strategy_grid(
    *,
    score_thresholds: Sequence[int] | None = None,
    dice_thresholds: Sequence[int] | None = None,
    smart_five_opts: Sequence[bool] | None = None,
    smart_one_opts: Sequence[bool] | None = None,
    consider_score_opts: Sequence[bool] = (True, False),
    consider_dice_opts: Sequence[bool] = (True, False),
    auto_hot_opts: Sequence[bool] = (False, True),
    run_up_score_opts: Sequence[bool] = (True, False),
) -> Tuple[List[ThresholdStrategy], pd.DataFrame]:
    """Create the Cartesian product of all parameter options.

    Returns a tuple *(strategies, meta_df)* where the first element is a
    list of fully-constructed ``ThresholdStrategy`` instances (safe to
    pass to the engine) and the second element is a *metadata* dataframe
    recording each parameter combo.
    Default grid should be len(grid) = 8160
    """

    if score_thresholds is None:
        score_thresholds = list(range(200, 1050, 50))
    if dice_thresholds is None:
        dice_thresholds = list(range(0, 5)) 
    if smart_five_opts is None:
        smart_five_opts = [True, False] 
    if smart_one_opts is None:
        smart_one_opts = [True, False] 
    combos: List[Tuple[int, int, bool, bool, bool, bool, bool, bool, bool, bool]] = []
    # We'll build combos of (st, dt, sm, cs, cd, require_both)
    for rs in run_up_score_opts:  # 2 options
        for hd in auto_hot_opts:  # Hot Dice 2 options
            for st in score_thresholds:  # 17 options
                for dt in dice_thresholds:  # 5 options
                    for sf in smart_five_opts:  # 2 options
                        for so in smart_one_opts:  # 1.5 options
                            if not sf and so:  # Can't be smart one without being smart five
                                continue  # You technically could but it makes no strategic sense irl
                            for cs in consider_score_opts:  # 2 options
                                for cd in consider_dice_opts:  # 2.5 options (1/4 of cs&cd gets an extra option on rb 4+1 = 5, 5/2 = 2.5)
                                    # Determine the two valid values of require_both:
                                    rb_values = [True, False] if cs and cd else [False]
                                    for rb in rb_values:
                                        # Now we need to pick prefer_score (ps) according to:
                                        #   cs  cd   ⟶  valid ps
                                        #   T   F       True
                                        #   F   T       False
                                        #   T   T       both {True, False}
                                        #   F   F       both {True, False}
                                        if cs and not cd:
                                            ps_values = [True]
                                        elif cd and not cs:
                                            ps_values = [False]
                                        else:
                                            # (cs,cd) is either (T,T) or (F,F)
                                            ps_values = [True, False]
                                        for ps in ps_values:  # 1.6 options (Increases cs,cd,rb truth table from 5 to 8 with cs,cd,rb,ps 8/5 = 1.6)
                                            combos.append((st, dt, sf, so, cs, cd, rb, hd, rs, ps))

    # Build actual strategy objects and a DataFrame
    strategies = [
        ThresholdStrategy(
            score_threshold = st,
            dice_threshold = dt,
            smart_five = sf,
            smart_one = so,
            consider_score = cs,
            consider_dice = cd,
            require_both = rb,
            auto_hot_dice = hd,
            run_up_score = rs,
            prefer_score = ps,
        )
        for st, dt, sf, so, cs, cd, rb, hd, rs, ps in combos
    ]

    meta = pd.DataFrame(
        combos,
        columns=[
            "score_threshold", 
            "dice_threshold", 
            "smart_five", 
            "smart_one", 
            "consider_score", 
            "consider_dice", 
            "require_both", 
            "auto_hot_dice", 
            "run_up_score", 
            "prefer_score",
        ],
    )
    meta["strategy_idx"] = meta.index
    return strategies, meta


def experiment_size(
    *,
    score_thresholds: Sequence[int] | None = None,
    dice_thresholds: Sequence[int] | None = None,
    smart_five_and_one_options: Sequence[Sequence[bool]] | None = None,
    consider_score_opts: Sequence[bool] = (True, False),
    consider_dice_opts: Sequence[bool] = (True, False),
    auto_hot_dice_opts: Sequence[bool] = (True, False),
    run_up_score_opts: Sequence[bool] = (True, False),
) -> int:
    """Compute *a priori* size of a strategy grid."""
    
    score_thresholds = score_thresholds or list(range(200, 1050, 50))
    dice_thresholds = dice_thresholds or list(range(0, 5))
    smart_five_and_one_options = smart_five_and_one_options or [[True, True], [True, False], [False, False]]
    base = (
        len(score_thresholds)
        * len(dice_thresholds)
        * len(smart_five_and_one_options)
        * len(auto_hot_dice_opts)
        * len(run_up_score_opts)
    )

    # ----- how many CS/CD pairs? ----------------------------------------
    n_cs, n_cd = len(consider_score_opts), len(consider_dice_opts)
    pair_count = n_cs * n_cd

    # Extra row for (True, True) when it's present
    if True in consider_score_opts and True in consider_dice_opts:
        pair_count += 1  # second variant with require_both=True
    
    pair_count += 3 # prefer score adds an extra option for each cs,cd TT (2 of them) or FF combination

    return base * pair_count

# ---------------------------------------------------------------------------
# Batch simulation helpers
# ---------------------------------------------------------------------------
def _make_players(strategies, seed):
    master = np.random.default_rng(seed)
    return [
      FarklePlayer(
        name=f"P{i+1}",
        strategy=s,
        rng=np.random.default_rng(master.integers(0, 2**32 - 1)),
      )
      for i, s in enumerate(strategies)
    ]


def _play_game(seed: int, strategies: Sequence[ThresholdStrategy], target_score: int=10000) -> Dict[str, Any]:
    # give every player an *independent* PRNG, but reproducible
    players = _make_players(strategies, seed)
    gm = FarkleGame(players, target_score=target_score).play()
    # 1) Determine the winning player's name from the PlayerStats block
    winner = next(name for name, ps in gm.players.items() if ps.rank == 1)
    flat: Dict[str, Any] = {
        "winner": winner,
        "winning_score": gm.players[winner].score,
        "n_rounds": gm.game.n_rounds,
    }
    # 3) Per-player metrics
    for pname, stats in gm.players.items():
        # asdict works on slots dataclasses
        for k, v in asdict(stats).items():
            flat[f"{pname}_{k}"] = v
            
    return flat


def simulate_many_games(
    *,
    n_games: int,
    strategies: Sequence[ThresholdStrategy],
    target_score: int = 10_000,
    seed: int | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run *n_games* in parallel (if *n_jobs>1*) and return tidy metrics."""
    
    master_rng = np.random.default_rng(seed)
    seeds = master_rng.integers(0, 2**32 - 1, size=n_games).tolist()
    args = [(s, strategies, target_score) for s in seeds]
    if n_jobs == 1:
        rows = [_play_game(*a) for a in args]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            rows = pool.starmap(_play_game, args)
    return pd.DataFrame(rows)


def simulate_one_game(
    *,
    strategies: Sequence[ThresholdStrategy],
    target_score: int = 10_000,
    seed: int | None = None,
):
    """Convenience wrapper around the *single* game engine."""
    
    players = _make_players(strategies, seed)
    return FarkleGame(players, target_score=target_score).play()


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def aggregate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a *dict* with high-level summary statistics."""
    
    return {
        "games": len(df),
        "avg_rounds": df["n_rounds"].mean(),
        "winner_freq": df["winner"].value_counts().to_dict(),
    }
