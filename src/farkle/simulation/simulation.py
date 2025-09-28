# src/farkle/simulation.py
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

from __future__ import annotations

import itertools
import multiprocessing as mp
from dataclasses import asdict
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

from farkle.game.engine import FarkleGame, FarklePlayer
from farkle.simulation.strategies import FavorDiceOrScore, ThresholdStrategy
from farkle.utils.random import MAX_UINT32, make_rng, spawn_seeds

__all__: list[str] = [
    "generate_strategy_grid",
    "experiment_size",
    "simulate_one_game",
    "simulate_many_games",
    "simulate_many_games_from_seeds",
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
    auto_hot_dice_opts: Sequence[bool] = (False, True),
    run_up_score_opts: Sequence[bool] = (True, False),
) -> Tuple[List[ThresholdStrategy], pd.DataFrame]:
    """Create the Cartesian product of all parameter choices.

    Parameters
    ----------
    score_thresholds, dice_thresholds, smart_five_opts, smart_one_opts,
    consider_score_opts, consider_dice_opts, auto_hot_dice_opts, run_up_score_opts
        Sequences of options for the corresponding ``ThresholdStrategy``
        fields. ``None`` selects sensible defaults for each parameter.

    Returns
    -------
    Tuple[List[ThresholdStrategy], pandas.DataFrame]
        The first element contains fully constructed strategies in the
        same order as the metadata ``DataFrame`` returned as the second
        element.  The default grid consists of 8 160 strategies.
    """

    if score_thresholds is None:
        score_thresholds = list(range(200, 1050, 50))
    if dice_thresholds is None:
        dice_thresholds = list(range(0, 5))
    if smart_five_opts is None:
        smart_five_opts = [True, False]
    if smart_one_opts is None:
        smart_one_opts = [True, False]
    combos: List[Tuple[int, int, bool, bool, bool, bool, bool, bool, bool, FavorDiceOrScore]] = []

    # Iterate over the basic option grid using itertools.product and filter
    for st, dt, sf, so, cs, cd, hd, rs in itertools.product(
        score_thresholds,
        dice_thresholds,
        smart_five_opts,
        smart_one_opts,
        consider_score_opts,
        consider_dice_opts,
        auto_hot_dice_opts,
        run_up_score_opts,
    ):
        # Can't be smart one without smart five
        if not sf and so:
            continue

        rb_values = [True, False] if cs and cd else [False]

        if cs and not cd:
            ps_values = [FavorDiceOrScore.SCORE]
        elif cd and not cs:
            ps_values = [FavorDiceOrScore.DICE]
        else:
            ps_values = [FavorDiceOrScore.SCORE, FavorDiceOrScore.DICE]

        for rb in rb_values:
            for ps in ps_values:
                combos.append((st, dt, sf, so, cs, cd, rb, hd, rs, ps))

    # Build actual strategy objects and a DataFrame
    strategies = [
        ThresholdStrategy(
            score_threshold=st,
            dice_threshold=dt,
            smart_five=sf,
            smart_one=so,
            consider_score=cs,
            consider_dice=cd,
            require_both=rb,
            auto_hot_dice=hd,
            run_up_score=rs,
            favor_dice_or_score=ps,
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
            "favor_dice_or_score",
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
    """Compute *a priori* size of a strategy grid.

    Parameters
    ----------
    score_thresholds, dice_thresholds, smart_five_and_one_options,
    consider_score_opts, consider_dice_opts, auto_hot_dice_opts,
    run_up_score_opts
        Option sequences mirroring those accepted by
        :func:`generate_strategy_grid`. ``None`` falls back to that
        function's defaults.

    Returns
    -------
    int
        Number of unique strategy combinations that would be generated.
    """
    score_thresholds = score_thresholds or list(range(200, 1050, 50))
    dice_thresholds = dice_thresholds or list(range(0, 5))
    smart_five_and_one_options = smart_five_and_one_options or [
        [True, True],
        [True, False],
        [False, False],
    ]
    base = (
        len(score_thresholds)
        * len(dice_thresholds)
        * len(smart_five_and_one_options)
        * len(auto_hot_dice_opts)
        * len(run_up_score_opts)
    )

    # ----- how many CS/CD pairs? ----------------------------------------
    # generate_strategy_grid loops over each (cs, cd) pair and then
    # selects ``require_both`` and ``favor_dice_or_score`` options based on the
    # truth table described in the function's docstring.  ``pair_count``
    # should therefore mirror that logic directly so that the predicted
    # size matches ``generate_strategy_grid`` for any subset of options.

    def _pair_variations(cs: bool, cd: bool) -> int:
        rb_choices = 2 if cs and cd else 1
        ps_choices = 1 if cs ^ cd else 2
        return rb_choices * ps_choices

    pair_count = sum(
        _pair_variations(cs, cd) for cs in consider_score_opts for cd in consider_dice_opts
    )

    return base * pair_count


# ---------------------------------------------------------------------------
# Batch simulation helpers
# ---------------------------------------------------------------------------
def _make_players(
    strategies: Sequence[ThresholdStrategy],
    seed: int | None,
) -> List[FarklePlayer]:
    """Instantiate ``FarklePlayer`` objects for a table.

    Parameters
    ----------
    strategies
        Strategies applied to each player in order.
    seed
        Seed used to create per-player random number generators.  ``None``
        yields non-deterministic behavior.

    Returns
    -------
    List[FarklePlayer]
        Players ready to be passed to ``FarkleGame``.
    """

    master = make_rng(seed)
    # derive stable per-player seeds from a single source of truth
    player_seeds = spawn_seeds(
        len(strategies),
        seed=int(master.integers(0, MAX_UINT32)) if seed is not None else None,
    )
    return [
        FarklePlayer(
            name=f"P{i+1}",
            strategy=s,
            rng=make_rng(int(ps)),
        )
        for i, (s, ps) in enumerate(zip(strategies, player_seeds, strict=True))
    ]


def _play_game(
    seed: int,
    strategies: Sequence[ThresholdStrategy],
    target_score: int = 10_000,
) -> Mapping[str, Any]:
    """Play a single game and return flattened metrics.

    Parameters
    ----------
    seed
        Base seed for constructing player RNGs.
    strategies
        ``ThresholdStrategy`` objects in seating order.
    target_score
        Score required to trigger the final round.

    Returns
    -------
    Mapping[str, Any]
        Mapping of metric names to values including the winner and
        per-player statistics.
    """
    # give every player an *independent* PRNG, but reproducible
    players = _make_players(strategies, seed)
    gm = FarkleGame(players, target_score=target_score).play()
    # Check for only one winner
    winners = [name for name, ps in gm.players.items() if ps.rank == 1]
    if len(winners) != 1:
        raise ValueError(
            "Expected exactly one player with rank == 1, " f"got {len(winners)}: {winners}"
        )
    # Determine the winner from the PlayerStats block
    winner = next(name for name, ps in gm.players.items() if ps.rank == 1)
    flat: dict[str, Any] = {
        "winner": winner,
        "winning_score": gm.players[winner].score,
        "n_rounds": gm.game.n_rounds,
    }
    # Per-player metrics
    for pname, stats in gm.players.items():
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
    """Run many games and return a tidy ``DataFrame`` of results.

    Parameters
    ----------
    n_games
        Number of games to simulate.
    strategies
        Strategies assigned to the players.
    target_score
        Score required to trigger the final round.
    seed
        Optional seed for determinism across runs.
    n_jobs
        Number of worker processes to spawn; ``1`` runs serially.

    Returns
    -------
    pandas.DataFrame
        One row per game as produced by :func:`_play_game`.
    """
    seeds = spawn_seeds(n_games, seed=seed)
    args = [(int(s), strategies, target_score) for s in list(seeds)]
    if n_jobs == 1:
        rows = [_play_game(*a) for a in args]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            rows = pool.starmap(_play_game, args)
    return pd.DataFrame(rows)


def simulate_many_games_from_seeds(
    *,
    seeds: Iterable[int],
    strategies: Sequence[ThresholdStrategy],
    target_score: int = 10_000,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run games for a predetermined list of seeds."""
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
    """Play a single game using the provided strategies.

    Parameters
    ----------
    strategies
        Strategy objects applied to the players.
    target_score
        Score needed to trigger the final round.
    seed
        Optional seed controlling all random number generators.

    Returns
    -------
    GameMetrics
        Dataclass returned by :class:`~farkle.engine.FarkleGame.play`.
    """
    players = _make_players(strategies, seed)
    return FarkleGame(players, target_score=target_score).play()


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------


def aggregate_metrics(df: pd.DataFrame) -> Mapping[str, Any]:
    """Summarize a DataFrame of game results.

    Parameters
    ----------
    df
        DataFrame produced by :func:`simulate_many_games`.

    Returns
    -------
    Mapping[str, Any]
        Mapping with the total number of games, the mean round count and
        a winner frequency dictionary.
    """
    return {
        "games": len(df),
        "avg_rounds": df["n_rounds"].mean(),
        "winner_freq": df["winner"].value_counts().to_dict(),
    }
