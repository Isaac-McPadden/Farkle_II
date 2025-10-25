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

import multiprocessing as mp
from dataclasses import asdict
from typing import Any, Iterable, List, Mapping, Sequence, Tuple, TypeVar

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

_T = TypeVar("_T")


def _coerce_options(options: Sequence[_T] | None, fallback: Iterable[_T]) -> list[_T]:
    """Return a mutable list using ``fallback`` when ``options`` is None."""
    return list(fallback) if options is None else list(options)


def _favor_options(sf: bool, cs: bool, cd: bool) -> tuple[FavorDiceOrScore, ...]:
    """Return valid FavorDiceOrScore choices for the given flag combination."""
    if cs and cd:
        return (FavorDiceOrScore.SCORE, FavorDiceOrScore.DICE) if sf else (FavorDiceOrScore.SCORE,)
    if cs:
        return (FavorDiceOrScore.SCORE,)
    if cd:
        return (FavorDiceOrScore.DICE,)
    return (FavorDiceOrScore.SCORE,)


def _iter_strategy_combos(
    *,
    score_thresholds: Sequence[int],
    dice_thresholds: Sequence[int],
    smart_five_opts: Sequence[bool],
    smart_one_opts: Sequence[bool],
    consider_score_opts: Sequence[bool],
    consider_dice_opts: Sequence[bool],
    auto_hot_dice_opts: Sequence[bool],
    run_up_score_opts: Sequence[bool],
    inactive_score_threshold: int,
    inactive_dice_threshold: int,
    allowed_smart_pairs: set[tuple[bool, bool]] | None = None,
) -> Iterable[Tuple[int, int, bool, bool, bool, bool, bool, bool, bool, FavorDiceOrScore]]:
    """
    Yield canonical strategy parameter combinations honoring the flag rules.
    """
    for sf in smart_five_opts:
        smart_one_candidates = [
            so
            for so in smart_one_opts
            if (sf or not so) and (allowed_smart_pairs is None or (sf, so) in allowed_smart_pairs)
        ]
        if not smart_one_candidates:
            continue

        for so in smart_one_candidates:
            for cs in consider_score_opts:
                score_values = score_thresholds if cs else [inactive_score_threshold]

                for cd in consider_dice_opts:
                    dice_values = dice_thresholds if cd else [inactive_dice_threshold]
                    rb_values = [True, False] if (cs and cd) else [False]
                    favor_choices = _favor_options(sf, cs, cd)

                    for st in score_values:
                        for dt in dice_values:
                            for hd in auto_hot_dice_opts:
                                for rs in run_up_score_opts:
                                    for rb in rb_values:
                                        for ps in favor_choices:
                                            yield (
                                                st,
                                                dt,
                                                sf,
                                                so,
                                                cs,
                                                cd,
                                                rb,
                                                hd,
                                                rs,
                                                ps,
                                            )


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
        element.  The default grid consists of 3,676 strategies.
        When a threshold flag is disabled the corresponding threshold column
        uses a sentinel value (one less than the minimum configured threshold)
        to indicate that it is inactive.
    """

    score_thresholds = _coerce_options(score_thresholds, range(200, 1400, 50))
    dice_thresholds = _coerce_options(dice_thresholds, range(0, 5))
    smart_five_opts = _coerce_options(smart_five_opts, (True, False))
    smart_one_opts = _coerce_options(smart_one_opts, (True, False))
    consider_score_opts = _coerce_options(consider_score_opts, (True, False))
    consider_dice_opts = _coerce_options(consider_dice_opts, (True, False))
    auto_hot_dice_opts = _coerce_options(auto_hot_dice_opts, (False, True))
    run_up_score_opts = _coerce_options(run_up_score_opts, (True, False))

    if not score_thresholds:
        raise ValueError("score_thresholds must contain at least one value")
    if not dice_thresholds:
        raise ValueError("dice_thresholds must contain at least one value")

    inactive_score_threshold = min(score_thresholds) - 1
    inactive_dice_threshold = min(dice_thresholds) - 1

    combos = list(
        _iter_strategy_combos(
            score_thresholds=score_thresholds,
            dice_thresholds=dice_thresholds,
            smart_five_opts=smart_five_opts,
            smart_one_opts=smart_one_opts,
            consider_score_opts=consider_score_opts,
            consider_dice_opts=consider_dice_opts,
            auto_hot_dice_opts=auto_hot_dice_opts,
            run_up_score_opts=run_up_score_opts,
            inactive_score_threshold=inactive_score_threshold,
            inactive_dice_threshold=inactive_dice_threshold,
        )
    )

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
    auto_hot_dice_opts: Sequence[bool] = (False, True),  # order doesn't matter; len does
    run_up_score_opts: Sequence[bool] = (True, False),
) -> int:
    """Compute *a priori* size of a strategy grid that matches generate_strategy_grid."""
    score_thresholds_list = _coerce_options(score_thresholds, range(200, 1400, 50))
    dice_thresholds_list = _coerce_options(dice_thresholds, range(0, 5))
    consider_score_opts_list = _coerce_options(consider_score_opts, (True, False))
    consider_dice_opts_list = _coerce_options(consider_dice_opts, (True, False))
    auto_hot_dice_opts_list = _coerce_options(auto_hot_dice_opts, (False, True))
    run_up_score_opts_list = _coerce_options(run_up_score_opts, (True, False))

    if not score_thresholds_list:
        raise ValueError("score_thresholds must contain at least one value")
    if not dice_thresholds_list:
        raise ValueError("dice_thresholds must contain at least one value")

    inactive_score_threshold = min(score_thresholds_list) - 1
    inactive_dice_threshold = min(dice_thresholds_list) - 1

    if smart_five_and_one_options is None:
        smart_five_opts_list = _coerce_options(None, (True, False))
        smart_one_opts_list = _coerce_options(None, (True, False))
        allowed_pairs: set[tuple[bool, bool]] | None = None
    else:
        normalized_pairs: list[tuple[bool, bool]] = []
        for opts in smart_five_and_one_options:
            if len(opts) != 2:
                raise ValueError("smart_five_and_one_options entries must contain exactly two booleans")
            normalized_pairs.append((bool(opts[0]), bool(opts[1])))
        allowed_pairs = set(normalized_pairs)
        smart_five_opts_list = list(dict.fromkeys(sf for sf, _ in normalized_pairs))
        smart_one_opts_list = list(dict.fromkeys(so for _, so in normalized_pairs))
        if not smart_five_opts_list or not smart_one_opts_list:
            return 0

    combo_iter = _iter_strategy_combos(
        score_thresholds=score_thresholds_list,
        dice_thresholds=dice_thresholds_list,
        smart_five_opts=smart_five_opts_list,
        smart_one_opts=smart_one_opts_list,
        consider_score_opts=consider_score_opts_list,
        consider_dice_opts=consider_dice_opts_list,
        auto_hot_dice_opts=auto_hot_dice_opts_list,
        run_up_score_opts=run_up_score_opts_list,
        inactive_score_threshold=inactive_score_threshold,
        inactive_dice_threshold=inactive_dice_threshold,
        allowed_smart_pairs=allowed_pairs,
    )
    return sum(1 for _ in combo_iter)


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
