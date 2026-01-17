# src/farkle/simulation/simulation.py
"""Utilities for constructing strategy grids and running Farkle simulations.

Key entry points include:

* ``generate_strategy_grid`` for building the canonical 800-strategy (or custom) grid
  and returning both strategies and a DataFrame view.
* ``simulate_many_games`` for executing batches of games, with optional parallelism.
* ``aggregate_metrics`` for summarizing simulation results into lightweight metrics.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

from farkle.game.engine import FarkleGame, FarklePlayer
from farkle.simulation.strategies import (
    STOP_AT_THRESHOLDS,
    StrategyGridOptions,
    ThresholdStrategy,
    build_stop_at_strategy,
    build_strategy_encoder,
    encode_strategy,
    iter_strategy_combos,
    strategy_tuple,
)
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
    include_stop_at: bool = False,
    include_stop_at_heuristic: bool = False,
) -> Tuple[List[ThresholdStrategy], pd.DataFrame]:
    """Create the Cartesian product of all parameter choices.

    Parameters
    ----------
    score_thresholds, dice_thresholds, smart_five_opts, smart_one_opts,
    consider_score_opts, consider_dice_opts, auto_hot_dice_opts, run_up_score_opts
        Sequences of options for the corresponding ``ThresholdStrategy``
        fields. ``None`` selects sensible defaults for each parameter.
    include_stop_at, include_stop_at_heuristic
        When enabled, append the named stop-at strategies (with or without
        heuristics) to the generated grid.

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

    options = StrategyGridOptions.from_inputs(
        score_thresholds=score_thresholds,
        dice_thresholds=dice_thresholds,
        smart_five_opts=smart_five_opts,
        smart_one_opts=smart_one_opts,
        consider_score_opts=consider_score_opts,
        consider_dice_opts=consider_dice_opts,
        auto_hot_dice_opts=auto_hot_dice_opts,
        run_up_score_opts=run_up_score_opts,
        include_stop_at=include_stop_at,
        include_stop_at_heuristic=include_stop_at_heuristic,
    )

    if not options.score_thresholds:
        raise ValueError("score_thresholds must contain at least one value")
    if not options.dice_thresholds:
        raise ValueError("dice_thresholds must contain at least one value")

    encoder = build_strategy_encoder(
        score_thresholds=options.score_thresholds,
        dice_thresholds=options.dice_thresholds,
        smart_five_opts=options.smart_five_opts,
        smart_one_opts=options.smart_one_opts,
        consider_score_opts=options.consider_score_opts,
        consider_dice_opts=options.consider_dice_opts,
        auto_hot_dice_opts=options.auto_hot_dice_opts,
        run_up_score_opts=options.run_up_score_opts,
        include_stop_at=options.include_stop_at,
        include_stop_at_heuristic=options.include_stop_at_heuristic,
    )

    combos = list(
        iter_strategy_combos(
            score_thresholds=options.score_thresholds,
            dice_thresholds=options.dice_thresholds,
            smart_five_opts=options.smart_five_opts,
            smart_one_opts=options.smart_one_opts,
            consider_score_opts=options.consider_score_opts,
            consider_dice_opts=options.consider_dice_opts,
            auto_hot_dice_opts=options.auto_hot_dice_opts,
            run_up_score_opts=options.run_up_score_opts,
            inactive_score_threshold=options.inactive_score_threshold,
            inactive_dice_threshold=options.inactive_dice_threshold,
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
            strategy_id=encoder.encode_tuple(
                (st, dt, sf, so, cs, cd, rb, hd, rs, ps)
            ),
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
    meta["strategy_id"] = [encoder.encode_tuple(combo) for combo in combos]

    stop_at_rows: list[list[object]] = []
    stop_at_ids: list[int] = []
    if include_stop_at:
        stop_at_strats = [
            build_stop_at_strategy(
                threshold, inactive_dice_threshold=options.inactive_dice_threshold
            )
            for threshold in STOP_AT_THRESHOLDS
        ]
        for strat in stop_at_strats:
            strat.strategy_id = encode_strategy(strat, encoder)
            strategies.append(strat)
            stop_at_rows.append(list(strategy_tuple(strat)))
            stop_at_ids.append(strat.strategy_id)

    if include_stop_at_heuristic:
        stop_at_strats = [
            build_stop_at_strategy(
                threshold,
                heuristic=True,
                inactive_dice_threshold=options.inactive_dice_threshold,
            )
            for threshold in STOP_AT_THRESHOLDS
        ]
        for strat in stop_at_strats:
            strat.strategy_id = encode_strategy(strat, encoder)
            strategies.append(strat)
            stop_at_rows.append(list(strategy_tuple(strat)))
            stop_at_ids.append(strat.strategy_id)

    if stop_at_rows:
        stop_at_frame = pd.DataFrame(
            stop_at_rows,
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
        stop_at_frame["strategy_id"] = stop_at_ids
        meta = pd.concat([meta, stop_at_frame], ignore_index=True)
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
    include_stop_at: bool = False,
    include_stop_at_heuristic: bool = False,
) -> int:
    """Compute the number of strategies a grid configuration will yield.

    Parameters
    ----------
    score_thresholds : Sequence[int] | None, optional
        Candidate turn-score thresholds. ``None`` falls back to ``range(200, 1400, 50)``.
    dice_thresholds : Sequence[int] | None, optional
        Candidate dice thresholds. ``None`` falls back to ``range(0, 5)``.
    smart_five_and_one_options : Sequence[Sequence[bool]] | None, optional
        Explicit ``(smart_five, smart_one)`` pairs to allow. ``None`` permits all valid
        combinations with the implicit dependency that smart-one requires smart-five.
    consider_score_opts : Sequence[bool], optional
        Boolean values indicating whether score thresholds are considered.
    consider_dice_opts : Sequence[bool], optional
        Boolean values indicating whether dice thresholds are considered.
    auto_hot_dice_opts : Sequence[bool], optional
        Allowed values for the auto-hot-dice flag.
    run_up_score_opts : Sequence[bool], optional
        Allowed values for the run-up-score flag.
    include_stop_at, include_stop_at_heuristic
        When enabled, accounts for the registered stop-at strategies (with or
        without heuristics) that will be appended to the generated grid.

    Returns
    -------
    int
        Total number of unique strategy configurations that would be generated.
    """
    options = StrategyGridOptions.from_inputs(
        score_thresholds=score_thresholds,
        dice_thresholds=dice_thresholds,
        consider_score_opts=consider_score_opts,
        consider_dice_opts=consider_dice_opts,
        auto_hot_dice_opts=auto_hot_dice_opts,
        run_up_score_opts=run_up_score_opts,
    )
    score_thresholds_list = options.score_thresholds
    dice_thresholds_list = options.dice_thresholds
    consider_score_opts_list = options.consider_score_opts
    consider_dice_opts_list = options.consider_dice_opts
    auto_hot_dice_opts_list = options.auto_hot_dice_opts
    run_up_score_opts_list = options.run_up_score_opts

    if not score_thresholds_list:
        raise ValueError("score_thresholds must contain at least one value")
    if not dice_thresholds_list:
        raise ValueError("dice_thresholds must contain at least one value")

    inactive_score_threshold = options.inactive_score_threshold
    inactive_dice_threshold = options.inactive_dice_threshold

    if smart_five_and_one_options is None:
        smart_five_opts_list = options.smart_five_opts
        smart_one_opts_list = options.smart_one_opts
        allowed_pairs: set[tuple[bool, bool]] | None = None
    else:
        normalized_pairs: list[tuple[bool, bool]] = []
        for opts in smart_five_and_one_options:
            if len(opts) != 2:
                raise ValueError(
                    "smart_five_and_one_options entries must contain exactly two booleans"
                )
            normalized_pairs.append((bool(opts[0]), bool(opts[1])))
        allowed_pairs = set(normalized_pairs)
        smart_five_opts_list = tuple(dict.fromkeys(sf for sf, _ in normalized_pairs))
        smart_one_opts_list = tuple(dict.fromkeys(so for _, so in normalized_pairs))
        if not smart_five_opts_list or not smart_one_opts_list:
            return 0

    combo_iter = iter_strategy_combos(
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

    base_count = sum(1 for _ in combo_iter)
    stop_at_count = 0
    if include_stop_at:
        stop_at_count += len(STOP_AT_THRESHOLDS)
    if include_stop_at_heuristic:
        stop_at_count += len(STOP_AT_THRESHOLDS)

    return base_count + stop_at_count


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
    """Run games for a predetermined list of seeds.

    Parameters
    ----------
    seeds : Iterable[int]
        Deterministic seeds to feed into :func:`_play_game`. Each seed produces one game.
    strategies : Sequence[ThresholdStrategy]
        Strategies assigned to players in every simulated game.
    target_score : int, optional
        Score required to trigger the final round; defaults to ``10_000``.
    n_jobs : int, optional
        Number of worker processes to use. ``1`` executes serially; larger values spawn a pool.

    Returns
    -------
    pandas.DataFrame
        One row per simulated game, identical in shape to the output of
        :func:`simulate_many_games`.
    """
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
