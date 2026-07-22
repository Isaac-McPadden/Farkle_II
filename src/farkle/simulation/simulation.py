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
import pyarrow as pa

from farkle.game.engine import FarkleGame, FarklePlayer, TerminationStatus
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
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose, coordinate_rng, spawn_seeds
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION, raw_simulation_schema_for

__all__: list[str] = [
    "generate_strategy_grid",
    "experiment_size",
    "simulate_one_game",
    "simulate_many_games",
    "simulate_many_games_from_seeds",
    "aggregate_metrics",
    "simulation_rows_to_table",
    "validate_simulation_row",
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
            strategy_id=encoder.encode_tuple((st, dt, sf, so, cs, cd, rb, hd, rs, ps)),
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
    seed: int,
) -> List[FarklePlayer]:
    """Instantiate ``FarklePlayer`` objects for a table.

    Parameters
    ----------
    strategies
        Strategies applied to each player in order.
    seed
        Root seed used to derive coordinate-owned player generators.

    Returns
    -------
    List[FarklePlayer]
        Players ready to be passed to ``FarkleGame``.
    """

    return [
        FarklePlayer(
            name=f"P{i + 1}",
            strategy=s,
            rng=coordinate_rng(
                RandomPurpose.PLAYER,
                root_seed=seed,
                k=len(strategies),
                seat_index=i,
            ),
        )
        for i, s in enumerate(strategies)
    ]


def validate_simulation_row(row: Mapping[str, Any]) -> None:
    """Validate the closed outcome invariants for one flattened game row."""

    try:
        n_players = int(row["k"])
        status = TerminationStatus(row["termination_status"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Simulation row has invalid k or termination_status") from exc
    if n_players < 1:
        raise ValueError("Simulation row k must be positive")
    if row.get("outcome_schema_version") != OUTCOME_SCHEMA_VERSION:
        raise ValueError(f"Simulation row must use outcome_schema_version={OUTCOME_SCHEMA_VERSION}")

    seats = [f"P{index}" for index in range(1, n_players + 1)]
    missing_ranks = [seat for seat in seats if f"{seat}_rank" not in row]
    if missing_ranks:
        raise ValueError(f"Simulation row missing participant ranks for {missing_ranks}")
    ranks = [row[f"{seat}_rank"] for seat in seats]
    winner_seat = row.get("winner_seat")
    winner_strategy = row.get("winner_strategy")

    if status is TerminationStatus.COMPLETED:
        rank_one_seats = [seat for seat, rank in zip(seats, ranks, strict=True) if rank == 1]
        if len(rank_one_seats) != 1 or winner_seat != rank_one_seats[0]:
            raise ValueError(
                "Completed simulation row must have exactly one winner matching its rank-1 seat"
            )
        if any(rank is None for rank in ranks) or sorted(ranks) != list(range(1, n_players + 1)):
            raise ValueError("Completed simulation row ranks must be the permutation 1..k")
        if winner_strategy is None or winner_strategy != row.get(f"{winner_seat}_strategy"):
            raise ValueError("Completed simulation row must identify the winning strategy")
        if row.get("winning_score") is None or row.get("victory_margin") is None:
            raise ValueError("Completed simulation row must retain winner-conditioned fields")
        if row.get("hit_safety_limit") is not False:
            raise ValueError("Completed simulation row cannot hit the safety limit")
        expected_seat_ranks = [
            seat for _, seat in sorted(zip(ranks, seats, strict=True), key=lambda item: item[0])
        ]
        if row.get("seat_ranks") != expected_seat_ranks:
            raise ValueError("Completed simulation row has inconsistent seat_ranks")
        return

    if row.get("hit_safety_limit") is not True:
        raise ValueError("Safety-limit simulation row must set hit_safety_limit=true")
    winner_conditioned = {
        "winner_seat": winner_seat,
        "winner_strategy": winner_strategy,
        "winning_score": row.get("winning_score"),
        "victory_margin": row.get("victory_margin"),
    }
    present = [name for name, value in winner_conditioned.items() if value is not None]
    if present:
        raise ValueError(f"Safety-limit simulation row cannot claim a winner: {present}")
    if any(rank is not None for rank in ranks):
        raise ValueError("Safety-limit simulation row cannot assign participant ranks")
    if row.get("seat_ranks") != [None] * n_players:
        raise ValueError("Safety-limit simulation row must retain k null seat-rank entries")
    if any(row.get(f"{seat}_loss_margin") is not None for seat in seats):
        raise ValueError("Safety-limit simulation row cannot assign loss margins")


def simulation_rows_to_table(rows: Sequence[Mapping[str, Any]], n_players: int) -> pa.Table:
    """Validate and materialize raw rows with deliberate Arrow nullability."""

    for row in rows:
        validate_simulation_row(row)
        if int(row["k"]) != n_players:
            raise ValueError(f"Simulation row k={row['k']} does not match schema k={n_players}")
    return pa.Table.from_pylist(list(rows), schema=raw_simulation_schema_for(n_players))


def _play_game(
    seed: int,
    strategies: Sequence[ThresholdStrategy],
    target_score: int = 10_000,
    provenance: Mapping[str, Any] | None = None,
    max_rounds: int = 200,
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
    max_rounds
        Safety cap on complete table rounds.

    Returns
    -------
    Mapping[str, Any]
        Mapping of metric names to values including the winner and
        per-player statistics.
    """
    # give every player an *independent* PRNG, but reproducible
    players = _make_players(strategies, seed)
    game = FarkleGame(players, target_score=target_score, table_seed=seed)
    gm = game.play() if max_rounds == 200 else game.play(max_rounds=max_rounds)
    winners = [name for name, ps in gm.players.items() if ps.rank == 1]
    status = gm.game.termination_status
    if status is TerminationStatus.COMPLETED and len(winners) != 1:
        raise ValueError(
            "Completed game must have exactly one player with rank == 1, "
            f"got {len(winners)}: {winners}"
        )
    if status is TerminationStatus.SAFETY_LIMIT and winners:
        raise ValueError(f"Safety-limit game cannot have rank-1 players: {winners}")
    winner = winners[0] if winners else None
    seat_ranks: list[str | None]
    if winner is None:
        seat_ranks = [None] * len(strategies)
    else:
        seat_ranks = [
            name
            for name, _ in sorted(
                gm.players.items(),
                key=lambda item: item[1].rank if item[1].rank is not None else len(strategies) + 1,
            )
        ]
    flat: dict[str, Any] = {
        "termination_status": status.value,
        "hit_safety_limit": gm.game.hit_safety_limit,
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "winner_seat": winner,
        "winner_strategy": None if winner is None else gm.players[winner].strategy,
        "seat_ranks": seat_ranks,
        "winning_score": None if winner is None else gm.players[winner].score,
        "victory_margin": gm.game.margin,
        "n_rounds": gm.game.n_rounds,
        "root_seed": seed,
        "k": len(strategies),
        "shuffle_index": None,
        "game_index": None,
        "deterministic_batch_id": None,
        "game_seed": seed,
        "rng_scheme_version": RNG_SCHEME_VERSION,
        "rng_purpose_namespace": int(RandomPurpose.INDEXED_SEED),
    }
    if provenance is not None:
        flat.update(provenance)
    # Per-player metrics
    for pname, stats in gm.players.items():
        for k, v in asdict(stats).items():
            flat[f"{pname}_{k}"] = v

    validate_simulation_row(flat)
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
    if seed is None:
        raise ValueError("simulate_many_games requires an explicit seed")
    seeds = spawn_seeds(n_games, seed=seed)
    args = [
        (
            int(game_seed),
            strategies,
            target_score,
            {
                "root_seed": seed,
                "k": len(strategies),
                "shuffle_index": None,
                "game_index": game_index,
                "deterministic_batch_id": None,
                "game_seed": int(game_seed),
                "rng_scheme_version": RNG_SCHEME_VERSION,
                "rng_purpose_namespace": int(RandomPurpose.INDEXED_SEED),
            },
        )
        for game_index, game_seed in enumerate(seeds)
    ]
    if n_jobs == 1:
        rows = [_play_game(*a) for a in args]
    else:
        with mp.Pool(processes=n_jobs) as executor:
            rows = executor.starmap(_play_game, args)
    return pd.DataFrame(rows)


def simulate_many_games_from_seeds(
    *,
    seeds: Iterable[int],
    strategies: Sequence[ThresholdStrategy],
    target_score: int = 10_000,
    n_jobs: int = 1,
    root_seed: int | None = None,
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
        Number of worker processes to use. ``1`` executes serially; larger values use an executor.

    Returns
    -------
    pandas.DataFrame
        One row per simulated game, identical in shape to the output of
        :func:`simulate_many_games`.
    """
    args = [
        (
            int(game_seed),
            strategies,
            target_score,
            {
                "root_seed": int(game_seed) if root_seed is None else root_seed,
                "k": len(strategies),
                "shuffle_index": None,
                "game_index": game_index,
                "deterministic_batch_id": None,
                "game_seed": int(game_seed),
                "rng_scheme_version": RNG_SCHEME_VERSION,
                "rng_purpose_namespace": int(RandomPurpose.INDEXED_SEED),
            },
        )
        for game_index, game_seed in enumerate(seeds)
    ]
    if n_jobs == 1:
        rows = [_play_game(*a) for a in args]
    else:
        with mp.Pool(processes=n_jobs) as executor:
            rows = executor.starmap(_play_game, args)
    return pd.DataFrame(rows)


def simulate_one_game(
    *,
    strategies: Sequence[ThresholdStrategy],
    target_score: int = 10_000,
    seed: int,
):
    """Play a single game using the provided strategies.

    Parameters
    ----------
    strategies
        Strategy objects applied to the players.
    target_score
        Score needed to trigger the final round.
    seed
        Required root seed controlling all random number generators.

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
        "winner_freq": df["winner_seat"].value_counts().to_dict(),
    }
