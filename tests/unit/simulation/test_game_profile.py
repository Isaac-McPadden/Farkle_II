from __future__ import annotations

import multiprocessing as mp
import pickle

import pytest

import farkle.simulation.run_tournament as tournament
from farkle.simulation.game_profile import (
    GameProfile,
    H2HMaxRoundsOverride,
    TournamentMaxRoundsOverride,
)
from farkle.simulation.simulation import generate_strategy_grid


def _spawned_profile_identity(profile: GameProfile) -> str:
    return profile.sha256


def test_game_profile_matches_only_complete_semantic_coordinate() -> None:
    profile = GameProfile(
        default_target_score=100,
        default_max_rounds=200,
        tournament_max_rounds_overrides=(TournamentMaxRoundsOverride(11, 2, 0, 0, 0),),
        h2h_max_rounds_overrides=(H2HMaxRoundsOverride(11, 1, 0, 0, 0),),
    )

    assert (
        profile.tournament_limits(root_seed=11, k=2, shuffle_index=0, game_index=0).max_rounds == 0
    )
    assert (
        profile.tournament_limits(root_seed=11, k=2, shuffle_index=0, game_index=1).max_rounds
        == 200
    )
    assert profile.h2h_limits(root_seed=11, pair_id=1, order=0, attempt_index=0).max_rounds == 0
    assert profile.h2h_limits(root_seed=22, pair_id=1, order=0, attempt_index=0).max_rounds == 200


def test_game_profile_is_picklable_and_has_order_independent_identity() -> None:
    overrides = (
        TournamentMaxRoundsOverride(22, 4, 1, 0, 7),
        TournamentMaxRoundsOverride(11, 2, 0, 0, 0),
    )
    profile = GameProfile(tournament_max_rounds_overrides=overrides)
    reordered = GameProfile(tournament_max_rounds_overrides=tuple(reversed(overrides)))

    assert pickle.loads(pickle.dumps(profile)) == profile
    assert reordered.sha256 == profile.sha256

    spawn_context = mp.get_context("spawn")
    with spawn_context.Pool(processes=1) as pool:
        assert pool.apply(_spawned_profile_identity, (profile,)) == profile.sha256


def test_game_profile_rejects_mutable_or_outcome_shaped_inputs() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        GameProfile(tournament_max_rounds_overrides=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GameProfile(winner_strategy=1)  # type: ignore[call-arg]


def test_omitted_profile_and_production_default_profile_play_identically() -> None:
    strategies, _ = generate_strategy_grid(
        score_thresholds=[500],
        dice_thresholds=[2],
        smart_five_opts=[False],
        smart_one_opts=[False],
        consider_score_opts=[True],
        consider_dice_opts=[True],
        auto_hot_dice_opts=[False],
        run_up_score_opts=[False],
    )
    config = tournament.TournamentConfig(
        n_players=2,
        n_strategies=2,
    )
    task = tournament.ShuffleTask(
        root_seed=31,
        k=2,
        shuffle_index=4,
        shuffle_seed=123,
        deterministic_batch_id=4,
    )

    tournament._init_worker(strategies, config)
    without_profile = tournament._play_one_shuffle(task, collect_rows=True)[3]
    tournament._init_worker(strategies, config, GameProfile())
    with_default_profile = tournament._play_one_shuffle(task, collect_rows=True)[3]

    assert with_default_profile == without_profile
