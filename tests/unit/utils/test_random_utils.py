import types
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from farkle.utils.random import (
    PURPOSE_NAMESPACES,
    RNG_SCHEME_VERSION,
    RandomPurpose,
    coordinate_entropy,
    coordinate_rng,
    coordinate_seed,
    make_rng,
    seed_everything,
    spawn_seeds,
)


def _coordinate_payload(index: int) -> tuple[int, bytes]:
    rng = coordinate_rng(
        RandomPurpose.TOURNAMENT_PLAYER,
        root_seed=92821,
        k=4,
        shuffle_index=index // 5,
        game_index=index % 5,
        seat_index=index % 4,
    )
    return index, rng.bytes(128)


def _v1_tournament_game_seed(*, root_seed: int, k: int, shuffle_index: int, game_index: int) -> int:
    """Independent oracle for the reviewed v1 uint32 narrowing path."""

    mask = 2**32 - 1
    entropy = [1, 102]
    for value in (root_seed, k, shuffle_index, 0, 0, game_index, 0, 0):
        entropy.extend((value & mask, value >> 32))
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def test_spawn_seeds_deterministic():
    seeds1 = spawn_seeds(5, seed=123)
    seeds2 = spawn_seeds(5, seed=123)
    assert np.array_equal(seeds1, seeds2)
    assert seeds1.dtype == np.uint32


def test_make_rng_respects_seed():
    rng1 = make_rng(5)
    rng2 = make_rng(5)
    assert rng1.random() == rng2.random()
    assert isinstance(rng1.bit_generator, np.random.PCG64DXSM)


def test_coordinate_streams_are_stable_and_purpose_separated() -> None:
    coordinates = {
        "root_seed": 123,
        "k": 4,
        "shuffle_index": 9,
        "game_index": 7,
        "seat_index": 2,
    }
    first = coordinate_rng(RandomPurpose.PLAYER, **coordinates).bytes(64)
    replay = coordinate_rng(RandomPurpose.PLAYER, **coordinates).bytes(64)
    other_purpose = coordinate_rng(RandomPurpose.TOURNAMENT_GAME, **coordinates).bytes(64)

    assert first == replay
    assert first != other_purpose
    assert RNG_SCHEME_VERSION == 2


def test_reviewed_v1_uint32_collision_has_distinct_v2_player_streams() -> None:
    first = {"root_seed": 32, "k": 2, "shuffle_index": 194, "game_index": 18}
    second = {"root_seed": 32, "k": 2, "shuffle_index": 4052, "game_index": 4}

    assert _v1_tournament_game_seed(**first) == 2_963_478_802
    assert _v1_tournament_game_seed(**second) == 2_963_478_802
    assert coordinate_rng(RandomPurpose.TOURNAMENT_PLAYER, **first, seat_index=0).bytes(
        128
    ) != coordinate_rng(RandomPurpose.TOURNAMENT_PLAYER, **second, seat_index=0).bytes(128)


def test_v2_separates_root_k_batch_seat_and_purpose_and_replays() -> None:
    base = {
        "root_seed": 32,
        "k": 2,
        "shuffle_index": 29,
        "game_index": 7,
        "seat_index": 0,
    }
    selected = coordinate_rng(RandomPurpose.TOURNAMENT_PLAYER, **base).bytes(128)
    variants = [
        coordinate_rng(RandomPurpose.TOURNAMENT_PLAYER, **{**base, "root_seed": 33}).bytes(128),
        coordinate_rng(RandomPurpose.TOURNAMENT_PLAYER, **{**base, "k": 4}).bytes(128),
        coordinate_rng(RandomPurpose.TOURNAMENT_PLAYER, **{**base, "shuffle_index": 30}).bytes(128),
        coordinate_rng(RandomPurpose.TOURNAMENT_PLAYER, **{**base, "seat_index": 1}).bytes(128),
        coordinate_rng(RandomPurpose.BOOTSTRAP, **base).bytes(128),
        coordinate_rng(
            RandomPurpose.H2H_PLAYER,
            root_seed=32,
            k=2,
            pair_id=0,
            order=0,
            attempt_index=7,
            seat_index=0,
        ).bytes(128),
    ]

    assert selected == coordinate_rng(RandomPurpose.TOURNAMENT_PLAYER, **base).bytes(128)
    assert len({selected, *variants}) == 1 + len(variants)


def test_production_scale_coordinate_identity_enumeration_without_games() -> None:
    """Enumerate one million seat identities without materializing outcomes."""

    previous: tuple[int, ...] | None = None
    enumerated = 0
    for shuffle_index in range(4_000):
        for game_index in range(125):
            for seat_index in range(2):
                identity = coordinate_entropy(
                    RandomPurpose.TOURNAMENT_PLAYER,
                    root_seed=102,
                    k=2,
                    shuffle_index=shuffle_index,
                    game_index=game_index,
                    seat_index=seat_index,
                )
                assert identity != previous
                previous = identity
                enumerated += 1

    assert enumerated == 1_000_000


def test_coordinate_changes_are_local_and_do_not_depend_on_draw_history() -> None:
    game_seeds = [
        coordinate_seed(
            RandomPurpose.TOURNAMENT_GAME,
            root_seed=11,
            k=2,
            shuffle_index=3,
            game_index=index,
        )
        for index in range(4)
    ]
    unrelated = coordinate_rng(RandomPurpose.BOOTSTRAP, root_seed=11)
    unrelated.random(10_000)
    replay = [
        coordinate_seed(
            RandomPurpose.TOURNAMENT_GAME,
            root_seed=11,
            k=2,
            shuffle_index=3,
            game_index=index,
        )
        for index in range(4)
    ]

    assert replay == game_seeds
    assert len(set(game_seeds)) == len(game_seeds)


def test_purpose_registry_is_unique_and_unknown_namespaces_fail() -> None:
    assert len(PURPOSE_NAMESPACES) == len(set(PURPOSE_NAMESPACES.values()))
    with pytest.raises(ValueError, match="unregistered RNG purpose"):
        coordinate_rng(999, root_seed=1)
    with pytest.raises(TypeError):
        PURPOSE_NAMESPACES["new"] = 999  # type: ignore[index]


def test_coordinate_bytes_ignore_executor_count_order_and_resume_boundary() -> None:
    indices = list(range(20))
    serial = dict(map(_coordinate_payload, indices))
    with ProcessPoolExecutor(max_workers=1) as executor:
        one_worker = dict(executor.map(_coordinate_payload, reversed(indices)))
    with ProcessPoolExecutor(max_workers=2) as executor:
        first_half = dict(executor.map(_coordinate_payload, indices[:7]))
    with ProcessPoolExecutor(max_workers=3) as executor:
        resumed = dict(executor.map(_coordinate_payload, indices[7:]))

    assert one_worker == serial
    assert {**first_half, **resumed} == serial


def test_seed_everything_optional_backends(monkeypatch):
    torch_calls: list[int] = []
    tf_calls: list[str] = []

    torch_mod = types.SimpleNamespace(
        manual_seed=lambda seed: torch_calls.append(seed),
        cuda=types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda _s: None),
    )

    tf_random = types.SimpleNamespace(set_seed=lambda seed: tf_calls.append(f"random:{seed}"))
    tf_keras_utils = types.SimpleNamespace(
        set_random_seed=lambda seed: tf_calls.append(f"keras:{seed}")
    )
    tf_mod = types.SimpleNamespace(
        random=tf_random, keras=types.SimpleNamespace(utils=tf_keras_utils)
    )

    def fake_import(name):
        if name == "torch":
            return torch_mod
        if name == "tensorflow":
            return tf_mod
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", fake_import)

    seed_everything(77)

    assert torch_calls == [77]
    assert tf_calls == ["random:77", "keras:77"]


def test_seed_everything_ignores_torch_import_error(monkeypatch) -> None:
    calls: list[str] = []

    tf_mod = types.SimpleNamespace(
        random=types.SimpleNamespace(set_seed=lambda _seed: calls.append("tf"))
    )

    def fake_import(name: str):
        if name == "torch":
            raise RuntimeError("torch backend unavailable")
        if name == "tensorflow":
            return tf_mod
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", fake_import)

    seed_everything(42)

    assert calls == ["tf"]


def test_seed_everything_suppresses_tensorflow_set_seed_error(monkeypatch) -> None:
    torch_mod = types.SimpleNamespace(
        manual_seed=lambda _seed: None,
        cuda=types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda _s: None),
    )
    tf_mod = types.SimpleNamespace(
        random=types.SimpleNamespace(
            set_seed=lambda _seed: (_ for _ in ()).throw(RuntimeError("bad tf seed"))
        ),
        keras=types.SimpleNamespace(
            utils=types.SimpleNamespace(set_random_seed=lambda _seed: None)
        ),
    )

    def fake_import(name: str):
        if name == "torch":
            return torch_mod
        if name == "tensorflow":
            return tf_mod
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", fake_import)

    seed_everything(9)


def test_spawn_seeds_child_stream_determinism() -> None:
    root = np.random.SeedSequence(2024)
    children_a = root.spawn(2)
    root_replay = np.random.SeedSequence(2024)
    children_b = root_replay.spawn(2)

    seq_a = [spawn_seeds(4, seed=int(child.generate_state(1)[0])) for child in children_a]
    seq_b = [spawn_seeds(4, seed=int(child.generate_state(1)[0])) for child in children_b]

    assert all(np.array_equal(a, b) for a, b in zip(seq_a, seq_b, strict=False))
