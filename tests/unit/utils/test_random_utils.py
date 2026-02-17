import types

import numpy as np

from farkle.utils.random import make_rng, seed_everything, spawn_seeds


def test_spawn_seeds_deterministic():
    seeds1 = spawn_seeds(5, seed=123)
    seeds2 = spawn_seeds(5, seed=123)
    assert np.array_equal(seeds1, seeds2)
    assert seeds1.dtype == np.uint32


def test_make_rng_respects_seed():
    rng1 = make_rng(5)
    rng2 = make_rng(5)
    assert rng1.random() == rng2.random()


def test_seed_everything_optional_backends(monkeypatch):
    torch_calls: list[int] = []
    tf_calls: list[str] = []

    torch_mod = types.SimpleNamespace(
        manual_seed=lambda seed: torch_calls.append(seed),
        cuda=types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda _s: None),
    )

    tf_random = types.SimpleNamespace(set_seed=lambda seed: tf_calls.append(f"random:{seed}"))
    tf_keras_utils = types.SimpleNamespace(set_random_seed=lambda seed: tf_calls.append(f"keras:{seed}"))
    tf_mod = types.SimpleNamespace(random=tf_random, keras=types.SimpleNamespace(utils=tf_keras_utils))

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



def test_spawn_seeds_child_stream_determinism() -> None:
    root = np.random.SeedSequence(2024)
    children_a = root.spawn(2)
    root_replay = np.random.SeedSequence(2024)
    children_b = root_replay.spawn(2)

    seq_a = [spawn_seeds(4, seed=int(child.generate_state(1)[0])) for child in children_a]
    seq_b = [spawn_seeds(4, seed=int(child.generate_state(1)[0])) for child in children_b]

    assert all(np.array_equal(a, b) for a, b in zip(seq_a, seq_b, strict=False))
