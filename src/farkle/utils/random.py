"""Versioned, coordinate-derived NumPy random streams."""

from __future__ import annotations

import contextlib
import importlib
from enum import IntEnum
from types import MappingProxyType
from typing import Final, Protocol

import numpy as np

RNG_SCHEME_VERSION: Final = 1
MAX_UINT32: Final = 2**32 - 1
MAX_UINT64: Final = 2**64 - 1


class RandomPurpose(IntEnum):
    """Permanent integer namespaces; existing values must never be renumbered."""

    INDEXED_SEED = 1
    PLAYER = 10
    STRATEGY = 11
    TOURNAMENT_SHUFFLE = 100
    SHUFFLE_PERMUTATION = 101
    TOURNAMENT_GAME = 102
    H2H_PAIR = 200
    H2H_ORDER = 201
    H2H_GAME = 202
    TRUESKILL_DIAGNOSTIC = 300
    BOOTSTRAP = 400
    TIE_BREAK = 500
    HGB = 600
    SEED_SELECTION = 700


PURPOSE_NAMESPACES = MappingProxyType(
    {purpose.name.lower(): int(purpose) for purpose in RandomPurpose}
)


class RngProtocol(Protocol):
    """Protocol for RNGs that behave like ``numpy.random.Generator``."""

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | None = None,
        dtype=np.int64,
        endpoint: bool = False,
    ) -> np.ndarray: ...


def _uint64_words(value: int, *, name: str) -> tuple[int, int]:
    if isinstance(value, bool) or not 0 <= int(value) <= MAX_UINT64:
        raise ValueError(f"{name} must be an integer in [0, 2**64 - 1]")
    normalized = int(value)
    return normalized & MAX_UINT32, normalized >> 32


def coordinate_seed_sequence(
    purpose: RandomPurpose | int,
    *,
    root_seed: int,
    k: int = 0,
    shuffle_index: int = 0,
    pair_index: int = 0,
    order: int = 0,
    game_index: int = 0,
    seat_index: int = 0,
    replicate_index: int = 0,
) -> np.random.SeedSequence:
    """Return a SeedSequence determined only by fixed semantic coordinates."""

    try:
        namespace = RandomPurpose(int(purpose))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unregistered RNG purpose namespace: {purpose!r}") from exc
    entropy: list[int] = [RNG_SCHEME_VERSION, int(namespace)]
    for name, value in (
        ("root_seed", root_seed),
        ("k", k),
        ("shuffle_index", shuffle_index),
        ("pair_index", pair_index),
        ("order", order),
        ("game_index", game_index),
        ("seat_index", seat_index),
        ("replicate_index", replicate_index),
    ):
        entropy.extend(_uint64_words(value, name=name))
    return np.random.SeedSequence(entropy)


def coordinate_rng(
    purpose: RandomPurpose | int,
    *,
    root_seed: int,
    k: int = 0,
    shuffle_index: int = 0,
    pair_index: int = 0,
    order: int = 0,
    game_index: int = 0,
    seat_index: int = 0,
    replicate_index: int = 0,
) -> np.random.Generator:
    """Build an explicit PCG64DXSM generator for semantic coordinates."""

    seed_sequence = coordinate_seed_sequence(
        purpose,
        root_seed=root_seed,
        k=k,
        shuffle_index=shuffle_index,
        pair_index=pair_index,
        order=order,
        game_index=game_index,
        seat_index=seat_index,
        replicate_index=replicate_index,
    )
    return np.random.Generator(np.random.PCG64DXSM(seed_sequence))


def coordinate_seed(
    purpose: RandomPurpose | int,
    *,
    root_seed: int,
    k: int = 0,
    shuffle_index: int = 0,
    pair_index: int = 0,
    order: int = 0,
    game_index: int = 0,
    seat_index: int = 0,
    replicate_index: int = 0,
    dtype: type[np.uint32] | type[np.uint64] = np.uint64,
) -> int:
    """Return a stable integer identity without consuming a generator stream."""

    sequence = coordinate_seed_sequence(
        purpose,
        root_seed=root_seed,
        k=k,
        shuffle_index=shuffle_index,
        pair_index=pair_index,
        order=order,
        game_index=game_index,
        seat_index=seat_index,
        replicate_index=replicate_index,
    )
    return int(sequence.generate_state(1, dtype=dtype)[0])


def make_rng(seed: int) -> np.random.Generator:
    """Return an explicit PCG64DXSM generator from a required root seed."""

    return coordinate_rng(RandomPurpose.INDEXED_SEED, root_seed=seed)


def spawn_seeds(n: int, *, seed: int) -> np.ndarray:
    """Return indexed uint32 seeds independent of draw history and worker order."""

    if isinstance(n, bool) or n < 0:
        raise ValueError("n must be a non-negative integer")
    return np.asarray(
        [
            coordinate_seed(
                RandomPurpose.INDEXED_SEED,
                root_seed=seed,
                game_index=index,
                dtype=np.uint32,
            )
            for index in range(n)
        ],
        dtype=np.uint32,
    )


def seed_everything(seed: int) -> None:
    """Seed optional external ML backends; project RNGs remain coordinate-owned."""

    try:  # optional dependency; ignore if unavailable
        torch = importlib.import_module("torch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CUDA not in CI
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional / CUDA absence
        pass

    try:  # optional dependency; ignore if unavailable
        tf = importlib.import_module("tensorflow")
        with contextlib.suppress(Exception):
            tf.random.set_seed(seed)
        with contextlib.suppress(Exception):
            keras_utils = getattr(tf.keras, "utils", None)
            if keras_utils and hasattr(keras_utils, "set_random_seed"):
                keras_utils.set_random_seed(seed)
    except Exception:  # pragma: no cover - tensorflow optional
        pass


__all__ = [
    "MAX_UINT32",
    "PURPOSE_NAMESPACES",
    "RNG_SCHEME_VERSION",
    "RandomPurpose",
    "RngProtocol",
    "coordinate_rng",
    "coordinate_seed",
    "coordinate_seed_sequence",
    "make_rng",
    "seed_everything",
    "spawn_seeds",
]
