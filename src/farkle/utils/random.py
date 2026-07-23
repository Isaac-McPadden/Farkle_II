"""Versioned, coordinate-derived NumPy random streams."""

from __future__ import annotations

import contextlib
import importlib
from enum import IntEnum
from types import MappingProxyType
from typing import Final, Protocol

import numpy as np

RNG_SCHEME_VERSION: Final = 2
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
    TOURNAMENT_PLAYER = 103
    H2H_PAIR = 200
    H2H_ORDER = 201
    H2H_GAME = 202
    H2H_PLAYER = 203
    TRUESKILL_DIAGNOSTIC = 300
    BOOTSTRAP = 400
    ROOT_STABILITY_BOOTSTRAP = 401
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


def _resolve_coordinate_alias(
    primary: int | None,
    alias: int | None,
    *,
    primary_name: str,
    alias_name: str,
) -> int:
    """Resolve two names for one semantic coordinate without ambiguity."""

    if primary is not None and alias is not None and int(primary) != int(alias):
        raise ValueError(f"{primary_name} and {alias_name} identify different coordinates")
    value = primary if primary is not None else alias
    return 0 if value is None else int(value)


def coordinate_entropy(
    purpose: RandomPurpose | int,
    *,
    root_seed: int,
    k: int = 0,
    shuffle_index: int = 0,
    pair_index: int | None = None,
    pair_id: int | None = None,
    order: int = 0,
    game_index: int | None = None,
    attempt_index: int | None = None,
    seat_index: int = 0,
    replicate_index: int = 0,
) -> tuple[int, ...]:
    """Return the lossless SeedSequence entropy for semantic coordinates."""

    try:
        namespace = RandomPurpose(int(purpose))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unregistered RNG purpose namespace: {purpose!r}") from exc
    resolved_pair_id = _resolve_coordinate_alias(
        pair_index,
        pair_id,
        primary_name="pair_index",
        alias_name="pair_id",
    )
    resolved_game_index = _resolve_coordinate_alias(
        game_index,
        attempt_index,
        primary_name="game_index",
        alias_name="attempt_index",
    )
    entropy: list[int] = [RNG_SCHEME_VERSION, int(namespace)]
    for name, value in (
        ("root_seed", root_seed),
        ("k", k),
        ("shuffle_index", shuffle_index),
        ("pair_id", resolved_pair_id),
        ("order", order),
        ("game_index", resolved_game_index),
        ("seat_index", seat_index),
        ("replicate_index", replicate_index),
    ):
        entropy.extend(_uint64_words(value, name=name))
    return tuple(entropy)


def coordinate_seed_sequence(
    purpose: RandomPurpose | int,
    *,
    root_seed: int,
    k: int = 0,
    shuffle_index: int = 0,
    pair_index: int | None = None,
    pair_id: int | None = None,
    order: int = 0,
    game_index: int | None = None,
    attempt_index: int | None = None,
    seat_index: int = 0,
    replicate_index: int = 0,
) -> np.random.SeedSequence:
    """Return a SeedSequence determined only by fixed semantic coordinates."""

    entropy = coordinate_entropy(
        purpose,
        root_seed=root_seed,
        k=k,
        shuffle_index=shuffle_index,
        pair_index=pair_index,
        pair_id=pair_id,
        order=order,
        game_index=game_index,
        attempt_index=attempt_index,
        seat_index=seat_index,
        replicate_index=replicate_index,
    )
    return np.random.SeedSequence(entropy)


def coordinate_rng(
    purpose: RandomPurpose | int,
    *,
    root_seed: int,
    k: int = 0,
    shuffle_index: int = 0,
    pair_index: int | None = None,
    pair_id: int | None = None,
    order: int = 0,
    game_index: int | None = None,
    attempt_index: int | None = None,
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
        pair_id=pair_id,
        order=order,
        game_index=game_index,
        attempt_index=attempt_index,
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
    pair_index: int | None = None,
    pair_id: int | None = None,
    order: int = 0,
    game_index: int | None = None,
    attempt_index: int | None = None,
    seat_index: int = 0,
    replicate_index: int = 0,
    dtype: type[np.uint32] | type[np.uint64] = np.uint64,
) -> int:
    """Return a diagnostic/external-boundary fingerprint for coordinates.

    The reduced scalar is never an authoritative coordinate or a root for a
    project-owned child stream.
    """

    sequence = coordinate_seed_sequence(
        purpose,
        root_seed=root_seed,
        k=k,
        shuffle_index=shuffle_index,
        pair_index=pair_index,
        pair_id=pair_id,
        order=order,
        game_index=game_index,
        attempt_index=attempt_index,
        seat_index=seat_index,
        replicate_index=replicate_index,
    )
    return int(sequence.generate_state(1, dtype=dtype)[0])


def tournament_player_rng(
    *,
    root_seed: int,
    k: int,
    shuffle_index: int,
    game_index: int,
    seat_index: int,
) -> np.random.Generator:
    """Return one tournament seat stream from its complete semantic coordinate."""

    return coordinate_rng(
        RandomPurpose.TOURNAMENT_PLAYER,
        root_seed=root_seed,
        k=k,
        shuffle_index=shuffle_index,
        game_index=game_index,
        seat_index=seat_index,
    )


def h2h_player_rng(
    *,
    root_seed: int,
    pair_id: int,
    order: int,
    attempt_index: int,
    seat_index: int,
) -> np.random.Generator:
    """Return one H2H seat stream from its complete semantic coordinate."""

    return coordinate_rng(
        RandomPurpose.H2H_PLAYER,
        root_seed=root_seed,
        k=2,
        pair_id=pair_id,
        order=order,
        attempt_index=attempt_index,
        seat_index=seat_index,
    )


def make_rng(seed: int) -> np.random.Generator:
    """Return an explicit PCG64DXSM generator from a required root seed."""

    return coordinate_rng(RandomPurpose.INDEXED_SEED, root_seed=seed)


def spawn_seeds(n: int, *, seed: int) -> np.ndarray:
    """Return legacy external-boundary seeds independent of draw history.

    Project-owned tournament and H2H streams must use full semantic coordinates
    directly instead of re-rooting from these reduced values.
    """

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
    "coordinate_entropy",
    "coordinate_rng",
    "coordinate_seed",
    "coordinate_seed_sequence",
    "h2h_player_rng",
    "make_rng",
    "seed_everything",
    "spawn_seeds",
    "tournament_player_rng",
]
