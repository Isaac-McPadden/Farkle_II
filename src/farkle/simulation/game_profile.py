"""Authenticated, limit-only game settings for deterministic workflow oracles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from farkle.utils.authenticated_contract import identity_sha256

GAME_PROFILE_CONTRACT_VERSION: Final = 1


def _require_coordinate(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_max_rounds(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_rounds must be a non-negative integer")


@dataclass(frozen=True, slots=True, order=True)
class TournamentMaxRoundsOverride:
    """One tournament semantic coordinate with a replacement safety limit."""

    root_seed: int
    k: int
    shuffle_index: int
    game_index: int
    max_rounds: int

    def __post_init__(self) -> None:
        for name in ("root_seed", "k", "shuffle_index", "game_index"):
            _require_coordinate(getattr(self, name), name=name)
        if self.k < 2:
            raise ValueError("k must be at least 2")
        _require_max_rounds(self.max_rounds)

    @property
    def coordinate(self) -> tuple[int, int, int, int]:
        """Return the complete tournament coordinate owned by this override."""

        return (self.root_seed, self.k, self.shuffle_index, self.game_index)


@dataclass(frozen=True, slots=True, order=True)
class H2HMaxRoundsOverride:
    """One H2H semantic coordinate with a replacement safety limit."""

    root_seed: int
    pair_id: int
    order: int
    attempt_index: int
    max_rounds: int

    def __post_init__(self) -> None:
        for name in ("root_seed", "pair_id", "order", "attempt_index"):
            _require_coordinate(getattr(self, name), name=name)
        if self.order not in (0, 1):
            raise ValueError("order must be 0 or 1")
        _require_max_rounds(self.max_rounds)

    @property
    def coordinate(self) -> tuple[int, int, int, int]:
        """Return the complete H2H coordinate owned by this override."""

        return (self.root_seed, self.pair_id, self.order, self.attempt_index)


@dataclass(frozen=True, slots=True)
class GameLimits:
    """Arguments already accepted by the production game engine."""

    target_score: int
    max_rounds: int


@dataclass(frozen=True, slots=True)
class GameProfile:
    """Immutable, picklable limit profile for deterministic workflow oracles.

    The descriptor can select only the target score and safety-round limit. It
    cannot provide winners, ranks, rows, counters, or other simulated outcomes.
    """

    default_target_score: int = 10_000
    default_max_rounds: int = 200
    tournament_max_rounds_overrides: tuple[TournamentMaxRoundsOverride, ...] = ()
    h2h_max_rounds_overrides: tuple[H2HMaxRoundsOverride, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.default_target_score, bool)
            or not isinstance(self.default_target_score, int)
            or self.default_target_score <= 0
        ):
            raise ValueError("default_target_score must be a positive integer")
        _require_max_rounds(self.default_max_rounds)
        if not isinstance(self.tournament_max_rounds_overrides, tuple):
            raise TypeError("tournament_max_rounds_overrides must be a tuple")
        if not isinstance(self.h2h_max_rounds_overrides, tuple):
            raise TypeError("h2h_max_rounds_overrides must be a tuple")
        tournament_coordinates = [
            override.coordinate for override in self.tournament_max_rounds_overrides
        ]
        if len(set(tournament_coordinates)) != len(tournament_coordinates):
            raise ValueError("tournament max-round overrides contain duplicate coordinates")
        h2h_coordinates = [override.coordinate for override in self.h2h_max_rounds_overrides]
        if len(set(h2h_coordinates)) != len(h2h_coordinates):
            raise ValueError("H2H max-round overrides contain duplicate coordinates")

    def canonical_payload(self) -> dict[str, object]:
        """Return the canonical, order-independent profile identity payload."""

        return {
            "game_profile_contract_version": GAME_PROFILE_CONTRACT_VERSION,
            "default_target_score": self.default_target_score,
            "default_max_rounds": self.default_max_rounds,
            "tournament_max_rounds_overrides": [
                asdict(override)
                for override in sorted(
                    self.tournament_max_rounds_overrides,
                    key=lambda item: item.coordinate,
                )
            ],
            "h2h_max_rounds_overrides": [
                asdict(override)
                for override in sorted(
                    self.h2h_max_rounds_overrides,
                    key=lambda item: item.coordinate,
                )
            ],
        }

    @property
    def sha256(self) -> str:
        """Return the canonical SHA-256 identity of this profile."""

        return identity_sha256(self.canonical_payload())

    def tournament_limits(
        self,
        *,
        root_seed: int,
        k: int,
        shuffle_index: int,
        game_index: int,
    ) -> GameLimits:
        """Resolve limits for exactly one tournament semantic coordinate."""

        coordinate = (root_seed, k, shuffle_index, game_index)
        max_rounds = self.default_max_rounds
        for override in self.tournament_max_rounds_overrides:
            if override.coordinate == coordinate:
                max_rounds = override.max_rounds
                break
        return GameLimits(
            target_score=self.default_target_score,
            max_rounds=max_rounds,
        )

    def h2h_limits(
        self,
        *,
        root_seed: int,
        pair_id: int,
        order: int,
        attempt_index: int,
    ) -> GameLimits:
        """Resolve limits for exactly one H2H semantic coordinate."""

        coordinate = (root_seed, pair_id, order, attempt_index)
        max_rounds = self.default_max_rounds
        for override in self.h2h_max_rounds_overrides:
            if override.coordinate == coordinate:
                max_rounds = override.max_rounds
                break
        return GameLimits(
            target_score=self.default_target_score,
            max_rounds=max_rounds,
        )


__all__ = [
    "GAME_PROFILE_CONTRACT_VERSION",
    "GameLimits",
    "GameProfile",
    "H2HMaxRoundsOverride",
    "TournamentMaxRoundsOverride",
]
