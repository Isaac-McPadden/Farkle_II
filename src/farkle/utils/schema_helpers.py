# src/farkle/utils/schema_helpers.py
"""
Schema utilities for game outputs. Defines the canonical seat-level columns,
builds expected schemas for a given player count, and offers helpers for
deriving metadata such as player counts or batch sizes.
"""

from __future__ import annotations

import re
from typing import Final

import pyarrow as pa

OUTCOME_SCHEMA_VERSION: Final[int] = 2
TOURNAMENT_METHOD_VERSION: Final[int] = 2

# ---------- static pieces -------------------------------------------------
_NULLABLE_STRING_LIST: Final[pa.ListType] = pa.list_(pa.field("item", pa.string(), nullable=True))

_BASE_FIELDS: Final[list[pa.Field]] = [
    pa.field("root_seed", pa.int64(), nullable=False),
    pa.field("k", pa.int16(), nullable=False),
    pa.field("shuffle_index", pa.int64(), nullable=False),
    pa.field("game_index", pa.int32(), nullable=False),
    pa.field("deterministic_batch_id", pa.int32(), nullable=False),
    pa.field("shuffle_seed", pa.int64(), nullable=False),
    pa.field("termination_status", pa.string(), nullable=False),
    pa.field("hit_safety_limit", pa.bool_(), nullable=False),
    pa.field("outcome_schema_version", pa.int16(), nullable=False),
    pa.field("winner_seat", pa.string(), nullable=True),
    pa.field("winner_strategy", pa.int32(), nullable=True),
    pa.field("game_seed", pa.int64(), nullable=False),
    pa.field("rng_scheme_version", pa.int16(), nullable=False),
    pa.field("rng_purpose_namespace", pa.int32(), nullable=False),
    pa.field("seat_ranks", _NULLABLE_STRING_LIST, nullable=False),
    pa.field("winning_score", pa.int32(), nullable=True),
    pa.field("victory_margin", pa.int32(), nullable=True),
    pa.field("n_rounds", pa.int16(), nullable=False),
]

_SEAT_TEMPLATE: Final[dict[str, tuple[pa.DataType, bool]]] = {
    "score": (pa.int32(), False),
    "farkles": (pa.int16(), False),
    "rolls": (pa.int16(), False),
    "highest_turn": (pa.int16(), False),
    "strategy": (pa.int32(), False),
    "rank": (pa.int8(), True),
    "loss_margin": (pa.int32(), True),
    "smart_five_uses": (pa.int16(), False),
    "n_smart_five_dice": (pa.int16(), False),
    "smart_one_uses": (pa.int16(), False),
    "n_smart_one_dice": (pa.int16(), False),
    "hot_dice": (pa.int16(), False),  # counts of hot-dice used this game
    "n_turns": (pa.int16(), False),
    "hit_max_rounds": (pa.bool_(), False),
    # add/remove seat-level cols here once
}

# ---------- public helpers -----------------------------------------------


def expected_schema_for(n_players: int) -> pa.Schema:
    """Return the canonical analysis schema for *n_players* seats."""

    seat_fields: list[pa.Field] = []
    for i in range(1, n_players + 1):
        for suffix, (dtype, _nullable) in _SEAT_TEMPLATE.items():
            seat_fields.append(pa.field(f"P{i}_{suffix}", dtype, nullable=True))
    # Analysis tables pad unavailable columns while combining k cells, so the
    # rectangular analysis schema must permit nulls independently of the stricter
    # persisted raw-row contract below.
    base_fields = [pa.field(field.name, field.type, nullable=True) for field in _BASE_FIELDS]
    return pa.schema(base_fields + seat_fields)


def raw_simulation_schema_for(n_players: int) -> pa.Schema:
    """Return the typed outcome-schema-v2 schema for persisted simulation rows."""

    if n_players < 1:
        raise ValueError("n_players must be positive")
    seat_fields = [
        pa.field(f"P{i}_{suffix}", dtype, nullable=nullable)
        for i in range(1, n_players + 1)
        for suffix, (dtype, nullable) in _SEAT_TEMPLATE.items()
    ]
    return pa.schema([*_BASE_FIELDS, *seat_fields])


_PNUM_RE = re.compile(r"^P(\d+)_")  # Regex for P<X>_


def n_players_from_schema(schema: pa.Schema) -> int:
    """Infer the maximum player index from seat-oriented schema fields."""
    pnums = []
    for name in schema.names:
        m = _PNUM_RE.match(name)
        if m:
            pnums.append(int(m.group(1)))
    return max(pnums) if pnums else 0


# Convenience: estimate rows per batch from a RAM budget (MB), column count,
# and value size (bytes). Used by streaming readers when you want a dynamic size.
def rows_for_ram(target_mb: int, n_cols: int, bytes_per_val: int = 4, safety: float = 1.5) -> int:
    """Convenience: estimate rows per batch from a RAM budget (MB), column count,
    and value size (bytes). Used by streaming readers when you want a dynamic size.
    """
    return max(10_000, int((target_mb * 1024**2) / (n_cols * bytes_per_val * safety)))
