from __future__ import annotations

import re
from typing import Final

import pyarrow as pa

__all__ = [
    "expected_schema_for",
    "n_players_from_schema",
    "rows_for_ram",
]

_BASE_FIELDS: Final[list[tuple[str, pa.DataType]]] = [
    ("winner_seat", pa.string()),
    ("winner_strategy", pa.string()),
    ("seat_ranks", pa.list_(pa.string())),
    ("winning_score", pa.int32()),
    ("n_rounds", pa.int16()),
]

_SEAT_TEMPLATE: Final[dict[str, pa.DataType]] = {
    "score": pa.int32(),
    "farkles": pa.int16(),
    "rolls": pa.int16(),
    "highest_turn": pa.int16(),
    "strategy": pa.string(),
    "rank": pa.int8(),
    "loss_margin": pa.int32(),
    "smart_five_uses": pa.int16(),
    "n_smart_five_dice": pa.int16(),
    "smart_one_uses": pa.int16(),
    "n_smart_one_dice": pa.int16(),
    "hot_dice": pa.int16(),
}


def expected_schema_for(n_players: int) -> pa.Schema:
    seat_fields = [
        pa.field(f"P{i}_{suffix}", dtype)
        for i in range(1, n_players + 1)
        for suffix, dtype in _SEAT_TEMPLATE.items()
    ]
    return pa.schema(_BASE_FIELDS + seat_fields)


_PNUM_RE = re.compile(r"^P(\d+)_")


def n_players_from_schema(schema: pa.Schema) -> int:
    seats: list[int] = []
    for name in schema.names:
        match = _PNUM_RE.match(name)
        if match:
            seats.append(int(match.group(1)))
    return max(seats) if seats else 0


def rows_for_ram(target_mb: int, n_cols: int, bytes_per_val: int = 4, safety: float = 1.5) -> int:
    return max(10_000, int((target_mb * 1024**2) / (n_cols * bytes_per_val * safety)))
