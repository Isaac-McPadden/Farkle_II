# src/farkle/utils/schema_helpers.py
from __future__ import annotations

import re
from typing import Final

import pyarrow as pa

# ---------- static pieces -------------------------------------------------
_BASE_FIELDS: Final[list[tuple[str, pa.DataType]]] = [
    ("winner_seat", pa.string()),  # P{n} label of the winner
    ("winner_strategy", pa.string()),  # strategy string of the winner
    ("seat_ranks", pa.list_(pa.string())),  # ["P7","P1","P3",...]
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
    "hot_dice": pa.int16(),  # counts of hot-dice used this game
    # add/remove seat-level cols here once
}

# ---------- public helpers -----------------------------------------------


def expected_schema_for(n_players: int) -> pa.Schema:
    """Return the canonical schema for *n_players* seats."""
    seat_fields: list[pa.Field] = []
    for i in range(1, n_players + 1):
        for suffix, dtype in _SEAT_TEMPLATE.items():
            seat_fields.append(pa.field(f"P{i}_{suffix}", dtype))
    return pa.schema(_BASE_FIELDS + seat_fields)


_PNUM_RE = re.compile(r"^P(\d+)_")  # Regex for P<X>_


def n_players_from_schema(schema: pa.Schema) -> int:
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