# src/farkle/analysis/stage_state.py
"""Backwards-compatible re-export of shared stage completion helpers."""

from __future__ import annotations

from farkle.utils.stage_completion import (
    CompletionState,
    read_stage_done,
    resolve_stage_state,
    stage_done_path,
    stage_is_up_to_date,
    write_stage_done,
)

__all__ = [
    "CompletionState",
    "read_stage_done",
    "resolve_stage_state",
    "stage_done_path",
    "stage_is_up_to_date",
    "write_stage_done",
]
