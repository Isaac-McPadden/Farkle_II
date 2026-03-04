"""Backwards-compatible re-export of shared stage completion helpers."""

from __future__ import annotations

from farkle.utils.stage_completion import (
    read_stage_done,
    stage_done_path,
    stage_is_up_to_date,
    write_stage_done,
)

__all__ = ["read_stage_done", "stage_done_path", "stage_is_up_to_date", "write_stage_done"]
