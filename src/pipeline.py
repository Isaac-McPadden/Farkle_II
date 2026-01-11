"""Compatibility shim for the relocated orchestration pipeline module."""

from __future__ import annotations

from farkle.orchestration.pipeline import *  # noqa: F403

from farkle.orchestration.pipeline import __all__  # noqa: F401
