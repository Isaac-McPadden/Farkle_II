"""Shared fixtures and helpers for analysis tests."""

from __future__ import annotations

# The analysis modules now provide internal fallbacks when optional dependencies
# such as ``pydantic`` are unavailable. The legacy tests used to skip entirely
# when these packages were missing, but the modern pipeline should operate (and
# therefore be exercised) without them. Keeping this file allows future shared
# fixtures to be added without reintroducing the hard dependency checks.
