"""Compatibility wrapper for the Farkle CLI.

Exports the :func:`main` entry point from :mod:`farkle.cli.farkle_cli` so
``pyproject.toml`` can reference ``farkle.cli.main:main``.
"""

from __future__ import annotations

from .farkle_cli import load_config, main

__all__ = ["load_config", "main"]
