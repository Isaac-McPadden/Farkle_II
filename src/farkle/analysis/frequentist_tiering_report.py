# src/farkle/analysis/frequentist_tiering_report.py
"""Alias module wiring tiering report into the frequentist stage.

Exports the :func:`run` entry point expected by the pipeline so downstream
invocations can reuse the shared tiering report implementation.
"""

from __future__ import annotations

from farkle.analysis.tiering_report import run

__all__ = ["run"]
