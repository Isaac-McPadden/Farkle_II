# src/farkle/analysis/frequentist_tiering_report.py
"""Deprecated alias forwarding to :mod:`farkle.analysis.tiering_report`.

This module remains for backward compatibility but should not be referenced by
new code. The analytics pipeline now imports :mod:`farkle.analysis.tiering_report`
directly so skip messaging reflects the real module name.
"""

from __future__ import annotations

import warnings

from farkle.analysis.tiering_report import run

warnings.warn(
    "farkle.analysis.frequentist_tiering_report is deprecated; use"
    " farkle.analysis.tiering_report instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["run"]
