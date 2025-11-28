# src/farkle/analysis/__init__.py
"""Lightweight orchestrator for downstream statistical analyses."""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

from farkle.config import AppConfig

LOGGER = logging.getLogger(__name__)


def _optional_import(module: str) -> ModuleType | None:
    """Attempt to import an analytics module while tolerating missing deps.

    Args:
        module: Fully qualified module path to import.

    Returns:
        Imported module object, or ``None`` when the dependency is absent.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tests
        LOGGER.info(
            "Analytics module skipped due to missing dependency",
            extra={
                "stage": "analysis",
                "missing_module": module,
                "missing": str(exc),
            },
        )
        return None


def run_seed_summaries(cfg: AppConfig, *, force: bool = False) -> None:
    """Wrapper around :mod:`farkle.analysis.seed_summaries`."""
    from farkle.analysis import seed_summaries

    seed_summaries.run(cfg, force=force)


def run_meta(cfg: AppConfig, *, force: bool = False) -> None:
    """Wrapper around :mod:`farkle.analysis.meta`."""
    from farkle.analysis import meta

    meta.run(cfg, force=force)


def _skip_message(step: str, reason: str) -> None:
    """Log a standardized skip message for optional analysis stages.

    Args:
        step: Name of the analytics step being skipped.
        reason: Human-readable explanation for the skip condition.
    """
    LOGGER.info(
        "Analytics: skipping %s",
        step,
        extra={"stage": "analysis", "step": step, "reason": reason},
    )


def run_all(cfg: AppConfig) -> None:
    """Run every analytics pass in sequence."""
    LOGGER.info("Analytics: starting all modules", extra={"stage": "analysis"})
    ts_mod = _optional_import("farkle.analysis.trueskill")
    if cfg.analysis.run_trueskill and ts_mod is not None:
        ts_mod.run(cfg)
    else:
        LOGGER.info(
            "Analytics: skipping trueskill",
            extra={
                "stage": "analysis",
                "reason": (
                    "run_trueskill=False" if not cfg.analysis.run_trueskill else "unavailable"
                ),
            },
        )

    h2h_mod = _optional_import("farkle.analysis.head2head")
    if cfg.analysis.run_head2head and h2h_mod is not None:
        h2h_mod.run(cfg)
    else:
        LOGGER.info(
            "Analytics: skipping head-to-head",
            extra={
                "stage": "analysis",
                "reason": (
                    "run_head2head=False" if not cfg.analysis.run_head2head else "unavailable"
                ),
            },
        )

    post_h2h_mod = _optional_import("farkle.analysis.h2h_analysis")
    if cfg.analysis.run_post_h2h_analysis and post_h2h_mod is not None:
        post_h2h_mod.run_post_h2h(cfg)
    else:
        LOGGER.info(
            "Analytics: skipping post head-to-head analysis",
            extra={
                "stage": "analysis",
                "reason": (
                    "run_post_h2h_analysis=False"
                    if not cfg.analysis.run_post_h2h_analysis
                    else "unavailable"
                ),
            },
        )

    hgb_mod = _optional_import("farkle.analysis.hgb_feat")
    if cfg.analysis.run_hgb and hgb_mod is not None:
        hgb_mod.run(cfg)
    else:
        LOGGER.info(
            "Analytics: skipping hist gradient boosting",
            extra={
                "stage": "analysis",
                "reason": "run_hgb=False" if not cfg.analysis.run_hgb else "unavailable",
            },
        )

    freq_mod = _optional_import("farkle.analysis.frequentist_tiering_report")
    if getattr(cfg.analysis, "run_frequentist", False) and freq_mod is not None:
        freq_mod.run(cfg)
    else:
        LOGGER.info(
            "Analytics: skipping tiering report",
            extra={
                "stage": "analysis",
                "reason": (
                    "run_frequentist=False"
                    if not getattr(cfg.analysis, "run_frequentist", False)
                    else "unavailable"
                ),
            },
        )

    run_seed_summaries(cfg)
    run_meta(cfg)
    LOGGER.info("Analytics: all modules finished", extra={"stage": "analysis"})
