# src/farkle/analysis/__init__.py
"""Lightweight orchestrator for downstream statistical analyses."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from types import ModuleType

from farkle.config import AppConfig

LOGGER = logging.getLogger(__name__)


def _optional_import(module: str, attribute: str | None = None) -> ModuleType | Callable[..., object] | None:
    """Import *module* (and optional *attribute*) if available, logging when missing."""

    try:
        mod = importlib.import_module(module)
        return getattr(mod, attribute) if attribute is not None else mod
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tests
        LOGGER.info(
            "Analytics module skipped due to missing dependency",
            extra={"stage": "analysis", "module": module, "missing": str(exc)},
        )
        return None


def run_seed_summaries(cfg: AppConfig, *, force: bool = False) -> None:
    """Wrapper around :mod:`farkle.analysis.seed_summaries`."""
    from farkle.analysis import seed_summaries

    seed_summaries.run(cfg, force=force)


def run_meta(cfg: AppConfig, *, force: bool = False) -> None:
    """Wrapper around :mod:`farkle.analysis.meta`."""

    meta_run = _optional_import("farkle.analysis.meta", "run")
    if meta_run is None:
        _log_skip("meta", reason="unavailable")
        return

    meta_run(cfg, force=force)


def _log_skip(label: str, *, reason: str) -> None:
    LOGGER.info(
        "Analytics: skipping %s",
        label,
        extra={"stage": "analysis", "module": label, "reason": reason},
    )


def run_all(cfg: AppConfig) -> None:
    """Run the analytics pipeline in a deterministic, resumable sequence."""
    LOGGER.info("Analytics: pipeline starting", extra={"stage": "analysis"})

    # Always refresh seed-level summaries first so downstream steps have inputs.
    run_seed_summaries(cfg)
    run_meta(cfg)

    ts_mod = _optional_import("farkle.analysis.run_trueskill")
    if cfg.analysis.run_trueskill and ts_mod is not None:
        ts_mod.run_trueskill_all_seeds(cfg)
    else:
        _log_skip(
            "trueskill",
            reason=(
                "run_trueskill=False"
                if not cfg.analysis.run_trueskill
                else "unavailable"
            ),
        )

    h2h_mod = _optional_import("farkle.analysis.head2head")
    if cfg.analysis.run_head2head and h2h_mod is not None:
        h2h_mod.run(cfg)
    else:
        _log_skip(
            "head-to-head",
            reason=(
                "run_head2head=False"
                if not cfg.analysis.run_head2head
                else "unavailable"
            ),
        )

    post_h2h_mod = _optional_import("farkle.analysis.h2h_analysis")
    if cfg.analysis.run_post_h2h_analysis and post_h2h_mod is not None:
        post_h2h_mod.run_post_h2h(cfg)
    else:
        _log_skip(
            "post head-to-head analysis",
            reason=(
                "run_post_h2h_analysis=False"
                if not cfg.analysis.run_post_h2h_analysis
                else "unavailable"
            ),
        )

    hgb_mod = _optional_import("farkle.analysis.hgb_feat")
    if cfg.analysis.run_hgb and hgb_mod is not None:
        hgb_mod.run(cfg)
    else:
        _log_skip(
            "hist gradient boosting",
            reason=(
                "run_hgb=False" if not cfg.analysis.run_hgb else "unavailable"
            ),
        )

    freq_tiering_mod = _optional_import("farkle.analysis.frequentist_tiering_report")
    if getattr(cfg.analysis, "run_frequentist", False) and freq_tiering_mod is not None:
        freq_tiering_mod.run(cfg)
    else:
        _log_skip(
            "frequentist tiering",
            reason=(
                "run_frequentist=False"
                if not getattr(cfg.analysis, "run_frequentist", False)
                else "unavailable"
            ),
        )

    agreement_mod = _optional_import("farkle.analysis.agreement")
    if getattr(cfg.analysis, "run_agreement", False) and agreement_mod is not None:
        agreement_mod.run(cfg)
    else:
        _log_skip(
            "agreement analysis",
            reason=(
                "run_agreement=False"
                if not getattr(cfg.analysis, "run_agreement", False)
                else "unavailable"
            ),
        )

    report_mod = _optional_import("farkle.analysis.reporting")
    if getattr(cfg.analysis, "run_report", True) and report_mod is not None:
        report_mod.run_report(cfg)
    else:
        _log_skip(
            "report",
            reason=(
                "run_report=False"
                if not getattr(cfg.analysis, "run_report", True)
                else "unavailable"
            ),
        )

    LOGGER.info("Analytics: pipeline complete", extra={"stage": "analysis"})
