# src/farkle/analysis/trueskill.py
"""Run canonical per-root/per-k TrueSkill screening diagnostics."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from farkle.analysis import run_trueskill, stage_logger
from farkle.analysis.trueskill_screening import (
    TRUESKILL_CONDITIONING,
    ScreeningRatingCell,
    build_percentile_contribution,
    trueskill_method_contract,
)
from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import validate_artifact_sidecar

if TYPE_CHECKING:
    from farkle.orchestration.run_contexts import SeedRunContext

LOGGER = logging.getLogger(__name__)


def run(cfg: AppConfig, *, force: bool = False) -> None:
    """Build per-root/per-k ratings and their screening-only contribution."""
    stage_log = stage_logger("trueskill", logger=LOGGER)
    stage_log.start()

    curated_parquet = cfg.curated_parquet
    roots = tuple(int(root) for root in (cfg.sim.seed_list or [cfg.sim.seed]))
    if len(roots) != 1:
        raise ValueError(f"TrueSkill root execution requires exactly one root, found {roots}")
    if not curated_parquet.exists():
        raise FileNotFoundError(
            f"TrueSkill requires canonical concatenated rows: {curated_parquet}"
        )

    out = cfg.trueskill_candidate_contribution_path()
    LOGGER.info(
        "TrueSkill analysis running",
        extra={"stage": "trueskill", "analysis_dir": str(cfg.analysis_dir)},
    )
    run_trueskill.run_trueskill_root(cfg, force=force)
    LOGGER.info(
        "TrueSkill analysis complete",
        extra={"stage": "trueskill", "path": str(out)},
    )


def run_root_pair(
    cfg: AppConfig,
    root_contexts: Sequence[SeedRunContext],
    *,
    force: bool = False,
) -> None:
    """Aggregate complete root-owned rating cells into one pair contribution."""

    configured_roots = tuple(int(root) for root in (cfg.sim.seed_list or [cfg.sim.seed]))
    contexts = tuple(root_contexts)
    context_roots = tuple(int(context.seed) for context in contexts)
    if len(configured_roots) != 2 or len(set(configured_roots)) != 2:
        raise ValueError(
            "TrueSkill root-pair aggregation requires two distinct configured roots; "
            f"found {configured_roots}"
        )
    if len(contexts) != 2 or context_roots != configured_roots:
        raise ValueError(
            "TrueSkill root contexts must match configured root order; "
            f"configured={configured_roots}, contexts={context_roots}"
        )

    required_k = tuple(sorted({int(k) for k in cfg.sim.n_players_list}))
    if not required_k:
        raise ValueError("TrueSkill root-pair aggregation requires configured player counts")
    cells: list[ScreeningRatingCell] = []
    missing: list[str] = []
    for context in contexts:
        for k in required_k:
            path = context.config.trueskill_rating_path(k, root_seed=context.seed)
            if not path.exists():
                missing.append(str(path))
                continue
            validate_artifact_sidecar(
                path,
                expected={
                    "scope": ArtifactScope.BY_K.value,
                    "operation": "sequential_rating",
                    "player_counts": [k],
                    "seed_scope": "single_root",
                    "conditioning": TRUESKILL_CONDITIONING,
                    "method_contract": trueskill_method_contract("sequential_rating"),
                },
            )
            cells.append(ScreeningRatingCell(root_seed=context.seed, k=k, ratings_path=path))
    if missing:
        raise FileNotFoundError(f"TrueSkill root-pair rating cells are missing: {missing}")

    expected = {(root, k) for root in configured_roots for k in required_k}
    observed = {(cell.root_seed, cell.k) for cell in cells}
    if observed != expected or len(cells) != len(expected):
        missing_cells = sorted(expected.difference(observed))
        extra_cells = sorted(observed.difference(expected))
        raise ValueError(
            "TrueSkill root-pair inputs must cover every root/k cell; "
            f"missing={missing_cells}, extra={extra_cells}"
        )
    output = build_percentile_contribution(cfg, cells, force=force)
    LOGGER.info(
        "TrueSkill root-pair contribution complete",
        extra={
            "stage": "trueskill",
            "roots": list(configured_roots),
            "player_counts": list(required_k),
            "path": str(output),
        },
    )


__all__ = ["run", "run_root_pair"]
