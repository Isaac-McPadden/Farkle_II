"""Exact ordered dice-outcome enumeration against the production scorer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa

from farkle.config import AppConfig, ArtifactScope
from farkle.game.scoring import score_roll_cached
from farkle.utils.artifact_contract import make_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done

_FACES: Final = range(1, 7)
_SELECTION_RULE: Final = "production_max_immediate_score_v1"


@dataclass(frozen=True)
class ExactRollArtifacts:
    """Paths published by exact ordered-roll enumeration."""

    distribution: Path
    summary: Path

    @property
    def all_paths(self) -> tuple[Path, Path]:
        return self.distribution, self.summary


def _histogram_quantile(score_counts: Counter[int], total: int, probability: float) -> float:
    """Return the exact-distribution linear quantile from integer score counts."""

    rank = (total - 1) * probability
    lower_rank = int(rank)
    upper_rank = min(lower_rank + 1, total - 1)

    def _value_at(target: int) -> int:
        cumulative = 0
        for score, count in sorted(score_counts.items()):
            cumulative += count
            if cumulative > target:
                return score
        raise RuntimeError("ordered-roll score histogram is incomplete")

    lower = _value_at(lower_rank)
    upper = _value_at(upper_rank)
    return lower + (rank - lower_rank) * (upper - lower)


def enumerate_ordered_roll_outcomes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate all ``6**d`` ordered outcomes for every dice count ``d=1..6``."""

    distribution_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for dice_count in range(1, 7):
        cells: Counter[tuple[int, int, bool, bool]] = Counter()
        score_counts: Counter[int] = Counter()
        total_score = 0
        total_scoring_dice = 0
        farkles = 0
        hot_dice = 0
        for outcome in product(_FACES, repeat=dice_count):
            score, used, *_ = score_roll_cached(outcome)
            is_farkle = score == 0
            is_hot_dice = score > 0 and used == dice_count
            cells[(score, used, is_farkle, is_hot_dice)] += 1
            score_counts[score] += 1
            total_score += score
            total_scoring_dice += used
            farkles += int(is_farkle)
            hot_dice += int(is_hot_dice)

        ordered_outcomes = 6**dice_count
        for (score, used, is_farkle, is_hot_dice), count in sorted(cells.items()):
            distribution_rows.append(
                {
                    "dice_count": dice_count,
                    "scoring_selection_rule": _SELECTION_RULE,
                    "max_immediate_score": score,
                    "scoring_dice": used,
                    "is_farkle": is_farkle,
                    "is_hot_dice": is_hot_dice,
                    "ordered_outcome_count": count,
                    "ordered_outcome_probability": count / ordered_outcomes,
                }
            )
        summary_rows.append(
            {
                "dice_count": dice_count,
                "scoring_selection_rule": _SELECTION_RULE,
                "ordered_outcomes": ordered_outcomes,
                "farkle_count": farkles,
                "farkle_probability": farkles / ordered_outcomes,
                "expected_max_immediate_score": total_score / ordered_outcomes,
                "p10_max_immediate_score": _histogram_quantile(score_counts, ordered_outcomes, 0.1),
                "median_max_immediate_score": _histogram_quantile(
                    score_counts, ordered_outcomes, 0.5
                ),
                "p90_max_immediate_score": _histogram_quantile(score_counts, ordered_outcomes, 0.9),
                "hot_dice_probability": hot_dice / ordered_outcomes,
                "expected_scoring_dice": total_scoring_dice / ordered_outcomes,
            }
        )
    return pd.DataFrame(distribution_rows), pd.DataFrame(summary_rows)


def build_exact_roll_enumeration(cfg: AppConfig, *, force: bool = False) -> ExactRollArtifacts:
    """Publish exact ordered-roll distribution and summary diagnostics."""

    artifacts = ExactRollArtifacts(
        distribution=cfg.exact_roll_distribution_path(),
        summary=cfg.exact_roll_summary_path(),
    )
    done = stage_done_path(cfg.game_stats_stage_dir, "exact_roll_enumeration")
    if not force and stage_is_up_to_date(
        done,
        inputs=[],
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="game_stats",
        sidecar_artifacts=list(artifacts.all_paths),
    ):
        return artifacts

    distribution, summary = enumerate_ordered_roll_outcomes()
    frames = (
        (
            distribution,
            artifacts.distribution,
            [
                "dice_count",
                "scoring_selection_rule",
                "max_immediate_score",
                "scoring_dice",
                "is_farkle",
                "is_hot_dice",
            ],
        ),
        (summary, artifacts.summary, ["dice_count", "scoring_selection_rule"]),
    )
    for frame, path, grouping_keys in frames:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        sidecar = make_artifact_sidecar(
            cfg,
            path,
            producer="roll_enumeration",
            scope=ArtifactScope.DIAGNOSTICS,
            source_scope=ArtifactScope.DIAGNOSTICS,
            operation="exact_enumeration",
            weighted_quantity="ordered_outcome_probability",
            support_count_role="ordered_die_outcomes",
            uncertainty_method="none_exact_finite_enumeration",
            replication_unit="ordered_roll_outcome",
            conditioning=_SELECTION_RULE,
            consistency_columns=table.schema.names,
            grouping_keys=grouping_keys,
            missing_cell_policy="not_applicable",
            seed_scope="not_applicable",
        )
        write_parquet_artifact_atomic(table, path, sidecar=sidecar, codec=cfg.parquet_codec)

    write_stage_done(
        done,
        inputs=[],
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="game_stats",
        sidecar_artifacts=list(artifacts.all_paths),
    )
    return artifacts


__all__ = [
    "ExactRollArtifacts",
    "build_exact_roll_enumeration",
    "enumerate_ordered_roll_outcomes",
]
