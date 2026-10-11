"""Bounded descriptive summaries of the first-release Task 7 artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from farkle.analysis.parameter_responses import PAIRS
from farkle.analysis.seat_behavior import OA, OPPORTUNITY_SCHEMA, Q7, _rows
from farkle.config import AppConfig
from farkle.orchestration.seed_utils import base_results_dir, resolve_results_dir
from farkle.utils.arrow_batches import iter_parquet_tables_by_bytes
from farkle.utils.artifact_contract import validate_artifact_sidecar

NOTES = [
    "Terminal behavior is descriptive association, not state-level risk or a causal mechanism.",
    "Primary behavior conditions on completed games; safety-limit and all-attempt views retain separate support.",
    "Points per actual turn uses total score / total turns (turn weighting); mean game points per turn equally weights player-game exposures.",
    "Score per round is a proxy. The displayed proxy gap is exact mean game return minus mean score per round.",
    "A zero-turn or zero-round safety exposure must have zero score; its game ratio is zero and its zero-denominator count is retained. Completed denominators must be positive.",
    "Rolls, farkles, hot dice and heuristic dice/use counts have actual-turn and exposure denominators; highest turn is an exposure mean. Dice per heuristic use is unavailable without a recorded use.",
    "Opportunity counts each completed game once, including self-play. Mean turn range is in turns; turn-share imbalance is descriptive and does not adjust win rates.",
    "The overall opportunity value is an equal-k mean within each root with every configured k and fixed weights; no completed support at any k makes it unavailable.",
    "Equal-k chance correction is mean absolute probability-point advantage over 1/k, not relative lift.",
    "Per-k specialists, equal-k rankings, Pareto membership and maximin performance are distinct descriptive summaries without new significance claims.",
    "Parameter responses use all-attempt observed win probabilities and identical canonical backgrounds with equal weights. Unsupported rectangles are unestimable, not zero effects.",
    "The parameter exploration covers the ten declared HGB features and only the three prespecified pairs; it is not an exhaustive interaction search or multiplicity-controlled discovery.",
    "Observed nonadditivity is a finite-grid association. Out-of-fold corner means may use different fitted folds and are never subtracted as a within-model interaction.",
    "Held-out MAE/R2 and permutation association importance describe predictive support. Weak predictive scores limit interpretation; new strategies require independent simulation.",
]


def task7_summary(cfg: AppConfig, roots: tuple[int, ...]) -> tuple[dict[str, Any], list[Path]]:
    sources: list[Path] = []
    cells = []
    overall = []
    budget = cfg.resources.stage_batch_bytes["analysis"]
    for root in roots:

        def directory(stage: str, root: int = root) -> Path:
            folder = cfg.root_input_stage_folder(stage)
            if folder is None:
                raise ValueError(f"required Task 7 root stage absent: {stage}")
            base = (
                cfg.analysis_dir
                if len(roots) == 1
                else resolve_results_dir(base_results_dir(cfg), root) / cfg.io.analysis_subdir
            )
            return base / folder

        equal_path = directory("metrics") / "across_k" / "turn_opportunity_equal_k.parquet"
        validate_artifact_sidecar(equal_path, expected={"operation": "turn_opportunity_v1"})
        equal = list(_rows(equal_path, OA, budget))
        ks = sorted(set(cfg.sim.n_players_list))
        if (
            len(equal) != 1
            or equal[0]["root_seed"] != root
            or equal[0]["required_ks"] != ks
            or equal[0]["weights"] != [1 / len(ks)] * len(ks)
        ):
            raise ValueError("Task 7 equal-k opportunity support mismatch")
        overall.append({**equal[0], "artifact": str(equal_path)})
        sources.append(equal_path)
        for k in ks:
            metrics = directory("metrics") / "by_k" / f"{k}p"
            hgb = directory("hgb") / "by_k" / f"{k}p"
            behavior_path = metrics / "seat_behavior_summaries.parquet"
            opportunity_path = metrics / "turn_opportunity.parquet"
            products = {
                "responses": hgb / "parameter_responses.parquet",
                "nonadditivity": hgb / "parameter_nonadditivity.parquet",
                "matched_support": hgb / "parameter_response_support.parquet",
            }
            folds_path = hgb / f"heldout_fold_metrics_{k}p.parquet"
            importance_path = hgb / f"feature_importance_{k}p.parquet"
            for path in (behavior_path, opportunity_path):
                validate_artifact_sidecar(path, expected={"operation": "seat_behavior_v1"})
            for path in products.values():
                validate_artifact_sidecar(path, expected={"operation": "parameter_response_v1"})
            validate_artifact_sidecar(
                folds_path, expected={"operation": "heldout_fold_diagnostics"}
            )
            validate_artifact_sidecar(
                importance_path, expected={"operation": "heldout_permutation_importance"}
            )
            sources.extend(
                [behavior_path, opportunity_path, *products.values(), folds_path, importance_path]
            )
            totals: dict[tuple[str, str], list[Any]] = {}
            for row in _rows(behavior_path, Q7, budget):
                if (row["root_seed"], row["k"]) != (root, k):
                    raise ValueError("Task 7 behavior root/k mismatch")
                if row["seat"] == 0:
                    value = totals.setdefault((row["conditioning"], row["metric"]), [0.0, 0])
                    value[0] += row["numerator"]
                    value[1] += row["denominator"]
            behavior = [
                {
                    "conditioning": c,
                    "metric": m,
                    "numerator": n,
                    "denominator": d,
                    "estimate": n / d if d else None,
                    "status": "ok" if d else "no_denominator",
                }
                for (c, m), (n, d) in sorted(totals.items())
            ]
            opportunity = list(_rows(opportunity_path, OPPORTUNITY_SCHEMA, budget))
            if [(r["root_seed"], r["k"], r["seat"]) for r in opportunity] != [
                (root, k, s) for s in range(k + 1)
            ]:
                raise ValueError("Task 7 opportunity root/k/seat mismatch")
            support_counts = {}
            for name, path in products.items():
                counts: Counter[str] = Counter()
                for _, _, table in iter_parquet_tables_by_bytes(
                    path,
                    columns=["root_seed", "k", "status"],
                    max_batch_bytes=budget // 4,
                    max_batch_rows=max(1, budget // 8192),
                ):
                    for row in table.to_pylist():
                        if (row["root_seed"], row["k"]) != (root, k):
                            raise ValueError("Task 7 parameter root/k mismatch")
                        counts[row["status"]] += 1
                support_counts[name] = {
                    "rows": sum(counts.values()),
                    "status_counts": dict(sorted(counts.items())),
                    "artifact": str(path),
                }
            predictive = pq.read_table(folds_path).to_pylist()
            importance = pq.read_table(importance_path).to_pylist()
            if any((r["root_seed"], r["players"]) != (root, k) for r in [*predictive, *importance]):
                raise ValueError("Task 7 held-out root/k mismatch")
            cells.append(
                {
                    "root_seed": root,
                    "k": k,
                    "behavior": behavior,
                    "behavior_weighting": "sum_numerators_and_denominators_over_all_strategy_exposures",
                    "behavior_artifact": str(behavior_path),
                    "opportunity": opportunity,
                    "opportunity_artifact": str(opportunity_path),
                    "parameters": support_counts,
                    "heldout_folds": predictive,
                    "heldout_fold_metrics_artifact": str(folds_path),
                    "permutation_association_importance": importance,
                    "importance_artifact": str(importance_path),
                }
            )
    return {
        "method_version": 1,
        "notes": NOTES,
        "prespecified_pairs": [list(p) for p in PAIRS],
        "overall_opportunity": overall,
        "root_k_cells": cells,
    }, sources


def task7_markdown(summary: dict[str, Any]) -> list[str]:
    lines = ["", "## Behavior, opportunity and parameter associations", ""]
    lines += [f"- {note}" for note in summary["notes"]]
    lines += ["", "### Equal-k opportunity by root", ""]
    for row in summary["overall_opportunity"]:
        lines.append(
            f"- Root {row['root_seed']}: mean turn range `{row['mean_turn_range_equal_k']}`; status `{row['status']}`; k `{row['required_ks']}`, weights `{row['weights']}`. [Exact opportunity summary]({Path(row['artifact']).as_posix()})."
        )
    for cell in summary["root_k_cells"]:
        lines += [
            "",
            f"### Root {cell['root_seed']}, k={cell['k']}",
            "",
            "| Completed behavior | Estimate | Denominator |",
            "| --- | ---: | ---: |",
        ]
        for row in cell["behavior"]:
            if row["conditioning"] == "completed" and row["metric"] in (
                "points_per_turn",
                "mean_game_points_per_turn",
                "mean_game_points_per_round",
                "round_proxy_gap",
                "turn_round_mismatch_rate",
                "mean_turns",
                "mean_rounds",
                "mean_highest_turn",
            ):
                lines.append(f"| {row['metric']} | {row['estimate']} | {row['denominator']} |")
        lines += [
            "",
            f"[Seat/strategy behavior, all conditioning strata]({Path(cell['behavior_artifact']).as_posix()}).",
            f"[Per-seat turn opportunity]({Path(cell['opportunity_artifact']).as_posix()}): `{cell['opportunity'][0]}`.",
            "",
        ]
        for name, product in cell["parameters"].items():
            lines.append(
                f"- [{name}]({Path(product['artifact']).as_posix()}): {product['rows']} rows; support statuses `{product['status_counts']}`."
            )
        lines += [
            f"- [Held-out fold metrics]({Path(cell['heldout_fold_metrics_artifact']).as_posix()}): `{cell['heldout_folds']}`.",
            f"- [Permutation association importance]({Path(cell['importance_artifact']).as_posix()}): `{cell['permutation_association_importance']}`.",
        ]
    return lines
