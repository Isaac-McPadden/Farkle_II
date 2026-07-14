"""Inventory ignored retired artifacts without deleting or reading them."""

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass
from pathlib import Path

from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import write_json_artifact_atomic


@dataclass(frozen=True)
class RetiredArtifactRule:
    pattern: str
    replacement: str


_RULES: tuple[RetiredArtifactRule, ...] = (
    RetiredArtifactRule("ratings_k_weighted.parquet", "canonical root/k TrueSkill ratings"),
    RetiredArtifactRule("frequentist_*.parquet", "descriptive screening artifacts"),
    RetiredArtifactRule("bonferroni_*.parquet", "canonical h2h_2p inference artifacts"),
    RetiredArtifactRule("bonferroni_*.json", "canonical h2h_2p inference artifacts"),
    RetiredArtifactRule("h2h_s_tiers.json", "dominance fronts and cycle evidence"),
    RetiredArtifactRule("*seed_summary*.parquet", "root-specific canonical estimates"),
    RetiredArtifactRule("*variance*.parquet", "cross_seed stability diagnostics"),
    RetiredArtifactRule("*meta*.parquet", "within-k root combination and stability"),
    RetiredArtifactRule("tiers.json", "descriptive screening or dominance fronts"),
    RetiredArtifactRule("agreement_*p.json", "selection-conditioned method agreement"),
    RetiredArtifactRule("agreement_across_k.json", "selection-conditioned method agreement"),
    RetiredArtifactRule("metrics_weighted.parquet", "canonical across_k performance"),
    RetiredArtifactRule("*isolated_metrics.parquet", "canonical all-player batch metrics"),
    RetiredArtifactRule("*seed_symmetry*.parquet", "canonical seat diagnostics"),
    RetiredArtifactRule("interseed_*.json", "canonical cross_seed artifacts"),
    RetiredArtifactRule("interseed_*.parquet", "canonical cross_seed artifacts"),
    RetiredArtifactRule("s_tier_*.parquet", "dominance fronts and cycle evidence"),
)


def _matching_rule(filename: str) -> RetiredArtifactRule | None:
    return next(
        (rule for rule in _RULES if fnmatch.fnmatchcase(filename, rule.pattern)),
        None,
    )


def inventory(cfg: AppConfig) -> list[dict[str, str | int]]:
    """Return a stable inventory of retired files under the configured results root."""

    root = cfg.results_root
    report_path = cfg.migration_report_path().resolve()
    rows: list[dict[str, str | int]] = []
    if not root.exists():
        return rows
    for directory, folder_names, filenames in os.walk(root):
        folder_names.sort()
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.resolve() == report_path:
                continue
            rule = _matching_rule(filename)
            if rule is None:
                continue
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "existing_name": filename,
                    "replacement": rule.replacement,
                    "disposition": "ignored_not_deleted",
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


def run(cfg: AppConfig) -> Path:
    """Write the canonical non-destructive migration report and sidecar."""

    output = cfg.migration_report_path()
    ignored = inventory(cfg)
    payload = {
        "migration_report_version": 2,
        "scan_root": str(cfg.results_root),
        "ignored_artifact_count": len(ignored),
        "ignored_artifacts": ignored,
        "artifacts_deleted": False,
        "current_code_reads_ignored_artifacts": False,
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing == payload:
            validate_artifact_sidecar(
                output,
                expected={"operation": "inventory_ignored_on_disk_artifacts"},
            )
            return output
    roots = sorted({int(root) for root in (cfg.sim.seed_list or [cfg.sim.seed])})
    sidecar = make_artifact_sidecar(
        cfg,
        output,
        producer="migration_audit",
        scope=ArtifactScope.DIAGNOSTICS,
        source_scope=ArtifactScope.DIAGNOSTICS,
        operation="inventory_ignored_on_disk_artifacts",
        conditioning="filesystem_inventory_only",
        seed_scope="both_roots_combined" if len(roots) == 2 else "single_root",
        player_counts=sorted({int(k) for k in cfg.sim.n_players_list}),
        required_player_counts=sorted({int(k) for k in cfg.sim.n_players_list}),
        missing_cell_policy="not_applicable",
    )
    write_json_artifact_atomic(payload, output, sidecar=sidecar)
    return output


__all__ = ["RetiredArtifactRule", "inventory", "run"]
