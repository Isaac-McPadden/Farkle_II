"""Shared publication helpers for first-release RNG evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import make_artifact_sidecar, sha256_file, sidecar_path
from farkle.utils.artifacts import write_json_artifact_atomic, write_parquet_artifact_atomic
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done

CONTRACT_ID = "first_release_task0_v1"
BLIND_SPOTS = [
    "Within-batch dependence preserving batch totals is invisible.",
    "Unselected strategies, actual seats, higher lags and rare tails are not covered.",
    "Root agreement does not establish independent realizations.",
    "Descriptive thresholds have no calibrated family-wise error guarantee.",
]


def metadata(cfg: AppConfig, path: Path, sources: list[Path], columns: list[str]):
    return make_artifact_sidecar(
        cfg,
        path,
        producer="rng_diagnostics",
        scope=ArtifactScope.DIAGNOSTICS,
        source_scope=ArtifactScope.DIAGNOSTICS,
        operation="first_release_rng_evidence",
        source_artifacts=sources,
        consistency_columns=columns,
        player_counts=sorted(set(cfg.sim.n_players_list)),
        required_player_counts=sorted(set(cfg.sim.n_players_list)),
        conditioning="validation_or_descriptive_monitoring_not_inferential_observations",
        replication_unit="deterministic_shuffle_batch",
        support_count_role="explicit_per_check_support",
        uncertainty_method="descriptive_only",
        method_contract={
            "kind": "operation",
            "procedure": "first_release_rng_evidence",
            "parameters": {
                "contract_id": CONTRACT_ID,
                "rng_preflight_method_version": 1,
                "rng_monitor_method_version": 1,
                "multiplicity": "descriptive_no_fwer",
            },
        },
    )


def publish(cfg: AppConfig, path: Path, payload: dict[str, Any], sources: list[Path]) -> None:
    write_json_artifact_atomic(payload, path, sidecar=metadata(cfg, path, sources, list(payload)))


def stamp(path: Path) -> Path:
    return stage_done_path(path.parent, path.stem)


def reusable(cfg: AppConfig, path: Path, sources: list[Path]) -> bool:
    return stage_is_up_to_date(
        stamp(path), sources, [path], cfg=cfg, stage="rng_diagnostics", sidecar_artifacts=[path]
    )


def finish(
    cfg: AppConfig, path: Path, payload: dict[str, Any] | pa.Table, sources: list[Path]
) -> None:
    if isinstance(payload, pa.Table):
        write_parquet_artifact_atomic(
            payload, path, sidecar=metadata(cfg, path, sources, payload.schema.names)
        )
    else:
        publish(cfg, path, payload, sources)
    write_stage_done(
        stamp(path),
        inputs=sources,
        outputs=[path],
        cfg=cfg,
        stage="rng_diagnostics",
        sidecar_artifacts=[path],
    )


def inventory(paths: list[Path], root: Path) -> list[dict[str, str]]:
    return [
        {
            "path": p.relative_to(root).as_posix(),
            "sha256": sha256_file(p),
            "sidecar_sha256": sha256_file(sidecar_path(p)),
        }
        for p in paths
    ]


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
