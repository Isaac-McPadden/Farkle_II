"""Required authenticated preflight and batch monitoring; optional exhaustive lags."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis import rng_diagnostics, rng_preflight
from farkle.analysis.batch_monitor import METHOD_ID, diagnose_counts
from farkle.analysis.outcome_distributions import _COUNTS, outcome_support_schema
from farkle.analysis.outcome_precision import _inputs, _tables, _validate_support
from farkle.analysis.outcome_publication import (
    _checked,
    _resolve_design,
    validate_outcome_checkpoints,
)
from farkle.analysis.rng_evidence import (
    BLIND_SPOTS,
    CONTRACT_ID,
    inventory,
    metadata,
    publish,
    read,
    reusable,
)
from farkle.config import AppConfig
from farkle.utils.artifact_contract import sha256_file, validate_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.authenticated_contract import AuthenticatedContractError, identity_sha256
from farkle.utils.parallel import process_map, resolve_mp_context, resolve_stage_parallel_policy
from farkle.utils.stage_completion import (
    CompletionState,
    resolve_stage_state,
    stage_done_path,
    write_stage_done,
)

LOGGER = logging.getLogger(__name__)


def freshness_key(cfg: AppConfig, exhaustive: bool, lags: Sequence[int] | None) -> dict[str, Any]:
    return {
        **cfg.freshness_key(),
        "exhaustive_enabled": exhaustive,
        "lags": list(cfg.analysis.rng_diagnostic_lags if lags is None else lags),
    }


def monitor_schema() -> pa.Schema:
    return pa.schema(
        cast(
            list[pa.Field],
            [
                pa.field("root_seed", pa.int64(), nullable=False),
                pa.field("k", pa.int16(), nullable=False),
                pa.field("strategy", pa.int32(), nullable=False),
                pa.field("seat", pa.int16(), nullable=False),
                pa.field("check_id", pa.string(), nullable=False),
                pa.field("batch_start", pa.int32()),
                pa.field("batch_stop_exclusive", pa.int32()),
                pa.field("support", pa.int64(), nullable=False),
                pa.field("statistic", pa.float64()),
                pa.field("reference_low", pa.float64()),
                pa.field("reference_high", pa.float64()),
                pa.field("effect_threshold", pa.float64()),
                pa.field("result", pa.string(), nullable=False),
                pa.field("reason", pa.string(), nullable=False),
                pa.field("method_id", pa.string(), nullable=False),
            ],
        )
    )


def sentinel_ids(roster: Sequence[int]) -> list[int]:
    """Select before outcomes, using the frozen Task 5B rule."""
    if list(roster) != sorted(set(roster)) or not roster:
        raise ValueError("monitor requires a nonempty canonical unique roster")
    return (
        list(roster)
        if len(roster) <= 16
        else [roster[i * (len(roster) - 1) // 15] for i in range(16)]
    )


def _monitor_k(cfg: AppConfig, k: int, proof: Path, *, force: bool) -> Path:
    path = cfg.rng_input_path(f"monitor_units/k{k}.parquet")
    sources = [*_inputs(cfg, k), proof]
    if not validate_outcome_checkpoints(cfg.outcome_checkpoint_inventory_path(k)):
        raise ValueError("production monitor requires complete authenticated outcome checkpoints")
    if not force and reusable(cfg, path, sources):
        return path
    design = _resolve_design(cfg, k)
    budget = int(cfg.resources.stage_batch_bytes["analysis"])
    _validate_support(cfg, design, budget)
    selected = sentinel_ids([s for s, _ in design.roster])
    if len(selected) * design.batches * len(_COUNTS) * 8 > budget // 4:
        raise MemoryError("selected monitor vectors exceed the configured byte allowance")
    values = {sid: np.zeros((design.batches, len(_COUNTS)), dtype=np.int64) for sid in selected}
    for table in _tables(
        cfg.outcome_batch_stats_path(k), outcome_support_schema(batch=True), budget
    ):
        for row in table.to_pylist():
            sid = row["strategy"]
            if sid in values:
                batch = row["deterministic_batch_id"]
                for j, count in enumerate(_COUNTS):
                    values[sid][batch, j] = _checked(int(values[sid][batch, j]) + row[count])
    rows = []
    for sid in selected:
        counts = {name: values[sid][:, i] for i, name in enumerate(_COUNTS)}
        for metric, checks in diagnose_counts(
            np.arange(design.batches), counts, expected_batches=design.batches
        ).items():
            for check in checks:
                rows.append(
                    {
                        "root_seed": cfg.sim.seed,
                        "k": k,
                        "strategy": sid,
                        "seat": 0,
                        "check_id": f"{metric}/{check.check_id}",
                        "batch_start": 0,
                        "batch_stop_exclusive": design.batches,
                        "support": check.support,
                        "statistic": check.statistic,
                        "reference_low": None,
                        "reference_high": None,
                        "effect_threshold": check.effect_threshold,
                        "result": check.result,
                        "reason": check.reason,
                        "method_id": check.method_id,
                    }
                )
    table = pa.Table.from_pylist(rows, schema=monitor_schema())
    write_parquet_artifact_atomic(
        table, path, sidecar=metadata(cfg, path, sources, table.schema.names)
    )
    write_stage_done(
        stage_done_path(path.parent, path.stem),
        inputs=sources,
        outputs=[path],
        cfg=cfg,
        stage="rng_diagnostics",
        sidecar_artifacts=[path],
    )
    return path


def _monitor_task(task: tuple[AppConfig, int, Path, bool]) -> tuple[int, Path]:
    cfg, k, proof, force = task
    return k, _monitor_k(cfg, k, proof, force=force)


def validate_preflight_inventory(publication: Path) -> bool:
    """Deep audit of the exact retained proof inventory, with no glob discovery."""
    try:
        from farkle.utils.artifact_contract import sidecar_path

        validate_artifact_sidecar(publication)
        payload = read(publication)
        planned = read(publication.parent / "plan.json")["required_source_paths"]
        paths: set[str] = set()
        for entry in payload["sources"]:
            relative = Path(entry["path"])
            if (
                relative.is_absolute()
                or relative.drive
                or ".." in relative.parts
                or entry["path"] in paths
            ):
                return False
            paths.add(entry["path"])
            path = publication.parent / relative
            validate_artifact_sidecar(path)
            if (
                sha256_file(path) != entry["sha256"]
                or sha256_file(sidecar_path(path)) != entry["sidecar_sha256"]
            ):
                return False
        return (
            bool(paths)
            and paths == set(planned)
            and len(planned) == len(paths)
            and payload["status"] == "complete_valid"
        )
    except (OSError, ValueError, KeyError, TypeError, AuthenticatedContractError):
        return False


def _envelope(cfg: AppConfig, proof: Path, result: dict[str, Any]) -> dict[str, Any]:
    plan = read(proof.parent / "plan.json")
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "root_seed": cfg.sim.seed,
        "required_ks": sorted(set(cfg.sim.n_players_list)),
        "method_id": METHOD_ID,
        "code_identity": result["code_identity"],
        "config_identity": cfg.stage_config_sha("rng_diagnostics"),
        "rng_identity": identity_sha256(plan["rng"]),
        "workload_identity": result["workload_identity"],
        "environment": result["environment"],
        "source_artifacts": inventory([proof], cfg.rng_input_path(".")),
        "applicability": {
            "schema_version": 1,
            "covered_ks": result["covered_ks"],
            "covered_workers": result["covered_workers"],
            "capability_method": METHOD_ID,
            "target_root": cfg.sim.seed,
            "validation_roots": list(rng_preflight.ROOTS),
            "production_observations": False,
            "target_profile_sha256": plan.get("target_game_profile_sha256"),
            "profile_scope": "declared_10000_200_matrix_and_named_supplemental_fixtures_only",
            "blind_spots": BLIND_SPOTS,
            "reuse": "current_code_config_environment_authenticated_units",
        },
    }


def run(
    cfg: AppConfig,
    *,
    lags: Sequence[int] | None = None,
    force: bool = False,
    exhaustive: bool | None = None,
) -> None:
    """Publish required evidence even when exhaustive lags are deliberately omitted."""
    enabled = not cfg.analysis.disable_rng_diagnostics if exhaustive is None else exhaustive
    if not set(cfg.sim.n_players_list).issubset(rng_preflight.KS):
        raise ValueError("required k lies outside the first-release preflight recipe")
    stamp = stage_done_path(cfg.rng_stage_dir, "rng_diagnostics")
    try:
        proof, result = rng_preflight.run(cfg, force=force)
        if not validate_preflight_inventory(proof):
            raise ValueError("preflight inventory is incomplete or corrupt")
    except Exception:
        stamp.unlink(missing_ok=True)
        raise
    pe, mc, mr = (
        cfg.rng_preflight_evidence_path(),
        cfg.rng_monitor_coverage_path(),
        cfg.rng_production_monitor_path(),
    )
    inputs = [
        proof,
        *(proof.parent / entry["path"] for entry in result["sources"]),
        *(path for k in sorted(set(cfg.sim.n_players_list)) for path in _inputs(cfg, k)),
    ]
    unit_paths = [
        cfg.rng_input_path(f"monitor_units/k{k}.parquet")
        for k in sorted(set(cfg.sim.n_players_list))
    ]
    proof_paths = [proof, *(proof.parent / entry["path"] for entry in result["sources"])]
    outputs = [pe, mc, mr, *unit_paths, *proof_paths]
    exhaustive_outputs: list[Path] = []
    key = freshness_key(cfg, enabled, lags)
    if enabled:
        exhaustive_outputs.extend(
            cfg.rng_input_path(name)
            for name in ("rng_diagnostics.parquet", "rng_diagnostics_summary.json")
        )
        outputs.extend(exhaustive_outputs)
    state = resolve_stage_state(
        stamp,
        inputs,
        outputs,
        cfg=cfg,
        stage="rng_diagnostics",
        freshness_key=key,
        sidecar_artifacts=outputs,
    )
    scope_matches = (
        state in {CompletionState.COMPLETE_VALID, CompletionState.BLOCKED_BY_CAP}
        and mc.exists()
        and read(mc).get("exhaustive_enabled") == enabled
        and read(mc).get("requested_lags") == key["lags"]
    )
    if (
        not force
        and scope_matches
        and state in {CompletionState.COMPLETE_VALID, CompletionState.BLOCKED_BY_CAP}
    ):
        return
    # Stale parent completion cannot survive a failed required publication.
    stamp.unlink(missing_ok=True)
    policy = resolve_stage_parallel_policy("rng_diagnostics", cfg.analysis, resources=cfg.resources)
    jobs = min(len(unit_paths), policy.process_workers)
    units = [
        path
        for _, path in sorted(
            process_map(
                _monitor_task,
                ((cfg, k, proof, force) for k in sorted(set(cfg.sim.n_players_list))),
                n_jobs=jobs,
                mp_context=resolve_mp_context(cfg.analysis.mp_start_method),
                window=jobs,
            )
        )
    ]
    tables = [pq.read_table(path) for path in units]  # <=16*12 rows per required k
    rows = pa.concat_tables(tables)
    write_parquet_artifact_atomic(rows, mr, sidecar=metadata(cfg, mr, units, rows.schema.names))
    counts = Counter(rows["result"].to_pylist())
    status = (
        "investigate"
        if counts["investigate"]
        else "insufficient_evidence" if counts["insufficient_evidence"] else "complete_valid"
    )
    if counts["investigate"]:
        LOGGER.warning(
            "Batch monitor requests investigation: %d descriptive warnings; sampling targets unchanged",
            counts["investigate"],
        )
    exhaustive_status, reason = (
        "omitted_by_design",
        "analysis.disable_rng_diagnostics or explicit exhaustive override",
    )
    if enabled:
        summary = cfg.rng_input_path("rng_diagnostics_summary.json")
        # Explicit call-level lag changes are execution requests outside the
        # public config. Do not reuse a child computed for another lag set.
        lag_change = False
        if summary.exists():
            try:
                lag_change = read(summary).get("normalized_lags") != key["lags"]
            except (ValueError, OSError):
                lag_change = True
        rng_diagnostics.run(
            cfg, lags=lags, force=force or lag_change, _completion_name="rng_exhaustive"
        )
        child = resolve_stage_state(
            stage_done_path(cfg.rng_stage_dir, "rng_exhaustive"),
            [],
            exhaustive_outputs,
            cfg=cfg,
            stage="rng_diagnostics",
        )
        exhaustive_status = child.value
        if exhaustive_status not in {"complete_valid", "blocked_by_cap"}:
            publish(
                cfg,
                cfg.rng_input_path("_failures/exhaustive.json"),
                {
                    "execution_status": "incomplete",
                    "completion_state": exhaustive_status,
                    "requested_lags": key["lags"],
                    "reason": "enabled_exhaustive_outputs_missing_or_invalid",
                },
                [proof, mr],
            )
            raise ValueError("enabled exhaustive lag execution is unexpectedly incomplete")
        reason = "descriptive legacy bands; no automatic band-crossing failure"
        if exhaustive_status == "blocked_by_cap" and status == "complete_valid":
            status = "insufficient_evidence"
    envelope = _envelope(cfg, proof, result)
    correctness = []
    for check in read(proof.parent / "plan.json")["recipe"]["checks"]:
        correctness.append(
            {
                "check_id": check["check_id"],
                "required": True,
                "coverage": check["name"],
                "minimum_support": None,
                "fault_recipe_id": "first_release_preflight_v1",
                "clean_fixture_count": None,
                "fault_fixture_count": None,
                "clean_alert_count": None,
                "detected_fault_count": None,
                "blind_spots": BLIND_SPOTS,
                "result_artifact": inventory([proof], proof.parent)[0],
                "kind": "correctness",
                "execution_status": "complete",
                "result": "pass",
                "reason": "declared_matrix_and_named_supplemental_checks_completed",
                "detail_artifact": "preflight/correctness.json",
            }
        )
    publish(
        cfg,
        pe,
        {
            **envelope,
            "checks": correctness,
            "coverage_complete": True,
            "status": "insufficient_evidence",
            "reason": "two_batch_correctness_matrix_is_not_dependence_evidence",
        },
        [proof],
    )
    check_records = []
    monitor_identity = inventory([mr], mr.parent)[0]
    for row in rows.to_pylist():
        check_records.append(
            {
                **row,
                "kind": "statistical",
                "execution_status": "complete",
                "required": True,
                "coverage": {"k": row["k"], "strategy": row["strategy"], "seat": 0},
                "minimum_support": 30,
                "fault_recipe_id": "batch_dependence_capability_v1",
                "clean_fixture_count": None,
                "fault_fixture_count": None,
                "clean_alert_count": None,
                "detected_fault_count": None,
                "blind_spots": BLIND_SPOTS,
                "result_artifact": monitor_identity,
            }
        )
    check_records.append(
        {
            "check_id": "exhaustive_game_lags",
            "required": False,
            "kind": "statistical",
            "execution_status": exhaustive_status,
            "coverage": exhaustive_status,
            "minimum_support": None,
            "fault_recipe_id": "legacy_lag_correctness_v4",
            "clean_fixture_count": None,
            "fault_fixture_count": None,
            "clean_alert_count": None,
            "detected_fault_count": None,
            "blind_spots": BLIND_SPOTS,
            "result_artifact": (
                inventory(exhaustive_outputs, mr.parent) if exhaustive_outputs else None
            ),
            "reason": reason,
        }
    )
    publish(
        cfg,
        mc,
        {
            **envelope,
            "checks": check_records,
            "status": status,
            "coverage_complete": not counts["insufficient_evidence"]
            and exhaustive_status != "blocked_by_cap",
            "statistical_warning_count": counts["investigate"],
            "check_opportunities": rows.num_rows,
            "exhaustive_status": exhaustive_status,
            "exhaustive_enabled": enabled,
            "requested_lags": key["lags"],
            "capability_assessment": result.get("groups", {}),
        },
        [proof, mr, *exhaustive_outputs],
    )
    write_stage_done(
        stamp,
        inputs=inputs,
        outputs=outputs,
        cfg=cfg,
        stage="rng_diagnostics",
        freshness_key=key,
        sidecar_artifacts=outputs,
        status="blocked_by_cap" if exhaustive_status == "blocked_by_cap" else "success",
    )
