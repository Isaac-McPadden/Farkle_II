# src/farkle/orchestration/two_seed_pipeline.py
"""Two-seed simulation + analysis pipeline orchestrator."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from farkle import analysis
from farkle.analysis.stage_runner import StageRunContext, StageRunner
from farkle.config import (
    AppConfig,
    apply_dot_overrides,
    assign_config_sha,
    load_app_config,
)
from farkle.orchestration.run_contexts import InterseedRunContext, SeedRunContext
from farkle.orchestration.seed_utils import (
    prepare_seed_config,
    resolve_seed_pair_args,
    seed_has_completion_markers,
    seed_pair_meta_root,
    seed_pair_root,
    seed_pair_seed_root,
    write_active_config,
)
from farkle.simulation import runner
from farkle.utils.logging import setup_info_logging
from farkle.utils.manifest import (
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_STAGE_END,
    append_manifest_event,
    ensure_manifest_v2,
    make_run_id,
)
from farkle.utils.parallel import (
    StageParallelPolicy,
    apply_native_thread_limits,
    normalize_n_jobs,
    resolve_stage_parallel_policy,
)
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SeedRunStatus:
    """Capture simulation and analysis outcomes for one seed in the pair."""

    seed: int
    context: SeedRunContext
    simulation_ok: bool
    analysis_ok: bool
    simulation_error: str | None = None
    analysis_error: str | None = None


def _per_seed_worker_budget(total_workers: int, seed_count: int) -> int:
    """Split the total worker budget across the number of concurrent seed runs.

    Args:
        total_workers: Total workers available to the orchestration run.
        seed_count: Number of seed runs that may execute concurrently.

    Returns:
        Minimum worker budget to assign to each seed.
    """
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    return max(1, total_workers // seed_count)


@dataclass(frozen=True)
class _PerSeedPolicyBundle:
    """Resolved stage-level parallelism policies for one seeded run family."""

    simulation: StageParallelPolicy
    ingest: StageParallelPolicy
    analysis: StageParallelPolicy

    def as_metadata(self) -> dict[str, dict[str, int]]:
        return {
            "simulation": {
                "total_cores": self.simulation.total_cores,
                "process_workers": self.simulation.process_workers,
                "python_threads": self.simulation.python_threads,
                "arrow_threads": self.simulation.arrow_threads,
                "native_threads_per_process": self.simulation.native_threads_per_process,
            },
            "ingest": {
                "total_cores": self.ingest.total_cores,
                "process_workers": self.ingest.process_workers,
                "python_threads": self.ingest.python_threads,
                "arrow_threads": self.ingest.arrow_threads,
                "native_threads_per_process": self.ingest.native_threads_per_process,
            },
            "analysis": {
                "total_cores": self.analysis.total_cores,
                "process_workers": self.analysis.process_workers,
                "python_threads": self.analysis.python_threads,
                "arrow_threads": self.analysis.arrow_threads,
                "native_threads_per_process": self.analysis.native_threads_per_process,
            },
        }


def _derive_per_seed_job_budgets(cfg: AppConfig, seed_count: int) -> _PerSeedPolicyBundle:
    """Resolve per-seed stage policies from the orchestration worker budget.

    Args:
        cfg: Application config providing stage-level ``n_jobs`` settings.
        seed_count: Number of seeds participating in the orchestration run.

    Returns:
        Per-seed simulation, ingest, and analysis policies derived from the budget.
    """
    total_workers = normalize_n_jobs(cfg.sim.n_jobs, default=1)
    seed_concurrency = seed_count if cfg.orchestration.parallel_seeds else 1
    per_seed_workers = _per_seed_worker_budget(total_workers, seed_concurrency)
    simulation_policy = resolve_stage_parallel_policy(
        "simulation",
        cfg.sim,
        n_jobs_override=per_seed_workers,
    )
    ingest_policy = resolve_stage_parallel_policy(
        "ingest",
        cfg.ingest,
        n_jobs_override=min(per_seed_workers, normalize_n_jobs(cfg.ingest.n_jobs)),
    )
    analysis_policy = resolve_stage_parallel_policy(
        "analysis",
        cfg.analysis,
        n_jobs_override=per_seed_workers,
    )
    policy_bundle = _PerSeedPolicyBundle(
        simulation=simulation_policy,
        ingest=ingest_policy,
        analysis=analysis_policy,
    )
    LOGGER.info(
        "Derived per-seed policy bundle",
        extra={
            "stage": "orchestration",
            "total_workers": int(total_workers),
            "seed_count": seed_count,
            "parallel_seeds": cfg.orchestration.parallel_seeds,
            "seed_concurrency": seed_concurrency,
            "per_seed_workers": per_seed_workers,
            "resolved_policy": policy_bundle.as_metadata(),
        },
    )
    return policy_bundle


def _build_seed_cfg(
    cfg: AppConfig,
    *,
    seed_pair: tuple[int, int],
    seed: int,
    meta_dir: Path,
    policy_bundle: _PerSeedPolicyBundle,
) -> AppConfig:
    """Clone ``cfg`` for one seed and apply the per-seed worker budget."""

    seed_cfg = prepare_seed_config(
        cfg,
        seed=seed,
        base_results_dir=seed_pair_seed_root(cfg, seed_pair, seed),
        meta_analysis_dir=meta_dir,
    )
    seed_cfg.sim.n_jobs = policy_bundle.simulation.process_workers
    seed_cfg.ingest.n_jobs = policy_bundle.ingest.process_workers
    seed_cfg.analysis.n_jobs = policy_bundle.analysis.process_workers
    return seed_cfg


def _run_one_seed(
    cfg: AppConfig,
    *,
    seed: int,
    seed_pair: tuple[int, int],
    meta_dir: Path,
    manifest_path: Path,
    run_id: str,
    force: bool,
    policy_bundle: _PerSeedPolicyBundle,
) -> _SeedRunStatus:
    """Run simulation and per-seed analysis for one member of the seed pair."""

    seed_cfg = _build_seed_cfg(
        cfg,
        seed_pair=seed_pair,
        seed=seed,
        meta_dir=meta_dir,
        policy_bundle=policy_bundle,
    )
    seed_context = SeedRunContext.from_config(seed_cfg)
    active_config_path = seed_context.active_config_path

    append_manifest_event(
        manifest_path,
        {
            "event": "seed_start",
            "seed": seed,
            "results_dir": str(seed_cfg.results_root),
            "active_config": str(active_config_path),
            "resolved_policy": policy_bundle.as_metadata(),
        },
        run_id=run_id,
        config_sha=seed_cfg.config_sha,
    )

    write_active_config(seed_cfg)
    LOGGER.info(
        "Using resolved config",
        extra={
            "stage": "orchestration",
            "seed": seed,
            "results_dir": str(seed_cfg.results_root),
            "active_config": str(active_config_path),
            "resolved_policy": policy_bundle.as_metadata(),
        },
    )

    apply_native_thread_limits(policy_bundle.simulation)

    if not force and seed_has_completion_markers(seed_cfg):
        LOGGER.info(
            "Skipping seed run (completion markers found)",
            extra={
                "stage": "orchestration",
                "seed": seed,
                "results_dir": str(seed_cfg.results_root),
            },
        )
        append_manifest_event(
            manifest_path,
            {
                "event": "seed_simulation_skipped",
                "seed": seed,
                "results_dir": str(seed_context.results_root),
            },
            run_id=run_id,
            config_sha=seed_cfg.config_sha,
        )
    else:
        LOGGER.info(
            "Running simulation",
            extra={
                "stage": "orchestration",
                "seed": seed,
                "results_dir": str(seed_cfg.results_root),
            },
        )
        try:
            runner.run_tournament(seed_cfg, force=force)
            append_manifest_event(
                manifest_path,
                {
                    "event": "seed_simulation_complete",
                    "seed": seed,
                    "results_dir": str(seed_context.results_root),
                },
                run_id=run_id,
                config_sha=seed_cfg.config_sha,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            append_manifest_event(
                manifest_path,
                {
                    "event": "seed_simulation_failed",
                    "seed": seed,
                    "results_dir": str(seed_context.results_root),
                    "error": error,
                },
                run_id=run_id,
                config_sha=seed_cfg.config_sha,
            )
            append_manifest_event(
                manifest_path,
                {
                    "event": "seed_analysis_skipped",
                    "seed": seed,
                    "results_dir": str(seed_context.results_root),
                    "reason": "simulation_failed",
                    "blocking_error": error,
                },
                run_id=run_id,
                config_sha=seed_cfg.config_sha,
            )
            return _SeedRunStatus(
                seed=seed,
                context=seed_context,
                simulation_ok=False,
                analysis_ok=False,
                simulation_error=error,
            )

    LOGGER.info(
        "Running per-seed analysis",
        extra={
            "stage": "orchestration",
            "seed": seed,
            "results_dir": str(seed_cfg.results_root),
        },
    )
    try:
        _run_per_seed_analysis(
            seed_cfg,
            seed=seed,
            force=force,
            policy_bundle=policy_bundle,
        )
        append_manifest_event(
            manifest_path,
            {
                "event": "seed_analysis_complete",
                "seed": seed,
                "results_dir": str(seed_context.results_root),
            },
            run_id=run_id,
            config_sha=seed_cfg.config_sha,
        )
        return _SeedRunStatus(
            seed=seed,
            context=seed_context,
            simulation_ok=True,
            analysis_ok=True,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        append_manifest_event(
            manifest_path,
            {
                "event": "seed_analysis_failed",
                "seed": seed,
                "results_dir": str(seed_context.results_root),
                "error": error,
            },
            run_id=run_id,
            config_sha=seed_cfg.config_sha,
        )
        return _SeedRunStatus(
            seed=seed,
            context=seed_context,
            simulation_ok=True,
            analysis_ok=False,
            analysis_error=error,
        )


@dataclass(frozen=True)
class _PipelineStageContract:
    name: str
    required_outputs: tuple[Path, ...]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ResolvedStageStatus:
    """Resolved artifact-validation result for one logical pipeline stage."""

    status: str
    diagnostics: list[str]
    diagnostic_codes: list[str]
    required_outputs: tuple[Path, ...]
    missing_outputs: tuple[Path, ...]


def _validate_artifact(path: Path) -> tuple[bool, str | None, str | None]:
    """Validate that a required artifact exists and is structurally usable.

    Args:
        path: Artifact path to validate.

    Returns:
        Tuple of ``(is_valid, reason, code)`` for pipeline health reporting.
    """
    if not path.exists():
        return False, "missing", "missing"
    if path.stat().st_size <= 0:
        return False, "empty", "empty"
    if path.suffix in {".json", ".jsonl"}:
        try:
            if path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if stripped:
                        json.loads(stripped)
                        break
                else:
                    return False, "empty jsonl", "empty"
            else:
                json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return False, "invalid json", "invalid_json"
    return True, None, None


def _resolve_stage_contract_status(
    contract: _PipelineStageContract,
    *,
    dependency_statuses: Sequence[str] = (),
    explicit_error: str | None = None,
    explicit_error_code: str | None = None,
) -> _ResolvedStageStatus:
    """Resolve a stage contract into a success/missing/failed health status.

    Args:
        contract: Stage contract describing required outputs.
        dependency_statuses: Upstream stage statuses that gate this contract.
        explicit_error: Optional explicit stage error text.
        explicit_error_code: Optional machine-readable code for ``explicit_error``.

    Returns:
        Resolved stage status with diagnostics and missing-output details.
    """
    diagnostics: list[str] = []
    diagnostic_codes: list[str] = []
    dependency_blocked = bool(dependency_statuses) and any(
        status != "success" for status in dependency_statuses
    )
    if explicit_error:
        diagnostics.append(explicit_error)
        diagnostic_codes.append(explicit_error_code or "stage_exception")
    if dependency_blocked:
        diagnostics.append(f"upstream incomplete: {', '.join(dependency_statuses)}")
        diagnostic_codes.append("dependency_failed")

    missing_outputs: list[Path] = []
    for output_path in contract.required_outputs:
        valid, reason, code = _validate_artifact(output_path)
        if not valid:
            missing_outputs.append(output_path)
            diagnostics.append(f"{output_path}: {reason}")
            diagnostic_codes.append(code or "missing")

    if explicit_error or (dependency_blocked and not missing_outputs):
        status = "failed"
    elif missing_outputs:
        status = "failed" if any(code in {"invalid_json", "stage_exception"} for code in diagnostic_codes) else "missing"
    else:
        status = "success"
    return _ResolvedStageStatus(
        status=status,
        diagnostics=diagnostics,
        diagnostic_codes=diagnostic_codes,
        required_outputs=contract.required_outputs,
        missing_outputs=tuple(missing_outputs),
    )


def _resolve_seed_family_statuses(
    seed: int,
    *,
    seed_cfg: AppConfig,
    simulation_error: str | None,
    analysis_error: str | None,
) -> dict[str, _ResolvedStageStatus]:
    """Resolve health statuses for the per-seed analysis family.

    Args:
        seed: Seed being summarized.
        seed_cfg: Resolved per-seed application config.
        simulation_error: Optional simulation failure message.
        analysis_error: Optional analysis failure message.

    Returns:
        Mapping of logical stage name to resolved health status.
    """
    analysis_manifest_candidates = (
        seed_cfg.analysis_dir / seed_cfg.manifest_name,
        seed_cfg.analysis_dir / "analysis_manifest.jsonl",
    )
    analysis_manifest = next(
        (path for path in analysis_manifest_candidates if path.exists()),
        analysis_manifest_candidates[0],
    )
    stage_contracts = (
        _PipelineStageContract(
            name=f"seed_{seed}.analysis",
            required_outputs=(analysis_manifest,),
        ),
        _PipelineStageContract(
            name=f"seed_{seed}.seed_symmetry",
            required_outputs=(seed_cfg.seed_symmetry_stage_dir / "seed_symmetry_summary.parquet",),
            depends_on=(f"seed_{seed}.analysis",),
        ),
        _PipelineStageContract(
            name=f"seed_{seed}.post_h2h",
            required_outputs=(seed_cfg.post_h2h_stage_dir / "h2h_s_tiers.json",),
            depends_on=(f"seed_{seed}.analysis",),
        ),
    )
    analysis_status = _resolve_stage_contract_status(
        stage_contracts[0],
        explicit_error=(
            analysis_error
            if analysis_error is not None
            else (
                f"simulation failed before analysis: {simulation_error}"
                if simulation_error is not None
                else None
            )
        ),
        explicit_error_code="stage_exception" if (analysis_error or simulation_error) else None,
    )
    return {
        stage_contracts[0].name: analysis_status,
        stage_contracts[1].name: _resolve_stage_contract_status(
            stage_contracts[1], dependency_statuses=(analysis_status.status,)
        ),
        stage_contracts[2].name: _resolve_stage_contract_status(
            stage_contracts[2], dependency_statuses=(analysis_status.status,)
        ),
    }


def _resolve_interseed_family_statuses(
    *,
    interseed_cfg: AppConfig,
    seed_pair: tuple[int, int],
    stage_statuses: dict[str, dict[str, Any]],
    interseed_error: str | None,
    h2h_error: str | None,
) -> dict[str, _ResolvedStageStatus]:
    """Resolve health statuses for interseed and post-interseed analysis stages.

    Args:
        interseed_cfg: Interseed application config with resolved output paths.
        seed_pair: Seed pair represented by the orchestration run.
        stage_statuses: Existing stage-status payloads from prior phases.
        interseed_error: Optional error from the interseed analysis run.
        h2h_error: Optional error from the H2H tier trends run.

    Returns:
        Mapping of logical stage name to resolved health status.
    """
    per_seed_post_h2h_stages = tuple(f"seed_{seed}.post_h2h" for seed in seed_pair)
    interseed_contract = _PipelineStageContract(
        name="interseed_analysis",
        required_outputs=(interseed_cfg.interseed_stage_dir / "interseed_summary.json",),
        depends_on=per_seed_post_h2h_stages,
    )
    interseed_status = _resolve_stage_contract_status(
        interseed_contract,
        dependency_statuses=tuple(stage_statuses.get(name, {}).get("status", "missing") for name in per_seed_post_h2h_stages),
        explicit_error=interseed_error,
        explicit_error_code="stage_exception" if interseed_error else None,
    )

    h2h_contract = _PipelineStageContract(
        name="h2h_tier_trends",
        required_outputs=(interseed_cfg.stage_dir("h2h_tier_trends") / "s_tier_trends.parquet",),
        depends_on=("interseed_analysis", *per_seed_post_h2h_stages),
    )
    h2h_status = _resolve_stage_contract_status(
        h2h_contract,
        dependency_statuses=(interseed_status.status,),
        explicit_error=h2h_error,
        explicit_error_code="stage_exception" if h2h_error else None,
    )
    return {
        interseed_contract.name: interseed_status,
        h2h_contract.name: h2h_status,
    }


def _write_pipeline_health(path: Path, payload: dict[str, Any]) -> None:
    """Write the pipeline health summary JSON atomically.

    Args:
        path: Destination JSON path.
        payload: Serializable health payload to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")




def _validate_required_config_sha_outputs(
    *,
    expected_sha: str,
    manifest_path: Path,
    run_id: str,
    seed_contexts: Mapping[int, SeedRunContext],
    interseed_cfg: AppConfig,
) -> list[str]:
    """Validate that key metadata artifacts carry the expected ``config_sha``.

    Args:
        expected_sha: Config SHA that should appear in validated outputs.
        manifest_path: Manifest path containing run events for this pipeline run.
        run_id: Run identifier used to filter manifest events.
        seed_contexts: Per-seed contexts supplying active-config artifact paths.
        interseed_cfg: Interseed config used to resolve optional stage outputs.

    Returns:
        Human-readable validation errors, or an empty list when all checks pass.
    """
    errors: list[str] = []

    def _check_json(path: Path) -> None:
        """Validate one JSON metadata artifact against the expected config SHA.

        Args:
            path: JSON artifact to inspect.
        """
        if not path.exists():
            errors.append(f"missing metadata: {path}")
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid metadata {path}: {type(exc).__name__}: {exc}")
            return
        value = payload.get("config_sha") if isinstance(payload, dict) else None
        if value != expected_sha:
            errors.append(f"{path} config_sha={value!r} expected {expected_sha!r}")

    if not manifest_path.exists():
        errors.append(f"missing metadata: {manifest_path}")
    else:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"invalid metadata {manifest_path}: {type(exc).__name__}: {exc}")
                continue
            if not isinstance(parsed, dict) or parsed.get("run_id") != run_id:
                continue
            record_event = parsed.get("event")
            if parsed.get("config_sha") != expected_sha:
                errors.append(
                    f"manifest event {record_event} has config_sha={parsed.get('config_sha')!r}"
                )

    for seed_context in seed_contexts.values():
        _check_json(seed_context.active_config_path.with_suffix(".done.json"))

    rng_stage_dir = interseed_cfg.stage_dir_if_active("rng_diagnostics")
    if rng_stage_dir is not None:
        rng_done = rng_stage_dir / "rng_diagnostics.done.json"
        if rng_done.exists():
            _check_json(rng_done)

    return errors

def _shared_meta_dir(cfg: AppConfig, pair_root: Path, seed_pair: tuple[int, int]) -> Path:
    """Resolve the shared meta-analysis directory for a seed pair.

    Args:
        cfg: Application config used to resolve configured meta roots.
        pair_root: Root results directory for the seed pair.
        seed_pair: Seed pair represented by the orchestration run.

    Returns:
        Directory where shared per-seed summary artifacts should be written.
    """
    configured_meta = seed_pair_meta_root(cfg, seed_pair)
    if configured_meta is not None:
        return configured_meta
    return pair_root / "interseed_analysis" / "seed_summaries_meta"


def _run_per_seed_analysis(
    cfg: AppConfig,
    *,
    seed: int,
    force: bool,
    policy_bundle: _PerSeedPolicyBundle,
) -> None:
    """Execute the per-seed analysis plan with resolved manifest metadata.

    Args:
        cfg: Per-seed application config.
        seed: Seed being analyzed.
        force: Recompute stages even when outputs appear current.
        policy_bundle: Resolved parallelism policies for the seeded run.
    """
    apply_native_thread_limits(policy_bundle.analysis)
    per_seed_manifest_path = cfg.analysis_dir / cfg.manifest_name
    per_seed_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    plan = analysis.build_per_seed_stage_plan(cfg, force=force)
    context = StageRunContext(
        config=cfg,
        manifest_path=per_seed_manifest_path,
        run_label=f"per_seed_pipeline_{seed}",
        run_metadata={
            "seed": seed,
            "results_dir": str(cfg.results_root),
            "analysis_dir": str(cfg.analysis_dir),
            "config_sha": cfg.config_sha,
            "resolved_policy": policy_bundle.as_metadata(),
        },
        run_end_metadata={
            "config_sha": cfg.config_sha,
            "resolved_policy": policy_bundle.as_metadata(),
        },
        continue_on_error=False,
        logger=LOGGER,
    )
    StageRunner.run(plan, context, raise_on_failure=True)


def run_pipeline(
    cfg: AppConfig,
    *,
    seed_pair: tuple[int, int],
    force: bool = False,
) -> None:
    """Run the full two-seed simulation and analysis orchestration."""

    if cfg.config_sha is None:
        assign_config_sha(cfg)
    pair_root = seed_pair_root(cfg, seed_pair)
    meta_dir = _shared_meta_dir(cfg, pair_root, seed_pair)
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pair_root / "two_seed_pipeline_manifest.jsonl"
    pipeline_health_path = pair_root / "pipeline_health.json"
    run_id = make_run_id(f"two_seed_pipeline_{seed_pair[0]}_{seed_pair[1]}")
    ensure_manifest_v2(manifest_path)
    stage_statuses: dict[str, dict[str, Any]] = {}
    stage_errors: dict[str, str] = {}
    policy_bundle = _derive_per_seed_job_budgets(cfg, len(seed_pair))

    append_manifest_event(
        manifest_path,
        {
            "event": EVENT_RUN_START,
            "seed_pair": list(seed_pair),
            "results_dir": str(pair_root),
            "meta_analysis_dir": str(meta_dir),
            "resolved_policy": policy_bundle.as_metadata(),
        },
        run_id=run_id,
        config_sha=cfg.config_sha,
    )

    seed_contexts: dict[int, SeedRunContext] = {}

    if cfg.orchestration.parallel_seeds:
        with ThreadPoolExecutor(max_workers=len(seed_pair)) as executor:
            futures = {
                seed: executor.submit(
                    _run_one_seed,
                    cfg,
                    seed=seed,
                    seed_pair=seed_pair,
                    meta_dir=meta_dir,
                    manifest_path=manifest_path,
                    run_id=run_id,
                    force=force,
                    policy_bundle=policy_bundle,
                )
                for seed in seed_pair
            }
            seed_results = {seed: futures[seed].result() for seed in seed_pair}
    else:
        seed_results = {
            seed: _run_one_seed(
                cfg,
                seed=seed,
                seed_pair=seed_pair,
                meta_dir=meta_dir,
                manifest_path=manifest_path,
                run_id=run_id,
                force=force,
                policy_bundle=policy_bundle,
            )
            for seed in seed_pair
        }

    for seed in seed_pair:
        seed_result = seed_results[seed]
        seed_context = seed_result.context
        seed_contexts[seed] = seed_context
        seed_cfg = seed_context.config
        if seed_result.analysis_error:
            stage_errors[f"seed_{seed}.analysis"] = seed_result.analysis_error
        elif seed_result.simulation_error:
            stage_errors[f"seed_{seed}.analysis"] = (
                f"simulation failed before analysis: {seed_result.simulation_error}"
            )

        seed_resolved = _resolve_seed_family_statuses(
            seed,
            seed_cfg=seed_cfg,
            simulation_error=seed_result.simulation_error,
            analysis_error=seed_result.analysis_error,
        )
        for stage_name, resolved in seed_resolved.items():
            stage_statuses[stage_name] = {
                "status": resolved.status,
                "required_outputs": [str(path) for path in resolved.required_outputs],
                "missing_required_outputs": [str(path) for path in resolved.missing_outputs],
                "diagnostics": resolved.diagnostics,
                "diagnostic_codes": resolved.diagnostic_codes,
            }
            append_manifest_event(
                manifest_path,
                {
                    "event": EVENT_STAGE_END,
                    "stage": stage_name,
                    "status": resolved.status,
                    "diagnostics": resolved.diagnostics,
                    "diagnostic_codes": resolved.diagnostic_codes,
                    "missing_required_outputs": [str(path) for path in resolved.missing_outputs],
                },
                run_id=run_id,
                config_sha=cfg.config_sha,
            )

    interseed_seed = seed_pair[0]
    interseed_seed_context = seed_contexts[interseed_seed]
    interseed_context = InterseedRunContext.from_seed_context(
        interseed_seed_context,
        seed_pair=seed_pair,
        analysis_root=pair_root / "interseed_analysis",
    )
    interseed_cfg = interseed_context.config

    apply_native_thread_limits(policy_bundle.analysis)

    per_seed_post_h2h_stages = tuple(f"seed_{seed}.post_h2h" for seed in seed_pair)
    interseed_should_run = all(
        stage_statuses.get(stage, {}).get("status") == "success" for stage in per_seed_post_h2h_stages
    )
    if interseed_should_run:
        LOGGER.info(
            "Running interseed analysis",
            extra={
                "stage": "orchestration",
                "seed": interseed_seed,
                "results_dir": str(interseed_cfg.results_root),
            },
        )
        append_manifest_event(
            manifest_path,
            {
                "event": "interseed_start",
                "seed": interseed_seed,
                "results_dir": str(interseed_cfg.results_root),
                "active_config": str(interseed_cfg.results_root / "active_config.yaml"),
                "resolved_policy": policy_bundle.as_metadata(),
            },
            run_id=run_id,
            config_sha=interseed_cfg.config_sha,
        )
        try:
            analysis.run_interseed_analysis(
                interseed_cfg,
                force=force,
                manifest_path=manifest_path,
            )
            append_manifest_event(
                manifest_path,
                {
                    "event": "interseed_complete",
                    "seed": interseed_seed,
                    "results_dir": str(interseed_cfg.results_root),
                    "resolved_policy": policy_bundle.as_metadata(),
                },
                run_id=run_id,
                config_sha=interseed_cfg.config_sha,
            )
        except Exception as exc:  # noqa: BLE001
            append_manifest_event(
                manifest_path,
                {
                    "event": "interseed_failed",
                    "seed": interseed_seed,
                    "results_dir": str(interseed_cfg.results_root),
                    "error": f"{type(exc).__name__}: {exc}",
                    "resolved_policy": policy_bundle.as_metadata(),
                },
                run_id=run_id,
                config_sha=interseed_cfg.config_sha,
            )
            stage_errors["interseed_analysis"] = f"{type(exc).__name__}: {exc}"
    seed_s_tier_paths = [
        seed_contexts[seed].config.post_h2h_stage_dir / "h2h_s_tiers.json" for seed in seed_pair
    ]
    missing_tier_inputs = [path for path in seed_s_tier_paths if not path.exists()]
    tier_should_run = interseed_should_run and "interseed_analysis" not in stage_errors and not missing_tier_inputs
    if tier_should_run:
        LOGGER.info(
            "Running head-to-head tier trends",
            extra={
                "stage": "orchestration",
                "seed": interseed_seed,
                "results_dir": str(interseed_cfg.results_root),
            },
        )
        append_manifest_event(
            manifest_path,
            {
                "event": "h2h_tier_trends_start",
                "seed": interseed_seed,
                "results_dir": str(interseed_cfg.results_root),
                "resolved_policy": policy_bundle.as_metadata(),
            },
            run_id=run_id,
            config_sha=interseed_cfg.config_sha,
        )
        try:
            analysis.run_h2h_tier_trends(
                interseed_cfg,
                force=force,
                seed_s_tier_paths=seed_s_tier_paths,
            )
            append_manifest_event(
                manifest_path,
                {
                    "event": "h2h_tier_trends_complete",
                    "seed": interseed_seed,
                    "results_dir": str(interseed_cfg.results_root),
                    "resolved_policy": policy_bundle.as_metadata(),
                },
                run_id=run_id,
                config_sha=interseed_cfg.config_sha,
            )
        except Exception as exc:  # noqa: BLE001
            stage_errors["h2h_tier_trends"] = f"{type(exc).__name__}: {exc}"

    interseed_resolved = _resolve_interseed_family_statuses(
        interseed_cfg=interseed_cfg,
        seed_pair=seed_pair,
        stage_statuses=stage_statuses,
        interseed_error=stage_errors.get("interseed_analysis"),
        h2h_error=stage_errors.get("h2h_tier_trends"),
    )
    for stage_name, resolved in interseed_resolved.items():
        stage_statuses[stage_name] = {
            "status": resolved.status,
            "depends_on": list(
                tuple(f"seed_{seed}.post_h2h" for seed in seed_pair)
                if stage_name == "interseed_analysis"
                else ("interseed_analysis", *(f"seed_{seed}.post_h2h" for seed in seed_pair))
            ),
            "required_outputs": [str(path) for path in resolved.required_outputs],
            "missing_required_outputs": [str(path) for path in resolved.missing_outputs],
            "diagnostics": resolved.diagnostics,
            "diagnostic_codes": resolved.diagnostic_codes,
        }
        append_manifest_event(
            manifest_path,
            {
                "event": EVENT_STAGE_END,
                "stage": stage_name,
                "status": resolved.status,
                "diagnostics": resolved.diagnostics,
                "diagnostic_codes": resolved.diagnostic_codes,
                "missing_required_outputs": [str(path) for path in resolved.missing_outputs],
                "resolved_policy": policy_bundle.as_metadata(),
            },
            run_id=run_id,
            config_sha=interseed_cfg.config_sha,
        )

    sha_errors = _validate_required_config_sha_outputs(
        expected_sha=cfg.config_sha or "",
        manifest_path=manifest_path,
        run_id=run_id,
        seed_contexts=seed_contexts,
        interseed_cfg=interseed_cfg,
    )
    if sha_errors:
        stage_statuses["config_sha_validation"] = {
            "status": "failed",
            "diagnostics": sha_errors,
            "diagnostic_codes": ["config_sha_mismatch"] * len(sha_errors),
            "required_outputs": [
                str(manifest_path),
                *[str(ctx.active_config_path.with_suffix(".done.json")) for ctx in seed_contexts.values()],
            ],
            "missing_required_outputs": [],
        }

    if any(payload["status"] in {"failed", "missing"} for payload in stage_statuses.values()):
        overall_status = "failed_blocked"
    else:
        overall_status = "complete_success"

    first_blocking_failure = next(
        (
            {
                "stage": name,
                "status": payload["status"],
                **(
                    {"missing_required_outputs": payload.get("missing_required_outputs", [])}
                    if payload.get("missing_required_outputs")
                    else {}
                ),
                **(
                    {"error": payload["diagnostics"][0]}
                    if payload.get("diagnostics")
                    else {}
                ),
            }
            for name, payload in stage_statuses.items()
            if payload["status"] in {"failed", "missing"}
        ),
        None,
    )

    health_payload = {
        "seed_pair": list(seed_pair),
        "status": overall_status,
        "config_sha": cfg.config_sha,
        "stage_statuses": stage_statuses,
        "missing_required_outputs": {
            name: payload.get("missing_required_outputs", [])
            for name, payload in stage_statuses.items()
            if payload.get("missing_required_outputs")
        },
        "first_blocking_failure": first_blocking_failure,
    }
    _write_pipeline_health(pipeline_health_path, health_payload)

    append_manifest_event(
        manifest_path,
        {
            "event": EVENT_RUN_END,
            "status": overall_status,
            "health_artifact": str(pipeline_health_path),
        },
        run_id=run_id,
        config_sha=cfg.config_sha,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the module-level parser for two-seed orchestration."""

    parser = argparse.ArgumentParser(prog="farkle two-seed-pipeline")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/fast_config.yaml"), help="Path to YAML config"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values",
    )
    parser.add_argument(
        "--seed-a",
        type=int,
        help="Override the first seed for dual-seed orchestration",
    )
    parser.add_argument(
        "--seed-b",
        type=int,
        help="Override the second seed for dual-seed orchestration",
    )
    parser.add_argument(
        "--seed-pair",
        type=int,
        nargs=2,
        metavar=("A", "B"),
        help="Override the dual-seed tuple (A B)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when completion markers exist",
    )
    parser.add_argument(
        "--parallel-seeds",
        action="store_true",
        help="Run per-seed simulation and analysis concurrently",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and dispatch the two-seed pipeline."""

    parser = build_parser()
    args = parser.parse_args(argv)
    setup_info_logging()

    cfg = load_app_config(Path(args.config), seed_list_len=2)
    cfg = apply_dot_overrides(cfg, list(args.overrides or []))
    if args.parallel_seeds:
        cfg.orchestration.parallel_seeds = True

    seed_pair = resolve_seed_pair_args(args, parser)
    if seed_pair is not None:
        cfg.sim.seed_list = list(seed_pair)
        cfg.sim.seed_pair = seed_pair
    resolved_seeds = cfg.sim.populate_seed_list(2)
    assign_config_sha(cfg)
    seed_pair = (resolved_seeds[0], resolved_seeds[1])

    run_pipeline(cfg, seed_pair=seed_pair, force=args.force)
    return 0


__all__ = ["main", "run_pipeline"]
