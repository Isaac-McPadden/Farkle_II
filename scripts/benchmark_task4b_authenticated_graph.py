"""Bounded Task 4B benchmark for authenticated graph finalization."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import psutil
import pyarrow as pa

from farkle.analysis.release_audit import audit_sidecar_completeness
from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig, assign_config_sha
from farkle.orchestration.run_contexts import (
    RootPairRunContext,
    SeedRunContext,
    load_run_context,
    write_run_context_atomic,
)
from farkle.orchestration.seed_utils import write_active_config
from farkle.orchestration.two_seed_pipeline import _final_release_gate, _SeedRunStatus
from farkle.utils.artifact_contract import make_artifact_sidecar, sha256_file
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.authenticated_contract import (
    CodeIdentity,
    CodeIdentityPolicy,
    resolve_code_identity,
)
from farkle.utils.authenticated_graph import (
    AuthenticatedGraphSnapshot,
    SnapshotGeneration,
    capture_authenticated_graph_snapshot,
)
from farkle.utils.authentication_telemetry import (
    AuthenticationTelemetry,
    use_authentication_telemetry,
)
from farkle.utils.release_identity import write_v3_stage_completion
from farkle.utils.stage_completion import CompletionState, resolve_stage_state
from farkle.utils.writer import atomic_path

BENCHMARK_VERSION = 1
OWNER_FILENAME = ".task4b_benchmark_owner.json"


@dataclass(frozen=True)
class BenchContext:
    context: SeedRunContext | RootPairRunContext
    stage_key: str
    outputs: tuple[Path, ...]
    completion: Path
    snapshot: AuthenticatedGraphSnapshot
    generation: SnapshotGeneration
    snapshot_seconds: float
    snapshot_telemetry: dict[str, int]


def _config(root: Path, seed: int) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=root / f"root_{seed}"),
        sim=SimConfig(seed=seed, seed_list=[seed], n_players_list=[2]),
    )
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    assign_config_sha(cfg)
    return cfg


def _publish_outputs(
    cfg: AppConfig,
    *,
    stage_key: str,
    count: int,
    pair_scope: bool,
) -> tuple[tuple[Path, ...], Path]:
    outputs: list[Path] = []
    for index in range(count):
        scope = ArtifactScope.CROSS_SEED if pair_scope else ArtifactScope.BY_K
        path = cfg.scope_path(
            stage_key,
            scope,
            f"benchmark/output_{index:05d}.parquet",
            k=None if pair_scope else 2,
        )
        table = pa.table({"coordinate": [index], "value": [index * 3 + 1]})
        metadata = make_artifact_sidecar(
            cfg,
            path,
            producer=stage_key,
            scope=scope,
            source_scope=scope,
            operation="task4b_benchmark_output",
            baseline="deterministic_fixture",
            weighted_quantity="none",
            support_count_role="fixture_rows",
            uncertainty_method="none",
            replication_unit="artifact",
            conditioning="unconditional",
            consistency_columns=table.schema.names,
            player_counts=[2],
            required_player_counts=[2],
            missing_cell_policy="fail",
            seed_scope="both_roots_combined" if pair_scope else "single_root",
            method_contract={
                "kind": "operation",
                "procedure": "task4b_benchmark_output",
                "parameters": {"method_version": 1},
            },
        )
        write_parquet_artifact_atomic(table, path, sidecar=metadata)
        outputs.append(path)
    completion = cfg.stage_dir(stage_key) / f"{stage_key}.done.json"
    write_v3_stage_completion(
        completion,
        cfg=cfg,
        stage_key=stage_key,
        inputs=[],
        outputs=outputs,
        status="success",
    )
    return tuple(outputs), completion


def _capture(
    context: SeedRunContext | RootPairRunContext,
    *,
    stage_key: str,
    outputs: tuple[Path, ...],
    completion: Path,
    code: CodeIdentity,
    pair_scope: bool,
) -> BenchContext:
    state = resolve_stage_state(
        completion,
        inputs=[],
        outputs=outputs,
        cfg=context.config,
        stage=stage_key,
    )
    if state is not CompletionState.COMPLETE_VALID:
        raise RuntimeError(f"benchmark context did not authenticate: {stage_key}: {state.value}")
    telemetry = AuthenticationTelemetry()
    generation = SnapshotGeneration()
    started = time.perf_counter()
    with use_authentication_telemetry(telemetry):
        snapshot = capture_authenticated_graph_snapshot(
            cfg=context.config,
            scope="pair" if pair_scope else "root",
            roots=(
                tuple(sorted(context.root_pair))
                if isinstance(context, RootPairRunContext)
                else (context.seed,)
            ),
            graph_root=(
                context.analysis_root
                if isinstance(context, RootPairRunContext)
                else context.results_root
            ),
            analysis_root=context.analysis_root,
            run_context_path=context.run_context_path,
            active_config_path=context.active_config_path,
            stage_states={stage_key: CompletionState.COMPLETE_VALID.value},
            completion_paths=[(stage_key, completion)],
            generation=generation,
            code_identity=code,
        )
    return BenchContext(
        context=context,
        stage_key=stage_key,
        outputs=outputs,
        completion=completion,
        snapshot=snapshot,
        generation=generation,
        snapshot_seconds=time.perf_counter() - started,
        snapshot_telemetry=telemetry.as_metadata(),
    )


def build_fixture(root: Path, artifact_count: int, code: CodeIdentity) -> tuple[BenchContext, ...]:
    per_context = [artifact_count // 3] * 3
    for index in range(artifact_count % 3):
        per_context[index] += 1
    root_contexts: list[SeedRunContext] = []
    captures: list[BenchContext] = []
    for seed, count in zip((11, 22), per_context[:2], strict=True):
        cfg = _config(root, seed)
        context = SeedRunContext.from_config(cfg)
        write_run_context_atomic(context, code_identity=code)
        write_active_config(cfg)
        outputs, completion = _publish_outputs(
            cfg,
            stage_key="metrics",
            count=max(1, count),
            pair_scope=False,
        )
        captured = _capture(
            context,
            stage_key="metrics",
            outputs=outputs,
            completion=completion,
            code=code,
            pair_scope=False,
        )
        root_contexts.append(context)
        captures.append(captured)
    pair = RootPairRunContext.from_root_contexts(
        (root_contexts[0], root_contexts[1]),
        pair_root=root / "pair",
    )
    write_run_context_atomic(
        pair,
        code_identity=code,
        parent_lifecycle_roots=tuple(item.snapshot.lifecycle_sha256 for item in captures),
    )
    write_active_config(pair.config, dest_dir=pair.pair_root)
    outputs, completion = _publish_outputs(
        pair.config,
        stage_key="root_stability",
        count=max(1, per_context[2]),
        pair_scope=True,
    )
    captures.append(
        _capture(
            pair,
            stage_key="root_stability",
            outputs=outputs,
            completion=completion,
            code=code,
            pair_scope=True,
        )
    )
    return tuple(captures)


def _baseline(contexts: Sequence[BenchContext]) -> tuple[dict[str, Any], dict[str, int]]:
    telemetry = AuthenticationTelemetry()
    failures: list[str] = []
    phases: dict[str, float] = {}
    started_total = time.perf_counter()
    with use_authentication_telemetry(telemetry):
        started = time.perf_counter()
        for item in contexts:
            state = resolve_stage_state(
                item.completion,
                inputs=[],
                outputs=item.outputs,
                cfg=item.context.config,
                stage=item.stage_key,
            )
            if state is not CompletionState.COMPLETE_VALID:
                failures.append(f"state:{item.stage_key}:{state.value}")
        phases["state_recheck_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        for item in contexts:
            load_run_context(
                item.context.run_context_path,
                active_config_path=item.context.active_config_path,
            )
            sha256_file(item.context.run_context_path)
        phases["run_context_authentication_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        for item in contexts:
            failures.extend(audit_sidecar_completeness(item.snapshot.graph_root))
        phases["final_audit_seconds"] = time.perf_counter() - started
    phases["total_finalization_seconds"] = time.perf_counter() - started_total
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        **phases,
    }, telemetry.as_metadata()


def _optimized(
    contexts: Sequence[BenchContext],
    code: CodeIdentity,
) -> tuple[dict[str, Any], dict[str, int]]:
    telemetry = AuthenticationTelemetry()
    roots = contexts[:2]
    pair = contexts[2]
    root_results = {
        int(item.context.seed): _SeedRunStatus(
            seed=int(item.context.seed),
            context=item.context,
            simulation_ok=True,
            analysis_ok=True,
            lifecycle_sha256=item.snapshot.lifecycle_sha256,
            stage_states=dict(item.snapshot.stage_states),
            graph_snapshot=item.snapshot,
            snapshot_generation=item.generation,
        )
        for item in roots
        if isinstance(item.context, SeedRunContext)
    }
    started_total = time.perf_counter()
    started = time.perf_counter()
    for item in contexts:
        item.generation.validate(
            item.snapshot,
            expected_scope=item.snapshot.scope,
            expected_roots=item.snapshot.roots,
            expected_run_context_path=item.snapshot.run_context_path,
            telemetry=telemetry,
        )
    reuse_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with use_authentication_telemetry(telemetry):
        result = _final_release_gate(
            root_results,
            pair.context,  # type: ignore[arg-type]
            code_identity=code,
            allow_oracle_code_identity=True,
            pair_snapshot=pair.snapshot,
            pair_generation=pair.generation,
            authentication_telemetry=telemetry,
        )
    audit_seconds = time.perf_counter() - started
    return (
        {
            "status": result["status"],
            "failures": result["failures"],
            "snapshot_reuse_seconds": reuse_seconds,
            "final_audit_seconds": audit_seconds,
            "total_finalization_seconds": time.perf_counter() - started_total,
            "top_level_invocations": result["top_level_invocations"],
            "internal_roots": result["internal_roots"],
        },
        telemetry.as_metadata(),
    )


def _measure(
    contexts: Sequence[BenchContext],
    code: CodeIdentity,
    mode: str,
) -> dict[str, Any]:
    process = psutil.Process()
    rss_before = process.memory_info().rss
    result, telemetry = _baseline(contexts) if mode == "baseline" else _optimized(contexts, code)
    return {
        "mode": mode,
        "result": result,
        "telemetry": telemetry,
        "peak_observed_rss_bytes": max(rss_before, process.memory_info().rss),
    }


def _bundle_digest(paths: Iterable[Path]) -> str:
    from hashlib import sha256

    digest = sha256()
    for path in sorted(paths):
        encoded = path.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def run_benchmark(
    root: Path,
    *,
    sizes: Sequence[int],
    repetitions: int,
    force: bool,
) -> dict[str, Any]:
    evidence_path = root / "task4b_authenticated_graph.json"
    if evidence_path.is_file() and not force:
        return json.loads(evidence_path.read_text(encoding="utf-8"))
    owner_path = root / OWNER_FILENAME
    if root.exists() and not owner_path.is_file():
        raise RuntimeError(f"refusing unowned Task 4B benchmark root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if not owner_path.exists():
        owner_path.write_text(
            json.dumps(
                {
                    "benchmark": "task4b_authenticated_graph",
                    "benchmark_version": BENCHMARK_VERSION,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    code = resolve_code_identity(
        Path(__file__).resolve().parents[1],
        policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY,
    )
    measurements: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for size in sizes:
        fixture_root = root / f"graph_{size:05d}"
        contexts = build_fixture(fixture_root, size, code)
        all_outputs = [path for item in contexts for path in item.outputs]
        canonical_before = _bundle_digest(all_outputs)
        _measure(contexts, code, "baseline")
        _measure(contexts, code, "optimized")
        for repetition in range(repetitions):
            order = ("baseline", "optimized") if repetition % 2 == 0 else ("optimized", "baseline")
            for position, mode in enumerate(order, start=1):
                measurement = _measure(contexts, code, mode)
                measurement.update(
                    artifact_count=size,
                    repetition=repetition + 1,
                    position=position,
                )
                measurements.append(measurement)
        canonical_after = _bundle_digest(all_outputs)
        by_mode = {
            mode: [
                item["result"]["total_finalization_seconds"]
                for item in measurements
                if item["artifact_count"] == size and item["mode"] == mode
            ]
            for mode in ("baseline", "optimized")
        }
        baseline_median = statistics.median(by_mode["baseline"])
        optimized_median = statistics.median(by_mode["optimized"])
        summaries.append(
            {
                "artifact_count": size,
                "context_artifact_counts": [
                    len(item.snapshot.graph_inventory) for item in contexts
                ],
                "baseline_median_seconds": baseline_median,
                "optimized_median_seconds": optimized_median,
                "optimized_to_baseline_ratio": optimized_median / baseline_median,
                "improvement_fraction": 1.0 - optimized_median / baseline_median,
                "canonical_outputs_unchanged": canonical_before == canonical_after,
                "snapshot_build_seconds": sum(item.snapshot_seconds for item in contexts),
                "snapshot_peak_inventory_entries": max(
                    len(item.snapshot.graph_inventory) for item in contexts
                ),
            }
        )

    tamper_contexts = build_fixture(root / "tamper", 9, code)
    tamper_path = tamper_contexts[0].outputs[0]
    original = tamper_path.read_bytes()
    tamper_path.write_bytes(original + b"tamper")
    baseline_tamper, _ = _baseline(tamper_contexts)
    optimized_tamper, _ = _optimized(tamper_contexts, code)
    tamper_path.write_bytes(original)
    evidence = {
        "task": "4B",
        "benchmark_version": BENCHMARK_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root.resolve()),
        "repository_head": code.commit,
        "python": sys.version,
        "platform": platform.platform(),
        "settings": {"sizes": list(sizes), "repetitions": repetitions, "warmups": 1},
        "measurements": measurements,
        "summary": summaries,
        "tamper_equivalence": {
            "baseline_status": baseline_tamper["status"],
            "optimized_status": optimized_tamper["status"],
            "both_fail_closed": (
                baseline_tamper["status"] == "failed" and optimized_tamper["status"] == "failed"
            ),
        },
    }
    with atomic_path(str(evidence_path)) as temporary:
        Path(temporary).write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return evidence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/farkle-task4b-authenticated-graph-v1"),
    )
    parser.add_argument("--sizes", default="9,128,512")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sizes = tuple(int(value) for value in args.sizes.split(",") if value.strip())
    if not sizes or any(value < 3 for value in sizes):
        raise ValueError("benchmark sizes must contain at least three artifacts")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    evidence = run_benchmark(
        args.root,
        sizes=sizes,
        repetitions=args.repetitions,
        force=args.force,
    )
    print(json.dumps(evidence["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
