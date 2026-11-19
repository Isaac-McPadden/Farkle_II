from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path

import pyarrow as pa
import pytest
from tests.helpers.artifact_sidecars import (
    clean_test_code_identity,
    make_authenticated_v3_config,
    publish_v3_parquet,
    publish_v3_simulation_run,
)

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import (
    AppConfig,
    assign_config_sha,
    compute_config_sha,
    load_app_config,
)
from farkle.orchestration.run_contexts import (
    RootPairRunContext,
    SeedRunContext,
    configure_run_lineage,
    load_run_context,
    write_run_context_atomic,
)
from farkle.orchestration.seed_utils import write_active_config
from farkle.simulation import runner
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils.artifact_contract import sidecar_path
from farkle.utils.authenticated_contract import CodeIdentity, reuse_stage_parquet_facts
from farkle.utils.authentication_telemetry import (
    AuthenticationTelemetry,
    use_authentication_telemetry,
)
from farkle.utils.stage_completion import (
    CompletionState,
    resolve_stage_state,
    stage_done_path,
    write_stage_done,
)


def _code(commit: str = "a" * 40) -> CodeIdentity:
    return clean_test_code_identity(commit)


def _cfg(tmp_path: Path, *, root: int = 11) -> AppConfig:
    cfg = make_authenticated_v3_config(
        tmp_path,
        name=f"root_{root}",
        root_seed=root,
    )
    cfg.screening.resolution_delta = 0.9
    cfg.batching.target_batches = 2
    cfg.batching.min_shuffles_per_batch = 1
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    assign_config_sha(cfg)
    cfg._code_identity = _code()
    cfg._run_lineage_sha256 = "1" * 64
    return cfg


def _publish_table(
    cfg: AppConfig,
    path: Path,
    *,
    producer: str,
    sources: list[Path] | None = None,
) -> None:
    table = pa.table({"value": [1]})
    publish_v3_parquet(
        cfg,
        path,
        table,
        stage_key=producer,
        producer=producer,
        operation=f"publish_{producer}_test",
        sources=tuple(sources or ()),
        source_scope="by_k",
    )


@pytest.mark.parametrize(
    "mutation",
    ("grid", "strategy_manifest", "output", "sidecar", "code", "method"),
)
def test_simulation_completion_mutation_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.sim.row_dir = Path("rows")
    assign_config_sha(cfg)
    published = publish_v3_simulation_run(
        cfg,
        pa.table({"value": [1]}),
        strategies=(
            ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=1),
            ThresholdStrategy(score_threshold=100, dice_threshold=5, strategy_id=2),
        ),
    )
    strategy_manifest = published.strategy_manifest
    output = published.shard
    assert runner.simulation_is_complete(cfg, 2)

    if mutation == "grid":
        cfg.sim.score_thresholds = [999]
    elif mutation == "strategy_manifest":
        strategy_manifest.write_bytes(strategy_manifest.read_bytes() + b"mutation")
    elif mutation == "output":
        output.write_bytes(b"output-v2")
    elif mutation == "sidecar":
        sidecar_path(output).write_text("sidecar-v2", encoding="utf-8")
    elif mutation == "code":
        cfg._code_identity = _code("b" * 40)
    else:
        metadata = json.loads(sidecar_path(output).read_text(encoding="utf-8"))
        metadata["versions"]["method_versions"]["tournament_method_version"] = 99
        sidecar_path(output).write_text(json.dumps(metadata), encoding="utf-8")

    assert not runner.simulation_is_complete(cfg, 2)


@pytest.mark.parametrize(
    "mutation",
    ("input", "output", "sidecar", "stage_config", "code", "method"),
)
def test_root_stage_completion_mutation_matrix(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    source = cfg.combined_rows_by_k(2)
    output = cfg.metrics_all_player_batch_path(2)
    _publish_table(cfg, source, producer="combine")
    _publish_table(cfg, output, producer="metrics", sources=[source])
    adjacent = sidecar_path(output)
    done = stage_done_path(cfg.metrics_stage_dir, "metrics")
    write_stage_done(done, inputs=[source], outputs=[output], cfg=cfg, stage="metrics")
    assert (
        resolve_stage_state(done, [source], [output], cfg=cfg, stage="metrics")
        is CompletionState.COMPLETE_VALID
    )

    if mutation == "input":
        source.write_bytes(b"input-v2")
    elif mutation == "output":
        output.write_bytes(b"output-v2")
    elif mutation == "sidecar":
        adjacent.write_text("sidecar-v2", encoding="utf-8")
    elif mutation == "stage_config":
        cfg.screening.resolution_delta = 0.8
    elif mutation == "code":
        cfg._code_identity = _code("b" * 40)
    else:
        from farkle.analysis import outcome_distributions

        monkeypatch.setattr(
            outcome_distributions,
            "OUTCOME_METHOD_VERSION",
            outcome_distributions.OUTCOME_METHOD_VERSION + 1,
        )

    assert (
        resolve_stage_state(done, [source], [output], cfg=cfg, stage="metrics")
        is CompletionState.COMPLETE_STALE
    )


def test_runtime_only_change_does_not_stale_root_stage(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    source = cfg.combined_rows_by_k(2)
    output = cfg.metrics_all_player_batch_path(2)
    _publish_table(cfg, source, producer="combine")
    _publish_table(cfg, output, producer="metrics", sources=[source])
    done = stage_done_path(cfg.metrics_stage_dir, "metrics")
    write_stage_done(done, inputs=[source], outputs=[output], cfg=cfg, stage="metrics")

    cfg.analysis.n_jobs = 99
    assign_config_sha(cfg)

    assert (
        resolve_stage_state(done, [source], [output], cfg=cfg, stage="metrics")
        is CompletionState.COMPLETE_VALID
    )


def test_cross_stage_parquet_reads_and_invocation_lifetime(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    source = cfg.combined_rows_by_k(2)
    output = cfg.metrics_all_player_batch_path(2)
    _publish_table(cfg, source, producer="combine")
    _publish_table(cfg, output, producer="metrics", sources=[source])
    for stage, path in (("combine", source), ("metrics", output)):
        write_stage_done(
            stage_done_path(cfg.stage_dir(stage), stage),
            inputs=[],
            outputs=[path],
            cfg=cfg,
            stage=stage,
        )

    def check(stage: str) -> AuthenticationTelemetry:
        counts = AuthenticationTelemetry()
        with use_authentication_telemetry(counts):
            assert (
                resolve_stage_state(
                    stage_done_path(cfg.stage_dir(stage), stage), [], [], cfg=cfg, stage=stage
                )
                is CompletionState.COMPLETE_VALID
            )
        return counts

    baseline = check("metrics")
    with reuse_stage_parquet_facts():
        assert check("combine").schema_validations == 1
        shared = check("metrics")
        assert shared.schema_validations == baseline.schema_validations - 1
        assert (
            shared.artifact_bytes_hashed == baseline.artifact_bytes_hashed - source.stat().st_size
        )
        assert shared.as_metadata()["total_opens"] == baseline.as_metadata()["total_opens"]
        assert check("metrics").artifact_bytes_hashed == 0
        publication = AuthenticationTelemetry()
        with use_authentication_telemetry(publication):
            write_stage_done(
                stage_done_path(cfg.stage_dir("metrics"), "metrics"),
                inputs=[source],
                outputs=[output],
                cfg=cfg,
                stage="metrics",
            )
        assert publication.artifact_bytes_hashed >= source.stat().st_size + output.stat().st_size
        copied = copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            foreign = pool.submit(copied.run, check, "metrics").result()
        assert foreign.schema_validations == baseline.schema_validations
    assert copied.run(check, "metrics").schema_validations == baseline.schema_validations
    with reuse_stage_parquet_facts():
        assert check("metrics").artifact_bytes_hashed == baseline.artifact_bytes_hashed


@pytest.mark.parametrize("mutation", ["writer", "rebind", "interrupted"])
def test_warm_completion_reopens_live_namespace(tmp_path: Path, mutation: str) -> None:
    cfg = _cfg(tmp_path)
    output = cfg.metrics_all_player_batch_path(2)
    _publish_table(cfg, output, producer="metrics")
    stamp = stage_done_path(cfg.metrics_stage_dir, "metrics")
    write_stage_done(stamp, inputs=[], outputs=[output], cfg=cfg, stage="metrics")

    def check() -> CompletionState:
        return resolve_stage_state(stamp, [], [output], cfg=cfg, stage="metrics")

    if mutation == "rebind":
        # Even a read-only leaf cannot pin its parent directory.
        output.chmod(stat.S_IREAD)
    with reuse_stage_parquet_facts():
        assert check() is CompletionState.COMPLETE_VALID
        if mutation == "writer":
            # A separate writer can change this file even with an ordinary reader open.
            before = output.stat()
            with output.open("rb"):
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; f=open(sys.argv[1],'r+b'); f.write(b'FAIL'); f.close()",
                        str(output),
                    ],
                    check=True,
                )
            os.utime(output, ns=(before.st_atime_ns, before.st_mtime_ns))
        elif mutation == "rebind":
            old = output.parent.with_name("previous_directory")
            output.parent.rename(old)
            shutil.copytree(old, output.parent)
            output.chmod(stat.S_IREAD | stat.S_IWRITE)
            (old / output.name).chmod(stat.S_IREAD | stat.S_IWRITE)
            with output.open("r+b") as handle:
                handle.write(b"FAIL")
        else:
            sidecar_path(output).rename(output.with_suffix(".staged"))
        assert check() is CompletionState.COMPLETE_STALE


def test_pair_public_config_and_context_round_trip_bind_parent_and_stage_hashes(
    tmp_path: Path,
) -> None:
    roots = (
        SeedRunContext.from_config(_cfg(tmp_path, root=11)),
        SeedRunContext.from_config(_cfg(tmp_path, root=22)),
    )
    pair = RootPairRunContext.from_root_contexts(roots, pair_root=tmp_path / "pair")
    parents = ("2" * 64, "3" * 64)
    write_run_context_atomic(pair, code_identity=_code(), parent_lifecycle_roots=parents)
    write_active_config(pair.config, pair.pair_root)

    reloaded = load_app_config(pair.active_config_path, seed_list_len=2)
    assert compute_config_sha(reloaded) == pair.config.config_sha
    serialized = pair.active_config_path.read_text(encoding="utf-8")
    assert "_analysis_root_override" not in serialized
    assert "_root_input_layout_override" not in serialized
    persisted_context = load_run_context(
        pair.run_context_path,
        active_config_path=pair.active_config_path,
    )
    assert tuple(persisted_context["parent_lifecycle_roots"]) == parents
    assert persisted_context["run_lineage_sha256"] == pair.config._run_lineage_sha256

    output = pair.config.root_discrepancies_path()
    _publish_table(pair.config, output, producer="root_stability")
    done = stage_done_path(pair.config.stage_dir("root_stability"), "root_stability")
    write_stage_done(
        done,
        inputs=[],
        outputs=[output],
        cfg=pair.config,
        stage="root_stability",
    )
    stamp = json.loads(done.read_text(encoding="utf-8"))
    assert stamp["state"] == CompletionState.COMPLETE_VALID.value
    assert stamp["outputs"]

    configure_run_lineage(
        pair,
        code_identity=_code(),
        parent_lifecycle_roots=(parents[0], "4" * 64),
    )
    assert (
        resolve_stage_state(
            done,
            [],
            [output],
            cfg=pair.config,
            stage="root_stability",
        )
        is CompletionState.COMPLETE_STALE
    )
