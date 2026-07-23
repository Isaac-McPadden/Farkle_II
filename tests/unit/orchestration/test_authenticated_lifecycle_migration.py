from __future__ import annotations

import json
from pathlib import Path

import pytest

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import (
    AppConfig,
    IOConfig,
    SimConfig,
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
from farkle.utils.authenticated_contract import CodeIdentity, CodeIdentityPolicy
from farkle.utils.stage_completion import (
    CompletionState,
    resolve_stage_state,
    stage_done_path,
    write_stage_done,
)


def _code(commit: str = "a" * 40) -> CodeIdentity:
    return CodeIdentity(
        commit=commit,
        policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
        state="clean",
        dirty_fingerprint_sha256=None,
    )


def _cfg(tmp_path: Path, *, root: int = 11) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / f"root_{root}"),
        sim=SimConfig(seed=root, seed_list=[root], n_players_list=[2]),
    )
    cfg.screening.resolution_delta = 0.9
    cfg.batching.target_batches = 2
    cfg.batching.min_shuffles_per_batch = 1
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    assign_config_sha(cfg)
    cfg._code_identity = _code()
    cfg._run_lineage_sha256 = "1" * 64
    return cfg


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
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True)
    strategy_manifest = cfg.strategy_manifest_root_path()
    strategy_manifest.parent.mkdir(parents=True, exist_ok=True)
    strategy_manifest.write_bytes(b"strategy-grid-v1")
    (n_dir / "simulation_workload_plan.json").write_text("{}", encoding="utf-8")
    output = n_dir / "2p_checkpoint.pkl"
    output.write_bytes(b"output-v1")
    output.with_name(f"{output.name}.sidecar.json").write_text("sidecar-v1", encoding="utf-8")
    runner.write_simulation_done(
        cfg,
        2,
        num_shuffles=2,
        shuffles_per_batch=1,
        n_strategies=2,
        outputs=[output],
    )
    assert runner.simulation_is_complete(cfg, 2)

    if mutation == "grid":
        cfg.sim.score_thresholds = [999]
    elif mutation == "strategy_manifest":
        strategy_manifest.write_bytes(b"strategy-grid-v2")
    elif mutation == "output":
        output.write_bytes(b"output-v2")
    elif mutation == "sidecar":
        output.with_name(f"{output.name}.sidecar.json").write_text("sidecar-v2", encoding="utf-8")
    elif mutation == "code":
        cfg._code_identity = _code("b" * 40)
    else:
        monkeypatch.setattr("farkle.utils.schema_helpers.TOURNAMENT_METHOD_VERSION", 99)

    assert not runner.simulation_is_complete(cfg, 2)


@pytest.mark.parametrize(
    "mutation",
    ("input", "output", "sidecar", "stage_config", "code", "method"),
)
def test_root_stage_completion_mutation_matrix(tmp_path: Path, mutation: str) -> None:
    cfg = _cfg(tmp_path)
    source = tmp_path / "source.parquet"
    output = tmp_path / "output.parquet"
    sidecar = output.with_name(f"{output.name}.sidecar.json")
    source.write_bytes(b"input-v1")
    output.write_bytes(b"output-v1")
    sidecar.write_text("sidecar-v1", encoding="utf-8")
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
        sidecar.write_text("sidecar-v2", encoding="utf-8")
    elif mutation == "stage_config":
        cfg.screening.resolution_delta = 0.8
    elif mutation == "code":
        cfg._code_identity = _code("b" * 40)
    else:
        cfg.artifact_contract.estimand_version += 1

    assert (
        resolve_stage_state(done, [source], [output], cfg=cfg, stage="metrics")
        is CompletionState.COMPLETE_STALE
    )


def test_runtime_only_change_does_not_stale_root_stage(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.write_text("input", encoding="utf-8")
    output.write_text("output", encoding="utf-8")
    done = stage_done_path(cfg.metrics_stage_dir, "metrics")
    write_stage_done(done, inputs=[source], outputs=[output], cfg=cfg, stage="metrics")

    cfg.analysis.n_jobs = 99
    assign_config_sha(cfg)

    assert (
        resolve_stage_state(done, [source], [output], cfg=cfg, stage="metrics")
        is CompletionState.COMPLETE_VALID
    )


def test_pair_public_config_and_context_round_trip_bind_parent_and_stage_hashes(
    tmp_path: Path,
) -> None:
    roots = tuple(SeedRunContext.from_config(_cfg(tmp_path, root=root)) for root in (11, 22))
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

    source = tmp_path / "root-input"
    source.write_text("root", encoding="utf-8")
    output = pair.config.root_discrepancies_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("pair-output", encoding="utf-8")
    done = stage_done_path(pair.config.stage_dir("root_stability"), "root_stability")
    write_stage_done(
        done,
        inputs=[source],
        outputs=[output],
        cfg=pair.config,
        stage="root_stability",
    )
    stamp = json.loads(done.read_text(encoding="utf-8"))
    assert stamp["config_sha"] == pair.config.config_sha
    assert stamp["run_lineage_sha256"] == persisted_context["run_lineage_sha256"]

    configure_run_lineage(
        pair,
        code_identity=_code(),
        parent_lifecycle_roots=(parents[0], "4" * 64),
    )
    assert (
        resolve_stage_state(
            done,
            [source],
            [output],
            cfg=pair.config,
            stage="root_stability",
        )
        is CompletionState.COMPLETE_STALE
    )
