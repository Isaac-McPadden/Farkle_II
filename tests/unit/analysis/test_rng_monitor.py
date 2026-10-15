from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.unit.analysis.test_outcome_precision import _fixture

from farkle import analysis
from farkle.analysis import rng_monitor as monitor
from farkle.analysis import rng_preflight
from farkle.analysis.rng_evidence import finish, inventory
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.stage_completion import CompletionState, resolve_stage_state, stage_done_path


def test_capability_unit_preserves_every_check_and_authenticates_bytes(tmp_path):
    from tests.helpers.artifact_sidecars import make_authenticated_v3_config

    from farkle.analysis.rng_evidence import read, reusable

    cfg = make_authenticated_v3_config(tmp_path)
    recipe = read(rng_preflight.ROOT / "configs/validation/task5b_capability.json")
    seed = recipe["fixture_root_seeds"][0]
    _, payload = rng_preflight._capability_unit((seed, recipe))
    path = cfg.rng_preflight_dir() / "capability" / f"{seed}.parquet"
    table = pa.Table.from_pylist(payload["fixtures"], schema=rng_preflight.capability_schema())
    finish(cfg, path, table, [])
    from farkle.utils.authenticated_contract import load_authenticated_sidecar

    assert load_authenticated_sidecar(path).method_contract.multiplicity == "descriptive_no_fwer"
    assert pq.read_table(path).to_pylist() == payload["fixtures"]
    assert table.num_rows == 12
    assert sum(len(row["checks"]) for row in payload["fixtures"]) == 2304
    assert reusable(cfg, path, [])
    with path.open("r+b") as handle:
        handle.seek(10)
        original = handle.read(1)
        handle.seek(10)
        handle.write(bytes([original[0] ^ 1]))
    assert not reusable(cfg, path, [])


def test_warning_surfaces_without_changing_game_targets(tmp_path, monkeypatch, caplog):
    from tests.helpers.artifact_sidecars import make_authenticated_v3_config
    from tests.helpers.outcome_fixture import publish_outcome_source
    from tests.unit.analysis.test_outcome_distributions import _row

    from farkle.analysis.outcome_publication import build_outcome_distributions
    from farkle.config import assign_config_sha

    cfg = make_authenticated_v3_config(tmp_path)
    cfg.sim.row_dir = Path("rows")
    cfg.analysis.n_jobs = 1
    cfg.ingest.n_jobs = 1
    assign_config_sha(cfg)
    rows = [
        _row([1000, 900] if b < 15 else [900, 1000], [10, 20], shuffle=b, width=1)
        for b in range(30)
    ]
    publish_outcome_source(cfg, rows, roster=(10, 20), width=1)
    build_outcome_distributions(cfg, 2)
    before = cfg.simulation_workload_plan_path(2).read_bytes()
    _proof(cfg, monkeypatch)
    monitor.run(cfg, exhaustive=False)
    coverage = json.loads(cfg.rng_monitor_coverage_path().read_text())
    assert coverage["status"] == "investigate"
    assert coverage["statistical_warning_count"] > 0
    assert "requests investigation" in caplog.text
    assert cfg.simulation_workload_plan_path(2).read_bytes() == before
    assert (
        resolve_stage_state(
            stage_done_path(cfg.rng_stage_dir, "rng_diagnostics"),
            [],
            [],
            cfg=cfg,
            stage="rng_diagnostics",
        )
        == CompletionState.COMPLETE_VALID
    )


def test_enabled_exhaustive_preserves_cap_state(tmp_path, monkeypatch):
    from farkle.analysis.rng_evidence import metadata, publish
    from farkle.utils.artifacts import write_parquet_artifact_atomic
    from farkle.utils.stage_completion import write_stage_done

    cfg = _fixture(tmp_path)
    _proof(cfg, monkeypatch)

    def capped(cfg, **kwargs):
        parquet = cfg.rng_input_path("rng_diagnostics.parquet")
        summary = cfg.rng_input_path("rng_diagnostics_summary.json")
        table = pa.table({"count": [1]})
        write_parquet_artifact_atomic(table, parquet, sidecar=metadata(cfg, parquet, [], ["count"]))
        publish(cfg, summary, {"completeness_status": "blocked_by_cap"}, [])
        write_stage_done(
            stage_done_path(cfg.rng_stage_dir, kwargs["_completion_name"]),
            inputs=[],
            outputs=[parquet, summary],
            cfg=cfg,
            stage="rng_diagnostics",
            sidecar_artifacts=[parquet, summary],
            status="blocked_by_cap",
        )

    monkeypatch.setattr(monitor.rng_diagnostics, "run", capped)
    monitor.run(cfg, exhaustive=True)
    coverage = json.loads(cfg.rng_monitor_coverage_path().read_text())
    assert coverage["exhaustive_status"] == "blocked_by_cap"
    assert coverage["coverage_complete"] is False
    assert (
        resolve_stage_state(
            stage_done_path(cfg.rng_stage_dir, "rng_diagnostics"),
            [],
            [],
            cfg=cfg,
            stage="rng_diagnostics",
        )
        == CompletionState.BLOCKED_BY_CAP
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_real_exhaustive_and_required_evidence_complete_and_resume(tmp_path, monkeypatch, workers):
    from farkle.analysis import combine
    from farkle.config import assign_config_sha
    from farkle.utils.authenticated_contract import load_authenticated_sidecar

    cfg = _fixture(tmp_path, workers=workers)
    cfg.analysis.rng_diagnostic_partitions = 2
    assign_config_sha(cfg)
    combine.run(cfg)
    _proof(cfg, monkeypatch)
    monitor.run(cfg, exhaustive=True)
    stamp = stage_done_path(cfg.rng_stage_dir, "rng_diagnostics")
    outputs = [
        cfg.rng_preflight_evidence_path(),
        cfg.rng_monitor_coverage_path(),
        cfg.rng_production_monitor_path(),
        cfg.rng_input_path("rng_diagnostics.parquet"),
        cfg.rng_input_path("rng_diagnostics_summary.json"),
    ]
    before = [(sha256_file(p), p.stat().st_mtime_ns) for p in [stamp, *outputs]]
    evidence = load_authenticated_sidecar(outputs[0])
    exhaustive = load_authenticated_sidecar(outputs[-1])
    assert evidence.method_contract.method_version == 1
    assert exhaustive.method_contract.method_version == 4
    assert evidence.versions == exhaustive.versions
    assert (
        resolve_stage_state(stamp, [], [], cfg=cfg, stage="rng_diagnostics")
        == CompletionState.COMPLETE_VALID
    )
    monitor.run(cfg, exhaustive=True)
    assert [(sha256_file(p), p.stat().st_mtime_ns) for p in [stamp, *outputs]] == before
    with outputs[-1].open("ab") as handle:
        handle.write(b"corrupt")
    assert (
        resolve_stage_state(stamp, [], [], cfg=cfg, stage="rng_diagnostics")
        == CompletionState.COMPLETE_STALE
    )


def test_parent_completion_binds_retained_proof_units(tmp_path, monkeypatch):
    cfg = _fixture(tmp_path)
    proof = _proof(cfg, monkeypatch)
    monitor.run(cfg, exhaustive=False)
    stamp = stage_done_path(cfg.rng_stage_dir, "rng_diagnostics")
    key = monitor.freshness_key(cfg, False, None)
    assert (
        resolve_stage_state(stamp, [], [], cfg=cfg, stage="rng_diagnostics", freshness_key=key)
        == CompletionState.COMPLETE_VALID
    )
    (proof.parent / "plan.json").unlink()
    assert (
        resolve_stage_state(stamp, [], [], cfg=cfg, stage="rng_diagnostics", freshness_key=key)
        != CompletionState.COMPLETE_VALID
    )


def _proof(cfg, monkeypatch):
    """Explicit test double for expensive preflight; never claims real capability."""
    plan = cfg.rng_preflight_dir() / "plan.json"
    finish(
        cfg,
        plan,
        {"rng": {"scheme": 2}, "recipe": {"checks": []}, "required_source_paths": ["plan.json"]},
        [],
    )
    proof = cfg.rng_preflight_dir() / "publication.json"
    result = {
        "status": "complete_valid",
        "code_identity": "a" * 64,
        "workload_identity": "b" * 64,
        "environment": {"schema_version": 1},
        "covered_ks": [2],
        "covered_workers": [1],
        "sources": inventory([plan], plan.parent),
    }
    finish(cfg, proof, result, [plan])
    monkeypatch.setattr(rng_preflight, "run", lambda *a, **kw: (proof, result))
    return proof


def test_disabled_exhaustive_keeps_required_stage_and_nonmutating_paths(tmp_path):
    cfg = _fixture(tmp_path)
    cfg.analysis.disable_rng_diagnostics = True
    assert "rng_diagnostics" in [item.name for item in analysis.build_root_stage_plan(cfg)]
    before = set(tmp_path.rglob("*"))
    cfg.rng_preflight_evidence_path()
    cfg.rng_production_monitor_path()
    cfg.rng_monitor_coverage_path()
    assert set(tmp_path.rglob("*")) == before


def test_monitor_omission_exact_reuse_and_missing_required_output(tmp_path, monkeypatch):
    cfg = _fixture(tmp_path)
    _proof(cfg, monkeypatch)
    monkeypatch.setattr(
        monitor.rng_diagnostics, "run", lambda *a, **kw: pytest.fail("exhaustive ran")
    )
    monitor.run(cfg, exhaustive=False)
    table = pq.read_table(cfg.rng_production_monitor_path())
    assert table.num_rows == 24
    assert set(table["result"].to_pylist()) == {"insufficient_evidence"}
    coverage = json.loads(cfg.rng_monitor_coverage_path().read_text())
    assert coverage["exhaustive_status"] == "omitted_by_design"
    paths = [
        cfg.rng_monitor_coverage_path(),
        cfg.rng_production_monitor_path(),
        cfg.rng_preflight_evidence_path(),
    ]
    before = [(sha256_file(p), p.stat().st_mtime_ns) for p in paths]
    monitor.run(cfg, exhaustive=False)
    assert [(sha256_file(p), p.stat().st_mtime_ns) for p in paths] == before
    cfg.rng_production_monitor_path().unlink()
    stamp = stage_done_path(cfg.rng_stage_dir, "rng_diagnostics")
    assert (
        resolve_stage_state(stamp, [], paths, cfg=cfg, stage="rng_diagnostics")
        != CompletionState.COMPLETE_VALID
    )
    monitor.run(cfg, exhaustive=False)
    assert [sha256_file(p) for p in paths] == [value[0] for value in before]


def test_missing_proof_unit_blocks_success(tmp_path, monkeypatch):
    cfg = _fixture(tmp_path)
    proof = _proof(cfg, monkeypatch)
    (proof.parent / "plan.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError), match=".*"):
        monitor.run(cfg, exhaustive=False)
    assert not stage_done_path(cfg.rng_stage_dir, "rng_diagnostics").exists()


def test_fixed_sentinel_selection():
    assert monitor.sentinel_ids([2, 8, 90]) == [2, 8, 90]
    assert monitor.sentinel_ids(list(range(120))) == [i * 119 // 15 for i in range(16)]
    with pytest.raises(ValueError):
        monitor.sentinel_ids([2, 1])
