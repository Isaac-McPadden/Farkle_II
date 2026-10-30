"""First-release Task 10: real protected CLI, operational stop and exact resume."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml
from tests.helpers.raw_simulation_oracle import (
    load_completed_pipeline_contexts,
    write_tiny_oracle_config,
)
from tests.helpers.simulation_to_report_oracle import snapshot_stable_workflow_files
from tests.helpers.task7_oracle import assert_task7_products

from farkle.analysis.analysis_guide import render_guide_markdown, validate_guide
from farkle.config import load_app_config
from farkle.utils.artifact_contract import sha256_file, validate_artifact_sidecar
from farkle.utils.writer import atomic_path


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "win32", reason="real Windows Job Object qualification")
def test_windows_launcher_cross_volume_cap_stop_and_resume(
    tmp_path: Path, record_property: Any
) -> None:
    """Use normal rules and explicit consent; recover a missing reporting product."""

    started = time.perf_counter()
    repo = Path(__file__).resolve().parents[2]
    config_path = write_tiny_oracle_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    production = yaml.safe_load((repo / "configs/farkle_mega_config.yaml").read_text())
    scratch = Path(tempfile.mkdtemp(prefix="farkle-task10-scratch-"))
    payload["io"]["temp_dir"] = str(scratch)
    payload["resources"] = production["resources"]
    for section in ("sim", "analysis", "ingest", "head2head"):
        payload[section]["n_jobs"] = 2
    payload["sim"]["mp_start_method"] = "spawn"
    payload["analysis"]["mp_start_method"] = "spawn"
    payload["head2head"]["total_game_cap"] = 23
    payload["head2head"]["delta_equivalence"] = 0.03
    payload["batching"] = {"target_batches": 20, "min_shuffles_per_batch": 30}
    payload["profile"] = {
        "purpose": "integration",
    }
    # No oracle rule/coordinate overrides: the public launcher plays normal games.
    with atomic_path(str(config_path)) as staged:
        Path(staged).write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    cfg = load_app_config(config_path, seed_list_len=2)
    cfg.validate_statistical_contract(require_two_roots=True)
    pair_root = Path(str(payload["io"]["results_dir_prefix"]) + "_seed_pair_11_22")
    base = [sys.executable, "-m", "farkle", "--config", str(config_path)]
    phases: list[dict[str, Any]] = []
    env = os.environ.copy()
    # The root venv is editable-installed against the user's worktree. Explicitly
    # make the stable snapshot first on sys.path in every launcher/spawn child.
    env["PYTHONPATH"] = str(repo / "src")

    def launch(name: str, extra: list[str], expected: int) -> None:
        command = [*base, *extra, "two-seed-pipeline"]
        log_path = tmp_path / f"{name}.log"
        before = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3840,
                check=False,
            )
        phases.append(
            {
                "phase": name,
                "command": command,
                "exit_code": result.returncode,
                "seconds": time.perf_counter() - before,
                "log": str(log_path),
                "log_sha256": sha256_file(log_path),
            }
        )
        # Keep phase evidence even if a later assertion/launcher interrupts the
        # long qualification. Canonical producer recovery remains pipeline-owned.
        with atomic_path(str(tmp_path / "task10_launcher_progress.json")) as staged:
            Path(staged).write_text(
                json.dumps(
                    {"config": str(config_path), "pair_root": str(pair_root), "phases": phases},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        assert result.returncode == expected, log_path.read_text(encoding="utf-8")[-12000:]

    launch("cap_stop", [], 3)
    health_path = pair_root / "pipeline_health.json"
    blocked = json.loads(health_path.read_text(encoding="utf-8"))
    assert blocked["status"] == "blocked_by_cap"
    stopped_telemetry = json.loads((pair_root / "pipeline_telemetry.json").read_text())
    assert stopped_telemetry["status"] == "blocked_by_cap"
    assert stopped_telemetry["active_scope_count"] == 0
    power_path = next((pair_root / "seed_pair_analysis").rglob("power_plan.json"))
    validate_artifact_sidecar(power_path)
    plan = json.loads(power_path.read_text(encoding="utf-8"))
    assert plan["maximum_total_attempts"] == 24
    _root_contexts, stopped_pair = load_completed_pipeline_contexts(cfg)
    assert not list(stopped_pair.config.h2h_block_results_dir().glob("*.parquet"))
    root_units = {
        str(path): (sha256_file(path), path.stat().st_mtime_ns)
        for path in pair_root.rglob("*.parquet")
        if "seed_pair_analysis" not in path.parts
    }
    assert root_units
    token = f"{plan['schedule_hash']}:{plan['maximum_total_attempts']}"
    launch("authorized_resume", ["--h2h-authorize", token], 0)
    assert all(
        (sha256_file(Path(path)), Path(path).stat().st_mtime_ns) == identity
        for path, identity in root_units.items()
    )
    contexts, pair = load_completed_pipeline_contexts(cfg)
    authorization = json.loads(pair.config.h2h_cap_authorization_path().read_text())
    assert authorization["authorization_source"] == "explicit_override"
    assert authorization["authorized_total_game_cap"] == 24
    assert authorization["configured_total_game_cap"] == 23
    assert cfg.head2head.total_game_cap == 23
    for context in contexts:
        run_context = json.loads(context.run_context_path.read_text(encoding="utf-8"))
        boundary = run_context["execution_controls"]["os_memory_boundary"]
        assert boundary["backend"] == "windows_job"
        assert boundary["enforced"] is True
        assert boundary["strict_required"] is True
        assert boundary["fallback_used"] is False
        assert_task7_products(context.config)
        for k in context.config.sim.n_players_list:
            support = pq.read_table(context.config.outcome_support_path(k)).to_pylist()
            inventory = pq.read_table(context.config.outcome_batch_inventory_path(k)).to_pylist()
            assert sum(r["attempted"] for r in support if r["seat"] == 0) == k * sum(
                r["observed_games"] for r in inventory
            )
        coverage = json.loads(context.config.rng_monitor_coverage_path().read_text())
        assert coverage["exhaustive_status"] == "omitted_by_design"
        assert context.config.rng_preflight_evidence_path().exists()

    def check_report_and_audit() -> dict[str, Any]:
        guide_path = pair.config.analysis_guide_json_path()
        validate_artifact_sidecar(guide_path)
        validate_artifact_sidecar(pair.config.analysis_guide_markdown_path())
        guide = json.loads(guide_path.read_text(encoding="utf-8"))
        validate_guide(guide)
        assert pair.config.analysis_guide_markdown_path().read_text(
            encoding="utf-8"
        ) == render_guide_markdown(guide)
        health = json.loads(health_path.read_text(encoding="utf-8"))
        assert health["status"] == "complete_success"
        assert health["profile"]["release_status"] == "passed"
        assert health["profile"]["release_eligible"] is False
        terminal = json.loads((pair_root / "pipeline_telemetry.json").read_text())
        assert terminal["telemetry_kind"] == "terminal"
        assert terminal["status"] == "complete_success"
        assert terminal["active_scopes"] == [] and terminal["active_scope_count"] == 0
        audit = health["release_audit"]
        assert audit["status"] == "passed" and not audit["failures"]
        assert audit["top_level_invocations"] == 1
        assert len(audit["internal_roots"]) == 3
        assert audit["release_eligible"] is False
        assert audit["production_eligible"] is False
        assert audit["code_identity"]["state"] == "clean"
        return health

    health = check_report_and_audit()
    stable = snapshot_stable_workflow_files(contexts, pair)
    launch("no_force_replay", [], 0)
    assert snapshot_stable_workflow_files(contexts, pair) == stable
    check_report_and_audit()
    # A lost final product cannot be accepted from its old completion. Recover
    # just reporting while preserving all producer and H2H durable Parquets.
    markdown = pair.config.analysis_guide_markdown_path()
    json_identity = (
        sha256_file(pair.config.analysis_guide_json_path()),
        pair.config.analysis_guide_json_path().stat().st_mtime_ns,
    )
    durable = {str(p): (sha256_file(p), p.stat().st_mtime_ns) for p in pair_root.rglob("*.parquet")}
    markdown.unlink()
    launch("missing_markdown_resume", [], 0)
    assert (
        sha256_file(pair.config.analysis_guide_json_path()),
        pair.config.analysis_guide_json_path().stat().st_mtime_ns,
    ) == json_identity
    assert all(
        (sha256_file(Path(p)), Path(p).stat().st_mtime_ns) == identity
        for p, identity in durable.items()
    )
    check_report_and_audit()
    telemetry = json.loads((pair_root / "pipeline_telemetry.json").read_text())
    evidence = {
        "schema_version": 1,
        "phases": phases,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "pair_root": str(pair_root),
        "scratch": str(scratch),
        "different_drive_letters": scratch.anchor.lower() != pair_root.anchor.lower(),
        "initial_complete_health": health,
        "final_telemetry": telemetry,
        "preserved_root_parquets": len(root_units),
        "preserved_recovery_parquets": len(durable),
        "no_force_stable_files": len(stable),
        "durable_bytes": sum(p.stat().st_size for p in pair_root.rglob("*") if p.is_file()),
        "scratch_bytes": sum(p.stat().st_size for p in scratch.rglob("*") if p.is_file()),
        "runtime_seconds": time.perf_counter() - started,
    }
    evidence_path = tmp_path / "task10_launcher_evidence.json"
    with atomic_path(str(evidence_path)) as staged:
        Path(staged).write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    record_property("qualification_evidence", str(evidence_path))
    record_property("qualification_runtime_seconds", round(evidence["runtime_seconds"], 3))
