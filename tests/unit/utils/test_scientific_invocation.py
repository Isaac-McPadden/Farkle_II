"""Active invocation, protected byte reuse, and real subprocess boundaries."""

from __future__ import annotations

import json
import mmap
import multiprocessing
import os
import py_compile
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from importlib.metadata import PackagePath
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from farkle.config import AppConfig
from farkle.utils import scientific_runtime as runtime
from farkle.utils.authenticated_contract import identity_sha256
from farkle.utils.scientific_files import MeasuredFiles, ScientificDrift
from farkle.utils.scientific_identity import worker_dependencies
from farkle.utils.scientific_invocation import (
    ROOT,
    current,
    preflight_request,
    publication_boundary,
    scientific_invocation,
)


@pytest.fixture
def small_environment(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "numerical_environment",
        lambda names, **_: (identity_sha256(np.geterr()), dict.fromkeys(names, "b" * 64)),
    )


def test_launch_snapshot_survives_execution_and_documentation_changes(tmp_path, small_environment):
    cfg = AppConfig()
    with scientific_invocation([(cfg, "simulation")]):
        first = cfg.stage_config_sha("simulation")
        cfg.sim.n_jobs = 4
        (tmp_path / "README.md").write_text("operator notes")
        assert cfg.stage_config_sha("simulation") == first
    with (
        pytest.raises(ScientificDrift, match="sim.n_players_list"),
        scientific_invocation([(cfg, "simulation")]),
    ):
        assert cfg.stage_config_sha("simulation") == first
        cfg.sim.n_players_list.append(23)
        with pytest.raises(ScientificDrift, match="sim.n_players_list"):
            publication_boundary()
        cfg.sim.n_players_list.pop()
        # Sticky rejection survives restoration of the config.
        publication_boundary()


def test_standalone_pair_runner_pins_both_root_input_chains(tmp_path, small_environment):
    from farkle.analysis.stage_runner import StagePlanItem, StageRunContext, StageRunner
    from farkle.orchestration.run_contexts import RootPairRunContext, SeedRunContext

    def root(seed):
        cfg = AppConfig()
        cfg.sim.seed, cfg.sim.seed_list = seed, [seed]
        cfg.io.results_dir_prefix = tmp_path / str(seed)
        return SeedRunContext.from_config(cfg)

    pair = RootPairRunContext.from_root_contexts((root(11), root(22)), pair_root=tmp_path / "pair")

    def check(_cfg):
        pin = current()
        assert pin is not None
        for stage in ("metrics", "trueskill", "simulation"):
            for seed in (11, 22):
                assert f"reuse_v1/root/{stage}/{seed}" in pin.launch_digests
        assert "reuse_v1/pair/trueskill/11,22" in pin.launch_digests

    StageRunner.run(
        [StagePlanItem("candidate_freeze", check)],
        StageRunContext(pair.config, tmp_path / "manifest.jsonl", "pair"),
    )


def test_numerical_and_spawn_placement_mismatch(small_environment):
    cfg = AppConfig()
    with (
        pytest.raises(ScientificDrift, match="numerical_environment"),
        scientific_invocation([(cfg, "simulation")]),
        np.errstate(over="raise"),
    ):
        publication_boundary()
    with (
        pytest.raises(ScientificDrift, match="import placement"),
        scientific_invocation([(cfg, "simulation")]),
    ):
        expected = worker_dependencies()
        assert expected is not None
        replace(expected, placement="0" * 64).verify()


@pytest.mark.parametrize("module", ["farkle.utils.random", "synthetic.new_extension"])
def test_redirected_dynamic_import_blocks_publication(
    module, tmp_path, small_environment, monkeypatch
):
    with (
        pytest.raises(ScientificDrift, match=module),
        scientific_invocation([(AppConfig(), "simulation")]),
    ):
        observation = runtime._OBSERVATION.get()
        assert observation
        observation.origins["synthetic"] = tmp_path / "__init__.py"
        foreign = ModuleType(module)
        foreign.__file__ = str(tmp_path / "different.py")
        monkeypatch.setitem(sys.modules, module, foreign)
        publication_boundary()


def test_wheel_bytes_held_until_invocation_ends(tmp_path, monkeypatch):
    file = tmp_path / "numerical_fixture.py"
    file.write_bytes(b"first")
    monkeypatch.syspath_prepend(str(tmp_path))
    dist = SimpleNamespace(
        files=[PackagePath(file.name)], requires=[], version="1", locate_file=lambda p: tmp_path / p
    )
    monkeypatch.setattr(runtime.metadata, "distribution", lambda _: dist)
    files = MeasuredFiles()
    try:
        with runtime.numerical_observation(files):
            first = runtime._installed_build("fixture", {})
            if os.name == "nt":
                with pytest.raises(PermissionError):
                    file.write_bytes(b"other")
                with pytest.raises(PermissionError):
                    file.rename(file.with_suffix(".old"))
            assert runtime._installed_build("fixture", {}) == first
            assert len(files.digests) == 1
            files.boundary()
    finally:
        files.close()
    file.write_bytes(b"other")
    assert runtime._installed_build("fixture", {}) != first


def test_dynamic_import_ignores_timestamp_valid_old_bytecode(
    tmp_path, small_environment, monkeypatch
):
    module = tmp_path / "cache_fixture.py"
    module.write_text("value = 1")
    py_compile.compile(str(module), doraise=True)
    stamp = module.stat()
    module.write_text("value = 2")
    os.utime(module, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    monkeypatch.syspath_prepend(str(tmp_path))
    with scientific_invocation([(AppConfig(), "simulation")]):
        assert import_module("cache_fixture").value == 2
    sys.modules.pop("cache_fixture")


def test_preexisting_writer_prevents_measurement(tmp_path):
    path = tmp_path / "scientific.py"
    files = MeasuredFiles()
    try:
        with path.open("wb"), pytest.raises(ScientificDrift, match="scientific.py"):
            files.digest(path)
    finally:
        files.close()


def test_writable_mapping_survives_file_close_and_prevents_measurement(tmp_path):
    path = tmp_path / "scientific.py"
    path.write_bytes(b"value = 1")
    with path.open("r+b") as stream:
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_WRITE)
    files = MeasuredFiles()
    try:
        with mapping, pytest.raises(ScientificDrift, match="scientific.py"):
            files.digest(path)
    finally:
        files.close()


def test_drift_allows_failure_receipt_and_rejects_success(tmp_path, small_environment):
    from farkle.orchestration.two_seed_pipeline import _write_pipeline_health

    cfg = AppConfig()
    path = tmp_path / "pipeline_health.json"
    with (
        pytest.raises(ScientificDrift, match="sim.n_players_list"),
        scientific_invocation([(cfg, "simulation")]),
    ):
        cfg.sim.n_players_list.append(3)
        with pytest.raises(ScientificDrift, match="sim.n_players_list"):
            publication_boundary()
        _write_pipeline_health(path, {"status": "failed", "error": "scientific drift"})
        _write_pipeline_health(path, {"status": "complete_success"})
    assert json.loads(path.read_bytes())["status"] == "failed"


def test_root_invocation_receipt_binds_launch_to_existing_manifest(tmp_path, small_environment):
    from farkle.orchestration.execution_history import invocation_dir, publish_invocation
    from farkle.orchestration.run_contexts import SeedRunContext
    from farkle.utils.artifact_contract import sha256_file
    from farkle.utils.scientific_invocation import configured_producers

    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "seed"
    cfg.sim.seed_list = [cfg.sim.seed]
    context = SeedRunContext.from_config(cfg)
    manifest = tmp_path / "manifest.jsonl"
    with scientific_invocation(configured_producers(cfg)):
        assert cfg._code_identity is not None
        publish_invocation(
            context,
            run_id="pin-test",
            manifest_path=manifest,
            completion_paths=[],
            code_identity=cfg._code_identity,
            worker_counts={},
        )
        record = invocation_dir(context.results_root, "pin-test") / "invocation.json"
        assert json.loads(record.read_bytes())["scientific_launch_v1"]["producers"][
            "root/simulation"
        ] == cfg.stage_config_sha("simulation")
        assert json.loads(manifest.read_text().splitlines()[-1])[
            "invocation_sha256"
        ] == sha256_file(record)


def test_new_unclassified_source_blocks_next_publication(tmp_path, small_environment):
    with (
        pytest.raises(ScientificDrift, match="changed"),
        scientific_invocation([(AppConfig(), "simulation")]),
    ):
        pin = current()
        assert pin
        pin.files.watch_names(tmp_path)
        (tmp_path / "new_dynamic_helper.py").write_text("value = 1")
        publication_boundary()


def test_preflight_interpreter_attests_one_real_node(tmp_path):
    from farkle.cli.preflight_worker import main

    cfg = AppConfig()
    envelope = tmp_path / "launch.json"
    node = "tests/unit/utils/test_random_utils.py::test_purpose_registry_is_unique_and_unknown_namespaces_fail"
    with scientific_invocation([(cfg, "rng_diagnostics")]):
        request = preflight_request(cfg, [node])
        digest = identity_sha256(request)
        envelope.write_text(json.dumps(request))
        env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""}
        child = subprocess.run(
            [sys.executable, "-m", "farkle.cli.preflight_worker", str(envelope), digest],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        assert envelope.with_suffix(".verified").read_text() == digest
        incompatible = tmp_path / "incompatible.json"
        incompatible.write_text(json.dumps(request))
        other = subprocess.run(
            [sys.executable, "-m", "farkle.cli.preflight_worker", str(incompatible), digest],
            cwd=ROOT,
            env={**env, "NUMBA_OPT": "0"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert other.returncode != 0 and "dispatch_settings.NUMBA_OPT" in other.stderr
        assert not incompatible.with_suffix(".verified").exists()
        envelope.write_text(json.dumps({**request, "nodes": []}))
        with pytest.raises(ValueError, match="authentication failed"):
            main([str(envelope), digest])


def _check_spawned_request(expected, report):
    try:
        expected.verify()
    except ScientificDrift as exc:
        report.write_text(str(exc))
        return
    raise AssertionError("worker accepted a different scientific configuration")


def test_spawned_worker_rejects_changed_scientific_configuration(tmp_path):
    cfg = AppConfig()
    with scientific_invocation([(cfg, "simulation")]):
        expected = worker_dependencies()
        assert expected is not None
        other = deepcopy(cfg)
        other.sim.n_players_list.append(23)
        report = tmp_path / "worker_failure.txt"
        child = multiprocessing.get_context("spawn").Process(
            target=_check_spawned_request,
            args=(replace(expected, config=other), report),
        )
        child.start()
        try:
            child.join(timeout=120)
            assert child.exitcode == 0
            assert "sim.n_players_list" in report.read_text()
        finally:
            if child.is_alive():
                child.terminate()
                child.join()


def test_unowned_rating_keeper_selection_prevents_launch(tmp_path, small_environment):
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.sim.n_players_list = [2]
    keeper = cfg.n_dir(2) / "keepers_2.npy"
    keeper.parent.mkdir(parents=True)
    np.save(keeper, np.array([1, 2]))
    with (
        pytest.raises(ValueError, match="Unclassified external TrueSkill keeper"),
        scientific_invocation([(cfg, "trueskill")]),
    ):
        pytest.fail("unclassified rating input was admitted")


def test_preflight_keeps_target_worker_topology_across_revisions():
    from farkle.analysis.rng_preflight import _preflight_design

    plan = {
        "schema_version": 2,
        "code": {"commit": "old"},
        "scientific_dependencies_v1": "a" * 64,
        "environment": {"scratch": "old", "simulation_workers": 2},
        "target_config": {"io": {}},
    }
    other = {
        **plan,
        "code": {"commit": "new"},
        "target_config": {"io": {"temp_dir": "new"}},
        "environment": {"scratch": "new", "simulation_workers": 2},
    }
    assert _preflight_design(plan) == _preflight_design(other)
    other["environment"]["simulation_workers"] = 4
    assert _preflight_design(plan) != _preflight_design(other)
