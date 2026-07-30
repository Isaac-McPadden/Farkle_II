"""Real CLI-orchestration oracle from raw games through final reporting."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from tests.helpers.raw_simulation_oracle import (
    ORACLE_ROOTS,
    load_completed_pipeline_contexts,
    oracle_game_profile,
    snapshot_frozen_family,
    write_tiny_oracle_config,
)
from tests.helpers.simulation_to_report_oracle import (
    assert_canonical_artifact_inventory,
    assert_pipeline_health_and_simulation_lifecycle,
    assert_report_oracle,
    snapshot_stable_workflow_files,
)
from tests.helpers.tournament_analysis_oracle import (
    assert_authenticated_analysis_graph,
    assert_pair_h2h_oracle,
    assert_root_pipeline_oracle,
)

from farkle.config import assign_config_sha, load_app_config
from farkle.orchestration.two_seed_pipeline import run_pipeline


@pytest.mark.integration
def test_raw_simulation_to_authenticated_report_via_normal_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Any,
) -> None:
    """Prove the complete tiny workflow and its no-force resume in routine-CI time."""

    started = time.perf_counter()
    known_bad_root = (
        Path(__file__).resolve().parents[2] / "data" / "results_seed_pair_32_33"
    ).resolve()
    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any):
        resolved = path.resolve()
        if resolved == known_bad_root or known_bad_root in resolved.parents:
            raise AssertionError(f"final oracle accessed known-bad fast-run tree: {resolved}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    config_path = write_tiny_oracle_config(tmp_path)
    cfg = load_app_config(config_path, seed_list_len=2)
    assign_config_sha(cfg)
    profile = oracle_game_profile()

    run_pipeline(
        cfg,
        seed_pair=ORACLE_ROOTS,
        oracle_game_profile=profile,
    )
    contexts, pair_context = load_completed_pipeline_contexts(cfg)
    frozen = snapshot_frozen_family(pair_context.config)

    assert_root_pipeline_oracle(contexts)
    assert_pair_h2h_oracle(
        pair_context,
        frozen,
        profile,
        include_reporting=True,
    )
    assert_authenticated_analysis_graph(
        contexts,
        pair_context,
        profile_sha256=profile.sha256,
        include_reporting=True,
    )
    assert_canonical_artifact_inventory(contexts, pair_context)
    assert_pipeline_health_and_simulation_lifecycle(
        cfg,
        contexts,
        pair_context,
        profile,
    )
    assert_report_oracle(pair_context)

    first_hashes = snapshot_stable_workflow_files(contexts, pair_context)
    first_block_count = len(list(pair_context.config.h2h_block_results_dir().glob("*.parquet")))
    run_pipeline(
        cfg,
        seed_pair=ORACLE_ROOTS,
        oracle_game_profile=profile,
    )
    assert snapshot_stable_workflow_files(contexts, pair_context) == first_hashes
    assert len(list(pair_context.config.h2h_block_results_dir().glob("*.parquet"))) == (
        first_block_count
    )
    assert_pipeline_health_and_simulation_lifecycle(
        cfg,
        contexts,
        pair_context,
        profile,
    )
    events = [
        json.loads(line)
        for line in (pair_context.pair_root / "two_seed_pipeline_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert events[-1]["event"] == "run_end"
    assert events[-1]["status"] == "complete_success"

    elapsed = time.perf_counter() - started
    record_property("raw_simulation_to_report_runtime_seconds", round(elapsed, 3))
    print(f"raw simulation-to-report oracle runtime: {elapsed:.3f}s")
    assert elapsed <= 90.0
