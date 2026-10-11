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

from farkle.analysis.h2h_authorization import cap_authorization_policy
from farkle.config import assign_config_sha, load_app_config
from farkle.orchestration.two_seed_pipeline import run_pipeline


@pytest.mark.integration
def test_raw_simulation_to_authenticated_report_via_normal_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Any,
) -> None:
    """Prove the tiny workflow, required proof inventories, and no-force resume."""

    started = time.perf_counter()
    # Task 5C adds two bounded preflights and their retained proof inventories.
    # Authentication also runs outside rng_preflight.run (health, reporting,
    # graph construction and byte-deep audit), so subtracting that function's
    # duration cannot isolate the previous 240-second workflow. Bound the full
    # oracle by that baseline plus the two declared 1800-second proof budgets.
    from farkle.analysis import rng_preflight

    preflight_seconds = 0.0
    original_preflight = rng_preflight.run

    def timed_preflight(*args, **kwargs):
        nonlocal preflight_seconds
        before = time.perf_counter()
        result = original_preflight(*args, **kwargs)
        duration = time.perf_counter() - before
        assert duration <= 1800.0
        preflight_seconds += duration
        return result

    monkeypatch.setattr(rng_preflight, "run", timed_preflight)
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
    cfg.head2head.total_game_cap = 23  # frozen oracle plans exactly 24 maximum attempts
    assign_config_sha(cfg)
    profile = oracle_game_profile()

    import sys

    prompts: list[str] = []

    def approve(prompt: str) -> str:
        prompts.append(prompt)
        assert len(prompts) == 1, "unchanged plan prompted more than once"
        return "y"

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", approve)
    with cap_authorization_policy(interactive=True):
        run_pipeline(
            cfg,
            seed_pair=ORACLE_ROOTS,
            oracle_game_profile=profile,
        )
    assert len(prompts) == 1
    contexts, pair_context = load_completed_pipeline_contexts(cfg)
    authorization = json.loads(pair_context.config.h2h_cap_authorization_path().read_text())
    assert authorization["authorization_source"] == "interactive"
    assert authorization["authorized_total_game_cap"] == 24
    assert authorization["configured_total_game_cap"] == 23
    assert cfg.head2head.total_game_cap == 23
    frozen = snapshot_frozen_family(pair_context.config)

    assert_root_pipeline_oracle(contexts)
    from tests.helpers.task7_oracle import assert_task7_products

    for context in contexts:
        assert_task7_products(context.config)
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
    from tests.helpers.task8_oracle import assert_task8_guide

    assert_task8_guide(contexts, pair_context)

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
    # Authenticated-v3 revalidates every output sidecar, immutable manifest,
    # source identity, and completion inventory on both the first run and the
    # no-force replay.
    record_property("required_preflight_runtime_seconds", round(preflight_seconds, 3))
    record_property("total_runtime_ceiling_seconds", 240 + 2 * 1800)
    assert elapsed <= 240.0 + 2 * 1800.0
