from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import pytest
from scripts import benchmark_task4c_simulation_execution as benchmark

from farkle.simulation import runner


def test_fixture_worker_count_is_execution_only(tmp_path: Path) -> None:
    prefix = tmp_path / "same"
    one = benchmark._fixture_config(prefix=prefix, workers=1, row_output=False)
    twelve = benchmark._fixture_config(prefix=prefix, workers=12, row_output=False)

    assert one.sim.n_jobs == 1
    assert twelve.sim.n_jobs == 12
    assert one.config_sha == twelve.config_sha
    assert one.sim.n_players_list == twelve.sim.n_players_list == [2]
    assert one.sim.mp_start_method == twelve.sim.mp_start_method == "spawn"
    assert one.orchestration.parallel_seeds is False


def test_cpu_fixture_uses_the_unchanged_fast_configuration_grid(tmp_path: Path) -> None:
    cfg = benchmark._fixture_config(
        prefix=tmp_path / "production_grid",
        workers=12,
        row_output=False,
        production_grid=True,
    )

    _strategies, grid_size, used_custom = runner._resolve_strategies(cfg, None)
    assert grid_size == 80
    assert used_custom is True


def test_canonical_digest_excludes_operational_pickle_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "results"
    n_dir = root / "2_players"
    n_dir.mkdir(parents=True)
    (n_dir / "2p_checkpoint.parquet").write_bytes(b"canonical")
    checkpoint = n_dir / "2p_checkpoint.pkl"
    checkpoint.write_bytes(b"operational-a")
    first = benchmark._canonical_bundle_sha256(root)
    checkpoint.write_bytes(b"operational-b")
    assert benchmark._canonical_bundle_sha256(root) == first


def test_durable_row_identities_exclude_unmanifested_shards(tmp_path: Path) -> None:
    row_dir = tmp_path / "rows"
    row_dir.mkdir()
    durable = row_dir / "rows_000.parquet"
    incomplete = row_dir / "rows_001.parquet"
    durable.write_bytes(b"durable")
    incomplete.write_bytes(b"not-yet-committed")
    (row_dir / "manifest.jsonl").write_text(
        json.dumps({"path": durable.name}) + "\n",
        encoding="utf-8",
    )

    identities = benchmark._durable_row_identities(row_dir)
    assert set(identities) == {durable.name}


def test_owned_root_requires_prefix_and_marker_for_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="must start"):
        benchmark._safe_owned_root(tmp_path / "unowned")

    root = tmp_path / "data" / "farkle-task4c-unit-owned-root"
    prepared = benchmark._prepare_owned_root(root, force=False)
    assert (prepared / benchmark.OWNERSHIP_MARKER).exists()
    assert benchmark._prepare_owned_root(root, force=True) == prepared


def test_logical_checkpoint_digest_ignores_mapping_insertion_order(tmp_path: Path) -> None:
    first = tmp_path / "first" / "2_players"
    second = tmp_path / "second" / "2_players"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    common = {
        "outcome_counts": {"games_attempted": 3, "games_completed": 3},
        "meta": {
            "completed_shuffle_indices": [0, 1, 2],
            "completed_process_block_indices": [0, 1, 2],
        },
    }
    (first / "2p_checkpoint.pkl").write_bytes(
        pickle.dumps({"win_totals": Counter({"a": 1, "b": 2}), **common})
    )
    (second / "2p_checkpoint.pkl").write_bytes(
        pickle.dumps({"win_totals": Counter({"b": 2, "a": 1}), **common})
    )

    assert benchmark._logical_checkpoint_sha256(first.parent) == (
        benchmark._logical_checkpoint_sha256(second.parent)
    )
