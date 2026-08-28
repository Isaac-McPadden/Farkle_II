from __future__ import annotations

import json
import stat
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts import benchmark_task7a_artifact_density as benchmark


def test_owned_root_requires_exact_name_and_preserves_mismatched_root(tmp_path: Path) -> None:
    settings = {"case": 1}
    with pytest.raises(ValueError, match="exact name"):
        benchmark.prepare_owned_root(tmp_path / "farkle-task7a-other", settings)

    root = tmp_path / benchmark.SAFE_ROOT_NAME
    benchmark.prepare_owned_root(root, settings)
    marker = root / benchmark.MARKER_NAME
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["settings"] = {"case": 2}
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="marker or settings mismatch"):
        benchmark.prepare_owned_root(root, settings, force=True)
    assert root.is_dir()


def test_force_cleans_only_owned_fixture_child(tmp_path: Path) -> None:
    root = tmp_path / benchmark.SAFE_ROOT_NAME
    settings = {"case": 1}
    benchmark.prepare_owned_root(root, settings)
    fixture = root / "fixtures" / "case"
    fixture.mkdir(parents=True)
    (fixture / "payload.bin").write_bytes(b"owned")
    (fixture / "payload.bin").chmod(stat.S_IREAD)
    unrelated = root / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")

    benchmark.prepare_owned_root(root, settings, force=True)

    assert not (root / "fixtures").exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_inventory_measures_footer_typed_empty_and_jsonl_structure(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    table = benchmark._fixture_table(8, 2)
    pq.write_table(table, root / "rows.parquet")
    empty_schema = pa.schema([pa.field("strategy_id", pa.int32(), nullable=False)])
    pq.write_table(
        pa.Table.from_arrays([pa.array([], type=pa.int32())], schema=empty_schema),
        root / "empty.parquet",
    )
    (root / "valid.jsonl").write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    (root / "pretty.jsonl").write_text('{\n  "a": 1\n}\n', encoding="utf-8")

    inventory = benchmark.inventory_tree(root, detailed_parquets=True)

    assert inventory["read_only"] is True
    assert inventory["parquet"]["files"] == 2
    assert inventory["parquet"]["footer_metadata_bytes"] > 0
    assert inventory["parquet"]["zero_row_files"][0]["schema"] == [
        {"name": "strategy_id", "type": "int32", "nullable": False}
    ]
    assert len(inventory["json"]["invalid_jsonl"]) == 1
    assert inventory["json"]["invalid_jsonl"][0]["path"] == "pretty.jsonl"


def test_fixture_padding_preserves_active_columns_and_types() -> None:
    native = benchmark._fixture_table(256, 2)
    padded = benchmark._fixture_table(256, 2, pad_to=12)

    assert native.num_rows == padded.num_rows == 256
    assert padded.schema == benchmark.expected_schema_for(12)
    assert padded["P1_strategy"].combine_chunks().equals(native["P1_strategy"].combine_chunks())
    assert padded["P12_strategy"].null_count == 256


def test_run_is_resumable_without_remeasurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / benchmark.SAFE_ROOT_NAME
    current = tmp_path / "current"
    historical = tmp_path / "historical"
    current.mkdir()
    historical.mkdir()
    monkeypatch.setattr(
        benchmark,
        "_layout_benchmarks",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        benchmark,
        "_authentication_benchmark",
        lambda *_args, **_kwargs: [],
    )
    payload = benchmark.run(
        root,
        current_root=current,
        historical_root=historical,
        rows=1,
        repetitions=3,
    )

    monkeypatch.setattr(
        benchmark,
        "inventory_tree",
        lambda *_args, **_kwargs: pytest.fail("resume recomputed inventory"),
    )
    resumed = benchmark.run(
        root,
        current_root=current,
        historical_root=historical,
        rows=1,
        repetitions=3,
    )

    assert resumed == payload
    assert (root / benchmark.CHECKPOINT_NAME).is_file()


def test_layout_summary_requires_each_layout_to_have_own_repetitions() -> None:
    rows = []
    for count in (4, 64):
        for repetition in range(1, 4):
            rows.append(
                {
                    "label": "case",
                    "file_count": count,
                    "repetition": repetition,
                    "write_wall_seconds": float(repetition),
                    "footer_open_wall_seconds": 0.1,
                    "scan_wall_seconds": 0.2,
                    "hash_wall_seconds": 0.3,
                    "file_bytes": count,
                    "footer_metadata_bytes": count,
                    "logical_digest": "a" * 64,
                }
            )

    summary = benchmark._summarize_measurements(rows)

    assert summary["case:4"]["repetitions"] == 3
    assert summary["case:64"]["repetitions"] == 3
