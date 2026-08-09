from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.artifact_sidecars import (
    make_authenticated_v3_config,
    publish_v3_parquet,
)

from farkle.analysis import combine
from farkle.analysis.checks import check_pre_metrics
from farkle.analysis.release_audit import audit_sidecar_completeness
from farkle.config import AppConfig
from farkle.utils.artifact_contract import sidecar_path, validate_artifact_sidecar
from farkle.utils.schema_helpers import expected_schema_for


def _cfg(tmp_path: Path, player_counts: tuple[int, ...] = (1, 2)) -> AppConfig:
    cfg = make_authenticated_v3_config(
        tmp_path,
        name="combine",
        root_seed=0,
        player_counts=player_counts,
    )
    cfg.analysis.n_jobs = 1
    return cfg


def _row(k: int, index: int = 0) -> dict[str, object]:
    row: dict[str, object] = {
        "root_seed": 0,
        "k": k,
        "shuffle_index": index,
        "game_index": 0,
        "deterministic_batch_id": index,
        "shuffle_seed": 400 + index,
        "winner_seat": "P1",
        "winner_strategy": 10 + k,
        "game_seed": 100 + index,
        "seat_ranks": [f"P{seat}" for seat in range(1, k + 1)],
        "n_rounds": 1 + index,
        "winning_score": 100 + k,
    }
    for seat in range(1, k + 1):
        row[f"P{seat}_strategy"] = 10 + seat
        row[f"P{seat}_rank"] = seat
    return row


def _write_curated(cfg: AppConfig, k: int, rows: list[dict[str, object]]) -> Path:
    path = cfg.ingested_rows_curated(k)
    publish_v3_parquet(
        cfg,
        path,
        pa.Table.from_pylist(rows, schema=expected_schema_for(k)),
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
        source_scope="by_k",
    )
    return path


def _write_all_sources(cfg: AppConfig) -> tuple[Path, ...]:
    return tuple(_write_curated(cfg, k, [_row(k, k)]) for k in sorted(cfg.sim.n_players_list))


def _manifest_units(cfg: AppConfig) -> list[dict[str, object]]:
    records = [
        json.loads(line)
        for line in cfg.combined_manifest_path().read_text(encoding="utf-8").splitlines()
    ]
    return [record for record in records if record["type"] == "unit"]


def test_partitioned_concat_is_logically_equivalent_and_deterministic(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_all_sources(cfg)

    combine.run(cfg)

    paths = combine.combined_partition_paths(cfg)
    assert paths == (cfg.combined_rows_by_k(1), cfg.combined_rows_by_k(2))
    assert not cfg.curated_parquet.exists()
    assert [record["unit_key"] for record in _manifest_units(cfg)] == [[1], [2]]
    assert [record["unit_metadata"]["row_count"] for record in _manifest_units(cfg)] == [1, 1]
    assert all(
        record["unit_metadata"]["schema_sha256"]
        == _manifest_units(cfg)[0]["unit_metadata"]["schema_sha256"]
        for record in _manifest_units(cfg)
    )
    projected = []
    for batch in combine.scan_concat_ks(
        cfg,
        columns=["k", "shuffle_index", "winner_strategy"],
        max_batch_bytes=1024,
        max_batch_rows=1,
    ):
        projected.extend(pa.Table.from_batches([batch]).to_pylist())
    assert projected == [
        {"k": 1, "shuffle_index": 1, "winner_strategy": 11},
        {"k": 2, "shuffle_index": 2, "winner_strategy": 12},
    ]
    check_pre_metrics(cfg, winner_col="winner_seat")
    validate_artifact_sidecar(
        cfg.combined_manifest_path(),
        expected={"scope": "concat_ks", "operation": "concatenate"},
    )
    stable = {
        path: path.read_bytes()
        for path in (
            *paths,
            cfg.combined_manifest_path(),
            sidecar_path(cfg.combined_manifest_path()),
        )
    }
    combine.run(cfg)
    assert {path: path.read_bytes() for path in stable} == stable


def test_partial_partition_set_never_publishes_completion(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_curated(cfg, 1, [_row(1)])

    with pytest.raises(FileNotFoundError, match="incomplete canonical curated by-k support"):
        combine.run(cfg)

    assert not cfg.combined_manifest_path().exists()
    assert not (cfg.combine_stage_dir / "combine.done.json").exists()


def test_interrupted_combination_resumes_only_missing_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    _write_all_sources(cfg)
    original = combine._PartitionWriter.__call__

    def interrupt(self, unit, output):
        if unit.key == (2,):
            raise RuntimeError("synthetic combine interruption")
        return original(self, unit, output)

    monkeypatch.setattr(combine._PartitionWriter, "__call__", interrupt)
    with pytest.raises(RuntimeError, match="synthetic combine interruption"):
        combine.run(cfg)
    first = cfg.combined_rows_by_k(1)
    first_bytes = first.read_bytes()
    assert first.exists()
    assert not cfg.combined_manifest_path().exists()

    monkeypatch.setattr(combine._PartitionWriter, "__call__", original)
    combine.run(cfg)

    assert first.read_bytes() == first_bytes
    assert combine.verify_concat_ks(cfg) == {
        "partitions": 2,
        "rows": 2,
        "deep_verified": False,
        "deep_scanned_rows": 0,
    }


def test_changed_one_source_precisely_rewrites_one_partition(tmp_path: Path, caplog) -> None:
    caplog.set_level("INFO")
    cfg = _cfg(tmp_path)
    _write_all_sources(cfg)
    combine.run(cfg)
    first = cfg.combined_rows_by_k(1)
    first_bytes = first.read_bytes()

    _write_curated(cfg, 2, [_row(2), _row(2, 9)])
    combine.run(cfg)

    record = [
        item for item in caplog.records if item.message == "Combine: partitioned dataset written"
    ][-1]
    assert record.reused_partitions == 1
    assert record.written_partitions == 1
    assert first.read_bytes() == first_bytes
    assert pq.read_metadata(cfg.combined_rows_by_k(2)).num_rows == 2


@pytest.mark.parametrize("mutation", ["bytes", "sidecar"])
def test_changed_source_bytes_or_sidecar_invalidates_dataset(tmp_path: Path, mutation: str) -> None:
    cfg = _cfg(tmp_path)
    sources = _write_all_sources(cfg)
    combine.run(cfg)
    source = sources[1]
    if mutation == "bytes":
        with source.open("ab") as handle:
            handle.write(b"mutation")
    else:
        with sidecar_path(source).open("a", encoding="utf-8") as handle:
            handle.write("\n")

    with pytest.raises(RuntimeError):
        combine.combined_partition_paths(cfg)
    if mutation == "bytes":
        with pytest.raises(RuntimeError):
            combine.run(cfg)
    else:
        combine.run(cfg)
        assert combine.verify_concat_ks(cfg)["partitions"] == 2


def test_complete_repository_code_identity_change_invalidates_all_partitions(
    tmp_path: Path, caplog
) -> None:
    caplog.set_level("INFO")
    cfg = _cfg(tmp_path)
    _write_all_sources(cfg)
    combine.run(cfg)
    assert cfg._code_identity is not None
    cfg._code_identity = replace(
        cfg._code_identity,
        commit="b" * 40,
    )

    combine.run(cfg)

    record = [
        item for item in caplog.records if item.message == "Combine: partitioned dataset written"
    ][-1]
    assert record.reused_partitions == 0
    assert record.written_partitions == 2


def test_missing_corrupt_and_reordered_partitions_are_not_complete(tmp_path: Path, caplog) -> None:
    caplog.set_level("INFO")
    cfg = _cfg(tmp_path)
    _write_all_sources(cfg)
    combine.run(cfg)
    cfg.combined_rows_by_k(2).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError):
        combine.combined_partition_paths(cfg)

    combine.run(cfg)
    assert combine.verify_concat_ks(cfg)["partitions"] == 2
    lines = cfg.combined_manifest_path().read_text(encoding="utf-8").splitlines()
    cfg.combined_manifest_path().write_text(
        "\n".join([lines[0], lines[2], lines[1], lines[3]]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        combine.combined_partition_paths(cfg)
    combine.run(cfg)
    assert [record["unit_key"] for record in _manifest_units(cfg)] == [[1], [2]]
    record = [
        item for item in caplog.records if item.message == "Combine: partitioned dataset written"
    ][-1]
    assert record.reused_partitions == 2
    assert record.written_partitions == 0

    cfg.sim.n_players_list = [1]
    combine.run(cfg)
    assert combine.combined_partition_paths(cfg) == (cfg.combined_rows_by_k(1),)
    assert not cfg.combined_rows_by_k(2).exists()


def test_release_materialization_and_deep_verification(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_all_sources(cfg)
    combine.run(cfg)
    materialized = combine.materialize_concat_ks(
        cfg,
        cfg.concat_ks_dir("combine") / "release_materialized" / "concat.parquet",
    )

    former = pq.read_table(materialized)
    current = pa.Table.from_batches(list(combine.scan_concat_ks(cfg)))
    assert current.equals(former, check_metadata=False)
    assert combine.verify_concat_ks(cfg, deep=True)["deep_scanned_rows"] == former.num_rows
    validate_artifact_sidecar(
        materialized,
        expected={"operation": "materialize_concat_ks_compatibility"},
    )
    assert audit_sidecar_completeness(cfg.analysis_dir) == []
    # The fixture demonstrates the eliminated canonical bytes: the materialized
    # compatibility file is exactly the extra physical representation no longer stored.
    assert materialized.stat().st_size > 0
    assert not cfg.curated_parquet.exists()


def test_materialization_cannot_restore_retired_canonical_path(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, (1,))
    _write_all_sources(cfg)
    combine.run(cfg)

    with pytest.raises(ValueError, match="retired canonical path"):
        combine.materialize_concat_ks(cfg, cfg.curated_parquet)


def test_padding_normalizes_types_and_nullability() -> None:
    target = pa.schema(
        [
            pa.field("value", pa.int64(), nullable=True),
            pa.field("missing", pa.int32(), nullable=True),
        ]
    )
    source = pa.Table.from_arrays(
        [pa.array([1, 2], type=pa.int32())],
        schema=pa.schema([pa.field("value", pa.int32(), nullable=False)]),
    )

    normalized = combine._pad_to_schema(source, target)

    assert normalized.schema.equals(target, check_metadata=False)
    assert normalized["missing"].null_count == 2
