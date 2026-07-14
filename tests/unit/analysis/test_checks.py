from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest
from tests.helpers.artifact_sidecars import write_parquet_test_artifact

from farkle.analysis.checks import check_post_combine, check_pre_metrics
from farkle.config import AppConfig, IOConfig
from farkle.utils.schema_helpers import expected_schema_for


def _canonical_concat(tmp_path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> Path:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    output = cfg.curated_parquet
    write_parquet_test_artifact(
        pa.Table.from_pylist(rows, schema=schema),
        output,
        scope="concat_ks",
    )
    output.with_suffix(".manifest.jsonl").write_text(
        json.dumps({"path": output.name, "rows": len(rows)}) + "\n",
        encoding="utf-8",
    )
    return output


def test_check_pre_metrics_accepts_canonical_concat(tmp_path: Path) -> None:
    output = _canonical_concat(
        tmp_path,
        [{"winner_seat": "P1", "n_rounds": 3}],
        pa.schema([("winner_seat", pa.string()), ("n_rounds", pa.int16())]),
    )
    check_pre_metrics(output, winner_col="winner_seat")


def test_check_pre_metrics_rejects_missing_winner(tmp_path: Path) -> None:
    output = _canonical_concat(tmp_path, [{"n_rounds": 1}], pa.schema([("n_rounds", pa.int16())]))
    with pytest.raises(RuntimeError, match="missing 'winner_seat' column"):
        check_pre_metrics(output, winner_col="winner_seat")


def test_check_pre_metrics_rejects_negative_counter(tmp_path: Path) -> None:
    output = _canonical_concat(
        tmp_path,
        [{"winner_seat": "P1", "bad_count": -1}],
        pa.schema([("winner_seat", pa.string()), ("bad_count", pa.int32())]),
    )
    with pytest.raises(RuntimeError, match="negative values present in bad_count"):
        check_pre_metrics(output, winner_col="winner_seat")


def test_check_pre_metrics_rejects_manifest_mismatch(tmp_path: Path) -> None:
    output = _canonical_concat(
        tmp_path,
        [{"winner_seat": "P1"}],
        pa.schema([("winner_seat", pa.string())]),
    )
    output.with_suffix(".manifest.jsonl").write_text('{"rows":2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="row-count mismatch 1 != 2"):
        check_pre_metrics(output, winner_col="winner_seat")


def test_check_pre_metrics_rejects_missing_or_multiple_manifest_records(tmp_path: Path) -> None:
    output = _canonical_concat(
        tmp_path,
        [{"winner_seat": "P1"}],
        pa.schema([("winner_seat", pa.string())]),
    )
    manifest = output.with_suffix(".manifest.jsonl")
    manifest.unlink()
    with pytest.raises(RuntimeError, match="missing canonical manifest"):
        check_pre_metrics(output, winner_col="winner_seat")
    manifest.write_text('{"rows":1}\n{"rows":1}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected exactly one manifest record"):
        check_pre_metrics(output, winner_col="winner_seat")


def test_check_pre_metrics_rejects_scope_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "unscoped" / "rows.parquet"
    write_parquet_test_artifact(
        pa.Table.from_pylist([{"winner_seat": "P1"}]),
        path,
        scope="concat_ks",
    )
    with pytest.raises(RuntimeError, match="scope mismatch"):
        check_pre_metrics(path, winner_col="winner_seat")


def test_check_post_combine_accepts_matching_rows_and_schema(tmp_path: Path) -> None:
    schema = expected_schema_for(1)
    curated = tmp_path / "1p" / "rows.parquet"
    combined = tmp_path / "combined.parquet"
    write_parquet_test_artifact(
        pa.Table.from_pylist([{"winning_score": 100}], schema=schema), curated
    )
    write_parquet_test_artifact(
        pa.Table.from_pylist([{"winning_score": 100}], schema=schema), combined
    )
    check_post_combine([curated], combined, max_players=1)


def test_check_post_combine_rejects_missing_and_row_mismatch(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"
    with pytest.raises(RuntimeError, match="missing"):
        check_post_combine([], missing)

    schema = expected_schema_for(1)
    curated = tmp_path / "1p" / "rows.parquet"
    combined = tmp_path / "combined.parquet"
    write_parquet_test_artifact(
        pa.Table.from_pylist([{"winning_score": 100}], schema=schema), curated
    )
    write_parquet_test_artifact(
        pa.Table.from_pylist([{"winning_score": 100}, {"winning_score": 200}], schema=schema),
        combined,
    )
    with pytest.raises(RuntimeError, match="row-count mismatch 2 != 1"):
        check_post_combine([curated], combined, max_players=1)
