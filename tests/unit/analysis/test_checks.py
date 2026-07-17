from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.artifact_sidecars import sidecar_metadata, write_parquet_test_artifact

from farkle.analysis import checks as checks_module
from farkle.analysis.checks import check_post_combine, check_pre_metrics
from farkle.config import AppConfig, IOConfig
from farkle.utils.artifact_contract import write_artifact_with_sidecar_atomic
from farkle.utils.schema_helpers import expected_schema_for


def _canonical_concat(
    tmp_path: Path,
    rows: list[dict[str, object]],
    schema: pa.Schema,
    *,
    write_statistics: bool = True,
) -> Path:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    output = cfg.curated_parquet
    table = pa.Table.from_pylist(rows, schema=schema)
    if write_statistics:
        write_parquet_test_artifact(table, output, scope="concat_ks")
    else:

        def _write(staged_path: Path) -> None:
            pq.write_table(table, staged_path, write_statistics=False)

        write_artifact_with_sidecar_atomic(output, sidecar_metadata(output), _write)
    output.with_suffix(".manifest.jsonl").write_text(
        json.dumps({"path": output.name, "rows": len(rows)}) + "\n",
        encoding="utf-8",
    )
    return output


def test_check_pre_metrics_accepts_canonical_concat(tmp_path: Path, monkeypatch) -> None:
    output = _canonical_concat(
        tmp_path,
        [{"winner_seat": "P1", "n_rounds": 3}],
        pa.schema([("winner_seat", pa.string()), ("n_rounds", pa.int16())]),
    )
    monkeypatch.setattr(
        checks_module,
        "_scan_negative_columns",
        lambda *_args, **_kwargs: pytest.fail("complete statistics must avoid a data scan"),
    )
    check_pre_metrics(output, winner_col="winner_seat")


def test_check_pre_metrics_rejects_missing_winner(tmp_path: Path) -> None:
    output = _canonical_concat(tmp_path, [{"n_rounds": 1}], pa.schema([("n_rounds", pa.int16())]))
    with pytest.raises(RuntimeError, match="missing 'winner_seat' column"):
        check_pre_metrics(output, winner_col="winner_seat")


def test_check_pre_metrics_rejects_negative_counter(tmp_path: Path, monkeypatch) -> None:
    output = _canonical_concat(
        tmp_path,
        [{"winner_seat": "P1", "bad_count": -1}],
        pa.schema([("winner_seat", pa.string()), ("bad_count", pa.int32())]),
    )
    monkeypatch.setattr(
        checks_module,
        "_scan_negative_columns",
        lambda *_args, **_kwargs: pytest.fail("metadata must detect the negative value"),
    )
    with pytest.raises(RuntimeError, match="negative values present in bad_count"):
        check_pre_metrics(output, winner_col="winner_seat")


def test_check_pre_metrics_treats_all_null_statistics_as_safe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = _canonical_concat(
        tmp_path,
        [{"winner_seat": "P1", "optional_count": None}],
        pa.schema([("winner_seat", pa.string()), ("optional_count", pa.int32())]),
    )
    monkeypatch.setattr(
        checks_module,
        "_scan_negative_columns",
        lambda *_args, **_kwargs: pytest.fail("all-null row groups must avoid a data scan"),
    )

    check_pre_metrics(output, winner_col="winner_seat")


def test_check_pre_metrics_scans_columns_without_statistics_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = _canonical_concat(
        tmp_path,
        [{"winner_seat": "P1", "bad_count": -1, "good_count": 2}],
        pa.schema(
            [
                ("winner_seat", pa.string()),
                ("bad_count", pa.int32()),
                ("good_count", pa.int32()),
            ]
        ),
        write_statistics=False,
    )
    calls: list[tuple[str, ...]] = []
    real_scan = checks_module._scan_negative_columns

    def _scan(path: Path, columns: list[str]) -> set[str]:
        calls.append(tuple(columns))
        return real_scan(path, columns)

    monkeypatch.setattr(checks_module, "_scan_negative_columns", _scan)

    with pytest.raises(RuntimeError, match="negative values present in bad_count"):
        check_pre_metrics(output, winner_col="winner_seat")

    assert calls == [("bad_count", "good_count")]


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
