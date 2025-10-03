from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.schema import expected_schema_for
from farkle.analysis.checks import check_post_combine, check_pre_metrics


def _combined_path(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    combined_dir = data_dir / "all_n_players_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, combined_dir / "all_ingested_rows.parquet"


def _write_table(path: Path, schema: pa.Schema, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


def _write_manifest(data_dir: Path, n_players: int, payload: dict[str, object] | str) -> Path:
    manifest = data_dir / f"{n_players}p" / f"manifest_{n_players}p.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        manifest.write_text(payload)
    else:
        manifest.write_text(json.dumps(payload))
    return manifest


def test_check_pre_metrics_missing_winner_column(tmp_path: Path) -> None:
    data_dir, combined = _combined_path(tmp_path)
    schema = pa.schema([("n_rounds", pa.int16())])
    _write_table(combined, schema, [{"n_rounds": 1}])
    _write_manifest(data_dir, 1, {"row_count": 1})

    with pytest.raises(RuntimeError, match="missing 'winner'"):
        check_pre_metrics(combined)


def test_check_pre_metrics_negative_integer_column(tmp_path: Path) -> None:
    data_dir, combined = _combined_path(tmp_path)
    schema = pa.schema(
        [
            ("winner", pa.string()),
            ("n_rounds", pa.int16()),
            ("bad_counts", pa.int32()),
        ]
    )
    _write_table(
        combined,
        schema,
        [{"winner": "P1", "n_rounds": 1, "bad_counts": -3}],
    )
    _write_manifest(data_dir, 1, {"row_count": 1})

    with pytest.raises(RuntimeError, match="negative values present"):
        check_pre_metrics(combined)


def test_check_pre_metrics_unreadable_manifest(tmp_path: Path) -> None:
    data_dir, combined = _combined_path(tmp_path)
    schema = pa.schema(
        [
            ("winner", pa.string()),
            ("n_rounds", pa.int16()),
            ("good_counts", pa.int32()),
        ]
    )
    _write_table(
        combined,
        schema,
        [{"winner": "P1", "n_rounds": 1, "good_counts": 5}],
    )
    _write_manifest(data_dir, 1, "{not json")

    with pytest.raises(RuntimeError, match="failed to parse"):
        check_pre_metrics(combined)


def test_check_pre_metrics_manifest_row_mismatch(tmp_path: Path) -> None:
    data_dir, combined = _combined_path(tmp_path)
    schema = pa.schema(
        [
            ("winner", pa.string()),
            ("n_rounds", pa.int16()),
            ("good_counts", pa.int32()),
        ]
    )
    _write_table(
        combined,
        schema,
        [
            {"winner": "P1", "n_rounds": 1, "good_counts": 5},
            {"winner": "P2", "n_rounds": 1, "good_counts": 6},
        ],
    )
    _write_manifest(data_dir, 1, {"row_count": 1})

    with pytest.raises(RuntimeError, match="row-count mismatch"):
        check_pre_metrics(combined)


def test_check_post_combine_missing_output(tmp_path: Path) -> None:
    combined = tmp_path / "missing.parquet"

    with pytest.raises(RuntimeError, match="missing"):
        check_post_combine([], combined)


def test_check_post_combine_unreadable_output(tmp_path: Path) -> None:
    combined = tmp_path / "bad.parquet"
    combined.write_text("not parquet data")

    with pytest.raises(RuntimeError, match="unable to read"):
        check_post_combine([], combined)


def test_check_post_combine_row_count_mismatch(tmp_path: Path) -> None:
    schema = expected_schema_for(1)
    curated = tmp_path / "1p" / "curated.parquet"
    _write_table(curated, schema, [{"winning_score": 100}])

    combined = tmp_path / "combined.parquet"
    _write_table(
        combined,
        schema,
        [{"winning_score": 100}, {"winning_score": 200}],
    )

    with pytest.raises(RuntimeError, match="row-count mismatch"):
        check_post_combine([curated], combined, max_players=1)


def test_check_post_combine_schema_mismatch(tmp_path: Path) -> None:
    schema = expected_schema_for(1)
    curated = tmp_path / "1p" / "curated.parquet"
    _write_table(curated, schema, [{"winning_score": 100}])

    combined = tmp_path / "combined.parquet"
    wrong_schema = pa.schema([("winner_seat", pa.string())])
    _write_table(combined, wrong_schema, [{"winner_seat": "P1"}])

    with pytest.raises(RuntimeError, match="output schema mismatch"):
        check_post_combine([curated], combined, max_players=1)
