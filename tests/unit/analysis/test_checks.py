from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.checks import (
    check_post_combine,
    check_pre_metrics,
    check_stage_artifact_families,
)
from farkle.config import AppConfig, IOConfig
from farkle.utils.schema_helpers import expected_schema_for


def _combined_path(tmp_path: Path) -> tuple[Path, Path]:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    data_dir = cfg.stage_dir("curate")
    combined_dir = cfg.stage_subdir("combine", "pooled")
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
    schema = pa.schema([pa.field("n_rounds", pa.int16())])
    _write_table(combined, schema, [{"n_rounds": 1}])
    _write_manifest(data_dir, 1, {"row_count": 1})

    with pytest.raises(RuntimeError, match="missing 'winner'"):
        check_pre_metrics(combined)


def test_check_pre_metrics_negative_integer_column(tmp_path: Path) -> None:
    data_dir, combined = _combined_path(tmp_path)
    schema = pa.schema(
        [
            pa.field("winner", pa.string()),
            pa.field("n_rounds", pa.int16()),
            pa.field("bad_counts", pa.int32()),
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
            pa.field("winner", pa.string()),
            pa.field("n_rounds", pa.int16()),
            pa.field("good_counts", pa.int32()),
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
            pa.field("winner", pa.string()),
            pa.field("n_rounds", pa.int16()),
            pa.field("good_counts", pa.int32()),
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


def test_check_pre_metrics_passes_with_manifest(tmp_path: Path, caplog) -> None:
    data_dir, combined = _combined_path(tmp_path)
    schema = pa.schema([("winner", pa.string()), ("n_rounds", pa.int16())])
    _write_table(combined, schema, [{"winner": "P1", "n_rounds": 3}])
    _write_manifest(data_dir, 1, {"row_count": 1})

    with caplog.at_level("INFO"):
        check_pre_metrics(combined)

    assert "check_pre_metrics passed" in caplog.text


def test_check_pre_metrics_manifest_fallback(tmp_path: Path, caplog) -> None:
    _, combined = _combined_path(tmp_path)
    combined.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("winner", pa.string())])
    _write_table(combined, schema, [{"winner": "P1"}])
    manifest_path = combined.with_suffix(".manifest.jsonl")
    manifest_path.write_text(json.dumps({"row_count": 1}))

    with caplog.at_level("INFO"):
        check_pre_metrics(combined)

    assert "check_pre_metrics passed" in caplog.text


def test_check_post_combine_success(tmp_path: Path, caplog) -> None:
    schema = expected_schema_for(1)
    curated = tmp_path / "1p" / "curated.parquet"
    _write_table(curated, schema, [{"winning_score": 100}])
    combined = tmp_path / "combined.parquet"
    _write_table(combined, schema, [{"winning_score": 100}])

    with caplog.at_level("INFO"):
        check_post_combine([curated], combined, max_players=1)

    assert "check_post_combine passed" in caplog.text


def test_check_stage_artifact_families_passes_expected_matrix(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))

    stage_dirs = {
        "combine": cfg.stage_dir("combine"),
        "metrics": cfg.stage_dir("metrics"),
        "game_stats": cfg.stage_dir("game_stats"),
    }
    k_values = (2, 3)

    combine_out = stage_dirs["combine"] / "pooled" / "all_ingested_rows.parquet"
    combine_out.parent.mkdir(parents=True, exist_ok=True)
    combine_out.touch()
    metrics_out = stage_dirs["metrics"] / "pooled" / "metrics.parquet"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.touch()
    for k in k_values:
        metrics_per_k = stage_dirs["metrics"] / f"{k}p" / f"{k}p_isolated_metrics.parquet"
        metrics_per_k.parent.mkdir(parents=True, exist_ok=True)
        metrics_per_k.touch()
    pooled_game_length = stage_dirs["game_stats"] / "pooled" / "game_length.parquet"
    pooled_game_length.parent.mkdir(parents=True, exist_ok=True)
    pooled_game_length.touch()
    pooled_margin = stage_dirs["game_stats"] / "pooled" / "margin_stats.parquet"
    pooled_margin.touch()
    (stage_dirs["game_stats"] / "pooled" / "game_length_k_weighted.parquet").touch()
    (stage_dirs["game_stats"] / "pooled" / "margin_k_weighted.parquet").touch()
    for k in k_values:
        per_k_game_length = stage_dirs["game_stats"] / f"{k}p" / "game_length.parquet"
        per_k_game_length.parent.mkdir(parents=True, exist_ok=True)
        per_k_game_length.touch()
        (stage_dirs["game_stats"] / f"{k}p" / "margin_stats.parquet").touch()

    check_stage_artifact_families(cfg.analysis_dir, stage_dirs, k_values)


def test_check_stage_artifact_families_flags_missing_and_layout_drift(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    stage_dir = cfg.stage_dir("game_stats")
    stage_dirs = {"game_stats": stage_dir}

    # Drift: pooled aggregate written to stage root instead of pooled/.
    drift = stage_dir / "game_length.parquet"
    drift.parent.mkdir(parents=True, exist_ok=True)
    drift.touch()
    # Missing: other required artifacts intentionally omitted.

    with pytest.raises(RuntimeError) as excinfo:
        check_stage_artifact_families(cfg.analysis_dir, stage_dirs, (2,))

    msg = str(excinfo.value)
    assert "missing per-k artifact" in msg
    assert "layout drift" in msg
