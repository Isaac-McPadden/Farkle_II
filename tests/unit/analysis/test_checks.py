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

    with pytest.raises(RuntimeError) as excinfo:
        check_pre_metrics(combined)

    assert str(excinfo.value) == f"check_pre_metrics: missing 'winner' column in {combined}"


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

    with pytest.raises(RuntimeError) as excinfo:
        check_pre_metrics(combined)

    assert str(excinfo.value) == "check_pre_metrics: negative values present in bad_counts"


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

    with pytest.raises(RuntimeError) as excinfo:
        check_pre_metrics(combined)

    manifest = data_dir / "1p" / "manifest_1p.json"
    assert str(excinfo.value).startswith(f"check_pre_metrics: failed to parse {manifest}:")


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

    with pytest.raises(RuntimeError) as excinfo:
        check_pre_metrics(combined)

    assert str(excinfo.value) == "check_pre_metrics: row-count mismatch 2 != 1"


def test_check_pre_metrics_missing_manifest_files(tmp_path: Path) -> None:
    data_dir, combined = _combined_path(tmp_path)
    schema = pa.schema([("winner", pa.string())])
    _write_table(combined, schema, [{"winner": "P1"}])

    with pytest.raises(RuntimeError) as excinfo:
        check_pre_metrics(combined)

    assert str(excinfo.value) == f"check_pre_metrics: no manifest files found under {data_dir}"


def test_check_post_combine_missing_output(tmp_path: Path) -> None:
    combined = tmp_path / "missing.parquet"

    with pytest.raises(RuntimeError) as excinfo:
        check_post_combine([], combined)

    assert str(excinfo.value) == f"check_post_combine: missing {combined}"


def test_check_post_combine_unreadable_output(tmp_path: Path) -> None:
    combined = tmp_path / "bad.parquet"
    combined.write_text("not parquet data")

    with pytest.raises(RuntimeError) as excinfo:
        check_post_combine([], combined)

    assert str(excinfo.value).startswith(f"check_post_combine: unable to read {combined}:")


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

    with pytest.raises(RuntimeError) as excinfo:
        check_post_combine([curated], combined, max_players=1)

    assert str(excinfo.value) == "check_post_combine: row-count mismatch 2 != 1"


def test_check_post_combine_schema_mismatch(tmp_path: Path) -> None:
    schema = expected_schema_for(1)
    curated = tmp_path / "1p" / "curated.parquet"
    _write_table(curated, schema, [{"winning_score": 100}])

    combined = tmp_path / "combined.parquet"
    wrong_schema = pa.schema([("winner_seat", pa.string())])
    _write_table(combined, wrong_schema, [{"winner_seat": "P1"}])

    with pytest.raises(RuntimeError) as excinfo:
        check_post_combine([curated], combined, max_players=1)

    assert str(excinfo.value) == "check_post_combine: output schema mismatch"


def test_check_post_combine_unreadable_curated_file(tmp_path: Path) -> None:
    combined = tmp_path / "combined.parquet"
    _write_table(combined, pa.schema([("winner", pa.string())]), [{"winner": "P1"}])
    curated = tmp_path / "broken.parquet"
    curated.write_text("not parquet")

    with pytest.raises(RuntimeError) as excinfo:
        check_post_combine([curated], combined, max_players=1)

    assert str(excinfo.value).startswith(f"check_post_combine: unable to read {curated}:")


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


def test_check_pre_metrics_all_n_players_combined_parent_resolution(
    tmp_path: Path,
    caplog,
) -> None:
    data_dir = tmp_path / "02_combine"
    combined = (
        tmp_path
        / "02_combine"
        / "all_n_players_combined"
        / "all_ingested_rows.parquet"
    )
    schema = pa.schema([("winner", pa.string())])
    _write_table(combined, schema, [{"winner": "P1"}])
    _write_manifest(data_dir, 1, {"row_count": 1})

    with caplog.at_level("INFO"):
        check_pre_metrics(combined)

    assert "check_pre_metrics passed" in caplog.text


def test_check_pre_metrics_pooled_without_curate_falls_back_to_analysis_root(
    tmp_path: Path,
    caplog,
) -> None:
    analysis_root = tmp_path / "analysis"
    combined = analysis_root / "02_combine" / "pooled" / "all_ingested_rows.parquet"
    schema = pa.schema([("winner", pa.string())])
    _write_table(combined, schema, [{"winner": "P1"}])
    _write_manifest(analysis_root, 1, {"row_count": 1})

    with caplog.at_level("INFO"):
        check_pre_metrics(combined)

    assert "check_pre_metrics passed" in caplog.text


def test_check_pre_metrics_default_parent_resolution(tmp_path: Path, caplog) -> None:
    combined_parent = tmp_path / "custom"
    combined = combined_parent / "all_ingested_rows.parquet"
    schema = pa.schema([("winner", pa.string())])
    _write_table(combined, schema, [{"winner": "P1"}])
    _write_manifest(combined_parent, 1, {"row_count": 1})

    with caplog.at_level("INFO"):
        check_pre_metrics(combined)

    assert "check_pre_metrics passed" in caplog.text


def test_check_pre_metrics_json_manifest_rows_key(tmp_path: Path, caplog) -> None:
    data_dir, combined = _combined_path(tmp_path)
    schema = pa.schema([("winner", pa.string())])
    _write_table(combined, schema, [{"winner": "P1"}])
    _write_manifest(data_dir, 1, {"rows": 1})

    with caplog.at_level("INFO"):
        check_pre_metrics(combined)

    assert "check_pre_metrics passed" in caplog.text


def test_check_pre_metrics_jsonl_manifest_mixed_records(tmp_path: Path, caplog) -> None:
    data_dir, combined = _combined_path(tmp_path)
    schema = pa.schema([("winner", pa.string())])
    _write_table(combined, schema, [{"winner": "P1"}, {"winner": "P2"}])
    manifest = data_dir / "1p" / "manifest_1p.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"row_count": 1}),
                json.dumps("ignore this record"),
                json.dumps({"rows": 1}),
            ]
        )
    )

    with caplog.at_level("INFO"):
        check_pre_metrics(combined)

    assert "check_pre_metrics passed" in caplog.text


def test_check_pre_metrics_manifest_candidate_priority_prefers_jsonl(
    tmp_path: Path,
    caplog,
) -> None:
    data_dir, combined = _combined_path(tmp_path)
    schema = pa.schema([("winner", pa.string())])
    _write_table(combined, schema, [{"winner": "P1"}])
    seat_dir = data_dir / "1p"
    seat_dir.mkdir(parents=True, exist_ok=True)
    (seat_dir / "manifest.jsonl").write_text(json.dumps({"row_count": 1}))
    (seat_dir / "manifest_1p.json").write_text(json.dumps({"row_count": 999}))

    with caplog.at_level("INFO"):
        check_pre_metrics(combined)

    assert "check_pre_metrics passed" in caplog.text


def test_check_pre_metrics_negative_loss_margin_allowed(tmp_path: Path, caplog) -> None:
    data_dir, combined = _combined_path(tmp_path)
    schema = pa.schema([("winner", pa.string()), ("loss_margin", pa.int32())])
    _write_table(combined, schema, [{"winner": "P1", "loss_margin": -5}])
    _write_manifest(data_dir, 1, {"row_count": 1})

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

    expected = (
        f"check_stage_artifact_families failed under {cfg.analysis_dir}:\n"
        f" - game_stats: missing per-k artifact {stage_dir / '2p' / 'game_length.parquet'}\n"
        f" - game_stats: layout drift; expected {stage_dir / '2p' / 'game_length.parquet'} "
        f"but found {stage_dir / 'game_length.parquet'}\n"
        f" - game_stats: missing per-k artifact {stage_dir / '2p' / 'margin_stats.parquet'}\n"
        f" - game_stats: missing pooled_concat artifact {stage_dir / 'pooled' / 'game_length.parquet'}\n"
        f" - game_stats: layout drift; expected {stage_dir / 'pooled' / 'game_length.parquet'} "
        f"but found {stage_dir / 'game_length.parquet'}\n"
        f" - game_stats: missing pooled_concat artifact {stage_dir / 'pooled' / 'margin_stats.parquet'}\n"
        f" - game_stats: missing pooled_weighted artifact "
        f"{stage_dir / 'pooled' / 'game_length_k_weighted.parquet'}\n"
        f" - game_stats: missing pooled_weighted artifact "
        f"{stage_dir / 'pooled' / 'margin_k_weighted.parquet'}"
    )
    assert str(excinfo.value) == expected


def test_check_stage_artifact_families_flags_duplicate_and_mismatched_family_outputs(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    stage_dir = cfg.stage_dir("metrics")
    stage_dirs = {"metrics": stage_dir}

    pooled = stage_dir / "pooled"
    pooled.mkdir(parents=True, exist_ok=True)
    # Mismatched-family output: per-k artifact written in pooled/.
    misplaced_per_k = pooled / "2p_isolated_metrics.parquet"
    misplaced_per_k.touch()
    # Duplicate output: pooled artifact also exists at stage root.
    root_duplicate = stage_dir / "metrics.parquet"
    root_duplicate.touch()

    with pytest.raises(RuntimeError) as excinfo:
        check_stage_artifact_families(cfg.analysis_dir, stage_dirs, (2,))

    expected = (
        f"check_stage_artifact_families failed under {cfg.analysis_dir}:\n"
        f" - metrics: missing per-k artifact {stage_dir / '2p' / '2p_isolated_metrics.parquet'}\n"
        f" - metrics: missing pooled_concat artifact {pooled / 'metrics.parquet'}\n"
        f" - metrics: layout drift; expected {pooled / 'metrics.parquet'} but found {root_duplicate}"
    )
    assert str(excinfo.value) == expected


def test_check_stage_artifact_families_skips_missing_stage_keys(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    check_stage_artifact_families(cfg.analysis_dir, {}, (2,))


def test_check_stage_artifact_families_skips_nonexistent_stage_dir(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    stage_dirs = {"metrics": tmp_path / "missing_metrics"}
    check_stage_artifact_families(cfg.analysis_dir, stage_dirs, (2,))


def test_check_stage_artifact_families_honors_custom_matrix_success(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    stage_dir = cfg.stage_dir("metrics")
    expected = stage_dir / "pooled" / "only_here.parquet"
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    check_stage_artifact_families(
        cfg.analysis_dir,
        {"metrics": stage_dir},
        (2,),
        matrix={"metrics": {"pooled_concat": ("only_here.parquet",)}},
    )


def test_check_stage_artifact_families_honors_custom_matrix_failure(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    stage_dir = cfg.stage_dir("metrics")

    with pytest.raises(RuntimeError) as excinfo:
        check_stage_artifact_families(
            cfg.analysis_dir,
            {"metrics": stage_dir},
            (2,),
            matrix={"metrics": {"pooled_concat": ("required.parquet",)}},
        )

    expected = (
        f"check_stage_artifact_families failed under {cfg.analysis_dir}:\n"
        f" - metrics: missing pooled_concat artifact {stage_dir / 'pooled' / 'required.parquet'}"
    )
    assert str(excinfo.value) == expected
