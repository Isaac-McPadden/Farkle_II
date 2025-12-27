import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis import combine
from farkle.config import AppConfig, IOConfig
from farkle.utils.schema_helpers import expected_schema_for


def _write_curated(path: Path, schema: pa.Schema, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(tbl, path)


def test_combine_pads_and_counts(tmp_results_dir: Path, capinfo, monkeypatch) -> None:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_results_dir, append_seed=False))
    # create per-N curated files
    p1 = cfg.ingested_rows_curated(1)
    schema1 = expected_schema_for(1)
    _write_curated(
        p1,
        schema1,
        [
            {"winner": "P1", "n_rounds": 1, "winning_score": 100, "P1_strategy": "A", "P1_rank": 1},
        ],
    )
    p2 = cfg.ingested_rows_curated(2)
    schema2 = expected_schema_for(2)
    _write_curated(
        p2,
        schema2,
        [
            {
                "winner": "P1",
                "n_rounds": 1,
                "winning_score": 200,
                "P1_strategy": "A",
                "P2_strategy": "B",
                "P1_rank": 1,
                "P2_rank": 2,
            },
        ],
    )
    calls: list[tuple[list[Path], Path, int]] = []

    def _capture(files: list[Path], combined: Path, max_players: int = 12) -> None:
        calls.append((files, combined, max_players))

    monkeypatch.setattr(combine, "check_post_combine", _capture)

    # run combine
    combine.run(cfg)
    out = cfg.combine_pooled_dir() / "all_ingested_rows.parquet"
    pf = pq.ParquetFile(out)
    assert pf.metadata.num_rows == 2
    assert pq.read_schema(out).names == expected_schema_for(12).names
    log = next(rec for rec in capinfo.records if rec.message == "Combine: parquet written")
    assert getattr(log, "stage", None) == "combine"
    assert getattr(log, "path", "") == str(out)
    assert getattr(log, "rows", None) == 2
    manifest_path = out.with_suffix(".manifest.jsonl")
    assert manifest_path.exists()
    assert calls and calls[0][0] == [p1, p2]
    assert calls[0][1] == out
    assert calls[0][2] == 12


def test_combine_logs_when_no_inputs(tmp_results_dir: Path, capinfo, monkeypatch) -> None:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_results_dir, append_seed=False))
    calls: list[tuple[list[Path], Path, int]] = []

    monkeypatch.setattr(
        combine,
        "check_post_combine",
        lambda *args, **kwargs: calls.append(([], Path(), 0)),
    )

    combine.run(cfg)

    assert any(
        rec.message == "Combine: no inputs discovered" and getattr(rec, "stage", None) == "combine"
        for rec in capinfo.records
    )
    assert not calls


def test_combine_skips_when_output_newer(tmp_results_dir: Path, capinfo, monkeypatch) -> None:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_results_dir, append_seed=False))
    calls: list[tuple[list[Path], Path, int]] = []

    monkeypatch.setattr(
        combine,
        "check_post_combine",
        lambda *args, **kwargs: calls.append(([], Path(), 0)),
    )

    schema = pa.schema([("winner", pa.string())])
    input_path = cfg.ingested_rows_curated(1)
    _write_curated(input_path, schema, [{"winner": "P1"}])

    out_dir = cfg.combine_pooled_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "all_ingested_rows.parquet"
    pq.write_table(pa.Table.from_pylist([{"winner": "P1"}], schema=schema), out)
    os.utime(input_path, (0, 0))

    combine.run(cfg)

    assert any(
        rec.message == "Combine: output up-to-date" and getattr(rec, "stage", None) == "combine"
        for rec in capinfo.records
    )
    assert not calls


def test_combine_zero_row_inputs_cleanup(tmp_results_dir: Path, capinfo, monkeypatch) -> None:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_results_dir, append_seed=False))
    calls: list[tuple[list[Path], Path, int]] = []

    monkeypatch.setattr(
        combine,
        "check_post_combine",
        lambda *args, **kwargs: calls.append(([], Path(), 0)),
    )

    schema = expected_schema_for(1)
    input_path = cfg.ingested_rows_curated(1)
    _write_curated(input_path, schema, [])

    out_dir = cfg.combine_pooled_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "all_ingested_rows.parquet"
    out.write_text("stale")
    manifest = out.with_suffix(".manifest.jsonl")
    manifest.write_text("old")
    os.utime(out, (0, 0))

    combine.run(cfg)

    assert any(
        rec.message == "Combine: inputs produced zero rows"
        and getattr(rec, "stage", None) == "combine"
        for rec in capinfo.records
    )
    assert not out.exists()
    assert not manifest.exists()
    assert not calls
