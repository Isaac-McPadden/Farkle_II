from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from tests.helpers.artifact_sidecars import write_parquet_test_artifact

from farkle.analysis import metrics
from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, IOConfig, SimConfig


def test_fresh_metrics_tracks_concat_without_semantic_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11], n_players_list=[2]),
    )
    cfg.screening.delta_across_k = 0.03
    cfg.set_stage_layout(resolve_stage_layout(cfg))

    concat = cfg.curated_parquet
    write_parquet_test_artifact(
        pa.table({"winner_seat": ["P1"], "n_rounds": pa.array([1], type=pa.int16())}),
        concat,
        scope="concat_ks",
    )
    concat.with_suffix(".manifest.jsonl").write_text(
        json.dumps({"path": concat.name, "rows": 1}) + "\n",
        encoding="utf-8",
    )
    for path in (cfg.ingested_rows_curated(2), cfg.combined_rows_by_k(2)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    captured: dict[str, Any] = {}

    def _up_to_date(_done: Path, *, inputs, **_kwargs: object) -> bool:
        captured["inputs"] = list(inputs)
        return True

    monkeypatch.setattr(metrics, "stage_is_up_to_date", _up_to_date)
    monkeypatch.setattr(
        metrics,
        "check_pre_metrics",
        lambda *_args, **_kwargs: pytest.fail("fresh metrics must not scan canonical rows"),
    )

    metrics.run(cfg)

    assert captured["inputs"][0] == cfg.curated_parquet
    assert cfg.curated_dataset not in captured["inputs"]


def test_stale_metrics_validates_before_building(tmp_path: Path, monkeypatch) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11], n_players_list=[2]),
    )
    cfg.screening.delta_across_k = 0.03
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    for path in (
        cfg.curated_parquet,
        cfg.ingested_rows_curated(2),
        cfg.combined_rows_by_k(2),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    events: list[str] = []

    def _stale(*_args: object, **_kwargs: object) -> bool:
        events.append("freshness")
        return False

    class ExpectedValidation(Exception):
        pass

    def _validate(path: Path, *, winner_col: str) -> None:
        events.append("validation")
        assert path == cfg.curated_parquet
        assert winner_col == "winner_seat"
        raise ExpectedValidation

    monkeypatch.setattr(metrics, "stage_is_up_to_date", _stale)
    monkeypatch.setattr(metrics, "check_pre_metrics", _validate)
    monkeypatch.setattr(
        metrics,
        "_all_player_metrics",
        lambda *_args, **_kwargs: pytest.fail("builders must follow validation"),
    )

    with pytest.raises(ExpectedValidation):
        metrics.run(cfg)

    assert events == ["freshness", "validation"]
