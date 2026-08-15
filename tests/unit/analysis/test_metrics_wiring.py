from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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

    concat = cfg.combined_manifest_path()
    concat.parent.mkdir(parents=True, exist_ok=True)
    concat.write_text("fixture\n", encoding="utf-8")
    for path in (cfg.ingested_rows_curated(2), cfg.combined_rows_by_k(2)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    captured: dict[str, Any] = {}

    def _up_to_date(_done: Path, *, inputs, outputs, **_kwargs: object) -> bool:
        captured["inputs"] = list(inputs)
        captured["outputs"] = list(outputs)
        return True

    monkeypatch.setattr(metrics, "stage_is_up_to_date", _up_to_date)
    monkeypatch.setattr(
        metrics,
        "check_pre_metrics",
        lambda *_args, **_kwargs: pytest.fail("fresh metrics must not scan canonical rows"),
    )

    metrics.run(cfg)

    assert captured["inputs"][0] == cfg.combined_manifest_path()
    assert cfg.curated_dataset not in captured["inputs"]
    assert (
        cfg.metrics_all_player_batch_path(2).with_suffix(".manifest.jsonl") in captured["outputs"]
    )
    assert cfg.seat_batch_counts_path(2).with_suffix(".manifest.jsonl") in captured["outputs"]


def test_stale_metrics_validates_before_building(tmp_path: Path, monkeypatch) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11], n_players_list=[2]),
    )
    cfg.screening.delta_across_k = 0.03
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    for path in (
        cfg.combined_manifest_path(),
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

    def _validate(path: AppConfig, *, winner_col: str) -> None:
        events.append("validation")
        assert path is cfg
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
