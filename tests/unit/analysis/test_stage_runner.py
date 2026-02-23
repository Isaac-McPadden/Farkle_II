"""Tests for stage runner manifest and artifact validation behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from farkle.analysis.stage_runner import (
    StagePlanItem,
    StageRunContext,
    StageRunner,
    StageValidationError,
)
from farkle.config import AppConfig, IOConfig


def _manifest_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_stage_runner_marks_failed_when_required_output_missing_and_stops_downstream(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    manifest_path = tmp_path / "manifest.jsonl"
    calls: list[str] = []

    def _head2head(_cfg: AppConfig) -> None:
        calls.append("head2head")

    def _downstream(_cfg: AppConfig) -> None:
        calls.append("downstream")

    expected = cfg.head2head_stage_dir / "bonferroni_pairwise.parquet"
    plan = [
        StagePlanItem("head2head", _head2head, required_outputs=(expected,)),
        StagePlanItem("seed_symmetry", _downstream),
    ]
    context = StageRunContext(config=cfg, manifest_path=manifest_path, run_label="single_seed_analysis")

    with pytest.raises(StageValidationError):
        StageRunner.run(plan, context, raise_on_failure=True)

    assert calls == ["head2head"]
    lines = _manifest_lines(manifest_path)
    head2head_end = next(
        line for line in lines if line.get("event") == "stage_end" and line.get("stage") == "head2head"
    )
    assert head2head_end["ok"] is False
    assert head2head_end["missing_outputs"] == [str(expected)]

    run_end = next(line for line in lines if line.get("event") == "run_end")
    assert run_end["ok"] is False
    assert run_end["failed_steps"] == ["head2head"]
