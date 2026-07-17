"""Tests for stage runner manifest and artifact validation behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from farkle.analysis.stage_runner import (
    StageCompletionError,
    StagePlanItem,
    StageRunContext,
    StageRunner,
    StageValidationError,
)
from farkle.config import AppConfig, IOConfig
from farkle.utils.stage_completion import stage_done_path, write_stage_done


def _manifest_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_stage_runner_marks_failed_when_required_output_missing_and_stops_downstream(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    manifest_path = tmp_path / "manifest.jsonl"
    calls: list[str] = []

    def _h2h_inference(_cfg: AppConfig) -> None:
        calls.append("h2h_inference")

    def _downstream(_cfg: AppConfig) -> None:
        calls.append("downstream")

    expected = cfg.stage_dir("h2h_inference") / "bonferroni_pairwise.parquet"
    plan = [
        StagePlanItem("h2h_inference", _h2h_inference, required_outputs=(expected,)),
        StagePlanItem("diagnostics", _downstream),
    ]
    context = StageRunContext(
        config=cfg, manifest_path=manifest_path, run_label="single_seed_analysis"
    )

    with pytest.raises(StageValidationError):
        StageRunner.run(plan, context, raise_on_failure=True)

    assert calls == ["h2h_inference"]
    lines = _manifest_lines(manifest_path)
    h2h_end = next(
        line
        for line in lines
        if line.get("event") == "stage_end" and line.get("stage") == "h2h_inference"
    )
    assert h2h_end["ok"] is False
    assert h2h_end["missing_outputs"] == [str(expected)]

    run_end = next(line for line in lines if line.get("event") == "run_end")
    assert run_end["ok"] is False
    assert run_end["failed_steps"] == ["h2h_inference"]
    assert not cfg.stage_dir("h2h_inference").exists()


@pytest.mark.parametrize("status", ["missing", "invalid", "failed", "blocked_by_cap"])
def test_stage_runner_rejects_unsuccessful_declared_completion_stamp(
    tmp_path: Path,
    status: str,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    manifest_path = tmp_path / f"{status}.jsonl"
    output = cfg.h2h_pairwise_inference_path()
    stamp = stage_done_path(cfg.stage_dir("h2h_inference"), "h2h_inference")
    calls: list[str] = []

    def _action(_cfg: AppConfig) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("complete", encoding="utf-8")
        if status == "invalid":
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.write_text("not json", encoding="utf-8")
        elif status in {"failed", "blocked_by_cap"}:
            if status == "failed":
                write_stage_done(
                    stamp,
                    inputs=[],
                    outputs=[output],
                    cfg=cfg,
                    stage="h2h_inference",
                    status="failed",
                    blocking_dependency="test",
                    upstream_stage="h2h_execute",
                )
            else:
                write_stage_done(
                    stamp,
                    inputs=[],
                    outputs=[output],
                    cfg=cfg,
                    stage="h2h_inference",
                    status="blocked_by_cap",
                )

    def _downstream(_cfg: AppConfig) -> None:
        calls.append("downstream")

    plan = [
        StagePlanItem(
            "h2h_inference",
            _action,
            required_outputs=(output,),
            completion_stamp=stamp,
        ),
        StagePlanItem("h2h_digest", _downstream),
    ]
    context = StageRunContext(config=cfg, manifest_path=manifest_path, run_label="pair")

    with pytest.raises(StageCompletionError, match=status):
        StageRunner.run(plan, context)

    assert calls == []
    assert not cfg.stage_dir("h2h_digest").exists()


def test_stage_runner_accepts_successful_declared_completion_stamp(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    output = cfg.h2h_pairwise_inference_path()
    stamp = stage_done_path(cfg.stage_dir("h2h_inference"), "h2h_inference")
    calls: list[str] = []

    def _action(_cfg: AppConfig) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("complete", encoding="utf-8")
        write_stage_done(
            stamp,
            inputs=[],
            outputs=[output],
            cfg=cfg,
            stage="h2h_inference",
        )

    StageRunner.run(
        [
            StagePlanItem(
                "h2h_inference",
                _action,
                required_outputs=(output,),
                completion_stamp=stamp,
            ),
            StagePlanItem("h2h_digest", lambda _cfg: calls.append("downstream")),
        ],
        StageRunContext(
            config=cfg,
            manifest_path=tmp_path / "success.jsonl",
            run_label="pair",
        ),
    )

    assert calls == ["downstream"]
    assert not cfg.stage_dir("h2h_digest").exists()
