"""Tests for stage runner manifest and artifact validation behavior."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.analysis.stage_runner import (
    StageCompletionError,
    StagePlanItem,
    StageRunContext,
    StageRunner,
    StageValidationError,
)
from farkle.config import AppConfig, ArtifactScope, IOConfig, assign_config_sha
from farkle.orchestration.run_contexts import SeedRunContext, write_run_context_atomic
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.authenticated_contract import CodeIdentity, CodeIdentityPolicy
from farkle.utils.parallel import process_map, resolve_mp_context
from farkle.utils.stage_completion import stage_done_path, write_stage_done
from farkle.utils.telemetry import SupervisorHeartbeatRecorder


def _bounded_synthetic_worker(value: int) -> tuple[int, int]:
    time.sleep(0.04)
    return os.getpid(), value * value


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.origin = datetime(2026, 1, 1, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.value

    def utc_now(self) -> datetime:
        return self.origin + timedelta(seconds=self.value)

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _fixed_resources() -> dict[str, object]:
    return {
        "process_tree_rss_bytes": 100,
        "peak_process_tree_rss_bytes": 100,
        "host_available_memory_bytes": 1_000,
    }


def _manifest_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_authenticated_output(cfg: AppConfig, output: Path) -> None:
    table = pa.table({"value": [1]})
    write_parquet_artifact_atomic(
        table,
        output,
        sidecar=make_artifact_sidecar(
            cfg,
            output,
            producer="h2h_inference",
            scope=ArtifactScope.H2H_2P,
            source_scope=ArtifactScope.H2H_2P,
            operation="test_h2h_inference_output",
            consistency_columns=table.schema.names,
            player_counts=[2],
            required_player_counts=[2],
            missing_cell_policy="fail",
        ),
    )


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
        _write_authenticated_output(cfg, output)
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

    expected_errors = (
        (StageCompletionError, ArtifactContractError)
        if status == "failed"
        else StageCompletionError
    )
    with pytest.raises(expected_errors):
        StageRunner.run(plan, context)

    assert calls == []
    assert not cfg.stage_dir("h2h_digest").exists()


def test_stage_runner_accepts_successful_declared_completion_stamp(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    output = cfg.h2h_pairwise_inference_path()
    stamp = stage_done_path(cfg.stage_dir("h2h_inference"), "h2h_inference")
    calls: list[str] = []

    def _action(_cfg: AppConfig) -> None:
        _write_authenticated_output(cfg, output)
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


def test_stage_runner_validates_declared_stage_specific_freshness(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    output = cfg.h2h_pairwise_inference_path()
    stamp = stage_done_path(cfg.stage_dir("h2h_inference"), "h2h_inference")
    freshness_key = {**cfg.freshness_key(), "method_version": 99}

    def _action(_cfg: AppConfig) -> None:
        _write_authenticated_output(cfg, output)
        write_stage_done(
            stamp,
            inputs=[],
            outputs=[output],
            cfg=cfg,
            stage="h2h_inference",
            freshness_key=freshness_key,
        )

    result = StageRunner.run(
        [
            StagePlanItem(
                "h2h_inference",
                _action,
                required_outputs=(output,),
                completion_stamp=stamp,
                freshness_key=freshness_key,
            )
        ],
        StageRunContext(
            config=cfg,
            manifest_path=tmp_path / "custom_freshness.jsonl",
            run_label="pair",
        ),
    )

    assert result.failed_steps == []


def test_stage_runner_records_separate_phase_timings_without_changing_manifest_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import farkle.analysis.stage_runner as stage_runner_module

    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    output = tmp_path / "output.bin"
    stamp = tmp_path / "stage.done.json"
    events: list[dict[str, object]] = []
    clock = _FakeClock()
    recorder = SupervisorHeartbeatRecorder(
        logging.getLogger("tests.stage_runner.timing"),
        run="timed",
        interval_seconds=45.0,
        resource_sampler=_fixed_resources,
        clock=clock.monotonic,
        utc_clock=clock.utc_now,
        autostart=False,
    )

    monkeypatch.setattr(stage_runner_module, "validate_manifest_contract", lambda _path: None)

    def append(_path: Path, payload: dict[str, object], **_kwargs: object) -> None:
        events.append(dict(payload))
        clock.advance(1.0)

    def action(_cfg: AppConfig) -> None:
        output.write_bytes(b"canonical")
        clock.advance(2.0)

    def read(_path: Path) -> dict[str, object]:
        clock.advance(3.0)
        return {"status": "success"}

    def resolve(*_args: object, **_kwargs: object) -> object:
        clock.advance(4.0)
        return stage_runner_module.CompletionState.COMPLETE_VALID

    monkeypatch.setattr(stage_runner_module, "append_manifest_event", append)
    monkeypatch.setattr(stage_runner_module, "read_stage_done", read)
    monkeypatch.setattr(stage_runner_module, "resolve_stage_state", resolve)

    result = StageRunner.run(
        [
            StagePlanItem(
                "timed",
                action,
                required_outputs=(output,),
                completion_stamp=stamp,
            )
        ],
        StageRunContext(
            config=cfg,
            manifest_path=tmp_path / "manifest.jsonl",
            run_label="timed",
            telemetry=recorder,
            monotonic_clock=clock.monotonic,
            utc_clock=clock.utc_now,
        ),
    )

    timing = result.stage_timings[0]
    assert timing.stage_start_publication_seconds == 1.0
    assert timing.action_seconds == 2.0
    assert timing.output_validation_seconds == 0.0
    assert timing.completion_read_seconds == 3.0
    assert timing.authentication_seconds == 4.0
    assert timing.stage_end_publication_seconds == 1.0
    assert timing.status == "success"
    assert timing.resource_start["process_tree_rss_bytes"] == 100
    assert all("timing" not in json.dumps(event).lower() for event in events)
    assert all("heartbeat" not in json.dumps(event).lower() for event in events)


def test_stage_runner_times_failed_action_and_preserves_failure_behavior(
    tmp_path: Path,
) -> None:
    clock = _FakeClock()
    recorder = SupervisorHeartbeatRecorder(
        logging.getLogger("tests.stage_runner.failure_timing"),
        run="failure",
        interval_seconds=45.0,
        resource_sampler=_fixed_resources,
        clock=clock.monotonic,
        utc_clock=clock.utc_now,
        autostart=False,
    )

    def fail(_cfg: AppConfig) -> None:
        clock.advance(2.5)
        raise RuntimeError("expected")

    result = StageRunner.run(
        [StagePlanItem("failure", fail)],
        StageRunContext(
            config=AppConfig(io=IOConfig(results_dir_prefix=tmp_path)),
            manifest_path=tmp_path / "failure.jsonl",
            run_label="failure",
            telemetry=recorder,
            monotonic_clock=clock.monotonic,
            utc_clock=clock.utc_now,
        ),
        raise_on_failure=False,
    )

    assert result.failed_steps == ["failure"]
    assert isinstance(result.first_failure, RuntimeError)
    assert result.stage_timings[0].status == "failed"
    assert result.stage_timings[0].action_seconds == 2.5


def test_bounded_workers_never_emit_heartbeat_logs(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    logger = logging.getLogger("tests.stage_runner.parent_heartbeat")
    recorder = SupervisorHeartbeatRecorder(
        logger,
        run="bounded",
        interval_seconds=0.005,
        resource_sampler=_fixed_resources,
    )
    worker_pids: list[int] = []

    def action(_cfg: AppConfig) -> None:
        results = list(
            process_map(
                _bounded_synthetic_worker,
                range(4),
                n_jobs=2,
                window=2,
                mp_context=resolve_mp_context("spawn"),
            )
        )
        worker_pids.extend(pid for pid, _value in results)

    with caplog.at_level(logging.INFO, logger=logger.name):
        StageRunner.run(
            [StagePlanItem("bounded", action)],
            StageRunContext(
                config=AppConfig(io=IOConfig(results_dir_prefix=tmp_path)),
                manifest_path=tmp_path / "bounded.jsonl",
                run_label="bounded",
                telemetry=recorder,
            ),
        )
    recorder.close()

    heartbeats = [
        record for record in caplog.records if getattr(record, "telemetry_kind", None) == "heartbeat"
    ]
    assert heartbeats
    assert worker_pids
    assert all(pid != os.getpid() for pid in worker_pids)
    assert {record.__dict__["owner_pid"] for record in heartbeats} == {os.getpid()}
    assert {record.process for record in heartbeats} == {os.getpid()}


def test_telemetry_does_not_change_authenticated_fixture_outputs_or_identity(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    assign_config_sha(cfg)
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    context = SeedRunContext.from_config(cfg)
    write_run_context_atomic(
        context,
        code_identity=CodeIdentity(
            commit="a" * 40,
            policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY.value,
            state="development_dirty",
            dirty_fingerprint_sha256="b" * 64,
        ),
    )
    run_context_before = context.run_context_path.read_bytes()
    output = cfg.h2h_pairwise_inference_path()
    stamp = stage_done_path(cfg.stage_dir("h2h_inference"), "h2h_inference")
    freshness_before = cfg.freshness_key()
    stage_sha_before = cfg.stage_config_sha("h2h_inference")

    def action(_cfg: AppConfig) -> None:
        _write_authenticated_output(cfg, output)
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
                action,
                required_outputs=(output,),
                completion_stamp=stamp,
            )
        ],
        StageRunContext(
            config=cfg,
            manifest_path=tmp_path / "disabled.jsonl",
            run_label="disabled",
            heartbeat_interval_seconds=0.0,
        ),
    )
    baseline = {
        "data": sha256_file(output),
        "sidecar": sha256_file(sidecar_path(output)),
        "completion": sha256_file(stamp),
    }

    output.unlink()
    sidecar_path(output).unlink()
    stamp.unlink()
    StageRunner.run(
        [
            StagePlanItem(
                "h2h_inference",
                action,
                required_outputs=(output,),
                completion_stamp=stamp,
            )
        ],
        StageRunContext(
            config=cfg,
            manifest_path=tmp_path / "enabled.jsonl",
            run_label="enabled",
            heartbeat_interval_seconds=0.005,
        ),
    )
    observed = {
        "data": sha256_file(output),
        "sidecar": sha256_file(sidecar_path(output)),
        "completion": sha256_file(stamp),
    }

    assert observed == baseline
    assert cfg.freshness_key() == freshness_before
    assert cfg.stage_config_sha("h2h_inference") == stage_sha_before
    assert context.run_context_path.read_bytes() == run_context_before
    authenticated_text = sidecar_path(output).read_text(encoding="utf-8") + stamp.read_text(
        encoding="utf-8"
    )
    assert "heartbeat" not in authenticated_text.lower()
    assert "timing_summary" not in authenticated_text.lower()
    assert "worker_messages_received" not in authenticated_text.lower()
    assert "completed_progress" not in authenticated_text.lower()
    assert "heartbeat" not in run_context_before.decode("utf-8").lower()
    assert "timing_summary" not in run_context_before.decode("utf-8").lower()
    for manifest in (tmp_path / "disabled.jsonl", tmp_path / "enabled.jsonl"):
        text = manifest.read_text(encoding="utf-8").lower()
        assert "heartbeat" not in text
        assert "timing_summary" not in text
        assert "worker_messages_received" not in text
        assert "completed_progress" not in text
