import json
import logging

from farkle.utils.logging import configure_logging, setup_info_logging, setup_warning_logging
from farkle.utils.telemetry import SupervisorHeartbeatRecorder


def test_setup_info_logging_creates_file_and_sets_level(tmp_path):
    log_file = tmp_path / "info.log"
    setup_info_logging(log_file)
    root = logging.getLogger()
    assert log_file.exists()
    assert root.level == logging.INFO


def test_setup_warning_logging_creates_file_and_sets_level(tmp_path):
    log_file = tmp_path / "warn.log"
    setup_warning_logging(log_file)
    root = logging.getLogger()
    assert log_file.exists()
    assert root.level == logging.WARNING


def test_handlers_replaced_and_console_logging(tmp_path, capsys):
    log1 = tmp_path / "first.log"
    log2 = tmp_path / "second.log"

    setup_info_logging(log1)
    logging.info("first")
    capsys.readouterr()  # clear captured output

    setup_info_logging(log2)
    logging.info("second")
    captured = capsys.readouterr()
    assert "second" in captured.err

    root = logging.getLogger()
    assert any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log2) for h in root.handlers
    )
    assert not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log1) for h in root.handlers
    )

    assert "first" in log1.read_text()
    assert "second" not in log1.read_text()
    assert "second" in log2.read_text()


def test_progress_summary_fields_reach_console_and_file(tmp_path, capsys):
    log_file = tmp_path / "progress.log"
    setup_info_logging(log_file)
    recorder = SupervisorHeartbeatRecorder(
        logging.getLogger("farkle.pipeline"), run="smoke", interval_seconds=0
    )
    recorder.record_completion_summary(
        "metrics:k=2", stage="metrics", summary={"total_units": 8, "reused_units": 3}
    )
    recorder.close()
    console = capsys.readouterr().err
    file_text = log_file.read_text(encoding="utf-8")
    for output in (console, file_text):
        assert "Pipeline progress summary" in output
        fields = json.loads(output.split(" | ", 1)[1])
        assert fields["scope"] == "metrics:k=2"
        assert fields["stage"] == "metrics"
        assert fields["progress"] == {"total_units": 8, "reused_units": 3}


def test_structured_logging_preserves_exception_and_serializes_paths(tmp_path, capsys):
    log_file = tmp_path / "error.log"
    setup_info_logging(log_file)
    try:
        raise ValueError("publication failed")
    except ValueError:
        logging.getLogger("farkle.pipeline").exception(
            "Stage failed: %s", "rng_diagnostics", extra={"output_path": tmp_path}
        )
    for output in (capsys.readouterr().err, log_file.read_text(encoding="utf-8")):
        first, traceback = output.split("\n", 1)
        assert "Stage failed: rng_diagnostics" in first
        assert json.loads(first.split(" | ", 1)[1]) == {"output_path": str(tmp_path)}
        assert "Traceback (most recent call last):" in traceback
        assert "ValueError: publication failed" in traceback


def test_info_heartbeat_is_readable_and_debug_retains_snapshot(tmp_path, capsys):
    log_file = tmp_path / "heartbeat.log"
    fields = {"telemetry_kind": "heartbeat", "active_scopes": [{"progress": {"games": 42}}]}
    message = "Heartbeat: stage=metrics phase=partition_execution rss=250MiB"
    configure_logging(log_file=log_file)
    logging.getLogger("farkle.pipeline").info(message, extra=fields)
    for output in (capsys.readouterr().err, log_file.read_text(encoding="utf-8")):
        assert message in output
        assert "active_scopes" not in output
        assert " | " not in output
    configure_logging(level="DEBUG", log_file=log_file)
    logging.getLogger("farkle.pipeline").info(message, extra=fields)
    assert json.loads(capsys.readouterr().err.split(" | ", 1)[1]) == fields
    assert "active_scopes" in log_file.read_text(encoding="utf-8")


def test_info_cleanup_omits_bulk_history_but_preserves_errors(tmp_path, capsys):
    log_file = tmp_path / "cleanup.log"
    setup_info_logging(log_file)
    logging.getLogger("farkle.pipeline").info(
        "Pipeline telemetry stopped",
        extra={
            "telemetry_kind": "telemetry_cleanup",
            "heartbeat_count": 46,
            "dropped_updates": 0,
            "telemetry_errors": ["sampler failed"],
            "completed_progress": {"metrics": {"total_units": 999}},
            "resource_summary": {"process_tree_rss_bytes": 123456789},
        },
    )
    for output in (capsys.readouterr().err, log_file.read_text(encoding="utf-8")):
        fields = json.loads(output.split(" | ", 1)[1])
        assert fields["heartbeat_count"] == 46
        assert fields["telemetry_errors"] == ["sampler failed"]
        assert "completed_progress" not in fields
        assert "resource_summary" not in fields


def test_info_summaries_keep_totals_and_debug_keeps_details(tmp_path, capsys):
    log_file = tmp_path / "summary.log"
    progress = {
        "completed_games": 50,
        "planned_games": 50,
        "per_worker_attempts_per_second": [10, 20],
    }
    phases = [
        {
            "scope": "audit",
            "status": "success",
            "elapsed_seconds": 2,
            "resource_start": {"process_tree_rss_bytes": 123},
        }
    ]
    for level in ("INFO", "DEBUG"):
        configure_logging(level=level, log_file=log_file)
        logger = logging.getLogger("farkle.pipeline")
        logger.info(
            "Pipeline progress summary",
            extra={"telemetry_kind": "completion_summary", "progress": progress},
        )
        logger.info(
            "Pipeline finalization timing summary",
            extra={"telemetry_kind": "timing_summary", "finalization_phases": phases},
        )
        console = capsys.readouterr().err.splitlines()
        file_lines = log_file.read_text(encoding="utf-8").splitlines()[-2:]
        for lines in (console, file_lines):
            summary = json.loads(lines[0].split(" | ", 1)[1])["progress"]
            timing = json.loads(lines[1].split(" | ", 1)[1])["finalization_phases"]
            assert summary["completed_games"] == summary["planned_games"] == 50
            assert timing[0]["scope"] == "audit"
            assert ("per_worker_attempts_per_second" in summary) == (level == "DEBUG")
            assert ("resource_start" in timing[0]) == (level == "DEBUG")
