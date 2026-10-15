import json
import logging

from farkle.utils.logging import setup_info_logging, setup_warning_logging
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
