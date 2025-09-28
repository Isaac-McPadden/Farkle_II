import logging

from farkle.utils.logging import setup_info_logging, setup_warning_logging


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
