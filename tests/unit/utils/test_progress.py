from __future__ import annotations

import logging
import math

from farkle.utils import progress


def test_progress_format_helpers_cover_edge_cases() -> None:
    normalized = progress.ProgressLogConfig(-1, -2, -3).normalized()

    assert normalized.frequent_interval_sec == 0.0
    assert normalized.info_phase_sec == 0.0
    assert normalized.ongoing_interval_sec == 0.0

    assert progress._advance_deadline(5.0, 0.0, 20.0) == math.inf
    assert progress._advance_deadline(5.0, 2.0, 10.0) == 11.0

    assert progress._format_duration(None) == "unknown"
    assert progress._format_duration(float("inf")) == "unknown"
    assert progress._format_duration(3661.2) == "1h01m01s"
    assert progress._format_duration(61.0) == "1m01s"
    assert progress._format_duration(-0.4) == "0s"

    assert progress._format_count(1_000) == "1,000"
    assert progress._format_count(1_000.0) == "1,000"
    assert progress._format_count(12.345) == "12.35"

    assert progress._format_rate(math.nan, "rows") == "rate unknown rows/s"
    assert progress._format_rate(150.0, "rows") == "rate 150 rows/s"
    assert progress._format_rate(15.5, "rows") == "rate 15.5 rows/s"
    assert progress._format_rate(1.234, "rows") == "rate 1.23 rows/s"


def test_scheduled_progress_logger_uses_info_and_debug_levels(
    monkeypatch,
    caplog,
) -> None:
    ticks = iter([5.0, 10.0, 25.0, 31.0])
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(ticks))

    logger = logging.getLogger("tests.progress.levels")
    logger.setLevel(logging.DEBUG)
    scheduled = progress.ScheduledProgressLogger(
        logger,
        label="work",
        schedule=progress.ProgressLogConfig(
            frequent_interval_sec=10.0,
            info_phase_sec=20.0,
            ongoing_interval_sec=30.0,
        ),
        unit="items",
        total=100,
        start_time=0.0,
    )

    with caplog.at_level(logging.DEBUG, logger="tests.progress.levels"):
        assert scheduled.maybe_log(1) is False
        assert scheduled.maybe_log(10, detail="first batch") is True
        assert scheduled.maybe_log(50) is True
        assert scheduled.maybe_log(80) is True

    levels = [record.levelno for record in caplog.records]
    messages = [record.getMessage() for record in caplog.records]

    assert levels == [logging.INFO, logging.DEBUG, logging.INFO]
    assert "10/100 items (10.0%)" in messages[0]
    assert "first batch" in messages[0]
    assert "50/100 items (50.0%)" in messages[1]
    assert "eta" in messages[2]


def test_scheduled_progress_logger_force_info_without_total(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(progress.time, "monotonic", lambda: 10.0)

    logger = logging.getLogger("tests.progress.force")
    logger.setLevel(logging.INFO)
    scheduled = progress.ScheduledProgressLogger(
        logger,
        label="export",
        schedule=progress.ProgressLogConfig(),
        unit="games",
        start_time=10.0,
    )

    with caplog.at_level(logging.INFO, logger="tests.progress.force"):
        assert scheduled.maybe_log(3.5, force_info=True) is True

    assert [record.levelno for record in caplog.records] == [logging.INFO]
    assert (
        caplog.records[0].getMessage()
        == "export progress:, 3.50 games, elapsed 0s, rate unknown games/s"
    )
