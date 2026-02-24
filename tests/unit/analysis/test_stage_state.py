import json
import os
from pathlib import Path

import pytest

from farkle.analysis.stage_state import (
    read_stage_done,
    stage_done_path,
    stage_is_up_to_date,
    write_stage_done,
)


def test_stage_up_to_date_respects_config_sha(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    done = stage_done_path(tmp_path, "demo")

    input_path.write_text("in")
    output_path.write_text("out")

    write_stage_done(done, inputs=[input_path], outputs=[output_path], config_sha="abc")

    assert stage_is_up_to_date(done, inputs=[input_path], outputs=[output_path], config_sha="abc")
    assert not stage_is_up_to_date(done, inputs=[input_path], outputs=[output_path], config_sha="other")


def test_stage_up_to_date_invalidates_on_newer_input(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    done = stage_done_path(tmp_path, "demo")

    input_path.write_text("in")
    output_path.write_text("out")
    write_stage_done(done, inputs=[input_path], outputs=[output_path], config_sha=None)

    assert stage_is_up_to_date(done, inputs=[input_path], outputs=[output_path], config_sha=None)

    input_path.write_text("updated")
    os.utime(input_path, (done.stat().st_mtime + 1, done.stat().st_mtime + 1))
    assert not stage_is_up_to_date(done, inputs=[input_path], outputs=[output_path], config_sha=None)


def test_read_stage_done_missing_invalid_and_default_status(tmp_path: Path) -> None:
    missing = tmp_path / "missing.done.json"
    payload = read_stage_done(missing)
    assert payload["status"] == "missing"
    assert payload["reason"] is None

    invalid = tmp_path / "invalid.done.json"
    invalid.write_text("{ not-json", encoding="utf-8")
    payload = read_stage_done(invalid)
    assert payload["status"] == "invalid"
    assert payload["reason"] == "invalid json"

    minimal = tmp_path / "minimal.done.json"
    minimal.write_text(json.dumps({"inputs": ["in"], "outputs": ["out"]}), encoding="utf-8")
    payload = read_stage_done(minimal)
    assert payload["status"] == "success"
    assert payload["inputs"] == ["in"]
    assert payload["outputs"] == ["out"]


def test_stage_up_to_date_rejects_non_success_and_missing_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    done = stage_done_path(tmp_path, "demo")
    input_path.write_text("in", encoding="utf-8")
    output_path.write_text("out", encoding="utf-8")

    write_stage_done(
        done,
        inputs=[input_path],
        outputs=[output_path],
        status="failed",
        reason="boom",
        blocking_dependency="dep",
        upstream_stage="upstream",
    )
    assert not stage_is_up_to_date(done, inputs=[input_path], outputs=[output_path], config_sha=None)

    write_stage_done(done, inputs=[input_path], outputs=[output_path], status="success")
    output_path.unlink()
    assert not stage_is_up_to_date(done, inputs=[input_path], outputs=[output_path], config_sha=None)


def test_write_stage_done_validation_errors(tmp_path: Path) -> None:
    done = stage_done_path(tmp_path, "demo")

    with pytest.raises(ValueError, match="Unsupported stage status"):
        write_stage_done(done, inputs=[], outputs=[], status="not-a-status")

    with pytest.raises(ValueError, match="blocking_dependency and upstream_stage are required"):
        write_stage_done(done, inputs=[], outputs=[], status="skipped")
