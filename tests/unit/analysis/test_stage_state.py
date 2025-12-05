import os
from pathlib import Path

from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done


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
