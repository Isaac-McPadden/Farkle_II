from __future__ import annotations

import time
from pathlib import Path

from farkle.utils.stage_completion import (
    read_stage_done,
    stage_done_path,
    stage_is_up_to_date,
    write_stage_done,
)


def test_stage_done_roundtrip_and_status_handling(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    done = stage_done_path(stage_dir, "metrics")
    inp = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    inp.write_text("in")
    out.write_text("out")

    assert not stage_is_up_to_date(done, [inp], [out], config_sha="abc")

    write_stage_done(done, inputs=[inp], outputs=[out], config_sha="abc")
    assert stage_is_up_to_date(done, [inp], [out], config_sha="abc")
    assert not stage_is_up_to_date(done, [inp], [out], config_sha="other")

    meta = read_stage_done(done)
    assert meta["status"] == "success"


def test_stage_is_up_to_date_false_when_input_newer_or_skipped(tmp_path: Path) -> None:
    done = tmp_path / "x.done.json"
    inp = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    inp.write_text("in")
    out.write_text("out")

    write_stage_done(done, inputs=[inp], outputs=[out])
    assert stage_is_up_to_date(done, [inp], [out])

    time.sleep(0.01)
    inp.write_text("newer")
    assert not stage_is_up_to_date(done, [inp], [out])

    write_stage_done(
        done,
        inputs=[inp],
        outputs=[out],
        status="skipped",
        blocking_dependency="y",
        upstream_stage="z",
    )
    assert not stage_is_up_to_date(done, [inp], [out])
