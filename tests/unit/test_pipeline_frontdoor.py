from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from farkle.config import AppConfig, IOConfig
from farkle.orchestration.pipeline import (
    _detect_player_counts,
    _done_path,
    _first_existing,
    analyze_trueskill,
    fingerprint,
    is_up_to_date,
    write_done,
)


def test_fingerprint_hash_and_directory_handling(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello")
    dir_path = tmp_path / "dir"
    dir_path.mkdir()

    result = fingerprint([file_path, dir_path])
    assert result[0]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert "sha256" not in result[1]


def test_is_up_to_date_tracks_inputs_and_outputs(tmp_path: Path) -> None:
    done_path = tmp_path / "artifact.done.json"
    inp = tmp_path / "input.txt"
    out = tmp_path / "out.txt"
    inp.write_text("input")
    out.write_text("output")
    write_done(done_path, [inp], [out], "tool")
    assert is_up_to_date(done_path, [inp], [out])

    # Touch the input to make it newer than the done file
    now = datetime.now().timestamp()
    done_ts = done_path.stat().st_mtime
    new_ts = max(now, done_ts) + 10
    inp.touch()
    os.utime(inp, (new_ts, new_ts))
    assert not is_up_to_date(done_path, [inp], [out])


def test_write_done_serializes_expected_payload(tmp_path: Path) -> None:
    done_path = tmp_path / "done.json"
    inputs = [tmp_path / "input.txt"]
    outputs = [tmp_path / "out.txt"]
    inputs[0].write_text("in")
    outputs[0].write_text("out")

    write_done(done_path, inputs, outputs, "toolname")
    stamp = json.loads(done_path.read_text())
    assert stamp["tool"] == "toolname"
    assert stamp["inputs"][0]["path"] == str(inputs[0])
    assert stamp["outputs"] == [{"path": str(outputs[0])}]


def test_first_existing_prefers_first_existing(tmp_path: Path) -> None:
    p1 = tmp_path / "missing"
    p2 = tmp_path / "exists"
    p2.write_text("ok")
    assert _first_existing([p1, p2]) == p2
    assert _first_existing([p1]) == p1


def test_detect_player_counts_reads_parquet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "exp"))
    metrics = cfg.metrics_stage_dir / "metrics.parquet"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.touch()

    df = pd.DataFrame({"n_players": [None, 2, 5]})
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "pandas" else None)
    monkeypatch.setattr("pandas.read_parquet", lambda *_args, **_kwargs: df)

    counts = _detect_player_counts(cfg.analysis_dir)
    assert counts == [2, 5]


def test_analyze_trueskill_skips_when_up_to_date(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "input.txt").write_text("data")
    cfg = AppConfig(io=IOConfig(results_dir_prefix=exp_dir))
    out = cfg.trueskill_stage_dir / "tiers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("{}")
    done = _done_path(out)
    write_done(done, [exp_dir / "input.txt"], [out], "tool")

    analyze_trueskill(exp_dir)
    captured = capsys.readouterr().out
    assert "SKIP trueskill" in captured


def test_analyze_trueskill_runs_current_entrypoint_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "input.txt").write_text("data")
    cfg = AppConfig(io=IOConfig(results_dir_prefix=exp_dir))

    expected_out = cfg.trueskill_stage_dir / "tiers.json"
    expected_out.parent.mkdir(parents=True, exist_ok=True)

    def fake_run(cfg: Any) -> None:  # noqa: ANN401
        cfg.trueskill_stage_dir.mkdir(parents=True, exist_ok=True)
        (cfg.trueskill_stage_dir / "tiers.json").write_text(json.dumps({"current": True}))

    monkeypatch.setattr(
        "farkle.analysis.run_trueskill.run_trueskill_all_seeds",
        fake_run,
        raising=True,
    )
    analyze_trueskill(exp_dir)

    out = expected_out
    done = _done_path(out)
    assert out.exists()
    assert done.exists()
