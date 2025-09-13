import importlib.util
import logging
import os
import time
from pathlib import Path

from farkle.analysis.analysis_config import PipelineCfg

_spec = importlib.util.spec_from_file_location(
    "head2head", Path(__file__).resolve().parents[3] / "src" / "farkle" / "analysis" / "head2head.py"
)
assert _spec is not None
assert _spec.loader is not None
head2head = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(head2head)


def test_run_skips_if_up_to_date(tmp_path, monkeypatch, caplog):
    cfg = PipelineCfg(results_dir=tmp_path)
    analysis_dir = cfg.analysis_dir
    data_dir = analysis_dir / "data"
    data_dir.mkdir(parents=True)

    curated = data_dir / cfg.curated_rows_name
    pairwise = analysis_dir / "bonferroni_pairwise.csv"

    now = time.time()
    curated.touch()
    pairwise.touch()
    os.utime(curated, (now - 10, now - 10))
    os.utime(pairwise, (now, now))

    called = False

    def fake_main(argv):  # noqa: ARG001
        nonlocal called
        called = True

    monkeypatch.setattr(head2head._h2h, "main", fake_main)

    with caplog.at_level(logging.INFO):
        head2head.run(cfg)

    assert not called
    assert "Head-to-Head: results up-to-date - skipped" in caplog.text


def test_run_handles_exception(tmp_path, monkeypatch, caplog):
    cfg = PipelineCfg(results_dir=tmp_path)
    analysis_dir = cfg.analysis_dir
    data_dir = analysis_dir / "data"
    data_dir.mkdir(parents=True)

    curated = data_dir / cfg.curated_rows_name
    pairwise = analysis_dir / "bonferroni_pairwise.csv"

    now = time.time()
    pairwise.touch()
    curated.touch()
    os.utime(pairwise, (now - 10, now - 10))
    os.utime(curated, (now, now))

    called = False

    def boom(argv):  # noqa: ARG001
        nonlocal called
        called = True
        raise RuntimeError("boom")

    monkeypatch.setattr(head2head._h2h, "main", boom)

    with caplog.at_level(logging.INFO):
        head2head.run(cfg)

    assert called
    assert any(
        rec.levelname == "WARNING" and "Head-to-Head: skipped" in rec.message
        for rec in caplog.records
    )
