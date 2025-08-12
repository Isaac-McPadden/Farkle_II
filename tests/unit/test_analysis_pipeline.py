import os
from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
import pytest

from farkle.analysis_config import PipelineCfg

pipeline = pytest.importorskip("pipeline")


def _write_fixture(root: Path) -> None:
    """Create a minimal results block under *root* for two players."""
    block = root / "2_players"
    block.mkdir()
    np.save(block / "keepers_2.npy", np.array(["A", "B"]))
    df = pd.DataFrame(
        {
            "winner": ["P1", "P2"],
            "n_rounds": [5, 6],
            "winning_score": [1000, 1100],
            "P1_strategy": ["A", "A"],
            "P2_strategy": ["B", "B"],
            "P1_rank": [1, 2],
            "P2_rank": [2, 1],
        }
    )
    df.to_csv(block / "winners.csv", index=False)


def test_pipeline_all_creates_outputs(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        pipeline.main(["all", "--root", str(tmp_path)])
    finally:
        os.chdir(cwd)

    analysis = tmp_path / "analysis"
    p2_dir = analysis / "data" / "2p"
    assert (p2_dir / "2p_ingested_rows.parquet").exists()
    assert not (p2_dir / "2p_ingested_rows.raw.parquet").exists()
    combined = analysis / "data" / "all_n_players_combined" / "all_ingested_rows.parquet"
    assert combined.exists()
    assert (analysis / "metrics.parquet").exists()
    assert (analysis / "seat_advantage.csv").exists()

    # analytics artefacts
    assert (tmp_path / "ratings_pooled.pkl").exists()
    assert (tmp_path / "hgb_importance.json").exists()
    figs = tmp_path / "notebooks" / "figs"
    assert any(figs.glob("pd_*.png"))


def test_pipeline_ingest_only(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        pipeline.main(["ingest", "--root", str(tmp_path)])
    finally:
        os.chdir(cwd)

    analysis = tmp_path / "analysis"
    p2_dir = analysis / "data" / "2p"
    raw = p2_dir / "2p_ingested_rows.raw.parquet"
    curated = p2_dir / "2p_ingested_rows.parquet"
    assert raw.exists()
    assert not curated.exists()
    assert not (analysis / "metrics.parquet").exists()
    assert not (tmp_path / "hgb_importance.json").exists()
    combined = analysis / "data" / "all_n_players_combined" / "all_ingested_rows.parquet"
    assert not combined.exists()


def test_pipeline_aggregate_only(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        pipeline.main(["ingest", "--root", str(tmp_path)])
        pipeline.main(["curate", "--root", str(tmp_path)])
        pipeline.main(["aggregate", "--root", str(tmp_path)])
    finally:
        os.chdir(cwd)

    combined = tmp_path / "analysis" / "data" / "all_n_players_combined" / "all_ingested_rows.parquet"
    assert combined.exists()


def test_pipeline_missing_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_fixture(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        def _boom(cfg):  # simulate analytics dependency failure  # noqa: ARG001
            raise RuntimeError("missing dependency")

        monkeypatch.setattr("farkle.analytics.run_all", _boom)
        with pytest.raises(RuntimeError):
            pipeline.main(["all", "--root", str(tmp_path)])
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize(
    "command,target",
    [
        ("ingest", "farkle.ingest.run"),
        ("curate", "farkle.curate.run"),
        ("aggregate", "farkle.aggregate.run"),
        ("metrics", "farkle.metrics.run"),
        ("analytics", "farkle.analytics.run_all"),
    ],
)
def test_individual_step_failure_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    target: str,
) -> None:
    log_file = tmp_path / "pipeline.log"

    def _boom(cfg: PipelineCfg) -> None:  # noqa: ARG001
        logging.getLogger().debug("running %s", command)
        raise RuntimeError("boom")

    monkeypatch.setattr(target, _boom)

    def _parse_cli(cls, argv):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--root", dest="results_dir", type=Path, default=tmp_path)
        parser.add_argument("-v", "--verbose", action="store_true")
        parser.add_argument("--log-file", type=Path)
        ns, remaining = parser.parse_known_args(argv)
        cfg = PipelineCfg(results_dir=ns.results_dir)
        cfg.log_file = ns.log_file
        if ns.verbose:
            cfg.log_level = "DEBUG"
        return cfg, ns, remaining

    monkeypatch.setattr(PipelineCfg, "parse_cli", classmethod(_parse_cli))

    rc = pipeline.main(
        [command, "--root", str(tmp_path), "--verbose", "--log-file", str(log_file)]
    )
    assert rc == 1
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG
    assert log_file.exists()
    assert f"running {command}" in log_file.read_text()


def test_pipeline_all_step_failure_prints_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _ok(cfg: PipelineCfg) -> None:  # noqa: ARG001
        return None

    def _boom(cfg: PipelineCfg) -> None:  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr("farkle.ingest.run", _ok)
    monkeypatch.setattr("farkle.curate.run", _ok)
    monkeypatch.setattr("farkle.aggregate.run", _boom)

    with pytest.raises(RuntimeError):
        pipeline.main(["all", "--root", str(tmp_path)])

    err = capsys.readouterr().err
    assert "aggregate step failed: boom" in err
