from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types

import yaml


class _Sentinel:
    pass


def test_analysis_config_fallback_without_pydantic(monkeypatch, tmp_path):
    """Ensure fallback dataclass base works when pydantic is unavailable."""

    sentinel = _Sentinel()
    original_pydantic = sys.modules.get("pydantic", sentinel)
    original_analysis = sys.modules.get("farkle.analysis.analysis_config", sentinel)

    sys.modules.pop("farkle.analysis.analysis_config", None)

    class _Missing(types.ModuleType):
        def __getattr__(self, name: str) -> object:  # pragma: no cover - attribute protocol
            raise ModuleNotFoundError("No module named 'pydantic'")

    monkeypatch.setitem(sys.modules, "pydantic", _Missing("pydantic"))

    try:
        analysis_config = importlib.import_module("farkle.analysis.analysis_config")
        experiment = analysis_config.Experiment(name="fallback", seed=7)
        io_cfg = analysis_config.IO(results_dir=tmp_path, analysis_subdir="fallback")
        cfg = analysis_config.Config(experiment=experiment, io=io_cfg)

        dumped = cfg.model_dump()
        assert dumped["experiment"]["name"] == "fallback"
        assert dumped["io"]["analysis_subdir"] == "fallback"

        pipeline_cfg = cfg.to_pipeline_cfg()
        assert pipeline_cfg.results_dir == tmp_path
        assert pipeline_cfg.analysis_subdir == "fallback"
    finally:
        sys.modules.pop("farkle.analysis.analysis_config", None)
        if original_pydantic is sentinel:
            sys.modules.pop("pydantic", None)
        else:
            sys.modules["pydantic"] = original_pydantic
        if original_analysis is sentinel:
            sys.modules.pop("farkle.analysis.analysis_config", None)
        else:
            sys.modules["farkle.analysis.analysis_config"] = original_analysis
            importlib.reload(original_analysis)


def test_pipeline_cfg_cli_and_helpers(tmp_path):
    analysis_config = importlib.import_module("farkle.analysis.analysis_config")

    argv = [
        "--results-dir",
        str(tmp_path),
        "--analysis-subdir",
        "custom",
        "-v",
        "--other-flag",
    ]
    cfg, ns, remaining = analysis_config.PipelineCfg.parse_cli(argv)

    assert cfg.results_dir == tmp_path
    assert cfg.analysis_subdir == "custom"
    assert cfg.log_level == "DEBUG"
    assert ns.verbose is True
    assert remaining == ["--other-flag"]

    assert cfg.analysis_dir == tmp_path / "custom"
    assert cfg.data_dir == tmp_path / "custom" / "data"

    raw_path = cfg.ingested_rows_raw(3)
    assert raw_path == cfg.data_dir / "3p" / "3p_ingested_rows.raw.parquet"
    assert raw_path.parent.exists()

    curated_path = cfg.ingested_rows_curated(4)
    assert curated_path == cfg.data_dir / "4p" / "4p_ingested_rows.parquet"

    manifest_path = cfg.manifest_for(5)
    assert manifest_path == cfg.data_dir / "5p" / "manifest_5p.json"

    logging = cfg.logging_params()
    assert logging == {"level": "DEBUG", "log_file": None}

    cols = cfg.wanted_ingest_cols(2)
    assert cols[:3] == ["winner", "n_rounds", "winning_score"]
    suffixes_p1 = {c.split("_", 1)[1] for c in cols if c.startswith("P1_")}
    suffixes_p2 = {c.split("_", 1)[1] for c in cols if c.startswith("P2_")}
    assert suffixes_p1 == suffixes_p2
    assert len(cols) == 3 + 2 * len(suffixes_p1)


def test_pipeline_cfg_curated_parquet_and_rows_for_ram(tmp_path):
    analysis_config = importlib.import_module("farkle.analysis.analysis_config")
    cfg = analysis_config.PipelineCfg(results_dir=tmp_path, analysis_subdir="analysis")

    legacy = cfg.analysis_dir / "data" / cfg.curated_rows_name
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy")
    assert cfg.curated_parquet == legacy

    combined = cfg.data_dir / "all_n_players_combined" / "all_ingested_rows.parquet"
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.write_text("combined")
    assert cfg.curated_parquet == combined

    combined.unlink()
    legacy.unlink()
    assert cfg.curated_parquet == combined

    assert analysis_config.rows_for_ram(1, 1_000) == 10_000
    expected = int((50 * 1024**2) / (20 * 4 * 1.5))
    assert analysis_config.rows_for_ram(50, 20) == expected


def test_load_config_assigns_sha(tmp_path):
    analysis_config = importlib.import_module("farkle.analysis.analysis_config")

    payload = {
        "experiment": {"name": "demo", "seed": 3},
        "io": {"results_dir": str(tmp_path / "results"), "analysis_subdir": "analysis"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))

    cfg, sha = analysis_config.load_config(path)
    assert cfg.config_sha == sha

    expected_cfg = analysis_config.Config(**payload)
    dumped = json.dumps(expected_cfg.model_dump(), sort_keys=True, default=str).encode()
    expected_sha = hashlib.sha256(dumped).hexdigest()[:12]
    assert sha == expected_sha
