from __future__ import annotations

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pyarrow")

from farkle.analysis.analysis_config import PipelineCfg
from farkle.app_config import AppConfig


def test_app_config_parse_cli_delegates(monkeypatch):
    sentinel_cfg = object()
    sentinel_ns = object()
    sentinel_remaining = ["--pipeline", "extra"]
    observed_args: list[object] = []

    def fake_parse_cli(cls, argv=None):  # type: ignore[override]
        observed_args.append(argv)
        return sentinel_cfg, sentinel_ns, sentinel_remaining

    monkeypatch.setattr(PipelineCfg, "parse_cli", classmethod(fake_parse_cli))

    argv = ["--results-dir", "data"]
    cfg, namespace, remaining = AppConfig.parse_cli(argv)

    assert observed_args == [argv]
    assert cfg.analysis is sentinel_cfg
    assert namespace is sentinel_ns
    assert remaining is sentinel_remaining
