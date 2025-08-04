from __future__ import annotations

from farkle.analysis_config import PipelineCfg


def test_git_sha_lazily_computed(monkeypatch):
    calls = 0

    def fake_load(self):
        nonlocal calls
        calls += 1
        return "abc123"

    monkeypatch.setattr(PipelineCfg, "_load_git_sha", fake_load)
    cfg = PipelineCfg()

    assert calls == 0

    assert cfg.git_sha == "abc123"
    assert calls == 1

    # Subsequent access uses cached value
    assert cfg.git_sha == "abc123"
    assert calls == 1
