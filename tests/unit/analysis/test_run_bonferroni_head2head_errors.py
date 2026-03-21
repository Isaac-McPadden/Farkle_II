from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import farkle.analysis.run_bonferroni_head2head as rb
from farkle.config import AppConfig


def _mock_sizing_result(games_per_strategy: int) -> SimpleNamespace:
    return SimpleNamespace(
        games_per_strategy_uncapped=games_per_strategy,
        games_per_strategy=games_per_strategy,
        applied_floor=False,
        applied_cap=False,
    )


def _mock_simulated_games(strategies: list[str], seeds: list[int]) -> pd.DataFrame:
    winners = [str(strategies[0])] * len(seeds)
    return pd.DataFrame(
        {
            "winner_strategy": winners,
            "P1_n_farkles": [1.0] * len(seeds),
            "P2_n_farkles": [2.0] * len(seeds),
            "P1_score": [300.0] * len(seeds),
            "P2_score": [250.0] * len(seeds),
        }
    )


def _prepare_minimal_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0}), encoding="utf-8")

    monkeypatch.setattr(rb, "games_for_power", lambda **_: _mock_sizing_result(1))
    monkeypatch.setattr(
        rb,
        "_load_top_strategies",
        lambda **_: (
            ["A", "B"],
            {
                "ratings_count": 0,
                "metrics_count": 0,
                "combined_count": 2,
                "ratings_path": "",
                "metrics_path": "",
            },
        ),
    )
    monkeypatch.setattr(rb, "parse_strategy_identifier", lambda name, **_: name)
    monkeypatch.setattr(
        rb,
        "simulate_many_games_from_seeds",
        lambda seeds, strategies, n_jobs: _mock_simulated_games(strategies, seeds),
    )
    return cfg


def test_run_bonferroni_warns_when_completed_selfplay_ids_fail_to_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _prepare_minimal_run(tmp_path, monkeypatch)
    selfplay_path = cfg.head2head_path("bonferroni_selfplay_symmetry.parquet")
    selfplay_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"strategy": ["legacy"]}).to_parquet(selfplay_path)

    real_read = rb.pd.read_parquet

    def flaky_read(path, columns=None, *args, **kwargs):  # noqa: ANN001
        if Path(path).resolve() == selfplay_path.resolve() and columns == ["strategy"]:
            raise RuntimeError("cannot read selfplay ids")
        return real_read(path, columns=columns, *args, **kwargs)

    monkeypatch.setattr(rb.pd, "read_parquet", flaky_read)

    with caplog.at_level("WARNING"):
        rb.run_bonferroni_head2head(cfg=cfg)

    assert any(
        record.message == "Failed to load completed self-play strategy ids"
        for record in caplog.records
    )


def test_run_bonferroni_warns_when_completed_ordered_ids_missing_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _prepare_minimal_run(tmp_path, monkeypatch)
    ordered_path = cfg.head2head_path("bonferroni_pairwise_ordered.parquet")
    ordered_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"other": [1]}).to_parquet(ordered_path)

    real_read = rb.pd.read_parquet

    def flaky_read(path, columns=None, *args, **kwargs):  # noqa: ANN001
        if Path(path).resolve() == ordered_path.resolve() and columns == ["pair_id", "ordering"]:
            return pd.DataFrame({"pair_id": [1]})
        return real_read(path, columns=columns, *args, **kwargs)

    monkeypatch.setattr(rb.pd, "read_parquet", flaky_read)

    with caplog.at_level("WARNING"):
        rb.run_bonferroni_head2head(cfg=cfg)

    assert any(
        record.message == "Existing ordered parquet missing pair_id/ordering columns"
        for record in caplog.records
    )


def test_run_bonferroni_raises_when_ordered_merge_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _prepare_minimal_run(tmp_path, monkeypatch)
    real_read = rb.pd.read_parquet

    def flaky_read(path, columns=None, *args, **kwargs):  # noqa: ANN001
        candidate = Path(path)
        if "bonferroni_pairwise_ordered_shards" in candidate.as_posix() and columns is None:
            raise RuntimeError("cannot read ordered shard")
        return real_read(path, columns=columns, *args, **kwargs)

    monkeypatch.setattr(rb.pd, "read_parquet", flaky_read)

    with pytest.raises(
        rb.Head2HeadPipelineError,
        match="No ordered pairwise frames available",
    ):
        rb.run_bonferroni_head2head(cfg=cfg, shard_size=1)


def test_run_bonferroni_raises_when_selfplay_merge_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _prepare_minimal_run(tmp_path, monkeypatch)
    real_read = rb.pd.read_parquet

    def flaky_read(path, columns=None, *args, **kwargs):  # noqa: ANN001
        candidate = Path(path)
        if "bonferroni_selfplay_shards" in candidate.as_posix() and columns is None:
            raise RuntimeError("cannot read selfplay shard")
        return real_read(path, columns=columns, *args, **kwargs)

    monkeypatch.setattr(rb.pd, "read_parquet", flaky_read)

    with pytest.raises(
        rb.Head2HeadPipelineError,
        match="No self-play frames available",
    ):
        rb.run_bonferroni_head2head(cfg=cfg, shard_size=1)
