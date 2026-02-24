import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import farkle.analysis.run_bonferroni_head2head as rb
from farkle.analysis.stage_state import read_stage_done, stage_done_path
from farkle.config import AppConfig


def _mock_sizing_result(
    games_per_strategy: int,
    *,
    games_per_strategy_uncapped: int | None = None,
    applied_floor: bool = False,
    applied_cap: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        games_per_strategy_uncapped=(
            games_per_strategy
            if games_per_strategy_uncapped is None
            else games_per_strategy_uncapped
        ),
        games_per_strategy=games_per_strategy,
        applied_floor=applied_floor,
        applied_cap=applied_cap,
    )


def _mock_simulated_games(
    strategies: list[str],
    seeds: list[int],
    winner: str | None = None,
) -> pd.DataFrame:
    winners = [winner if winner is not None else str(strategies[0])] * len(seeds)
    return pd.DataFrame(
        {
            "winner_strategy": winners,
            "P1_n_farkles": [1.0] * len(seeds),
            "P2_n_farkles": [2.0] * len(seeds),
            "P1_score": [300.0] * len(seeds),
            "P2_score": [250.0] * len(seeds),
        }
    )


def test_head2head_pipeline_error_payload_and_union_payload_validation() -> None:
    err = rb.Head2HeadPipelineError(
        "bad payload",
        error_code="invalid_union_candidates_schema",
        context={"field": "value"},
    )
    payload = err.to_payload()
    assert payload["type"] == "Head2HeadPipelineError"
    assert payload["error_code"] == "invalid_union_candidates_schema"
    assert payload["message"] == "bad payload"
    assert payload["context"] == {"field": "value"}

    with pytest.raises(rb.Head2HeadPipelineError, match="missing required fields"):
        rb._validate_union_candidates_payload(
            {
                "candidates": ["A", "B"],
                "ratings_count": 2,
                "metrics_count": 2,
                "combined_count": 2,
            }
        )

    with pytest.raises(rb.Head2HeadPipelineError, match="invalid field types"):
        rb._validate_union_candidates_payload(
            {
                "candidates": ["A", "B"],
                "ratings_count": "2",
                "metrics_count": 2,
                "combined_count": 2,
                "ratings_path": "ratings.parquet",
                "metrics_path": "metrics.parquet",
            }
        )

    with pytest.raises(rb.Head2HeadPipelineError, match="combined_count does not match"):
        rb._validate_union_candidates_payload(
            {
                "candidates": ["A", "B"],
                "ratings_count": 2,
                "metrics_count": 2,
                "combined_count": 1,
                "ratings_path": "ratings.parquet",
                "metrics_path": "metrics.parquet",
            }
        )


def test_player_metric_mean_and_weighted_mean_guardrails() -> None:
    df = pd.DataFrame({"P1_score": [100.0]})

    with pytest.raises(KeyError, match="Unsupported metric alias lookup"):
        rb._player_metric_mean(df, 1, "unknown")

    with pytest.raises(KeyError, match="Missing expected columns"):
        rb._player_metric_mean(df, 2, "farkles")

    assert math.isnan(rb._weighted_mean(1.0, 0, 2.0, 0))
    assert rb._weighted_mean(2.5, 2, 10.0, 0) == 2.5


def test_warn_legacy_stage_dirs_noop_when_layout_has_no_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    (cfg.analysis_dir / "99_head2head").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(type(cfg.stage_layout), "folder_for", lambda _self, _key: None)

    with caplog.at_level("WARNING"):
        rb._warn_legacy_stage_dirs(cfg, "head2head")

    assert not caplog.records


def test_load_top_strategies_warns_when_sort_column_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ratings = tmp_path / "ratings.parquet"
    metrics = tmp_path / "metrics.parquet"
    pd.DataFrame(
        {
            "strategy": pd.Series(dtype="string"),
            "mu": pd.Series(dtype="float64"),
        }
    ).to_parquet(ratings)
    pd.DataFrame({"strategy": ["B"], "win_rate": [0.75]}).to_parquet(metrics)

    with caplog.at_level("WARNING"):
        strategies, info = rb._load_top_strategies(ratings_path=ratings, metrics_path=metrics)

    assert strategies == ["B"]
    assert info["ratings_count"] == 0
    assert info["metrics_count"] == 1
    assert any("missing data" in rec.message for rec in caplog.records)


def test_run_bonferroni_head2head_uses_legacy_ratings_path_and_records_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text('{"A": 0, "B": 0}', encoding="utf-8")

    fallback_ratings = cfg.trueskill_stage_dir / "ratings_k_weighted.parquet"
    fallback_ratings.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"strategy": ["A"], "mu": [1.0]}).to_parquet(fallback_ratings)

    metrics_path = cfg.metrics_input_path("metrics.parquet")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"strategy": ["A"], "win_rate": [0.5]}).to_parquet(metrics_path)

    observed: dict[str, Path] = {}

    def _load_top(**kwargs):  # noqa: ANN001
        observed["ratings_path"] = kwargs["ratings_path"]
        observed["metrics_path"] = kwargs["metrics_path"]
        return (
            ["A", "B"],
            {
                "ratings_count": 2,
                "metrics_count": 2,
                "combined_count": 2,
                "ratings_path": str(kwargs["ratings_path"]),
                "metrics_path": str(kwargs["metrics_path"]),
            },
        )

    monkeypatch.setattr(rb, "_load_top_strategies", _load_top)
    monkeypatch.setattr(rb, "games_for_power", lambda **_: _mock_sizing_result(0))  # noqa: ANN001

    with caplog.at_level("WARNING"):
        rb.run_bonferroni_head2head(cfg=cfg)

    assert observed["ratings_path"] == fallback_ratings
    assert observed["metrics_path"] == metrics_path
    assert any("Using legacy pooled ratings path" in rec.message for rec in caplog.records)

    done = read_stage_done(stage_done_path(cfg.head2head_stage_dir, "bonferroni_head2head"))
    inputs = {Path(p) for p in done["inputs"]}  # type: ignore[arg-type]
    assert fallback_ratings in inputs
    assert metrics_path in inputs


def test_run_bonferroni_head2head_warns_when_selfplay_strategy_column_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text('{"A": 0, "B": 0}', encoding="utf-8")

    selfplay_path = cfg.head2head_path("bonferroni_selfplay_symmetry.parquet")
    selfplay_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"other": [1]}).to_parquet(selfplay_path)

    monkeypatch.setattr(rb, "games_for_power", lambda **_: _mock_sizing_result(1))  # noqa: ANN001
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

    real_read = rb.pd.read_parquet

    def _read(path, columns=None, *args, **kwargs):  # noqa: ANN001
        if Path(path).resolve() == selfplay_path.resolve() and columns == ["strategy"]:
            return pd.DataFrame({"other": [1]})
        return real_read(path, *args, columns=columns, **kwargs)

    monkeypatch.setattr(rb.pd, "read_parquet", _read)
    with caplog.at_level("WARNING"):
        rb.run_bonferroni_head2head(cfg=cfg)

    assert any(
        rec.message == "Existing self-play parquet missing strategy column"
        for rec in caplog.records
    )


def test_run_bonferroni_head2head_warns_on_unreadable_existing_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text('{"A": 0, "B": 0}', encoding="utf-8")

    pairwise_shard = cfg.head2head_stage_dir / "bonferroni_pairwise_shards" / "bonferroni_pairwise_shard_0000.parquet"
    pairwise_shard.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "players": 2,
                "seed": 0,
                "pair_id": 99,
                "a": "X",
                "b": "Y",
                "games": 1,
                "wins_a": 1,
                "wins_b": 0,
                "win_rate_a": 1.0,
                "pval_one_sided": 0.1,
                "mean_farkles_a": 1.0,
                "mean_farkles_b": 2.0,
                "mean_score_a": 300.0,
                "mean_score_b": 250.0,
            }
        ]
    ).to_parquet(pairwise_shard)

    selfplay_shard = cfg.head2head_stage_dir / "bonferroni_selfplay_shards" / "bonferroni_selfplay_shard_0000.parquet"
    selfplay_shard.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "players": 2,
                "seed": 0,
                "strategy": "legacy",
                "games": 1,
                "wins_seat1": 1,
                "wins_seat2": 0,
                "seat1_win_rate": 1.0,
                "seat2_win_rate": 0.0,
                "seat_win_rate_diff": 1.0,
                "mean_farkles_seat1": 1.0,
                "mean_farkles_seat2": 2.0,
                "mean_score_seat1": 300.0,
                "mean_score_seat2": 250.0,
            }
        ]
    ).to_parquet(selfplay_shard)

    monkeypatch.setattr(rb, "games_for_power", lambda **_: _mock_sizing_result(1))  # noqa: ANN001
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

    real_read = rb.pd.read_parquet

    def _read(path, columns=None, *args, **kwargs):  # noqa: ANN001
        candidate = Path(path).resolve()
        if candidate in {pairwise_shard.resolve(), selfplay_shard.resolve()} and columns is None:
            raise RuntimeError("unreadable shard")
        return real_read(path, *args, columns=columns, **kwargs)

    monkeypatch.setattr(rb.pd, "read_parquet", _read)
    with caplog.at_level("WARNING"):
        rb.run_bonferroni_head2head(cfg=cfg, shard_size=1)

    assert any(rec.message == "Failed to load shard data" for rec in caplog.records)
    assert any(rec.message == "Failed to load self-play shard data" for rec in caplog.records)


def test_run_bonferroni_head2head_completed_pairs_keep_ordered_output_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text('{"A": 0, "B": 0}', encoding="utf-8")

    shard_path = cfg.head2head_stage_dir / "bonferroni_pairwise_shards" / "bonferroni_pairwise_shard_0000.parquet"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "players": 2,
                "seed": 0,
                "pair_id": 0,
                "a": "A",
                "b": "B",
                "games": 1,
                "wins_a": 1,
                "wins_b": 0,
                "win_rate_a": 1.0,
                "pval_one_sided": 0.5,
                "mean_farkles_a": 1.0,
                "mean_farkles_b": 2.0,
                "mean_score_a": 300.0,
                "mean_score_b": 250.0,
            }
        ]
    ).to_parquet(shard_path)

    monkeypatch.setattr(rb, "games_for_power", lambda **_: _mock_sizing_result(1))  # noqa: ANN001
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

    rb.run_bonferroni_head2head(cfg=cfg, completed_pair_ids=[0])

    ordered = pd.read_parquet(cfg.head2head_path("bonferroni_pairwise_ordered.parquet"))
    assert ordered.empty
