import json
from pathlib import Path

import pandas as pd
import pytest

import farkle.analysis.run_bonferroni_head2head as rb
from farkle.analysis.stage_registry import StageDefinition, StageLayout, StagePlacement
from farkle.analysis.stage_state import read_stage_done, stage_done_path
from farkle.config import AppConfig
from farkle.simulation.simulation import simulate_many_games_from_seeds
from farkle.simulation.strategies import ThresholdStrategy


def _mock_simulated_games(strategies: list[str], seeds: list[int], winner: str | None = None) -> pd.DataFrame:
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


def _mock_simulated_games_legacy_farkles(
    strategies: list[str], seeds: list[int], winner: str | None = None
) -> pd.DataFrame:
    winners = [winner if winner is not None else str(strategies[0])] * len(seeds)
    return pd.DataFrame(
        {
            "winner_strategy": winners,
            "P1_farkles": [3.0] * len(seeds),
            "P2_farkles": [4.0] * len(seeds),
            "P1_score": [300.0] * len(seeds),
            "P2_score": [250.0] * len(seeds),
        }
    )


def test_simulate_many_games_from_seeds(monkeypatch):
    def fake_play(seed, strategies, target_score=10_000):  # noqa: ARG001
        return {
            "winner": f"P{seed}",
            "winning_score": seed,
            "n_rounds": seed,
            "P1_strategy": str(strategies[0]),
        }

    import farkle.simulation.simulation as sim

    monkeypatch.setattr(sim, "_play_game", fake_play, raising=True)
    seeds1 = [1, 2, 3]
    seeds2 = [4, 5, 6]
    strat = ThresholdStrategy(300, 1, True, True)
    df1 = simulate_many_games_from_seeds(seeds=seeds1, strategies=[strat], n_jobs=1)
    df2 = simulate_many_games_from_seeds(seeds=seeds2, strategies=[strat], n_jobs=1)

    assert len(df1) == len(seeds1)
    assert len(df2) == len(seeds2)
    assert not df1.equals(df2)


def test_run_bonferroni_head2head_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An informative error is raised when tiers.json is absent."""
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Tier file not found"):
        rb.run_bonferroni_head2head(seed=1, cfg=cfg)


def test_run_bonferroni_head2head_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "tiers.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="No tiers found"):
        rb.run_bonferroni_head2head(cfg=cfg)


def test_count_pair_wins_prefers_strategy_column() -> None:
    df = pd.DataFrame({"winner_strategy": ["A", "B", "A", "A"]})
    wins = rb._count_pair_wins(df, "A", "B")
    assert wins == (3, 1)


def test_count_pair_wins_falls_back_to_seat_column() -> None:
    df = pd.DataFrame({"winner": ["P1", "P2", "P2", "P1", "P1"]})
    wins = rb._count_pair_wins(df, "A", "B")
    assert wins == (3, 2)


def test_run_bonferroni_head2head_resumes_and_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"S1": 0, "S2": 0, "S3": 0}))

    existing = pd.DataFrame(
        [
            {
                "players": 2,
                "seed": 0,
                "pair_id": 0,
                "a": "S1",
                "b": "S2",
                "games": 1,
                "wins_a": 1,
                "wins_b": 0,
                "win_rate_a": 1.0,
                "pval_one_sided": 0.5,
            }
        ]
    )
    shard_dir = cfg.head2head_stage_dir / "bonferroni_pairwise_shards"
    shard_dir.mkdir()
    existing_path = shard_dir / "bonferroni_pairwise_shard_0000.parquet"
    existing.to_parquet(existing_path)

    call_counter = {"calls": 0}

    def fake_games_for_power(**kwargs):  # noqa: ANN001
        return 2

    def fake_simulate(seeds, strategies, n_jobs):  # noqa: ANN001,ARG001
        call_counter["calls"] += 1
        return _mock_simulated_games(strategies, seeds)

    monkeypatch.setattr(rb, "games_for_power", fake_games_for_power)
    monkeypatch.setattr(
        rb,
        "_load_top_strategies",
        lambda **_: (
            ["S1", "S2", "S3"],
            {
                "ratings_count": 3,
                "metrics_count": 3,
                "combined_count": 3,
                "ratings_path": "",
                "metrics_path": "",
            },
        ),
    )
    monkeypatch.setattr(
        rb,
        "parse_strategy_identifier",
        lambda name, manifest=None, parse_legacy=None: name,
    )
    monkeypatch.setattr(rb, "simulate_many_games_from_seeds", fake_simulate)

    rb.run_bonferroni_head2head(seed=1, cfg=cfg, shard_size=1)

    pairwise_path = cfg.head2head_path("bonferroni_pairwise.parquet")
    ordered_path = cfg.head2head_path("bonferroni_pairwise_ordered.parquet")
    selfplay_path = cfg.head2head_path("bonferroni_selfplay_symmetry.parquet")
    assert pairwise_path.exists()
    assert ordered_path.exists()
    assert selfplay_path.exists()
    df = pd.read_parquet(pairwise_path)
    assert set(df["pair_id"]) == {0, 1, 2}
    assert {
        "mean_farkles_a",
        "mean_farkles_b",
        "mean_score_a",
        "mean_score_b",
    }.issubset(df.columns)
    populated_pairs = df[(df["games"] > 0) & (df["pair_id"] != 0)]
    assert populated_pairs["mean_farkles_a"].notna().all()
    assert populated_pairs["mean_farkles_b"].notna().all()
    assert populated_pairs["mean_score_a"].notna().all()
    assert populated_pairs["mean_score_b"].notna().all()
    assert populated_pairs["mean_farkles_a"].between(1.0, 2.0).all()
    assert populated_pairs["mean_farkles_b"].between(1.0, 2.0).all()
    assert populated_pairs["mean_score_a"].between(250.0, 300.0).all()
    assert populated_pairs["mean_score_b"].between(250.0, 300.0).all()
    ordered = pd.read_parquet(ordered_path)
    assert set(ordered["ordering"]) == {"a_b", "b_a"}
    assert {
        "mean_farkles_seat1",
        "mean_farkles_seat2",
        "mean_score_seat1",
        "mean_score_seat2",
    }.issubset(ordered.columns)
    populated = ordered[ordered["games"] > 0]
    assert populated["mean_farkles_seat1"].notna().all()
    assert populated["mean_farkles_seat2"].notna().all()
    assert populated["mean_score_seat1"].notna().all()
    assert populated["mean_score_seat2"].notna().all()
    selfplay = pd.read_parquet(selfplay_path)
    assert set(selfplay["strategy"]) == {"S1", "S2", "S3"}
    assert {
        "mean_farkles_seat1",
        "mean_farkles_seat2",
        "mean_score_seat1",
        "mean_score_seat2",
    }.issubset(selfplay.columns)
    assert (selfplay["mean_farkles_seat1"] == 1.0).all()
    assert (selfplay["mean_farkles_seat2"] == 2.0).all()
    assert (selfplay["mean_score_seat1"] == 300.0).all()
    assert (selfplay["mean_score_seat2"] == 250.0).all()

    shards = sorted(shard_dir.glob("*.parquet"))
    assert len(shards) >= 2
    # Two pending pairs -> each runs two seat orderings, plus one self-play run per elite.
    assert call_counter["calls"] == 7


def test_run_bonferroni_head2head_accepts_legacy_farkles_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **_: 2)  # noqa: ANN001
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
    monkeypatch.setattr(
        rb,
        "parse_strategy_identifier",
        lambda name, manifest=None, parse_legacy=None: name,
    )
    monkeypatch.setattr(
        rb,
        "simulate_many_games_from_seeds",
        lambda seeds, strategies, n_jobs: _mock_simulated_games_legacy_farkles(strategies, seeds),
    )

    rb.run_bonferroni_head2head(seed=1, cfg=cfg)

    pairwise_path = cfg.head2head_path("bonferroni_pairwise.parquet")
    selfplay_path = cfg.head2head_path("bonferroni_selfplay_symmetry.parquet")
    assert pairwise_path.exists()
    assert selfplay_path.exists()

    pairwise = pd.read_parquet(pairwise_path)
    assert (pairwise["mean_farkles_a"] == 3.5).all()
    assert (pairwise["mean_farkles_b"] == 3.5).all()

    selfplay = pd.read_parquet(selfplay_path)
    assert (selfplay["mean_farkles_seat1"] == 3.0).all()
    assert (selfplay["mean_farkles_seat2"] == 4.0).all()


def test_run_bonferroni_head2head_progress_cadence_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0, "C": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **kwargs: 1)  # noqa: ANN001
    monkeypatch.setattr(
        rb,
        "parse_strategy_identifier",
        lambda name, manifest=None, parse_legacy=None: name,
    )
    monkeypatch.setattr(
        rb,
        "simulate_many_games_from_seeds",
        lambda seeds, strategies, n_jobs: _mock_simulated_games(strategies, seeds),
    )

    ticks = iter([0.0, 0.0, 1.0, 2.1, 4.3, 8.6])
    monkeypatch.setattr(rb.time, "monotonic", lambda: next(ticks))

    with caplog.at_level("INFO"):
        rb.run_bonferroni_head2head(
            cfg=cfg,
            n_jobs=1,
            progress_schedule=[0.5, 1.5, 2.0],
        )

    progress_logs = [rec for rec in caplog.records if rec.message == "Head-to-head progress"]
    assert len(progress_logs) >= 2


def test_run_bonferroni_head2head_progress_schedule_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **kwargs: 1)  # noqa: ANN001
    monkeypatch.setattr(
        rb,
        "parse_strategy_identifier",
        lambda name, manifest=None, parse_legacy=None: name,
    )
    monkeypatch.setattr(
        rb,
        "simulate_many_games_from_seeds",
        lambda seeds, strategies, n_jobs: _mock_simulated_games(strategies, seeds, winner="A"),
    )

    with pytest.raises(ValueError, match="progress_schedule must have three values"):
        rb.run_bonferroni_head2head(root=cfg.results_root, progress_schedule=[1, 2])


def test_run_bonferroni_limits_pair_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0, "C": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **kwargs: 1)  # noqa: ANN001
    monkeypatch.setattr(
        rb,
        "parse_strategy_identifier",
        lambda name, manifest=None, parse_legacy=None: name,
    )

    pair_jobs: list[int] = []

    def fake_simulate(seeds, strategies, n_jobs):  # noqa: ANN001,ARG001
        pair_jobs.append(n_jobs)
        return _mock_simulated_games(strategies, seeds)

    monkeypatch.setattr(rb, "simulate_many_games_from_seeds", fake_simulate)

    rb.run_bonferroni_head2head(root=cfg.results_root, n_jobs=4)

    assert pair_jobs
    assert all(job == 2 for job in pair_jobs)
    assert len(pair_jobs) == 9


def test_run_bonferroni_head2head_safeguard_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.head2head.bonferroni_total_games_safeguard = 1
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0, "C": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **_: 10)  # noqa: ANN001
    monkeypatch.setattr(
        rb,
        "_load_top_strategies",
        lambda **_: (
            ["A", "B", "C"],
            {
                "ratings_count": 3,
                "metrics_count": 3,
                "combined_count": 3,
                "ratings_path": "",
                "metrics_path": "",
            },
        ),
    )

    rb.run_bonferroni_head2head(cfg=cfg)

    pairwise_path = cfg.head2head_stage_dir / "bonferroni_pairwise.parquet"
    assert pairwise_path.exists()
    df = pd.read_parquet(pairwise_path)
    assert df.empty

    done = read_stage_done(stage_done_path(cfg.head2head_stage_dir, "bonferroni_head2head"))
    assert done["status"] == "skipped"
    assert "safeguard" in str(done["reason"])


def test_load_top_strategies_handles_missing_and_invalid(tmp_path: Path, caplog):
    ratings = tmp_path / "ratings.parquet"
    metrics = tmp_path / "metrics.parquet"

    caplog.set_level("INFO")
    strategies, info = rb._load_top_strategies(ratings_path=ratings, metrics_path=metrics)

    assert strategies == []
    assert info["combined_count"] == 0
    assert any("Fallback selection skipped" in rec.message for rec in caplog.records)

    ratings.write_text("not a parquet")
    invalid_again, invalid_info = rb._load_top_strategies(ratings_path=ratings, metrics_path=metrics)
    assert invalid_again == []
    assert invalid_info["ratings_count"] == 0

    ratings_df = pd.DataFrame({"strategy": ["S1", "S2"], "mu": [1.0, 2.0]})
    metrics_df = pd.DataFrame({"strategy": ["S2", "S3"], "win_rate": [0.2, 0.8]})
    ratings_df.to_parquet(ratings)
    metrics_df.to_parquet(metrics)

    combined, combined_info = rb._load_top_strategies(ratings_path=ratings, metrics_path=metrics)
    assert combined == ["S2", "S1", "S3"]
    assert combined_info["combined_count"] == 3


def test_tiers_path_prefers_stage_layout_and_warns(tmp_path: Path, caplog):
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.set_stage_layout(
        StageLayout(
            placements=[
                StagePlacement(
                    definition=StageDefinition(key="head2head", group="analytics"),
                    index=0,
                    folder_name="11_head2head",
                ),
                StagePlacement(
                    definition=StageDefinition(key="tiering", group="analytics"),
                    index=1,
                    folder_name="11_head2head",
                ),
            ]
        )
    )
    stage_dir = cfg.analysis_dir / "11_head2head"
    stage_dir.mkdir(parents=True)
    legacy_tiers = cfg.analysis_dir / "tiers.json"
    legacy_tiers.write_text("{}")
    preferred = stage_dir / "tiers.json"
    preferred.write_text("{}")

    caplog.set_level("WARNING")
    path = rb._tiers_path(cfg)

    assert path == preferred


def test_count_pair_wins_errors_without_winner(monkeypatch):
    df = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(KeyError):
        rb._count_pair_wins(df, "A", "B")


def test_warn_legacy_stage_dirs_only_warns_for_legacy(tmp_path: Path, caplog) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    expected = cfg.head2head_stage_dir
    expected.mkdir(parents=True, exist_ok=True)
    legacy = cfg.analysis_dir / "99_head2head"
    legacy.mkdir(parents=True, exist_ok=True)

    with caplog.at_level("WARNING"):
        rb._warn_legacy_stage_dirs(cfg, "head2head")

    warnings = [rec for rec in caplog.records if rec.message == "Legacy stage directory detected; prefer layout-aware helpers"]
    assert len(warnings) == 1
    assert warnings[0].legacy_path == str(legacy)


def test_tiers_path_fallback_and_no_existing_candidates(tmp_path: Path, caplog) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.tiering_stage_dir.mkdir(parents=True, exist_ok=True)
    legacy = cfg.analysis_dir / "tiers.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("{}")

    with caplog.at_level("WARNING"):
        path = rb._tiers_path(cfg)

    assert path == legacy
    assert any(rec.message == "Legacy tiers path detected; prefer layout-aware locations" for rec in caplog.records)

    legacy.unlink()
    missing = rb._tiers_path(cfg)
    assert missing == cfg.tiering_stage_dir / "tiers.json"
    assert not missing.exists()


@pytest.mark.parametrize(
    ("design", "message"),
    [
        ({"method": "holm"}, "requires method='bonferroni'"),
        ({"full_pairwise": False}, "requires full_pairwise comparisons"),
        ({"tail": "sideways"}, "tail must be 'one_sided' or 'two_sided'"),
        ({"k_players": 3}, "only supports k_players=2"),
    ],
)
def test_run_bonferroni_head2head_design_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, design: dict[str, object], message: str
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0}))

    monkeypatch.setattr(rb, "_load_top_strategies", lambda **_: (["A", "B"], {"ratings_count": 0, "metrics_count": 0, "combined_count": 2, "ratings_path": "", "metrics_path": ""}))

    with pytest.raises(ValueError, match=message):
        rb.run_bonferroni_head2head(cfg=cfg, design=design)


def test_run_bonferroni_head2head_returns_when_games_per_pair_non_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0, "C": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **_: 0)  # noqa: ANN001
    monkeypatch.setattr(rb, "_load_top_strategies", lambda **_: (["A", "B", "C"], {"ratings_count": 0, "metrics_count": 0, "combined_count": 3, "ratings_path": "", "metrics_path": ""}))

    called = {"simulate": 0}

    def fail_if_called(*args, **kwargs):  # noqa: ANN002,ANN003
        called["simulate"] += 1
        raise AssertionError("simulate should not be called")

    monkeypatch.setattr(rb, "simulate_many_games_from_seeds", fail_if_called)
    rb.run_bonferroni_head2head(cfg=cfg)
    assert called["simulate"] == 0


def test_run_bonferroni_head2head_single_elite_no_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"only": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **_: 5)  # noqa: ANN001
    monkeypatch.setattr(rb, "_load_top_strategies", lambda **_: ([], {"ratings_count": 0, "metrics_count": 0, "combined_count": 0, "ratings_path": "", "metrics_path": ""}))
    monkeypatch.setattr(rb, "parse_strategy_identifier", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("parse should not be called")))

    rb.run_bonferroni_head2head(cfg=cfg)
    pairwise_path = cfg.head2head_path("bonferroni_pairwise.parquet")
    ordered_path = cfg.head2head_path("bonferroni_pairwise_ordered.parquet")
    selfplay_path = cfg.head2head_path("bonferroni_selfplay_symmetry.parquet")
    assert pairwise_path.exists()
    assert ordered_path.exists()
    assert selfplay_path.exists()
    assert pd.read_parquet(pairwise_path).empty
    assert pd.read_parquet(ordered_path).empty
    assert pd.read_parquet(selfplay_path).empty

    done = read_stage_done(stage_done_path(cfg.head2head_stage_dir, "bonferroni_head2head"))
    assert done["status"] == "skipped"
    assert "insufficient candidate strategies" in str(done["reason"])


def test_run_bonferroni_head2head_shard_pair_id_read_exception_warns_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0}))

    shard_dir = cfg.head2head_stage_dir / "bonferroni_pairwise_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "bonferroni_pairwise_shard_0000.parquet"
    pd.DataFrame([{"pair_id": 99}]).to_parquet(shard_path)

    monkeypatch.setattr(rb, "games_for_power", lambda **_: 1)  # noqa: ANN001
    monkeypatch.setattr(rb, "_load_top_strategies", lambda **_: (["A", "B"], {"ratings_count": 0, "metrics_count": 0, "combined_count": 2, "ratings_path": "", "metrics_path": ""}))
    monkeypatch.setattr(rb, "parse_strategy_identifier", lambda name, **_: name)
    monkeypatch.setattr(rb, "simulate_many_games_from_seeds", lambda seeds, strategies, n_jobs: _mock_simulated_games(strategies, seeds))

    real_read = rb.pd.read_parquet

    def flaky_read(path, columns=None, *args, **kwargs):  # noqa: ANN001
        if Path(path) == shard_path and columns == ["pair_id"]:
            raise RuntimeError("broken shard pair-id read")
        return real_read(path, columns=columns, *args, **kwargs)

    monkeypatch.setattr(rb.pd, "read_parquet", flaky_read)
    with caplog.at_level("WARNING"):
        rb.run_bonferroni_head2head(cfg=cfg)

    assert any(rec.message == "Failed to load completed pair ids" for rec in caplog.records)
    out = pd.read_parquet(cfg.head2head_path("bonferroni_pairwise.parquet"))
    assert set(out["pair_id"]) == {0, 99}


def test_run_bonferroni_head2head_missing_pair_id_column_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0}))

    pairwise_path = cfg.head2head_path("bonferroni_pairwise.parquet")
    pairwise_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"players": 2}]).to_parquet(pairwise_path)

    monkeypatch.setattr(rb, "games_for_power", lambda **_: 1)  # noqa: ANN001
    monkeypatch.setattr(rb, "_load_top_strategies", lambda **_: (["A", "B"], {"ratings_count": 0, "metrics_count": 0, "combined_count": 2, "ratings_path": "", "metrics_path": ""}))
    monkeypatch.setattr(rb, "parse_strategy_identifier", lambda name, **_: name)
    monkeypatch.setattr(rb, "simulate_many_games_from_seeds", lambda seeds, strategies, n_jobs: _mock_simulated_games(strategies, seeds))

    real_read = rb.pd.read_parquet

    def read_missing_pair(path, columns=None, *args, **kwargs):  # noqa: ANN001
        if Path(path).resolve() == pairwise_path.resolve() and columns and "pair_id" in columns:
            return pd.DataFrame({"other": [1]})
        return real_read(path, columns=columns, *args, **kwargs)

    monkeypatch.setattr(rb.pd, "read_parquet", read_missing_pair)
    with caplog.at_level("WARNING"):
        rb.run_bonferroni_head2head(cfg=cfg)

    assert any(rec.message == "Existing parquet missing pair_id column" for rec in caplog.records)


def test_run_bonferroni_head2head_warns_when_final_pairwise_load_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0}))

    pairwise_path = cfg.head2head_path("bonferroni_pairwise.parquet")
    pairwise_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"pair_id": 88}]).to_parquet(pairwise_path)

    monkeypatch.setattr(rb, "games_for_power", lambda **_: 1)  # noqa: ANN001
    monkeypatch.setattr(rb, "_load_top_strategies", lambda **_: (["A", "B"], {"ratings_count": 0, "metrics_count": 0, "combined_count": 2, "ratings_path": "", "metrics_path": ""}))
    monkeypatch.setattr(rb, "parse_strategy_identifier", lambda name, **_: name)
    monkeypatch.setattr(rb, "simulate_many_games_from_seeds", lambda seeds, strategies, n_jobs: _mock_simulated_games(strategies, seeds))

    real_read = rb.pd.read_parquet

    def read_with_final_failure(path, columns=None, *args, **kwargs):  # noqa: ANN001
        if Path(path) == pairwise_path and columns is None:
            raise RuntimeError("cannot read final parquet")
        return real_read(path, columns=columns, *args, **kwargs)

    monkeypatch.setattr(rb.pd, "read_parquet", read_with_final_failure)
    with caplog.at_level("WARNING"):
        rb.run_bonferroni_head2head(cfg=cfg)

    assert any(rec.message == "Failed to load existing pairwise parquet" for rec in caplog.records)
    out = real_read(pairwise_path)
    assert set(out["pair_id"]) == {0}


def test_run_bonferroni_head2head_errors_on_ties_or_missing_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **_: 1)  # noqa: ANN001
    monkeypatch.setattr(rb, "_load_top_strategies", lambda **_: (["A", "B"], {"ratings_count": 0, "metrics_count": 0, "combined_count": 2, "ratings_path": "", "metrics_path": ""}))
    monkeypatch.setattr(rb, "parse_strategy_identifier", lambda name, **_: name)
    monkeypatch.setattr(
        rb,
        "simulate_many_games_from_seeds",
        lambda seeds, strategies, n_jobs: _mock_simulated_games(strategies, seeds, winner="unknown"),
    )

    with pytest.raises(RuntimeError, match="Tie or missing outcome detected"):
        rb.run_bonferroni_head2head(cfg=cfg)


def test_run_bonferroni_head2head_root_seed_override_updates_cfg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("data") / "custom_seed_17"
    cfg = AppConfig()
    analysis_dir = Path("data/custom_seed_17/analysis")
    tiers_path = analysis_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"A": 0, "B": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **_: 0)  # noqa: ANN001
    monkeypatch.setattr(rb, "_load_top_strategies", lambda **_: (["A", "B"], {"ratings_count": 0, "metrics_count": 0, "combined_count": 2, "ratings_path": "", "metrics_path": ""}))

    rb.run_bonferroni_head2head(cfg=cfg, root=root)

    assert cfg.io.results_dir_prefix == Path("custom")
    assert cfg.sim.seed == 17
