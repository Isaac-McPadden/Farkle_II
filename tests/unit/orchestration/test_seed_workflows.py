from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from farkle.analysis.stage_registry import StageLayout, resolve_stage_layout
from farkle.config import AppConfig, IOConfig
from farkle.orchestration import pipeline, run_contexts, seed_utils, two_seed, two_seed_pipeline


@pytest.fixture
def base_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results_seed_5"))
    cfg.sim.n_players_list = [2, 5]
    cfg.sim.seed_pair = (11, 22)
    return cfg


def test_seed_utils_paths_and_split(base_cfg: AppConfig) -> None:
    root = seed_utils.resolve_results_dir(Path("data/results"), 12)
    assert root == Path("data/results_seed_12")

    seeded_base, seed = seed_utils.split_seeded_results_dir(Path("abc_seed_123"))
    assert seeded_base == Path("abc")
    assert seed == 123

    plain_base, plain_seed = seed_utils.split_seeded_results_dir(Path("abc"))
    assert plain_base == Path("abc")
    assert plain_seed is None

    pair_root = seed_utils.seed_pair_root(base_cfg, (7, 8))
    assert pair_root.name.endswith("results_seed_pair_7_8")

    per_seed_root = seed_utils.seed_pair_seed_root(base_cfg, (7, 8), 7)
    assert per_seed_root == pair_root / "results_seed_7"


def test_seed_pair_meta_root_and_completion_marker_checks(
    base_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_cfg.io.meta_analysis_dir = Path("meta")
    meta_root = seed_utils.seed_pair_meta_root(base_cfg, (3, 9))
    assert meta_root is not None
    assert meta_root.name == "meta_3_9"

    base_cfg.io.meta_analysis_dir = None
    assert seed_utils.seed_pair_meta_root(base_cfg, (3, 9)) is None

    calls: list[int] = []

    def fake_is_complete(cfg: AppConfig, n_players: int) -> bool:
        calls.append(n_players)
        return n_players != 5

    monkeypatch.setattr(seed_utils.runner, "simulation_is_complete", fake_is_complete)
    assert not seed_utils.seed_has_completion_markers(base_cfg)
    assert calls == [2, 5]


def test_prepare_seed_config_and_write_active_config(tmp_path: Path, base_cfg: AppConfig) -> None:
    prepared = seed_utils.prepare_seed_config(
        base_cfg,
        seed=44,
        base_results_dir=Path("data/custom_prefix"),
        meta_analysis_dir=tmp_path / "meta",
    )

    assert prepared.sim.seed == 44
    assert prepared.io.results_dir_prefix == Path("custom_prefix")
    assert prepared.io.meta_analysis_dir == tmp_path / "meta"

    seed_utils.write_active_config(prepared, dest_dir=tmp_path)
    out = yaml.safe_load((tmp_path / "active_config.yaml").read_text(encoding="utf-8"))
    assert out["io"]["results_dir_prefix"].endswith("custom_prefix")
    assert out["io"]["meta_analysis_dir"].endswith("meta")


def test_two_seed_resolve_seed_pair_and_parser_validation() -> None:
    parser = two_seed.build_parser()
    args = parser.parse_args(["--seed-pair", "1", "2"])
    assert two_seed._resolve_seed_pair(args, parser) == (1, 2)

    args = parser.parse_args(["--seed-a", "4", "--seed-b", "5"])
    assert two_seed._resolve_seed_pair(args, parser) == (4, 5)

    with pytest.raises(SystemExit):
        args = parser.parse_args(["--seed-a", "4"])
        two_seed._resolve_seed_pair(args, parser)

    with pytest.raises(SystemExit):
        args = parser.parse_args(["--seed-a", "4", "--seed-b", "5", "--seed-pair", "1", "2"])
        two_seed._resolve_seed_pair(args, parser)


def test_two_seed_resolve_seed_pair_empty_args_returns_none() -> None:
    parser = two_seed.build_parser()
    args = parser.parse_args([])
    assert two_seed._resolve_seed_pair(args, parser) is None


def test_two_seed_build_parser_defaults() -> None:
    parser = two_seed.build_parser()
    args = parser.parse_args([])
    assert args.config == Path("configs/fast_config.yaml")
    assert args.force is False
    assert args.overrides == []


def test_two_seed_main_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.sim.seed_pair = (10, 20)

    called: dict[str, Any] = {}

    monkeypatch.setattr(two_seed, "setup_info_logging", lambda: called.setdefault("logging", True))
    monkeypatch.setattr(two_seed, "load_app_config", lambda path, seed_list_len=0: cfg)
    monkeypatch.setattr(two_seed, "apply_dot_overrides", lambda c, overrides: c)

    def fake_run_seeds(run_cfg: AppConfig, *, seed_pair: tuple[int, int], force: bool) -> None:
        called["seed_pair"] = seed_pair
        called["force"] = force
        called["cfg"] = run_cfg

    monkeypatch.setattr(two_seed, "run_seeds", fake_run_seeds)

    rc = two_seed.main(["--seed-a", "7", "--seed-b", "8", "--force"])
    assert rc == 0
    assert called["logging"] is True
    assert called["seed_pair"] == (7, 8)
    assert called["force"] is True


@pytest.mark.parametrize(
    ("force", "markers", "expected_runs", "expected_marker_checks"),
    [
        (False, {3: True, 4: True}, [], [3, 4]),
        (False, {3: False, 4: False}, [3, 4], [3, 4]),
        (True, {3: True, 4: True}, [3, 4], []),
    ],
)
def test_two_seed_run_seeds_skip_normal_and_forced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    force: bool,
    markers: dict[int, bool],
    expected_runs: list[int],
    expected_marker_checks: list[int],
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    seed_pair = (3, 4)

    call_log: dict[str, list[Any]] = {
        "seed_pair_seed_root": [],
        "prepare_seed_config": [],
        "marker_checks": [],
        "write_active_config": [],
        "run_tournament": [],
    }

    def fake_seed_pair_seed_root(_cfg: AppConfig, pair: tuple[int, int], seed: int) -> Path:
        call_log["seed_pair_seed_root"].append((pair, seed))
        return tmp_path / f"seed_{seed}"

    def fake_prepare_seed_config(
        _cfg: AppConfig,
        *,
        seed: int,
        base_results_dir: Path,
        meta_analysis_dir: Path | None = None,
    ) -> AppConfig:
        del meta_analysis_dir
        call_log["prepare_seed_config"].append((seed, base_results_dir))
        seed_cfg = AppConfig(io=IOConfig(results_dir_prefix=base_results_dir))
        seed_cfg.sim.seed = seed
        return seed_cfg

    def fake_seed_has_completion_markers(seed_cfg: AppConfig) -> bool:
        seed = seed_cfg.sim.seed
        assert seed is not None
        call_log["marker_checks"].append(seed)
        return markers[seed]

    def fake_write_active_config(seed_cfg: AppConfig) -> None:
        call_log["write_active_config"].append(seed_cfg.sim.seed)

    def fake_run_tournament(seed_cfg: AppConfig, *, force: bool = False) -> None:
        call_log["run_tournament"].append((seed_cfg.sim.seed, force))

    monkeypatch.setattr(two_seed, "seed_pair_seed_root", fake_seed_pair_seed_root)
    monkeypatch.setattr(two_seed, "prepare_seed_config", fake_prepare_seed_config)
    monkeypatch.setattr(two_seed, "seed_has_completion_markers", fake_seed_has_completion_markers)
    monkeypatch.setattr(two_seed, "write_active_config", fake_write_active_config)
    monkeypatch.setattr(two_seed.runner, "run_tournament", fake_run_tournament)

    two_seed.run_seeds(cfg, seed_pair=seed_pair, force=force)

    assert call_log["seed_pair_seed_root"] == [((3, 4), 3), ((3, 4), 4)]
    assert [seed for seed, _ in call_log["prepare_seed_config"]] == [3, 4]
    assert call_log["marker_checks"] == expected_marker_checks
    assert call_log["write_active_config"] == expected_runs
    assert call_log["run_tournament"] == [(seed, force) for seed in expected_runs]


def test_two_seed_main_uses_populated_seed_pair_when_no_cli_seed_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))

    called: dict[str, Any] = {}

    def fake_populate_seed_list(count: int) -> list[int]:
        called["populate_count"] = count
        return [101, 202]

    monkeypatch.setattr(cfg.sim, "populate_seed_list", fake_populate_seed_list)
    monkeypatch.setattr(two_seed, "setup_info_logging", lambda: called.setdefault("logging", True))
    monkeypatch.setattr(two_seed, "load_app_config", lambda path, seed_list_len=0: cfg)
    monkeypatch.setattr(two_seed, "apply_dot_overrides", lambda c, overrides: c)

    def fake_run_seeds(run_cfg: AppConfig, *, seed_pair: tuple[int, int], force: bool) -> None:
        called["cfg"] = run_cfg
        called["seed_pair"] = seed_pair
        called["force"] = force

    monkeypatch.setattr(two_seed, "run_seeds", fake_run_seeds)

    rc = two_seed.main([])
    assert rc == 0
    assert called["logging"] is True
    assert called["populate_count"] == 2
    assert called["cfg"] is cfg
    assert called["seed_pair"] == (101, 202)
    assert called["force"] is False


def test_two_seed_pipeline_helpers(base_cfg: AppConfig, tmp_path: Path) -> None:
    parser = two_seed_pipeline.build_parser()
    args = parser.parse_args(["--seed-pair", "13", "14"])
    assert two_seed_pipeline._resolve_seed_pair(args, parser) == (13, 14)

    pair_root = tmp_path / "pair"
    base_cfg.io.meta_analysis_dir = None
    assert two_seed_pipeline._shared_meta_dir(base_cfg, pair_root, (1, 2)) == pair_root / "interseed_analysis" / "seed_summaries_meta"

    base_cfg.io.meta_analysis_dir = Path("meta")
    assert two_seed_pipeline._shared_meta_dir(base_cfg, pair_root, (1, 2)).name == "meta_1_2"


def test_run_per_seed_analysis_invokes_stage_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    captured: dict[str, Any] = {}
    delegated: list[tuple[AppConfig, Path]] = []

    def fake_run(plan: list[Any], context: Any, raise_on_failure: bool) -> None:
        captured["plan_names"] = [item.name for item in plan]
        captured["run_label"] = context.run_label
        captured["raise_on_failure"] = raise_on_failure
        for item in plan:
            if item.name == "single_seed_analysis":
                item.action(context.config)

    monkeypatch.setattr(two_seed_pipeline.StageRunner, "run", staticmethod(fake_run))
    monkeypatch.setattr(
        two_seed_pipeline.analysis,
        "run_single_seed_analysis",
        lambda app_cfg, *, manifest_path: delegated.append((app_cfg, manifest_path)),
    )

    manifest_path = tmp_path / "manifest.jsonl"
    two_seed_pipeline._run_per_seed_analysis(cfg, manifest_path=manifest_path, seed=9)
    assert captured["plan_names"] == [
        "ingest",
        "curate",
        "combine",
        "metrics",
        "game_stats",
        "single_seed_analysis",
    ]
    assert captured["run_label"] == "per_seed_pipeline_9"
    assert captured["raise_on_failure"] is True
    assert delegated == [(cfg, manifest_path)]


def test_run_pipeline_skip_vs_force(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.sim.seed_pair = (1, 2)

    sim_calls: list[int] = []
    per_seed_calls: list[int] = []
    manifest_events: list[str] = []

    monkeypatch.setattr(two_seed_pipeline, "seed_has_completion_markers", lambda _cfg: True)
    monkeypatch.setattr(two_seed_pipeline.runner, "run_tournament", lambda seed_cfg, force=False: sim_calls.append(seed_cfg.sim.seed))
    monkeypatch.setattr(two_seed_pipeline, "_run_per_seed_analysis", lambda _cfg, manifest_path, seed: per_seed_calls.append(seed))
    monkeypatch.setattr(two_seed_pipeline, "append_manifest_line", lambda _path, rec: manifest_events.append(rec["event"]))

    interseed_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "interseed"))
    interseed_context = SimpleNamespace(config=interseed_cfg)
    monkeypatch.setattr(
        two_seed_pipeline.InterseedRunContext,
        "from_seed_context",
        classmethod(lambda cls, seed_context, seed_pair, analysis_root: interseed_context),
    )
    monkeypatch.setattr(two_seed_pipeline.analysis, "run_interseed_analysis", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(two_seed_pipeline.analysis, "run_h2h_tier_trends", lambda *_args, **_kwargs: None)

    two_seed_pipeline.run_pipeline(cfg, seed_pair=(1, 2), force=False)
    assert sim_calls == []
    assert per_seed_calls == [1, 2]
    assert manifest_events == [
        "run_start",
        "seed_start",
        "seed_simulation_skipped",
        "seed_analysis_complete",
        "seed_start",
        "seed_simulation_skipped",
        "seed_analysis_complete",
        "interseed_start",
        "interseed_complete",
        "h2h_tier_trends_start",
        "h2h_tier_trends_complete",
        "run_end",
    ]

    sim_calls.clear()
    per_seed_calls.clear()
    manifest_events.clear()
    two_seed_pipeline.run_pipeline(cfg, seed_pair=(1, 2), force=True)
    assert sim_calls == [1, 2]
    assert per_seed_calls == [1, 2]
    assert manifest_events == [
        "run_start",
        "seed_start",
        "seed_simulation_complete",
        "seed_analysis_complete",
        "seed_start",
        "seed_simulation_complete",
        "seed_analysis_complete",
        "interseed_start",
        "interseed_complete",
        "h2h_tier_trends_start",
        "h2h_tier_trends_complete",
        "run_end",
    ]


def test_two_seed_pipeline_main_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.sim.seed_pair = (50, 60)

    called: dict[str, Any] = {}

    monkeypatch.setattr(two_seed_pipeline, "setup_info_logging", lambda: called.setdefault("logging", True))
    monkeypatch.setattr(two_seed_pipeline, "load_app_config", lambda path, seed_list_len=0: cfg)
    monkeypatch.setattr(two_seed_pipeline, "apply_dot_overrides", lambda c, overrides: c)
    monkeypatch.setattr(
        two_seed_pipeline,
        "run_pipeline",
        lambda run_cfg, *, seed_pair, force: called.update(seed_pair=seed_pair, force=force),
    )

    rc = two_seed_pipeline.main(["--seed-pair", "3", "4"])
    assert rc == 0
    assert called["logging"] is True
    assert called["seed_pair"] == (3, 4)
    assert called["force"] is False


def test_pipeline_main_and_detect_player_counts_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(__import__("sys").modules, "farkle.analysis.pipeline", SimpleNamespace(main=lambda argv=None: 123))
    assert pipeline.main(["--anything"]) == 123

    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "res"))
    metrics = cfg.analysis_dir / "04_metrics" / "metrics.parquet"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text("stub")

    monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr("pandas.read_parquet", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")))
    assert pipeline._detect_player_counts(cfg.analysis_dir) == [5]


def test_run_contexts_edge_branches(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "res"))
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    seed_context = run_contexts.SeedRunContext.from_config(cfg)

    run_cfg = run_contexts.RunContextConfig.from_base(
        cfg,
        analysis_root=tmp_path / "override_analysis",
        interseed_input_dir=tmp_path / "override_input",
        interseed_input_layout={"combine": "custom_combine"},
    )
    assert run_cfg.analysis_dir == tmp_path / "override_analysis"
    assert run_cfg.interseed_input_dir == tmp_path / "override_input"
    assert run_cfg._interseed_input_folder("combine") == "custom_combine"
    assert run_cfg._interseed_input_folder("missing") is None

    run_cfg._interseed_input_layout_override = 123  # type: ignore[assignment]
    assert run_cfg._interseed_input_folder("combine") is None

    cfg_without_combine = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "res_no_combine"))
    cfg_without_combine.set_stage_layout(StageLayout(placements=[]))
    empty_seed_context = run_contexts.SeedRunContext.from_config(cfg_without_combine)
    with pytest.raises(KeyError):
        run_contexts.InterseedRunContext.from_seed_context(
            empty_seed_context,
            seed_pair=(1, 2),
            analysis_root=tmp_path / "pair" / "interseed",
        )

    interseed = run_contexts.InterseedRunContext.from_seed_context(
        seed_context,
        seed_pair=(9, 10),
        analysis_root=tmp_path / "pair" / "interseed",
    )
    assert interseed.config.analysis.tiering_seeds == [9, 10]
