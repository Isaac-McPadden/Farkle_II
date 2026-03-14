from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from farkle.analysis.stage_registry import StageLayout, resolve_stage_layout
from farkle.config import AppConfig, IOConfig, assign_config_sha
from farkle.orchestration import pipeline, run_contexts, seed_utils, two_seed_pipeline


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

    base_cfg.io.meta_analysis_dir = Path("data/meta_analysis")
    nested_meta_root = seed_utils.seed_pair_meta_root(base_cfg, (3, 9))
    assert nested_meta_root is not None
    assert nested_meta_root.name == "meta_analysis_3_9"
    assert nested_meta_root.parent.name.startswith("results_seed_pair_")

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

    assign_config_sha(prepared)
    seed_utils.write_active_config(prepared, dest_dir=tmp_path)
    out = yaml.safe_load((tmp_path / "active_config.yaml").read_text(encoding="utf-8"))
    assert out["io"]["results_dir_prefix"].endswith("custom_prefix")
    assert out["io"]["meta_analysis_dir"].endswith("meta")
    done_meta = json.loads((tmp_path / "active_config.done.json").read_text(encoding="utf-8"))
    assert done_meta["config_sha"] == prepared.config_sha


def test_two_seed_pipeline_helpers(base_cfg: AppConfig, tmp_path: Path) -> None:
    parser = two_seed_pipeline.build_parser()
    args = parser.parse_args(["--seed-pair", "13", "14", "--parallel-seeds"])
    assert seed_utils.resolve_seed_pair_args(args, parser) == (13, 14)
    assert args.parallel_seeds is True

    pair_root = tmp_path / "pair"
    base_cfg.io.meta_analysis_dir = None
    assert two_seed_pipeline._shared_meta_dir(base_cfg, pair_root, (1, 2)) == pair_root / "interseed_analysis" / "seed_summaries_meta"

    base_cfg.io.meta_analysis_dir = Path("meta")
    assert two_seed_pipeline._shared_meta_dir(base_cfg, pair_root, (1, 2)).name == "meta_1_2"


def test_run_per_seed_analysis_invokes_stage_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    captured: dict[str, Any] = {}

    def fake_run(plan: list[Any], context: Any, raise_on_failure: bool) -> None:
        captured["plan_names"] = [item.name for item in plan]
        captured["run_label"] = context.run_label
        captured["manifest_path"] = context.manifest_path
        captured["raise_on_failure"] = raise_on_failure

    monkeypatch.setattr(two_seed_pipeline.StageRunner, "run", staticmethod(fake_run))
    policy_bundle = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=1)
    two_seed_pipeline._run_per_seed_analysis(
        cfg,
        seed=9,
        force=False,
        policy_bundle=policy_bundle,
    )
    assert captured["plan_names"] == [
        "ingest",
        "curate",
        "combine",
        "metrics",
        "coverage_by_k",
        "game_stats",
        "seed_summaries",
        "trueskill",
        "tiering",
        "head2head",
        "seed_symmetry",
        "post_h2h",
        "hgb",
    ]
    assert captured["plan_names"].index("coverage_by_k") < captured["plan_names"].index("game_stats")
    assert captured["run_label"] == "per_seed_pipeline_9"
    assert captured["manifest_path"] == cfg.analysis_dir / cfg.manifest_name
    assert captured["raise_on_failure"] is True


def test_run_per_seed_analysis_writes_per_seed_manifest_and_health_inputs_independent_of_pair_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    pair_manifest_path = tmp_path / "pair" / "two_seed_pipeline_manifest.jsonl"
    policy_bundle = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=1)
    assign_config_sha(cfg)

    def fake_run(_plan: list[Any], context: Any, raise_on_failure: bool) -> None:
        del raise_on_failure
        context.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        context.manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "run_id": "test_run",
                    "event": "run_start",
                    "config_sha": cfg.config_sha,
                    "run": context.run_label,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(two_seed_pipeline.StageRunner, "run", staticmethod(fake_run))

    seed = 123
    two_seed_pipeline._run_per_seed_analysis(
        cfg,
        seed=seed,
        force=False,
        policy_bundle=policy_bundle,
    )

    per_seed_manifest = cfg.analysis_dir / cfg.manifest_name
    assert per_seed_manifest.exists()
    events = [json.loads(line) for line in per_seed_manifest.read_text(encoding="utf-8").splitlines()]
    assert all(event["run"] == f"per_seed_pipeline_{seed}" for event in events)

    stage_statuses = {
        key: {"status": value.status, "diagnostics": list(value.diagnostics)}
        for key, value in two_seed_pipeline._resolve_seed_family_statuses(
            seed,
            seed_cfg=cfg,
            analysis_error=None,
        ).items()
    }
    assert pair_manifest_path.exists() is False
    assert stage_statuses[f"seed_{seed}.analysis"]["status"] == "success"
    assert stage_statuses[f"seed_{seed}.seed_symmetry"]["status"] == "missing"
    assert stage_statuses[f"seed_{seed}.post_h2h"]["status"] == "missing"


def test_run_pipeline_skip_vs_force(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.sim.seed_pair = (1, 2)

    sim_calls: list[int] = []
    per_seed_calls: list[int] = []
    manifest_events: list[str] = []

    monkeypatch.setattr(two_seed_pipeline, "seed_has_completion_markers", lambda _cfg: True)
    monkeypatch.setattr(two_seed_pipeline.runner, "run_tournament", lambda seed_cfg, force=False: sim_calls.append(seed_cfg.sim.seed))
    def _fake_per_seed(
        seed_cfg: AppConfig,
        *,
        seed: int,
        force: bool = False,
        policy_bundle: object | None = None,
    ) -> None:
        del force, policy_bundle
        per_seed_calls.append(seed)
        seed_cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.analysis_dir / "analysis_manifest.jsonl").write_text(
            json.dumps({"event": "run_end"}) + "\n",
            encoding="utf-8",
        )
        seed_cfg.seed_symmetry_stage_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.seed_symmetry_stage_dir / "seed_symmetry_summary.parquet").write_text("ok")
        seed_cfg.post_h2h_stage_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.post_h2h_stage_dir / "h2h_s_tiers.json").write_text("{}")

    monkeypatch.setattr(two_seed_pipeline, "_run_per_seed_analysis", _fake_per_seed)
    monkeypatch.setattr(
        two_seed_pipeline,
        "append_manifest_event",
        lambda _path, rec, **_kwargs: manifest_events.append(rec["event"]),
    )

    interseed_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "interseed"))
    interseed_cfg.config_sha = cfg.config_sha
    interseed_cfg.stage_dir("h2h_tier_trends").mkdir(parents=True, exist_ok=True)
    interseed_context = SimpleNamespace(config=interseed_cfg)
    monkeypatch.setattr(
        two_seed_pipeline.InterseedRunContext,
        "from_seed_context",
        classmethod(lambda cls, seed_context, seed_pair, analysis_root: interseed_context),
    )
    monkeypatch.setattr(two_seed_pipeline.analysis, "run_interseed_analysis", lambda *_args, **_kwargs: None)
    h2h_calls: list[dict[str, object]] = []

    def _record_h2h(_cfg, **kwargs):
        h2h_calls.append(kwargs)
        (_cfg.stage_dir("h2h_tier_trends") / "s_tier_trends.parquet").write_text("ok")

    monkeypatch.setattr(two_seed_pipeline.analysis, "run_h2h_tier_trends", _record_h2h)

    two_seed_pipeline.run_pipeline(cfg, seed_pair=(1, 2), force=False)
    assert sim_calls == []
    assert per_seed_calls == [1, 2]
    assert manifest_events[0] == "run_start"
    assert manifest_events[-1] == "run_end"
    for event in [
        "seed_start",
        "seed_simulation_skipped",
        "seed_analysis_complete",
        "interseed_start",
        "interseed_complete",
        "h2h_tier_trends_start",
        "h2h_tier_trends_complete",
    ]:
        assert event in manifest_events
    assert len(h2h_calls) == 1
    assert h2h_calls[0]["force"] is False
    pair_root = seed_utils.seed_pair_root(cfg, (1, 2))
    assert h2h_calls[0]["seed_s_tier_paths"] == [
        pair_root / "results_seed_1" / "analysis" / "11_post_h2h" / "h2h_s_tiers.json",
        pair_root / "results_seed_2" / "analysis" / "11_post_h2h" / "h2h_s_tiers.json",
    ]

    sim_calls.clear()
    per_seed_calls.clear()
    manifest_events.clear()
    h2h_calls.clear()
    two_seed_pipeline.run_pipeline(cfg, seed_pair=(1, 2), force=True)
    assert sim_calls == [1, 2]
    assert per_seed_calls == [1, 2]
    assert manifest_events[0] == "run_start"
    assert manifest_events[-1] == "run_end"
    for event in [
        "seed_start",
        "seed_simulation_complete",
        "seed_analysis_complete",
        "interseed_start",
        "interseed_complete",
        "h2h_tier_trends_start",
        "h2h_tier_trends_complete",
    ]:
        assert event in manifest_events
    assert len(h2h_calls) == 1
    assert h2h_calls[0]["force"] is True




def test_two_seed_pipeline_parallel_seed_smoke_equivalence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interseed_roots: dict[str, AppConfig] = {}

    def _setup_common_stubs() -> None:
        monkeypatch.setattr(two_seed_pipeline, "seed_has_completion_markers", lambda _cfg: True)
        monkeypatch.setattr(two_seed_pipeline.runner, "run_tournament", lambda *_args, **_kwargs: None)

        def _fake_per_seed(
            seed_cfg: AppConfig,
            *,
            seed: int,
            force: bool = False,
            policy_bundle: object | None = None,
        ) -> None:
            del force, policy_bundle
            seed_cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
            seed_cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
            (seed_cfg.analysis_dir / "analysis_manifest.jsonl").write_text(
                json.dumps({"event": "run_end"}) + "\n",
                encoding="utf-8",
            )
            seed_cfg.seed_symmetry_stage_dir.mkdir(parents=True, exist_ok=True)
            (seed_cfg.seed_symmetry_stage_dir / "seed_symmetry_summary.parquet").write_text("ok")
            seed_cfg.post_h2h_stage_dir.mkdir(parents=True, exist_ok=True)
            (seed_cfg.post_h2h_stage_dir / "h2h_s_tiers.json").write_text(json.dumps({"seed": seed}))

        monkeypatch.setattr(two_seed_pipeline, "_run_per_seed_analysis", _fake_per_seed)

        def _from_seed_context(cls, seed_context, seed_pair, analysis_root):
            del cls, seed_pair
            key = str(analysis_root)
            if key not in interseed_roots:
                interseed_cfg = AppConfig(io=IOConfig(results_dir_prefix=analysis_root / "results_seed_1"))
                interseed_cfg.config_sha = seed_context.config.config_sha
                interseed_roots[key] = interseed_cfg
            return SimpleNamespace(config=interseed_roots[key])

        monkeypatch.setattr(
            two_seed_pipeline.InterseedRunContext,
            "from_seed_context",
            classmethod(_from_seed_context),
        )
        def _record_interseed(cfg: AppConfig, **_kwargs) -> None:
            cfg.interseed_stage_dir.mkdir(parents=True, exist_ok=True)
            (cfg.interseed_stage_dir / "interseed_summary.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(two_seed_pipeline.analysis, "run_interseed_analysis", _record_interseed)

        def _record_h2h(_cfg, **_kwargs):
            (_cfg.stage_dir("h2h_tier_trends") / "s_tier_trends.parquet").write_text("ok")

        monkeypatch.setattr(two_seed_pipeline.analysis, "run_h2h_tier_trends", _record_h2h)

    _setup_common_stubs()

    def _run_case(name: str, *, parallel: bool) -> dict[str, Any]:
        cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / name / "results"))
        cfg.sim.seed_pair = (7, 8)
        cfg.orchestration.parallel_seeds = parallel
        two_seed_pipeline.run_pipeline(cfg, seed_pair=(7, 8), force=False)
        pair_root = seed_utils.seed_pair_root(cfg, (7, 8))
        health = json.loads((pair_root / "pipeline_health.json").read_text(encoding="utf-8"))
        return health

    sequential = _run_case("sequential", parallel=False)
    parallel = _run_case("parallel", parallel=True)

    assert sequential["status"] == "complete_success"
    assert parallel["status"] == "complete_success"
    assert {
        stage: payload["status"] for stage, payload in sequential["stage_statuses"].items()
    } == {
        stage: payload["status"] for stage, payload in parallel["stage_statuses"].items()
    }

    manifest_records = [
        json.loads(line)
        for line in (seed_utils.seed_pair_root(cfg=AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "sequential" / "results")), seed_pair=(7, 8)) / "two_seed_pipeline_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stage_end_records = [record for record in manifest_records if record.get("event") == "stage_end"]
    assert stage_end_records
    assert all(record.get("status") != "missing" for record in stage_end_records)

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
        lambda run_cfg, *, seed_pair, force: called.update(
            seed_pair=seed_pair,
            force=force,
            parallel=run_cfg.orchestration.parallel_seeds,
        ),
    )

    rc = two_seed_pipeline.main(["--seed-pair", "3", "4", "--parallel-seeds"])
    assert rc == 0
    assert called["logging"] is True
    assert called["seed_pair"] == (3, 4)
    assert called["force"] is False


def test_two_seed_pipeline_head2head_failure_chain_integration_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.sim.seed_pair = (101, 202)
    pair_root = seed_utils.seed_pair_root(cfg, (101, 202))

    monkeypatch.setattr(two_seed_pipeline, "seed_has_completion_markers", lambda _cfg: True)
    monkeypatch.setattr(two_seed_pipeline.runner, "run_tournament", lambda *_args, **_kwargs: None)

    def _failing_per_seed(
        seed_cfg: AppConfig,
        *,
        seed: int,
        force: bool = False,
        policy_bundle: object | None = None,
    ) -> None:
        del force, policy_bundle
        seed_cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.analysis_dir / "analysis_manifest.jsonl").write_text(
            json.dumps({"event": "run_end"}) + "\n",
            encoding="utf-8",
        )
        seed_cfg.seed_symmetry_stage_dir.mkdir(parents=True, exist_ok=True)
        if seed == 101:
            (seed_cfg.seed_symmetry_stage_dir / "seed_symmetry_summary.parquet").write_text("ok")
            seed_cfg.post_h2h_stage_dir.mkdir(parents=True, exist_ok=True)
            (seed_cfg.post_h2h_stage_dir / "h2h_s_tiers.json").write_text("{}")
            return
        raise RuntimeError("simulated head2head failure")

    monkeypatch.setattr(two_seed_pipeline, "_run_per_seed_analysis", _failing_per_seed)
    monkeypatch.setattr(
        two_seed_pipeline.InterseedRunContext,
        "from_seed_context",
        classmethod(
            lambda cls, seed_context, seed_pair, analysis_root: SimpleNamespace(
                config=AppConfig(io=IOConfig(results_dir_prefix=analysis_root / "results_seed_101"))
            )
        ),
    )
    monkeypatch.setattr(two_seed_pipeline.analysis, "run_interseed_analysis", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(two_seed_pipeline.analysis, "run_h2h_tier_trends", lambda *_args, **_kwargs: None)

    two_seed_pipeline.run_pipeline(cfg, seed_pair=(101, 202), force=False)

    health = json.loads((pair_root / "pipeline_health.json").read_text(encoding="utf-8"))
    assert health["status"] == "failed_blocked"
    assert health["stage_statuses"]["seed_101.seed_symmetry"]["status"] == "success"
    assert health["stage_statuses"]["seed_101.post_h2h"]["status"] == "success"
    assert health["stage_statuses"]["seed_202.seed_symmetry"]["status"] == "missing"
    assert health["stage_statuses"]["seed_202.post_h2h"]["status"] == "missing"
    assert health["stage_statuses"]["h2h_tier_trends"]["status"] == "missing"
    assert health["first_blocking_failure"]["stage"] == "seed_202.analysis"


def test_two_seed_pipeline_blocks_stale_post_h2h_outputs_after_analysis_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.sim.seed_pair = (301, 302)
    pair_root = seed_utils.seed_pair_root(cfg, (301, 302))

    monkeypatch.setattr(two_seed_pipeline, "seed_has_completion_markers", lambda _cfg: True)
    monkeypatch.setattr(two_seed_pipeline.runner, "run_tournament", lambda *_args, **_kwargs: None)

    def _stale_post_h2h_then_fail(
        seed_cfg: AppConfig,
        *,
        seed: int,
        force: bool = False,
        policy_bundle: object | None = None,
    ) -> None:
        del force, policy_bundle
        seed_cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.analysis_dir / "analysis_manifest.jsonl").write_text(
            json.dumps({"event": "run_end"}) + "\n",
            encoding="utf-8",
        )
        seed_cfg.seed_symmetry_stage_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.seed_symmetry_stage_dir / "seed_symmetry_summary.parquet").write_text(
            "ok",
            encoding="utf-8",
        )
        seed_cfg.post_h2h_stage_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.post_h2h_stage_dir / "h2h_s_tiers.json").write_text("{}", encoding="utf-8")
        raise RuntimeError(f"seed {seed} metrics failed")

    monkeypatch.setattr(two_seed_pipeline, "_run_per_seed_analysis", _stale_post_h2h_then_fail)
    monkeypatch.setattr(
        two_seed_pipeline.InterseedRunContext,
        "from_seed_context",
        classmethod(
            lambda cls, seed_context, seed_pair, analysis_root: SimpleNamespace(
                config=AppConfig(io=IOConfig(results_dir_prefix=analysis_root / "results_seed_301"))
            )
        ),
    )

    interseed_calls: list[str] = []
    monkeypatch.setattr(
        two_seed_pipeline.analysis,
        "run_interseed_analysis",
        lambda *_args, **_kwargs: interseed_calls.append("interseed"),
    )
    monkeypatch.setattr(
        two_seed_pipeline.analysis,
        "run_h2h_tier_trends",
        lambda *_args, **_kwargs: interseed_calls.append("h2h"),
    )

    two_seed_pipeline.run_pipeline(cfg, seed_pair=(301, 302), force=False)

    health = json.loads((pair_root / "pipeline_health.json").read_text(encoding="utf-8"))
    assert interseed_calls == []
    assert health["status"] == "failed_blocked"
    for seed in (301, 302):
        post_h2h = health["stage_statuses"][f"seed_{seed}.post_h2h"]
        seed_symmetry = health["stage_statuses"][f"seed_{seed}.seed_symmetry"]
        assert post_h2h["status"] == "failed"
        assert seed_symmetry["status"] == "failed"
        assert "upstream incomplete: failed" in post_h2h["diagnostics"]
        assert "upstream incomplete: failed" in seed_symmetry["diagnostics"]
    assert health["stage_statuses"]["interseed_analysis"]["status"] == "missing"
    assert health["stage_statuses"]["h2h_tier_trends"]["status"] == "missing"
    assert health["first_blocking_failure"]["stage"] == "seed_301.analysis"


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

def test_two_seed_pipeline_emits_consistent_config_sha_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.sim.seed_pair = (3, 4)
    assign_config_sha(cfg)

    monkeypatch.setattr(two_seed_pipeline, "seed_has_completion_markers", lambda _cfg: True)
    monkeypatch.setattr(two_seed_pipeline.runner, "run_tournament", lambda *_args, **_kwargs: None)

    def _fake_per_seed(
        seed_cfg: AppConfig,
        *,
        seed: int,
        force: bool = False,
        policy_bundle: object | None = None,
    ) -> None:
        del force, policy_bundle, seed
        seed_cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.analysis_dir / "manifest.jsonl").write_text('{"ok": true}\n', encoding="utf-8")
        seed_cfg.seed_symmetry_stage_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.seed_symmetry_stage_dir / "seed_symmetry_summary.parquet").write_text("ok")
        seed_cfg.post_h2h_stage_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.post_h2h_stage_dir / "h2h_s_tiers.json").write_text("{}")

    monkeypatch.setattr(two_seed_pipeline, "_run_per_seed_analysis", _fake_per_seed)

    def _fake_interseed(interseed_cfg: AppConfig, **_kwargs: object) -> None:
        interseed_cfg.interseed_stage_dir.mkdir(parents=True, exist_ok=True)
        (interseed_cfg.interseed_stage_dir / "interseed_summary.json").write_text("{}")
        interseed_cfg.rng_stage_dir.mkdir(parents=True, exist_ok=True)
        (interseed_cfg.rng_stage_dir / "rng_diagnostics.done.json").write_text(
            json.dumps({"config_sha": interseed_cfg.config_sha}),
            encoding="utf-8",
        )

    monkeypatch.setattr(two_seed_pipeline.analysis, "run_interseed_analysis", _fake_interseed)

    def _fake_h2h(interseed_cfg: AppConfig, **_kwargs: object) -> None:
        (interseed_cfg.stage_dir("h2h_tier_trends") / "s_tier_trends.parquet").write_text("ok")

    monkeypatch.setattr(two_seed_pipeline.analysis, "run_h2h_tier_trends", _fake_h2h)

    two_seed_pipeline.run_pipeline(cfg, seed_pair=(3, 4), force=False)

    expected_sha = cfg.config_sha
    assert expected_sha is not None
    pair_root = seed_utils.seed_pair_root(cfg, (3, 4))

    manifest_records = [
        json.loads(line)
        for line in (pair_root / "two_seed_pipeline_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required_events = [
        rec
        for rec in manifest_records
        if rec.get("event") in {"run_start", "run_end", "seed_start", "stage_end"}
    ]
    assert required_events
    with_sha = [rec for rec in required_events if "config_sha" in rec]
    assert with_sha
    assert all(rec.get("config_sha") == expected_sha for rec in with_sha)

    for seed in (3, 4):
        active_done = pair_root / f"results_seed_{seed}" / "active_config.done.json"
        payload = json.loads(active_done.read_text(encoding="utf-8"))
        assert payload["config_sha"] == expected_sha

    rng_candidates = list((pair_root / "interseed_analysis").glob("*/rng_diagnostics.done.json"))
    assert len(rng_candidates) == 1
    assert json.loads(rng_candidates[0].read_text(encoding="utf-8"))["config_sha"] == expected_sha

    health = json.loads((pair_root / "pipeline_health.json").read_text(encoding="utf-8"))
    assert health["config_sha"] == expected_sha
    assert "config_sha_validation" not in health["stage_statuses"]


def test_two_seed_pipeline_ignores_prior_manifest_sha_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.sim.seed_pair = (31, 32)
    assign_config_sha(cfg)

    pair_root = seed_utils.seed_pair_root(cfg, (31, 32))
    pair_root.mkdir(parents=True, exist_ok=True)
    stale_manifest = pair_root / "two_seed_pipeline_manifest.jsonl"
    stale_manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "run_start",
                        "seed_pair": [31, 32],
                        "results_dir": str(pair_root),
                        "config_sha": "stale_sha",
                    }
                ),
                json.dumps(
                    {
                        "event": "stage_end",
                        "stage": "seed_31.analysis",
                        "status": "missing",
                        "config_sha": "stale_sha",
                    }
                ),
                json.dumps({"event": "run_end", "status": "failed_blocked", "config_sha": "stale_sha"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(two_seed_pipeline, "seed_has_completion_markers", lambda _cfg: True)
    monkeypatch.setattr(two_seed_pipeline.runner, "run_tournament", lambda *_args, **_kwargs: None)

    def _fake_per_seed(
        seed_cfg: AppConfig,
        *,
        seed: int,
        force: bool = False,
        policy_bundle: object | None = None,
    ) -> None:
        del force, policy_bundle, seed
        seed_cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.analysis_dir / "manifest.jsonl").write_text('{"ok": true}\n', encoding="utf-8")
        seed_cfg.seed_symmetry_stage_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.seed_symmetry_stage_dir / "seed_symmetry_summary.parquet").write_text("ok", encoding="utf-8")
        seed_cfg.post_h2h_stage_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.post_h2h_stage_dir / "h2h_s_tiers.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(two_seed_pipeline, "_run_per_seed_analysis", _fake_per_seed)

    def _fake_interseed(interseed_cfg: AppConfig, **_kwargs: object) -> None:
        interseed_cfg.interseed_stage_dir.mkdir(parents=True, exist_ok=True)
        (interseed_cfg.interseed_stage_dir / "interseed_summary.json").write_text("{}", encoding="utf-8")
        interseed_cfg.rng_stage_dir.mkdir(parents=True, exist_ok=True)
        (interseed_cfg.rng_stage_dir / "rng_diagnostics.done.json").write_text(
            json.dumps({"config_sha": interseed_cfg.config_sha}),
            encoding="utf-8",
        )

    monkeypatch.setattr(two_seed_pipeline.analysis, "run_interseed_analysis", _fake_interseed)

    def _fake_h2h(interseed_cfg: AppConfig, **_kwargs: object) -> None:
        (interseed_cfg.stage_dir("h2h_tier_trends") / "s_tier_trends.parquet").write_text(
            "ok", encoding="utf-8"
        )

    monkeypatch.setattr(two_seed_pipeline.analysis, "run_h2h_tier_trends", _fake_h2h)

    two_seed_pipeline.run_pipeline(cfg, seed_pair=(31, 32), force=False)

    health = json.loads((pair_root / "pipeline_health.json").read_text(encoding="utf-8"))
    assert health["status"] == "complete_success"
    assert "config_sha_validation" not in health["stage_statuses"]


def test_two_seed_pipeline_worker_budget_and_artifact_validation(
    base_cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="seed_count must be positive"):
        two_seed_pipeline._per_seed_worker_budget(total_workers=4, seed_count=0)

    base_cfg.ingest.n_jobs = 9
    monkeypatch.setattr("farkle.utils.parallel.os.cpu_count", lambda: 8)

    # Serial seed execution keeps the full worker budget on each seed.
    base_cfg.orchestration.parallel_seeds = False
    base_cfg.sim.n_jobs = 6
    bundle = two_seed_pipeline._derive_per_seed_job_budgets(base_cfg, seed_count=2)
    assert bundle.simulation.process_workers == 6
    assert bundle.ingest.process_workers == 6
    assert bundle.analysis.process_workers == 6

    # Parallel seed execution splits the worker budget across concurrent seeds.
    base_cfg.orchestration.parallel_seeds = True
    base_cfg.sim.n_jobs = 6
    bundle = two_seed_pipeline._derive_per_seed_job_budgets(base_cfg, seed_count=2)
    assert bundle.simulation.process_workers == 3
    assert bundle.ingest.process_workers == 3
    assert bundle.analysis.process_workers == 3

    # Ingest workers remain capped by cfg.ingest.n_jobs.
    base_cfg.ingest.n_jobs = 2
    bundle = two_seed_pipeline._derive_per_seed_job_budgets(base_cfg, seed_count=2)
    assert bundle.simulation.process_workers == 3
    assert bundle.ingest.process_workers == 2
    assert bundle.analysis.process_workers == 3
    base_cfg.ingest.n_jobs = 9

    # Edge case: non-positive sim.n_jobs falls back to cpu_count() and still splits.
    base_cfg.sim.n_jobs = 0
    bundle = two_seed_pipeline._derive_per_seed_job_budgets(base_cfg, seed_count=2)
    assert bundle.simulation.process_workers == 4
    assert bundle.ingest.process_workers == 4
    assert bundle.analysis.process_workers == 4

    # Edge case: seed_count=1 should not reduce workers when running in parallel mode.
    bundle = two_seed_pipeline._derive_per_seed_job_budgets(base_cfg, seed_count=1)
    assert bundle.simulation.process_workers == 8
    assert bundle.ingest.process_workers == 8
    assert bundle.analysis.process_workers == 8

    missing = tmp_path / "missing.json"
    assert two_seed_pipeline._validate_artifact(missing) == (False, "missing", "missing")

    empty = tmp_path / "empty.bin"
    empty.touch()
    assert two_seed_pipeline._validate_artifact(empty) == (False, "empty", "empty")

    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("\n \n", encoding="utf-8")
    assert two_seed_pipeline._validate_artifact(empty_jsonl) == (
        False,
        "empty jsonl",
        "empty",
    )

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    assert two_seed_pipeline._validate_artifact(bad_json) == (
        False,
        "invalid json",
        "invalid_json",
    )

    valid_jsonl = tmp_path / "ok.jsonl"
    valid_jsonl.write_text("\n \n{\"ok\": true}\n", encoding="utf-8")
    assert two_seed_pipeline._validate_artifact(valid_jsonl) == (True, None, None)


def test_two_seed_pipeline_validate_required_config_sha_outputs_collects_errors(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                "",
                "not-json",
                "[]",
                json.dumps({"schema_version": 2, "run_id": "current", "event": "run_start", "config_sha": "wrong"}),
                json.dumps({"schema_version": 2, "run_id": "current", "event": "stage_end", "config_sha": "wrong"}),
                json.dumps({"schema_version": 2, "run_id": "prior", "event": "run_end", "config_sha": "stale"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    seed_a_active = tmp_path / "seed_a_active.yaml"
    seed_b_active = tmp_path / "seed_b_active.yaml"
    seed_b_active.with_suffix(".done.json").write_text("{bad-json", encoding="utf-8")
    seed_contexts = {
        1: SimpleNamespace(active_config_path=seed_a_active),
        2: SimpleNamespace(active_config_path=seed_b_active),
    }

    interseed_cfg = SimpleNamespace(stage_dir_if_active=lambda _name: tmp_path / "rng_stage")

    errors = two_seed_pipeline._validate_required_config_sha_outputs(
        expected_sha="expected",
        manifest_path=manifest_path,
        run_id="current",
        seed_contexts=seed_contexts,
        interseed_cfg=interseed_cfg,
    )

    assert any("invalid metadata" in err and "manifest.jsonl" in err for err in errors)
    assert any(err == "manifest event run_start has config_sha='wrong'" for err in errors)
    assert any(err == "manifest event stage_end has config_sha='wrong'" for err in errors)
    assert any("missing metadata" in err and "seed_a_active.done.json" in err for err in errors)
    assert any("invalid metadata" in err and "seed_b_active.done.json" in err for err in errors)


def test_two_seed_pipeline_resolve_seed_pair_validation_errors() -> None:
    parser = two_seed_pipeline.build_parser()

    with pytest.raises(SystemExit):
        args = parser.parse_args(["--seed-a", "9"])
        seed_utils.resolve_seed_pair_args(args, parser)

    with pytest.raises(SystemExit):
        args = parser.parse_args(["--seed-a", "1", "--seed-b", "2", "--seed-pair", "3", "4"])
        seed_utils.resolve_seed_pair_args(args, parser)


def test_two_seed_pipeline_main_without_cli_seed_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    called: dict[str, Any] = {}

    monkeypatch.setattr(two_seed_pipeline, "setup_info_logging", lambda: called.setdefault("logging", True))
    monkeypatch.setattr(two_seed_pipeline, "load_app_config", lambda _path, seed_list_len=0: cfg)
    monkeypatch.setattr(two_seed_pipeline, "apply_dot_overrides", lambda c, _overrides: c)
    monkeypatch.setattr(cfg.sim, "populate_seed_list", lambda count: [91, 92] if count == 2 else [])
    monkeypatch.setattr(
        two_seed_pipeline,
        "run_pipeline",
        lambda run_cfg, *, seed_pair, force: called.update(
            seed_pair=seed_pair,
            force=force,
            parallel=run_cfg.orchestration.parallel_seeds,
        ),
    )

    rc = two_seed_pipeline.main([])
    assert rc == 0
    assert called["logging"] is True
    assert called["seed_pair"] == (91, 92)
    assert called["force"] is False
    assert called["parallel"] is False


@pytest.mark.parametrize("mode", ["interseed_fail", "h2h_fail"])
def test_two_seed_pipeline_records_interseed_and_h2h_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.sim.seed_pair = (13, 14)

    manifest_events: list[str] = []
    monkeypatch.setattr(
        two_seed_pipeline,
        "append_manifest_event",
        lambda _path, rec, **_kwargs: manifest_events.append(rec["event"]),
    )
    monkeypatch.setattr(two_seed_pipeline, "seed_has_completion_markers", lambda _cfg: True)
    monkeypatch.setattr(two_seed_pipeline.runner, "run_tournament", lambda *_args, **_kwargs: None)

    def _fake_per_seed(
        seed_cfg: AppConfig,
        *,
        seed: int,
        force: bool = False,
        policy_bundle: object | None = None,
    ) -> None:
        del force, policy_bundle, seed
        seed_cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.analysis_dir / "analysis_manifest.jsonl").write_text(
            json.dumps({"event": "run_end"}) + "\n",
            encoding="utf-8",
        )
        seed_cfg.seed_symmetry_stage_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.seed_symmetry_stage_dir / "seed_symmetry_summary.parquet").write_text("ok", encoding="utf-8")
        seed_cfg.post_h2h_stage_dir.mkdir(parents=True, exist_ok=True)
        (seed_cfg.post_h2h_stage_dir / "h2h_s_tiers.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(two_seed_pipeline, "_run_per_seed_analysis", _fake_per_seed)

    def _from_seed_context(cls, seed_context, seed_pair, analysis_root):
        del cls, seed_pair
        interseed_cfg = AppConfig(io=IOConfig(results_dir_prefix=analysis_root / "results_seed_13"))
        interseed_cfg.config_sha = seed_context.config.config_sha
        return SimpleNamespace(config=interseed_cfg)

    monkeypatch.setattr(
        two_seed_pipeline.InterseedRunContext,
        "from_seed_context",
        classmethod(_from_seed_context),
    )

    if mode == "interseed_fail":
        monkeypatch.setattr(
            two_seed_pipeline.analysis,
            "run_interseed_analysis",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("interseed boom")),
        )
        monkeypatch.setattr(two_seed_pipeline.analysis, "run_h2h_tier_trends", lambda *_args, **_kwargs: None)
    else:
        def _interseed_ok(interseed_cfg: AppConfig, **_kwargs: object) -> None:
            interseed_cfg.interseed_stage_dir.mkdir(parents=True, exist_ok=True)
            (interseed_cfg.interseed_stage_dir / "interseed_summary.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(two_seed_pipeline.analysis, "run_interseed_analysis", _interseed_ok)
        monkeypatch.setattr(
            two_seed_pipeline.analysis,
            "run_h2h_tier_trends",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("h2h boom")),
        )

    two_seed_pipeline.run_pipeline(cfg, seed_pair=(13, 14), force=False)

    pair_root = seed_utils.seed_pair_root(cfg, (13, 14))
    health = json.loads((pair_root / "pipeline_health.json").read_text(encoding="utf-8"))
    if mode == "interseed_fail":
        assert "interseed_failed" in manifest_events
        assert health["stage_statuses"]["interseed_analysis"]["status"] == "failed"
    else:
        assert "h2h_tier_trends_start" in manifest_events
        assert health["stage_statuses"]["h2h_tier_trends"]["status"] == "failed"
