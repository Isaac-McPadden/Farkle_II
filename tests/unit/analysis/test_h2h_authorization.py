"""Operational consent must survive the entire immutable H2H lifecycle."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.helpers.h2h_oracle import execute_oracle, plan_oracle, publish_fixture_family
from tests.unit.analysis.test_h2h_schedule import _cfg, _write_frozen_family

from farkle.analysis import h2h_authorization as consent
from farkle.analysis import h2h_schedule as schedule
from farkle.analysis.stage_runner import StagePlanItem, StageRunContext, StageRunner
from farkle.cli import main as cli
from farkle.config import AppConfig
from farkle.utils.artifact_contract import sidecar_path
from farkle.utils.stage_completion import CompletionState, read_stage_done, stage_done_path


@pytest.fixture
def planned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[AppConfig, dict[str, Any]]:
    cfg = _cfg(tmp_path)
    cfg.head2head.total_game_cap = 1
    _write_frozen_family(cfg, (1, 2))
    monkeypatch.setattr(schedule, "_minimum_block_games", lambda **kwargs: 2)
    monkeypatch.setattr(schedule, "_worst_scenario_power", lambda **kwargs: 0.9)
    plan_oracle(cfg)
    return cfg, json.loads(cfg.h2h_power_plan_path().read_text())


def test_planning_requires_authenticated_mapping_before_publication(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _write_frozen_family(cfg, (1, 2))
    path = cfg.strategy_manifest_root_path()
    sidecar_path(path).unlink()
    with pytest.raises(RuntimeError, match="authenticated strategy manifest"):
        schedule.plan_h2h_schedule(cfg)
    assert not cfg.h2h_power_plan_path().exists()
    assert not cfg.h2h_block_manifest_path().exists()
    assert not cfg.h2h_cap_authorization_path().exists()


def test_equal_candidate_ids_cannot_transfer_strategy_authorization(tmp_path, monkeypatch):
    from tests.helpers.artifact_sidecars import (
        clean_test_code_identity,
        publish_v3_strategy_manifest,
    )

    from farkle.simulation.strategies import ThresholdStrategy

    monkeypatch.setattr(schedule, "_minimum_block_games", lambda **kwargs: 2)
    monkeypatch.setattr(schedule, "_worst_scenario_power", lambda **kwargs: 0.9)
    plans = []
    for threshold in (300, 900):
        cfg = _cfg(tmp_path / str(threshold))
        cfg._code_identity = clean_test_code_identity()
        cfg.head2head.total_game_cap = 1
        publish_fixture_family(cfg, [1, 2])
        publish_v3_strategy_manifest(
            cfg,
            tuple(ThresholdStrategy(score_threshold=threshold, strategy_id=sid) for sid in (1, 2)),
        )
        schedule.plan_h2h_schedule(cfg)
        plans.append(json.loads(cfg.h2h_power_plan_path().read_bytes()))
    assert plans[0]["family_hash"] == plans[1]["family_hash"]
    assert plans[0]["schedule_hash"] != plans[1]["schedule_hash"]
    with (
        consent.cap_authorization_policy(override=consent.authorization_token(plans[0])),
        pytest.raises(consent.H2HAuthorizationRequired, match="override_mismatch"),
    ):
        consent.resolve_cap_authorization(cfg, plans[1], request=True)

    frozen = cfg.h2h_cap_authorization_path().read_bytes()
    with pytest.raises(RuntimeError, match="Immutable root strategy manifest"):
        publish_v3_strategy_manifest(
            cfg, tuple(ThresholdStrategy(strategy_id=sid) for sid in (1, 2))
        )
    # Replace both files externally with another valid publication to exercise the
    # independent consumer check even when a writer bypasses frozen publication.
    import shutil

    other = _cfg(tmp_path / "300").strategy_manifest_root_path()
    shutil.copyfile(other, cfg.strategy_manifest_root_path())
    shutil.copyfile(sidecar_path(other), sidecar_path(cfg.strategy_manifest_root_path()))
    with pytest.raises(RuntimeError, match="frozen source|frozen H2H"):
        schedule.execute_h2h_schedule(cfg)
    assert cfg.h2h_cap_authorization_path().read_bytes() == frozen


def test_mixed_release_audit_checks_terminal_blocks_and_original_consent(tmp_path, monkeypatch):
    from farkle.analysis.candidate_family import verify_family_identity
    from farkle.analysis.release_audit import (
        _audit_frozen_execution,
        _build_audit_index,
        _validate_indexed_artifacts,
    )

    cfg = _cfg(tmp_path)
    cfg.head2head.total_game_cap = 1
    _write_frozen_family(cfg, (1, 2))
    monkeypatch.setattr(schedule, "_minimum_block_games", lambda **_: 2)
    monkeypatch.setattr(schedule, "_worst_scenario_power", lambda **_: 0.9)
    schedule.plan_h2h_schedule(cfg)
    plan = json.loads(cfg.h2h_power_plan_path().read_bytes())
    with consent.cap_authorization_policy(override=consent.authorization_token(plan)):
        schedule.execute_h2h_schedule(cfg)
    index, failures = _build_audit_index(cfg.results_root.resolve())
    assert not failures
    assert not _validate_indexed_artifacts(index)
    _audit_frozen_execution(cfg, index)
    family = json.loads(cfg.h2h_candidate_family_manifest_path().read_bytes())
    family["final_cutoffs"]["win_rate"] += 1
    with pytest.raises(ValueError, match="candidate family differs"):
        verify_family_identity(cfg, family)
    approval = cfg.h2h_cap_authorization_path()
    original = approval.read_bytes()
    forged = json.loads(original)
    forged["block_manifest_sha256"] = "b" * 64
    approval.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="authorization source changed"):
        _audit_frozen_execution(cfg, index)
    approval.write_bytes(original)
    block = next(cfg.h2h_block_results_dir().glob("*.parquet"))
    block.unlink()
    with pytest.raises(ValueError, match="block differs"):
        _audit_frozen_execution(cfg, index)


def test_legacy_authorization_reader_preserves_identity_but_cannot_issue_consent(tmp_path):
    from farkle.utils.frozen_h2h import verify_frozen_strategy_design

    legacy = {"schedule_hash": "a" * 64, "maximum_total_attempts": 10}
    # Golden identity from the pre-scientific-design schedule contract.
    assert (
        schedule._schedule_hash(
            family_hash="a" * 64,
            roots=(11, 22),
            effect=0.03,
            family_alpha=0.05,
            alpha_per_pair=0.01,
            target_power=0.8,
            scenarios=(0.0, 0.05),
            block_games=4,
            max_attempt_multiplier=2.0,
            min_candidate_completion_rate=0.99,
            game_profile_sha256=None,
        )
        == "6438181305bfdcdca0de0300d1d7db39386084ffcdcc0dca900cc63762c2251c"
    )
    assert consent.authorization_token(legacy) == f"{'a' * 64}:10"
    with pytest.raises(RuntimeError, match="Historical H2H plan"):
        verify_frozen_strategy_design(_cfg(tmp_path), legacy)


def test_root_force_preserves_authenticated_mapping_before_and_after_external_pair_freeze(
    tmp_path, monkeypatch
):
    from tests.helpers.artifact_sidecars import (
        make_authenticated_v3_config,
        publish_v3_strategy_manifest,
    )

    from farkle.orchestration.run_contexts import RootPairRunContext, SeedRunContext
    from farkle.simulation import runner
    from farkle.simulation.strategies import ThresholdStrategy
    from farkle.utils.frozen_h2h import verify_frozen_strategy_design

    first, second = (
        SeedRunContext.from_config(
            make_authenticated_v3_config(tmp_path, name=f"root_{seed}", root_seed=seed)
        )
        for seed in (11, 22)
    )
    root = first.config
    old = [ThresholdStrategy(strategy_id=i, score_threshold=100 * i) for i in (1, 2)]
    new = [ThresholdStrategy(strategy_id=i, score_threshold=200 * i) for i in (1, 2)]
    mapping = publish_v3_strategy_manifest(root, tuple(old))
    files = (mapping, sidecar_path(mapping))
    original = [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]
    checkpoint = root.n_dir(2) / "2p_checkpoint.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"existing checkpoint")

    def require_refusal():
        with pytest.raises(ValueError, match="manifest mismatch.*new output root"):
            runner.run_single_n(root, 2, new, force=True)
        assert checkpoint.read_bytes() == b"existing checkpoint"
        assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in files] == original

    require_refusal()
    pair = RootPairRunContext.from_root_contexts(
        (first, second), pair_root=tmp_path / "external_pair"
    ).config
    publish_fixture_family(pair, [1, 2])
    monkeypatch.setattr(schedule, "_minimum_block_games", lambda **_: 1)
    monkeypatch.setattr(schedule, "_worst_scenario_power", lambda **_: 0.9)
    schedule.plan_h2h_schedule(pair)
    plan = json.loads(pair.h2h_power_plan_path().read_bytes())
    consent_bytes = pair.h2h_cap_authorization_path().read_bytes()
    require_refusal()
    with pytest.raises(ValueError, match="Cannot establish simulation cleanup ownership"):
        runner.run_single_n(root, 2, old, force=True)
    assert checkpoint.read_bytes() == b"existing checkpoint"
    checkpoint.unlink()

    def stop_before_games(**_):
        raise InterruptedError("fixture stops before games")

    monkeypatch.setattr(runner.tournament_mod, "run_tournament", stop_before_games)
    with pytest.raises(InterruptedError, match="fixture stops before games"):
        runner.run_single_n(root, 2, old, force=True)
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in files] == original
    verify_frozen_strategy_design(pair, plan)
    assert pair.h2h_cap_authorization_path().read_bytes() == consent_bytes


def test_builtin_and_oracle_consent_cannot_cross_the_execution_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule, "_minimum_block_games", lambda **kwargs: 2)
    monkeypatch.setattr(schedule, "_worst_scenario_power", lambda **kwargs: 0.9)
    tokens = []
    for engine, planner in (("scientific", schedule.plan_h2h_schedule), ("oracle", plan_oracle)):
        cfg = _cfg(tmp_path / engine)
        cfg.head2head.total_game_cap = 1
        _write_frozen_family(cfg, (1, 2))
        planner(cfg)
        plan = json.loads(cfg.h2h_power_plan_path().read_bytes())
        token = consent.authorization_token(plan)
        tokens.append(token)
        original = cfg.h2h_cap_authorization_path().read_bytes()
        with (
            consent.cap_authorization_policy(override=token),
            pytest.raises(RuntimeError, match="engine does not match"),
        ):
            if engine == "scientific":
                execute_oracle(cfg, block_runner=lambda *_: pytest.fail("unapproved engine ran"))
            else:
                schedule.execute_h2h_schedule(cfg)
        assert cfg.h2h_cap_authorization_path().read_bytes() == original
        assert not cfg.h2h_block_results_dir().exists()
    assert tokens[0] != tokens[1]


def _runner(block: dict[str, Any], _manifest: Path, _chunk: int) -> dict[str, Any]:
    games = int(block["n_completed_required"])
    return {**block, "games_completed": games, "wins_seat1": 1, "wins_seat2": games - 1}


def _terminal(monkeypatch: pytest.MonkeyPatch, answers: list[Any]) -> list[str]:
    monkeypatch.setattr(consent.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    prompts: list[str] = []
    iterator = iter(answers)

    def reply(prompt: str) -> str:
        prompts.append(prompt)
        answer = next(iterator)
        if isinstance(answer, BaseException):
            raise answer
        return str(answer)

    monkeypatch.setattr("builtins.input", reply)
    return prompts


@pytest.mark.parametrize("cap", [16, 17, None])
def test_under_equal_unlimited_never_prompt(planned, monkeypatch, cap) -> None:
    cfg, plan = planned
    cfg.head2head.total_game_cap = cap
    prompts = _terminal(monkeypatch, [])
    with consent.cap_authorization_policy(interactive=True):
        plan_oracle(cfg)
        execute_oracle(cfg, block_runner=_runner)
    assert not prompts
    assert plan["maximum_total_attempts"] == 16


def test_103_percent_description() -> None:
    description = consent.budget_description(
        {
            "schedule_hash": "a" * 64,
            "maximum_total_attempts": 10300,
            "total_completed_required": 5150,
        },
        10000,
    )
    assert "103.00% of cap" in description
    assert "percentage over cap: 3.00%" in description
    assert "5,150" in description and "10,300" in description


@pytest.mark.parametrize(
    "answer,reason",
    [
        ("n", "refused"),
        ("N", "refused"),
        ("", "blank"),
        ("  ", "blank"),
        (EOFError(), "eof"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_refusal_preserves_work_and_never_executes(planned, monkeypatch, answer, reason) -> None:
    cfg, _ = planned
    before = {p: p.read_bytes() for p in cfg.analysis_dir.rglob("*") if p.is_file()}
    _terminal(monkeypatch, [answer])
    with (
        consent.cap_authorization_policy(interactive=True),
        pytest.raises(consent.H2HAuthorizationRequired) as error,
    ):
        execute_oracle(cfg, block_runner=lambda *args: pytest.fail("unauthorized work"))
    assert error.value.reason == reason
    assert "--h2h-authorize" in str(error.value)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not cfg.h2h_block_results_dir().exists()
    assert (
        read_stage_done(stage_done_path(cfg.stage_dir("h2h_power"), "h2h_power"))["status"]
        == "success"
    )


def test_invalid_then_lowercase_approval_survives_replay_and_finalization(
    planned, monkeypatch
) -> None:
    cfg, plan = planned
    immutable = {
        p: p.read_bytes() for p in (cfg.h2h_power_plan_path(), cfg.h2h_block_manifest_path())
    }
    prompts = _terminal(monkeypatch, ["yes", "?", " y "])
    with consent.cap_authorization_policy(interactive=True):
        execute_oracle(cfg, block_runner=_runner)
    assert len(prompts) == 3 and all(consent.PROMPT in p for p in prompts)
    authorization = json.loads(cfg.h2h_cap_authorization_path().read_text())
    assert authorization["authorized_total_game_cap"] == plan["maximum_total_attempts"]
    assert authorization["authorization_source"] == "interactive"
    assert cfg.head2head.total_game_cap == 1
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in cfg.analysis_dir.rglob("*")
        if p.is_file()
    }
    assert not plan_oracle(cfg).execution_blocked
    execute_oracle(cfg, block_runner=lambda *args: pytest.fail("replayed work"))
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == data for p, data in before.items())
    assert all(p.read_bytes() == data for p, data in immutable.items())
    done = read_stage_done(stage_done_path(cfg.stage_dir("h2h_execute"), "h2h_execute"))
    assert "cap_authorization.json" in json.dumps(done)
    state = json.loads(cfg.h2h_execution_state_path().read_text())
    assert state["execution_authorization"] == "ready"
    assert state["authorized_total_game_cap"] == 16


@pytest.mark.parametrize(
    "override,reason", [(None, "noninteractive"), ("wrong:16", "override_mismatch")]
)
def test_noninteractive_does_not_read_stdin(planned, monkeypatch, override, reason) -> None:
    cfg, _ = planned
    monkeypatch.setattr(consent.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("read stdin"))
    with (
        consent.cap_authorization_policy(interactive=True, override=override),
        pytest.raises(consent.H2HAuthorizationRequired) as error,
    ):
        execute_oracle(cfg, block_runner=_runner)
    assert error.value.reason == reason


def test_explicit_override_and_partial_resume_require_independent_consent(
    planned, monkeypatch
) -> None:
    cfg, plan = planned
    calls = []

    def interrupt(block, manifest, chunk):
        calls.append(block["block_id"])
        if len(calls) == 2:
            raise RuntimeError("injected interruption")
        return _runner(block, manifest, chunk)

    with (
        consent.cap_authorization_policy(override=consent.authorization_token(plan)),
        pytest.raises(RuntimeError, match="injected"),
    ):
        execute_oracle(cfg, block_runner=interrupt)
    blocks = list(cfg.h2h_block_results_dir().glob("*.parquet"))
    assert len(blocks) == 1
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in blocks}
    prior = json.loads(cfg.h2h_execution_state_path().read_text())
    assert prior["completed_block_count"] == 1
    # Progress alone must never grant permission, even though it was formerly approved.
    cfg.h2h_cap_authorization_path().unlink()
    sidecar_path(cfg.h2h_cap_authorization_path()).unlink()
    assert plan_oracle(cfg).execution_blocked
    _terminal(monkeypatch, ["N"])
    with (
        consent.cap_authorization_policy(interactive=True),
        pytest.raises(consent.H2HAuthorizationRequired) as refusal,
    ):
        execute_oracle(cfg, block_runner=_runner)
    assert refusal.value.reason == "refused"
    state = json.loads(cfg.h2h_execution_state_path().read_text())
    assert state["completed_block_count"] == 1
    assert state["execution_state"] == CompletionState.PARTIAL_RESUMABLE.value
    with consent.cap_authorization_policy(override=consent.authorization_token(plan)):
        consent.resolve_cap_authorization(cfg, plan, request=True)
    refreshed = schedule._refresh_execution_authorization(cfg, plan, blocked=True)
    for field in prior.keys() - {
        "execution_authorization",
        "authorized_total_game_cap",
        "cap_authorization_sha256",
    }:
        assert refreshed[field] == prior[field]
    remaining = []

    def resumed(block, manifest, chunk):
        remaining.append(block["block_id"])
        return _runner(block, manifest, chunk)

    execute_oracle(cfg, block_runner=resumed)
    assert len(remaining) == 3 and calls[0] not in remaining
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == data for p, data in before.items())


@pytest.mark.parametrize("boundary", ["data_only", "published"])
@pytest.mark.parametrize("partial", [False, True])
def test_authorization_publication_interruption(planned, monkeypatch, boundary, partial) -> None:
    cfg, plan = planned
    if partial:
        calls = []

        def stop_after_one(block, manifest, chunk):
            calls.append(block["block_id"])
            if len(calls) == 2:
                raise RuntimeError("pause after one block")
            return _runner(block, manifest, chunk)

        with (
            consent.cap_authorization_policy(override=consent.authorization_token(plan)),
            pytest.raises(RuntimeError, match="pause after one"),
        ):
            execute_oracle(cfg, block_runner=stop_after_one)
        cfg.h2h_cap_authorization_path().unlink()
        sidecar_path(cfg.h2h_cap_authorization_path()).unlink()
        assert plan_oracle(cfg).execution_blocked
    blocks = list(cfg.h2h_block_results_dir().glob("*.parquet"))
    retained = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in blocks}
    assert len(blocks) == int(partial)
    publish = consent.write_json_artifact_atomic

    def interrupted(payload, path, **kwargs):
        if payload["authorization_source"] == "explicit_override":
            if boundary == "published":
                publish(payload, path, **kwargs)
            else:
                path.write_text(json.dumps(payload))
            raise RuntimeError("publication interruption")
        return publish(payload, path, **kwargs)

    monkeypatch.setattr(consent, "write_json_artifact_atomic", interrupted)
    with (
        consent.cap_authorization_policy(override=consent.authorization_token(plan)),
        pytest.raises(RuntimeError, match="publication interruption"),
    ):
        execute_oracle(cfg, block_runner=lambda *args: pytest.fail("early work"))
    monkeypatch.setattr(consent, "write_json_artifact_atomic", publish)
    assert len(list(cfg.h2h_block_results_dir().glob("*.parquet"))) == int(partial)
    if boundary == "data_only":
        with pytest.raises(consent.H2HAuthorizationRequired):
            execute_oracle(cfg, block_runner=_runner)
        with consent.cap_authorization_policy(override=consent.authorization_token(plan)):
            execute_oracle(cfg, block_runner=_runner)
    else:
        assert not plan_oracle(cfg).execution_blocked
        execute_oracle(cfg, block_runner=_runner)
    assert all(
        (path.read_bytes(), path.stat().st_mtime_ns) == data for path, data in retained.items()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schedule_hash", "b" * 64),
        ("family_hash", "c" * 64),
        ("maximum_total_attempts", 17),
        ("authorized_total_game_cap", 17),
        ("authorized_total_game_cap", True),
    ],
)
def test_authenticated_but_mismatched_authorization_rejected(planned, field, value) -> None:
    cfg, plan = planned
    with consent.cap_authorization_policy(override=consent.authorization_token(plan)):
        auth = consent.resolve_cap_authorization(cfg, plan, request=True)
    consent._publish(cfg, plan, {**auth, field: value})
    with pytest.raises(ValueError):
        plan_oracle(cfg)


def test_stage_runner_records_operational_stop_and_stops_even_continue_on_error(
    planned, tmp_path
) -> None:
    cfg, _ = planned
    manifest = tmp_path / "events.jsonl"
    context = StageRunContext(
        config=cfg,
        manifest_path=manifest,
        run_label="consent",
        logger=logging.getLogger(__name__),
        continue_on_error=True,
    )

    def execute(inner: AppConfig) -> None:
        execute_oracle(inner, block_runner=_runner)

    with pytest.raises(consent.H2HAuthorizationRequired):
        StageRunner().run(
            [
                StagePlanItem(
                    "h2h_execute",
                    execute,
                ),
                StagePlanItem("h2h_inference", lambda inner: pytest.fail("downstream ran")),
            ],
            context,
        )
    events = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert any(
        e.get("status") == "blocked_by_cap" and e.get("reason") == "noninteractive" for e in events
    )
    assert events[-1]["health"] == "blocked_by_cap"


def test_cli_policy_is_scoped_and_refusal_has_distinct_exit(monkeypatch) -> None:
    def dispatch(args, parser):
        assert consent._POLICY.get() == (True, "a:16")
        raise consent.H2HAuthorizationRequired("refused", "resume instructions")

    monkeypatch.setattr(cli, "_dispatch", dispatch)
    with pytest.raises(SystemExit) as error:
        cli.main(["--h2h-authorize", "a:16", "two-seed-pipeline"])
    assert error.value.code == 3
    assert consent._POLICY.get() == (False, None)


def test_final_byte_audit_retains_and_checks_authorization(planned) -> None:
    from farkle.analysis.release_audit import audit_sidecar_completeness

    cfg, plan = planned
    with consent.cap_authorization_policy(override=consent.authorization_token(plan)):
        execute_oracle(cfg, block_runner=_runner)
    failures = audit_sidecar_completeness(cfg.analysis_dir)
    assert len(failures) == 1 and "ineligible for scientific release" in failures[0]
    auth = json.loads(cfg.h2h_cap_authorization_path().read_text())
    consent._publish(cfg, plan, {**auth, "authorized_total_game_cap": 17})
    failures = audit_sidecar_completeness(cfg.analysis_dir)
    assert any("cap authorization" in failure for failure in failures)


def test_corrupt_progress_publication_keeps_approval_and_block_resume(planned) -> None:
    cfg, plan = planned
    with consent.cap_authorization_policy(override=consent.authorization_token(plan)):
        execute_oracle(cfg, block_runner=_runner)
    authorization_bytes = cfg.h2h_cap_authorization_path().read_bytes()
    cfg.h2h_execution_state_path().write_text("interrupted state publication")
    plan_oracle(cfg)
    execute_oracle(cfg, block_runner=lambda *args: pytest.fail("replayed block"))
    assert cfg.h2h_cap_authorization_path().read_bytes() == authorization_bytes


def test_renewed_consent_with_same_amount_refreshes_exact_provenance(planned) -> None:
    from farkle.analysis.release_audit import audit_sidecar_completeness

    cfg, plan = planned
    with consent.cap_authorization_policy(override=consent.authorization_token(plan)):
        execute_oracle(cfg, block_runner=_runner)
    prior = json.loads(cfg.h2h_execution_state_path().read_text())
    authorization = json.loads(cfg.h2h_cap_authorization_path().read_text())
    # A different consent record can authorize the same amount. Refresh its
    # source binding too, while retaining every accumulated progress field.
    consent._publish(cfg, plan, {**authorization, "authorization_source": "interactive"})
    assert not plan_oracle(cfg).execution_blocked
    refreshed = json.loads(cfg.h2h_execution_state_path().read_text())
    assert refreshed["cap_authorization_sha256"] != prior["cap_authorization_sha256"]
    assert {k: v for k, v in refreshed.items() if k != "cap_authorization_sha256"} == {
        k: v for k, v in prior.items() if k != "cap_authorization_sha256"
    }
    execute_oracle(cfg, block_runner=lambda *args: pytest.fail("replayed block"))
    failures = audit_sidecar_completeness(cfg.analysis_dir)
    assert len(failures) == 1 and "ineligible for scientific release" in failures[0]


def test_external_frozen_pair_retains_root_rating_and_upstream_evidence(tmp_path):
    import pandas as pd
    from tests.helpers.artifact_sidecars import make_authenticated_v3_config
    from tests.helpers.h2h_oracle import write_fixture_contribution
    from tests.unit.analysis.test_trueskill_screening import (
        _publish_game_source,
        _publish_strategy_control,
        _ratings,
    )

    from farkle.analysis.candidate_family import freeze_h2h_candidate_family
    from farkle.analysis.h2h_schedule import plan_h2h_schedule
    from farkle.analysis.release_audit import (
        _build_audit_index,
        _validate_cross_context_provenance,
        _validate_indexed_artifacts,
    )
    from farkle.analysis.run_trueskill import _load_ratings_parquet, _save_ratings_parquet
    from farkle.analysis.trueskill_screening import (
        ScreeningRatingCell,
        build_percentile_contribution,
        publish_rating_cell_contract,
    )
    from farkle.config import ArtifactScope
    from farkle.orchestration.run_contexts import RootPairRunContext, SeedRunContext
    from farkle.utils.artifact_contract import sha256_file, sidecar_path
    from farkle.utils.authenticated_contract import load_authenticated_sidecar
    from farkle.utils.frozen_h2h import verify_frozen_strategy_design
    from farkle.utils.scientific_invocation import scientific_invocation

    root = tmp_path
    roots = tuple(
        SeedRunContext.from_config(
            make_authenticated_v3_config(root, name=f"root_{seed}", root_seed=seed)
        )
        for seed in (11, 17)
    )
    pair = RootPairRunContext.from_root_contexts(
        (roots[0], roots[1]), pair_root=root / "external_pair"
    ).config
    pair.screening.candidate_contribution_size = 2
    stages = [(r.config, s) for r in roots for s in ("simulation", "curate", "trueskill")]
    stages.extend(
        (pair, s)
        for s in (
            "trueskill",
            "root_stability",
            "candidate_freeze",
            "h2h_power",
            "h2h_execute",
            "simulation",
        )
    )
    with scientific_invocation(stages):
        fixture = _ratings(root / "sample.parquet", {"1": (30.0, 2.0), "2": (20.0, 3.0)})
        cells = []
        for context in roots:
            cfg = context.config
            _publish_strategy_control(cfg)
            source_path = _publish_game_source(cfg, 2)
            rating = cfg.trueskill_rating_path(2, root_seed=cfg.sim.seed)
            _save_ratings_parquet(rating, _load_ratings_parquet(fixture))
            cell = ScreeningRatingCell(cfg.sim.seed, 2, rating, source_path)
            publish_rating_cell_contract(cfg, cell, completed_artifact_sha256=sha256_file(rating))
            cells.append(cell)
        contrib = build_percentile_contribution(pair, cells)
        write_fixture_contribution(
            pair,
            pair.root_combined_performance_across_k_path(),
            pd.DataFrame(
                {
                    "strategy": [1, 2],
                    "complete_support": [True, True],
                    "across_k_score": [0.7, 0.3],
                    "estimate_scope": ["combined_roots"] * 2,
                }
            ),
            scope=ArtifactScope.CROSS_SEED,
            operation="equal_k_mean",
            seed_scope="root_pair_stability",
        )
        freeze_h2h_candidate_family(pair)
        plan_h2h_schedule(pair)
        plan = json.loads(pair.h2h_power_plan_path().read_bytes())
        records = [
            pair.h2h_candidate_family_manifest_path(),
            pair.h2h_power_plan_path(),
            pair.h2h_block_manifest_path(),
            pair.h2h_cap_authorization_path(),
            contrib,
            roots[0].config.strategy_manifest_root_path(),
        ]
        originals = [(sha256_file(p), sha256_file(sidecar_path(p))) for p in records]
        cell = cells[0]
        path = cell.ratings_path

        def audit():
            indexes = []
            for graph in [r.results_root for r in roots] + [root / "external_pair"]:
                index, _ = _build_audit_index(graph)
                assert not _validate_indexed_artifacts(index)
                indexes.append(index)
            return [m for m in _validate_cross_context_provenance(indexes) if str(contrib) in m]

        assert not audit()
        previous = load_authenticated_sidecar(path)
        source = next(
            s
            for s in load_authenticated_sidecar(contrib).source_artifacts
            if s.sidecar_contract_sha256 == previous.sidecar_contract_sha256
        )
        values = _load_ratings_parquet(path)
        times = (path.stat().st_mtime_ns, sidecar_path(path).stat().st_mtime_ns)
        _save_ratings_parquet(path, values)
        publish_rating_cell_contract(
            roots[0].config, cell, completed_artifact_sha256=sha256_file(path)
        )
        values["1"].mu += 1
        with pytest.raises(RuntimeError, match="Immutable scientific input.*new output root"):
            _save_ratings_parquet(path, values)
        assert times == (path.stat().st_mtime_ns, sidecar_path(path).stat().st_mtime_ns)
        assert source.artifact.content_sha256 == sha256_file(path)
        assert source.sidecar_sha256 == sha256_file(sidecar_path(path))
        assert originals == [(sha256_file(p), sha256_file(sidecar_path(p))) for p in records]
        assert not audit()
        verify_frozen_strategy_design(pair, plan)
