"""Operational consent must survive the entire immutable H2H lifecycle."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
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
    schedule.plan_h2h_schedule(cfg)
    manifest = cfg.strategy_manifest_root_path()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"strategy_id": pa.array([], type=pa.int64())}), manifest)
    return cfg, json.loads(cfg.h2h_power_plan_path().read_text())


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
        schedule.plan_h2h_schedule(cfg)
        schedule.execute_h2h_schedule(cfg, block_runner=_runner)
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
        schedule.execute_h2h_schedule(
            cfg, block_runner=lambda *args: pytest.fail("unauthorized work")
        )
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
        schedule.execute_h2h_schedule(cfg, block_runner=_runner)
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
    assert not schedule.plan_h2h_schedule(cfg).execution_blocked
    schedule.execute_h2h_schedule(cfg, block_runner=lambda *args: pytest.fail("replayed work"))
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
        schedule.execute_h2h_schedule(cfg, block_runner=_runner)
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
        schedule.execute_h2h_schedule(cfg, block_runner=interrupt)
    blocks = list(cfg.h2h_block_results_dir().glob("*.parquet"))
    assert len(blocks) == 1
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in blocks}
    prior = json.loads(cfg.h2h_execution_state_path().read_text())
    assert prior["completed_block_count"] == 1
    # Progress alone must never grant permission, even though it was formerly approved.
    cfg.h2h_cap_authorization_path().unlink()
    sidecar_path(cfg.h2h_cap_authorization_path()).unlink()
    assert schedule.plan_h2h_schedule(cfg).execution_blocked
    _terminal(monkeypatch, ["N"])
    with (
        consent.cap_authorization_policy(interactive=True),
        pytest.raises(consent.H2HAuthorizationRequired) as refusal,
    ):
        schedule.execute_h2h_schedule(cfg, block_runner=_runner)
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

    schedule.execute_h2h_schedule(cfg, block_runner=resumed)
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
            schedule.execute_h2h_schedule(cfg, block_runner=stop_after_one)
        cfg.h2h_cap_authorization_path().unlink()
        sidecar_path(cfg.h2h_cap_authorization_path()).unlink()
        assert schedule.plan_h2h_schedule(cfg).execution_blocked
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
        schedule.execute_h2h_schedule(cfg, block_runner=lambda *args: pytest.fail("early work"))
    monkeypatch.setattr(consent, "write_json_artifact_atomic", publish)
    assert len(list(cfg.h2h_block_results_dir().glob("*.parquet"))) == int(partial)
    if boundary == "data_only":
        with pytest.raises(consent.H2HAuthorizationRequired):
            schedule.execute_h2h_schedule(cfg, block_runner=_runner)
        with consent.cap_authorization_policy(override=consent.authorization_token(plan)):
            schedule.execute_h2h_schedule(cfg, block_runner=_runner)
    else:
        assert not schedule.plan_h2h_schedule(cfg).execution_blocked
        schedule.execute_h2h_schedule(cfg, block_runner=_runner)
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
        schedule.plan_h2h_schedule(cfg)


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
        schedule.execute_h2h_schedule(inner, block_runner=_runner)

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
        schedule.execute_h2h_schedule(cfg, block_runner=_runner)
    assert audit_sidecar_completeness(cfg.analysis_dir) == []
    auth = json.loads(cfg.h2h_cap_authorization_path().read_text())
    consent._publish(cfg, plan, {**auth, "authorized_total_game_cap": 17})
    failures = audit_sidecar_completeness(cfg.analysis_dir)
    assert any("cap authorization" in failure for failure in failures)


def test_corrupt_progress_publication_keeps_approval_and_block_resume(planned) -> None:
    cfg, plan = planned
    with consent.cap_authorization_policy(override=consent.authorization_token(plan)):
        schedule.execute_h2h_schedule(cfg, block_runner=_runner)
    authorization_bytes = cfg.h2h_cap_authorization_path().read_bytes()
    cfg.h2h_execution_state_path().write_text("interrupted state publication")
    schedule.plan_h2h_schedule(cfg)
    schedule.execute_h2h_schedule(cfg, block_runner=lambda *args: pytest.fail("replayed block"))
    assert cfg.h2h_cap_authorization_path().read_bytes() == authorization_bytes


def test_renewed_consent_with_same_amount_refreshes_exact_provenance(planned) -> None:
    from farkle.analysis.release_audit import audit_sidecar_completeness

    cfg, plan = planned
    with consent.cap_authorization_policy(override=consent.authorization_token(plan)):
        schedule.execute_h2h_schedule(cfg, block_runner=_runner)
    prior = json.loads(cfg.h2h_execution_state_path().read_text())
    authorization = json.loads(cfg.h2h_cap_authorization_path().read_text())
    # A different consent record can authorize the same amount. Refresh its
    # source binding too, while retaining every accumulated progress field.
    consent._publish(cfg, plan, {**authorization, "authorization_source": "interactive"})
    assert not schedule.plan_h2h_schedule(cfg).execution_blocked
    refreshed = json.loads(cfg.h2h_execution_state_path().read_text())
    assert refreshed["cap_authorization_sha256"] != prior["cap_authorization_sha256"]
    assert {k: v for k, v in refreshed.items() if k != "cap_authorization_sha256"} == {
        k: v for k, v in prior.items() if k != "cap_authorization_sha256"
    }
    schedule.execute_h2h_schedule(cfg, block_runner=lambda *args: pytest.fail("replayed block"))
    assert audit_sidecar_completeness(cfg.analysis_dir) == []
