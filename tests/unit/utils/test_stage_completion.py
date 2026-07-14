from __future__ import annotations

import json
import time
from pathlib import Path

from farkle.utils.stage_completion import (
    CompletionState,
    freshness_sha256,
    read_stage_done,
    resolve_stage_state,
    stage_done_path,
    stage_is_up_to_date,
    write_stage_done,
)


def test_stage_done_roundtrip_and_status_handling(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    done = stage_done_path(stage_dir, "metrics")
    inp = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    inp.write_text("in")
    out.write_text("out")

    assert not stage_is_up_to_date(done, [inp], [out], config_sha="abc")

    write_stage_done(done, inputs=[inp], outputs=[out], config_sha="abc")
    assert stage_is_up_to_date(done, [inp], [out], config_sha="abc")
    assert not stage_is_up_to_date(done, [inp], [out], config_sha="other")

    meta = read_stage_done(done)
    assert meta["status"] == "success"


def test_stage_is_up_to_date_false_when_input_newer_or_skipped(tmp_path: Path) -> None:
    done = tmp_path / "x.done.json"
    inp = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    inp.write_text("in")
    out.write_text("out")

    write_stage_done(done, inputs=[inp], outputs=[out], stage_config_sha="cache")
    assert stage_is_up_to_date(done, [inp], [out], stage_config_sha="cache")

    time.sleep(0.01)
    inp.write_text("newer")
    assert not stage_is_up_to_date(done, [inp], [out], stage_config_sha="cache")

    write_stage_done(
        done,
        inputs=[inp],
        outputs=[out],
        stage_config_sha="cache",
        status="skipped",
        blocking_dependency="y",
        upstream_stage="z",
    )
    assert not stage_is_up_to_date(done, [inp], [out], stage_config_sha="cache")


def test_resolve_stage_state_distinguishes_all_lifecycle_states(tmp_path: Path) -> None:
    done = tmp_path / "stage.done.json"
    output = tmp_path / "output.parquet"
    checkpoint = tmp_path / "checkpoint.json"

    assert resolve_stage_state(done, [], [output]) is CompletionState.NOT_STARTED

    checkpoint.write_text("partial", encoding="utf-8")
    assert (
        resolve_stage_state(done, [], [output], partial_paths=[checkpoint])
        is CompletionState.PARTIAL_RESUMABLE
    )

    output.write_text("complete", encoding="utf-8")
    freshness = {"estimand_version": 1, "rng_scheme_version": 1}
    write_stage_done(
        done,
        inputs=[],
        outputs=[output],
        stage_config_sha="stage-sha",
        freshness_key=freshness,
    )
    assert (
        resolve_stage_state(
            done,
            [],
            [output],
            stage_config_sha="stage-sha",
            freshness_key=freshness,
        )
        is CompletionState.COMPLETE_VALID
    )
    assert (
        resolve_stage_state(
            done,
            [],
            [output],
            stage_config_sha="stage-sha",
            freshness_key={**freshness, "estimand_version": 2},
        )
        is CompletionState.COMPLETE_STALE
    )
    assert (
        resolve_stage_state(done, [], [output], cap_reached=True) is CompletionState.BLOCKED_BY_CAP
    )


def test_blocked_by_cap_stamp_is_explicit_and_resumable_after_cap_change(tmp_path: Path) -> None:
    done = tmp_path / "stage.done.json"
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("batch=10", encoding="utf-8")
    write_stage_done(
        done,
        inputs=[],
        outputs=[checkpoint],
        status="blocked_by_cap",
        reason="configured workload cap reached",
    )

    assert resolve_stage_state(done, [], [checkpoint]) is CompletionState.BLOCKED_BY_CAP
    done.unlink()
    assert (
        resolve_stage_state(done, [], [], partial_paths=[checkpoint])
        is CompletionState.PARTIAL_RESUMABLE
    )


def test_old_completion_schema_cannot_validate_replacement_artifact(tmp_path: Path) -> None:
    done = tmp_path / "stage.done.json"
    output = tmp_path / "replacement.parquet"
    output.write_text("replacement", encoding="utf-8")
    done.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "success",
                "stage_config_sha": "same",
                "cache_key_version": 2,
                "outputs": [str(output)],
            }
        ),
        encoding="utf-8",
    )

    assert (
        resolve_stage_state(done, [], [output], stage_config_sha="same")
        is CompletionState.COMPLETE_STALE
    )
    assert not stage_is_up_to_date(done, [], [output], stage_config_sha="same")


def test_freshness_digest_is_canonical_and_stamp_records_contract(tmp_path: Path) -> None:
    first = {"weighting_version": 1, "required_player_counts": [2, 4]}
    second = {"required_player_counts": [2, 4], "weighting_version": 1}
    assert freshness_sha256(first) == freshness_sha256(second)

    done = tmp_path / "stage.done.json"
    output = tmp_path / "output"
    output.write_text("done", encoding="utf-8")
    write_stage_done(done, inputs=[], outputs=[output], freshness_key=first)
    metadata = read_stage_done(done)
    assert metadata["freshness_key"] == first
    assert metadata["freshness_sha256"] == freshness_sha256(first)


def test_stage_hash_may_bind_freshness_for_streaming_substage_stamps(tmp_path: Path) -> None:
    done = tmp_path / "substage.done.json"
    output = tmp_path / "output"
    output.write_text("done", encoding="utf-8")
    write_stage_done(
        done,
        inputs=[],
        outputs=[output],
        stage_config_sha="hash-includes-freshness",
    )

    assert (
        resolve_stage_state(
            done,
            [],
            [output],
            stage_config_sha="hash-includes-freshness",
            freshness_key={"estimand_version": 1},
        )
        is CompletionState.COMPLETE_VALID
    )
