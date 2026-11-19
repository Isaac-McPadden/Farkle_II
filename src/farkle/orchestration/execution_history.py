"""Retain execution evidence at the two-seed publisher's existing ownership boundaries."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from farkle.orchestration.run_contexts import (
    RootPairRunContext,
    SeedRunContext,
    load_run_context,
    write_run_context_atomic,
)
from farkle.orchestration.seed_utils import write_active_config
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.authenticated_contract import CodeIdentity, canonical_json_bytes
from farkle.utils.authenticated_graph import AuthenticatedGraphSnapshot
from farkle.utils.manifest import append_manifest_event
from farkle.utils.stage_completion import freshness_sha256
from farkle.utils.writer import atomic_path

_FILES = ("run_context.json", "active_config.yaml", "active_config.done.json")


def invocation_dir(root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError(f"invalid invocation ID: {run_id!r}")
    return root / "_provenance" / "invocations" / run_id


def _digest(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def resume_context_path(root: Path) -> Path:
    """Include a committed but interrupted transition in the normal compatibility check."""
    pending = root / "_provenance" / "publishing"
    if pending.exists():
        context = pending / "run_context.json"
        if not context.is_file():
            raise ValueError(f"interrupted publication is missing its run context: {pending}")
        return context
    return root / "run_context.json"


def _finish_publication(root: Path) -> None:
    pending = root / "_provenance" / "publishing"
    if not pending.exists():
        return
    receipt = json.loads((pending / "invocation.json").read_bytes())
    final = invocation_dir(root, receipt["run_id"])
    if receipt["version"] != 1:
        raise ValueError(f"unsupported invocation record: {pending}")
    if final.exists():
        raise FileExistsError(f"execution history already exists: {final}")
    for name in _FILES:
        old, new = receipt["previous"][name], receipt["current"][name]
        if (
            _digest(pending / "previous" / name) != old
            or sha256_file(pending / name) != new
            or _digest(root / name) not in (old, new)
        ):
            raise ValueError(f"interrupted publication has conflicting evidence for {root / name}")
    for index, (stage, _, digest) in enumerate(receipt["previous_completion_stamps"]):
        if _digest(pending / "previous" / f"{index}.stamp") != digest:
            raise ValueError(f"retained pre-resume completion differs for {stage}: {pending}")
    load_run_context(
        pending / "run_context.json", active_config_path=pending / "active_config.yaml"
    )
    for name in _FILES:
        if _digest(root / name) != receipt["current"][name]:
            with atomic_path(str(root / name)) as temporary:
                shutil.copyfile(pending / name, temporary)
    pending.rename(final)


def publish_invocation(
    context: SeedRunContext | RootPairRunContext,
    *,
    run_id: str,
    manifest_path: Path,
    completion_paths: Sequence[tuple[str, Path]],
    code_identity: CodeIdentity,
    parent_lifecycle_roots: tuple[str, ...] = (),
    cli_overrides: tuple[str, ...] = (),
    worker_counts: Mapping[str, Mapping[str, object]] | None = None,
    game_profile_sha256: str | None = None,
) -> None:
    """Commit both generations before replacing any live file or starting a producer."""
    root = context.run_context_path.parent
    final = invocation_dir(root, run_id)
    final.parent.mkdir(parents=True, exist_ok=True)
    _finish_publication(root)
    if final.exists():
        raise FileExistsError(f"execution history already exists: {final}")
    history = final.parent.parent
    with TemporaryDirectory(dir=history) as scratch:
        bundle = Path(scratch) / "bundle"
        previous = bundle / "previous"
        previous.mkdir(parents=True)
        for name in _FILES:
            if (root / name).is_file():
                shutil.copyfile(root / name, previous / name)
        # This inventory is an observation before resume, including partial legacy
        # runs. Only a completed snapshot can publish an authenticated lifecycle.
        prior_stamps = []
        for index, (stage, path) in enumerate(completion_paths):
            archived = previous / f"{index}.stamp"
            if path.is_file():
                shutil.copyfile(path, archived)
            prior_stamps.append([stage, str(path.resolve()), _digest(path)])
        write_run_context_atomic(
            context,
            code_identity=code_identity,
            parent_lifecycle_roots=parent_lifecycle_roots,
            cli_overrides=cli_overrides,
            worker_counts=worker_counts,
            game_profile_sha256=game_profile_sha256,
            destination=bundle / "run_context.json",
        )
        write_active_config(context.config, bundle, recorded_path=context.active_config_path)
        receipt = {
            "version": 1,
            "run_id": run_id,
            "previous": {name: _digest(previous / name) for name in _FILES},
            "current": {name: sha256_file(bundle / name) for name in _FILES},
            "previous_completion_stamps": prior_stamps,
        }
        (bundle / "invocation.json").write_bytes(canonical_json_bytes(receipt))
        bundle.rename(history / "publishing")
    _finish_publication(root)
    record = final / "invocation.json"
    append_manifest_event(
        manifest_path,
        {
            "event": "invocation_published",
            "invocation_record": str(record.resolve()),
            "invocation_sha256": sha256_file(record),
            "run_lineage_sha256": context.config._run_lineage_sha256,
        },
        run_id=run_id,
        config_sha=context.config.config_sha,
    )


def retain_lifecycle(
    snapshot: AuthenticatedGraphSnapshot, run_id: str, completion_paths: Sequence[tuple[str, Path]]
) -> None:
    """Seal the ordered lifecycle preimage before the pair can reference it."""
    invocation = invocation_dir(snapshot.run_context_path.parent, run_id)
    with TemporaryDirectory(dir=invocation) as scratch:
        folder = Path(scratch) / "lifecycle"
        folder.mkdir()
        stamps = []
        for index, (stage, path) in enumerate(completion_paths):
            archived = folder / f"{index}.stamp"
            shutil.copyfile(path, archived)
            stamps.append([stage, str(path.resolve()), sha256_file(archived)])
        inventory = {
            "version": 1,
            "run_context_file_sha256": snapshot.run_context_sha256,
            "run_context_identity_sha256": snapshot.run_context_identity_sha256,
            "run_lineage_sha256": snapshot.run_lineage_sha256,
            "lifecycle_sha256": snapshot.lifecycle_sha256,
            "completion_stamps": stamps,
        }
        (folder / "inventory.json").write_bytes(canonical_json_bytes(inventory))
        _verify_evidence(snapshot, invocation, folder)
        folder.rename(invocation / "lifecycle")


def _verify_evidence(snapshot: AuthenticatedGraphSnapshot, invocation: Path, folder: Path) -> None:
    receipt = json.loads((invocation / "invocation.json").read_bytes())
    expected = receipt["current"]
    if (
        receipt["version"] != 1
        or receipt["run_id"] != invocation.name
        or expected["run_context.json"] != snapshot.run_context_sha256
        or expected["active_config.yaml"] != snapshot.active_config_sha256
        or any(sha256_file(invocation / name) != expected[name] for name in _FILES)
    ):
        raise ValueError(
            f"retained invocation does not match the authenticated context: {invocation}"
        )
    inventory = json.loads((folder / "inventory.json").read_bytes())
    members = {item.stage_key: (str(item.path), item.sha256) for item in snapshot.completions}
    ordered = []
    if (
        inventory["version"] != 1
        or inventory["run_context_file_sha256"] != snapshot.run_context_sha256
        or inventory["run_context_identity_sha256"] != snapshot.run_context_identity_sha256
        or inventory["run_lineage_sha256"] != snapshot.run_lineage_sha256
        or inventory["lifecycle_sha256"] != snapshot.lifecycle_sha256
    ):
        raise ValueError(f"retained lifecycle does not bind the authenticated context: {folder}")
    for index, (stage, path, digest) in enumerate(inventory["completion_stamps"]):
        if (
            members.pop(stage, None) != (path, digest)
            or sha256_file(folder / f"{index}.stamp") != digest
        ):
            raise ValueError(f"retained lifecycle stamp differs for {stage}: {folder}")
        ordered.append(digest)
    if (
        members
        or freshness_sha256(
            {"run_lineage_sha256": snapshot.run_lineage_sha256, "completion_stamps": ordered}
        )
        != snapshot.lifecycle_sha256
    ):
        raise ValueError(f"retained lifecycle membership or order differs: {folder}")


def audit_invocation(snapshot: AuthenticatedGraphSnapshot, run_id: str) -> dict[str, str]:
    invocation = invocation_dir(snapshot.run_context_path.parent, run_id)
    folder = invocation / "lifecycle"
    _verify_evidence(snapshot, invocation, folder)
    return {
        "invocation_record": str(invocation / "invocation.json"),
        "invocation_sha256": sha256_file(invocation / "invocation.json"),
        "retained_run_context": str(invocation / "run_context.json"),
        "lifecycle_inventory": str(folder / "inventory.json"),
        "lifecycle_sha256": snapshot.lifecycle_sha256,
    }
