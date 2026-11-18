"""Side-by-side, exact-byte recovery of the November 106/107 frozen H2H inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from farkle.analysis.h2h_authorization import validate_authorization, validate_authorization_sources
from farkle.analysis.h2h_schedule import _block_path, _read_authenticated_block
from farkle.analysis.stage_registry import resolve_root_pair_stage_layout
from farkle.config import AppConfig, load_app_config
from farkle.orchestration.run_contexts import RunContextConfig, load_run_context
from farkle.utils.artifact_contract import sha256_file, sidecar_path
from farkle.utils.authenticated_contract import (
    ArtifactIdentity,
    AuthenticatedSidecar,
    ManifestEntry,
    ManifestRootIdentity,
    SourceArtifactIdentity,
    compute_manifest_root,
    load_immutable_manifest_sidecar,
    make_authenticated_sidecar,
    make_stage_identity,
    validate_authenticated_artifact_unbound,
)
from farkle.utils.frozen_h2h import frozen_family_inputs
from farkle.utils.writer import atomic_path

SEEDS = (106, 107)
FAMILY = "ba270075a721293ac42903c9248ad2e4114e7660ed26d892bf08244946d56406"
SCHEDULE = "42ace906c2e4c898401bce604f62392cf626e2694648375158f5d6d75beb1aea"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(actual: str, expected: str, label: object) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected}, observed {actual}")


def _historical_sidecar(
    template: AuthenticatedSidecar,
    artifact: ArtifactIdentity,
    sources: tuple[SourceArtifactIdentity, ...],
    lineage: str,
) -> AuthenticatedSidecar:
    stage = template.stage_identity
    if set(stage.immutable_design_identities) != {"run_lineage_sha256"} or template.manifest_roots:
        raise ValueError(f"unexpected historical reconstruction inputs for {artifact.location}")
    historical_stage = make_stage_identity(
        stage_key=stage.stage_key,
        stage_cache_key_version=stage.stage_cache_key_version,
        stage_config=stage.stage_config,
        versions=stage.versions,
        code=stage.code,
        method_contract=template.method_contract,
        upstream_identity_sha256=[source.sha256 for source in sources],
        immutable_design_identities={"run_lineage_sha256": lineage},
    )
    return make_authenticated_sidecar(
        artifact=artifact,
        stage_identity=historical_stage,
        method_contract=template.method_contract,
        sources=sources,
    )


def reconstruct_family(cfg: AppConfig, power: AuthenticatedSidecar) -> dict[Path, bytes]:
    """Return candidates only after both historical family edges bind all reconstructed bytes."""
    lineage = power.stage_identity.immutable_design_identities["run_lineage_sha256"]
    family_hash = power.stage_identity.immutable_design_identities["family_hash"]
    manifest_path = cfg.h2h_candidate_family_manifest_path()
    member_path = cfg.h2h_candidate_family_path()
    pins = {source.artifact.location.path(cfg): source for source in power.source_artifacts}
    if set(pins) != {manifest_path, member_path}:
        raise ValueError("unexpected frozen family inventory")
    current = {
        path: validate_authenticated_artifact_unbound(path, validate_provenance=False)
        for path in pins
    }
    inputs = current[manifest_path].source_artifacts
    expected_inputs = {
        cfg.trueskill_candidate_contribution_path(),
        cfg.root_combined_performance_across_k_path(),
    }
    if (
        current[member_path].source_artifacts != inputs
        or len(inputs) != 2
        or {s.artifact.location.path(cfg) for s in inputs} != expected_inputs
    ):
        raise ValueError("candidate publications disagree on source inventory")
    staged: dict[Path, bytes] = {}
    edges = []
    for source in inputs:
        path = source.artifact.location.path(cfg)
        template = validate_authenticated_artifact_unbound(path, validate_provenance=False)
        _require(_digest(sidecar_path(path).read_bytes()), source.sidecar_sha256, path)
        if template.artifact != source.artifact:
            raise ValueError(f"changed input payload at {path}")
        restored = _historical_sidecar(
            template, template.artifact, template.source_artifacts, lineage
        )
        raw = restored.canonical_bytes()
        staged[sidecar_path(path)] = raw
        edges.append(
            replace(
                source,
                sidecar_sha256=_digest(raw),
                sidecar_contract_sha256=restored.sidecar_contract_sha256,
            )
        )
        if path == cfg.trueskill_candidate_contribution_path():
            historical_rating = restored

    # Keep all historical strings, including original producer paths and the
    # Windows text writer's CRLF convention. Do not serialize recovery paths.
    manifest = json.loads(manifest_path.read_bytes())
    rating = manifest["source_identity"]["trueskill"]
    rating["sidecar_sha256"] = _digest(historical_rating.canonical_bytes())
    rating["stage_identity_sha256"] = historical_rating.stage_identity.sha256
    manifest["family_hash"] = family_hash
    raw_json = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    staged[manifest_path] = raw_json.replace("\n", "\r\n").encode()
    table = pq.read_table(member_path)
    index = table.schema.get_field_index("family_hash")
    field = table.schema.field(index)
    table = table.set_column(index, field, pa.array([family_hash] * len(table), type=field.type))
    stream = pa.BufferOutputStream()
    pq.write_table(table, stream, compression=cfg.parquet_codec)
    staged[member_path] = stream.getvalue().to_pybytes()
    for path, pin in pins.items():
        _require(_digest(staged[path]), pin.artifact.content_sha256, path)
        sidecar = _historical_sidecar(current[path], pin.artifact, tuple(edges), lineage)
        raw = sidecar.canonical_bytes()
        _require(_digest(raw), pin.sidecar_sha256, sidecar_path(path))
        _require(sidecar.sidecar_contract_sha256, pin.sidecar_contract_sha256, path)
        staged[sidecar_path(path)] = raw
    return staged


def _pair_config(public: AppConfig, root: Path) -> RunContextConfig:
    return RunContextConfig.from_base(
        public,
        analysis_root=root / "seed_pair_analysis",
        stage_layout=resolve_root_pair_stage_layout(public),
    )


class Recovery:
    """Private, create-only destination. Every copied byte has a named historical witness."""

    def __init__(self, source: Path, dest: Path) -> None:
        self.source = source.resolve(strict=True)
        self.dest = dest.resolve()
        if self.source.is_relative_to(self.dest) or self.dest.is_relative_to(self.source):
            raise ValueError("recovery requires disjoint source and destination trees")
        self.dest.mkdir(parents=True, exist_ok=False)
        self.evidence = self.dest / "recovery_evidence"
        self.evidence.mkdir()
        self.inventory = (self.evidence / "verified_files.jsonl").open("x", encoding="utf-8")
        self.files: dict[Path, str] = {}
        self.reconstructions: dict[Path, bytes] = {}
        self.checked: dict[Path, tuple[ArtifactIdentity, str, str]] = {}
        self.configs: dict[int, AppConfig] = {}
        self.lineages: dict[int, str] = {}
        self.shards = 0

    def copy(self, src: Path, expected: str, witness: Path) -> Path:
        relative = src.relative_to(self.source)
        target = self.dest / relative
        if src in self.files:
            _require(self.files[src], expected, src)
            return target
        if src.resolve(strict=True) != self.source / relative or target.exists():
            raise ValueError(f"expected ordinary source and new destination for {src}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(str(target)) as temp:
            staged = Path(temp)
            with staged.open("wb") as out:
                if src in self.reconstructions:
                    out.write(self.reconstructions[src])
                else:
                    with src.open("rb") as inp:
                        shutil.copyfileobj(inp, out, length=1024 * 1024)
            _require(sha256_file(staged), expected, f"{relative} pinned by {witness}")
        self.files[src] = expected
        self.inventory.write(
            json.dumps(
                {
                    "path": relative.as_posix(),
                    "sha256": expected,
                    "witness": witness.relative_to(self.source).as_posix(),
                    "mode": "reconstructed" if src in self.reconstructions else "copied",
                }
            )
            + "\n"
        )
        return target

    def configure(self, base: Path, seed: int) -> AppConfig:
        public_path = base / "active_config.yaml"
        stamp = base / "active_config.done.json"
        _require(
            sha256_file(public_path),
            json.loads(stamp.read_bytes())["active_config_sha256"],
            public_path,
        )
        context = load_run_context(base / "run_context.json", active_config_path=public_path)
        cfg = load_app_config(public_path)
        target_root = self.dest if seed == 0 else self.dest / base.name
        # Configuring physical recovery paths is an operator action. Keep this
        # YAML separate from the historical producer's active configuration.
        data = yaml.safe_load(public_path.read_bytes())
        data["io"]["results_dir_prefix"] = str(self.dest / f"results_seed_{seed or 106}")
        data["io"]["temp_dir"] = str(self.dest / "scratch")
        target_root.mkdir(parents=True, exist_ok=True)
        (target_root / "recovery_config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        cfg.io.results_dir_prefix = self.source / f"results_seed_{seed or 106}"
        if not seed:
            cfg = _pair_config(cfg, self.source)
        if cfg.stage_layout.to_resolved_layout() != context["resolved_stage_layout"]:
            raise ValueError(f"historical stage layout differs at {base}")
        self.configs[seed] = cfg
        self.lineages[seed] = context["run_lineage_sha256"]
        return cfg

    def source_seed(self, source: SourceArtifactIdentity, owner: int) -> int:
        if owner or source.artifact.location.scope != "by_k":
            return owner
        match = re.search(r"\.roots_(106|107)\.", source.logical_role)
        if match is None:
            raise ValueError(f"ambiguous seed for {source.logical_role}")
        return int(match[1])

    def artifact(self, ref: SourceArtifactIdentity, owner: int, witness: Path) -> None:
        location = ref.artifact.location
        cfg = self.configs[owner]
        src = location.path(cfg)
        key = (ref.artifact, ref.sidecar_sha256, ref.sidecar_contract_sha256)
        target = self.copy(src, ref.artifact.content_sha256, witness)
        self.copy(sidecar_path(src), ref.sidecar_sha256, witness)
        if src in self.checked:
            if self.checked[src] != key:
                raise ValueError(f"conflicting historical references at {src}")
            return
        metadata = validate_authenticated_artifact_unbound(
            target, expected_location=location, validate_provenance=False
        )
        if (
            metadata.artifact != ref.artifact
            or metadata.sidecar_contract_sha256 != ref.sidecar_contract_sha256
        ):
            raise ValueError(f"historical artifact or contract differs at {src}")
        self.check_producer(metadata, owner, src)
        for direct in metadata.source_artifacts:
            self.artifact(direct, self.source_seed(direct, owner), sidecar_path(src))
        for manifest in metadata.manifest_roots:
            self.manifest(manifest, owner, sidecar_path(src))
        for name, digest in metadata.stage_identity.immutable_design_identities.items():
            if name.startswith("control:"):
                if name != "control:simulation.done.json:0000" or not location.player_count:
                    raise ValueError(f"unclassified historical control {name} at {src}")
                self.copy(
                    cfg.results_root / f"{location.player_count}_players" / "simulation.done.json",
                    digest,
                    sidecar_path(src),
                )
        self.checked[src] = key

    def check_producer(self, metadata: AuthenticatedSidecar, owner: int, path: Path) -> None:
        stage = metadata.stage_identity
        if (
            stage.code != self.code
            or stage.immutable_design_identities.get("run_lineage_sha256") != self.lineages[owner]
        ):
            raise ValueError(f"historical code or lineage mismatch at {path}")

    def manifest(self, ref: ManifestRootIdentity, owner: int, witness: Path) -> None:
        cfg = self.configs[owner]
        src = ref.location.path(cfg)
        target = self.copy(src, ref.manifest_sha256, witness)
        self.copy(sidecar_path(src), ref.sidecar_sha256, witness)
        metadata = load_immutable_manifest_sidecar(target)
        if (
            metadata.location != ref.location
            or metadata.summary != ref.summary
            or metadata.sidecar_contract_sha256 != ref.sidecar_contract_sha256
            or metadata.stage_identity.code != self.code
            or metadata.stage_identity.immutable_design_identities.get("run_lineage_sha256")
            != self.lineages[owner]
        ):
            raise ValueError(f"immutable manifest differs at {src}")
        entries = []
        with target.open(encoding="utf-8") as records:
            for line in records:
                record = json.loads(line)
                shard = (src.parent / record["path"]).resolve()
                if shard.parent != src.parent or record["root_seed"] != owner:
                    raise ValueError(f"shard outside its historical manifest scope: {shard}")
                out = self.copy(shard, record["data_sha256"], src)
                self.copy(sidecar_path(shard), record["sidecar_sha256"], src)
                sc = validate_authenticated_artifact_unbound(out, validate_provenance=False)
                self.check_producer(sc, owner, shard)
                schema = sc.artifact.arrow_schema
                if (
                    schema is None
                    or schema.fingerprint_sha256 != record["schema_fingerprint_sha256"]
                    or sc.artifact.byte_length != record["byte_length"]
                    or sc.artifact.location.path(cfg) != shard
                    or sc.manifest_roots
                ):
                    raise ValueError(f"shard schema, length or location mismatch at {shard}")
                for source in sc.source_artifacts:
                    self.artifact(source, owner, sidecar_path(shard))
                entries.append(
                    ManifestEntry(
                        coordinate=(int(record["shuffle_index"]),),
                        canonical_relative_path=shard.relative_to(cfg.results_root).as_posix(),
                        data_sha256=record["data_sha256"],
                        sidecar_sha256=record["sidecar_sha256"],
                        schema_fingerprint_sha256=schema.fingerprint_sha256,
                    )
                )
        if compute_manifest_root(entries) != ref.summary:
            raise ValueError(f"shard coverage differs at {src}")
        self.shards += len(entries)
        print(
            f"Authenticated {len(entries)} shards from {src.relative_to(self.source)}", flush=True
        )

    def recover(self) -> dict[str, Any]:
        for seed in SEEDS:
            self.configure(self.source / f"results_seed_{seed}", seed)
        cfg = self.configure(self.source, 0)
        # A retained downstream consumer independently pins the frozen power
        # plan, schedule, execution state and aggregate. Its own stage stamp
        # pins the consumer bytes. These witnesses stay in the original tree.
        consumer = cfg.h2h_combined_order_counts_path()
        stamp = cfg.stage_dir("h2h_inference") / "h2h_inference.done.json"
        completions = json.loads(stamp.read_bytes())
        if completions["state"] != "complete_valid":
            raise ValueError(f"incomplete downstream witness {stamp}")
        pins = [
            o
            for o in completions["outputs"]
            if o["artifact"]["location"]["relative_path"] == consumer.name
        ]
        if len(pins) != 1:
            raise ValueError(f"missing independent pin for {consumer}")
        _require(sha256_file(consumer), pins[0]["artifact"]["content_sha256"], consumer)
        _require(sha256_file(sidecar_path(consumer)), pins[0]["sidecar_sha256"], consumer)
        downstream = validate_authenticated_artifact_unbound(consumer, validate_provenance=False)
        refs = {s.artifact.location.path(cfg): s for s in downstream.source_artifacts}
        power_path = cfg.h2h_power_plan_path()
        schedule_path = cfg.h2h_block_manifest_path()
        state_path = cfg.h2h_execution_state_path()
        aggregate = cfg.h2h_order_counts_path()
        if set(refs) != {power_path, schedule_path, state_path, aggregate}:
            raise ValueError("unexpected downstream witness inventory")
        _require(sha256_file(sidecar_path(power_path)), refs[power_path].sidecar_sha256, power_path)
        power = validate_authenticated_artifact_unbound(power_path, validate_provenance=False)
        self.code = power.stage_identity.code
        if (
            self.code.commit != "6bf52543fcece163f5d8a8f117f95376c215f7f7"
            or self.code.state != "clean"
        ):
            raise ValueError(f"unexpected historical producer at {power_path}")
        self.lineages[0] = power.stage_identity.immutable_design_identities["run_lineage_sha256"]
        plan = json.loads(power_path.read_bytes())
        if (
            plan["family_hash"] != FAMILY
            or plan["schedule_hash"] != SCHEDULE
            or plan["root_seeds"] != list(SEEDS)
            or plan["total_block_count"] != 22684
        ):
            raise ValueError("unexpected historical design")
        self.reconstructions = reconstruct_family(cfg, power)
        for path in (power_path, schedule_path, state_path, aggregate):
            print(f"Recovering dependencies of {path.name}", flush=True)
            self.artifact(refs[path], 0, sidecar_path(consumer))
        destination = _pair_config(load_app_config(self.dest / "recovery_config.yaml"), self.dest)
        frozen_family_inputs(destination)
        auth_path = destination.h2h_cap_authorization_path()
        auth = json.loads(auth_path.read_bytes())
        validate_authorization(auth, plan, require_ready=True)
        validate_authorization_sources(
            auth, destination.h2h_power_plan_path(), destination.h2h_block_manifest_path()
        )
        state = json.loads(destination.h2h_execution_state_path().read_bytes())
        _require(sha256_file(auth_path), state["cap_authorization_sha256"], auth_path)
        records = pq.read_table(destination.h2h_block_manifest_path()).to_pylist()
        counts = pq.read_table(destination.h2h_order_counts_path()).to_pylist()
        if len(records) != 22684 or len({row["block_id"] for row in records}) != 22684:
            raise ValueError("schedule lacks 22,684 distinct blocks")
        inventory = {
            destination.h2h_cap_authorization_path(),
            destination.h2h_execution_state_path(),
            destination.h2h_order_counts_path(),
        }
        totals = dict.fromkeys(
            (
                "games_attempted",
                "games_completed",
                "games_safety_limit",
                "replacement_attempt_count",
            ),
            0,
        )
        for block, aggregate_row in zip(records, counts, strict=True):
            path = _block_path(destination, block)
            row = _read_authenticated_block(path, block)
            if (
                row is None
                or row != aggregate_row
                or row["games_completed"] != block["n_completed_required"]
                or any(row[k] != v for k, v in block.items())
                or path in inventory
            ):
                raise ValueError(f"incomplete or conflicting scheduled block at {path}")
            inventory.add(path)
            for k in totals:
                totals[k] += int(row[k])
        if (
            state["execution_state"] != "complete_valid"
            or state["completed_block_count"] != len(records)
            or totals["games_completed"] != plan["total_completed_required"]
            or any(state[k] != v for k, v in totals.items())
        ):
            raise ValueError("block totals differ from authenticated execution state or plan")
        self.check_completion(cfg, "h2h_execute", inventory)
        self.check_completion(
            cfg,
            "h2h_power",
            {destination.h2h_power_plan_path(), destination.h2h_block_manifest_path()},
        )
        return {
            "witnesses": {
                path.relative_to(self.source).as_posix(): sha256_file(path)
                for path in (
                    stamp,
                    consumer,
                    sidecar_path(consumer),
                    cfg.stage_dir("h2h_execute") / "h2h_execute.done.json",
                    cfg.stage_dir("h2h_power") / "h2h_power.done.json",
                )
            },
            "family_hash": FAMILY,
            "schedule_hash": SCHEDULE,
            "historical_commit": self.code.commit,
            "historical_pair_lineage": self.lineages[0],
            "completed_blocks": len(records),
            "games_replayed": 0,
            **totals,
            "verified_files": len(self.files),
            "raw_shards": self.shards,
            "reconstructed": {
                p.relative_to(self.source).as_posix(): _digest(b)
                for p, b in self.reconstructions.items()
            },
            "authorization_source": auth["authorization_source"],
            "authorized_total_game_cap": auth["authorized_total_game_cap"],
        }

    def check_completion(self, cfg: AppConfig, stage: str, paths: set[Path]) -> None:
        stamp = cfg.stage_dir(stage) / f"{stage}.done.json"
        content = json.loads(stamp.read_bytes())
        if content["state"] != "complete_valid" or len(content["outputs"]) != len(paths):
            raise ValueError(f"invalid frozen completion inventory at {stamp}")
        for output in content["outputs"]:
            artifact = output["artifact"]
            location = artifact["location"]
            if location["stage_key"] != stage or location["scope"] != "h2h_2p":
                raise ValueError(f"unexpected completion location in {stamp}")
            src = cfg.h2h_2p_dir(stage) / location["relative_path"]
            paths.remove(self.dest / src.relative_to(self.source))
            _require(self.files[src], artifact["content_sha256"], src)
            _require(self.files[sidecar_path(src)], output["sidecar_sha256"], src)


def source_inventory(root: Path) -> str:
    """Fingerprint original names, lengths and modification times without opening payloads."""
    digest = hashlib.sha256()
    for base, folders, files in os.walk(root):
        folders.sort()
        for name in sorted(files):
            p = Path(base) / name
            stat = p.stat()
            digest.update(
                json.dumps(
                    [p.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns]
                ).encode()
                + b"\n"
            )
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    run = Recovery(args.source, args.destination)
    report: dict[str, Any] = {
        "source": str(run.source),
        "destination": str(run.dest),
        "status": "incomplete",
        "release_qualified": False,
        "started_utc": datetime.now(UTC).isoformat(),
        "recovery_tool_sha256": sha256_file(Path(__file__)),
    }
    try:
        report["source_inventory_before"] = source_inventory(run.source)
        report.update(run.recover())
        report["source_inventory_after"] = source_inventory(run.source)
        _require(
            report["source_inventory_after"],
            report["source_inventory_before"],
            "source file inventory",
        )
        report["status"] = "phase_1b_verified"
    except BaseException as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run.inventory.close()
        report["verified_files"] = len(run.files)
        report["finished_utc"] = datetime.now(UTC).isoformat()
        (run.evidence / "recovery.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{report['status']}: {run.evidence}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
