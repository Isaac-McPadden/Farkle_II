"""Scoped scientific dependency projection for production and comparison.

Call on an already stable source tree. Inputs are evidence from the same producer invocation:
each role covers its complete coordinate-labeled inventory, including its upstream science.
Native evidence describes the effective process (including workers), not merely a lock file.
Capturing/verifying those facts and admitting reuse are separate integration responsibilities.
"""

from __future__ import annotations

import json
import re
import sys
import sysconfig
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from farkle.config import AppConfig
from farkle.protocols import implementation_contract
from farkle.utils.authenticated_contract import StageConfigIdentity, identity_sha256
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION, TOURNAMENT_METHOD_VERSION
from farkle.utils.scientific_dependencies import (
    CLASSIFIED_PACKAGE_FILES,
    LOCAL_DESIGNS,
    MANAGED_NATIVE_STAGES,
    OPERATIONAL_FIELDS,
    PREFLIGHT_LOCK,
    PREFLIGHT_RECIPE,
    RELEASE_FIELDS,
    RESOURCE_FIELDS,
    RUNTIME_FIELDS,
    SCIENTIFIC_FIELDS,
    UNRESOLVED_FIELDS,
    Producer,
    classified_operation,
    dependencies_for,
    stage_producer,
)

SCIENCE_VERSION = 1
_SHA256 = re.compile("[0-9a-f]{64}")


@dataclass(frozen=True)
class ScientificInput:
    """Semantic content/order plus upstream producer science for derived inputs.

    A local design can omit upstream science when the scoped code owns its complete
    implementation. External strategy implementations need classified source evidence.
    """

    semantic_sha256: str
    producer_science: ScientificFingerprint | None = None


@dataclass(frozen=True)
class NumericalEnvironment:
    """Verified build identities for the declared distributions and their runtime closure.

    The runtime digest must cover the interpreter/build, ISA/dispatch, effective
    floating-point/thread settings, optional backends and transitive native linkage
    (including BLAS/OpenMP and LLVM where used). Each distribution digest covers
    version and installed implementation/build content. Leave evidence absent if
    only version strings or unverified wheel metadata are available.
    """

    runtime_sha256: str
    distribution_sha256: Mapping[str, str]


@dataclass(frozen=True)
class ScientificFingerprint:
    version: int
    sha256: str | None
    unresolved: str | None = None

    @property
    def classified(self) -> bool:
        return (
            type(self.version) is int
            and self.version == SCIENCE_VERSION
            and self.unresolved is None
            and isinstance(self.sha256, str)
            and _SHA256.fullmatch(self.sha256) is not None
        )


class Compatibility(Enum):
    COMPATIBLE = "compatible"
    CHANGED = "changed"
    UNCLASSIFIED = "unclassified"


def _digest(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"Missing or invalid SHA-256 for {label}")
    return value


def _require_keys(label: str, actual: set[str], expected: frozenset[str]) -> None:
    if actual != expected:
        raise ValueError(
            f"{label}: missing={sorted(expected - actual)}, unclassified={sorted(actual - expected)}"
        )


def _raise_os_error(error: OSError) -> None:
    raise error


def _source_projection(repo_root: Path, required: frozenset[str]) -> dict[str, str]:
    # Enumerate untracked files too. Unknown modules, extension modules, or resources
    # in the import root cannot silently enter a later simulation/worker.
    from farkle.utils.scientific_runtime import _OBSERVATION, _file_digest

    observation = _OBSERVATION.get()
    files = observation.files if observation else None
    if files is not None and (repo_root, required) in files.sources:
        files.check()
        return files.sources[repo_root, required]
    for directory, dirs, files_here in (repo_root / "src").walk(on_error=_raise_os_error):
        if files is not None:
            files.watch_names(directory)
            # The watch must precede the enumeration it protects.
            if set(dirs + files_here) != {p.name for p in directory.iterdir()}:
                raise ValueError(f"Scientific import directory changed during startup: {directory}")
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")
        for name in files_here:
            relative = (directory / name).relative_to(repo_root).as_posix()
            if relative not in CLASSIFIED_PACKAGE_FILES:
                raise ValueError(f"Unclassified source: {relative}")
    paths = set(required)
    if PREFLIGHT_RECIPE in required:
        # Preflight executes these nodes via pytest, outside the import root.
        # The supplied suite identity additionally binds fixture/plugin dependencies.
        if files is not None:
            files.digest(repo_root / PREFLIGHT_LOCK)
            files.digest(repo_root / PREFLIGHT_RECIPE)
        locks = json.loads((repo_root / PREFLIGHT_LOCK).read_bytes())["sha256"]
        if set(locks) - required:
            raise ValueError("Unclassified dependency in preflight capability lock")
        checks = json.loads((repo_root / PREFLIGHT_RECIPE).read_bytes())["checks"]
        for check in checks:
            for node in check["existing_test_nodes"]:
                test = Path(node.split("::")[0])
                if test.parts[:1] != ("tests",) or ".." in test.parts or test.suffix != ".py":
                    raise ValueError(f"Unclassified preflight node: {node}")
                paths.add(test.as_posix())
        # The RNG contract node inspects Python files across all three trees.
        # src is classified above. Pytest also imports helpers and conftests here.
        for root in ("tests", "scripts"):
            for folder, dirs, names in (repo_root / root).walk(on_error=_raise_os_error):
                if files is not None:
                    files.watch_names(folder)
                    if set(dirs + names) != {p.name for p in folder.iterdir()}:
                        raise ValueError(f"Preflight directory changed during startup: {folder}")
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                paths.update(
                    (folder / name).relative_to(repo_root).as_posix()
                    for name in names
                    if name.endswith(".py")
                )
    implementation = {}
    for path in sorted(paths):
        implementation[path] = _file_digest(repo_root / path)
    if files is not None:
        files.check()
        files.sources[repo_root, required] = implementation
    return implementation


def _public_fields(section: Any, prefix: str = "") -> dict[str, Any]:
    result = {}
    for field in fields(section):
        if not field.init and not prefix and field.name in RUNTIME_FIELDS:
            continue
        name = prefix + field.name
        value = getattr(section, field.name)
        if is_dataclass(value) and not isinstance(value, type):
            result.update(_public_fields(value, name + "."))
        else:
            result[name] = value
    return result


def _config_projection(
    cfg: AppConfig, producer: Producer, scope: tuple[str, ...]
) -> dict[str, Any]:
    public = _public_fields(cfg)
    _require_keys(
        "public configuration",
        set(public),
        SCIENTIFIC_FIELDS
        | OPERATIONAL_FIELDS
        | RESOURCE_FIELDS
        | RELEASE_FIELDS
        | UNRESOLVED_FIELDS,
    )
    if cfg.sim.per_n:
        raise ValueError("Unclassified effective overrides in sim.per_n")
    unknown_outputs = set(cfg.analysis.outputs) - {"curated_rows_name", "manifest_name"}
    if unknown_outputs:
        raise ValueError(f"Unclassified analysis.outputs keys: {sorted(unknown_outputs)}")

    if "resources.stage_batch_bytes.game_stats" in scope:
        # This bound sets the floating-point accumulator's input batch size.
        budgets = cfg.resources.stage_batch_bytes
        public["resources.stage_batch_bytes.game_stats"] = budgets.get(
            "game_stats", budgets["analysis"]
        )
    selection = {key: public[key] for key in scope}
    # The standalone analysis entry points also accept an explicit ordered root
    # list. Retain all supplied coordinates even in their default root layout.
    roots = list(cfg.sim.seed_list or [cfg.sim.seed])
    if producer.scope == "pair":
        roots = cfg.sim.resolve_seed_list(2)
    if len(set(roots)) != len(roots) or roots[0] != cfg.sim.seed:
        raise ValueError("Inconsistent configured root coordinates")
    selection["sim.seed_list"] = roots
    for key in ("sim.row_dir", "sim.metric_chunk_dir"):
        if key in selection:
            # Enabling row/metric collection selects a product. Its location does not.
            selection[key] = selection[key] is not None
    for key in ("k_aggregation.k_weights", "screening.practical_delta_by_k"):
        if selection.get(key) is not None:
            # The config schema owns integer k keys; canonical JSON uses strings.
            weights = selection[key]
            if any(type(k) is not int for k in weights):
                raise ValueError(f"Invalid player-count keys in {key}")
            selection[key] = {str(k): v for k, v in weights.items()}
    return selection


def producer_config_identity(cfg: AppConfig, stage: str) -> StageConfigIdentity:
    """Persist the normalized allowlist under the existing typed config contract."""
    producer = stage_producer(cfg, stage)
    fields = dependencies_for(producer).config_scope
    selected = _config_projection(cfg, producer, fields)
    paths = tuple(sorted(selected))
    payload = {"stage_key": stage, "field_paths": paths, "selected_config": selected}
    return StageConfigIdentity(stage, paths, selected, identity_sha256(payload))


def require_input_roles(cfg: AppConfig, stage: str, roles: set[str]) -> None:
    """Require the complete scientific inventory for a stage-level reduction."""
    _require_keys(
        "direct scientific inputs",
        roles,
        dependencies_for(stage_producer(cfg, stage)).required_inputs(cfg),
    )


def _producer_components(
    producer: Producer, cfg: AppConfig, repo_root: Path, environment: NumericalEnvironment
) -> dict[str, Any]:
    deps = dependencies_for(producer)
    _digest(environment.runtime_sha256, "effective numerical runtime")
    builds = environment.distribution_sha256
    _require_keys("numerical builds", set(builds), deps.distributions)
    return {
        "scientific_dependency_version": SCIENCE_VERSION,
        "producer": producer,
        "implementation": _source_projection(repo_root, deps.source_files),
        "config": _config_projection(cfg, producer, deps.config_scope),
        "protocol": implementation_contract(),
        "outcome_schema": OUTCOME_SCHEMA_VERSION,
        "tournament_method": TOURNAMENT_METHOD_VERSION,
        "game_profile": (
            None
            if cfg._game_profile_sha256 is None
            else _digest(cfg._game_profile_sha256, "game profile")
        ),
        "numerical_environment": {
            "python": sys.version,
            "platform": sysconfig.get_platform(),
            "runtime": environment.runtime_sha256,
            "packages": {
                name: {"version": version(name), "build": _digest(digest, name)}
                for name, digest in builds.items()
            },
        },
    }


def live_components(
    cfg: AppConfig, producer: Producer, *, observe_worker: bool = False
) -> dict[str, Any]:
    from farkle.utils.scientific_runtime import numerical_environment

    stage = producer.stage
    if stage == "trueskill" and producer.scope == "root":
        for k in cfg.sim.n_players_list:
            keeper = cfg.n_dir(k) / f"keepers_{k}.npy"
            if keeper.exists():
                raise ValueError(f"Unclassified external TrueSkill keeper selection at {keeper}")
    deps = dependencies_for(producer)
    native_threads = (
        cfg.resources.native_threads_per_worker
        if stage in MANAGED_NATIVE_STAGES and cfg.resource_limiter_mode.value == "enabled"
        else None
    )
    runtime, builds = numerical_environment(
        deps.distributions, native_threads=native_threads, observe_worker=observe_worker
    )
    return _producer_components(
        producer, cfg, Path(__file__).resolve().parents[3], NumericalEnvironment(runtime, builds)
    )


def stage_dependency_sha256(
    cfg: AppConfig, stage: str, *, partition: str | None = None, observe_worker: bool = False
) -> str:
    """Scoped producer components shared by same-revision publications and durable units."""
    from farkle.utils.scientific_invocation import checked_components

    owner = stage_producer(cfg, stage)
    components = checked_components(cfg, owner, observe_worker=observe_worker)
    if partition is not None:
        producer = Producer(owner.scope, stage, partition)
        dependencies_for(producer)  # Validate the family before deriving its distinct digest.
        components = {**components, "producer": producer}
    return identity_sha256(components)


_ACTIVE_PRODUCER: ContextVar[tuple[AppConfig, str] | None] = ContextVar(
    "active_scientific_producer", default=None
)


@contextmanager
def producer_scope(cfg: AppConfig, stage: str) -> Iterator[None]:
    """Pass the owning stage to scientific work launched through an executor."""
    from threadpoolctl import threadpool_limits

    from farkle.utils.scientific_invocation import checked_components, current

    pin = current()
    managed = stage in MANAGED_NATIVE_STAGES and cfg.resource_limiter_mode.value == "enabled"
    token = _ACTIVE_PRODUCER.set((cfg, stage))
    try:
        with (
            threadpool_limits(limits=cfg.resources.native_threads_per_worker)
            if pin is not None and managed
            else nullcontext()
        ):
            if pin is not None:
                checked_components(cfg, stage_producer(cfg, stage))
            yield
            if pin is not None:
                checked_components(cfg, stage_producer(cfg, stage))
    finally:
        _ACTIVE_PRODUCER.reset(token)


@dataclass(frozen=True)
class WorkerDependencies:
    config: AppConfig
    stage: str
    expected: str
    placement: str = ""
    components: dict[str, Any] | None = None
    launch_digests: dict[str, str] | None = None

    def attach(self) -> None:
        from farkle.utils.scientific_invocation import attach_worker

        if self.placement:
            attach_worker(self)

    def verify(self) -> None:
        self.attach()
        if not compatible_dependency_digests(
            stage_dependency_sha256(self.config, self.stage, observe_worker=True), self.expected
        ):
            raise ValueError(f"Worker scientific dependencies differ for {self.stage}")


def worker_dependencies() -> WorkerDependencies | None:
    from farkle.utils.scientific_invocation import current, describe_components, process_placement

    pin = current()
    producer = _ACTIVE_PRODUCER.get() or (pin.default if pin else None)
    if producer is None:
        return None
    cfg, stage = producer
    if stage == "h2h_execute" and pin is None:
        # This executor independently verifies the digest in the authorized plan.
        return None
    return WorkerDependencies(
        cfg,
        stage,
        stage_dependency_sha256(cfg, stage),
        process_placement(cfg, stage),
        describe_components(pin.components(cfg, stage_producer(cfg, stage))) if pin else None,
        dict(pin.launch_digests) if pin else None,
    )


def project_scientific_identity(
    producer: Producer,
    cfg: AppConfig,
    *,
    repo_root: Path,
    environment: NumericalEnvironment | None,
    inputs: Mapping[str, ScientificInput],
    coordinates: tuple[int | str, ...] = (),
) -> ScientificFingerprint:
    """Project a canonical stage or durable unit. Incomplete evidence yields no digest.

    Partition coordinates follow the catalog. For multi-step operations the ordinal
    indexes the authenticated operation manifest. unit_design binds its exact semantic
    key and resolved range/tile/route plan. unit_sources binds ordered predecessors.
    """
    try:
        deps = dependencies_for(producer)
        if len(coordinates) != len(deps.coordinate_fields):
            raise ValueError(f"Expected partition coordinates {deps.coordinate_fields}")
        unit = dict(zip(deps.coordinate_fields, coordinates, strict=True))
        for name, value in unit.items():
            if name == "operation":
                if not classified_operation(producer, value):
                    raise ValueError(f"Unclassified partition operation {value!r}")
            elif type(value) is not int or value < 0:
                raise ValueError(f"Invalid partition coordinate {name}")
        if "k" in unit and unit["k"] not in cfg.sim.n_players_list:
            raise ValueError("Partition k outside configured support")
        if "root_seed" in unit and unit["root_seed"] not in (cfg.sim.seed_list or [cfg.sim.seed]):
            raise ValueError("Partition root outside configured support")
        if "order" in unit and unit["order"] not in (0, 1):
            raise ValueError("Invalid H2H order")
        if "stop" in unit and int(unit["stop"]) <= int(unit["start"]):
            raise ValueError("Empty partition range")
        if "partitions" in unit and int(unit["partition"]) >= int(unit["partitions"]):
            raise ValueError("Partition index outside declared support")
        _require_keys("direct scientific inputs", set(inputs), deps.required_inputs(cfg))
        for role, binding in inputs.items():
            if not isinstance(binding, ScientificInput):
                raise ValueError(f"Missing scientific input binding for {role}")
            origin = binding.producer_science
            if (origin is None and role not in LOCAL_DESIGNS) or (
                origin is not None and not origin.classified
            ):
                raise ValueError(f"Unclassified upstream producer for {role}")
            _digest(binding.semantic_sha256, role)
        if environment is None:
            raise ValueError("Missing verified numerical environment")
        science = _producer_components(producer, cfg, repo_root, environment)
        science.update(coordinates=unit, inputs=dict(inputs))
        return ScientificFingerprint(SCIENCE_VERSION, identity_sha256(science))
    except (OSError, KeyError, TypeError, ValueError, AttributeError, PackageNotFoundError) as exc:
        return ScientificFingerprint(SCIENCE_VERSION, None, str(exc))


def scientific_compatibility(
    previous: ScientificFingerprint, current: ScientificFingerprint
) -> Compatibility:
    """Compare proposed science only. Missing, malformed or unknown versions fail closed."""
    if not previous.classified or not current.classified:
        return Compatibility.UNCLASSIFIED
    return Compatibility.COMPATIBLE if previous.sha256 == current.sha256 else Compatibility.CHANGED


def compatible_dependency_digests(recorded: str | None, current: str | None) -> bool:
    """Compare two digests decoded from the version-one dependency field."""
    return (
        scientific_compatibility(
            ScientificFingerprint(SCIENCE_VERSION, recorded),
            ScientificFingerprint(SCIENCE_VERSION, current),
        )
        is Compatibility.COMPATIBLE
    )
