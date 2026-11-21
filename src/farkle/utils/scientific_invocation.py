"""Launch evidence shared by scientific entry points, workers and publication checks.

Invocation-local paths authenticate process placement. They are intentionally absent
from persisted producer fingerprints and do not authorize cross-revision reuse.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Concatenate, NoReturn, ParamSpec, TypeVar
from weakref import WeakKeyDictionary

from farkle.config import AppConfig
from farkle.utils.authenticated_contract import (
    canonical_json_bytes,
    identity_sha256,
    producer_code_identity,
)
from farkle.utils.scientific_dependencies import Producer, dependencies_for, stage_producer
from farkle.utils.scientific_files import MeasuredFiles, ScientificDrift
from farkle.utils.scientific_identity import (
    _ACTIVE_PRODUCER,
    WorkerDependencies,
    _config_projection,
    _source_projection,
    live_components,
)
from farkle.utils.scientific_runtime import _OBSERVATION, numerical_observation

ROOT = Path(__file__).resolve().parents[3]
Owner = tuple[Producer, int, tuple[int, ...]]
P = ParamSpec("P")
T = TypeVar("T")
_RESIDENT_DIGESTS: WeakKeyDictionary[ModuleType, str] = WeakKeyDictionary()


@dataclass
class Invocation:
    files: MeasuredFiles
    expected: dict[Owner, dict[str, Any]] = field(default_factory=dict)
    configs: dict[Owner, AppConfig] = field(default_factory=dict)
    default: tuple[AppConfig, str] | None = None
    pid: int = field(default_factory=os.getpid)
    failure: str = ""
    verified_imports: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    worker: bool = False

    def components(self, cfg: AppConfig, owner: Producer) -> dict[str, Any]:
        key = _key(cfg, owner)
        if key not in self.expected:
            self.reject(f"Unpinned scientific producer or seed coordinates: {key}")
        return self.expected[key]

    def imports(self) -> None:
        self.files.boundary()
        if self.failure:
            raise ScientificDrift(self.failure)
        if (
            sys.pycache_prefix != str(ROOT / "src/farkle/__init__.py")
            or not sys.dont_write_bytecode
        ):
            self.reject("Scientific Python bytecode-cache policy changed during invocation")
        observation = _OBSERVATION.get()
        assert observation is not None
        for name, module in sys.modules.copy().items():
            top = name.partition(".")[0]
            if module is None or (top != "farkle" and top not in observation.origins):
                continue
            path = getattr(module, "__file__", None)
            if not path:
                if name in self.verified_imports or name in observation.origins:
                    self.reject(f"Resolved scientific import {name} lost its file origin")
                continue  # namespace package; its concrete submodules have origins
            spec = getattr(module, "__spec__", None)
            placement = (path, getattr(spec, "origin", None))
            previous = self.verified_imports.get(name)
            if previous is not None:
                if previous != placement:
                    self.reject(
                        f"Resolved scientific import {name} changed from {previous} to {placement}"
                    )
                continue
            if top == "farkle":
                package = ROOT / "src" / Path(*name.split("."))
                legal = (str(package.with_suffix(".py")), str(package / "__init__.py"))
                if path not in legal or placement[1] not in {None, path}:
                    self.reject(f"Resolved scientific import {name} at {path}, expected {package}")
            elif (
                Path(path) not in self.files.digests
                or (name == top and Path(path) != observation.origins[top])
                or (placement[1] not in {None, path})
            ):
                self.reject(f"Resolved numerical import {name} outside pinned build at {placement}")
            measured = self.files.digests.get(Path(path))
            if measured and _RESIDENT_DIGESTS.setdefault(module, measured) != measured:
                self.reject(f"Loaded module {name} differs from {path}. Start a fresh interpreter.")
            self.verified_imports[name] = placement

    def reject(self, message: str) -> NoReturn:
        self.failure = message
        raise ScientificDrift(message)


_CURRENT: ContextVar[Invocation | None] = ContextVar("scientific_invocation", default=None)


def current() -> Invocation | None:
    pin = _CURRENT.get()
    return pin if pin is not None and pin.pid == os.getpid() else None


def _key(cfg: AppConfig, producer: Producer) -> Owner:
    return producer, cfg.sim.seed, tuple(cfg.sim.seed_list or [cfg.sim.seed])


def describe_components(values: dict[str, Any]) -> dict[str, Any]:
    """Expand the runtime hash for diagnostics while retaining the producer's exact digest."""
    observation = _OBSERVATION.get()
    environment = values.get("numerical_environment")
    if observation and environment and isinstance(environment["runtime"], str):
        controls = observation.controls.get(environment["runtime"])
        if controls is not None:
            return {**values, "numerical_environment": {**environment, "runtime": controls}}
    return values


def difference(field: str, before: Any, after: Any) -> str:
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            path = f"{field}.{key}"
            if key not in before:
                return f"{path} appeared with value {after[key]!r}"
            if key not in after:
                return f"{path} disappeared (launch value {before[key]!r})"
            if before[key] != after[key]:
                return difference(path, before[key], after[key])
    return f"{field}: {before!r} -> {after!r}"


def _compare(pin: Invocation, stage: str, expected: dict[str, Any], live: dict[str, Any]) -> None:
    if expected.keys() != live.keys():
        pin.reject(
            f"Scientific dependency fields changed for {stage}: {expected.keys() ^ live.keys()}"
        )
    for name, value in live.items():
        if value != expected[name]:
            before = describe_components(expected)[name]
            after = describe_components(live)[name]
            pin.reject(f"Scientific drift in {difference(f'{stage}.{name}', before, after)}")


def checked_components(
    cfg: AppConfig, owner: Producer, *, observe_worker: bool = False
) -> dict[str, Any]:
    pin = current()
    if pin is None:
        return live_components(cfg, owner, observe_worker=observe_worker)
    pin.imports()
    expected = pin.components(cfg, owner)
    active = _ACTIVE_PRODUCER.get()
    actual_threads = pin.worker or (
        active is not None and active[0] is cfg and active[1] == owner.stage
    )
    try:
        observed = live_components(cfg, owner, observe_worker=observe_worker or actual_threads)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        pin.reject(f"Cannot establish live scientific dependencies for {owner.stage}: {exc}")
    _compare(pin, owner.stage, expected, observed)
    pin.configs[_key(cfg, owner)] = cfg
    return observed


@contextmanager
def scientific_invocation(producers: Sequence[tuple[AppConfig, str]]) -> Iterator[None]:
    if not producers:
        yield
        return
    with ExitStack() as stack:
        for cfg, _ in producers:
            if cfg._code_identity is None:
                cfg._code_identity = producer_code_identity(cfg)
                stack.callback(setattr, cfg, "_code_identity", None)
        pin = current()
        if pin is None:
            observation = _OBSERVATION.get()
            files = observation.files if observation else None
            if files is None or files.pid != os.getpid():
                files = MeasuredFiles()
                stack.callback(files.close)
                stack.enter_context(numerical_observation(files))
            else:
                # Preflight fixtures have independent designs within the same
                # guarded interpreter. The enclosing invocation owns the handles.
                files.boundary()
            # This existing regular file cannot contain a bytecode-cache tree.
            # Later dynamic imports and spawned interpreters compile measured
            # source even if the editable checkout has timestamp-valid old pycs.
            prefix = str(ROOT / "src/farkle/__init__.py")
            stack.callback(setattr, sys, "pycache_prefix", sys.pycache_prefix)
            stack.callback(setattr, sys, "dont_write_bytecode", sys.dont_write_bytecode)
            sys.pycache_prefix, sys.dont_write_bytecode = prefix, True
            for variable, value in (
                ("PYTHONPYCACHEPREFIX", prefix),
                ("PYTHONDONTWRITEBYTECODE", "1"),
            ):
                if variable in os.environ:
                    stack.callback(os.environ.__setitem__, variable, os.environ[variable])
                else:
                    stack.callback(os.environ.pop, variable, None)
                os.environ[variable] = value
            pin = Invocation(files, default=producers[0] if len(producers) == 1 else None)
            for cfg, stage in producers:
                group = dependencies_for(stage_producer(cfg, stage))
                _source_projection(ROOT, group.source_files)
            for cfg, stage in producers:
                owner = stage_producer(cfg, stage)
                values = live_components(cfg, owner)
                key = _key(cfg, owner)
                pin.configs[key] = cfg
                if key in pin.expected:
                    _compare(pin, stage, pin.expected[key], values)
                else:
                    pin.expected[key] = deepcopy(values)
            pin.imports()
            token = _CURRENT.set(pin)
            stack.callback(_CURRENT.reset, token)
        else:
            for cfg, stage in producers:
                checked_components(cfg, stage_producer(cfg, stage))
        yield
        for cfg, stage in producers:
            checked_components(cfg, stage_producer(cfg, stage))


_PUBLISHING: ContextVar[bool] = ContextVar("scientific_publication", default=False)


def publication_boundary() -> None:
    pin = current()
    if pin is not None and not _PUBLISHING.get():
        active = _ACTIVE_PRODUCER.get() or pin.default
        if active:
            cfg, stage = active
            checked_components(cfg, stage_producer(cfg, stage))
        else:
            pin.imports()
            for key, config in pin.configs.items():
                owner = key[0]
                try:
                    projection = _config_projection(
                        config, owner, dependencies_for(owner).config_scope
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    pin.reject(f"Unclassified scientific configuration for {owner.stage}: {exc}")
                _compare(
                    pin,
                    owner.stage,
                    {"config": pin.expected[key]["config"]},
                    {"config": projection},
                )


@contextmanager
def separate_execution() -> Iterator[None]:
    """Keep launch files protected while recording diagnostics or running an independent fixture."""
    token = _CURRENT.set(None)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def verify_launch() -> None:
    pin = current()
    if pin is not None:
        for (owner, _, _), cfg in tuple(pin.configs.items()):
            checked_components(cfg, owner)


@contextmanager
def publication_transaction() -> Iterator[None]:
    """Check a fully staged data/sidecar pair before changing either canonical name."""
    publication_boundary()
    token = _PUBLISHING.set(True)
    try:
        yield
    finally:
        _PUBLISHING.reset(token)


def checked_worker_result(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Verify worker-local state before handing a completed unit to its parent."""
    result = fn(*args, **kwargs)
    publication_boundary()
    return result


def process_placement(cfg: AppConfig, stage: str) -> str:
    from importlib.metadata import distribution

    pin = current()
    if pin is None:
        return ""
    # A worker may measure fewer distributions than its multi-stage parent.
    # Paths of the installed distributions are checked when their modules resolve.
    names = dependencies_for(stage_producer(cfg, stage)).distributions
    return identity_sha256(
        {
            "checkout": str(ROOT),
            "executable": sys.executable,
            "distributions": {n: str(distribution(n).locate_file("")) for n in names},
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        }
    )


def attach_worker(expected: WorkerDependencies) -> None:
    import atexit

    active = current()
    if active is not None:
        if process_placement(expected.config, expected.stage) != expected.placement:
            active.reject(f"Worker import placement differs for {expected.stage} at {ROOT}")
    else:
        stack = ExitStack()
        try:
            stack.enter_context(scientific_invocation([(expected.config, expected.stage)]))
            if process_placement(expected.config, expected.stage) != expected.placement:
                raise ScientificDrift(
                    f"Worker import placement differs for {expected.stage} at {ROOT}"
                )
            observed = checked_components(
                expected.config, stage_producer(expected.config, expected.stage)
            )
            pin = current()
            assert pin is not None
            if expected.components is not None:
                _compare(pin, expected.stage, expected.components, describe_components(observed))
            if identity_sha256(observed) != expected.expected:
                raise ScientificDrift(
                    f"Worker {expected.stage} dependencies differ from launch: expected "
                    f"{expected.expected}, observed {identity_sha256(observed)}"
                )
            pin.worker = True
        except BaseException:
            stack.close()
            raise
        atexit.register(stack.close)


def configured_producers(
    cfg: AppConfig, stages: Sequence[str] | None = None
) -> list[tuple[AppConfig, str]]:
    """Resolve the plan and its upstream checks in the owning layout."""
    definitions = {p.definition.key: p.definition for p in cfg.stage_layout.placements}
    wanted = set(definitions if stages is None else stages)
    if not wanted:
        return []
    if "h2h_power" in wanted or "h2h_execute" in wanted:
        wanted.update(("h2h_power", "h2h_execute"))
    for stage, definition in reversed(definitions.items()):
        if stage in wanted:
            wanted.update(definition.depends_on)
    if any(p.group != "root_pair" for p in definitions.values()):
        wanted.add("simulation")
    return [(cfg, stage) for stage in sorted(wanted)]


def launch_evidence(cfg: AppConfig) -> dict[str, Any]:
    pin = current()
    if pin is None:
        return {}
    digests = {}
    for config, stage in configured_producers(cfg):
        owner = stage_producer(config, stage)
        digests[f"{owner.scope}/{stage}"] = identity_sha256(checked_components(config, owner))
    return {"producers": digests}


def stage_action(
    stage: str, *, also: tuple[str, ...] | None = None
) -> Callable[[Callable[Concatenate[AppConfig, P], T]], Callable[Concatenate[AppConfig, P], T]]:
    """Scope scientific work invoked through a direct CLI or producer entry point."""
    from functools import wraps

    from farkle.utils.scientific_identity import producer_scope

    def decorate(
        fn: Callable[Concatenate[AppConfig, P], T],
    ) -> Callable[Concatenate[AppConfig, P], T]:
        @wraps(fn)
        def run(cfg: AppConfig, *args: P.args, **kwargs: P.kwargs) -> T:
            producers = (
                configured_producers(cfg, [stage])
                if also is None
                else [(cfg, key) for key in (stage, *also)]
            )
            with scientific_invocation(producers), producer_scope(cfg, stage):
                return fn(cfg, *args, **kwargs)

        return run

    return decorate


def preflight_request(cfg: AppConfig, nodes: list[str]) -> dict[str, Any]:
    import yaml

    from farkle.config import effective_config_dict

    stage = "rng_diagnostics"
    with scientific_invocation([(cfg, stage)]):
        pin = current()
        assert pin is not None
        values = pin.components(cfg, stage_producer(cfg, stage))
        return {
            "version": 1,
            "config_yaml": yaml.safe_dump(effective_config_dict(cfg), sort_keys=True),
            "root_count": len(cfg.sim.seed_list) if cfg.sim.seed_list is not None else None,
            "mode": cfg.resource_limiter_mode.value,
            "profile": cfg._game_profile_sha256,
            "expected": cfg.stage_config_sha(stage),
            "components": json.loads(canonical_json_bytes(describe_components(values))),
            "placement": process_placement(cfg, stage),
            "nodes": nodes,
        }
