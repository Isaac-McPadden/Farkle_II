"""Lightweight public CLI launcher that establishes OS memory enforcement."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from farkle.config import AppConfig, apply_dot_overrides, load_app_config
from farkle.utils.os_memory import (
    SETUP_FAILURE_EXIT_CODE,
    MemoryBoundaryError,
    supervise_process,
)

_PROTECTED_COMMANDS = frozenset({"run", "analyze", "two-seed-pipeline"})
_NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _protected_command(argv: Sequence[str]) -> str | None:
    """Return the public command requiring the pipeline boundary."""

    return next((token for token in argv if token in _PROTECTED_COMMANDS), None)


def _resource_config(argv: Sequence[str]) -> AppConfig:
    """Load only enough public configuration to establish the hard boundary."""

    config_path: Path | None = None
    overrides: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--config" and index + 1 < len(argv):
            config_path = Path(argv[index + 1])
            index += 2
            continue
        if token.startswith("--config="):
            config_path = Path(token.split("=", 1)[1])
        elif token == "--set" and index + 1 < len(argv):
            overrides.append(argv[index + 1])
            index += 2
            continue
        elif token.startswith("--set="):
            overrides.append(token.split("=", 1)[1])
        index += 1
    cfg = load_app_config(config_path) if config_path is not None else AppConfig()
    apply_dot_overrides(cfg, overrides)
    cfg.validate_resource_contract()
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch non-pipeline commands directly and supervise pipeline commands."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if _protected_command(arguments) is None:
        from farkle.cli.main import main as cli_main

        if argv is None:
            cli_main()
        else:
            cli_main(arguments)
        return 0

    cfg = _resource_config(arguments)
    child = [sys.executable, "-m", "farkle.cli.protected", *arguments]
    child_env = os.environ.copy()
    native_threads = str(max(1, int(cfg.resources.native_threads_per_worker)))
    for variable in _NATIVE_THREAD_ENV_VARS:
        child_env[variable] = native_threads
    child_env["PYARROW_NUM_THREADS"] = native_threads
    try:
        return supervise_process(child, cfg.resources, env=child_env)
    except MemoryBoundaryError as exc:
        print(f"Farkle OS memory boundary setup failed closed: {exc}", file=sys.stderr)
        return SETUP_FAILURE_EXIT_CODE


__all__ = ["main"]
