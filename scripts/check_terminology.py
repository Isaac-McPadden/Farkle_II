"""Enforce precise product terms on maintained, normative repository surfaces."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENFORCED_TREES = ("src", "configs", "scripts")
ENFORCED_FILES = (
    "README.md",
    "pyproject.toml",
    "docs/config_reference.md",
    "docs/data_artifacts.md",
    "docs/rng_contract.md",
    "docs/terminology.md",
    "docs/turn_and_row_contract.md",
)
SPELLING_ENFORCED_TREES = ("src", "configs", "scripts", "tests", "docs")
SPELLING_ENFORCED_FILES = (
    "README.md",
    "cli_args.md",
    "farkle_rules.md",
    "analysis/README.md",
    "pyproject.toml",
)
HISTORICAL_PREFIXES = (
    Path("docs/reviews"),
    Path("docs/remediation"),
)
SKIP_SUFFIXES = {".pyc", ".parquet", ".png", ".jpg", ".jpeg", ".gif", ".html"}
# Hex escapes keep the checker subject to its own rule without embedding the
# prohibited spelling in repository-owned source.
FORBIDDEN = re.compile(
    r"(?<![A-Za-z0-9])\x70\x6f\x6f\x6c(?:s|ed|ing)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
EXTERNAL_API_SYMBOLS = (
    re.compile(r"\bmultiprocessing\.\x50\x6f\x6f\x6c\b"),
    re.compile(r"\b(?:ctx|context|mp|spawn_context)\.\x50\x6f\x6f\x6c\b"),
    re.compile(r"\b(?:Process|Thread)\x50\x6f\x6f\x6cExecutor\b"),
)
QUALIFIED_PROCESS_POOL_TERMS = (
    re.compile(r"\bprocess[- ]\x70\x6f\x6f\x6cs?\b", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_]*process_pool[A-Za-z0-9_]*\b", re.IGNORECASE),
)
BRITISH_TO_AMERICAN = {
    "\x68\x6f\x6e\x6f\x75\x72\x73": "honors",
    "\x69\x6e\x69\x74\x69\x61\x6c\x69\x73\x65": "initialize",
    "\x69\x6e\x69\x74\x69\x61\x6c\x69\x73\x65\x64": "initialized",
    "\x6c\x61\x62\x65\x6c\x6c\x65\x64": "labeled",
    "\x6c\x61\x62\x65\x6c\x6c\x69\x6e\x67": "labeling",
    "\x6d\x61\x74\x65\x72\x69\x61\x6c\x69\x73\x61\x74\x69\x6f\x6e": "materialization",
    "\x6f\x72\x67\x61\x6e\x69\x73\x65\x64": "organized",
    "\x72\x65\x6c\x61\x62\x65\x6c\x6c\x65\x64": "relabeled",
    "\x73\x75\x6d\x6d\x61\x72\x69\x73\x69\x6e\x67": "summarizing",
}
BRITISH_SPELLING = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(BRITISH_TO_AMERICAN) + r")(?![A-Za-z])",
    re.IGNORECASE,
)


def _is_allowed_concurrency_usage(line: str, start: int, end: int) -> bool:
    """Return whether one prohibited-token match is process-qualified."""

    return any(
        symbol.start() <= start and end <= symbol.end()
        for pattern in (*EXTERNAL_API_SYMBOLS, *QUALIFIED_PROCESS_POOL_TERMS)
        for symbol in pattern.finditer(line)
    )


def _enforced_paths(root: Path) -> list[Path]:
    paths = [root / relative for relative in ENFORCED_FILES]
    for relative in ENFORCED_TREES:
        tree = root / relative
        if tree.exists():
            paths.extend(item for item in tree.rglob("*") if item.is_file())
    return sorted(path for path in paths if path.is_file())


def _spelling_enforced_paths(root: Path) -> list[Path]:
    paths = [root / relative for relative in SPELLING_ENFORCED_FILES]
    for relative in SPELLING_ENFORCED_TREES:
        tree = root / relative
        if tree.exists():
            paths.extend(item for item in tree.rglob("*") if item.is_file())
    return sorted(
        path
        for path in set(paths)
        if path.is_file()
        and not any(path.relative_to(root).is_relative_to(prefix) for prefix in HISTORICAL_PREFIXES)
    )


def find_violations(root: Path = ROOT) -> list[str]:
    """Return repository-relative terminology violations."""

    violations: list[str] = []
    for path in _enforced_paths(root):
        relative = path.relative_to(root)
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if any(
                not _is_allowed_concurrency_usage(line, match.start(), match.end())
                for match in FORBIDDEN.finditer(line)
            ):
                violations.append(f"{relative}:{line_number}: {line.strip()}")
    for path in _spelling_enforced_paths(root):
        relative = path.relative_to(root)
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            match = BRITISH_SPELLING.search(line)
            if match is None:
                continue
            replacement = BRITISH_TO_AMERICAN[match.group(0).lower()]
            violations.append(
                f"{relative}:{line_number}: use {replacement!r} in maintained prose: "
                f"{line.strip()}"
            )
    return sorted(set(violations))


def main() -> int:
    """Print violations and return a process status suitable for CI."""

    violations = find_violations()
    if not violations:
        return 0
    print("Repository terminology or spelling violations found:", file=sys.stderr)
    for violation in violations:
        print(violation, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
