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


def _is_allowed_external_api(line: str, start: int, end: int) -> bool:
    """Return whether one prohibited-token match is inside an external symbol."""

    return any(
        symbol.start() <= start and end <= symbol.end()
        for pattern in EXTERNAL_API_SYMBOLS
        for symbol in pattern.finditer(line)
    )


def _enforced_paths(root: Path) -> list[Path]:
    paths = [root / relative for relative in ENFORCED_FILES]
    for relative in ENFORCED_TREES:
        tree = root / relative
        if tree.exists():
            paths.extend(item for item in tree.rglob("*") if item.is_file())
    return sorted(path for path in paths if path.is_file())


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
                not _is_allowed_external_api(line, match.start(), match.end())
                for match in FORBIDDEN.finditer(line)
            ):
                violations.append(f"{relative}:{line_number}: {line.strip()}")
    return violations


def main() -> int:
    """Print violations and return a process status suitable for CI."""

    violations = find_violations()
    if not violations:
        return 0
    print("Ambiguous repository terminology found:", file=sys.stderr)
    for violation in violations:
        print(violation, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
