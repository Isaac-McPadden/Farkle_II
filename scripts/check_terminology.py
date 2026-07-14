"""Fail when the prohibited ambiguous word family appears in project text."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = ("src", "tests", "configs", "docs", "scripts")
SKIP_SUFFIXES = {".pyc", ".parquet", ".png", ".jpg", ".jpeg", ".gif", ".html"}
# Hex escapes keep the checker subject to its own rule without embedding the
# prohibited spelling in repository-owned source.
FORBIDDEN = re.compile(r"\x70\x6f\x6f\x6c(?:s|ed|ing)?", re.IGNORECASE)
EXTERNAL_API_ALLOWLIST = (
    re.compile(r"\bmultiprocessing\.\x50\x6f\x6f\x6c\b"),
    re.compile(r"\b(?:ctx|context|mp)\.\x50\x6f\x6f\x6c\("),
    re.compile(r"\b(?:Process|Thread)\x50\x6f\x6f\x6cExecutor\b"),
)


def _is_allowed_external_api(line: str) -> bool:
    return any(pattern.search(line) for pattern in EXTERNAL_API_ALLOWLIST)


def find_violations() -> list[str]:
    """Return repository-relative terminology violations."""

    violations: list[str] = []
    for root_name in SEARCH_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(ROOT)
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if FORBIDDEN.search(line) and not _is_allowed_external_api(line):
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
