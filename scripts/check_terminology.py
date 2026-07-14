"""Fail when ambiguous aggregation vocabulary appears in project-owned text."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = ("src", "tests", "configs", "docs", "scripts")
SKIP_FILES = {
    Path("docs/terminology.md"),
    Path("scripts/check_terminology.py"),
}
SKIP_SUFFIXES = {".pyc", ".parquet", ".png", ".jpg", ".jpeg", ".gif", ".html"}
FORBIDDEN = re.compile(r"\bpool(?:ed|ing)?\b", re.IGNORECASE)
EXTERNAL_API_ALLOWLIST = (
    re.compile(r"\bmultiprocessing\.Pool\b"),
    re.compile(r"\b(?:ctx|context|mp)\.Pool\("),
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
            if relative in SKIP_FILES or path.suffix.lower() in SKIP_SUFFIXES:
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
