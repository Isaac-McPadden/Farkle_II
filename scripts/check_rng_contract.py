"""Audit project Python for RNG APIs that violate coordinate ownership."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = ("src", "tests", "scripts")


class _RngContractVisitor(ast.NodeVisitor):
    def __init__(self, *, reject_hash: bool) -> None:
        self.reject_hash = reject_hash
        self.violations: list[tuple[int, str]] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        if any(alias.name == "random" for alias in node.names):
            self.violations.append((node.lineno, "Python random module import"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.level == 0 and node.module == "random":
            self.violations.append((node.lineno, "Python random module import"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in {"default_rng", "Random"}:
            self.violations.append((node.lineno, f"forbidden RNG constructor {function.attr}"))
        if self.reject_hash and isinstance(function, ast.Name) and function.id == "hash":
            self.violations.append((node.lineno, "built-in hash use in production code"))
        self.generic_visit(node)


def find_violations() -> list[str]:
    """Return deterministic repository-relative RNG contract violations."""

    violations: list[str] = []
    for root_name in SEARCH_ROOTS:
        root = ROOT / root_name
        for path in sorted(root.rglob("*.py")):
            if path.resolve() == Path(__file__).resolve():
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                violations.append(f"{path.relative_to(ROOT)}: unable to audit: {exc}")
                continue
            visitor = _RngContractVisitor(reject_hash=root_name in {"src", "scripts"})
            visitor.visit(tree)
            violations.extend(
                f"{path.relative_to(ROOT)}:{line}: {message}"
                for line, message in visitor.violations
            )
    return violations


def main() -> int:
    violations = find_violations()
    if violations:
        print("RNG contract violations:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
