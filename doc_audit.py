# doc_audit.py
"""Documentation audit tool for repository modules."""
from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List


@dataclass
class Issue:
    description: str
    line: int | None = None

    def format(self) -> str:
        if self.line is None:
            return f"- {self.description}"
        return f"- L{self.line}: {self.description}"


@dataclass
class FileReport:
    path: Path
    issues: List[Issue] = field(default_factory=list)

    def add(self, description: str, line: int | None = None) -> None:
        self.issues.append(Issue(description, line))

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    def format(self, repo_root: Path) -> str:
        rel = self.path.relative_to(repo_root)
        parts = [f"{rel.as_posix()}"]
        for issue in self.issues:
            parts.append(f"  {issue.format()}")
        return "\n".join(parts)


IGNORED_DIRS = {
    ".git",
    "results",
    "results_seed_0",
    "__pycache__",
    "data",
    "docs",
    "notebooks",
    "experiments",
}


def git_tracked_py_files(repo_root: Path) -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "ls-files", "*.py"], text=True
        )
        files = [repo_root / line.strip() for line in output.splitlines() if line.strip()]
        return [path for path in files if not is_ignored(path, repo_root)]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [
            path
            for path in repo_root.rglob("*.py")
            if not is_ignored(path, repo_root)
        ]


def is_ignored(path: Path, repo_root: Path) -> bool:
    rel_parts = path.relative_to(repo_root).parts
    return any(part in IGNORED_DIRS for part in rel_parts)


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def check_path_comment(path: Path, lines: list[str], report: FileReport, repo_root: Path) -> int | None:
    """Return the index of the path comment line if valid, else None."""
    index = 0
    if lines and lines[0].startswith("#!/"):
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        report.add("Missing path comment (file is empty)")
        return None

    line = lines[index].strip()
    expected = f"# {path.relative_to(repo_root).as_posix()}"
    if not line.startswith("#"):
        report.add("First comment is missing or not at top of file")
        return None

    content = line.lstrip("#").strip()
    if content != path.relative_to(repo_root).as_posix():
        report.add(f"Path comment mismatch (expected: '{expected}')", index + 1)
        return None

    return index


def module_docstring_is_immediate(lines: list[str], path_comment_idx: int) -> bool:
    next_idx = path_comment_idx + 1
    return next_idx < len(lines) and lines[next_idx].lstrip().startswith(("\"\"\"", "'''"))


def is_trivial_body(body: list[ast.stmt]) -> bool:
    body_iter = iter(body)
    first = next(body_iter, None)
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(
        first.value.value, str
    ):
        # Skip docstring expression
        first = next(body_iter, None)

    remaining = [first] + list(body_iter) if first is not None else []
    if not remaining:
        return True

    for stmt in remaining:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
            continue
        return False
    return True


def collect_docstring_issues(tree: ast.AST, report: FileReport) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            if is_trivial_body(node.body):
                continue
            if ast.get_docstring(node) is None:
                report.add(f"Missing docstring for {node.__class__.__name__} '{node.name}'", node.lineno)


def audit_file(path: Path, repo_root: Path) -> FileReport:
    report = FileReport(path)
    lines = read_lines(path)

    path_idx = check_path_comment(path, lines, report, repo_root)

    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError as exc:
        report.add(f"Syntax error prevents parsing: {exc}")
        return report

    module_doc = ast.get_docstring(tree)
    if module_doc is None:
        report.add("Missing module docstring")
    elif path_idx is not None and not module_docstring_is_immediate(lines, path_idx):
        report.add("Module docstring is not immediately after path comment")

    collect_docstring_issues(tree, report)
    return report


def generate_report(repo_root: Path) -> list[FileReport]:
    reports: list[FileReport] = []
    for path in git_tracked_py_files(repo_root):
        reports.append(audit_file(path, repo_root))
    return reports


def print_report(reports: Iterable[FileReport], repo_root: Path) -> None:
    reports = list(reports)
    any_issues = False
    for report in reports:
        if report.has_issues:
            any_issues = True
            print(report.format(repo_root))
            print()
    if not any_issues:
        print("All files passed documentation audit.")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    print_report(generate_report(ROOT), ROOT)
