# doc_audit.py
"""Documentation audit tool for Python modules.

This script audits path headers, module docstrings, and symbol docstrings for
tracked Python files. It is designed to support an iterative documentation
backlog by scoping to a subtree such as ``src`` and by prioritizing findings
that are most likely to matter to contributors.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Sequence

IssueCode = Literal[
    "missing_path_comment",
    "path_comment_mismatch",
    "missing_module_docstring",
    "module_docstring_not_immediate",
    "syntax_error",
    "missing_public_docstring",
    "missing_private_complex_docstring",
]
PrivatePolicy = Literal["all", "complex", "none"]

IGNORED_DIRS = {
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "docs",
    "experiments",
    "notebooks",
    "results",
    "results_seed_0",
}

CONTROL_FLOW_NODES = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Match,
)

COMPLEX_CALL_NAMES = {
    "check_call",
    "check_output",
    "glob",
    "iterdir",
    "mkdir",
    "open",
    "read_bytes",
    "read_csv",
    "read_json",
    "read_parquet",
    "read_table",
    "read_text",
    "replace",
    "rename",
    "rglob",
    "run",
    "to_csv",
    "to_json",
    "to_parquet",
    "touch",
    "unlink",
    "write_bytes",
    "write_json",
    "write_table",
    "write_text",
}


@dataclass(slots=True)
class Issue:
    """Represent a single documentation issue discovered in a file.

    Attributes:
        code: Stable machine-readable issue identifier.
        description: Human-readable explanation of the issue.
        line: Optional line number associated with the issue.
    """

    code: IssueCode
    description: str
    line: int | None = None

    def format(self) -> str:
        """Format the issue for human-readable report output.

        Returns:
            String containing the issue description with an optional line number.
        """
        if self.line is None:
            return f"- [{self.code}] {self.description}"
        return f"- L{self.line}: [{self.code}] {self.description}"

    def to_dict(self) -> dict[str, object]:
        """Return a machine-readable representation of the issue.

        Returns:
            Dictionary containing the stable issue code, description, and line.
        """
        payload: dict[str, object] = {
            "code": self.code,
            "description": self.description,
        }
        if self.line is not None:
            payload["line"] = self.line
        return payload


@dataclass(slots=True)
class FileReport:
    """Aggregate documentation issues for a single file.

    Attributes:
        path: Filesystem path to the Python source being audited.
        issues: Collection of issues discovered during auditing.
    """

    path: Path
    issues: list[Issue] = field(default_factory=list)

    def add(self, code: IssueCode, description: str, line: int | None = None) -> None:
        """Append a discovered issue to the report.

        Args:
            code: Stable machine-readable issue identifier.
            description: Explanation of the missing documentation detail.
            line: Optional line number where the issue was detected.
        """
        self.issues.append(Issue(code=code, description=description, line=line))

    @property
    def has_issues(self) -> bool:
        """Indicate whether any issues have been recorded.

        Returns:
            ``True`` if issues are present; otherwise ``False``.
        """
        return bool(self.issues)

    @property
    def issue_count(self) -> int:
        """Return the number of issues recorded for the file.

        Returns:
            Total issue count for the file.
        """
        return len(self.issues)

    def counts_by_code(self) -> Counter[IssueCode]:
        """Count issues grouped by stable code.

        Returns:
            Counter of issue-code occurrences for the file.
        """
        return Counter(issue.code for issue in self.issues)

    def format(self, repo_root: Path) -> str:
        """Return a multi-line string summarizing all issues for a file.

        Args:
            repo_root: Repository root used to render relative paths.

        Returns:
            Formatted string starting with the relative path followed by issues.
        """
        rel = relative_path(self.path, repo_root)
        parts = [f"{rel} ({self.issue_count})"]
        for issue in self.issues:
            parts.append(f"  {issue.format()}")
        return "\n".join(parts)

    def to_dict(self, repo_root: Path) -> dict[str, object]:
        """Return a machine-readable representation of the file report.

        Args:
            repo_root: Repository root used to render relative paths.

        Returns:
            Dictionary containing the file path, counts, and issue payloads.
        """
        counts = self.counts_by_code()
        return {
            "path": relative_path(self.path, repo_root),
            "issue_count": self.issue_count,
            "counts_by_code": dict(sorted(counts.items())),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(slots=True)
class AuditSummary:
    """Summarize a completed documentation audit run.

    Attributes:
        repo_root: Repository root for relative path rendering.
        scopes: Scope paths that were audited.
        reports: Completed file reports, including files without issues.
        private_policy: Policy used for private symbol docstring enforcement.
    """

    repo_root: Path
    scopes: list[Path]
    reports: list[FileReport]
    private_policy: PrivatePolicy

    @property
    def file_count(self) -> int:
        """Return the number of Python files scanned.

        Returns:
            Total number of files audited.
        """
        return len(self.reports)

    @property
    def reports_with_issues(self) -> list[FileReport]:
        """Return only file reports that contain issues.

        Returns:
            List of file reports whose issue collections are non-empty.
        """
        return [report for report in self.reports if report.has_issues]

    @property
    def files_with_issues(self) -> int:
        """Return the number of files with at least one issue.

        Returns:
            Count of files that failed the audit.
        """
        return len(self.reports_with_issues)

    @property
    def total_issues(self) -> int:
        """Return the total issue count across all files.

        Returns:
            Sum of all issue counts from every report.
        """
        return sum(report.issue_count for report in self.reports)

    def counts_by_code(self) -> Counter[IssueCode]:
        """Count issues grouped by stable code across the whole audit.

        Returns:
            Counter of issue-code occurrences for the audit run.
        """
        counts: Counter[IssueCode] = Counter()
        for report in self.reports:
            counts.update(report.counts_by_code())
        return counts

    def sorted_reports(self) -> list[FileReport]:
        """Return issue-bearing reports sorted by priority.

        Returns:
            Reports ordered by descending issue count, then by relative path.
        """
        return sorted(
            self.reports_with_issues,
            key=lambda report: (-report.issue_count, relative_path(report.path, self.repo_root)),
        )

    def top_files(self, limit: int) -> list[FileReport]:
        """Return the highest-priority files based on issue count.

        Args:
            limit: Maximum number of files to include.

        Returns:
            Sorted list of the most issue-dense file reports.
        """
        return self.sorted_reports()[:limit]

    def to_dict(self) -> dict[str, object]:
        """Return a machine-readable representation of the audit summary.

        Returns:
            Dictionary containing summary counts and per-file findings.
        """
        counts = self.counts_by_code()
        return {
            "repo_root": str(self.repo_root),
            "scopes": [relative_path(scope, self.repo_root) for scope in self.scopes],
            "private_policy": self.private_policy,
            "file_count": self.file_count,
            "files_with_issues": self.files_with_issues,
            "total_issues": self.total_issues,
            "counts_by_code": dict(sorted(counts.items())),
            "files": [report.to_dict(self.repo_root) for report in self.sorted_reports()],
        }


class DocstringVisitor(ast.NodeVisitor):
    """Collect symbol-level docstring issues from an AST.

    Private symbols are filtered according to the selected policy so the audit
    can focus on public APIs or on private helpers that are likely to be
    non-obvious to maintainers.
    """

    def __init__(self, report: FileReport, private_policy: PrivatePolicy) -> None:
        """Initialize the visitor.

        Args:
            report: Report that will receive any missing-docstring issues.
            private_policy: Enforcement policy for private helpers.
        """
        self.report = report
        self.private_policy = private_policy
        self.parents: list[ast.AST] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Audit a class definition and recurse into its members.

        Args:
            node: Class definition node being visited.
        """
        self._check_symbol(node)
        self.parents.append(node)
        self.generic_visit(node)
        self.parents.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Audit a function definition and recurse into nested symbols.

        Args:
            node: Function definition node being visited.
        """
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Audit an async function definition and recurse into nested symbols.

        Args:
            node: Async function definition node being visited.
        """
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Audit a function-like node and recurse into nested symbols.

        Args:
            node: Function or async-function definition node being visited.
        """
        self._check_symbol(node)
        self.parents.append(node)
        self.generic_visit(node)
        self.parents.pop()

    def _check_symbol(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Record a missing-docstring issue when a symbol requires one.

        Args:
            node: Class or function-like node to evaluate.
        """
        if node.name.startswith("__") and node.name.endswith("__"):
            return
        if is_trivial_body(node.body):
            return
        if ast.get_docstring(node) is not None:
            return

        kind = symbol_kind(node, self.parents)
        visibility = symbol_visibility(node, self.parents)

        if visibility == "private":
            if not private_symbol_requires_docstring(node, self.private_policy):
                return
            self.report.add(
                "missing_private_complex_docstring",
                f"Missing docstring for {kind} '{node.name}'",
                node.lineno,
            )
            return

        self.report.add(
            "missing_public_docstring",
            f"Missing docstring for {kind} '{node.name}'",
            node.lineno,
        )


def relative_path(path: Path, repo_root: Path) -> str:
    """Render a repository-relative path using POSIX separators.

    Args:
        path: Path to render relative to the repository root.
        repo_root: Repository root used for relative path calculations.

    Returns:
        Repository-relative path string with forward slashes.
    """
    return path.relative_to(repo_root).as_posix()


def normalize_scopes(repo_root: Path, raw_scopes: Sequence[str] | None) -> list[Path]:
    """Resolve requested audit scopes against the repository root.

    Args:
        repo_root: Repository root used for resolving relative scope paths.
        raw_scopes: Raw scope strings from CLI arguments.

    Returns:
        Sorted, de-duplicated absolute scope paths that lie under the repo root.
    """
    if not raw_scopes:
        return [repo_root]

    normalized: list[Path] = []
    for raw_scope in raw_scopes:
        scope = (repo_root / raw_scope).resolve()
        try:
            scope.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(f"Scope must live under the repository root: {raw_scope}") from exc
        normalized.append(scope)

    unique_scopes = sorted(set(normalized))
    if not unique_scopes:
        return [repo_root]
    return unique_scopes


def git_tracked_py_files(repo_root: Path) -> list[Path]:
    """List tracked Python files, falling back to globbing if Git is unavailable.

    Args:
        repo_root: Root directory of the repository to inspect.

    Returns:
        Paths to Python files that are not ignored by the audit rules.
    """
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "ls-files", "*.py"],
            text=True,
        )
        files = [repo_root / line.strip() for line in output.splitlines() if line.strip()]
        return [path for path in files if not is_ignored(path, repo_root)]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [path for path in repo_root.rglob("*.py") if not is_ignored(path, repo_root)]


def scoped_py_files(repo_root: Path, scopes: Sequence[Path]) -> list[Path]:
    """Return tracked Python files limited to the requested scopes.

    Args:
        repo_root: Repository root used to discover tracked files.
        scopes: Absolute scope paths to include in the audit.

    Returns:
        Sorted list of Python files that live within any requested scope path.
    """
    files = []
    for path in git_tracked_py_files(repo_root):
        if any(path.is_relative_to(scope) for scope in scopes):
            files.append(path)
    return sorted(files)


def is_ignored(path: Path, repo_root: Path) -> bool:
    """Determine whether a path should be excluded from the audit.

    Args:
        path: Candidate file path.
        repo_root: Repository root used to compute relative path segments.

    Returns:
        ``True`` if any path segment matches the ignored-directory list.
    """
    rel_parts = path.relative_to(repo_root).parts
    return any(part in IGNORED_DIRS for part in rel_parts)


def read_lines(path: Path) -> list[str]:
    """Read a UTF-8 text file into individual lines.

    Args:
        path: File to read.

    Returns:
        List of lines without trailing newline characters.
    """
    return path.read_text(encoding="utf-8").splitlines()


def check_path_comment(
    path: Path,
    lines: list[str],
    report: FileReport,
    repo_root: Path,
) -> int | None:
    """Validate the expected line-one path comment for a file.

    Args:
        path: File being audited.
        lines: File contents split into lines.
        report: Report that should receive any discovered issues.
        repo_root: Repository root for relative path calculations.

    Returns:
        Index of the path-comment line when valid; otherwise ``None``.
    """
    index = 0
    if lines and lines[0].startswith("#!"):
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        report.add("missing_path_comment", "Missing path comment (file is empty)")
        return None

    line = lines[index].strip()
    expected = f"# {relative_path(path, repo_root)}"
    if not line.startswith("#"):
        report.add("missing_path_comment", "First comment is missing or not at top of file")
        return None
    if line != expected:
        report.add(
            "path_comment_mismatch",
            f"Path comment mismatch (expected: '{expected}')",
            index + 1,
        )
        return None
    return index


def module_docstring_is_immediate(lines: list[str], path_comment_idx: int) -> bool:
    """Check whether a module docstring directly follows the path comment.

    Args:
        lines: File contents split into lines.
        path_comment_idx: Index of the verified path-comment line.

    Returns:
        ``True`` when the docstring begins immediately after the path comment.
    """
    next_idx = path_comment_idx + 1
    return next_idx < len(lines) and lines[next_idx].lstrip().startswith(('"""', "'''"))


def non_docstring_body(body: list[ast.stmt]) -> list[ast.stmt]:
    """Return a node body with its leading docstring expression removed.

    Args:
        body: AST statements representing a function or class body.

    Returns:
        Body statements excluding an initial string-literal docstring expression.
    """
    if not body:
        return []
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return body[1:]
    return body


def is_trivial_body(body: list[ast.stmt]) -> bool:
    """Determine whether a node body contains only no-op statements.

    Args:
        body: AST statements representing a function or class body.

    Returns:
        ``True`` if the body is empty or only contains pass/ellipsis placeholders.
    """
    remaining = non_docstring_body(body)
    if not remaining:
        return True
    for stmt in remaining:
        if isinstance(stmt, ast.Pass):
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        ):
            continue
        return False
    return True


def iter_body_nodes(statements: Sequence[ast.stmt]) -> Iterator[ast.AST]:
    """Yield AST nodes from a body while skipping nested symbol definitions.

    Args:
        statements: Statements to inspect for complexity signals.

    Yields:
        AST nodes contained in the body, excluding nested functions and classes.
    """
    stack: list[ast.AST] = list(reversed(statements))
    while stack:
        node = stack.pop()
        yield node
        for child in reversed(list(ast.iter_child_nodes(node))):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            stack.append(child)


def parameter_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count function parameters excluding a leading ``self`` or ``cls``.

    Args:
        node: Function-like node whose signature should be summarized.

    Returns:
        Number of parameters that callers are expected to provide or understand.
    """
    positional = [arg.arg for arg in node.args.posonlyargs + node.args.args]
    if positional and positional[0] in {"self", "cls"}:
        positional = positional[1:]

    count = len(positional) + len(node.args.kwonlyargs)
    if node.args.vararg is not None:
        count += 1
    if node.args.kwarg is not None:
        count += 1
    return count


def callee_name(func: ast.AST) -> str | None:
    """Extract a simple function name from a call target when possible.

    Args:
        func: AST node representing the callable part of a function call.

    Returns:
        Bare identifier or attribute name when it can be determined.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def private_symbol_requires_docstring(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    private_policy: PrivatePolicy,
) -> bool:
    """Apply the private-helper docstring policy to a symbol.

    Args:
        node: Symbol being evaluated.
        private_policy: Policy controlling how aggressively to require docs.

    Returns:
        ``True`` when the symbol should be treated as needing a docstring.
    """
    if private_policy == "none":
        return False
    if private_policy == "all":
        return True

    body = non_docstring_body(node.body)
    if len(body) >= 4:
        return True

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and parameter_count(node) >= 4:
        return True

    return_count = 0
    for child in iter_body_nodes(body):
        if isinstance(child, CONTROL_FLOW_NODES):
            return True
        if isinstance(child, ast.Return):
            return_count += 1
            if return_count >= 2:
                return True
        if isinstance(child, ast.Call):
            name = callee_name(child.func)
            if name in COMPLEX_CALL_NAMES:
                return True

    return False


def symbol_kind(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    parents: Sequence[ast.AST],
) -> str:
    """Classify a symbol as a class, method, or function.

    Args:
        node: Symbol currently being audited.
        parents: Parent stack describing the current AST traversal context.

    Returns:
        Human-readable symbol kind for diagnostics.
    """
    if isinstance(node, ast.ClassDef):
        return "class"
    if any(isinstance(parent, ast.ClassDef) for parent in reversed(parents)):
        return "method"
    return "function"


def symbol_visibility(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    parents: Sequence[ast.AST],
) -> Literal["public", "private"]:
    """Determine whether a symbol should be treated as public or private.

    Args:
        node: Symbol currently being audited.
        parents: Parent stack describing the current AST traversal context.

    Returns:
        ``"public"`` for public API symbols; otherwise ``"private"``.
    """
    enclosing_function = next(
        (
            parent
            for parent in reversed(parents)
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )
    if enclosing_function is not None:
        return "private"

    enclosing_class = next(
        (parent for parent in reversed(parents) if isinstance(parent, ast.ClassDef)),
        None,
    )
    if node.name.startswith("_"):
        return "private"
    if enclosing_class is not None and enclosing_class.name.startswith("_"):
        return "private"
    return "public"


def audit_file(path: Path, repo_root: Path, private_policy: PrivatePolicy) -> FileReport:
    """Audit a single file for required path comments and docstrings.

    Args:
        path: File path to inspect.
        repo_root: Repository root for relative path calculations.
        private_policy: Enforcement policy for private symbol docstrings.

    Returns:
        Report containing any discovered documentation issues.
    """
    report = FileReport(path=path)
    lines = read_lines(path)
    path_idx = check_path_comment(path, lines, report, repo_root)

    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError as exc:
        report.add("syntax_error", f"Syntax error prevents parsing: {exc}")
        return report

    module_doc = ast.get_docstring(tree)
    if module_doc is None:
        report.add("missing_module_docstring", "Missing module docstring")
    elif path_idx is not None and not module_docstring_is_immediate(lines, path_idx):
        report.add(
            "module_docstring_not_immediate",
            "Module docstring is not immediately after path comment",
        )

    DocstringVisitor(report, private_policy=private_policy).visit(tree)
    return report


def generate_report(
    repo_root: Path,
    scopes: Sequence[Path],
    private_policy: PrivatePolicy,
) -> list[FileReport]:
    """Run the audit across tracked Python files within the requested scopes.

    Args:
        repo_root: Root directory for the repository being checked.
        scopes: Absolute scope paths limiting which files are audited.
        private_policy: Enforcement policy for private helper docstrings.

    Returns:
        Collection of per-file reports representing audit findings.
    """
    return [
        audit_file(path, repo_root, private_policy=private_policy)
        for path in scoped_py_files(repo_root, scopes)
    ]


def format_text(summary: AuditSummary, top_files: int) -> str:
    """Render the audit summary as a human-readable text report.

    Args:
        summary: Completed audit summary to render.
        top_files: Number of top-priority files to include in the summary table.

    Returns:
        Multi-line plain-text report.
    """
    lines = [
        "Documentation audit summary",
        f"Scope: {', '.join(relative_path(scope, summary.repo_root) for scope in summary.scopes)}",
        f"Private policy: {summary.private_policy}",
        f"Files scanned: {summary.file_count}",
        f"Files with issues: {summary.files_with_issues}",
        f"Total issues: {summary.total_issues}",
    ]

    counts = summary.counts_by_code()
    if not counts:
        lines.append("All files passed documentation audit.")
        return "\n".join(lines)

    lines.extend(["", "Issue counts:"])
    for code, count in sorted(counts.items()):
        lines.append(f"- {code}: {count}")

    lines.extend(["", "Top files by issue count:"])
    for report in summary.top_files(top_files):
        lines.append(f"- {relative_path(report.path, summary.repo_root)}: {report.issue_count}")

    lines.append("")
    lines.append("Detailed findings:")
    for report in summary.sorted_reports():
        lines.append(report.format(summary.repo_root))
        lines.append("")

    return "\n".join(lines).rstrip()


def format_markdown(summary: AuditSummary, top_files: int) -> str:
    """Render the audit summary as a prioritized Markdown backlog.

    Args:
        summary: Completed audit summary to render.
        top_files: Number of top-priority files to include in the hotspot table.

    Returns:
        Markdown report suitable for committing to the repository.
    """
    scope_label = ", ".join(relative_path(scope, summary.repo_root) for scope in summary.scopes)
    counts = summary.counts_by_code()

    lines = [
        "# Src Documentation Backlog",
        "",
        "Generated by `doc_audit.py`.",
        "",
        "## Scope",
        "",
        f"- Scope: `{scope_label}`",
        f"- Private helper policy: `{summary.private_policy}`",
        f"- Files scanned: `{summary.file_count}`",
        f"- Files with issues: `{summary.files_with_issues}`",
        f"- Total issues: `{summary.total_issues}`",
        "",
        "Private helper findings under the `complex` policy are heuristic. Public symbols,"
        " path headers, and module docstrings are always required.",
        "",
    ]

    if not counts:
        lines.extend(["## Status", "", "All scoped files passed the documentation audit."])
        return "\n".join(lines)

    lines.extend(["## Issue Counts", "", "| Issue code | Count |", "| --- | ---: |"])
    for code, count in sorted(counts.items()):
        lines.append(f"| `{code}` | {count} |")

    path_wave = [
        report
        for report in summary.sorted_reports()
        if any(
            issue.code
            in {
                "missing_path_comment",
                "path_comment_mismatch",
                "missing_module_docstring",
                "module_docstring_not_immediate",
            }
            for issue in report.issues
        )
    ]
    if path_wave:
        lines.extend(["", "## Wave 1: Header And Module Cleanup", ""])
        for report in path_wave:
            lines.append(f"- `{relative_path(report.path, summary.repo_root)}`")

    lines.extend(
        [
            "",
            "## Top Hotspots",
            "",
            "| File | Issues | Public | Private complex |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for report in summary.top_files(top_files):
        counts_by_code = report.counts_by_code()
        lines.append(
            "| "
            f"`{relative_path(report.path, summary.repo_root)}` | "
            f"{report.issue_count} | "
            f"{counts_by_code.get('missing_public_docstring', 0)} | "
            f"{counts_by_code.get('missing_private_complex_docstring', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Detailed Backlog",
            "",
            "| File | Issues | Header | Module | Public | Private complex |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for report in summary.sorted_reports():
        counts_by_code = report.counts_by_code()
        header_issues = counts_by_code.get("missing_path_comment", 0) + counts_by_code.get(
            "path_comment_mismatch", 0
        )
        module_issues = counts_by_code.get("missing_module_docstring", 0) + counts_by_code.get(
            "module_docstring_not_immediate", 0
        )
        lines.append(
            "| "
            f"`{relative_path(report.path, summary.repo_root)}` | "
            f"{report.issue_count} | "
            f"{header_issues} | "
            f"{module_issues} | "
            f"{counts_by_code.get('missing_public_docstring', 0)} | "
            f"{counts_by_code.get('missing_private_complex_docstring', 0)} |"
        )

    return "\n".join(lines)


def render_report(summary: AuditSummary, report_format: str, top_files: int) -> str:
    """Render the audit summary in the requested output format.

    Args:
        summary: Completed audit summary to render.
        report_format: Output format name.
        top_files: Number of hotspot files to include where relevant.

    Returns:
        Rendered report string.
    """
    if report_format == "json":
        return json.dumps(summary.to_dict(), indent=2, sort_keys=True)
    if report_format == "markdown":
        return format_markdown(summary, top_files=top_files)
    return format_text(summary, top_files=top_files)


def write_output(output_path: Path, text: str) -> None:
    """Persist rendered report output to disk.

    Args:
        output_path: File that should receive the rendered report.
        text: Report contents to write.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface for the audit tool.

    Returns:
        Configured argument parser for the documentation audit script.
    """
    parser = argparse.ArgumentParser(prog="doc_audit.py")
    parser.add_argument(
        "--scope",
        action="append",
        dest="scopes",
        help="Repository-relative path to audit. Repeat to include multiple scopes.",
    )
    parser.add_argument(
        "--private-policy",
        choices=("all", "complex", "none"),
        default="complex",
        help="How aggressively to require docstrings for private symbols.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Report output format.",
    )
    parser.add_argument(
        "--top-files",
        type=int,
        default=15,
        help="Number of highest-priority files to include in summary sections.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the rendered report instead of stdout.",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with status 1 when any issues are found.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the documentation audit from the command line.

    Args:
        argv: Optional CLI arguments. When omitted, ``sys.argv`` is used.

    Returns:
        Process exit status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    try:
        scopes = normalize_scopes(repo_root, args.scopes)
    except ValueError as exc:
        parser.error(str(exc))

    reports = generate_report(
        repo_root=repo_root,
        scopes=scopes,
        private_policy=args.private_policy,
    )
    summary = AuditSummary(
        repo_root=repo_root,
        scopes=scopes,
        reports=reports,
        private_policy=args.private_policy,
    )
    rendered = render_report(summary, report_format=args.format, top_files=args.top_files)

    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        write_output(output_path, rendered)
    else:
        print(rendered)

    return 1 if args.fail_on_issues and summary.total_issues else 0


if __name__ == "__main__":
    sys.exit(main())
