# Documentation Audit Workflow

## Documentation Standard

For Python modules under `src`, use this baseline:

- Line 1 is the repository-relative module path in a basic comment, for example
  `# src/farkle/orchestration/pipeline.py`.
- The module docstring starts immediately after the path comment and explains the
  module's responsibility, key side effects, and primary outputs when relevant.
- Public classes, functions, and methods always get docstrings.
- Private helpers get docstrings when they are not immediately self-explanatory.
  For this project, that usually means coordination logic, non-trivial data
  shaping, statistical transforms, filesystem work, checkpoint handling, or
  seed/config-driven behavior.
- Symbol docstrings should include a short summary plus expected inputs and
  outputs. Side effects and file artifacts should be called out when they are
  important to callers or maintainers.

## Audit Commands

Use the project virtual environment when running the audit locally:

```powershell
.\.venv\Scripts\python.exe doc_audit.py --scope src --format text --top-files 15
```

Regenerate the committed Markdown backlog:

```powershell
.\.venv\Scripts\python.exe doc_audit.py --scope src --private-policy complex --format markdown --output docs/src_documentation_backlog.md
```

Emit machine-readable output for automation:

```powershell
.\.venv\Scripts\python.exe doc_audit.py --scope src --private-policy complex --format json
```

Make CI fail when scoped issues remain:

```powershell
.\.venv\Scripts\python.exe doc_audit.py --scope src --private-policy complex --fail-on-issues
```

## Iterative Process

1. Regenerate `docs/src_documentation_backlog.md`.
2. Clear all header and module-level issues first.
3. Work the highest-priority hotspot files in batches of roughly 3 to 6 files or
   25 to 40 docstrings.
4. Prefer documenting public APIs and orchestration entry points before private
   helpers in deeper analysis modules.
5. Re-run the audit after each batch and commit the refreshed backlog.
6. Once the backlog is small, enable `--fail-on-issues` in CI for changed files,
   then ratchet to the whole `src` tree.

## Priority Order

Use this order when selecting the next batch:

1. Missing path comments and any module docstring placement issues.
2. Public classes, functions, and methods.
3. Large orchestration and analysis hotspot files.
4. Remaining private-complex helpers in utility and simulation modules.

The current prioritized queue is committed in `docs/src_documentation_backlog.md`
and should be treated as generated output rather than hand-maintained notes.
