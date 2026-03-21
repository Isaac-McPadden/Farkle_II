# Documentation audit review

## Audit command

- Run `python doc_audit.py` from the repository root to regenerate the current
  documentation audit report.

## Current status

The current audit is not clean. The latest pass reports:

- 37 `src/` files with documentation issues
- 260 `src/` issues total

Most findings are missing docstrings on private helpers inside large modules,
not missing module-level documentation. The heaviest files are:

- `src/farkle/analysis/game_stats.py` - 44 issues
- `src/farkle/analysis/rng_diagnostics.py` - 26 issues
- `src/farkle/analysis/interseed_analysis.py` - 24 issues
- `src/farkle/analysis/__init__.py` - 17 issues
- `src/farkle/orchestration/two_seed_pipeline.py` - 17 issues

## Recommended priority order

1. Public and front-door helpers
   Add docstrings for `AppConfig` path helpers, CLI entry points, and
   orchestration functions that other modules and contributors are expected to
   call directly.
2. Large analysis orchestrators
   Document stage planners, fan-out helpers, and interseed orchestration logic
   before spending time on tiny local helper functions.
3. High-complexity accumulator modules
   Add short docstrings to the key accumulator classes and pooled reduction
   helpers in `game_stats.py` and `rng_diagnostics.py`.

## Deliberate non-goals

- Full docstring coverage for tests
- Exhaustive prose for every small private helper
- Hand-maintained statements that claim the audit is "clean"

## Maintenance notes

- Keep this file as a current summary, not a one-time milestone note.
- If the project adopts CI enforcement for `doc_audit.py`, update this page with
  the latest counts whenever the baseline changes.
