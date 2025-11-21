# Documentation audit review

## Audit command
- Ran `python doc_audit.py > audit_report.txt` to regenerate the current list of documentation issues across tracked Python files.

## True-positive findings
- Core analytics modules (`src/farkle/analysis/*.py`) lack function/class docstrings that would clarify preprocessing, reporting, and metric computations.
- Pipeline and configuration entry points (`src/farkle/cli/main.py`, `src/farkle/config.py`, `src/pipeline.py`) are missing docstrings that describe CLI wiring, configuration schemas, and pipeline helpers.
- Game and simulation helpers (`src/farkle/game/scoring.py`, `src/farkle/simulation/power_helpers.py`, `src/farkle/simulation/run_tournament.py`, `src/farkle/simulation/watch_game.py`) need docstrings to explain gameplay logic and simulation flows.
- Utility modules (`src/farkle/utils/*.py`) need module/function/class docstrings to document logging, manifests, parallelism, sinks, statistics, streaming loops, and writer helpers.
- Package initializers are currently compliant; any `__init__.py` missing the required first-line path string should still be fixed if new gaps arise, even if other docstring content is trivial.

## Items marked as skip (false positives / low-value)
| File(s) | Rationale |
| --- | --- |
| Test suites under `tests/` (including `tests/conftest.py`, integration tests, and unit tests) | Test function names already describe intent; adding module/function docstrings would add noise without improving maintainability. |
| Temporary scratch helpers (`tmp_debug.py`, `tmp_debug2.py`, `tmp_test.py`, `tmp_test2.py`) | Local debugging or experimentation scripts not part of the supported codebase; documenting them provides little value. |
