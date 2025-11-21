# Documentation audit review

## Audit command
- Ran `python doc_audit.py > audit_report.txt` to refresh the documentation issues currently flagged across tracked Python files.

## True-positive findings
- `doc_audit.py` itself still lacks docstrings for its helper dataclasses and functions, so the tool continues to fail its own checks.
- Core analytics code under `src/farkle/analysis/` (e.g., `agreement.py`, `combine.py`, `head2head.py`, `reporting.py`, `seed_summaries.py`, `tiering_report.py`) is missing function and class docstrings that would help explain preprocessing and reporting behaviors.
- Entry points and configuration helpers (`src/farkle/cli/main.py`, `src/farkle/config.py`, `src/pipeline.py`) are missing docstrings for CLI wiring, config schemas, and pipeline helpers and should be filled in.
- Game and simulation helpers (`src/farkle/game/scoring.py`, `src/farkle/simulation/power_helpers.py`, `run_tournament.py`, `watch_game.py`) have undocumented helpers that should be described.
- Utility modules (`src/farkle/utils/artifacts.py`, `logging.py`, `manifest.py`, `parallel.py`, `schema_helpers.py`, `sinks.py`, `stats.py`, `streaming_loop.py`, `writer.py`, `yaml_helpers.py`) need module and helper docstrings to clarify their behaviors.
- Package initializers `src/farkle/game/tests/__init__.py` and `tests/__init__.py` need the required first-line path comment strings; other docstring gaps in those files can be considered trivial.

## Items marked as skip (false positives / low-value)
| File(s) | Rationale |
| --- | --- |
| Test modules across `tests/` (integration, unit, helpers, and `tests/conftest.py`) | Test names already communicate intent; adding module or function docstrings would add noise without improving readability for maintainers. |
| Temporary scratch files (`tmp_debug.py`, `tmp_debug2.py`, `tmp_test.py`, `tmp_test2.py`) | Local debugging/testing harnesses not part of shipped code; documenting them is unnecessary unless they are promoted into supported modules. |
