# Codex Repo Map

Generated for Codex orientation. Treat this file as a cache, not authority.
Verify source files directly before making code changes or accepting statistical
claims.

- Last verified commit: `6cd587c3cb0625567e7d8412a2f97d1e343ef3e4`
- Working tree note: generated from a dirty tree with existing edits in
  `src/farkle/utils/mdd_helpers.py` and `tests/unit/utils/test_mdd.py`
- Confidence: high for repository shape and entry points, medium for details
  that may move during active refactors

## Purpose

Farkle Mk II is a deterministic Monte Carlo simulation and analytics toolkit for
comparing Farkle strategies across player counts, seeds, and analysis methods.
The main engineering constraints are explicit seeded randomness, idempotent and
resumable stages, config-derived paths, atomic writes, streaming/low-RAM data
processing, and reuse of helpers under `src/farkle/utils`.

## Main Entry Points

- Package CLI: `src/farkle/cli/main.py`, exposed as `farkle`.
- Module CLI: `src/farkle/__main__.py`.
- Simulation front door: `src/farkle/simulation/runner.py`.
- Low-level tournament engine: `src/farkle/simulation/run_tournament.py`.
- Analysis CLI pipeline: `src/farkle/analysis/pipeline.py`.
- Analysis plan builders: `src/farkle/analysis/__init__.py`.
- Two-seed orchestration: `src/farkle/orchestration/two_seed_pipeline.py`.
- Legacy/top-level analytics helper: `src/pipeline.py`.

## Source Areas

- `src/farkle/config.py`: dataclass config model, seed normalization, stage
  layout, and canonical artifact/path helpers. Prefer `AppConfig` path helpers
  over manual path assembly.
- `src/farkle/game`: scoring lookup, Farkle game engine, final-round behavior,
  player/game metrics.
- `src/farkle/simulation`: strategy grid, tournament execution, resumability,
  per-player-count runners, power sizing wrappers.
- `src/farkle/analysis`: ingest, curate, combine, metrics, statistical stages,
  reporting, agreement, and interseed analysis.
- `src/farkle/orchestration`: seed-specific config preparation and two-seed
  pipeline coordination.
- `src/farkle/utils`: atomic writes, manifests, RNG helpers, stats helpers,
  pooling, schema helpers, streaming loops, logging, stage I/O.
- `tests`: unit and integration tests, with broad coverage but varying evidence
  strength. See `docs/codex_context/testing_and_review_map.md`.

## Runtime And Tooling

- Python target: 3.12 or newer.
- Local dependency note from `AGENTS.md`: use `.venv` at the repository root
  when operating locally in VSCode.
- Core dependencies include NumPy, pandas, PyArrow, SciPy, scikit-learn,
  matplotlib, numba, trueskill, tqdm, freezegun, pytest, Ruff, Black, mypy, and
  Pyright.
- Type checking is configured mainly for `src`.

## Project Invariants

- Randomness should come from explicit seeds and project RNG helpers, especially
  `src/farkle/utils/random.py`.
- Avoid hidden ambient randomness. Python `random` exists in a few places for
  compatibility or deterministic local selection and should be reviewed when
  touched.
- Long writes should use `atomic_path`, `write_parquet_atomic`,
  `write_csv_atomic`, `ParquetShardWriter`, or `run_streaming_shard`.
- Heavy stages should skip when `.done.json` stamps and outputs are current
  unless `--force` or equivalent is used.
- Stages should write under config-derived roots such as `cfg.analysis_dir`,
  `cfg.stage_dir(...)`, `cfg.metrics_output_path(...)`,
  `cfg.trueskill_path(...)`, and `cfg.head2head_path(...)`.
- For large data, stream or shard. Avoid whole-dataset materialization unless
  the data and computation are genuinely small.

## Review First Files

When starting a new task, read only the relevant files first:

- Config/path/stage issue: `src/farkle/config.py`,
  `src/farkle/analysis/stage_registry.py`,
  `src/farkle/analysis/stage_state.py`.
- Simulation issue: `src/farkle/simulation/runner.py`,
  `src/farkle/simulation/run_tournament.py`,
  `src/farkle/simulation/simulation.py`.
- Game rule issue: `src/farkle/game/engine.py`,
  `src/farkle/game/scoring.py`,
  `src/farkle/game/scoring_lookup.py`, `farkle_rules.md`.
- Statistical issue: start with
  `docs/codex_context/statistical_methods_map.md`, then verify the listed
  modules directly.
- Test evidence issue: `docs/codex_context/testing_and_review_map.md`,
  `pytest.ini`, `tests/conftest.py`, and the specific test files named by
  the method or subsystem.

## Existing Human Review Docs

- `docs/human_review_checklist.md`: main claim-review workflow.
- `docs/test_suite_review_checklist.md`: review whether tests are trustworthy
  evidence.
- `docs/data_artifacts.md`: artifact/path contract overview.
- `docs/config_reference.md`: concise config model summary.
- `statistical_analyses.md`: one-line map of statistical modules.

## Maintenance Rule

Update this pack after meaningful changes to architecture, stage ordering,
statistical methods, config semantics, or artifact contracts. Update the commit
SHA and dirty-tree note at the top. Do not use these summaries as proof without
spot-checking source.
