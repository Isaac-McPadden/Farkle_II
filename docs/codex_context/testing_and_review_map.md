# Codex Testing And Review Map

Generated for Codex orientation. Treat this file as a cache, not authority.
Always rerun targeted tests or inspect source before accepting conclusions.

- Last verified commit: `6cd587c3cb0625567e7d8412a2f97d1e343ef3e4`
- Working tree note: generated from a dirty tree with existing edits in
  `src/farkle/utils/mdd_helpers.py` and `tests/unit/utils/test_mdd.py`
- Sources inspected: `pytest.ini`, `tests/conftest.py`, `tests/FAILURE_NOTES.md`,
  `tests/helpers`, representative unit/integration test names, existing review
  checklists

## Test Harness

- Pytest config: `pytest.ini`.
- Test roots: all tests under `tests`.
- File pattern: `test_*.py`.
- Markers: `unit`, `integration`.
- Addopts: `-q -ra`.
- Strict xfail: enabled.
- Warning filters suppress selected pandas/numpy FutureWarning and RuntimeWarning
  noise.

`tests/conftest.py` behavior to remember:

- Installs compatibility shims for missing optional modules in some cases.
- Patches numba JIT/NJIT to identity behavior for tests.
- Provides lightweight scikit-learn stubs if sklearn is absent.
- Uses freezegun for deterministic timestamps.
- Autouse seed fixture calls `random.seed(1337)`, creates a NumPy generator
  with seed 1337, and sets `PYTHONHASHSEED=0`.

## Useful Test Targets By Task

- Stats helpers: `tests/unit/utils/test_stats.py`,
  `tests/test_stats_wilson.py`.
- MDD/frequentist: `tests/unit/utils/test_mdd.py`,
  `tests/unit/analysis/test_frequentist_ranking.py`.
- Meta-analysis: `tests/unit/analysis/test_meta.py`.
- Variance: `tests/unit/analysis/test_variance.py`,
  `tests/unit/analysis/test_variance_branch_closure.py`.
- Head-to-head: `tests/unit/analysis/test_head2head.py`,
  `tests/unit/analysis/test_run_bonferroni_head2head*.py`,
  `tests/unit/analysis/test_h2h_analysis.py`.
- TrueSkill: `tests/unit/analysis/test_run_trueskill_*.py`,
  `tests/unit/analysis/test_analytics_trueskill.py`.
- Agreement: `tests/unit/analysis/test_agreement_payload.py`,
  `tests/unit/analysis/test_agreement_ties.py`.
- Metrics: `tests/unit/analysis/test_metrics*.py`,
  `tests/integration/test_metrics_stage.py`,
  `tests/unit/analysis_light/test_pipeline_stabilizers.py`.
- Stage/path/cache behavior: `tests/unit/analysis/test_stage_registry.py`,
  `tests/unit/analysis/test_stage_runner.py`,
  `tests/unit/analysis/test_stage_state.py`,
  `tests/unit/analysis/test_artifact_contracts.py`.
- Simulation and resume: `tests/unit/simulation/test_run_tournament*.py`,
  `tests/unit/simulation/test_runner*.py`,
  `tests/integration/test_run_tournament_integration.py`.
- Game rules: `tests/unit/game/test_scoring*.py`,
  `tests/unit/game/test_engine*.py`,
  `tests/integration/test_farkle_integration.py`.
- Manifests/atomic streaming: `tests/unit/utils/test_manifest*.py`,
  `tests/unit/utils/test_writer.py`,
  `tests/unit/utils/test_parallel_files.py`.

## Evidence Strength

Stronger evidence areas:

- Wilson interval arithmetic has direct expected-value tests.
- Variance helper arithmetic has explicit small-frame tests.
- MDD helper tests include a weighted-dispersion identity check in the current
  dirty tree.
- Stage state and registry behavior have focused tests.
- Manifest and streaming helpers have many edge-case tests.

Qualified evidence areas:

- Many analysis tests assert private helpers and artifact mechanics. They are
  useful regression tests but not full proof of statistical validity.
- Integration tests often use synthetic fixtures, patches, or goldens.
- TrueSkill tests exercise streaming, resume, and pooling mechanics, but model
  validity remains a separate review question.
- Head-to-head tests cover many branches and errors, but final method validity
  depends on candidate selection, dependence assumptions, and power design.

Known review gaps to check manually:

- Whether every reported uncertainty field has a clearly documented estimator.
- Whether artifact names using `pooled` state the pooling scope.
- Whether game-order-dependent TrueSkill outputs are interpreted correctly.
- Whether head-to-head one-sided simulation p-values and two-sided post-H2H
  decisions are intentionally distinct.
- Whether k-player fair baselines use `1/k` rather than `0.5` when appropriate.

## Failure Notes Caveat

`tests/FAILURE_NOTES.md` is dated `2025-11-19T06:47:10Z`. Treat it as historical
triage, not current truth, until rerunning the suite. It lists old failures in
CLI, metrics, game engine, simulation metrics, and simulation stats. Verify
current behavior before relying on it.

## Recommended Review Workflow

For a statistical subsystem:

1. Read the relevant card in `statistical_methods_map.md`.
2. Read the source files listed in the card.
3. Extract the intended formula or model assumptions into a short note.
4. Map each formula term to code.
5. Build one tiny hand-checkable example.
6. Run the smallest matching tests.
7. Decide whether remaining risk is implementation mismatch, model assumption,
   weak test evidence, or report wording.

For a pipeline/artifact subsystem:

1. Start with `analysis_pipeline_map.md`.
2. Verify active stage path helpers in `config.py`.
3. Inspect `.done.json`, manifest, and atomic-write logic.
4. Run targeted stage-state or artifact-contract tests.
5. Avoid changing legacy fallback behavior unless the task is explicitly about
   migration or cleanup.

## Common Commands

Use the repository venv when running locally.

```powershell
.\.venv\Scripts\python -m pytest tests/unit/utils/test_stats.py tests/test_stats_wilson.py
.\.venv\Scripts\python -m pytest tests/unit/utils/test_mdd.py
.\.venv\Scripts\python -m pytest tests/unit/analysis/test_meta.py
.\.venv\Scripts\python -m pytest tests/unit/analysis/test_h2h_analysis.py
.\.venv\Scripts\python -m pytest tests/unit/analysis/test_run_trueskill_pooling.py
.\.venv\Scripts\python -m pytest tests/unit/analysis/test_stage_state.py
```

For broad checks:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m mypy
```
