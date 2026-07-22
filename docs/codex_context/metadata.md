# Codex Context Metadata

The orientation pack is a cache, not authority. Verify relevant source before
changing code or accepting a statistical conclusion.

- Reconciled date: `2026-07-22`
- Source baseline: SCP-11 commit `eee9fbb` plus the natural seed-pair staging rework
- Reconciliation scope: all CSRP/SCP source changes through the release
  closeout, including configuration, scopes, sidecars, lifecycle, RNG,
  estimators, root/pair orchestration, H2H, dominance, reporting, migration,
  release audits, the structural toy oracle, stage-owned pair artifacts,
  non-mutating path resolution, the immutable power/execution-state split,
  provider-neutral artifact I/O retries, and completed-H2H stamp recovery.
  The current artifact contract also covers streamed per-root intermediates and
  recognizes the two canonical descriptive-screening files at their stage root.
  Outcome-schema-v2 and tournament-method-v2 add explicit safety-limit rows,
  all-attempt tournament denominators, completed-only winner products, and
  observational-unit-labelled game statistics and reports.
- Files checked: `config.py`, `analysis/__init__.py`, `stage_registry.py`,
  `run_contexts.py`, `two_seed_pipeline.py`, canonical analysis modules,
  sidecar/lifecycle utilities, CLI dispatch, and current tests.
- Release commands are recorded in `testing_and_review_map.md`.

Update this file whenever the orientation pack is materially corrected. An
exact commit hash is not a substitute for checking the source relevant to the
next task.
