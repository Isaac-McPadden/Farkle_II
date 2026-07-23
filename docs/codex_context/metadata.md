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
  observational-unit-labelled game statistics and reports. RNG-scheme-v2
  derives tournament and H2H seat streams directly from complete semantic
  coordinates and makes shuffle indices, rather than scalar fingerprints, the
  recovery identity. Task 5 adds an opt-in version-3 authenticated-contract
  primitive layer for canonical physical paths, actual Arrow schemas, exact
  source/sidecar or immutable-manifest identities, typed method/version/code
  identity, and five-state lifecycle classification. Task 6 migrates simulation
  and shared root/pair completion to schema-4 exact-byte identities, gives every
  root stage a final health-checked stamp, makes HGB and root TrueSkill obey the
  shared lifecycle, separates reloadable public YAML from authenticated runtime
  context/lineage, binds pair lineage to both parent lifecycle roots, and derives
  final health from a last-moment canonical-state audit. Artifact producers that
  still emit v2 sidecars remain ineligible for v3 release evidence; no existing
  bytes or completion stamps are promoted. Task 7 seals each resumable
  TrueSkill root/k cell to exact ordered rows, coordinates, hyperparameters,
  method/code identity, rating bytes, and sidecar bytes; HGB completion binds
  its target/features, whole-strategy folds, hyperparameters, RNG/method/code
  identity, outputs, and sidecars. HGB's row-preserving per-k association union
  now lives in `concat_ks`; only the equal-k association summary lives in
  `across_k`.
- Files checked: `config.py`, `analysis/__init__.py`, `stage_registry.py`,
  `run_contexts.py`, `two_seed_pipeline.py`, canonical analysis modules,
  sidecar/lifecycle utilities, CLI dispatch, and current tests.
- Release commands are recorded in `testing_and_review_map.md`.

Update this file whenever the orientation pack is materially corrected. An
exact commit hash is not a substitute for checking the source relevant to the
next task.
