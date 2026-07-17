# Codex Analysis Pipeline Map

Treat this as orientation. Source authority is `analysis/__init__.py`,
`stage_registry.py`, `orchestration/run_contexts.py`, and
`orchestration/two_seed_pipeline.py`.

## Root workflow

`build_root_stage_plan` executes:

1. `ingest`
2. `curate`
3. `combine`
4. `metrics`
5. `game_stats`
6. `rng_diagnostics` when enabled
7. `trueskill`
8. `hgb`
9. `screening`

It then stops. In a two-root run, H2H never executes inside either root.

## Root-pair workflow

`RootPairRunContext` owns `results_seed_pair_X_Y/seed_pair_analysis` and
validated links to the two root layouts. `build_root_pair_stage_plan` executes
once:

1. `root_stability`
2. root/k TrueSkill candidate contribution
3. `candidate_freeze`
4. `h2h_power`
5. `h2h_execute`
6. `h2h_inference`
7. `h2h_digest`
8. `agreement`
9. `reporting`

Standalone analysis runs the root workflow and appends the same H2H tail with
`execution_scope=single_root`.

## Path and state rules

- Stage numbers are resolved at runtime; never hard-code them.
- Stage and scope path lookup is non-mutating. Writers and completion publishers
  create directories only when a stage actually materializes work.
- Use only `by_k`, `concat_ks`, `across_k`, `cross_seed`, `diagnostics`, and
  `h2h_2p`.
- Pair phases own their outputs. H2H execution state and checkpoints never
  modify the immutable power-plan artifact.
- A root-pair config writes under its dedicated `seed_pair_analysis` directory.
- Every canonical derived artifact requires a compatible adjacent sidecar.
- A completion stamp is valid only when inputs, output identities, stage hash,
  freshness key, and sidecars validate.
- The lifecycle is `not_started`, `partial_resumable`, `complete_valid`,
  `complete_stale`, or `blocked_by_cap`.

## Public CLI mapping

- `analyze pipeline` and `analyze analytics`: canonical standalone-root
  workflow with labelled H2H tail.
- `two-seed-pipeline`: both root workflows, then one root-pair workflow.
- `analyze ingest|curate|combine|metrics|preprocess`: focused root operations.

Missing canonical upstream artifacts are hard failures. Current workflows do
not resolve old stage names or alternate directories.
