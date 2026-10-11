# Analysis pipeline map

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
6. `rng_diagnostics` (required preflight/monitoring; exhaustive lags optional)
7. `trueskill`
8. `hgb`
9. `screening`

Metrics owns outcome publication, completed-outcome precision and seat behavior
as authenticated children. HGB owns matched parameter responses after held-out
prediction/fold outputs. Reporting owns the JSON/Markdown analysis guide;
required children/units enter parent completion and final audit.

The root then stops. In a two-root run, H2H never executes inside either root.

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
- A valid H2H power plan completes even above the operational cap. Execution
  requires its own authenticated `cap_authorization.json`; terminal approval or
  `--h2h-authorize PLAN_SHA256:MAX_ATTEMPTS` authorizes only that exact plan.
  Refusal preserves progress and stops with `blocked_by_cap` (CLI exit 3).
  Planning replay/checkpoints retain approval; final audit verifies it.
- A root-pair config writes under its dedicated `seed_pair_analysis` directory.
- Every canonical derived artifact requires a compatible adjacent sidecar.
- Ingest and combine source discovery is parent-owned: each root/k is
  authenticated once into an immutable worker snapshot, and partition workers
  never rediscover the global upstream tree.
- A completion stamp is valid only when inputs, output identities, stage hash,
  freshness key, and sidecars validate.
- A completed context may be captured once as an immutable process-local graph
  snapshot for same-process lifecycle/state reuse. Snapshots are never
  persisted or resumed and remain subordinate to one fresh byte-deep
  root/root/pair release audit before pipeline success.
- The lifecycle is `not_started`, `partial_resumable`, `complete_valid`,
  `complete_stale`, or `blocked_by_cap`.

## Public CLI mapping

- `run`, `analyze`, and `two-seed-pipeline` enter through the lightweight OS
  memory launcher before importing the real CLI. Windows uses an inherited Job
  Object; Linux/containers/WSL require delegated cgroup v2 in strict mode.
- `analyze pipeline` and `analyze analytics`: canonical standalone-root
  workflow with labeled H2H tail.
- `two-seed-pipeline`: both root workflows, then one root-pair workflow.
- `analyze ingest|curate|combine|metrics|preprocess`: focused root operations.

Missing canonical upstream artifacts are hard failures. Current workflows do
not resolve old stage names or alternate directories.
