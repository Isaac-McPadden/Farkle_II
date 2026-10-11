# Codex Analysis Pipeline Map

Task 10 advances ingest cache to 4 and its physical layout/partition method to
3. Adjacent input shard batches coalesce under a quarter of the ingest byte
allowance, with separate input/compaction/writer reserves and bounded chunk
metadata. The canonical output still preserves every source row in order;
downstream curate retains its copy boundary. No stage order, row schema or
statistical method changes. The shared ingest identity authenticates the new
physical layout; stale units are rebuilt rather than re-sidecarred. See
[Task 10](../implementation/task10_report.md).

Task 8 extends the existing reporting owner with authenticated `analysis_guide.json` and `analysis_guide.md` in diagnostics. The executable plan and completion require both; Markdown binds only the JSON as its source. Reporting contract is 9, guide method is 1; metrics cache 6 and HGB cache 2 remain. Missing required guide inputs fail publication. Final audit checks question/reference coverage, JSON-to-Markdown consistency and completion ownership. See [Task 8](../implementation/task8_report.md).

Task 7 runs `seat_behavior` inside metrics after certified outcome support,
then publishes equal-k opportunity within each root. `parameter_responses`
runs inside HGB after held-out prediction/fold outputs. No new top-level stage
or canonical scope is added. Their required artifacts and retained behavior
units enter parent completions, the graph snapshot, and final byte-deep audit.
Reporting consumes each root separately for these products.

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
