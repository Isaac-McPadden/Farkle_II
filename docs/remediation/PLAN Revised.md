# Revised fast-pipeline observability and performance plan

## Summary and evidence baseline

This plan remains grounded in:

- Git HEAD: `fb083fa462f8ea261c230a6061b5e0bc1d7c17b0`
- Commit: `Fixed RNG collision bug`
- Worktree: clean, `main` aligned with `origin/main`
- Inspected artifacts: `data/results_guardrails_cleanup_1-4_fast_seed_pair_48_49`
- Roots: 48 and 49
- Artifact identity tuple: `[3, 2, 2, 2, 2, 2]`
- RNG identity: method v4, cache key v6
- H2H method identity: v2
- Outcome identity: v2
- Completed runtime: 13h 32m 31s

Evidence labels:

- **Measured:** logs, file metadata, Parquet footers, manifests, sidecars, or completion records.
- **Derived:** deterministic counts from executable code and artifact cardinalities.
- **Inferred:** likely resource behavior not directly recorded.

Primary conclusions are unchanged:

1. RNG diagnostics consumed 6h 43m 57s primarily because of fine-grained routing, enormous file/open/spill counts, external merging, Python key handling, and repeated authentication—not because lag-1 accumulation itself was large.
2. H2H execution consumed 4h 17m 33s primarily because 77 candidates generated 2,926 pairs, 11,704 blocks, and 23.18 million attempts under production-like inferential targets.
3. The current fast configuration is a reduced production run mixed with integration settings, not a coherent fast integration profile.
4. The stable-matchup collision repair did not widen identifiers, records, or sort keys and did not change asymptotic complexity.
5. Observability must precede optimization. RNG coarsening is the highest-value code optimization; explicit integration sizing is the highest-value overall runtime reduction.
6. Production readiness remains blocked until a production-shaped bounded benchmark yields a defensible runtime within a human-approved budget.

## Repository identity and output-root rules

The current authentication implementation binds stage state to the complete repository code revision. Therefore:

- Any committed source change creates a new repository identity, even when statistical methods, schemas, configuration SHA, canonical layout, and canonical values remain unchanged.
- The seed-48/49 tree remains valid as immutable historical evidence.
- That tree is not resumable under a new source-changing HEAD.
- No task should backfill, reseal, reinterpret, or attempt to salvage old stage state under a new HEAD.
- Focused fixtures and disposable smoke roots should be used between tasks.
- One fresh end-to-end integration root pair should be run only after all accepted source-changing work is committed, the worktree is clean, and the review gates have passed.
- “No statistical identity change” and “no canonical layout change” do not imply resumability under the new HEAD.

Any future proposal to narrow code identity to stage-specific dependency closures is a separate authentication-policy change requiring human approval.

## Actual executable pipeline and artifact graph

### Artifact and lifecycle conventions

`C` = canonical data; `P` = resumable partition/checkpoint state; `M` = manifest; `S` = authenticated sidecar; `D` = completion stamp; `T` = temporary spill; `R` = human-facing report.

V3 analysis freshness generally binds:

- Stage-specific configuration slices
- Authenticated input identities
- Run lineage
- Method/schema versions
- Complete repository code revision

Publication is atomic and authenticated. Partitioned stages can resume only when unit state matches the current repository/configuration/source identity.

### Per-root pipeline, repeated independently for seeds 48 and 49

| Stage/subphase | Principal implementation | Inputs → outputs and consumers | Units/workers | Completion and resume |
|---|---|---|---|---|
| Simulation | `two_seed_pipeline._run_one_seed` → tournament runner/game engine | Config, strategy grid, explicit RNG coordinates → workload plan, strategy manifest, 12,900 shuffle Parquets/S, checkpoints/P, immutable M, simulation D; consumed by ingest | 4,300 shuffles for each of k=2/4/5; requested/effective 12 | Shuffle-level resume under matching identity; atomic manifest/publication |
| Ingest | `ingest.run`, `_run_partitioned_ingest` | Simulation M and shards → three per-k raw Parquets, stream M, S/D; consumed by curate | Per k; requested/effective 1 | Manifest is both inventory and work queue |
| Curate | Curate stage action | Three raw Parquets → three canonical curated game-row Parquets, schema metadata, S/D | Per k; serial | Reflink/copy where possible; no statistical transformation |
| Combine | `combine.run` | Curated per-k files → three normalized/padded combined Parquets plus logical `concat_ks` M | Per k; requested 4, effective 3 | Real normalized rewrites; `concat_ks` is an inventory, not a duplicate monolithic Parquet |
| Metrics: all-player | `metrics.run` → `build_all_player_batch_metrics` | Combined/curated rows → per-k batch metrics and stream M | Per k; requested 4, effective 3 | Per-k resumable units |
| Metrics: performance/bootstrap | `build_canonical_performance` | Batch metrics → matrices, per-k/across-k tables, 2,000-replicate bootstrap ranges, contrasts/effects | Bootstrap ranges; requested/effective 4 | Range/P state and final aggregation |
| Metrics: seat analysis | `build_canonical_seat_analysis` | Curated/combined rows → seat counts/effects and diagnostics | Per k/range; up to 4 | Range/P state and canonical outputs |
| Game statistics | `game_stats.run` and rare-event builders | Curated rows and exact roll enumeration → game/turn/score/rare-event artifacts | Per k and rare-event shards; requested 4, effective 3 for k fan-out | Ordinary and rare-event paths traverse sources independently |
| RNG count route | `rng_diagnostics.run`, `_extract_batch_arrays`, `_matchup_ids`, partitioned-stage runner | Three combined Parquets → 12,900 Arrow IPC route files, stamps, M | One source row group/unit, 32 batches/file; requested/effective 4 | Unit output is fsynced, hashed, stamped, immediately validated, and revalidated during manifest publication |
| RNG count reduce | Partition reducers, `_collapse_count_runs` | Count-route files → local `%TEMP%` runs/merges → 32 eligibility Parquets | 32 reducers; requested/effective 4 | Every reducer opens all 12,900 route files; three merge passes |
| RNG selection | Eligibility selector | Eligibility files → membership Parquet/report | Serial | Two scans of the 32 eligibility files |
| RNG statistics route | Second route | Combined rows and selected membership → another 12,900 IPC files/stamps/M | One row group/unit; requested/effective 4 | Repeats source decoding, canonicalization, hashing, routing, publication, and authentication |
| RNG statistics reduce | Stats reducers/lag accumulators | Stats-route files → temporary runs/merges → 32 result partitions → final diagnostic C/R/S/D | 32 reducers; requested/effective 4; lag `[1]` | External sort precedes deterministic group/lag accumulation |
| TrueSkill ratings | `run_trueskill_root` → `run_trueskill` | Curated per-k rows → three root/k rating cells | Per k; requested 4, effective 3 | Ordered within each k; k cells resume independently |
| TrueSkill screening | Screening-rating builders | Ratings/source rows → percentile and reverse-order artifacts | Root/k; up to 3 | Uses local reverse spool and authenticated publication |
| HGB | HGB analysis stage | Performance/strategy artifacts → 13 per-k fold/model artifacts | Serial at this scale | Standard v3 completion |
| Screening | Screening stage | Across-k performance/bootstrap → two selection/diagnostic outputs | Serial | Standard v3 completion |
| Root lifecycle sealing | `_root_lifecycle_identity` | Root stage graph → lifecycle SHA and states | Serial | Reauthenticates already immutable upstream state |

### Pair-level pipeline

| Stage/subphase | Principal implementation | Inputs → outputs | Units/workers | Completion and resume |
|---|---|---|---|---|
| Root stability | `build_two_root_stability` | Both roots’ matrices → stability/discrepancy/rank/convergence/bootstrap artifacts | Per k and ranges; up to 4 | Range manifests and 13 canonical outputs |
| Pair TrueSkill contribution | Pair contribution builder | Six root/k ratings → pair contribution ranking | Serial | One canonical output |
| Candidate freeze | Candidate-family builder | Win-rate/TrueSkill rankings → membership Parquet and JSON | Serial | Frozen candidate identity feeds all H2H stages |
| H2H power | Exact power planner | Candidate family and inferential settings → power plan | Candidate pair; serial | Exact first-crossing allocation |
| H2H schedule | Schedule builder | Power plan, roots, seat orders → immutable block schedule | Pair × root × order | 2,926 pairs, 11,704 blocks |
| H2H execution | `h2h_schedule` workers/game engine | Schedule, strategies, RNG coordinates, prior matching block state → block Parquet/S, aggregate counts, execution state, D | Requested/effective 15; window 15 | Block resume; each checkpoint rewrites its block Parquet and sidecar |
| H2H inference | Inference builder | Aggregate counts, schedule, power → four inference artifacts | Serial/vectorized | Authenticated publication |
| Digest/dominance | Digest builder | Inference and screening/stability → four digest/graph outputs | Serial | Authenticated publication |
| Agreement | Agreement builder | Candidate, inference, stability/digest → two outputs | Serial | Authenticated publication |
| Reporting | Pair reporting builders | Pair/root artifacts → JSON, Markdown, PNG, migration/report outputs | Serial | Four canonical/report outputs |
| Pair state recheck | `_current_plan_states` | Pair stage graph | Serial | Repeats deep stage-state authentication after the logged stage completion |
| Root lifecycle rechecks | `_root_lifecycle_identity` twice | Both root graphs | Serial | Repeats root authentication |
| Final release gate | `_final_release_gate`, `audit_sidecar_completeness` | Three run contexts and analysis trees → audit result | Serial | Reloads contexts and conducts three byte-deep graph audits |
| Health publication | `_write_pipeline_health` | Lifecycle/audit state → health metadata/run-end record | Serial | Atomic final publication |

### Repeated work

- Ingest consolidates simulation shards; curate copies/reflinks them; combine decodes and re-encodes them into normalized schema.
- Combined/curated sources are independently traversed by metrics, game statistics, RNG, and TrueSkill.
- RNG repeats source extraction, canonical matchup hashing, routing, sorting, and authentication in its count and statistics passes.
- Every RNG partition reducer opens every route file.
- H2H blocks are rewritten per checkpoint and then hashed during multiple completion/lifecycle/audit passes.
- Stage outputs may be hashed during sidecar construction, provisional completion validation, final completion sealing, StageRunner resolution, lifecycle resolution, and final release audit.
- The pair workflow’s “Stage run complete” precedes approximately 23 minutes of additional authenticated-state processing.
- Repeated `seed_list overrides seed` warnings correspond to persisted configuration reloads during final release validation.
- Whole-repository identity is broader than most stages’ actual implementation dependency. This is conservative policy, distinct from redundant work performed inside that policy.

## Empirical runtime reconstruction

### Root timelines

| Operation | Seed 48 | Seed 49 | Evidence-supported classification |
|---|---:|---:|---|
| Simulation | 17m 41s | 17m 26s | 326,800 games and 1.032m player rows/root; CPU-heavy plus publication |
| Ingest | 7m 56s action; 8m 03s stage | 7m 15s action; 7m 23s stage | Single-worker filesystem/serialization |
| Curate | 27s action; 37s stage | 30s action; 41s stage | Copy/reflink and metadata |
| Combine | 4m 05s action; 4m 25s stage | 3m 54s action; 4m 13s stage | Approximately 379.4 MB/root; Parquet decode/encode and filesystem |
| Metrics | 6m 23s | 5m 56s | Mixed scanning, bootstrap CPU, serialization, authentication |
| Game statistics | 12m 05s | 12m 29s | Repeated scans and CPU aggregation |
| RNG diagnostics | 3h 09m 51s | 3h 34m 06s | Metadata/file operations, external merge, Python keys/sorting, authentication |
| TrueSkill | 5m 39s | 6m 00s | Ordered CPU processing and source scanning |
| HGB | 11s | 12s | Negligible |
| Screening | about 1s | about 1s | Negligible |
| Root lifecycle work | about 3m 04s | about 3m 15s | Deep validation/hashing |
| Entire root | 4h 04m 57s | 4h 28m 29s | Seed-49 excess is almost entirely RNG |

This was a clean run: no resumed units, retry, worker downshift, or recorded memory pause.

### RNG evidence

Each root contains:

- 326,800 games
- 1.032m player exposures
- 12,900 Parquet row groups
- About 379.4 MB of combined source data

| Phase | Seed 48 | Seed 49 | Work |
|---|---:|---:|---|
| Count route | 84m 29s | 95m 40s | 12,900 units, four workers |
| Eligibility reduction | 8m 46s | 10m 03s | 32 reducers, three merge passes |
| Selection | under 1s | under 1s | Two scans of 32 files |
| Statistics route | 87m 14s | 98m 24s | Second 12,900-unit source route |
| Statistics reduction | 9m 21s | 9m 58s | 32 reducers, three merge passes |
| Final construction | under 1s before stage authentication | under 1s | Final concatenation/reports |

Per seed:

- 25,800 route files and 25,800 route stamps
- About 1.119 GB of route data, approximately 22 MB of stamps, and approximately 8 MB of manifests
- 412,800 initial route-batch opens per reduction
- Approximately 426,176 local run/merge files per reduction where partition batches are nonempty
- Approximately 852,352 spill/merge files across both reductions
- Spill volume unknown because `%TEMP%` was cleaned and no byte telemetry exists

Group counts were nearly identical:

| Cardinality | Seed 48 | Seed 49 |
|---|---:|---:|
| Candidate groups | 155,848 | 155,717 |
| Matchup groups | 155,608 | 155,477 |
| Eligible groups | 3,444 | 3,446 |
| Eligible matchup groups | 3,204 | 3,206 |
| Below minimum | 152,404 | 152,271 |
| Capped groups | 0 | 0 |

The 150,000-group cap does not reduce execution because only about 3,200 matchup groups survive eligibility, and the expensive count path must complete before that is known.

The collision repair retained the existing 64-bit digest, full canonical player-coordinate fields, record width, sort width, and merge depth. It made the complete canonical key authoritative and repaired resource cleanup. It does not explain the runtime.

### Seed-49 RNG difference

Seed 49’s additional time was concentrated in:

- Count route: approximately 11m 11s slower
- Statistics route: approximately 11m 10s slower
- Reductions combined: approximately 1m 14s slower

Rows, bytes, group counts, merge passes, and workers were effectively identical. No memory pause or retry was recorded. The evidence therefore rules out meaningful workload-cardinality differences but cannot distinguish:

- OneDrive/provider latency
- Local filesystem/cache behavior
- Native allocator or retained parent-process state
- Thermal or CPU-frequency variance
- Incidental system variance

Sampled RSS increased by approximately 418 MB for the second root but remained below warning thresholds. This is suggestive, not sufficient evidence for a subprocess architecture.

### H2H evidence

Candidate freeze selected 77 of 80 strategies:

- 73 shared by both methods
- Two TrueSkill-only
- Two win-rate-only
- No cap

This produced:

- 2,926 candidate pairs
- Two roots × two seat orders
- 11,704 blocks
- 1,974 required completed games per block
- 23,103,696 initially planned completed games
- 23,182,656 actual attempts
- 23,024,736 completed games
- 157,920 safety attempts
- 78,960 replacements

Of the blocks:

- 11,664 completed exactly 1,974 games.
- 40 nonviable blocks reached 3,948 attempts with no completed games.

With a 1,000-game chunk size, all blocks were checkpointed after 1,000 and 1,974 attempts, while the 40 exceptional blocks required two additional chunks. Approximately 23,488 block publication events rewrote a Parquet and sidecar.

Measured/inferred decomposition:

- Approximately 3h 45m 36s to the latest final block output
- Approximately 3m 30s for aggregate counts
- Approximately 3m 51s for the aggregate sidecar
- Approximately 1m 44s for execution state
- Approximately 12m 44s for completion publication/authentication
- Approximately 10m 08s for StageRunner revalidation

The global rate was about 1,712 attempts/s, equivalent to roughly 114 attempts/s/worker if 15 workers were equally occupied. The bulk schedule can occupy 15 workers; only the 40-block exceptional tail has substantially fewer tasks. Actual CPU utilization and worker-idle time were not recorded.

### Final unlabelled 23 minutes

After pair StageRunner completion, code executes:

1. Pair `_current_plan_states`
2. Root-48 `_root_lifecycle_identity`
3. Root-49 `_root_lifecycle_identity`
4. Three run-context reloads/authentications
5. Three `audit_sidecar_completeness` graph audits
6. Final memory check
7. Pipeline-health/run-end publication

Approximately 19m 09s elapsed before the first configuration-reload warning. The three warnings then align with root-48, root-49, and pair context loading. Root audits took approximately 35–36 seconds each; pair audit and final publication took approximately 2m 49s. Exact attribution within the first 19 minutes is unavailable because no named timers exist.

## Configuration taxonomy

Do not automatically create three new overlapping configuration families.

Map purposes onto existing repository configuration where practical:

- Use an existing minimal/test configuration for smoke coverage. Add only a thin smoke override if no current file can express the required branches.
- Treat the existing fast configuration as the candidate integration profile after human approval.
- Keep the existing default/mega production configuration as the production authority rather than duplicating it.
- Give each resolved run an explicit purpose label: `smoke`, `integration`, or `production`.

Purposes:

| Purpose | Required coverage | Runtime/statistical role |
|---|---|---|
| Smoke | Basic two-root orchestration, schemas, method branches, publication, and a small H2H path | Quick, non-inferential correctness |
| Integration | Real multiprocessing, two roots, all principal methods, deterministic coordinates, interruption/resume fixtures, cross-seed and pair behavior | Reduced-resolution, non-release development evidence |
| Production | Official strategy/k coverage and inferential targets | Release-grade statistical artifacts |

Proposed integration settings, pending approval:

- Approximately 600 shuffles/k through a 0.08 resolution target
- 500 bootstrap replicates
- Eight RNG partitions
- Top-eight contribution selection
- Explicit frozen-candidate cap of 12
- Existing two roots and k=2/4/5
- Existing H2H method and seat/root dimensions

Reports, run-context metadata, and health summaries must state that this profile is reduced-resolution, non-release, and non-production. They must expose the achieved resolution, bootstrap count, candidate cap, pair count, and planned H2H games.

Data-dependent exceptional behavior must not be forced into integration data merely to exercise it. Nonviable blocks, replacement attempts, retry/downshift, collision handling, and long-tail behavior require deterministic focused fixtures.

Candidate-count effects remain:

| Candidates | Pairs | Games/root/order | Planned completed games |
|---:|---:|---:|---:|
| 6 | 15 | 1,132 | 67,920 |
| 8 | 28 | 1,234 | 138,208 |
| 12 | 66 | 1,372 | 362,208 |
| 20 | 190 | 1,547 | 1,175,720 |
| 77 | 2,926 | 1,974 | 23,103,696 |
| 150 | 11,175 | 2,191 | 97,937,700 |

## Observability design

### Shared heartbeat contract

Emit structured progress to the normal log stream every 45 seconds, immediately on state changes, and once at completion. Heartbeats must not create additional canonical or OneDrive checkpoint artifacts.

Common fields:

- Root/pair, stage, phase, state
- UTC and monotonic elapsed time
- Completed/total units
- Active/pending units and oldest-active-unit age
- Records/games/bytes read and written
- Rolling/global rate and ETA when meaningful
- Requested/effective workers and process-pool generation
- Process-tree RSS, Job commit current/peak, host-available memory
- Memory-pause, retry, and downshift state
- Last durable checkpoint identifier/time
- Time since the last completed unit

States:

- `working`
- `waiting_on_futures`
- `paused_for_memory`
- `serial_sort_or_merge`
- `publishing`
- `authenticating`
- `cleaning_up`
- `stalled_suspected`

A suspected stall requires at least two missed progress intervals with no counter movement and no known long-running operation.

### Specialized RNG fields

- Phase: snapshot, count route, count reduce, selection, stats route, stats reduce, output build, authentication, cleanup, validation
- Source row groups/games scanned
- Route units complete/total
- Candidate, eligible, selected, skipped, and capped groups
- Spill runs and bytes
- Merge pass and runs created/remaining
- Partitions and lags complete
- Local-temp path/device classification
- Temporary and durable read/write bytes
- Route/checkpoint file counts and hashing time

### Specialized H2H fields

- Candidate and pair count
- Blocks active/pending/checkpointed/terminal
- Attempts, completed games, replacements, and safety attempts
- Root/order/replicate coordinates
- Current chunk/pass and pool generation
- Rolling/global game rate
- Long-tail block count and oldest block
- Parent checkpoint-write queue time
- Checkpoint count/bytes
- Resume-discovery and authentication progress

### Named finalization phases

- `pair_state_recheck`
- `root_48_lifecycle_recheck`
- `root_49_lifecycle_recheck`
- `run_context_authenticate`
- `root_48_graph_audit`
- `root_49_graph_audit`
- `pair_graph_audit`
- `pipeline_health_publish`

One completion summary should retain action, publication, authentication, cleanup, unit, byte, worker, retry, and peak-memory measurements in pipeline-health metadata. Operational telemetry must not enter statistical freshness or canonical artifact identity.

## Performance and investigation work

### A. Observability first

Implement the shared foundation separately from specialized RNG/H2H counters. Neither task changes statistical methods or canonical layout, but both change repository identity and therefore prevent seed-48/49 state from being resumed under the new HEAD.

### B. Approved integration sizing

Apply the approved integration settings to the existing fast configuration family, using disposable roots. This changes statistical configuration SHA and workload identity. It does not redefine production claims.

Expected runtime under current algorithms: approximately 90–120 minutes.

### C. Bounded storage-location experiment

Before attributing RNG or H2H metadata cost exclusively to the algorithms, run identical deterministic bounded workloads against:

1. A disposable directory under the current OneDrive-backed tree.
2. A disposable local nonsynchronized filesystem directory.

Use the same source fixture, RNG seed/coordinates, record counts, route partitions, merge fan-in, H2H blocks, checkpoint cadence, and authentication calls.

Measure separately:

- Source-read bytes and elapsed time
- File create/open/close counts and latency
- Sidecar publication count/bytes/time
- Hashing bytes/time
- Spill-run creation and merge throughput
- Checkpoint rewrite throughput
- Cleanup time and failures
- CPU time, wall time, RSS/commit, and host-available memory

Do not run the full pipeline. Delete only the explicitly created disposable benchmark directories after validating their resolved paths.

Decision rule:

- If synchronized-tree overhead is material and repeatable, design a configurable local working/checkpoint root whose final canonical publication still targets the configured output root.
- Do not move canonical artifacts, weaken authentication, or alter canonical paths without a separate artifact-flow, failure-recovery, and migration design.
- Local work state must remain deterministic, bounded, resumable under its current repository identity, and atomically published to the canonical root.

### D. RNG route-unit coarsening

Replace one-row-group durable units with deterministic contiguous row-group or byte-bounded ranges.

Expected benefit:

- Reduce 25,800 route files and 51,600 stamps per seed by approximately the coarsening factor.
- Reduce reducer file opens and initial spill runs by the same order.
- Preserve bounded memory and useful resume granularity.
- Target 25–45 minutes RNG/root at the current workload and 5–12 minutes/root at the proposed integration size, subject to telemetry and the storage A/B result.

Required equivalence:

- Exact integer counts
- Exact canonical keys and stable identifiers
- Exact membership and eligibility decisions
- Exact RNG/game/player coordinate coverage
- Exact schemas
- Exact deterministic row ordering
- Deterministic canonical floating-point results wherever the existing algorithm guarantees them

Do not assume final-file byte equality before proving arithmetic order is preserved. If different merge grouping changes floating-point accumulation order:

1. Measure and report the exact differing fields and values.
2. Determine whether a deterministic reduction order can retain existing values.
3. Do not accept a tolerance silently.
4. If exact preservation is infeasible, specify the statistical consequence and required RNG method/cache/artifact-version change for human approval before implementation is accepted.

Operational checkpoint layout may require a new cache/layout version even if the statistical method remains v4.

### E. H2H checkpoint efficiency

Tune checkpoint size independently from authentication work:

- Choose chunks aligned with the exact target when this remains within a human-approved 30–60-second recovery window.
- Avoid unnecessary process-pool generations.
- Preserve exact coordinate coverage, root/order balance, replacement behavior, deterministic block state, and resume semantics.
- Test nonviable and replacement branches with deterministic fixtures.

This is expected to reduce checkpoint publications and pool startup overhead but will not remove the dominant cost of a 23-million-game schedule.

### F. Authenticated graph/final-audit deduplication

Consider only after telemetry quantifies action versus repeated-authentication time.

Design:

- Build one immutable authenticated graph snapshot per completed run context.
- Reuse that snapshot for same-process stage-state/lifecycle queries.
- Remove redundant validation passes proven to authenticate the same unchanged bytes.
- Retain exactly one explicit byte-deep release audit before success.
- Ensure interruption never leaves a snapshot that can authenticate incomplete publication.

This is higher-risk authentication work. It must have independent tamper, interruption, stale-input, and changed-source testing. If the snapshot becomes a persisted canonical object or changes accepted authentication semantics, require an artifact-contract identity/version review.

### G. Conditional process isolation

Do not automatically introduce subprocess isolation across the pipeline.

Trigger a design task only if:

- Task 1 telemetry shows material RSS/commit retention across completed stages;
- a fresh-process A/B demonstrates meaningful recovery or speed benefit; or
- another repeatable host-memory/resource failure occurs.

If triggered, first isolate only the implicated stage boundary—likely RNG or H2H—rather than creating a generic architecture. Verify Job-limit inheritance, worker shutdown, exact outputs, failure propagation, and resume behavior. `gc.collect()` is not an architectural remedy unless telemetry specifically shows unreachable Python objects are responsible.

## Identity and output-root matrix

| Task | Statistical/method identity | Canonical layout | Seed-48/49 historical evidence | Resume seed-48/49 under new HEAD | Fresh root requirement |
|---|---|---|---|---|---|
| 1A shared observability | No intended change | No intended change | Remains valid read-only evidence | **No** after committed source change | Disposable fixtures during work; one fresh integrated pair after accepted source work |
| 1B specialized progress | No intended change | No intended change | Remains valid read-only evidence | **No** after committed source change | Same as 1A |
| 2 integration profile | Changes statistical configuration/workload SHA, not method implementation | No intended layout change | Remains historical evidence of the old profile | **No** | Fresh disposable smoke/integration roots; later fresh integrated pair |
| 3A storage A/B | No method change if performed with existing executable behavior | No canonical change; benchmark state is disposable | Unchanged/read-only | Not applicable; old tree must not be used as work state | Disposable synchronized and local benchmark roots only |
| 3B RNG coarsening | No intended method change unless floating-point order cannot be preserved | Canonical layout intended unchanged; checkpoint layout may change | Remains historical evidence | **No** after source change | Disposable RNG fixtures/roots; later fresh integrated pair |
| 4A H2H checkpoint efficiency | No intended method/coordinate change | Canonical layout intended unchanged | Remains historical evidence | **No** after source change | Disposable H2H roots; later fresh integrated pair |
| 4B graph/audit deduplication | No statistical method change; authentication contract review required | Intended unchanged unless snapshot is persisted canonically | Remains historical evidence | **No** after source change | Disposable authenticated graphs; later fresh integrated pair |
| 5A production-shaped benchmark | No statistical change when using bounded synthetic/noncanonical fixtures | No canonical production layout change | Remains historical evidence | Not applicable | Disposable benchmark roots |
| 5B conditional isolation | No intended statistical/method change | No intended canonical layout change | Remains historical evidence | **No** after source change | Disposable stage roots; later fresh integrated pair if accepted |

## Implementation tasks and review gates

### Task 1A — Shared observability foundation

- Type: observability-only.
- Scope: StageRunner action/publication/authentication/cleanup timings; shared heartbeat recorder; memory/resource state; named root/pair finalization phases.
- Excludes: RNG/H2H algorithm changes and specialized counters.
- Acceptance:
  - Fake-clock cadence and no-log-flood tests.
  - Explicit state transitions and final summaries.
  - Finalization phases appear by name.
  - Operational fields do not affect freshness.
  - Canonical outputs are byte-identical on bounded fixtures.
- Identity/root handling: source change; no statistical or canonical-layout change; old artifacts remain historical but cannot resume; use disposable fixtures.
- Benefit: required foundation.

**Gate:** Implement, review, and accept Task 1A before Task 1B.

### Task 1B — RNG and H2H specialized progress

- Type: observability-only.
- Scope: RNG route/reduce/spill/merge counters; H2H pair/block/game/checkpoint counters; rate, ETA, long-tail, and pool-generation reporting.
- Acceptance:
  - Tiny deterministic RNG and H2H fixtures cover all phases.
  - Memory pause, retry/downshift, long-tail, and nonviable-block states have deterministic tests.
  - Canonical fixture outputs are byte-identical with telemetry enabled/disabled.
  - Telemetry overhead remains below 1% in bounded benchmarks.
- Identity/root handling: source change; no statistical or canonical-layout change; old artifacts cannot resume under the new HEAD.
- Dependency: Task 1A.

**Gate:** Review and accept Task 1B before changing profile or algorithms.

### Task 2 — Existing-config taxonomy and integration profile

- Type: configuration-only/statistical-scope change.
- Scope: map smoke/integration/production purposes onto existing configs; apply only approved integration settings; add reduced-resolution/non-release labels.
- Acceptance:
  - Resolved-config tests assert exact shuffle, bootstrap, RNG partition, contribution, candidate, pair, and H2H-game limits.
  - Reports expose achieved resolution and non-production status.
  - Exceptional H2H branches remain covered in focused fixtures.
- Identity/root handling: changes config/workload identity; fresh disposable roots required; old artifacts remain historical and are not resumed.
- Benefit: projected 13h32 → 90–120m.
- Dependency: Tasks 1A/1B and human approval.

**Gate:** Obtain human approval for the integration settings before modifying the config. Review with disposable smoke/integration roots.

### Task 3A — Bounded storage-location A/B

- Type: diagnostic benchmark.
- Scope: identical deterministic RNG/H2H publication workloads on OneDrive-backed and local nonsynchronized disposable roots.
- Acceptance:
  - Report operation counts, bytes, phase timings, CPU/wall time, memory, hashing, merge, sidecar, checkpoint, and cleanup measurements.
  - Repeat sufficiently to distinguish stable provider overhead from incidental variance.
  - Make no canonical-root recommendation without measured effect size.
- Identity/root handling: no production artifact changes; existing tree read-only; benchmark roots disposable.
- Dependency: Task 1B.

**Gate:** Review results before introducing a local working-root design or attributing metadata cost to OneDrive.

### Task 3B — RNG route-unit coarsening

- Type: computational/checkpoint-I/O.
- Scope: deterministic contiguous ranges, reducer input changes, checkpoint-layout versioning, bounded benchmark.
- Acceptance:
  - Exact integer/key/identifier/membership/eligibility/coordinate/schema/order equivalence.
  - Deterministic floating-point equivalence analysis as specified above.
  - Exact interruption/resume behavior under the new HEAD.
  - Collision fixtures remain protected.
  - Bounded-memory and 256–1,024-row-group benchmarks demonstrate material file/open/spill reduction.
- Identity/root handling: source change; old tree not resumable; canonical layout intended unchanged; method/cache review required if arithmetic values change.
- Benefit: highest-value code optimization.
- Dependency: Tasks 1B and 3A.

**Gate:** Implement and benchmark this task independently. Do not proceed on a tolerance-based floating-point result without human approval.

### Task 4A — H2H execution/checkpoint efficiency

- Type: computational/I/O without intended statistical change.
- Scope: target-aligned checkpoint sizing, process-pool generation behavior, coordinate/resume equivalence.
- Acceptance:
  - Exact attempted/completed coordinate coverage.
  - Exact root/order balance and aggregate integer counts.
  - Interruption at each checkpoint boundary resumes without duplication.
  - Nonviable/replacement fixtures remain deterministic.
  - Bounded benchmark reports simulation versus checkpoint/pool time.
- Identity/root handling: source change; old H2H state not resumable under the new HEAD; canonical layout intended unchanged.
- Dependency: Task 1B; independent of Task 4B.

**Gate:** Review H2H optimization separately from authentication changes.

### Task 4B — Authenticated graph and final-audit deduplication

- Type: higher-risk authentication work.
- Scope: one immutable graph snapshot, removal of redundant passes, exactly one explicit byte-deep release audit.
- Acceptance:
  - Independent tamper tests for data, sidecar, completion, manifest, and run-context changes.
  - Interruption cannot publish a falsely complete snapshot.
  - Stale input or code identity is rejected.
  - Exactly one final byte-deep audit is observable.
  - Canonical output set/layout remains unchanged unless separately approved and versioned.
- Identity/root handling: source change; old artifacts remain historical but cannot resume; authentication-policy/version review may be required.
- Dependency: Task 1 telemetry must show sufficient repeated-authentication cost to justify the risk.

**Gate:** Do not schedule automatically after Task 4A. Obtain review of telemetry and authentication design first.

### Task 5A — Production-shaped bounded benchmark and readiness gate

- Type: diagnostic/capacity qualification.
- Scope: benchmark production row-group width, record counts sampled at bounded scale, four-pass merge structure, expected candidate counts, and publication/authentication costs.
- Acceptance:
  - Fit separate coefficients for source-unit overhead, record processing, merge passes, checkpoint/publication, and hashing.
  - Produce a reproducible projection for simulation, RNG, H2H, other analyses, and finalization.
  - State uncertainty and hardware/storage assumptions.
  - Compare the projection to a human-approved runtime budget.
- Identity/root handling: disposable noncanonical benchmark roots; no reuse of seed-48/49 work state.
- Dependency: accepted Tasks 1B, 3B, and 4A; include 4B only if separately approved.

### Task 5B — Conditional stage isolation

- Type: conditional resource/process-lifetime work.
- Trigger: telemetry-supported cross-stage retention or a repeated resource failure.
- Scope: isolate only the implicated stage boundary first.
- Acceptance:
  - Fresh-process A/B shows a material memory or runtime benefit.
  - Job/native-thread limits propagate.
  - Outputs, coordinates, and failure/resume behavior remain equivalent.
  - Child and worker processes terminate cleanly.
- Identity/root handling: source change; old artifacts cannot resume; no intended method/layout change.
- Dependency: Task 1 telemetry and trigger evidence.

## Execution gates

Do not implement the tasks continuously or run a full pair after each commit.

1. Implement and review Task 1A.
2. Implement and review Task 1B.
3. Obtain human approval for integration-profile settings.
4. Implement the profile and verify it with disposable smoke/integration roots.
5. Run and review the bounded storage-location A/B.
6. Implement and bounded-benchmark RNG coarsening.
7. Implement and review H2H checkpoint optimization separately.
8. Consider authenticated-graph deduplication only if telemetry quantifies a worthwhile repeated-authentication cost.
9. Consider subprocess isolation only if telemetry or a repeated resource failure supports it.
10. Commit all accepted source/configuration changes and require a clean worktree.
11. Run one fresh end-to-end integration root pair under the final repository identity.
12. Run the production-shaped bounded benchmark and apply the production-readiness stop gate.

## Runtime projections

| Cumulative state | Projected integration runtime | Confidence |
|---|---:|---|
| Current completed pipeline | 13h 32m | Measured |
| Task 1A/1B only | 13h 32m–13h 40m | High; telemetry overhead target below 1% |
| Approved integration sizing | 90–120m | Medium |
| Plus RNG coarsening | 55–90m | Medium-low until benchmarked |
| Plus H2H checkpoint tuning | 50–85m | Medium-low |
| Plus justified authentication deduplication | 45–75m | Conditional |
| Conditional subprocess isolation | No promised speedup | Intended for demonstrated retention/reliability |

Production dimensions remain:

- 39,013,900 games/root
- 177,504,000 player exposures/root
- 34,400 row groups/root
- Approximately 216.5m RNG count-route records/root
- Four RNG merge passes
- Up to 150 candidates and approximately 97.94m planned H2H completed games

The existing architecture’s production projection remains a broad multiple-day range because source-unit overhead scales about 2.67× while record work scales roughly 159–172× and merge depth increases. The provisional post-improvement 5–12-day range is a planning warning, not evidence of production readiness.

## Production-readiness stop gate

Do not declare the pipeline production-ready until Task 5A provides a defensible projection within a human-approved unattended/overnight-or-away runtime budget.

If the projection remains multiple days, stop and identify the dominant remaining dimensions:

- Game-engine simulation volume
- RNG record routing/hash/sort/merge throughput
- Production strategy/player exposure count
- H2H candidate-pair/power allocation
- Analysis source rescans
- Authentication/publication volume
- Storage-provider latency
- Available CPU, RAM, and local storage bandwidth

The resulting decision must explicitly choose among:

- Additional algorithmic/I/O work
- A separately approved reduction in official statistical scope
- Faster or larger hardware/local scratch capacity
- Acceptance of a longer production runtime

Official statistical targets must not be changed merely to meet a runtime goal without human statistical approval and corresponding configuration/report identity changes.

## Final verdict

1. RNG’s 6h44 runtime is best explained by two fine-grained routes per seed, approximately 51,600 route/stamp artifacts per seed, 825,600 reducer opens, roughly 852,000 temporary spill/merge files, Python key/sort work, and repeated authentication.
2. Seed 49’s additional 24 minutes occurred almost entirely in the two route phases. Existing evidence cannot distinguish provider/filesystem variance from inherited process or thermal state.
3. H2H’s 4h18 runtime is primarily the configured 23.18-million-attempt inferential workload; roughly 32 minutes were aggregation/publication/authentication tail.
4. The final 23 minutes were pair/root state rechecks, run-context reloads, three authenticated graph audits, and health publication.
5. Current fast-config is a reduced production run mixed with integration settings.
6. Proposed integration reductions preserve branch and contract coverage but require approval and explicit non-production labeling.
7. The minimum prerequisite for another long run is reviewed Tasks 1A and 1B.
8. The highest-value overall change is approved integration sizing; the highest-value code change is RNG route-unit coarsening.
9. OneDrive versus algorithm cost must be resolved with the bounded storage A/B before designing a local working root.
10. Authentication deduplication and subprocess isolation are conditional, independently reviewed work.
11. Existing seed-48/49 artifacts remain read-only historical evidence but are not resumable after any committed source change under the current identity implementation.
12. Only one fresh end-to-end integration pair should run after accepted changes are committed and the worktree is clean.
13. Production readiness remains blocked until a bounded production-shaped benchmark meets a human-approved runtime budget.
