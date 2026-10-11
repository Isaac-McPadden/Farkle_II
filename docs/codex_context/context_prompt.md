# Codex Context Prompt

Use this prompt in a new Codex conversation for this repository.

For the September 24 first-release implementation backlog, Task 0's completed
design is [implementation/task0_contract.md](../implementation/task0_contract.md),
with question coverage and artifact/schema companions. It specifies future
Tasks 1–10; it does not claim those producers or runtime contracts are already
implemented. In particular, verified zero exposure requires authenticated full
coverage; missing required inputs must never be converted to zero counts.

Task 9 is implemented: see [its report](../implementation/task9_report.md).
Seat histories now use byte-bounded roster aggregation and authenticated per-k
reuse; metrics cache-key version is 3 and seat output schemas are explicit.
Task 1A is also implemented: [report](../implementation/task1a_report.md).
`outcome_distributions` computes exact partial frequencies and batch statistics
from bounded curated slices, with typed empty outputs and honest conditional
support. It does not certify zero exposure or publish final distributions.
Task 1B plus Task 2 now add `outcome_publication`: authenticated source slices,
bounded merges, complete batch/roster certification, F/B/I/S/Q publication,
and exact inverse-CDF summaries. That work advanced the metrics cache-key to 4
and added `outcome_distribution_method_version=1` to its shared version identity.
The graph snapshot and final audit retain `_outcome_units` artifacts and
validate their complete unit manifests; quarantine/execution scratch is excluded.
See [the implementation report](../implementation/task1b2_report.md).
Task 3 is implemented: `outcome_precision` publishes the frozen five selected
completed-outcome summaries with pointwise batch percentile intervals and MCSE.
Purpose 402 owns shared root/k/replicate draws; sparse, undefined, and degenerate
support withholds intervals explicitly. Authenticated vector/range/reduction
tiles retain exact resume and enter the release graph. Metrics cache-key version
is 5, with `outcome_precision_method_version=1`. Simulation targets are unchanged.
See [the Task 3 report](../implementation/task3_report.md).
Task 4 implements independent H2H statistical/practical/equivalence conclusions,
with explicit unresolved/nonviable flags and combined display labels. Production
equivalence tolerance is 0.03; practical thresholds and game allocations are
unchanged. Equivalence is pairwise, uses strict simultaneous bounds, and is not
conditioned on Holm non-rejection. Reporting contract is 6. See
[the Task 4 report](../implementation/task4_report.md).
Task 5A's [preflight contract](../implementation/task5a_contract.md) and
[bounded recipe](../implementation/task5a_recipe.json) are specified. They define
correctness coverage, evidence/status/reuse rules, and applicability limits;
they do not implement preflight or establish statistical detection capability.
Task 5B adds bounded diagnostic arithmetic and a seeded capability driver under
the [frozen protocol](../implementation/task5b_protocol.md). Its
[report](../implementation/task5b_report.md) records detection, false alerts,
unsupported cases, and batch/common-root blind spots; consult that evidence
before accepting applicability. This does not run the Task 5A all-k matrix.
Task 5C adds authenticated preflight and required batch monitoring. The existing
`disable_rng_diagnostics` switch now omits only exhaustive game-level lags;
PE/MC/MR remain required. RNG stage cache is 7 and reporting contract is 7.
The bounded all-k matrix includes an explicit requested-worker extension up to
16 workers; validation observations remain separate from production. See
[the Task 5C report](../implementation/task5c_report.md) for evidence and limits.
Task 6 adds plan-specific H2H cap authorization: terminal Y/N consent or the
runtime-only `--h2h-authorize PLAN_SHA256:MAX_ATTEMPTS` option, durable approval
separate from progress, and exact resume without editing the configured cap.
Execution cache version is 6; statistical planning and RNG identities are
unchanged. Refusal is an operational `blocked_by_cap` stop (CLI exit 3), and
final audit retains/checks authorization. See [Task 6](../implementation/task6_report.md).
Task 7 adds authenticated seat behavior and once-per-game turn opportunity,
plus bounded matched parameter responses/nonadditivity from observed all-attempt
performance and held-out HGB predictions. Metrics cache is 6, HGB cache is 2,
and reporting contract is 8. Check [Task 7](../implementation/task7_report.md)
for acceptance evidence and limits.
Task 8 is complete: authenticated `analysis_guide.json`/`.md` at reporting diagnostics,
covering all ten workbook groups and 62 targets. Reporting contract is 9 and
guide method is 1; metrics cache 6 and HGB cache 2 are unchanged. Required missing
inputs fail; verified zeros, unestimable estimates, deferred work and intentionally
omitted diagnostics stay distinct. Markdown renders the JSON, and final audit
checks both products. See [Task 8](../implementation/task8_report.md).

Task 10 is complete at its validation/documentation boundary, with all 1,584
collected functional cases passing across isolated runs and clean static gates.
Its integrated qualification is documented in
[its report](../implementation/task10_report.md), with a conditional capacity
refresh and [operating review](../implementation/task10_operations.md).
The required checks include a protected Windows launcher, strict Job Object,
S: results/C: scratch, exact-plan authorization, no-force replay, missing-product
recovery and final root/root/pair audit. Ingest now coalesces adjacent shard
batches within its existing byte allowance: physical layout/partition method 3,
ingest cache 4. This repairs a production footer-admission blocker without
changing ordered rows, coordinates, schemas, RNG or statistical methods.
The four inherited terminology findings are resolved, including the internal
`all_batch_merge` checkpoint name; current code fingerprints reject incompatible
old units. No production run, cap increase or optimization follows qualification.

```text
You are working in Farkle Mk II, a deterministic Monte Carlo simulation and
statistical analysis project. Read AGENTS.md and the files under
docs/codex_context before broad source exploration. Treat those maps as a
verified orientation cache, then inspect the relevant source before editing.

Preserve these invariants:
- explicit coordinate-owned PCG64DXSM streams;
- atomic, idempotent, resumable work;
- AppConfig-owned paths and only the six canonical artifact scopes;
- exactly one compatible sidecar for each derived artifact;
- streaming or partitioned handling for large data;
- complete root/k support and explicit cross-k weights;
- explicit outcome-schema-v2 termination status: a safety-limit attempt has no
  winner or ranks, counts as a loss for every tournament participant, and is
  excluded from winner-conditioned products;
- tournament primary rates and batch MCSE use all attempted player-game
  exposures; completed-only rates are labeled diagnostics;
- separation of descriptive screening, inference, dominance, and display order;
- RNG lag diagnostics use authenticated external partitions: strategy seat
  exposures follow the full RNG-v2 tournament-player coordinate, matchup games
  follow the game coordinate without participant expansion, the compact matchup
  digest is only a partition/display accelerator and every identity comparison
  includes the sorted padded participant key, eligibility precedes fixed-ring
  allocation, deterministic cap exhaustion is blocked-by-cap, and zero-centered
  descriptive reference bands make no independence claim;
- two-root stability is fixed-design descriptive reproducibility: intervals are
  Monte Carlo precision, practical threshold positions are not tests, and joint
  discrepancy reference quantiles/frequencies are not multiple-testing inference;
- current code never reads old on-disk analysis artifacts;
- pair analysis lives under `results_seed_pair_X_Y/seed_pair_analysis`, path
  lookup is non-mutating, and every H2H phase owns its artifacts;
- explicit artifact-contract-v3 runs publish simulation through frozen
  candidate-family artifacts with canonical path/scope, actual Arrow or typed
  non-Parquet format, complete version/method, code/config, exact source or
  immutable-manifest identities, and authenticated completion; old completion
  schemas are stale and v2 bytes cannot be promoted or re-sidecarred;
- TrueSkill root/k completion binds ordered row bytes and rating sidecars, and
  HGB completion binds target/features, whole-strategy folds, model/RNG method,
  outputs, and sidecars; HGB per-k row unions live in `concat_ks`;
- active configs contain only reloadable public fields; resolved paths, layouts,
  parent lifecycle roots, code identity, and execution controls live in the
  separately authenticated `run_context.json`;
- run profiles explicitly declare smoke/integration/production purpose and
  production/release claim eligibility; the existing fast family is
  reduced-resolution integration evidence, profile labels stay outside
  statistical compute identity but are authenticated in run contexts and
  label-publishing stages, and a non-release profile cannot pass the release
  eligibility gate;
- public process-tree execution defaults to one aggregate Windows Job Object or
  delegated cgroup-v2 `memory.max` boundary using an explicit aggregate
  hard limit; scheduler admission memory, RSS high-water warning, host-available
  reserve, parent-process estimate, logical CPU budget, and per-stage worker
  estimates have separate meanings, and requested/resolved/effective execution
  policy is authenticated outside statistical freshness; setup fails closed in
  strict mode, nested process pools are suppressed, and partition manifests publish only
  after every unit validates; RSS high water provides nonsticky submission
  backpressure, Windows Job committed-memory peak and cgroup current/peak remain
  distinct telemetry, recognized resource failures cancel pending work and may
  retry pending partition coordinates once at half workers, valid publications
  are never quarantined for a soft warning, and roots fail fast before pair work;
- the CLI runtime-only resource mode is `enabled`, `memory-only`, or `disabled`;
  memory-only retains all memory controls while leaving CPU/native threads to
  the OS, disabled bypasses host ceilings, and both modes retain explicit
  `n_jobs`, nesting suppression, bounded queues, and byte-bounded streaming;
- optional absolute `io.temp_dir` moves growing per-file data, metadata, and
  analysis workspaces out of synchronized results trees; publication is data,
  then sidecar, then completion, with atomic same-filesystem rename or a bounded
  destination-local cross-filesystem bridge, while canonical checkpoint cadence
  and exact resume remain unchanged;
- byte-bounded streaming, atomic durable units, and exact resume are permanent
  engineering invariants; a fixed 1 GiB process-tree ceiling is not;
- RNG diagnostic checkpoint route-layout v3 greedily groups the canonical
  source-row-group sequence by expanded fixed-width bytes (64 MiB target, 96 MiB
  hard ceiling). Combine emits coalesced configured-size row groups; route workers
  reuse readers, reduce counts across whole route units, and publish one batch per
  partition. Reducers fill 256 MiB runs across route boundaries and use bounded
  array merges; exact compiled BLAKE2b and lag kernels remove Python record loops.
  Partition checkpoint schema v3 prevents v2 reuse; RNG method v4, cache-key v6,
  statistical identity, and canonical output schemas are unchanged;
- simulation v3 publication is producer-owned: staged data is hashed and
  sidecarred without a canonical data rewrite, coordinate-sorted native
  manifests reuse those identities, routine completion/resume is metadata-fast,
  and the explicit release audit remains byte-deep;
- curate preserves its authenticated provenance boundary with independent
  copy-on-write reflinks where supported and deterministic bounded physical-copy
  fallback; combine keeps normalized by-k Parquets as its only canonical data
  bytes and exposes `concat_ks` through one authenticated ordered partition
  manifest and deterministic scanner, with explicit deep verification and
  non-canonical release materialization;
- ingest and combine resolve authenticated upstream lifecycle/artifact identity
  once per root/k in the parent, pass immutable pickle-safe source snapshots to
  workers and validators, bind ingest freshness to simulation manifest/completion
  roots, classify allocator failures separately from malformed Parquet, and cap
  process workers by pending independent units;
- TrueSkill projected reads use byte-bounded Arrow batches; screening
  diagnostics publish an authenticated root/k result and completion stamp
  before deterministic aggregation, using one forward projected scan and a
  bounded reverse replay spool;
- all-player and seat batch metrics publish deterministic single-entry stream
  manifests with source-bound sidecars before per-k or stage completion;
  all-player metrics use k/schema-aware byte-bounded projections and stable
  row/seat accumulation; canonical performance uses memory-mapped root/k count
  matrices plus authenticated deterministic replicate-range shards, and only
  publishes bootstrap summaries after a complete validated range manifest;
- two-root stability reuses those immutable root/k matrices (or publishes a
  compatible fallback cache once), with independently authenticated top-N and
  joint-discrepancy ranges; final completion is stale unless both manifests validate;
- the H2H power plan and block manifest are immutable after publication;
  exact-power allocation is the global first admissible crossing, power is
  conditional on completed support, and cap authorization/execution progress
  is a separate resumable artifact;
- H2H targets completed games per pair/root/order, retains safety-limit attempts
  outside score-test counts, resumes deterministic contiguous replacements up
  to the frozen 2.0x attempt cap, keeps no-test pairs in the multiplicity family,
  and evaluates the frozen candidates against the 0.99 incident-attempt
  completion threshold without shrinking the family;
- H2H execution uses a deterministic execution-only cap of 5,000 attempted
  coordinates per durable block extension. Current 1,372-2,191 targets and
  frozen 2.0x caps therefore finish in one executor generation; larger future
  blocks split into exact contiguous ranges. The cap is outside statistical
  identity, valid authenticated prefixes may resume under a changed cap, and
  terminal block column order is independent of checkpoint history;
- final reporting authenticates the migration inventory as a stage output and
  states TrueSkill's completed-game conditioning and safety-limit exclusion;
- transient artifact I/O uses bounded, provider-neutral retries, and completed
  H2H execution can finalize a missing stamp without replaying block data.
- completed root and pair contexts create one immutable, explicitly owned,
  nonserializable process-local authenticated graph snapshot; same-process
  lifecycle/state finalization reuses require the same process, context, and
  generation, while one top-level `final_byte_deep_release_audit` freshly
  authenticates all root/root/pair bytes, schemas, contexts, configurations,
  completion inventories, and cross-context provenance before success.
- completion-like files have one semantic namespace contract: executable-plan
  stage completions and simulation lifecycle stamps are canonical; typed
  partition-unit stamps with their adjacent output are resumability state;
  active-config completions, checkpoints, substages, and malformed done-like
  files remain distinct. Release auditing freshly authenticates the exact
  canonical inventory and fails on missing, relocated, duplicated, mutated, or
  unexpected canonical completions without treating durable unit stamps as
  canonical graph additions;
- two-seed orchestration is strictly sequential and rejects
  `orchestration.parallel_seeds: true` before run-start/health publication;
  player counts are also sequential, while the active tournament owns the full
  resolved simulation process budget;
- a top-level simulation process pool is suppressed only when both the process marker
  and a genuine multiprocessing parent prove nested execution. Process-pool
  topology is observed (requested/resolved/created/live/peak), mismatches fail
  early, Windows cancellation owns a bounded terminate/join/kill sequence, and
  authenticated row/metric manifests independently authorize exact resume;
- simulation heartbeats and `pipeline_telemetry.json` expose seed, root pair,
  scope, active stage/phase, k, shuffle/chunk/game progress, rates/ETA,
  executor state, pending work, checkpoints, process-tree memory, and elapsed
  time without artifact-tree scans. Parent-owned durable counters are the only
  displayed completion numerators; semantic worker event IDs deduplicate
  delayed, retried, duplicate, and out-of-order observations. Unknown
  denominators and ETA are explicitly indeterminate, and terminal log/sink
  output comes from one locked snapshot. Recognized atomic staging files from
  interrupted workers are never completion inputs and are removed before resume;
- ordinary two-seed resume checks the existing authenticated pair public-config
  and code identity before health, manifest, context, or active-config
  publication. Compatible no-force replay preserves authenticated run-context
  bytes when only live host-memory telemetry changes; incompatible identity
  fails before publication.
- Task 5A's historical pre-optimization capacity gate derived the production plan as 5,160
  strategies, 39,013,900 games and 177,504,000 player exposures per root,
  34,400 source row groups, and 1,075 route units at the accepted width of 32.
  The historical fan-in-32 RNG reduction had three projected merge generations per
  reducer route (the historical four-pass count predates route coarsening). The
  150-candidate envelope has 11,175 pairs, 44,700 blocks, and 97,937,700 planned
  completed H2H games. The machine/storage projection is 10.47 days central,
  7.48-16.44 days plausible, and 19.73 days conservative; the readiness verdict
  is `not_capacity_ready`. This is capacity evidence only: it does not authorize
  Task 5B, a production run, optimization, local scratch, or statistical-scope
  changes.
- The current RNG route-layout v3 production dimensions are 613 coalesced combine
  row groups and 393 byte-planned route units per root. The largest projected
  expanded unit is 78,208,000 bytes under the 96 MiB hard ceiling. Accepted
  bounded benchmarks preserve the historical final digests and measure 5.06x to
  9.06x speedups; the production-`k` mix benchmark covers all configured game
  sizes with full 12-seat records. The conservative central two-root RNG update
  is 33.1 hours. The refreshed full-pipeline model is 114.50 hours central with
  a 9.03-day planning upper, so this is not a production-run authorization and
  the pipeline remains `not_capacity_ready`.
- Task 7A proved that the approximately 25-row combine groups and 18.42 MB of
  padded-player-column bytes belong to the pre-Task-6 seeds-54/55 tree. The
  accepted seeds-56/57 tree has six combine groups averaging 15,200 rows and only
  34,900 compressed padded-column bytes. Its remaining material density issue is
  H2H: 180 finalized block Parquets use 4.00 MB of 4.82 MB in footer metadata,
  and accepted telemetry attributes 100.76 s of the 171.69 s H2H stage to block
  checkpoints. A diagnostic-only 16-block bundle fixture preserved the exact
  logical digest while reducing write/scan time by about 93%. Task 7A recommends
  at most one separately approved Task 7B for deterministic H2H block-range
  bundles with an H2H physical-layout identity advance; it does not authorize
  implementation. Using the slower accepted Task 6 rerun, current pre-remediation
  capacity is 127.30 hours central and 240.44 hours planning upper, still
  `not_capacity_ready`. The older remediation Task 8 remains untriggered;
  this is distinct from the September 24 first-release analysis-guide Task 8.
- Task 7B evaluated deterministic width-16 H2H block-range bundles on the real
  Windows-spawn path but closed `performance_gate_failed`. Exact schedule,
  logical Arrow, schema, execution-state, and inference equivalence held.
  Median publication improved 66.64% for the preregistered primary 180-block
  case and 79.70% for 760 blocks; the primary result missed the non-waivable
  70% gate, and critical-worker simulation medians also exceeded the 5%
  regression threshold. The candidate implementation was rolled back. Current
  H2H storage remains one authenticated Parquet plus sidecar per block, Task 7A
  remains the operative capacity model, and no compatibility reader or
  production authorization was added.

Before a statistical change, identify the estimand, conditioning, chance
baseline, replication unit, uncertainty procedure, multiplicity rule, and
permitted report claim. Before a recovery change, compare coordinate manifests
and logical outputs across worker counts and interruption/resume.

Check git status before editing and use the repository .venv for validation.
```

Update this prompt and `metadata.md` when stage order, statistical contracts,
path/scope behavior, sidecars, RNG identity, or release gates materially change.
