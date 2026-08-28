# Task 7A post-Task-6 capacity and artifact-density diagnosis

## Verdict

`bounded_remediation_recommended`

Recommend exactly one separately approved Task 7B: deterministically bundle at
most 16 contiguous H2H root/order block rows per atomic physical checkpoint and
canonical publication unit, while exposing the unchanged coordinate-ordered
logical block stream to every consumer.

This is not an implementation authorization. Task 7A changed no production
module, configuration, schema, authentication policy, checkpoint, or statistical
behavior. The current pipeline remains `not_capacity_ready`.

The accepted H2H run measured 100.765 s of block-checkpoint work for 180 blocks.
The benchmark-only representation retained the exact 180 rows and schema but
reduced 180 Parquets to 12: median write time fell from 8.049 to 0.588 s, scan
time from 4.001 to 0.290 s, footer metadata from 3.979 MB to 0.265 MB, and hash
time from 0.895 to 0.010 s. All three repetitions on both layouts had the same
logical digest. Scaling the accepted checkpoint timer to 44,700 production
blocks gives 6.95 h before final authentication; the central net opportunity is
about 7.04 h (5.53% of the refreshed total), with 3.3 h retained after the
precommitted conservative haircut. It is the only candidate that clears the
published gate.

Machine-readable evidence is in
[`task7a_artifact_density.json`](task7a_artifact_density.json), raw repetitions
and inventories in
[`task7a_artifact_density_raw.json`](task7a_artifact_density_raw.json), and the
separate current-state projection in
[`task7a_production_capacity.json`](task7a_production_capacity.json).

## Repository and evidence provenance

The benchmark plan was written before the first Task 7A measurement:
[`task7a_artifact_density_benchmark_plan.md`](task7a_artifact_density_benchmark_plan.md).
It fixed a 6 h and 5% central materiality gate, a 3 h conservative-net gate,
equivalence oracles, repetition/order rules, resource bounds, and safe ownership.

| Item | Value | Classification |
| --- | --- | --- |
| Branch / upstream | `main` / `origin/main` | measured |
| HEAD and upstream tip | `9f879de1cdb4e6b92808799899c1e6af06980ca1` | measured |
| Tracked status before Task 7A | clean | measured |
| Pre-existing untracked status | eight `task6_rerun_20260827_*` documents | measured; user evidence preserved |
| Current immutable run | `data/results_rng_opt_fast_seed_pair_56_57` | measured/read-only |
| Historical immutable run | `data/results_efficiency_updates_4D_fast_seed_pair_54_55` | measured/read-only |
| Owned Task 7A root | `data/farkle-task7a-artifact-density-v1` | disposable/retained benchmark evidence |

The untracked Task 6 rerun was treated as accepted additional operational
evidence, not as Task 7A work. It reproduced every digest but was 1.28-1.39x
slower than the original Task 6 dense/mix session; its lower 3.650x historical
speedup is therefore used for the refreshed current-state capacity projection.
No supplied hash was used as a precondition.

The driver initially stopped before checkpoint publication on an invalid
benchmark fixture (`int8` overflow), then a Windows read-only attribute prevented
verified owned-fixture cleanup. Both diagnostic-only defects were fixed and
regression-tested. The final retained checkpoint contains only the conforming
three-repetition measurements; accepted evidence trees were never written.

## Task 5A, Task 6, and integration reconciliation

Task 5A's historical model projected 251.3 h centrally, of which RNG was 167.5 h.
Task 6 changed combine physical layout and RNG checkpoint/access/reduction layout,
preserved canonical digests, and projected 114.50 h centrally using its 5.06x
conservative speedup. The fresh same-code rerun supports 127.30 h centrally using
3.650x. These are session-sensitive bounded projections, not contradictory
logical results.

The accepted seeds-56/57 telemetry envelope is 1,349.188 s (22m29.2s), close to
the prompt's approximate 22m39s CLI observation. It ran 91,200 tournament games
per root (182,400 across two sequential roots) and 237,060 H2H games. The profile
is integration-only and has no production statistical resolution or release
eligibility.

| Phase | Seconds | Label |
| --- | ---: | --- |
| Simulation, two roots | 255.469 | measured |
| Ingest | 147.032 | measured |
| Curate | 13.312 | measured |
| Combine | 50.359 | measured |
| Metrics/performance/bootstrap/seat work | 80.172 | measured wrapper |
| Game statistics | 224.187 | measured |
| RNG diagnostics | 36.532 | measured |
| Root TrueSkill | 109.688 | measured |
| HGB | 21.843 | measured |
| Screening/final root decisions | 1.969 | measured |
| Root lifecycle finalization/snapshots | 65.281 | measured |
| Root stability | 20.234 | measured |
| Pair TrueSkill / candidate freeze | 0.687 / 0.594 | measured |
| H2H power / execution / inference | 8.328 / 171.688 / 6.094 | measured |
| Dominance digest / agreement / reporting | 1.296 / 0.625 / 13.485 | measured |
| Pair wrapper | 223.281 | measured; contains the pair children, not additive |
| Pair snapshot | 18.125 | measured |
| One final byte-deep release audit | 81.078 | measured |
| Resource final check | 0.094 | measured |
| Residual orchestration/publication | 20.766 | derived |

Summing each root phase once, the pair wrapper once, and the post-pair phases
gives 1,328.422 s. The 20.766 s (1.54%) residual covers run-context/config reload,
root transitions, health and final manifest publication, logging, and timer gaps.
The pair children sum to 223.031 s, 0.250 s below their wrapper. No overlapping
`pair_analysis` timer was double-counted.

The old Task 5A coefficient that became decisively stale was 34,400 combine row
groups and 1,075 fixed-width RNG route units per root. Current production derives
613 combine row groups and 393 byte-planned route units. The old simulation,
H2H-game, H2H-block, other-analysis, and fixed/finalization coefficients remain
wide extrapolations. The new integration is smaller than the Task 4D/Task 5A
anchor in several dimensions, so it validates stage order and fixed terms but
must not be forced to reproduce either production model.

## Current RNG runtime decomposition

Accepted integration evidence gives the enclosing timers: 23.438 s for root 56
and 13.094 s for root 57. Both used three source row groups, one count-route unit,
one statistics-route unit, eight reducer partitions, no reuse/retry/downshift,
one routing worker, and four reducer workers. The difference is operational
variation; output completion and authentication passed.

The smallest production-shaped Task 6 mix supplies the detailed decomposition.
It has 32,768 source rows across every production `k`, full 12-seat width, 64 row
groups, two route units per route, 32 reducers, and one worker. Values below are
medians of the two accepted measured repetitions.

| Component | Wall / operations | Attribution |
| --- | ---: | --- |
| Count route | 1.011 s | measured phase; includes source open/read/decode, row-group iteration, extraction, key construction, exact hash, route-local reduction, partitioning, IPC |
| Count reduce/eligibility | 1.523 s | measured phase; includes route opens, spill construction, eligibility publication and selection |
| Statistics route | 4.656 s | measured phase; includes second source scan, two membership loads, filtering, partitioning, IPC |
| Statistics reduce/final construction | 10.868 s | measured phase; includes reducer reads, spill/merge logic, compiled lag accumulation, final statistics/publication |
| Enclosing wall / CPU | 18.139 / 18.016 s | measured |
| Four-phase sum / residual | 18.058 / 0.081 s (0.45%) | derived |
| Source | 32,768 rows, 64 groups, 1,013,780 compressed file bytes, 1,755,176 projected read bytes, 128 Arrow batches | measured |
| Count records | 188,928 raw; 163,605 batch-reduced; 71,689 routed after route-local reduction | measured |
| Statistics records | 188,928 raw/routed | measured |
| Route publication | 4 files, 128 record batches, 22,706,024 bytes | measured |
| Reducer operations | 128 route opens; 2 membership loads | measured |
| Spill/merge | 64 initial runs, 22,342,434 bytes; zero merge generation in this bounded mix | measured |
| Hash/authentication | 4 calls/files, 22,706,024 bytes, 0.0223 s | measured nested timer |
| Parent / worker CPU | 18.016 / 17.938 s | measured; parent includes in-process worker accounting |
| Resources | 266,356,736-byte peak RSS; 37 native threads; 1 requested/effective worker | measured |
| Cleanup | zero failures | measured; time is within phase/residual |

The accepted 1,024-group sparse fixture identifies merge behavior absent from the
mix case: 32 initial spills/166,131 bytes, 12 reducer-generation passes, 22 merge
outputs, and 387,072 merged bytes. Production planning derives 393 route units
per route and bounded vectorized merge generations; it must not use the mix
fixture's zero merge count literally.

Several requested subcomponents are not independently timed by accepted
production code. Discovery/inventory resolution and row-group planning occur
before or within phase entry; source decode is inseparable from route compute;
manifest/stamp/sidecar/completion and cleanup are within phase timers. The report
does not invent false precision: their operations/bytes are measured where
available and their wall time is explicitly embedded in the four phase totals.
This accounts for the enclosing timer to a 0.081 s residual without rerunning a
large workload. The current conservative production RNG projection is 45.91 h
central, 33.05-71.15 h plausible.

## Physical artifact audit

### Whole-tree inventory

| Quantity | Historical seeds 54/55 | Current seeds 56/57 |
| --- | ---: | ---: |
| Total files / bytes | 8,850 / 501,201,165 | 8,474 / 318,804,353 |
| Parquets / bytes | 3,997 / 325,551,773 | 4,033 / 225,421,564 |
| Parquet rows / row groups | 654,867 / 22,151 | 653,512 / 18,617 |
| Footer bytes / share | 168,215,290 / 51.67% | 113,518,908 / 50.36% |
| Sidecars / bytes | 4,004 / 50,743,095 | 4,040 / 51,270,648 |
| Completions / bytes | 98 / 1,223,919 | 98 / 1,349,743 |
| Partition stamps / bytes | 352 / 347,256 | 128 / 126,534 |
| Manifests / bytes | 54 / 2,170,718 | 54 / 2,093,488 |

The release audit's canonical graph independently reports 3,993 files in each
root and 481 pair files, with 1,913/1,913/214 artifacts, equal sidecar counts,
and 12/12/9 canonical completions. Top-level config/context/health files bring
the full physical inventory to 8,474.

Current stage bytes include 53.79 MB ingest, 53.81 MB curate, and 5.02 MB combine.
Ingest writes normalized rows; curate preserves its independent authenticated
boundary through reflink/copy semantics, so logical file bytes are duplicated
even where physical allocation may be shared; combine is a normalized rewrite.
No Task 7A estimate treats reflink logical bytes as guaranteed allocated bytes.

There are 3,600 simulation row shards (600 shuffles x three `k` x two roots), one
sidecar per shard, plus 12 metric chunks. Simulation preserves shuffle-level
recovery and source provenance. Production derives 68,800 row shards across two
roots.

Current H2H execution has 180 one-row block Parquets plus the two-row aggregate:
4,000,949 of 4,818,436 Parquet bytes (83.03%) are footer metadata. Its 193 sidecar
publications occupy 2,768,584 bytes. This is current, not historical-only.

The current final audit measured 100,443 opens, 46,042 SHA-256 calls, 7.407 GB
hashed, 13,238 schema validations, and 37,859 metadata validations. Its one fresh
byte-deep traversal took 81.078 s and passed. No recommendation removes or makes
that audit metadata-only.

### Structural-finding dispositions

| Finding | Disposition |
| --- | --- |
| 310.47 MiB / 3,997 Parquets and 160.42 MiB footer metadata | Historical seeds 54/55 exactly; current Parquet fell to 215.0 MiB and 108.3 MiB footer metadata |
| ~48.39 MiB / 4,004 sidecars | Historical numbers; same inefficiency remains current at 48.90 MiB / 4,040 |
| Combine averages ~25 rows/group | Historical only: 3,600 groups, 25.33 rows/group. Task 6 corrected current output to six groups averaging 15,200 rows |
| All-null padded player columns | Current but negligible: 34,900 compressed bytes, down from 18,420,000 historical |
| Metadata-heavy H2H blocks | Present and material at production block count |
| Repeated invariant JSON content | Present: commit and `release_clean` each repeat 4,044 times; `PCG64DXSM` repeats 3,822 times |
| Null-typed empty outputs | Nine current empty Parquets; some are correctly typed, while pandas-created seat/HGB/dominance empties contain `null` fields. Real representation/contract defect, negligible capacity, separate migration |
| Six `manifest.jsonl` pretty objects | Present in both generations. All six are curate schema manifests intentionally written/read as one JSON object. Misleading path/format contract, not corrupt JSONL input; repair requires authenticated rename or format migration |

Typed-empty details, including every field/type/nullability, and every structural
JSONL result are retained in raw JSON. No artifact was fixed while inspecting.

## Bounded benchmark results

All cases ran on the synchronized repository volume with production Arrow
schemas, Snappy and atomic Parquet writer. One warm-up preceded three measured
repetitions. Each logical digest was invariant. Cache state, OneDrive activity,
antivirus, CPU frequency and host load remain confounders; reversed order and
conservative ranking limit their effect.

| Case | Files | Write | Footer open | Full scan | Hash | Bytes / footer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k=2 native | 64 | 3.961 s | 0.395 s | 1.063 s | 0.216 s | 3.796 / 0.606 MB |
| k=2 native | 4 | 0.298 s | 0.004 s | 0.021 s | 0.083 s | 3.221 / 0.038 MB |
| k=2 padded to 12 | 64 | 7.145 s | 0.771 s | 2.592 s | 0.375 s | 5.898 / 2.272 MB |
| k=2 padded to 12 | 4 | 0.442 s | 0.011 s | 0.208 s | 0.008 s | 3.353 / 0.142 MB |
| k=12 native | 64 | 6.725 s | 1.167 s | 2.607 s | 0.276 s | 16.572 / 2.399 MB |
| k=12 native | 4 | 0.557 s | 0.012 s | 0.226 s | 0.098 s | 14.267 / 0.152 MB |
| Actual H2H block rows | 180 | 8.049 s | 3.134 s | 4.001 s | 0.895 s | 4.772 / 3.979 MB |
| Actual H2H block rows, 16/bundle | 12 | 0.588 s | 0.136 s | 0.290 s | 0.010 s | 0.339 / 0.265 MB |

Production sidecar validation over 8/32/128 accepted small artifacts had median
wall times 0.0916/0.5597/2.6817 s for 1,832/13,547/662,150 artifact bytes. This
demonstrates a strong per-artifact/provider term; it is not used as a pure byte
throughput coefficient because sidecar parsing, schema checks and cache state are
mixed into the operation.

Peak benchmark RSS and native-thread samples, CPU times, every repetition,
cleanup time/failures, physical/logical digests, and environment versions are in
the raw evidence. No case used a worker pool, spill, or merge; those counters are
not falsely reported as production RNG evidence.

## Candidate ranking

1. **H2H deterministic block-range bundles — recommend.** Physical/checkpoint
   layout change with one H2H layout-identity advance. Expected central saving
   7.04 h, conservative net 3.3 h, about 83,812 files and 1.7 GB. It changes
   canonical physical bytes and recovery quantum but not logical rows or science.
2. **Simulation row-shard coalescing — do not recommend now.** Physical/checkpoint
   layout change. The measured write/open/ingest/final-audit opportunity is about
   3.5 h centrally and 1.5 h conservatively, below both gates. It also trades
   shuffle-level recovery for a larger range.
3. **Sidecar factoring — do not recommend.** It could reduce repeated bytes and
   opens, but is a broad authentication/public artifact-contract migration, not
   simple I/O cleanup, and independently remains below the gate.
4. **Further RNG architecture work — no bounded candidate.** RNG is still 36.1%
   of the refreshed central path and statistics reduction dominates the bounded
   mix, but Task 7A did not identify one independent change with measured gain.
5. **Typed-empty/manifest-format repair — separate correctness migration.** Real
   representation/naming defects, negligible runtime, multiple producers and
   authenticated paths. Do not combine with performance work.
6. **No remediation.** Rejected because the bounded H2H candidate clears the gate.

The H2H estimate scales the measured 100.765 s / 180-block checkpoint term by
44,700/180 to 6.951 h, applies the measured 92.70% write reduction, and adds a
conservative 0.6 h file-count-proportional share of final authentication. The
plausible benefit is 4.5-8.0 h. The conservative 3.3 h value halves the combined
benefit and charges bundle planning/validation. Candidate count is the principal
sensitivity: a smaller frozen family reduces block count quadratically.

At 16 blocks/bundle, production data/sidecar inventory falls by about 83,812
files. Current-format block data and sidecars project near 1.83 GB; measured data
compression plus one sidecar per bundle gives roughly 0.13 GB, a rounded 1.7 GB
opportunity. This storage estimate is secondary to the wall-time gate.

Candidate-specific contract and operational details:

- **H2H bundles.** Producer/consumers are H2H execution, resume/finalization,
  inference source resolution, completion and release audit. This is one physical
  checkpoint/publication migration; no global authentication policy is needed.
  Peak memory adds at most 16 one-row records and bounded writer state; temporary
  space adds one bundle staging file. Worker pools and Windows spawn are unchanged.
  Recovery increases from one block to 16 contiguous blocks. Physical/canonical
  bytes and paths differ under the new layout identity; logical rows and all
  downstream numerics must not. Old roots remain immutable and non-resumable by
  the new identity; no fallback reader or in-place migration is added. Missing,
  duplicate, overlap, omission, relocation, corruption, stale identity and
  unexpected inventory are required rejection tests. Breadth/risk is medium and
  no second migration is inseparable.
- **Simulation coalescing.** Target is the simulation row-shard producer plus
  ingest/resume/release consumers. Current support is 3,600 shards/two-root
  integration and 68,800 projected production shards. Production-width fixtures
  imply roughly 1.2-1.9 h write saving, 0.73 h ingest saving and at most about
  1 h authentication/finalization saving: ~3.5 h central, 1.5 h conservative.
  A 16-shuffle range would reduce about 129,000 data/sidecar files but make the
  recovery quantum 16 shuffles. Memory can remain byte-bounded; temporary space
  adds one range staging file; Windows spawn is unchanged. Simulation layout
  identity and canonical bytes/paths would change. The oracle is exact typed rows
  in shuffle/game coordinate order plus manifest/completion roots. Required tests
  mirror H2H inventory/recovery cases. Old roots cannot resume. Risk is medium-high
  because producer durability and canonical source provenance move together; it
  fails the gate and must not be combined with sidecar factoring.
- **Sidecar factoring.** Target is the shared artifact contract and every producer,
  validator, graph snapshot and release-audit consumer. Current cost is 4,040
  files/51.27 MB and production projects about 114,430 sidecars; small-artifact
  validation measured 2.68 s for 128 files. A proportional bound is at most ~2 h
  central finalization/publication and ~1.5 GB, with high uncertainty. Recovery,
  RAM and spawn do not materially change, but sidecar bytes, paths/lookup and
  public authentication semantics do. Logical artifacts/numerics are the oracle;
  every missing/common-metadata mutation and old/new version combination needs
  testing. This is a broad backward-compatibility policy migration, high risk,
  and inseparable across many producers. It is ineligible here.
- **Further RNG work.** Target would be statistics reducers/lag construction,
  because that bounded phase is 10.868 of 18.139 s and current RNG projects
  45.91 h. No concrete alternative was measured, so wall/storage/file opportunity,
  canonical byte effects and recovery changes are unknown. Any future proposal
  must preserve the final digest, grouping/lag order, partition/worker invariance,
  bounded spills, corruption rejection and Windows spawn. It may require another
  layout identity. Without an independently measured design, implementation breadth
  and migration coupling are unbounded; no Task 7B is specified.
- **Typed-empty/manifest repairs.** Targets are the pandas-created empty producers
  in seat analysis, HGB and dominance, plus the curate schema-manifest path/format.
  Current cost is nine empty Parquets and six ~0.58 KB manifests: effectively zero
  production runtime/storage opportunity. Typed repairs can alter authenticated
  Arrow schema; manifest repair alters path or bytes. Consumers, sidecars,
  completions and audit need exact empty/nonempty, stale-schema, relocation and
  mixed-version tests. RAM, temporary space, recovery and spawn are unchanged.
  These are multiple correctness/contract migrations, not one performance change,
  and require separate approval if pursued.
- **No remediation.** It has zero implementation/migration/recovery risk and
  preserves every byte, but leaves the measured 6.95 h production-scaled H2H
  checkpoint term. It would be selected if the H2H conservative benefit fell below
  3 h for a smaller realized candidate family; the current 150-candidate capacity
  envelope clears the published gate.

## Proposed Task 7B boundary

Task 7B should change only H2H execution physical checkpoint/publication layout:

- group canonical block coordinates into contiguous deterministic ranges of at
  most 16;
- atomically publish one Parquet and one sidecar per complete range;
- expose a deterministic scanner yielding the original block rows in exact
  coordinate order;
- advance only the H2H physical layout/checkpoint identity; do not advance H2H
  method v2, RNG/outcome/statistical identity, power plan, or logical schema;
- never migrate, promote, reseal, or re-sidecar old per-block bytes in place.

Expected production files are `h2h_schedule.py`, `h2h_inference.py`, narrowly
necessary H2H path/layout helpers in `config.py`, and shared completion code only
if the bundled inventory cannot be expressed through existing primitives.
Simulation, RNG, power/statistical methods, candidate freeze, inference formulas,
dominance, agreement, reporting semantics, global authentication policy, Task 8,
and local scratch are explicitly excluded.

Capture a baseline fixture and digest before implementation. Permitted differences
are Parquet bytes, bundle paths/inventory, row-group grouping, new-layout sidecars
and completion provenance, and physical telemetry. Forbidden differences are
coordinate support/order, Arrow field types/nullability, attempt/completion/safety
counts, wins, statuses, plan/family membership, inference values/decisions,
dominance/agreement/report values, or any scientific identity.

Required interruption tests stop before the first bundle; with a partial range
only in memory; after staging before replace; after data before sidecar/stamp;
with mixed complete/incomplete ranges; and after all bundles before aggregate and
completion. Corruption tests cover corrupt, missing, duplicate, overlapping,
omitted, relocated, stale-identity and unexpected inventory entries.

Run one-worker, multi-worker, worker-count-invariance and Windows-spawn cases.
Telemetry must report blocks buffered/published/reused/replayed; bundles
planned/published/reused/rejected; data/sidecar/footer bytes; write/hash/validation/
aggregation/cleanup wall and CPU; opens/hash calls; workers; RSS; and native
threads. The benchmark matrix must include 180 blocks and a production-shaped
bounded count, three reversed repetitions, interruption/resume, and final audit.
Acceptance requires at least 70% median block-publication wall reduction and exact
logical/downstream equivalence.

Stop Task 7B for approval if it needs a global authentication-policy migration,
more than 16 blocks per recovery unit, a logical schema/statistical change,
in-place historical migration, a resource/process-isolation change, or if either
the 70% performance gate or any equivalence oracle fails.

## Pre-remediation capacity projection

Task 7A applies no candidate savings to current state. The fresh conservative
Task 6 rerun is the planning basis.

| Stage | Central | Plausible |
| --- | ---: | ---: |
| Simulation | 63.80 h | 45.93-98.89 h |
| RNG diagnostics | 45.91 h | 33.05-71.15 h |
| Other root analyses | 1.65 h | 0.74-3.97 h |
| H2H and pair analyses | 13.94 h | 10.04-21.61 h |
| Authentication/finalization | 1.50 h | 0.75-3.75 h |
| Fixed orchestration | 0.50 h | 0.25-1.00 h |
| **Total** | **127.30 h / 5.30 d** | **90.77-200.36 h / 3.78-8.35 d** |

The correlated conservative planning upper is 240.44 h / 10.02 d. Six 18-hour
windows provide 108 h: the plausible lower fits, the central exceeds it by 19.30
h, and plausible/conservative uppers do not fit. Runtime-budget acceptance is a
user decision; this report does not declare production readiness.

Central durable storage remains 235 GiB (180-350 GiB plausible), peak temporary
24 GiB (48 GiB upper), and 231,857 final files. Peak RSS remains 4.5 GiB central,
8 GiB planning, under the 12 GiB hard limit; peak topology is 16 processes and
160 native threads centrally (190 planning). These are streamed/sequential
capacity estimates, not record-proportional RAM scaling.

Pre-remediation recovery quanta are one simulation shuffle shard; at most a
96 MiB expanded RNG route unit (393 units/route/root); and one H2H extension of
at most 5,000 attempts. The proposed Task 7B would increase only H2H recovery to
at most 16 contiguous blocks and must quantify replay cost.

## Task 8 disposition

No trigger. The accepted run peaked at 4,261,208,064 bytes RSS and 101 native
threads with zero warning crossings, near-hard-boundary events, downshifts,
retries, or resource failures. There is no measured cross-stage RSS/commit
retention, repeatable inherited-state failure, or fresh-process A/B benefit.
Multiple-day runtime and normal bounded worker memory are not triggers.

## Limitations and risks

- Task 7A file-count timing is one synchronized Windows/OneDrive host session.
- The production model extrapolates simulation exposure ~1,100x, H2H games ~413x
  from this integration run (and blocks 248x), and current RNG records manyfold.
- H2H bundling adds recovery coupling and needs careful out-of-order worker-result
  buffering, corruption rejection, and deterministic range completion.
- Sidecar savings assume one full authenticated sidecar per bundle; no factoring
  or authentication-policy change is included.
- The exact production H2H family may be below 150 candidates; benefit then falls
  quadratically and must be reported against the frozen realized plan.
- Physical allocation under reflinks and OneDrive hydration is not inferable from
  logical file length.
- RNG microphase timing cannot separate decode from compute without production
  instrumentation; embedded attribution is explicit rather than fabricated.

## Changed files, retention, and verification

Task 7A files:

- `docs/remediation/task7a_artifact_density_benchmark_plan.md`
- `scripts/benchmark_task7a_artifact_density.py`
- `tests/unit/scripts/test_benchmark_task7a_artifact_density.py`
- `docs/remediation/task7a_artifact_density_raw.json`
- `docs/remediation/task7a_artifact_density.json`
- `docs/remediation/task7a_production_capacity.json`
- `docs/remediation/task7a_artifact_density_report.md`
- orientation updates under `docs/codex_context`

Retained: the exact owned root, ownership marker, final atomic checkpoint, and
bounded fixture cases. Safely cleaned: only superseded named children under the
verified `fixtures` root during `--force`; read-only provider attributes were
cleared only there. No historical, accepted, user, or unrelated benchmark root
was removed.

Final verification commands and results are recorded after the orientation
updates and final diff review below. No full production or fresh two-seed
integration run was performed.

Verification results:

- `\.venv\Scripts\python -m pytest` over the Task 7A, Task 5A capacity, and
  Task 3B RNG benchmark tests: **17 passed** (6 + 6 + 5).
- Five Task 7A JSON documents parse: report evidence, capacity evidence, raw
  evidence, retained ownership marker, and retained checkpoint.
- Repository-wide Ruff: **passed**.
- Black `--check` on both changed Python files: **passed**.
- Mypy over `src`: **passed, 87 source files**.
- Pyright over the Task 7A driver and tests: **0 errors, 0 warnings** (only the
  tool's informational newer-version notice).
- `git diff --check`: **passed**; Git emitted only existing checkout-policy
  LF-to-CRLF notices for the three tracked context Markdown files.
- The final Task 7A owned root contains 1,190 files / 160,652,575 bytes and is
  retained for review.

Final worktree review: three tracked context files are modified; the seven Task
7A deliverables are untracked; the eight pre-existing untracked Task 6 rerun
documents remain unmodified and untracked. There are no production-module or
configuration changes. No commit, push, or pull request was performed.
