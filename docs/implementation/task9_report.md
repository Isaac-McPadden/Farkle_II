# Task 9: bounded seat aggregation and child reuse

Implemented September 24, 2026 against the updated implementation backlog,
Section 12 / execution-order step 2. This report supersedes historical source
findings only for the changes and evidence described here.

## Implementation

- Seat batch histories stream through the existing byte-aware Parquet reader.
  Numeric accumulation is bounded by the admitted strategy roster and seat
  count, independent of the number of games or deterministic batches. Only
  compact totals enter pandas. The batch-level artifact remains available for
  future uncertainty procedures; this task adds no uncertainty estimator.
- Count production also uses byte-bounded reads and numeric accumulation.
  Missing required strategy/coordinate/status fields, invalid winners, count
  conservation failures, and integer overflow fail explicitly. Unobserved
  strategy/seat combinations are not fabricated as zero observations. The
  future complete-support/verified-zero inventory remains a Task 1 contract.
- Independent per-k cells use the existing process scheduling and memory guard.
  Cross-k summaries align the common support by indexed intersections instead
  of repeatedly scanning every strategy frame. Existing equal-k, declared-k,
  attempt/completion conditioning, and mirrored pairing definitions remain.
- Self-play reads are byte bounded and its roster map has an admission limit.
  Mirrored partition collection checks its record limit before appending.
  Mirrored reduction shares a byte allowance across at most 16 readers; larger
  inventories use atomic temporary merge levels. Final publication is atomic;
  interrupted final merges reuse authenticated durable source partitions.
- Metrics no longer passes `force=True` merely because the parent is stale.
  Seat analysis adds authenticated per-k completions and a separate completion
  for each batch-count artifact plus its manifest. Thus an interrupted parent
  reuses finished cells, and a damaged summary can reuse valid count history.
  Explicit seat `force=True` still rebuilds the cells. Existing source, sidecar,
  configuration, code, and completion authentication remains in force.
- Empty and nonempty seat outputs now share explicit Arrow schemas, including
  canonical int32 strategy identifiers, int16 k/seat values, int64 counts, and
  list<int16> common-k support. Metrics cache-key version advances from 2 to 3.
  Historical outputs are not relabeled or automatically exempted from code
  identity invalidation.
- The touched count manifests already use actual authenticated single-entry
  JSONL through `run_streaming_shard`; their producers and consumers agree.
  No cosmetic manifest renaming or unrelated format migration was needed.

## Memory qualification

Reproduce with the repository Python 3.12 environment:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_task9_seat_aggregation.py
```

The deterministic synthetic history has 5,160 strategies, 12 seats, and 100
batch IDs: **6,192,000 count rows**, reduced to 61,920 strategy/seat cells and
12 population cells. A fresh child process measures aggregation after imports;
fixture construction runs in its parent. Exact exposure, win, safety-limit,
and population totals are asserted.

The final measurement used the clean verification snapshot below; an earlier
fresh-process repetition also passed at 25.17 MiB incremental RSS. Machine-readable
evidence is in [task9_memory.json](task9_memory.json).

The allowance was declared before measurement: 32 MiB each for reader/state
admission and 128 MiB incremental sampled process RSS for this workload.

| Measurement | Result |
| --- | ---: |
| Imported-process baseline RSS | 207,601,664 bytes |
| Peak sampled RSS | 230,019,072 bytes |
| Increment above baseline | 22,417,408 bytes (21.38 MiB) |
| Declared incremental allowance | 134,217,728 bytes (128 MiB) |
| Aggregation elapsed | 1.043 seconds |
| RSS sampling interval | 5 milliseconds |
| Exact-count assertions | Passed |

This qualifies the count-history reduction, not a complete production pipeline
or a universal process-tree ceiling. Native decoder allocations, pandas
temporaries, parallel worker count, and other stages still contribute to RSS.
The reader/state setting resolves from `resources.stage_batch_bytes.seat_analysis`
when present, otherwise `analysis`. Oversized rosters/compact summaries fail
with an explicit memory admission error rather than silently growing without
bound; raising the configured allowance is the current larger-roster path.

## Recovery and correctness evidence

- 21 focused seat, metrics wiring, and safety-limit tests passed. These include
  an independent count oracle across 31 batches and two byte settings, a ban on
  whole-file count reads, missing-field failures, overflow/admission limits,
  empty schemas, a 35-shard merge, existing mirrored shard corruption/recovery
  with a changed worker count, interrupted-parent reuse, explicit-force and
  method-change invalidation, and exact canonical results across one/two
  per-k workers.
- Recovery checks compare both bytes and modification times of reused child
  artifacts. Corrupting a summary repairs it while preserving its count file;
  changing a source rebuilds the affected cell while preserving the other k;
  removing a required source fails rather than accepting old publication.
- 16 stage-registry and stage-completion tests passed.
- Repository Ruff, Mypy over all 87 source files, terminology checks, and
  Black/Pyright on every changed Python file passed.
- Repository-wide Black reports three existing untouched files:
  `scripts/benchmark_task1b_telemetry.py`,
  `src/farkle/orchestration/run_contexts.py`, and `src/farkle/utils/parallel.py`.
  Repository-wide Pyright reports 29 existing errors in untouched tests:
  all-player metrics, RNG diagnostic branches, root stability, TrueSkill
  screening, and OS memory. These were not folded into Task 9.

Both structural toy-oracle tests and the real simulation-to-authenticated-report
oracle passed: **3 tests in 321.23 seconds**. The latter exercises the normal
two-root pipeline, fresh artifact/graph audit, report claims, pipeline health,
and byte-stable ordinary replay. Combined with the tests above, 40 relevant
test cases passed. The full repository test suite was not run for this bounded
task; full release qualification remains Task 10.

The integration tests and final memory measurement ran from a clean local Git
snapshot at `C:/Users/imcpa/AppData/Local/Temp/farkle-task9-verify-ld5pz6jr`,
commit `0d171b6c377edc2e269dd550643036f901d34e18`. The three changed production
modules and benchmark script were checked byte-for-byte against the workspace;
their SHA-256 hashes are retained in the memory evidence JSON. The isolated
snapshot includes the implementation without committing or changing the user's
working branch. Two supplementary tests (explicit force/method invalidation
and per-k worker equivalence) subsequently passed in the workspace.

An initial integration attempt in the dirty synchronized workspace stalled
during Git provenance collection and was stopped. A diagnostic rerun progressed
through the first structural test but was replaced by the clean-snapshot run.
No provenance checks or dirty-code restrictions were disabled. The successful
snapshot run is the integrated acceptance evidence; its temporary outputs are
not historical production artifacts.

## Scope and handoff

Future distribution producers must follow the existing Task 0 artifact contract:
authenticate durable children separately, reuse valid children after parent
interruption, and invalidate changed inputs/methods/code. Tasks 1A/1B/2 must
include the corresponding corruption and resume tests when those producers
exist. This task does not claim their implementation is complete.

The next execution-order work package is **Task 1A**. Production execution and
the full Task 10 qualification remain separate. No historical result tree was
modified and no production simulation was launched.
