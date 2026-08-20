# Task 4C simulation execution remediation

## Verdict

Task 4C is complete. The protected Windows CLI now demonstrates a real 12-process
spawn pool, useful live progress, sequential roots and player counts, bounded
interruption cleanup, exact manifest-based resume, and byte-identical canonical
outputs. No scientific, RNG, schema, canonical-layout, or artifact-contract
identity changed.

The Task 4C review gate can unblock Task 5A. This does not itself declare the
full pipeline production-ready; the existing Task 5A capacity/readiness stop gate
still applies.

## Proven root cause and incident reconstruction

The original health-versus-process observation combined two different kinds of
state. `pipeline_health.json` recorded a static plan (`requested=12`,
`resolved=12`, `effective=12`), not a live executor observation. The late process
snapshot recorded a later instant when no tournament workers were present.

The preserved incident root proves that the pool was not serial and was not
suppressed at startup:

- `2p_rows/manifest.jsonl` has 340 durable records;
- those records name 12 distinct worker PIDs;
- their timestamps span `2026-08-20T03:05:44Z` through
  `2026-08-20T03:06:00Z`, beginning 21 seconds after the rendered tournament
  start;
- all 12 workers wrote canonical row shards.

Therefore, the statement "the executor topology contained none" was true only
for the later inspection, not for pool construction or initial execution. The
old implementation emitted no actual topology, scheduler state, or worker
progress, so the preserved logs cannot identify the event that removed the pool
or the exact parent wait state. That uncertainty is retained rather than filled
with an assumption.

The code investigation and bounded reproductions did establish the corrective
causes:

1. Production v3 ran a historical 2,000-game serial throughput calibration
   before submitting pool work. A protected reproduction spent about 90 seconds
   there before worker creation.
2. The nested guard trusted `FARKLE_PROCESS_POOL_ACTIVE=1` alone, even if a
   top-level process inherited a stale marker.
3. `process_map` had no observed created/live/peak worker contract and therefore
   could not reject plan/topology divergence.
4. The Windows cancellation path called `shutdown(wait=True)` after terminating
   workers, an unbounded wait if an interrupted result pipe or executor manager
   did not settle.
5. Resume used row/metric manifests only when the periodic pickle checkpoint
   also existed, replaying already authenticated work after an interruption.
6. Hard termination could leave `._tmp_*` atomic staging files. Recovery ignored
   them, but final publication later enumerated them as ordinary outputs.
7. Operational fields lived only in logging `extra`; the terminal formatter
   printed the same opaque heartbeat repeatedly.

## Corrected architecture

The two root workflows now run in a plain fail-fast loop. Root two cannot start
simulation or analysis until root one returns successfully. The compatibility
setting `orchestration.parallel_seeds: true` is rejected before run-start or
health publication. `runner.run_multi` remains sequential across `k`; it passes
the full resolved `sim.n_jobs` budget to each active tournament.

The nested guard now requires both the environment marker and a genuine
`multiprocessing.parent_process()`. A stale inherited marker cannot collapse the
top-level protected process, while real executor children still suppress nested
pools.

`process_map` now records requested, resolved, effective, created, live,
peak-live, and cleanly terminated workers, PIDs, executor mode, construction and
shutdown state, parent PID/thread, multiprocessing parent, and nesting marker.
After initial submission, a real executor whose created/live topology does not
match `min(resolved workers, pending work)` raises `ProcessPoolTopologyError`.
There is no silent serial fallback.

Cancellation is bounded: cancel queued futures, terminate workers, request
nonblocking executor shutdown, join to a deadline, kill survivors, join again,
and bound the executor-manager join. Native thread caps, the Windows Job Object,
memory admission/backpressure, deterministic bounded submission, and resource
failure classification remain in force.

The unnecessary production serial calibration was removed. The legacy direct
API retains calibration only when it explicitly owns workload-plan publication.

## Telemetry

Simulation workers report completed shuffles/games through a bounded shared
endpoint. The supervisor heartbeat combines those counters with scheduler,
checkpoint, and process-tree state without scanning artifact trees. Structured
snapshots are written atomically to `pipeline_telemetry.json`.

The terminal distinguishes intentional serial, pool startup, pre-first
submission, memory backpressure, waiting futures, cancellation/cleanup, pool
creation failure, checkpoint publication, and completion. A captured example is:

```text
Heartbeat: seed=94512 phase=simulation k=2 games=1872/2400 (78.0%) shuffles=468/600 chunks=12/20 rate=14.6/s recent,14.6/s avg eta=1m05s workers=12/12/12 requested=12 mode=process_pool pending=8 in_flight=8 checkpoint=2026-08-20T08:25:28Z@588 rss=2854.1MiB phase_elapsed=1m31s run_elapsed=1m31s
```

## Windows-spawn benchmark

The standalone driver is
`scripts/benchmark_task4c_simulation_execution.py`. It uses the real protected
CLI and disposable owned root
`data/farkle-task4c-simulation-execution-v1`. The speed fixture retains the
unchanged fast-config 80-strategy grid: seed 94512, `k=2`, 600 shuffles, 40 games
per shuffle, 24,000 games, and 20 deterministic process blocks.

| Workers | Simulation | Protected CLI | Simulation games/s | Created/peak live | CPU seconds | Peak RSS | Peak threads |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 86.0 s | 96.25 s | 279.1 | 0/0 (intentional serial) | 92.17 | 487.8 MiB | 57 |
| 2 | 43.0 s | 52.59 s | 558.1 | 2/2 | 103.31 | 937.0 MiB | 71 |
| 12 | 21.0 s | 30.55 s | 1,142.9 | 12/12 | 182.00 | 3,410.8 MiB | 149 |

The identical simulation phase is 4.095x faster at 12 workers. End-to-end
protected-CLI speedup is 3.151x because the short fixture includes about 14.4
seconds from launcher start until all 12 spawned workers are observable plus
fixed finalization. The result meets the approximately 4x CPU-bound target while
also explaining why total wall speedup is lower. The 12-worker process tree
averaged multiple CPU cores and stayed well below the fixture's 6 GiB hard
boundary. This is bounded `k=2` evidence, not a full-pipeline runtime estimate.

All worker-count cases produced canonical digest
`712514720c7f7526e9d0ed30e61d7979cdc2ce8d3b9c69c4bca0aab61f3cb616`
and logical digest
`c1d395355c893e91dd731fc0bf2e486f3a1c96d4d4333762ece3abede3fa2f66`.

## Interruption and exact resume

The benchmark launched a 12-worker row-producing tournament, waited for durable
units, delivered the normal console break, verified process quiescence, and then
resumed the same root without `--force`.

- shutdown latency: 0.171 seconds;
- orphan workers: none;
- manifest-qualified rows before resume: 78;
- durable rows reused with unchanged mtime and SHA-256: 78;
- incomplete units: safely recomputed;
- worst-case exposure: at most one in-progress shuffle per worker, 48 games;
- canonical reference/resume digest:
  `a2cca89487e5d6e1532b6c96058f5ca51a2e54e2b1c19f3578157828f272f939`;
- logical reference/resume digest:
  `b3a7461e46967280d0f47e0b144c74f815c270ea3fcf59f34fb4b139a3df707e`.

Authenticated manifests are now independent resume authorities. Recognized
atomic staging names are removed before resume and excluded from completion
enumeration; arbitrary hidden files are preserved. This closes the finalization
failure found during the first benchmark attempt.

## Sequencing and failure contracts

Focused tests prove:

- root event order is `start 1`, `finish 1`, `start 2`, `finish 2` with no active
  interval overlap;
- root-one failure prevents root two and pair work;
- `parallel_seeds: true` publishes neither run-start work nor health success;
- `k` calls are non-overlapping and each sees the unchanged full `sim.n_jobs`;
- stale environment markers do not suppress the protected parent;
- real spawned children resolve nested process pools to one;
- topology mismatch, worker failure, interruption, and cleanup cannot publish
  false success.

## Verification

- Standalone protected benchmark: passed.
- Focused simulation/orchestration/parallel/telemetry/publication benchmark suite
  after final contract additions: 336 passed.
- Repository-wide Ruff: passed.
- Mypy over 86 source files: passed.
- Black over changed Python: passed.
- Pyright over changed production, benchmark, and tests: passed.
- JSON parsing and `git diff --check`: recorded in the final handoff.

The broader focused command had 342 passes and six failures, all six parameters
of `test_simulation_completion_mutation_matrix`. This is a pre-existing test/code
incompatibility: the test requests promotion of unsealed v3 bytes, while the
unchanged accepted Task 4B production path fails closed. Task 4C does not touch
that decision path, and the failure is not counted as a regression.

## Files and hygiene

Production changes are confined to the CLI heartbeat owner, two-seed
orchestration, simulation runner/executor, and shared parallel/telemetry helpers.
Focused tests and the Task 4C benchmark were added or updated. This report,
machine evidence, and Codex context notes are the only documentation changes.

Changed files:

- `src/farkle/cli/main.py`
- `src/farkle/orchestration/two_seed_pipeline.py`
- `src/farkle/simulation/run_tournament.py`
- `src/farkle/simulation/runner.py`
- `src/farkle/utils/parallel.py`
- `src/farkle/utils/telemetry.py`
- `scripts/benchmark_task4c_simulation_execution.py`
- `tests/unit/orchestration/test_seed_workflows.py`
- `tests/unit/simulation/test_runner_branches.py`
- `tests/unit/simulation/test_simulation_publication_v3.py`
- `tests/unit/utils/test_parallel_files.py`
- `tests/unit/scripts/test_benchmark_task4c_simulation_execution.py`
- `docs/remediation/task4c_simulation_execution_report.md`
- `docs/remediation/task4c_simulation_execution.json`
- `docs/codex_context/context_prompt.md`
- `docs/codex_context/repo_map.md`
- `docs/codex_context/testing_and_review_map.md`

The starting commit was `baecd8becd564217d43d366bb07f108f76ddb9f3`
with a clean worktree. No historical result tree was modified or deleted; the
interrupted production root remains preserved. No full pipeline was run, Task 5
was not begun, and no commit or push was performed.

## Remaining risks

- The old incident lacks the executor/scheduler events needed to identify the
  exact trigger for its post-worker wait; future runs now record that state.
- Spawn/import and fixed finalization costs are visible for short fixtures and
  limit end-to-end scaling even when the tournament exceeds 4x.
- Peak RSS scales with worker count. The measured 12-worker tree used about
  2.85 GiB more than intentional serial, an explained process cost that remains
  subject to the configured Job Object and scheduler limits.
- Task 4C is a bounded simulation result, not evidence for RNG, H2H, OneDrive,
  or complete two-root runtime. Task 5A remains responsible for that readiness
  assessment.
