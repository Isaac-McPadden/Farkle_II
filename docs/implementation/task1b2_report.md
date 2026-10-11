# Task 1B and Task 2: durable outcomes and summaries

Implemented 2026-09-25 against the frozen Task 0 contracts. The existing Task 1A
kernel feeds a companion `outcome_publication` producer owned by metrics.
No simulation rules, seeds, game targets, performance estimands, or uncertainty
procedures change. Task 3 remains the next dependency.

## Published results

All paths are non-mutating AppConfig getters under the existing metrics/by_k
scope. The producer publishes:

| Artifact | Contents |
| --- | --- |
| `outcome_frequencies.parquet` | Exact positive-count score, competition-rank, doubled-midrank, losing-margin, winning-margin, and stopped-score bins; actual seats and seat=0 |
| `outcome_batch_stats.parquet` | Complete batch/strategy/actual-seat grid of seven aligned counts/sums for Task 3 |
| `outcome_batch_inventory.parquet` | Certified workload coverage, termination totals, roster/source hashes, coordinate digest |
| `outcome_support.parquet` | Counts and support status for every roster strategy and actual/overall seat |
| `outcome_summaries.parquet` | Six distribution means and q05/q25/q50/q75/q95; completed/attempted win/loss rates, safety rate, mean completed deficit, conditional losing/winning means |

Q has 44 rows per strategy/view, ordered by its declared primary key. Quantiles
use the first value whose cumulative count reaches `ceil(n*p/100)`, with no
interpolation. Midranks are divided by two only when producing estimates.
Conditioning is explicit, including `all_attempted`. Undefined estimates are
null, distinguished as `verified_zero`, `no_completed_support`, or
`no_conditional_support`. Observed zero margins retain positive count bins.
With positive completed support and no losses, mean deficit is zero while the
conditional losing margin remains undefined. For positive loss support,
`deficit/completed = (losses/completed)*(deficit/losses)`.

Overall views sum player exposures and histogram counts across actual seats.
They do not average seat rates. Repeated strategy appearances are distinct
exposures when the declared roster multiplicities support them.

## Completeness and provenance

The producer requires current simulation and curate completion, the curated
source, the workload plan, and the strategy manifest. It checks the planned
row count, exact contiguous `(shuffle, game)` coordinates at every source-slice
position, and each strategy's multiplicity in every shuffle. Each complete
batch must reconcile attempted/completed/safety exposures, official wins,
losses, frequency masses, and margin sums. Only then may its absent seat cells
be emitted as verified zeros. Missing batches, records, partitions, or required
roster support fail; none are filled with zeros.

The coordinate digest uses the frozen little-endian `<qhqi` encoding. It is
generated from the scheduled keys only after every source position has matched
that schedule and authenticated source units have completely covered the batch.
It supplements the input identities rather than replacing them.

A unique-ID manifest is sufficient when its cardinality equals the scheduled
strategy count. If the schedule includes duplicate IDs, their multiplicities
must be explicitly authenticated in the roster; they are never inferred from
surviving rows. Existing incompatible manifests fail explicitly.

Metrics cache-key version is 4. Its shared version registry now includes
`outcome_distribution_method_version=1`, so the new and existing metrics
outputs have compatible completion identities. The new producer declares
actual schemas and consistency columns in all sidecars.

Metrics completion enumerates F/B/I/S/Q plus `plan.json` and `publication.json`.
The authenticated checkpoint inventory names every unit, input identity,
stage identity, and final manifest. The graph snapshot and final byte-deep
audit include the retained `_outcome_units` artifacts and validate their typed
unit stamps and manifests. Execution telemetry and quarantine files remain
operational scratch, excluded from the release graph.

## Bounded processing and recovery

The configured analysis byte allowance controls projected input slices,
fixed-width NumPy records, sorting, stream reads, Python row buffers, and Arrow
coalescing. Merge fan-in is at most 16. Reads and reductions never accumulate
a run-wide Python histogram. Roster/plan metadata and conservative Parquet
footer estimates also have admission checks. Before upstream authentication or
Arrow decoding, required source files' serialized footer sizes are checked
against one quarter of the analysis allowance. A large number of tiny row
groups can otherwise consume substantial native metadata memory despite small
projected row batches. Exceeding the admission limits fails
explicitly and requires a larger configured allowance.

`_outcome_units/plan.json` freezes physical slice boundaries. Named source,
batch-merge, certified-batch, normalized, and pooled-merge subdirectories each
own a partition manifest and exact enumerated unit paths. This is a physical
refinement of Task 0's illustrative slice/merge layout, within its prescribed
checkpoint root. Independent units use the existing parallel executor; merges
are streamed, staged atomically, and independently resumable.

Valid units survive interruption, corruption elsewhere, worker-count changes,
and larger byte allowances. A smaller allowance can reuse a fully complete
publication, but unfinished work requires the original allowance or explicit
force to repartition. A changed source/method/code discards only this producer's
validated, config-owned checkpoint directory before establishing a new plan.
Unchanged plans persist even during a forced recomputation.

Canonical B retains all batch numerator/denominator vectors selected by Task 0.
Tests reconstruct completed deficit and conditional margins using B alone,
including zero-loss batches. No quantile intervals were selected in Task 0;
the retained batch frequency runs are recovery artifacts, not a promise of
additional uncertainty estimates.

## Qualification

Focused tests cover hand-calculated ties and quantiles, safety-only support,
certified zero exposure, explicit repeated-strategy multiplicities, retained
batch arithmetic, coordinate/roster faults, slice and row-group boundaries,
integer overflow, output corruption, lost units/sidecars, actual Windows-spawn
resume with changed worker counts, forced recomputation, logical equality under
changed byte allowances, and final-audit inclusion/rejection.
The final combined outcome, metrics-wiring, release-audit, partition-executor,
stage-registry, and completion run passed **110 tests in 71.37 seconds**.

The reproducible benchmark runs with `python -m scripts.benchmark_outcome_publication`.
It declares a 32 MiB analysis allowance and a 128 MiB incremental RSS measurement
allowance before execution. Fixture generation is excluded from producer timing.
The default fixture uses k=12, 5,160 strategies, four shuffles/two batches,
1,720 games, and 20,640 exposures, with broad distinct scores and safety outcomes.
Detailed measurements and source hashes are retained in `task1b2_memory.json`.

An initial measurement exceeded the allowance because tiny output row groups
made Parquet footer metadata dominate memory. Bounded Arrow coalescing fixed
that problem and substantially reduced storage. The latest measurement used
about 29 MiB incremental RSS and 28.9 seconds, with approximately 0.50 seconds
for authenticated resume. Final artifacts total about 1.59 MiB and retained
checkpoints about 1.75 MiB. These are finite-fixture measurements, not a
production-capacity approval or a universal process-tree memory bound.

A second fixture concentrates broad support into 12 strategies over 1,024
shuffles; its separate measurement is `task1b2_broad_support_memory.json`.
The initial highly fragmented source exceeded the 128 MiB incremental RSS
measurement allowance (about 289 MiB incremental RSS). This is preserved in
`task1b2_fragmented_source_initial_memory.json`, not represented as a passing
measurement. The admitted rerun uses a declared 128 MiB analysis allowance and
512 MiB incremental RSS measurement allowance. Native source-footer overhead
is therefore an explicit resource limitation, separate from histogram growth.
Its 16.7 MiB source footer passes that larger admission; the rerun used 286.4
MiB incremental RSS, 27.4 seconds for publication, and 0.96 seconds for resume.
This does not retroactively pass the initial 128 MiB measurement allowance.

The real simulation-to-report integration test passed, including normal
two-root orchestration, root/pair artifact and statistical oracles, exact
completion inventories, the final byte-deep release audit, and byte-stable
ordinary replay. The measured workflow took **210.141 seconds**; pytest reported
**1 passed in 212.40 seconds**. Its existing 240-second assertion was unchanged.
An earlier concurrent validation run passed all functional assertions but took
291.372 seconds and failed that timing assertion. The successful rerun ran
without competing benchmark/test jobs.

Integration ran in the isolated clean Git snapshot
`C:/Users/imcpa/AppData/Local/Temp/farkle-outcomes-verify-2vh8oi9h`, commit
`21a76697d3fe52fdc36ec4bfb1f838e78e4beaae`. All eight relevant production modules
were checked byte-for-byte against the workspace. The working branch was not
committed or changed by that qualification checkout. Both final benchmark
JSON files retain SHA-256 identities for the final producer, kernel, and script.

Repository Ruff and Mypy over all 89 source files passed; targeted Black and
Pyright cover all Python files changed by this work package. This does not
claim a fresh pass of repository-wide Black/Pyright or the entire test suite;
the pre-existing unrelated findings remain described in the Task 9 report.
Full release qualification remains Task 10. No production simulation or
historical-results migration was performed.
