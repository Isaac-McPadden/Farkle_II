# Task 1A: exact outcome accumulator and semantic tests

Completed September 24, 2026, following Section 14 of the updated implementation
backlog. The implementation is
[`outcome_distributions.py`](../../src/farkle/analysis/outcome_distributions.py);
the semantic oracle is
[`test_outcome_distributions.py`](../../tests/unit/analysis/test_outcome_distributions.py).

## Implemented computation

`accumulate_outcome_slice` accepts one bounded Arrow slice of compatible curated
rows plus the explicit root, k, and workload-plan `shuffles_per_batch`. It
returns an `OutcomeSlice` with exact, sorted, explicitly typed tables:

| Table/API | Contents |
| --- | --- |
| `batch_frequencies` | Actual-seat frequencies keyed by root/k/batch/strategy/seat/distribution/value |
| `batch_stats` | Actual-seat attempted, completed, safety, official-win, completed-loss, deficit-sum, and victory-margin-sum counts, keyed by root/k/batch/strategy/seat |
| `coordinates` | Consumed root/k/batch/shuffle/game keys and termination status for later coverage certification |
| `frequency_views()` | Partial F, including seat=0 by summing actual-seat frequencies across the slice |
| `support_views()` | Partial S with actual-seat and overall counts/sums; only observed support is emitted |

The six distributions are `completed_score`, `competition_rank`, `midrank`,
`losing_margin`, `winning_margin`, and `stopped_score`. Counts and sums are
int64 with overflow rejection; strategy IDs are int32. Midranks are stored as
doubled integers with `value_scale=2`; other distributions use scale 1.
The F/S/B field definitions match Task 0. Batch frequencies additionally retain
the batch key needed for future frequency-based resampling. No summary estimates
or intervals are calculated by this task.

For scores 10000/8000/8000/5000, descriptive competition ranks are 1/2/2/4 and
doubled midranks are 2/5/5/8. Recorded ordinal engine ranks remain 1/2/3/4.
The existing engine winner, score ordering, and seat tie order are validated;
the input is never rewritten. Official seat identity determines wins and losses,
including when the same strategy occupies multiple seats. A tied official loser
retains a zero losing margin, and a tied official winner retains a zero victory
margin. Descriptive first place does not add an official win.

Safety-limit games contribute stopped-score frequencies and attempted/safety
support only. Their recorded winner, rank, and margin fields must be null.
Completed games must retain consistent winner/rank/margin fields and safety
flags. Missing actual-seat data, wrong physical types or outcome versions,
invalid roots/k, wrong batch assignment, and duplicate/out-of-order game keys
fail explicitly. Padded seats greater than k are ignored. Repeated-strategy
exposures are valid under Task 0, so the existing simulation-row validator's
distinct-strategy restriction is deliberately not imported into this analysis.

Empty slices return the same nonnullable Arrow schemas as nonempty slices.
Zero conditional support is represented by explicit support counts and no
frequency rows for that conditional distribution. For example, completed=0
with positive safety support is distinguishable from completed>0, losses=0.
An observed zero-margin bin has positive frequency count; it is not missingness.

## Read interface and bounds

`iter_curated_outcome_slices(cfg, k, shuffles_per_batch=...)` reads the existing
`cfg.ingested_rows_curated(k)` path, authenticates its sidecar and curated/by-k
scope, and projects only required fields through the existing byte-aware
Parquet iterator. The caller supplies the resolved workload-plan batch width;
I/O row/byte limits cannot redefine statistical batches. Duplicate/out-of-order
coordinates across I/O slices are rejected. A missing or empty required source
fails; the adapter publishes no files or completion markers.

The slice budget is `resources.stage_batch_bytes['analysis']`. Admission reserves
a quarter each for input, fixed-width records, grouping, and output views, with
a conservative per-exposure allowance. Tiny budgets and oversized direct slices
fail before expanded record allocation. Frequency records use fixed-width
NumPy arrays; dictionary growth cannot track an entire run's score support.
Malformed variable-length rank lists are rejected before Python expansion.
Only an admitted source slice becomes Python rows. The reader yields slices
incrementally; callers must consume them incrementally as well.

The kernel performs small independent slice arithmetic synchronously. Task 1B
will schedule independent durable slices through the existing process helpers;
the current forward adapter preserves order for boundary validation. This task
does not add a process pool around individual tiny game calculations.

## Completion boundary and next work

These tables are **partial reductions**, not authenticated final F/B/I/S
publications. They do not certify the absence of an exposure. In particular:

- No `verified_zero` row is emitted. An absent strategy/seat cell remains absent
  until the complete authenticated workload and roster have been checked.
- Task 1B must bind workload-plan/roster/source identities, validate complete
  game-coordinate coverage and roster multiplicities, and merge partial counts
  using bounded durable units. Merely adding partial row counts is insufficient
  to establish completeness or detect a missing batch.
- Only that certified merge may emit inventory I and the full B support grid,
  including genuine zero-exposure cells. Sparse missing value bins and lost
  required support records must remain distinct.
- Task 1B/2 must publish the specified atomic artifacts, reuse authenticated
  completed children after interruption, add AppConfig output getters and
  metrics/final-audit inventory, and generate Q summaries. Task 3 adds U.

No simulation code, targets, RNG, existing performance inputs, pipeline stage
order, or existing metrics output contract was changed. The new arithmetic
method version is 1. Existing repository-wide code freshness still applies;
this does not make historical outputs compatible with changed code. The next
execution-order package is **Task 1B plus Task 2**.

## Verification

The repository Python 3.12 environment ran:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/analysis/test_outcome_distributions.py tests/unit/utils/test_schema_helpers.py -o addopts='' -q
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pyright src/farkle/analysis/outcome_distributions.py tests/unit/analysis/test_outcome_distributions.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m black --check src/farkle/analysis/outcome_distributions.py tests/unit/analysis/test_outcome_distributions.py
.\.venv\Scripts\python.exe scripts/check_terminology.py
```

**47 tests passed**: 44 new semantic/reader cases and 3 existing schema cases.
They cover hand-computed ties and margins, all-tied repeated-strategy self-play,
undefined conditionals, missing/malformed outcomes, typed empties, source
authentication, duplicate coordinates across slices, padding, exact integers
above 2^53, overflow, and slice admission. An independent Counter oracle checks
all frequencies and seven sufficient statistics at k=2/4/12 under 1/7/43-game
slice sizes, including multiple statistical batches and safety-limit games.
Overall histograms reconcile bin-for-bin with seat counts; game/exposure/win
conservation and attempted/completed/safety identities hold.

Mypy passed for all 88 source files. Pyright and Black passed for the new files;
repository Ruff and terminology checks passed. The inherited repository-wide
Black/Pyright findings recorded in the Task 9 report were not changed here.
No full production-sized distribution-memory qualification, full test-suite
run, or pipeline publication is claimed: those are later work packages, not
requirements satisfied by this arithmetic-only acceptance run.
