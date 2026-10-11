# Task 5B — diagnostic capability demonstrated within declared limits

Date: 2026-09-28. Contract: `first_release_task0_v1`.
Method: `batch_lag_half_drift_descriptive_v1`.

Task 5B is complete: the frozen fault-to-check matrix, seeded clean/fault
fixtures, measured detection and false alerts, integrity/support checks, and
explicit blind spots meet the declared acceptance criteria. This is **bounded
synthetic capability evidence**, not validated universal independence, production
preflight, or a replacement for exhaustive game-level diagnostics. Tasks 5A's
all-k correctness execution and 5C's production integration remain outstanding.

## Delivered work

- [Protocol](task5b_protocol.md), [pre-assessment lock](task5b_protocol_lock.json),
  and [config-seeded recipe](../../configs/validation/task5b_capability.json).
  Thresholds, seeds, strengths, and acceptance criteria were frozen before the
  full assessment and have not changed after inspecting its results.
- [Pure diagnostic arithmetic](../../src/farkle/analysis/batch_monitor.py):
  complete expected batch inventory, integer count/conservation checks, six
  ratio responses, lag-one Pearson association and standardized half drift.
  Missing support, zero-denominator batches, constant or undefined series,
  fewer than 30 batches, and unassessed batch counts cannot yield reassuring
  labels. Boundaries are strict. Corrupt inputs fail explicitly.
- [Reproducible driver](../../scripts/task5b_capability.py): 768 atomic fixture
  units, deterministic reduction, exact source/recipe/code identity, bounded
  two-worker execution, compatible replay and per-unit recovery. The driver
  performs no production games and adds no RNG namespace.
- [Retained numerical results](task5b_capability.json) and
  [validation record](task5b_validation.json). The full plan, per-series results,
  exact unit inventory, and final summary digest remain at the recipe-owned
  `data/task5b-capability-v1` path. These are standalone evidence files, not
  canonical PE/MC/MR artifacts or production completion stamps.

## Measured capability

Every row below uses 64 configured seeded families. Each family contains 16
synthetic sentinel count vectors, six metrics per vector, and two checks per
metric: **96 series and 192 check opportunities**. Faults affect sentinel zero;
credit requires the intended check to detect that injected sentinel. The other
15 sentinels remain clean. Counts below are family-level outcomes, not 64 times
192 independent replications.

| Scenario | Intended check | 30 batches | 100 batches |
| --- | --- | ---: | ---: |
| Clean | Any warning anywhere in the family | 0/64 | 0/64 |
| AR(1), +0.85 | Lag one | 59/64 | 64/64 |
| AR(1), -0.85 | Lag one | 64/64 | 64/64 |
| 4-latent-SD half step | Half drift | 64/64 | 64/64 |
| Weak AR(1), +0.2 | Lag one | 0/64 | 1/64 |
| Weak 0.5-latent-SD half step | Half drift | 0/64 | 0/64 |

All primary B=100 gates passed: at least 58/64 detections for each strong
alternative and at most 4/64 clean families with any warning. All nine targeted
summary integrity faults failed explicitly; all four unsupported/degenerate
cases withheld statistical conclusions. Existing certified-publication tests
also passed for missing games, a same-count duplicate/hole, and a wrong roster.
Those upstream checks are essential: internally consistent aggregate counts
alone cannot authenticate the original games or catch every data fault.

The clean false-alert rate observed here was 0/64 at each B, across 12,288
check opportunities per B. This does not establish a zero or tightly calibrated
false-alert probability. The responses share counts, the checks share series,
and paired fault/clean cases share innovations. This is one finite synthetic
model, with completed win rates near one half and positive conditional support;
it is not an empirical survey of the production strategy grid, all k regimes,
or rare events. No per-check or dataset-wide family-wise guarantee is claimed.
The 30-batch results are disclosed stress evidence, not a new power guarantee.

The weak alternatives document low sensitivity deliberately. A no-concern result
cannot exclude a smaller effect, other lags, different drift shapes, faults in
unselected strategies/seats, or problems outside this model. The minimum batch
count is not a promise of independence or adequate uncertainty coverage.

## Demonstrated blind spots

For 100 batches of 1,000 binary observations, the randomized and fault-ordered
versions have identical batch totals. Game-level lag-one correlation changes
from **-0.003335** to **0.996020**, while both versions have exactly the same
batch lag (-0.150075), half drift (0.043080), and no-concern classifications.
The batch monitor therefore cannot substitute for observation-level lag checks.

In the common-fault example, the same step-contaminated summaries are copied
into two nominal roots. Their measured between-root difference is zero, yet
both within-root monitors emit 12 warnings. This deliberately shared corruption
is not two independent clean realizations: root agreement alone cannot certify
correctness. Retain independent-root comparisons alongside integrity checks and
within-root monitoring; do not make one erase the others.

## Interpretation and runtime boundary

The warning rules are descriptive effect policies: absolute lag above
`max(0.5, 4/sqrt(B-1))` or absolute standardized half drift above 1.5. They are
not significance tests. The current 1.96/sqrt(n) descriptive bands are not used
as automatic failures, and H2H's 2% superiority control is unrelated.

Future production use is limited initially to complete certified all-seat
vectors at the assessed B=30 or B=100, with at most 16 deterministically selected
strategies per required root/k. Other B values are insufficient until a versioned
assessment extends coverage. Any zero-denominator batch withholds that entire
series rather than deleting or reordering it. A degenerate safety-rate vector
also withholds; zero observed safety events do not establish independence.

Task 5C must check capability identity/applicability and authenticated source
coverage before using substantive labels. The helper's arithmetic results do
not grant that permission on their own. An `investigate` result must retain
statistic, threshold, support and coverage and surface a warning qualifying
batch-based precision. It must not silently stop/replan statistical sampling,
increase games or bootstrap replicates, alter candidates, or relax integrity
failures. Excluded or unsupported diagnostics must never be described as passed.

## Engineering verification and recovery

The final 768-unit assessment took **17.213 seconds** on this machine using two
workers. A one-worker no-force replay reused every unit in **5.981 seconds**;
all **771 evidence files** retained exact bytes and timestamps. After deliberate
corruption of one unit and the summary, verification rejected the graph. Recovery
took **6.169 seconds**, reused 767 units, and restored every original byte hash;
only the two corrupted files changed modification time. These bounded-fixture
timings are not production capacity estimates.

An earlier replay check caught a summary-order bug: parallel completion order
affected manifest ordering. The final implementation sorts inventory paths
before publication. A regression test reverses completion order and verifies
unchanged bytes/mtime. Another checks that missing final digest, changed summary
bytes, or a missing referenced unit invalidate the evidence graph. No statistical
threshold, seed, strength, or acceptance criterion changed during these fixes.

Validation: **50 focused checks passed**, followed by **22 final monitor/driver
checks**, for **52 distinct passing tests**. Repository Ruff, targeted Black and
Pyright, Mypy across 91 source files, protocol/source identity checks, and final
evidence graph verification passed. A full production-pipeline oracle was not
rerun: no existing stage or consumer imports this new monitor yet. Existing
uncommitted implementation/config/test files recorded by Task 4 remain unchanged.

Next is **Task 5C**: implement the authenticated bounded preflight and its actual
workload, resolve production-topology applicability (the current mega requests
16 workers), wire lightweight monitoring into required coverage, and make
exhaustive lag diagnostics independently optional across config, stage execution,
reporting, completion, and release audit. Production diagnostic behavior and
all game targets remain unchanged in Task 5B.
