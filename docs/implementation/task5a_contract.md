# Task 5A — bounded preflight workload and result contract

Date: 2026-09-28. Parent contract: `first_release_task0_v1` (September 24).
Recipe: `first_release_preflight_v1`. Status: **specification complete;
runtime implementation and diagnostic capability remain Tasks 5B/5C**.

Read with [Task 0](task0_contract.md), its [artifact envelope](task0_artifacts.md),
and the machine-readable [workload recipe](task5a_recipe.json). This is the
September 24 backlog's Task 5A, not the historical capacity task with the same
number. It changes no production config, simulation targets, streams, stage
order, or artifact versions. No new preflight success artifact is published by
this design task. The complete Task 5 is not yet complete.

## 1. Reconciled foundation and implementation gaps

- [RNG streams](../../src/farkle/utils/random.py) use scheme 2, PCG64DXSM,
  registered purposes, and lossless 64-bit semantic coordinate components.
  Purpose 402 now belongs to outcome resampling. No namespace is allocated here.
- [The static audit](../../scripts/check_rng_contract.py) checks maintained
  Python for prohibited RNG APIs. It is a useful source check, not proof that
  every possible dynamic draw is correctly assigned or independent.
- [Current diagnostics](../../src/farkle/analysis/rng_diagnostics.py) use
  method 4, exact full strategy/matchup identities, semantic ordering, bounded
  partitions, and descriptive lag bands. Stage cache version is 6. Cap exhaustion
  is `blocked_by_cap`, not complete coverage. The reducer maps a nonpositive cap
  to unlimited, but [public validation](../../src/farkle/config.py) rejects
  configured zero; null uses the reducer's default cap. Neither is a disable API.
- `analysis.disable_rng_diagnostics` currently disables the **whole stage**.
  It cannot be reused unchanged to omit exhaustive lags while still requiring
  PE/MC/MR. Task 5C must handle that distinction in config, stage execution,
  completion, reports, and the final audit together.
- [Root stability](../../src/farkle/analysis/root_stability.py) already retains
  independent-root descriptive comparisons and half drift. These remain
  descriptive fixed-design reproducibility summaries, not independence tests.
- [Batch outcome publication](../../src/farkle/analysis/outcome_publication.py)
  already certifies coordinates, batch coverage, roster multiplicity, support,
  and conservation. Reuse its certified sufficient statistics and existing
  all-player batches; do not rescan every game just to build a batch monitor.

The current raw authenticated oracle covers k=2 and k=4; distribution arithmetic
fixtures also cover k=12. These do **not** establish worker/resume coverage across
all eight production k values. The additional matrix below is required before
the new preflight can claim that coverage.

## 2. Frozen correctness workload

The recipe is a standalone **integration** workload, with reduced resolution,
`production_eligible=false`, and `release_eligible=false`. Its game rows must
never become production tournament, screening, bootstrap, or H2H observations.
Use an explicit bounded workload through the existing fixture/low-level
simulation interfaces. Do not weaken the production planner's 100-batch or
minimum-shuffle validation to run this fixture.

| Dimension | Required value |
| --- | --- |
| Validation roots | 1101 and 2202, from the frozen recipe |
| k | 2, 3, 4, 5, 6, 8, 10, 12, each explicitly covered |
| Roster | 120 unique legal strategy IDs, frozen before execution |
| Selection | Sort the generated mega-config strategy manifest by canonical ID; select indices `floor(i*(N-1)/119)` for i=0..119; require N>=120 |
| Shuffles / batches | Shuffles 0..3; two deterministic batches of two shuffles |
| Game profile | Target score 10,000; max rounds 200; existing turn safeguards |
| Safety fixture | For each root/k, only shuffle=0, game=0 has max_rounds=0 |
| Execution variants | workers=1; workers=2; interrupted workers=2 resumed with workers=1 |
| Multiplicity | Exactly 120 roster exposures per shuffle for every k |
| Native threads | One per worker; at most two active simulation workers overall |
| Work order | Stable root/k/variant unit IDs; independent units may execute concurrently within that aggregate worker budget |

Freeze the actual roster, grid-config hash, game-profile hash, expected coordinate
inventory, execution topology, and recipe digest in the plan. The selection is
deterministic design coverage, not a representative random sample of strategies.
No strategy ranking, control win rate, expected advantage, or ordering of strong
and weak strategies is a pass condition. Exact scoring-oracle outcomes are
permitted because they test the game rules, not a performance hypothesis.

For one root/shuffle the game counts are 60,40,30,24,20,15,12,10, totaling 211.
One complete variant therefore owns **1,688 game coordinates and 7,680 player
exposures**. The three variants execute 5,064 attempts, apart from explicitly
bounded replay of unauthenticated work. Safety attempts are retained and do not
trigger replacement in the tournament fixture.

For every root/k in the interrupted variant, authenticate shuffles 0 and 1;
interrupt during staged publication of shuffle 2 before its durable ownership
record. Resume must reuse the first two shuffles and regenerate only missing or
unauthenticated units. One-shuffle durable units bound the extra replay to 422
games and 1,920 exposures over all cells. Thus this matrix has an upper bound
of **5,486 executed game attempts and 24,960 player exposures**, including one
injected failure per cell. Persist injection progress so an ordinary resume does
not inject the same failure again. A second unexpected failure is incomplete
execution; it does not authorize unbounded retries or a larger validation plan.

The supplementary existing fixtures below retain their own finite inputs,
counts, seeds, and profiles. Record their resolved counts separately; do not
include them in the 5,486-game matrix bound. The whole preflight has a declared
30-minute execution ceiling, a 32 MiB source-read allowance, and 64 MiB per
reduction buffer. Stop and checkpoint with incomplete coverage if a bound cannot
be met; never shrink checks silently. These are initial operational budgets to
measure in 5B/5C, not demonstrated capacity or a statistical validity theorem.
The current mega config requests 16 workers: this initial two-worker matrix
does not qualify that topology. Production applicability requires a separately
declared bounded topology extension and its evidence in 5B/5C.
Record peak process-tree RSS, worker topology, elapsed time, and actual I/O.
There is no new fixed total-RSS scientific acceptance requirement.

## 3. Required checks and existing reusable oracles

The recipe lists exact source/test references. Existing tests are reusable
assertion foundations, not an implemented authenticated preflight runner.

| ID | Correctness requirement | Existing foundation / addition needed |
| --- | --- | --- |
| C01 | No prohibited RNG APIs; registered purposes and explicit seeds | Static audit, `test_random_utils.py` purpose and draw-history tests |
| C02 | Same semantic coordinate replays identical entropy/stream bytes; changing root/k/purpose/seat separates selected streams | Existing RNG tests; extend their coordinate fixtures to the declared k set |
| C03 | Exact expected game/exposure coordinate coverage and unique full coordinate ownership | Outcome support certification and duplicate/hole tests; never substitute scalar seed/hash uniqueness |
| C04 | workers=1 and workers=2 give identical canonical logical rows, counts, and scientific summaries | Existing raw/worker publication oracles; add the full matrix above |
| C05 | Interruption/resume gives the same result and reuses authenticated work | Existing publication and checkpoint boundary tests; full matrix with 2-to-1 worker change |
| C06 | Scoring, final-round, tie/official-winner, and safety termination semantics | Engine/scoring tests, raw game oracle, tied/self-play outcome fixtures |
| C07 | A=C+S; C=W+L; summed exposures equal k times corresponding games; official wins equal completed games | Outcome arithmetic and certified publication tests, all required root/k/batch cells |
| C08 | Stale code/config, corrupt sidecars/bytes, missing units, and same-count duplicate/hole fail; no false completion | Authenticated simulation publication and outcome support fault tests |
| C09 | H2H attempts use the immutable pair/root/order/attempt coordinates, preserve safety/replacement accounting and resume | Existing H2H engine-block and replacement-resume tests; their reduced plans remain separate from the tournament matrix |
| C10 | The legacy optional lag path retains semantic ordering, full-key collision handling, honest insufficient/cap status, and exact resume | Existing RNG branch tests; diagnostic arithmetic/coverage correctness only |

Use canonical logical Arrow content sorted by full coordinates when worker count
changes physical chunk boundaries. Require exact integer fields and the existing
canonical floating reduction order; do not introduce a new tolerance to hide a
change. Byte identity is additionally required for immutable plans and already
authenticated durable units reused in the same run. Verify the expected roster
and every coordinate from the frozen plan, independently of observed row counts.
Padded seats are not exposures; repeated-strategy seats in the supplementary
self-play fixture are legitimate distinct exposures.

The million-coordinate RNG test currently compares adjacent enumerated entropy
tuples; it is useful bounded enumeration but is not a global collision search.
The new small matrix must check the **complete expected tuple set** against
observed ownership. Neither distinct sampled bytes nor this finite enumeration
proves universal independence of streams.

## 4. Statistical diagnostics and label assignment

Correctness assertions have `pass` or `fail`. Missing execution, timeout,
unsupported topology, or a missing required check is `not_run`/`incomplete`,
never pass. Keep this axis separate from statistical investigation warnings.

Candidate production-monitor inputs for Task 5B are ordered certified batches
within each root/k/strategy (all-seat sums initially): attempted win and safety
rates from all-player batches; completed win rate, completed deficit, losing
margin and winning margin from outcome B. Completed loss rate is the complement
of win rate, not an independent extra check. Maintain original batch IDs; never
concatenate roots, k values, or actual seats into a synthetic time series. At
least 30 certified batches is the initial support floor; a positive conditional
denominator in fewer than 30 batches is insufficient. Missing required batches
are integrity failures, not short series to salvage by dropping gaps. Constant
series and undefined lag/drift statistics cannot become no-concern evidence.
The final recipe must also define handling of zero-denominator batches without
silently deleting them or changing the temporal order.

**No new numerical statistical alert threshold is accepted in 5A.** Task 5B must
freeze the lag/drift statistics, selection of monitored strategy/seat cells,
support and degeneracy rules, effect sizes of concern, and multiplicity policy
*before* measuring detection and clean-fixture alerts. Its versioned capability
record is a required input to label assignment. This is an explicit acceptance
gate, not permission to copy the current descriptive band into a pass/fail test.

For each declared statistical check, apply this ordered decision rule:

1. An integrity failure invalidates its source; publish failure evidence outside
   a successful PE/MR completion. Do not classify corrupted data statistically.
2. An explicitly optional omitted check is `omitted_by_design`, with its reason.
   Required omission is incomplete execution and blocks complete coverage.
3. An absent/incompatible capability record, missing applicable threshold rule,
   inadequate support, undefined statistic, or cap/truncation in required
   statistical coverage yields `insufficient_evidence`, with exact reasons.
4. With valid support and a compatible capability record, evaluate its frozen
   alert predicate. True yields `investigate`; false yields
   `no_concern_detected` **only for that stated coverage and sensitivity**.

Until 5B supplies that evidence, step 3 is mandatory: neither of the two
substantive statistical labels may be inferred from a successful software test.
In particular, the two-batch correctness workload cannot support batch-dependence
claims. It must report statistical evidence as insufficient. Record per-check
labels even if another check warns; never hide uncovered checks in an aggregate.

The current `+/-1.96/sqrt(lagged_pairs)` bands stay zero-centered descriptive
references. Crossing one is not automatically a failure or an investigation
alert; staying inside does not establish independence. A future formal alert
must identify the full tested family and its multiplicity treatment as well as
a meaningful effect threshold. A descriptive triage rule must expressly disclaim
calibrated false-positive control and report the number of opportunities to
alert. Do not borrow the unrelated H2H 2% superiority guarantee.

An investigation warning records evidence and qualifies interpretation of
batch-based uncertainty. It does not stop statistical sampling, add games,
change candidates, increase replicates, or redefine batch boundaries. Existing
deterministic correctness failures keep their failure semantics. Two roots can
agree on the same fault. Batch-only summaries cannot detect within-batch
dependence engineered to preserve every retained batch total.

## 5. Evidence envelope and durable publication

Keep Task 0's existing PE/MC/MR locations and `schema_version:1`:

- `cfg.scope_dir('rng_diagnostics', DIAGNOSTICS) / 'preflight_evidence.json'` (PE).
- The same owner / `monitor_coverage.json` (MC) and `production_monitor.parquet` (MR).
- Supporting validation-only files under that owner's `diagnostics/preflight/`
  subtree, with exact planned relative paths and an authenticated inventory.

These are planned producer paths, not APIs or artifacts added by 5A. Add
non-mutating AppConfig getters and narrow integration in 5C. Do not search prior
result directories, invent a new artifact scope/stage, or adopt legacy bytes by
attaching new sidecars. Production PE's root_seed is its owning root; the separate
validation roots and scope are explicit in workload_identity's referenced plan.
Record every source/sidecar identity; a reused proof needs an authenticated
current-run applicability record, not an unchanged foreign completion stamp.

Retain all required Task 0 fields. Identity fields are canonical digest strings
with their full typed payloads available through inventoried plan/result files.
Add the following named fields to the version-1 recipe's typed nested records:

| Record | Required content |
| --- | --- |
| code identity payload | Full commit, clean/development-dirty state, existing dirty fingerprint when applicable, source inventory and test/driver hashes |
| config identity payload | Effective validation config, relevant production/design config, strategy-grid/roster/profile hashes; operational variant settings separately |
| RNG identity payload | Scheme 2, PCG64DXSM, NumPy version, complete purpose map and semantic coordinate layout |
| workload identity payload | Recipe ID/hash, fixture root seeds, every k, roster, batches/shuffles, expected coordinates/counts, unit plan, safety overrides, execution variants and bounds |
| environment v1 | OS/build, architecture, Python version, dependency versions, start method, resolved worker/native thread counts, storage/scratch configuration, process-tree memory and elapsed/resource measurements |
| applicability v1 | Target code/config/environment identities, covered roots-as-domains/k/rules/profiles/topologies, excluded conditions, reuse decision/reasons, source proof manifest and capability record identity |
| each check | Task 0 fields plus kind, execution status, correctness result or statistical result, reason, actual support/statistics, threshold/multiplicity rule identity, timing and result artifact identity |

For an integrity check with no statistical capability exercise, fixture/alert
counts are null, not fabricated zeros. Its `fault_recipe_id` names its deterministic
fault fixture; `not_applicable_correctness_only_v1` is an explicit identifier for
checks without injected faults. `result_artifact` references a retained typed
result including a checksum, not a transient console log. A machine-readable
test report can be retained, but its invocation, selected cases, and code identity
must also be bound. A pytest pass alone is not authenticated PE publication.

PE/MC status reduction preserves Task 0's enum: required execution/integrity
failure produces **no successful completion**; for successfully executed checks,
any investigation gives `investigate`, otherwise required insufficient statistical
coverage gives `insufficient_evidence`, otherwise `complete_valid`. Record
`coverage_complete` separately, so investigate cannot hide insufficient checks.
`omitted_by_design` is for optional components, never the whole required stage.
A structured failure report is retained inside the supporting unit inventory;
do not extend the success enum with a disguised success-for-failure status.
`complete_valid` authenticates execution/evidence, not universal independence.

Plan and unit identity must include recipe, code/config/RNG identities, test
inventory, and applicable capability version. Use existing partition/unit
helpers and `atomic_path`; authenticate each durable check/root/k/variant before
marking it complete. A process exit between bytes and sidecar/stamp cannot create
success. On resume validate units individually, reuse compatible units, and
recompute absent/corrupt units. Final completion binds PE/MC/MR and every retained
proof unit they reference. Unexpected missing required outputs remain incomplete.

## 6. Reuse and applicability

Reuse is exact and explicit, not a blanket lifetime certificate:

1. Validate proof bytes, sidecars, immutable plan, and completion using the
   existing authenticated helpers; no file-exists-only or timestamp-only reuse.
2. Require the same recipe and capability method, code identity (including dirty
   fingerprint), RNG registry/engine/version, relevant config/grid/game rules,
   and software environment. Unrelated-looking code changes still obey the
   repository's existing identity rules; this task grants no freshness bypass.
3. Require target k/profile/topology coverage. A new k or backend, changed
   worker-count pattern, native threading, storage/publication environment,
   compiler/numeric dependency, or changed safety/rule semantics requires rerun
   of the affected proof. Do not extrapolate the two-worker fixture to a tested
   claim about a 16-worker execution. Pure coordinate identities can remain a
   narrower reusable subcheck when the enclosing policy permits it.
4. Different production root seeds do not require rerunning the fixed seeded
   fixture by themselves: the fixture checks coordinate assignment on declared
   validation domains, not every future stream realization. Record this limited
   applicability. Actual production data integrity, batch support, monitoring,
   and independent-root comparisons run on every new dataset and cannot be
   replaced by old preflight results.
5. A new production artifact inventory or monitor scope invalidates its current
   MC/MR even if the software preflight is reusable. Statistical warnings remain
   visible on reuse; unsuccessful/incomplete evidence cannot become successful.

No validated ranking, production capacity, universal independence, exhaustive
lag replacement, all-seat conditional precision guarantee, or production-release
authorization follows from this bounded workload. Use wording such as
“declared correctness checks passed; statistical coverage insufficient” or
“no concerning dependence detected by [named check] within [stated coverage].”
Never summarize the result as “RNG passed.”

## 7. Handoff gates

Task 5A is complete when the recipe arithmetic, source references, status/reuse
rules, and Task 0 envelope mapping are checked and recorded. The validation
record for this design handoff is [task5a_validation.json](task5a_validation.json).
The recipe arithmetic, deterministic roster selection, all 27 existing test
references, relative links, current mega k/worker settings, and whitespace were
checked. **32 existing foundation tests passed** in 30.40 seconds, covering RNG
ownership/replay, scoring examples, conservation, missing-support faults,
insufficient half-drift support, and diagnostic method/config identity. This
does not execute the proposed all-k preflight matrix or measure statistical
detection capability. No runtime/config files were changed for 5A, and existing
uncommitted implementation work was preserved.

Task 5B is next: freeze and evaluate a compact fault-to-check matrix including
serial dependence, between-batch drift, targeted integrity faults, clean
counterparts, and the unchanged-batch-total blind spot. Record seeded fixture
strengths/support, detections and misses, clean alert counts/rates, multiplicity
and effect policy, and common-fault agreement across roots. If a candidate check
fails its declared capability, revise/version it or expose the limitation.
Do not tune thresholds on the clean assessment set and then call it calibration.

Task 5C then implements this bounded preflight, accepts the demonstrated monitor,
and separates required PE/MC/MR from explicitly optional exhaustive lags across
config, stage registry, reporting, completion, and final audit. It must test
deliberate omission versus unexpected missing coverage, interruption/reuse, cap
exhaustion, and warning propagation. Until those gates pass, existing production
diagnostic behavior remains in place and no replacement coverage is claimed.
