# Post-remediation review 08: tests and code quality

## Scope and verdict

This review assessed whether the current tests and general validation structure
would expose statistically or operationally wrong behavior. It inspected the
current checkout and the completed `configs/fast_config.yaml` run at
`data/results_seed_pair_32_33`. It did not modify code, configuration, tests, or
existing artifacts.

**Verdict: questionable within this review's scope.** The most sensitive H2H
arithmetic, power, dominance, exact-turn, seat, and root-combination contracts
have useful hand-checkable tests. However, completed TrueSkill work can be
silently reused after its source or model contract changes, and neither the
test suite nor `--force` exposes that defect. The suite also lacks a true
raw-input-to-report workflow oracle and discriminating tests for several
important statistical procedures. A successful run and many passing tests are
therefore not yet strong evidence that stale or statistically altered outputs
would be caught.

Finding classification in this report is explicit: findings 1, 3, and part of
6 are confirmed implementation or gate defects; findings 2, 4, and 5 are test
gaps against currently plausible or currently correct implementations; finding
7 is a hardening gap.

## Validation performed

- `python -m pytest -q`: 578 tests passed and one terminology test failed. The
  failure was caused by pre-existing untracked review Markdown under
  `docs/reviews`, not by application behavior or this report.
- `python -m pytest --ignore=tests/unit/test_terminology.py --cov=src/farkle
  --cov-report=term-missing -q`: all selected tests passed, but measured branch
  coverage was 86.09%, below the configured 90% threshold.
- Ruff, Black check, Mypy over `src`, and Pyright all passed.
- The structural release audit passed both with its default invocation and with
  the three completed fast-run analysis roots supplied explicitly.
- A clean interpreter confirmed that representative production Numba functions
  are `CPUDispatcher` objects and compile on this machine. Pytest itself does
  not exercise that mode.
- An isolated temporary-directory probe reproduced finding 1: with an existing
  rating parquet and version-1 block stamp, `_rate_block_worker(...,
  resume=False, env_kwargs={"beta": 999, "tau": 999, ...})` returned the
  stamped `("2", 123)` and left the stale file unchanged.
- The completed fast run reports `complete_success` for both roots and the pair
  workflow (`data/results_seed_pair_32_33/pipeline_health.json:1-24`). Its H2H
  execution state is `complete_valid` with 11,400/11,400 blocks, and its 2,850
  inference rows satisfy `balanced_a_win_rate_alias = 0.5 + d_ab` to a maximum
  observed absolute error of approximately `1.11e-16`. These are useful wiring
  checks, not a statistical validation of the run.

## Findings

### 1. High — confirmed defect: completed TrueSkill and HGB work can bypass changed contracts and `--force`

**Confidence: High.**

**Evidence.** The root plan accepts `force`, passes it to game statistics, RNG
diagnostics, and screening, but calls TrueSkill as `trueskill.run` without the
flag (`src/farkle/analysis/__init__.py:71-75,97-116`). The block stamp records
only a shard key, parquet path, row count, creation time, and version
(`src/farkle/analysis/run_trueskill.py:315-321`). Loading it is described as
minimal validation and does not bind the source artifact, source digest,
TrueSkill parameters, code revision, root, or player count beyond unvalidated
strings (`src/farkle/analysis/run_trueskill.py:332-352`).

More importantly, `_rate_block_worker` returns immediately whenever that stamp
names an existing rating parquet (`src/farkle/analysis/run_trueskill.py:567-581`).
This happens before any use of the `resume` argument and without comparing the
curated input or `env_kwargs`. Afterward, `run_trueskill_root` loads the old
ratings and publishes a fresh current-config sidecar for them
(`src/farkle/analysis/run_trueskill.py:977-1010`). The fast-run stamp illustrates
the weak schema: it contains only those five fields and claims 172,000 rows
(`data/results_seed_pair_32_33/results_seed_32/analysis/06_trueskill/by_k/2p/ratings_2_seed32.done.json:1-7`).

The isolated probe described above proves the short circuit with `resume=False`
and changed `beta`/`tau`. Existing tests exercise continuation from an
in-progress checkpoint and cleanup (`tests/unit/analysis/test_run_trueskill_streaming.py:125-179`),
but not completed-cell staleness. The coverage run likewise left the completed
stamp branch at source lines 569-581 unexecuted.

HGB has the same contract-level weakness through a different guard. The root
plan calls `hgb_feat.run` without `force` (`src/farkle/analysis/__init__.py:111-116`),
and that function considers outputs fresh from file mtimes, sidecar presence,
and generic sidecar validity (`src/farkle/analysis/hgb_feat.py:62-91`). It does
not compare the sidecar's config hash with the active config. A change to fold
count, tree depth, estimator count, or permutation repetitions with unchanged
input files therefore skips recomputation, and root-level `force=True` cannot
override it. The existing up-to-date test constructs placeholder outputs and
specifically asserts the model runner is not called
(`tests/unit/analysis/test_hgb_feat.py:52-93`); it never changes an HGB contract
input.

**Consequence.** Changing the curated rows, TrueSkill parameters, rating
implementation, or HGB fitting/validation parameters can produce a nominally
successful run whose model outputs came from the old contract. Re-publishing a
current rating sidecar can make stale data appear provenance-compatible, and
either stale model can alter screening or finalist admission.

**Smallest reasonable remediation.** Put source artifact identity/hash,
root/k, rating parameters, and the applicable freshness/config digest into each
block stamp; validate all of them before skipping. Make `resume=False` and root
workflow `force=True` invalidate completed block stamps and rating outputs. Give
HGB the same stage-completion/freshness contract used by other canonical stages
and pass `force` through the root plan. Add tests that change one TrueSkill and
one HGB contract input at a time and prove recomputation, plus tests that
unchanged inputs still skip.

### 2. High — test gap: there is no raw-input-to-report oracle for the canonical workflow or fast run

**Confidence: High.**

**Evidence.** The testing map calls `test_structure_toy_oracle.py` the “Full
workflow” test (`docs/codex_context/testing_and_review_map.md:10-28`), but that
test starts by manufacturing all-player batch metrics and rating parquets
(`tests/integration/test_structure_toy_oracle.py:80-200`). It replaces H2H game
simulation with a deterministic result fabricator
(`tests/integration/test_structure_toy_oracle.py:37-56,237-242`) and invokes only
root stability, candidate freezing, the H2H tail, agreement, and reporting. It
does not run tournament simulation, ingest, curate, combine, all-player
metrics, performance, seat analysis, root TrueSkill, or HGB as one connected
workflow.

The separate tournament integration test heavily replaces the grid, worker
initializer, constants, and games per shuffle
(`tests/integration/test_run_tournament_integration.py:144-176`). Its final
assertions require only a checkpoint, a nonempty win counter, and keys that are
a subset of the test grid (`tests/integration/test_run_tournament_integration.py:202-211`).
No test references `fast_config.yaml` as an executed workflow or reads
`data/results_seed_pair_32_33`. The release script audits checked-in configs by
default but receives artifact roots only through optional CLI arguments
(`scripts/check_structure_release.py:11-16,21-38`); the documented default gate
does not supply them (`docs/codex_context/testing_and_review_map.md:46-56`).

**Consequence.** Schema-compatible but semantically wrong handoffs between
simulation, root analysis, pair analysis, and reporting can pass every test.
The completed fast run proves those stages executed, but CI does not recheck
its cross-stage identities, support, sidecars, completion states, or selected
observed values.

**Smallest reasonable remediation.** Add a deliberately tiny, fully configured
two-root CLI workflow starting from actual simulated rows and ending at the
report. Use a very small frozen family and schedule, but do not replace stage
implementations. Assert hand-calculated row/turn totals, equal-k estimates,
source hashes, lifecycle states, H2H orientation, and report claims. Separately
add a read-only fast-run audit command that accepts the designated completed
artifact root and checks invariant identities without asserting strategy
effectiveness.

### 3. Medium — confirmed statistical-contract defect: zero-exposure rows are rejected instead of excluded

**Confidence: High.**

**Evidence.** The governing contract says zero-game frequentist rows are
excluded from estimates. Canonical performance instead raises if *any* input
row has exposure less than or equal to zero
(`src/farkle/analysis/performance.py:60-88`), and two-root stability applies the
same rule (`src/farkle/analysis/root_stability.py:105-123`). The estimator then
uses every accepted row in batch-rate variation
(`src/farkle/analysis/performance.py:100-117`).

The performance fixture supplies exposure 100 for every row and only asserts
that output exposure is positive (`tests/unit/analysis/test_performance.py:32-42,81-88,101-109`).
The root fixture similarly defaults every row to exposure 10
(`tests/unit/analysis/test_root_stability.py:45-62`). There is no zero-exposure
oracle, while negative and impossible counts are properly treated as invalid
production input.

**Consequence.** A legitimate zero-game cell in a sparse deterministic batch
can abort an otherwise analyzable run. Alternatively, upstream code may omit
such cells, leaving the effective batch support implicit and untested in joint
resampling. Both behaviors disagree with the stated exclusion rule.

**Smallest reasonable remediation.** Reject negative exposure and impossible
wins, but explicitly remove zero-exposure rows before frequentist estimation
and record their count as provenance. Define how absent strategy/batch cells
enter joint-vector resampling. Add fixtures showing that adding a zero-game row
does not change the estimate or MCSE and that negative exposure still fails.

### 4. Medium — test gap: several named statistical tests do not discriminate the required method from a wrong substitute

**Confidence: High for the gaps; Medium-to-High that current implementations are correct.**

**Evidence.** Important positive examples exist: exact returns are checked by
hand (`tests/unit/analysis/test_all_player_metrics.py:89-133`); equal-k means and
analytic MCSE propagation are numerical (`tests/unit/analysis/test_performance.py:91-121`);
seat effects are checked within k before standardization
(`tests/unit/analysis/test_seat_analysis.py:87-131`); H2H has constrained-null
arithmetic, an external score-test oracle, boundaries, equivalence, and
unbalanced-order rejection (`tests/unit/analysis/test_h2h_inference.py:171-178,207-253,287-320,395-443`);
and power is compared with a brute-force enumeration and one-smaller allocation
(`tests/unit/analysis/test_h2h_schedule.py:110-132,148-171`).

The following advertised contracts remain weakly distinguished:

- The test named for joint batch resampling checks only repeatability after a
  forced rerun (`tests/unit/analysis/test_performance.py:183-196`). It would
  also pass if each strategy were resampled independently. Current production
  does appear to use one selected batch vector for all strategies within k
  (`src/farkle/analysis/performance.py:312-329`), but there is no oracle whose
  covariance/rank result changes under independent resampling.
- Wilson output is merely asserted positive while batch MCSE is checked
  separately; no test checks the t critical value, degrees of freedom, or
  interval endpoints (`tests/unit/analysis/test_performance.py:101-116`). A
  normal critical value or Wilson interval substituted as Monte Carlo
  uncertainty would not be directly rejected.
- Root stability numerically checks raw-count combination and the equal-k mean,
  but most discrepancy, joint-bootstrap, convergence, and half-drift assertions
  are row counts, column presence, bounds, or determinism
  (`tests/unit/analysis/test_root_stability.py:130-192`).
- TrueSkill contribution verifies complete support, the winning candidate, and
  absence of `sigma`, but not a hand-calculated percentile mean
  (`tests/unit/analysis/test_trueskill_screening.py:44-79`). Its fixture does not
  use cell-dependent rating scales that would make raw-mu averaging disagree
  with percentile aggregation.
- HGB's integration test validates sidecars and one prediction per strategy,
  but not disjoint train/held-out strategy identifiers, fold assignment, fold
  metrics, or retained between-fold importance variation
  (`tests/unit/analysis/test_hgb_feat.py:160-225`). Current code visibly assigns
  whole strategy rows to folds and retains fold standard deviation
  (`src/farkle/analysis/run_hgb.py:277-305,325-367`), but those properties are
  not protected by tests.
- No targeted oracle covers maximin ties, common-support player-count rank
  correlations, or Holm tied-p-value ordering. The Holm implementation uses a
  stable mergesort (`src/farkle/analysis/h2h_inference.py:274-289`), but that
  behavior is untested.

**Consequence.** A refactor can keep deterministic output shapes and plausible
values while changing the replication unit, interval family, dependence
structure, or HGB validation design. The present tests could then confirm the
wrong implementation reproduces itself.

**Smallest reasonable remediation.** Add compact fixtures designed so each
wrong substitute produces a different known result: perfectly correlated batch
vectors for the bootstrap, three batches for an exact t interval, unequal root
counts for diagnostic identities, cell-specific affine TrueSkill scales, and
an HGB spy estimator that records disjoint strategy IDs and predetermined fold
importance values. Add direct tie/order tests for maximin and Holm.

### 5. Medium — test gap: pytest globally disables production Numba behavior

**Confidence: High.**

**Evidence.** Even when Numba is installed, `tests/conftest.py` replaces
`numba.jit` and `numba.njit` with identity decorators during import and again
in `pytest_configure` (`tests/conftest.py:74-113,268-284`). A dedicated test
asserts that this patch occurs (`tests/unit/test_conftest_numba.py:12-30`). The
production scoring and strategy decision kernels use `@nb.njit(cache=True)`
(`src/farkle/game/scoring.py:47-48,152-153`,
`src/farkle/game/scoring_lookup.py:27-28,123-124`, and
`src/farkle/simulation/strategies.py:124-125`).

The isolated clean-interpreter probe compiled representative kernels
successfully, so this is not evidence of a current Numba failure. It is evidence
that the normal suite cannot detect one. The tournament integration test's
heavy worker and workload substitutions, followed by weak checkpoint
assertions, do not supply a clear compiled-kernel oracle.

**Consequence.** Type-inference failures, unsupported operations, cache issues,
or compiled/Python behavioral divergence can enter production while all normal
tests pass. These failures are particularly likely to be platform-sensitive.

**Smallest reasonable remediation.** Keep no-JIT unit runs for coverage speed,
but add a separate clean-subprocess test job that imports without pytest's
patch, verifies dispatcher compilation, and runs a hand-checkable scoring case
plus a tiny seeded simulation. Run it on every supported operating-system class.

### 6. Low — confirmed gate defects: the coverage threshold is inactive and the terminology test is non-hermetic

**Confidence: High.**

**Evidence.** `pyproject.toml` sets `fail_under = 90`
(`pyproject.toml:76-86`), but the documented release commands invoke plain
pytest without coverage (`docs/codex_context/testing_and_review_map.md:46-56`).
The ordinary suite therefore did not enforce the threshold. When coverage was
enabled explicitly, total coverage was 86.09% and failed. The shortfall itself
is not a statistical-quality metric, but key missing branches included the
TrueSkill stale-done guard from finding 1.

The terminology checker recursively scans every file under `docs`, whether or
not Git tracks it (`scripts/check_terminology.py:9-14,26-44`). Consequently the
current full suite failed because earlier untracked review reports quoted
retired vocabulary. `test_repository_terminology_is_precise` merely asserts
that global scan is empty (`tests/unit/test_terminology.py:1-7`). This makes test
results depend on unrelated local documents and means an adversarial review can
break the application gate by accurately naming historical behavior.

**Consequence.** The stated coverage policy gives false assurance, while the
ordinary suite can fail for workspace content unrelated to the checked-out
commit. That obscures real regressions and makes local/CI results disagree.

**Smallest reasonable remediation.** Add `pytest-cov` arguments to the actual
release gate (or remove the unenforced threshold until it is intentionally
restored). Make repository text checks operate on tracked files or an explicit
maintained allowlist, and exclude generated/adversarial review output.

### 7. Low — hardening gap: identifier-null and malformed-row failure paths are under-tested

**Confidence: Medium.**

**Evidence.** Core performance and root analyses repeatedly coerce strategy
identifiers with `astype(int)` or `int(...)` rather than validating a shared
non-null canonical identifier schema (for example,
`src/farkle/analysis/performance.py:121-123,174-196` and
`src/farkle/analysis/root_stability.py:183-186,231-240`). TrueSkill's row parser
silently skips incomplete seats and malformed winner labels
(`src/farkle/analysis/run_trueskill.py:493-516`). HGB logs and skips unparseable
strategy feature rows (`src/farkle/analysis/run_hgb.py:90-110`) and performs an
inner join afterward (`src/farkle/analysis/run_hgb.py:526-544`).

There are useful missing-coordinate and incomplete-support tests, but no
cross-module fixture with nullable Arrow integer IDs, pandas `Int64`, numeric
strings, missing IDs, and nonnumeric IDs proving that joins either preserve an
identical strategy set or fail loudly. The HGB test checks only that the final
prediction IDs match its clean integer inputs.

**Consequence.** A malformed identifier can cause a late conversion error or,
in permissive paths, silently remove a strategy and change support. The latter
is especially dangerous for screening and finalist selection.

**Smallest reasonable remediation.** Centralize canonical strategy-ID
validation at artifact boundaries, require non-null IDs, and make permissive
row dropping return an explicit rejected-row artifact or fail for canonical
inputs. Add Arrow/pandas round-trip and join tests across each consumer.

## Coverage of the requested adversarial checklist

Strong or materially adequate coverage was observed for exact `n_turns`
returns versus the rounds proxy, chance baselines, equal-k point estimates and
analytic MCSE propagation, seat standardization, complete root/k support in
screening and root combination, raw-count root combination, H2H orientation,
constrained-null score arithmetic, score-interval boundaries, asymmetric-order
rejection, explicit equivalence, exact implemented-test power, one-smaller
allocation, cap resume, candidate-tail contraction, cycles, unresolved pairs,
direct dominance for unique best, multi-k external-diagnostic reporting, and
fixed-root language.

Material gaps remain for completed-shard staleness, `--force` propagation,
zero-game exclusion, a discriminating joint-vector bootstrap oracle, t-interval
degrees of freedom, root diagnostic identities, TrueSkill percentile arithmetic,
HGB held-out/fold behavior, maximin and Holm ties, family-hash mismatch at every
H2H boundary, nullable strategy identifiers, production Numba execution, and a
complete workflow beginning with simulated rows. These gaps are assessed by
the defects they permit, not by line count alone.
