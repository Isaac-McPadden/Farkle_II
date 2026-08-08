# Farkle II post-Task-16 release readiness review

## 1. Disposition and executive finding

**Disposition: suitable only after specified fixes.**

The new roots 38/39 fast tree is authentic to clean commit
`e1379160edd015b3648004d80436a73e477f17b6`. All three exhaustive release
audits pass, the independent canonical recount passes every tournament and H2H
integer/nullability/support identity, the repaired pair audit completes inside
its budget, the plain functional/static gates are green, and the reports
preserve conservative claim language. Task 16 findings R1-R3 are fixed and
independently demonstrated.

Production authorization is nevertheless withheld. Both authenticated RNG
diagnostic sidecars report that the configured 150,000 matchup-group capacity
was exhausted. Root 38 skipped 534,618 matchup groups and root 39 skipped
534,533; the official-run requirement is zero. The top-level
`complete_success` state therefore overstates readiness for this required
diagnostic. This does not invalidate the independently verified tournament or
H2H estimands, but it is a release-gate regression of M5 and must be corrected
and demonstrated in a fresh clean fast tree.

A second independent blocker is the contract's documented 90% coverage gate.
The coverage-enforced full suite measured 85.50%. Coverage instrumentation also
pushed the otherwise-passing raw simulation-to-report oracle to 276.282 seconds
against its 240-second non-instrumented budget. The plain 1,067-test suite and
separate toy oracle pass, so this is a non-hermetic release/coverage-gate
failure rather than evidence that the normal oracle or production run is slow.
R4 is consequently regressed elsewhere even though Black and terminology are
repaired.

A third, low-severity release-documentation defect was found: the documented
no-argument `scripts/check_structure_release.py` command now fails because the
current audit correctly requires an explicit fresh artifact root. The three
required explicit-root invocations all pass.

## 2. Identity, scope, and source-gate basis

| Item | Independently verified value |
| --- | --- |
| Reviewed commit | `e1379160edd015b3648004d80436a73e477f17b6` |
| Current/public repository identity | `HEAD == origin/main == e1379160...`; worktree clean before review |
| Checked-in fast-config canonical SHA-256 | `8d0cfa0281c303216597dc5eb03a96b69a2f5a5672b70e25fe356575f0f335f2` |
| Pair public-config SHA-256 | `98b95aae371515c4d171f537e539e9f1005405a955ebdb7ae3b15da2b0303add` |
| Root-38 public-config SHA-256 | `7f01093e61fe3cee0d3cb01909971dafcdeaea81356ba69179fdf55f1337cf4d` |
| Root-39 public-config SHA-256 | `3af2528cc897661d9ecaefc3a7e8041fb76a1cb8d837613c61582770cecd9d4a` |
| Pair run-context identity | `56b5bfe3be69a9bfd01e7c2ce9138be733341cc2f435f6c8c4c2c2a39015d413` |
| Root-38 run-context identity | `8a118a3c47172ff229544a865ad52173567ebcffd8a68680fc6c0a8da3b39032` |
| Root-39 run-context identity | `584665f115e6c2381ae8d7077be08b11498b5744b97a7940dbeafb273ea33ccf` |
| Parent lifecycle roots | `3f23e238...` and `e25845fc...`, exactly bound by the pair context |
| Roots / k | roots 38 and 39; k = 2, 4, 5 |
| Frozen finalist family | 77 strategies; 2,926 unordered pairs |

The exact reviewed paths are:

- pair root: `data/results_post_task16_repaired_fast_20260803_seed_pair_38_39`;
- root 38: `.../results_post_task16_repaired_fast_20260803_seed_38`;
- root 39: `.../results_post_task16_repaired_fast_20260803_seed_39`; and
- pair analysis: `.../seed_pair_analysis`.

All three `run_context.json` files authenticate, reproduce their adjacent
materialized public-config hashes, and record `release_clean`, no dirty
fingerprint, and the exact reviewed commit. The checked-in config independently
reproduces the prospective Task 7 canonical hash. The tree therefore belongs
to the clean public identity asserted as cleared by Task 8.

Task 8's full-suite/static-gate transcript is not present in the repository or
the Task 16 inventory. Under the review instruction's explicit exception for
unavailable evidence, those unchanged gates were rerun. Section 8 records the
fresh results. No production-scale workload was rerun.

The older roots 36/37 tree was read only for timestamps and file counts in
Section 7. It is not used as scientific, lifecycle, or release evidence for
commit `e1379160...`. The known-bad pre-remediation tree was not inspected.

## 3. Complete release audits and artifact graph

| Explicit artifact root | Result | Wall time |
| --- | ---: | ---: |
| root 38 `analysis` | pass, zero failures | 16.077 s |
| root 39 `analysis` | pass, zero failures | 16.119 s |
| pair `seed_pair_analysis` | pass, zero failures | 134.683 s |

The pair audit is 465.317 seconds (77.6%) inside the accepted 600-second
budget. It is also 12.380 seconds faster than Task 5's complete 147.063-second
roots-36/37 benchmark. The original pre-index pair audit exceeded 900 seconds,
so R2's multiplicative lookup behavior is no longer present.

The audits exhaustively validated canonical physical scope, Arrow or typed
non-Parquet schema, content and sidecar hashes, source/manifest identities,
method contracts, stage-config hashes, code identity, named method versions,
and completion outputs. A separate schema-aware inventory found:

- 37,754 artifact-contract-v3 sidecars: 37,748 ordinary artifact sidecars and
  six immutable row-manifest sidecars;
- 75,566 explicit source-artifact identities;
- 69 `complete_valid` v3 stage completions binding 37,824 outputs;
- six sealed TrueSkill root/k cell stamps, six internal TrueSkill shard stamps,
  and three authenticated active-config round-trip completions; and
- zero wrong versions, missing adjacent artifacts, dirty/unknown code
  identities, incomplete source identities, missing method/stage/config hashes,
  or unrecognized completion schemas.

Every ordinary sidecar carries the exact release tuple
`(artifact_contract, schema, estimand, conditioning, rng, outcome) =
(3, 2, 2, 2, 2, 2)` and commit `e1379160...`. The six immutable manifest
sidecars carry the same versions through their stage identity and valid
manifest-contract-v1 roots. The first generic inventory attempt incorrectly
treated those special manifest sidecars and TrueSkill cell/shard stamps as
ordinary stage artifacts; the corrected schema-aware inventory above passed.
This was a reviewer-script classification error, not an artifact failure.

## 4. Independent canonical recount

The recount streamed selected columns directly from curated Parquet and used
local Python integer counters and PyArrow only. It did not import production
aggregation, performance, TrueSkill, H2H, or release-audit helpers. Integer
comparisons used zero tolerance.

### Tournament outcomes and exposures

| Root | k | Attempted A | Completed C | Safety S | Exposures `k*A` | Wins | Losses |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 38 | 2 | 172,000 | 170,461 | 1,539 | 344,000 | 170,461 | 173,539 |
| 38 | 4 | 86,000 | 85,997 | 3 | 344,000 | 85,997 | 258,003 |
| 38 | 5 | 68,800 | 68,800 | 0 | 344,000 | 68,800 | 275,200 |
| 39 | 2 | 172,000 | 170,391 | 1,609 | 344,000 | 170,391 | 173,609 |
| 39 | 4 | 86,000 | 85,996 | 4 | 344,000 | 85,996 | 258,004 |
| 39 | 5 | 68,800 | 68,800 | 0 | 344,000 | 68,800 | 275,200 |

Global totals are `A=653,600`, `C=650,445`, `S=3,155`, attempted exposures
`E=2,064,000`, completed exposures `E_C=2,057,676`, safety exposures
`E_S=6,324`, wins `W=650,445`, and losses `L=1,413,555`. Every root/k cell and
the global sum satisfy:

```text
A = C + S
E = k*A
E_C = k*C
E_S = k*S
W = C
L = E - W = (k - 1)*C + k*S
```

All 653,600 semantic game coordinates were unique. Every safety row had null
winner seat, winner strategy, winning score, victory margin, all seat ranks,
the compact rank entries, and all loss margins. Every completed row had all
winner fields populated, exactly one rank 1, and ranks equal to the permutation
`1..k`.

For every root/k/strategy, independently accumulated attempted, completed,
safety, win, and loss exposures exactly matched `performance.parquet`.
TrueSkill strategy support exactly matched the same attempted/completed/safety
counts. Each root/k rating cell recorded attempted `A`, completed `C`, excluded
safety `S`, and performed updates `C`. Thus updates condition only on completed
games while excluded support and prior-only status remain explicit.

### H2H support, replacements, and multiplicity

| Quantity | Independent recount |
| --- | ---: |
| Candidate table rows / initial family / final family | 80 / 77 / 77 |
| Unordered final-family pairs | 2,926 |
| Root/order cells | 11,704 |
| Attempts | 23,182,656 |
| Completed | 23,024,736 |
| Safety-limit attempts | 157,920 |
| Replacement attempts | 78,960 |
| Formal-test pairs | 2,916 |
| Unsupported/no-test pairs | 10 |
| Multiplicity-family members | 2,926 |

Every executed row exactly matched the immutable block manifest on family and
schedule hashes, pair, strategies, root, order, seats, completed target,
attempt cap, versions, and block ID. Every cell satisfied:

```text
attempted = completed + safety_limit
wins_seat1 + wins_seat2 = completed
wins_a + wins_b = completed
max_attempts = ceil(2.0 * n_completed_required)
replacement_attempt_count = max(0, attempted - n_completed_required)
completed = n_completed_required OR attempted = max_attempts
authenticated attempt range = [0, attempted)
```

No cell exceeded its cap or attempted work after reaching its target. Initial
and final families are identical, all 2,926 pair rows retain multiplicity
membership, and one family hash and one schedule hash cover the full Cartesian
root/order support.

For the ten unsupported pairs, all effect estimates, score statistics,
intervals, p-values, adjusted p-values, and balanced formal aliases are null;
Holm rejection and equivalence are false. No dominance or equivalence edge is
created from these rows. The wider claim guard also withholds candidate-level
claims on all incident pairs involving a nonviable finalist.

## 5. RNG diagnostic release failure

Both RNG sidecars correctly authenticate the configured effective cap and
normalized lag set, identify method version 3, use the semantic tournament
player coordinate, label their reference bands descriptive, and disclaim an
independence claim. Their capacity metadata is:

| Root | Effective cap | Normalized lags | Tracked groups | Skipped groups | Skipped rows |
| ---: | ---: | --- | ---: | ---: | ---: |
| 38 | 150,000 | `[1]` | 150,000 | 534,618 | 537,992 |
| 39 | 150,000 | `[1]` | 150,000 | 534,533 | 538,192 |

The authenticated configuration and method metadata agree; the failure is not
stale metadata. The cap admits only 150,000 of 684,618 encountered unique
matchup/strategy/k groups for root 38 and 150,000 of 684,533 for root 39.
Therefore the required skipped-group count of zero is not met. The pipeline log
also records `rng-diagnostics matchup grouping capped`, but the finding rests
on authenticated artifact metadata rather than the log.

The source currently permits a capped diagnostic to publish `complete_valid`
and the top-level health to publish `complete_success`. Release eligibility
must instead fail when an official run reports any skipped matchup group. A
fix must either provide bounded complete grouping (for example, partitioned
aggregation) or authenticate a resource-tested capacity large enough for all
groups; merely hiding the count or weakening the official zero-skip rule is not
acceptable.

## 6. Final-report claim review

The final report is conservative about the evidence it does contain:

- it explicitly states that the analysis is conditional on the finite
  simulated strategy grid;
- it reports 3,155 tournament safety attempts and all H2H attempts,
  completions, safety attempts, and replacements;
- it retains five operationally nonviable finalists (`0`, `1`, `20`, `40`,
  `41`) and reports 900 unresolved comparisons rather than converting
  noncompletion or nonsignificance to equivalence;
- it reports zero equivalent pairs, while `delta_equivalence=null` keeps the
  formal equivalence procedure disabled;
- it labels tournament and TrueSkill screening as descriptive, and the
  candidate-family source identity contains the exact completed-game
  TrueSkill conditioning and safety-exclusion disclaimer; and
- `dominance_summary.json` has `unique_best=null`,
  `unique_best_claim_permitted=false`, and explicitly says display order does
  not add inferential edges.

No unsupported unique-best, equivalence-from-nonsignificance,
equivalence-from-noncompletion, or dominance-from-unsupported-support claim was
found. The report does not expose the RNG skip counts, however, so its apparent
overall completeness must not override Section 5.

## 7. Performance, ownership, and file-count comparison

The comparison below uses roots 36/37 only as operational regression data.
Direct subtraction of its recorded timestamps shows that the prior review's
“12 hours 53 minutes” total was a six-hour arithmetic error; the actual elapsed
time was 6:53:25.

| Segment | Roots 36/37 | Roots 38/39 | Change |
| --- | ---: | ---: | ---: |
| Total pipeline | 6:53:25 | 6:05:48 | 47:37 faster (11.5%) |
| Root simulation plus publication | about 26 min/root | 26:18 and 24:08 | comparable/slightly faster |
| Root analysis | about 49-55 min/root | 49:47 and 54:46 | materially unchanged |
| Exact H2H planning | 0:13 | 0:13 | unchanged |
| H2H execution | 2:38:42 | 2:28:55 | 9:47 faster (6.2%) |
| H2H inference | 5:44 | 4:16 | 1:28 faster (25.6%) |
| Reporting | 6:20 | 4:54 | 1:26 faster (22.6%) |
| Final-validation tail | 1:07:03 | 0:32:08 | 34:55 faster (52.1%) |
| Complete pair release audit | Task 5: 147.063 s | 134.683 s | 12.380 s faster |

The final-validation tail materially improved, and the pair audit remains
comfortably bounded while preserving full hash/schema/source validation.

The new run context also demonstrates corrected worker ownership. Simulation
owns 12 process workers with one native thread each; ingest owns three
processes with five Arrow/Python/native threads each; analysis owns the
configured four processes with four threads each; and H2H resolves `n_jobs=0`
to 16 one-thread process workers. The prior roots 36/37 context incorrectly
gave analysis 12 simulation-owned workers. The new manifest and all three run
contexts consistently record the corrected policy.

Both trees contain exactly 75,609 files: 26,059 in each root and 23,485 in the
pair tree, including 37,754 sidecars, 37,679 Parquet files, and 84 files named
`*.done.json`. The projected high-cardinality upper envelope was 76,880 files:
25,800 tournament row shards plus a candidate-80 envelope of 12,640 H2H blocks,
each with a sidecar. The observed high-cardinality count was 75,008 because the
frozen family had 77 candidates and 11,704 blocks; the full tree adds 601
bounded workflow/config/report files for 75,609 total. Projection therefore
exceeded its intended observed scope by 1,872 files and did not understate
capacity.

## 8. Complete source and release-gate results

| Gate | Result |
| --- | --- |
| Complete pytest suite | 1,067 passed in 821.04 s; 828.029 s wall |
| Coverage-enforced complete suite | failed: 1,066 passed, one timing failure; 85.50% versus 90%; 1,483.068 s wall |
| Tiny two-root structure oracle | 2 passed in 97.32 s; 99.995 s wall |
| Compiled-Numba subprocess | passed as part of the complete suite |
| Terminology | passed; 0.430 s wall |
| Ruff | passed; 0.206 s wall |
| Black | passed; 181 files unchanged; 0.971 s wall |
| Mypy | passed; no issues in 75 source files; 2.018 s wall |
| Pyright | passed; 0 errors/warnings/information; 17.790 s wall |
| Root/pair structure release audit | all three explicit roots passed; timings in Section 3 |
| Documented no-root structure invocation | failed as expected under current explicit-fresh-root policy; documentation is stale |

The plain full suite includes the migrated ingest, lifecycle, HGB, TrueSkill,
simulation-helper, strict-ID, resume, exact-power, H2H recovery, report-claim,
and compiled subprocess cases. The coverage run's sole test failure is the
same raw end-to-end oracle that passes without instrumentation; its 276.282 s
instrumented runtime exceeded a 240 s operational assertion. Independently of
that timing failure, 85.50% is below the governing 90% threshold. The release
therefore has two substantive failures: coverage-gate hermeticity/threshold and
the artifact-specific RNG zero-skip condition.

## 9. Task 16 R1-R4 classifications

| Finding | Classification | Independent post-run evidence |
| --- | --- | --- |
| R1: 84 hermetic-suite failures | **fixed and independently demonstrated** | Complete suite: 1,067 passed; authenticated-v3 negative/mutation fixtures now reach their consumers. |
| R2: quadratic pair release audit | **fixed and independently demonstrated** | Complete pair audit passed in 134.683 s, inside 600 s and faster than Task 5's benchmark. |
| R3: strict IDs regressed public helpers | **fixed and independently demonstrated** | Complete helper/parallel/ID-boundary suite passed; canonical persisted boundaries remain strict. |
| R4: release hygiene gates | **regressed elsewhere** | Black, terminology, and Ruff pass, but the governing coverage gate reaches only 85.50% and makes a 240 s speed assertion fail under instrumentation. The stale no-root audit command is an additional documentation defect. |

## 10. Original-finding regression classifications

| Finding | Regression classification | Evidence |
| --- | --- | --- |
| B1 safety-limit winner fabrication | **fixed and independently demonstrated** | Direct row recount proves null safety outcomes, completed rank permutations, and every count identity. |
| B2 unauthenticated lifecycle/provenance | **fixed and independently demonstrated** | Three exhaustive audits, 37,754-sidecar scan, authenticated contexts, and full mutation suite pass. |
| H1 scalar-root RNG collisions | **fixed and independently demonstrated** | All tournament semantic coordinates are unique; RNG-v2/source gates pass. |
| H2 metric-only checkpoint ordering | **fixed and independently demonstrated** | Complete interruption/resume suite passes with unchanged v3 ownership semantics. |
| H3 game-stat winner duplication | **fixed and independently demonstrated** | Direct exposure totals are exactly `k*A`, with no extra winner observation. |
| H4 scope/schema/method/source authentication | **fixed and independently demonstrated** | All explicit-root audits and schema-aware identity inventory pass. |
| H5 permissive CLI parsing | **fixed and independently demonstrated** | Complete unit/installed CLI suite passes. |
| H6 no actual end-to-end oracle | **fixed and independently demonstrated** | The separate tiny two-root structure oracle passes both cases. |
| M1 nonmonotone exact-power search | **fixed and independently demonstrated** | Full exact-power suite passes; immutable plan targets and caps recount exactly. |
| M2 root/order imbalance accepted | **fixed and independently demonstrated** | All 11,704 cells exactly match both roots and both orders in the frozen manifest. |
| M3 mutable plan and block recovery | **fixed and independently demonstrated** | Immutable family/schedule hashes, contiguous attempt ranges, and recovery tests pass. |
| M4 unadjusted root significance labels | **fixed and independently demonstrated** | Final reports preserve descriptive root language and no unadjusted root rejection claim. |
| M5 incorrectly ordered/incompletely authenticated RNG diagnostic | **regressed elsewhere** | Ordering/method metadata remains correct, but both official roots authenticate large nonzero skipped-group counts while health says complete. |
| M6 TrueSkill calibration claim mismatch | **intentionally accepted limitation** | `mu_softmax_heuristic` remains explicitly descriptive and non-calibrated; exact completed-only conditioning is retained. |
| P1 weak ingest semantic validation | **fixed and independently demonstrated** | Migrated negative matrix passes in the 1,067-test suite; clean ingest artifacts audit and recount exactly. |
| P2 sparse/zero-exposure semantics | **fixed and independently demonstrated** | Complete sparse-support tests pass; no regression appears in complete root/k support. |
| P3 compiled behavior and ID boundaries | **fixed and independently demonstrated** | Compiled subprocess and strict scalar/pandas/Arrow/helper-ID tests all pass. |

## 11. Required correction and release sequence

1. Make the official RNG diagnostic complete under bounded resources. A
   capacity-only change must cover at least the observed 684,618 groups and be
   demonstrated not to violate RAM limits; a partitioned streaming design is
   preferable if that bound is unsafe.
2. Make official release health fail closed whenever
   `rng_skipped_matchup_group_count != 0`, while retaining the authenticated
   tracked/skipped metadata.
3. Update the documented structure-release invocation to require the exact
   fresh root and pair paths.
4. Restore a hermetic coverage gate at the accepted 90% threshold. Add relevant
   branch evidence and separate the raw-oracle correctness gate from its
   non-instrumented wall-time gate so coverage overhead cannot create a false
   performance regression.
5. Run the focused RNG capacity/ordering/health tests, the non-instrumented raw
   oracle, and all complete source/coverage/static gates from a clean commit.
6. Generate a new isolated two-root fast tree from empty paths under the new
   commit/config identity and repeat the complete explicit-root audits,
   independent recount, RNG metadata check, and final report review.

## 12. Handoff

- **Files changed:** only
  `docs/reviews/Farkle_II_post_Task16_release_readiness_review.md`.
- **Reviewed commit/config/output roots:** clean public commit
  `e1379160edd015b3648004d80436a73e477f17b6`; checked-in fast-config hash
  `8d0cfa...`; materialized pair/root hashes `98b95aae...`, `7f01093e...`, and
  `3af2528c...`; pair roots 38/39 and their exact paths from Section 2.
- **Audit and recount results:** root audits passed in 16.077/16.119 s and pair
  audit passed in 134.683 s; all tournament, TrueSkill, H2H, replacement,
  attempt-cap, family, multiplicity, null-result, and report-claim checks pass;
  RNG skips fail at 534,618 and 534,533 groups.
- **Complete gate results:** the plain 1,067-test suite and two toy-oracle tests
  pass; terminology, Ruff, Black, Mypy, Pyright, and all explicit-root audits
  pass; the coverage gate fails at 85.50% with one instrumentation-sensitive
  timing failure, and the documented no-root audit invocation is stale.
- **Performance comparison:** total time improved from 6:53:25 to 6:05:48,
  final tail from 1:07:03 to 0:32:08, and pair audit from 147.063 to 134.683 s;
  worker ownership is corrected and observed high-cardinality files remain
  below the declared envelope.
- **Artifact-staleness implications:** roots 38/39 remain internally authentic
  to `e1379160...` and must not be rewritten or re-sidecarred, but they are not
  qualifying release evidence because their authenticated RNG diagnostic is
  incomplete. Any cap/config, health-gate, or implementation correction changes
  freshness identity and requires a new output prefix; roots 36/37 remain
  comparison-only evidence for their older commit.
- **Exact next action:** implement bounded zero-skip RNG diagnostic processing
  plus a fail-closed official health gate, restore a hermetic 90% coverage gate
  with a separate non-instrumented oracle timing check, update the explicit-root
  gate documentation, commit all green gates, and run a new isolated fast roots
  pair for the same independent release audit before authorizing any full
  production run.
