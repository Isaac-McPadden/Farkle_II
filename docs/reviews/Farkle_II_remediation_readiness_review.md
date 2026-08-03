# Farkle II remediation readiness review

## 1. Disposition and executive finding

Reviewed HEAD `93c7660a250b6ed21e10bee64146b580a8c261b5` correctly repairs the
central statistical defects in the post-fast-config synthesis. The clean
roots 36/37 run independently conserves attempted, completed, and safety-limit
outcomes; RNG v2 uses full semantic coordinates; H2H excludes noncompletions,
attempts deterministic replacements, retains the frozen family, and emits no
formal result for unsupported pairs. The original tournament, equal-k, and
seat-adjusted H2H estimands were preserved.

The revision is not release-ready. The accepted Task 0 contract requires a
green hermetic suite and release gates, but the fresh review found:

1. 84 failures among 1,017 collected tests. The failures are broad, spanning
   simulation helpers, ingest, combine, game statistics, HGB, TrueSkill,
   configuration/CLI fixtures, and terminology.
2. A public simulation-helper regression: ordinary `ThresholdStrategy`
   objects produce descriptive string identifiers, while
   `validate_simulation_row` now unconditionally requires canonical integers.
   `simulate_many_games` and related helpers therefore fail before returning
   results (`src/farkle/simulation/simulation.py:399-421`,
   `src/farkle/utils/strategy_ids.py:68-73`).
3. Many lifecycle and negative-ingest tests were not migrated to authenticated
   v3 inputs. They fail before exercising their advertised mutation or
   malformed-row oracle. This weakens B2, P1, HGB, and TrueSkill regression
   evidence even though the clean artifacts and lower-level v3 tests are
   coherent.
4. `black --check .` reports 23 maintained files requiring formatting, and
   the terminology gate still scans historical reviews and misidentifies some
   process-executor syntax (`scripts/check_terminology.py:13-39`).
5. The pair release audit is impractically slow. The 41,108,669-byte
   `h2h_execute.done.json` lists 11,706 outputs. For each output,
   `_completion_artifact_path` starts a fresh recursive search
   (`src/farkle/analysis/release_audit.py:120-135`), over a pair tree containing
   23,485 files. A pair-only audit did not finish within 15 minutes; the
   original clean run spent approximately 67 minutes between reporting
   completion and pipeline completion. This traversal needs an indexed,
   canonical location lookup before production-scale release auditing.

These are release-engineering and API/gate failures, not evidence that the
repaired clean-run estimands are statistically wrong. Production authorization
should nevertheless remain withheld because the governing contract makes the
failed gates mandatory.

## 2. Review identity and scope

| Item | Reviewed value |
|---|---|
| Defective review base | `6be5f5fa11df77155621bfc81188c7515f38f8de` |
| Remediation HEAD | `93c7660a250b6ed21e10bee64146b580a8c261b5` |
| Remediation commits after base | 12 |
| Diff size | 129 files; 24,674 insertions; 1,967 deletions |
| Governing contract | `docs/remediation/post_fast_config_remediation_contract.md` |
| Clean fast tree | `data/results_official_fast_20260801_seed_pair_36_37` |
| Recorded clean-run code identity | `release_clean`, commit `93c7660a250b6ed21e10bee64146b580a8c261b5` |
| Roots / k | roots 36 and 37; k = 2, 4, 5 |
| Frozen finalist family | 77 strategies; 2,926 unordered pairs |
| Worktree before review | clean |
| Known-bad reviewed tree | not read, modified, repaired, or deleted |

The review read the original synthesis, Task 0 contract, repository context,
the entire changed-file inventory and commit series, all changed configuration
families, the principal changed source/test families, the clean run contexts,
logs, health state, reports, aggregate Parquet products, and representative v3
sidecars/completion identities. The complete test collection was executed, so
unchanged and changed code paths were also checked for cross-task regressions.

No production-scale pipeline was run. Artifact inspection was read-only. The
only repository file written is this review.

## 3. New release-blocking and operational findings

### R1. The hermetic release suite has 84 failures

- **Severity/confidence:** Blocker; high confidence.
- **Evidence:** `python -m pytest -q --tb=short` completed in 844.5 seconds.
  A follow-up last-failure run counted 84 current failures from 1,017 collected
  tests.
- **Main failure families:** 16 ingest integration cases; public simulation
  helper and process-executor cases; all-player metrics; combine/curate/game
  statistics; HGB freshness; TrueSkill streaming/orchestration/freshness;
  config/CLI fixtures; runner lifecycle helpers; and terminology.
- **Interpretation:** Some failures are stale v2/string-ID fixtures that now
  correctly fail closed. They still invalidate the release gate and leave the
  advertised mutation/negative oracle unexecuted. Other failures are genuine
  compatibility regressions, notably the public simulation helpers.
- **Required fix:** Migrate every fixture to canonical v3 locations, sidecars,
  completion payloads, and integer IDs where the boundary contract requires
  them. Restore a deliberate public-helper ID policy rather than failing
  ordinary helper usage incidentally. Require all 1,017 tests to pass.

### R2. Exhaustive pair release audit has quadratic path-discovery behavior

- **Severity/confidence:** High operational release blocker; high confidence.
- **Evidence:** Both root audits passed in 43.2 and 42.6 seconds. The pair-only
  audit exceeded 15 minutes. The pair tree has 23,485 files; the H2H completion
  inventory alone has 11,706 outputs. `_audit_completion` calls a recursive
  name search separately for every output (`release_audit.py:139-166`).
- **Clean-run timing evidence:** reporting ended at 22:11:39 local and the
  pipeline completed at 23:18:42, an approximately 67-minute tail dominated by
  final validation/context work.
- **Required fix:** Build one canonical `(scope, k, relative path) -> path`
  index per audit root, reject duplicates while constructing it, and resolve
  every completion entry in O(1) lookup time. Preserve full byte/schema/sidecar
  validation; do not replace it with size or mtime checks.

### R3. Strict strategy-ID hardening regressed public simulation helpers

- **Severity/confidence:** High API/regression defect; high confidence.
- **Evidence:** `simulate_many_games`, `simulate_many_games_from_seeds`,
  tournament integration helpers, and their parallel variants fail because a
  normal `ThresholdStrategy.strategy` value is descriptive text while row
  validation requires an integer. The clean production grid is unaffected
  because it assigns integer manifest IDs before execution.
- **Required fix:** Define and test the helper boundary explicitly: either
  assign deterministic local canonical IDs before row construction or require
  a documented caller-provided integer-ID mapping. Do not weaken canonical
  persisted artifact validation.

### R4. Release hygiene gates are not green

- **Severity/confidence:** Medium release blocker; high confidence.
- **Evidence:** Black would reformat 23 files. The terminology gate flags
  historical review evidence, contract prose, metadata, and a legitimate
  multiprocessing API reference, so it is not hermetic to production
  terminology.
- **Required fix:** Format the maintained tree and scope/allowlist terminology
  checking so historical evidence and legitimate external API names do not
  fail the gate, while production artifact/estimand terminology remains
  enforced.

## 4. Original finding-by-finding classification

The required classifications below describe the finding itself. A finding can
be repaired while the release remains blocked by R1-R4.

| Finding | Classification | Fresh adversarial evidence |
|---|---|---|
| B1 safety-limit winner fabrication | **fixed and independently demonstrated** | Forced zero/nonzero safety games have null winner/ranks; mixed hand oracle passed; independent fast recount found `W=C` and null winner-conditioned fields in every safety row. |
| B2 unauthenticated lifecycle/provenance | **fixed but weakly tested** | v3 primitive/mutation tests and both root audits passed; all 37,754 sidecars scanned contained no unknown/incompatible identity. HGB/TrueSkill high-level mutation fixtures currently fail before their mutation, and the pair audit could not finish promptly. |
| H1 scalar-root RNG collisions | **fixed and independently demonstrated** | One-million-coordinate enumeration, deliberate reviewed v1 collision, purpose/seat/root/k separation, and worker/resume byte oracle passed. Gameplay uses `coordinate_rng` directly (`random.py:80-188`); scalar seeds remain diagnostic only. |
| H2 metric-only checkpoint ordering | **fixed and independently demonstrated** | Interrupted two-shuffle metric-only test passed with sums, square sums, wins, outcome counts, and ownership identical to uninterrupted execution. Merge occurs before ownership/checkpoint publication (`run_tournament.py:1478-1519`). |
| H3 game-stat winner duplication | **fixed and independently demonstrated** | Seat selection is anchored by `^P[1-9][0-9]*_strategy$` (`game_stats.py:95-106`). Hand oracle and independent fast recount found strategy observations exactly equal to `k*A`, not exposures plus wins. |
| H4 scope/schema/method/source authentication | **fixed and independently demonstrated** | Wrong scope, column, dtype, nullability, source, and version tests passed; root graph audits passed; representative clean sidecars bind canonical path, Arrow schema, sources, methods, design hashes, and clean code. Pair-audit performance is R2. |
| H5 permissive CLI parsing | **fixed and independently demonstrated** | Installed CLI tests passed documented post-subcommand seed selection and pre-write rejection of unknown options. The entry point now uses strict `parse_args` (`cli/main.py:308`). |
| H6 no actual end-to-end oracle | **fixed and independently demonstrated** | Real two-root raw-simulation-to-report orchestration passed in 201.8 seconds (internal 196.621 seconds), including authenticated no-force reuse without rewriting outputs. |
| M1 nonmonotone exact-power search | **fixed and independently demonstrated** | Independent small-case joint-binomial oracle, the review's `n=1` counterexample, and brute-force first-crossing fixtures passed. Current implementation scans admitted positive integers (`h2h_schedule.py:323-345`). |
| M2 root/order imbalance accepted | **fixed and independently demonstrated** | Compensating imbalance is rejected, exact Cartesian support accepted, and all 11,704 clean cells exactly match the frozen manifest with two roots and two orders per pair. |
| M3 mutable plan and block recovery | **fixed and independently demonstrated** | Cap authorization leaves plan bytes unchanged; deterministic replacement resumes at the smallest unauthenticated attempt; interrupted data/sidecar publication replays only the affected coordinate. |
| M4 unadjusted root significance labels | **fixed and independently demonstrated** | Root-stability tests passed and clean artifacts/report contain no `statistically_*`, significance, or rejection classification. Descriptive estimates remain. |
| M5 incorrectly ordered RNG diagnostic | **fixed and independently demonstrated** | Split-batch/seat global-order tests passed. Clean RNG stages are `complete_valid`, bind lags/cap/method, and use the semantic tournament-player order with zero-centered descriptive bands. |
| M6 TrueSkill calibration claim mismatch | **intentionally accepted limitation** | The estimator remains mu-only but is explicitly named `mu_softmax_heuristic`; fields, method contract, and claim text disclaim model calibration. Percentile candidate screening is unchanged. |
| P1 weak ingest semantic validation | **fixed but weakly tested** | Source now validates exact schema, internal coordinates, versions, contiguous unique game indices, canonical IDs, and outcome invariants while streaming (`ingest.py:234-306`). The clean run ingested successfully, but 16 negative integration cases fail earlier on obsolete completion fixtures and therefore do not independently demonstrate their named corruptions. |
| P2 sparse/zero-exposure semantics | **fixed and independently demonstrated** | Zero-exposure batch tests pass with exclusion recorded and estimates unchanged; partial/missing rectangular cells follow explicit failure/exclusion rules through root stability. |
| P3 compiled behavior and ID boundaries | **regressed elsewhere** | Compiled-Numba subprocess and canonical nullable/nonnumeric/mixed-ID boundary tests pass, but the same strict-ID change breaks supported simulation helpers (R3). |

## 5. Independent clean-fast recount

The recount read selected columns directly from curated Parquet files and used
independent integer accumulation rather than production aggregation helpers.
All count comparisons used tolerance zero; rate comparisons used `1e-15`.

| Root | k | Attempted A | Completed C | Safety S | Exposures `k*A` | Wins |
|---:|---:|---:|---:|---:|---:|---:|
| 36 | 2 | 172,000 | 170,459 | 1,541 | 344,000 | 170,459 |
| 36 | 4 | 86,000 | 85,996 | 4 | 344,000 | 85,996 |
| 36 | 5 | 68,800 | 68,800 | 0 | 344,000 | 68,800 |
| 37 | 2 | 172,000 | 170,499 | 1,501 | 344,000 | 170,499 |
| 37 | 4 | 86,000 | 85,996 | 4 | 344,000 | 85,996 |
| 37 | 5 | 68,800 | 68,800 | 0 | 344,000 | 68,800 |

Global tournament totals are `A=653,600`, `C=650,550`, and `S=3,050`.
Every safety row had null winner, winner strategy, winning score, victory
margin, all seat ranks, and all loss margins. Every completed row had one
rank-1 player and ranks `1..k`. Full semantic game coordinates were unique.

For every root/k/strategy, independent exposures, completed exposures,
safety-limit exposures, wins, losses, and per-attempt win rates matched
`performance.parquet`. Game-stat population observations equalled `A`; the
sum of strategy observations equalled `k*A`. TrueSkill cell and strategy totals
matched `A`, `C`, `S`, `k*A`, `k*C`, and `k*S`, demonstrating completed-only
rating updates with separately retained exclusions.

### H2H recount

| Quantity | Recount |
|---|---:|
| Candidates | 77 |
| Unordered pairs | 2,926 |
| Root/order cells | 11,704 |
| Attempts | 23,182,656 |
| Completed | 23,024,736 |
| Safety-limit attempts | 157,920 |
| Replacement attempts | 78,960 |
| Formal-test pairs | 2,916 |
| No-test pairs | 10 |

Every counts row exactly joined the immutable manifest on pair, strategies,
root, order, seats, target, cap, and block ID. The following held with integer
tolerance zero:

```text
attempted = completed + safety_limit
wins_seat1 + wins_seat2 = completed
wins_a + wins_b = completed
attempted <= max_attempts
completed = required target OR attempted = cap
replacement_attempt_count = attempted - initial target
```

The ten no-test pairs are the all-safety comparisons among strategies 0, 1,
20, 21, and 40. Each used 15,792 capped attempts and produced zero completed
games. All formal effects, intervals, p-values, and adjusted p-values are null;
Holm rejection and equivalence are false; family membership remains true.
Five candidates are reported operationally nonviable, the frozen family is not
shrunk, and no unique-best claim is permitted.

## 6. Artifact, lifecycle, atomicity, and resumability assessment

- The clean tree contains 75,609 files and 4,047,403,091 bytes: 26,059 files
  per root and 23,485 in the pair tree. There are 37,754 sidecars and 37,679
  Parquet files. The large file count is the principal operational concern.
- Both root artifact graphs independently pass current v3 scope/schema/hash
  validation. A scan found zero sidecars carrying v1/v2 contract identity,
  `unknown` code identity, or `development_dirty` identity.
- Active configs reload and reproduce their recorded hashes. Run contexts bind
  exact clean commit, parent lifecycle roots, resolved layouts, and execution
  controls. The pair/root contexts are internally coherent.
- Data, sidecar, and completion publication use staged atomic replacement;
  transient I/O has bounded provider-neutral retries. Tournament checkpoints
  use `atomic_path` (`run_tournament.py:585-605`). H2H writes one authenticated
  coordinate block at a time and advances execution state after durable block
  publication (`h2h_schedule.py:825-879`, `1318-1378`).
- Metric-only interruption, H2H replacement interruption, missing final stamp,
  and worker-count/resume oracles pass. The real end-to-end oracle's no-force
  rerun verifies idempotent reuse.
- The pair graph's exhaustive audit did not complete within the review window
  because of R2. Therefore the stored `release_audit.status=passed` was not
  fully reproduced from every block byte during this review, although aggregate
  manifest/count identities and all root graphs passed independently.

## 7. Performance and parallel-processing assessment

The clean fast run took approximately 12 hours 53 minutes from first dispatch
to final completion. It is an integration run, not a production runtime
benchmark, but it exposes important scaling behavior.

| Segment | Observed duration / behavior |
|---|---|
| Tournament game compute per k | about 1.2-1.5 minutes after workload preparation |
| One root simulation plus publication | about 26 minutes |
| One root analysis | about 49-55 minutes |
| Exact H2H planning | 13 seconds |
| H2H execution | about 2 hours 39 minutes for 23.18 million attempts |
| H2H inference | about 5 minutes 44 seconds |
| Reporting | about 6 minutes 20 seconds |
| Final validation tail | about 67 minutes |

Parallel execution is present and bounded: tournament/root work used 12
process workers with native threads capped to one; ingest used three processes;
H2H `n_jobs=0` resolved to available cores and used `ProcessPoolExecutor`
(`h2h_schedule.py:1581-1670`). Roots ran sequentially, which avoids launching
two 12-worker roots simultaneously on a 16-core machine. This is a reasonable
resource choice.

Two qualifications remain:

1. The orchestration worker policy derives root-analysis workers from
   `sim.n_jobs` and materialized 12 even though checked-in
   `analysis.n_jobs` is 4 (`two_seed_pipeline.py:103-120`, `147-149`). The run
   context records the effective 12, so provenance is transparent, but the
   public ownership/override semantics should be clarified or corrected.
2. Per-shuffle row files and per-cell H2H blocks provide strong coordinate
   recovery but create 75,609 files for this fast run. Publication, combining,
   and auditing are I/O/metadata dominated. Production capacity planning must
   include file-count and validation costs, not only simulated games/second.

No evidence supports reducing atomicity or resumability to improve speed.
Optimization should use manifest-indexed lookup, bounded shard aggregation, and
clear worker ownership while preserving exact coordinate recovery.

## 8. Commands and results

Inspection commands (`git status`, `git log`, `git diff --stat/name-status`,
`rg`, Parquet schema reads, JSON/log reads, and file inventories) were all
read-only and succeeded unless noted below. Validation commands were:

| Command | Result |
|---|---|
| `python scripts/check_structure_release.py --artifact-root <root36> --artifact-root <root37> --artifact-root <pair>` with 120-second limit | Timed out, exit 124. |
| Initial 17-module focused pytest command with 120-second limit | Timed out, exit 124; no pass assumed. |
| Same 17-module focused pytest command with 600-second limit | Timed out, exit 124; no pass assumed. |
| Split safety/RNG/lifecycle/CLI/diagnostic/compiled pytest set | 5 failures after 147.9 s: one strict-ID tournament fixture and four unauthenticated/noncanonical TrueSkill fixtures. |
| Targeted safety, RNG, metric resume, v3 primitives, lifecycle, installed CLI, diagnostic, heuristic-label, and compiled subprocess set | 97 tests passed in 95.5 s. |
| Targeted exact-power, H2H replacement/cap/recovery/noncompletion, fixed-family, and root/order balance set | 17 tests passed in 76.0 s. |
| `python -m pytest -q -s tests/integration/test_simulation_to_report_oracle.py` | Passed in 201.8 s; oracle reported 196.621 s. |
| Independent artifact recount, first attempt | Reviewer script error (`summary_level` label assumption); no product failure inferred. |
| Corrected independent artifact recount | Passed in 35.0 s with all counts/nullability/rates/manifests/guards exact. |
| Full three-root release audit with 600-second limit | Timed out, exit 124. |
| Root-36 release audit | Passed in 43.2 s. |
| Root-37 release audit | Passed in 42.6 s. |
| Pair-only release audit with 900-second limit | Timed out, exit 124; R2. |
| `python -m ruff check .` | Passed. |
| `python -m mypy src` | Passed: 75 source files, no issues. |
| `python -m pyright` | Passed: 0 errors, 0 warnings; newer tool version notice only. |
| `python -m black --check .` | Failed: 23 files would be reformatted. |
| `python scripts/check_terminology.py` | Failed on legitimate external API text and historical review/contract prose. |
| `python -m pytest -q --tb=short` | Failed after 844.5 s. Follow-up count: 84 failed of 1,017 collected. |
| Representative HGB/TrueSkill freshness-matrix rerun | 15 failures: HGB fixtures lack v3 strategy-manifest sidecars; TrueSkill fixtures use string IDs. The mutations were not reached. |
| Sidecar identity scan | Passed: zero v1/v2, unknown, or dirty sidecars found. |

No command modified the clean artifact tree. Pytest used its normal ignored
cache and temporary directories.

## 9. Required fixes and release sequence

1. Restore the public simulation-helper ID contract without weakening
   persisted canonical-ID validation.
2. Migrate all 84 failing tests to v3 inputs/canonical IDs or correct the
   implementation where the public behavior is still supported. In
   particular, make P1 negative corruptions and HGB/TrueSkill mutation matrices
   reach and assert their intended oracle.
3. Replace repeated recursive completion resolution with one canonical audit
   index and demonstrate a complete pair audit in a practical bounded time.
4. Make terminology enforcement hermetic and run Black on the maintained tree.
5. Rerun terminology, structure release audit, actual end-to-end oracle, full
   pytest, Ruff, Black, Mypy, and Pyright from a clean commit.
6. Because the fixes change the release commit identity, generate a new clean
   fast run in a new output root and repeat the independent recount/audit. Do
   not re-sidecar or promote the current bytes under the new commit.

## 10. Handoff

- **Files changed:** only
  `docs/reviews/Farkle_II_remediation_readiness_review.md`.
- **Behavior before/after remediation:** the reviewed base fabricated
  safety-limit winners, used reduced RNG roots, weak lifecycle identity, unsafe
  resume ordering, duplicated game-stat winners, permissive CLI parsing, and
  unsupported H2H claims. Current production artifacts implement the accepted
  outcome/RNG/lifecycle/H2H semantics, but the repository release gates and
  some public/test interfaces regressed during the v3/ID cutover.
- **Tests run:** all commands and outcomes are enumerated in Section 8; the
  principal statistical oracles pass, while the full suite, Black,
  terminology, and timely pair audit do not.
- **Remaining risks:** incomplete high-level mutation/negative-ingest evidence,
  public helper ID compatibility, pair-audit scaling, file-count/I/O scaling,
  and ambiguous analysis-worker ownership. The two-root fast run remains
  integration evidence only and cannot establish production precision or
  rank stability.
- **Existing artifacts stale:** the current clean fast artifacts remain bound
  to and internally coherent for commit `93c7660a...`; this review does not
  alter them. Any required code/gate fix produces a new commit identity, making
  these artifacts stale for the next release and requiring a new output root.
- **Exact next task unblocked:** implement a bounded release-gate recovery
  change that (a) restores public helper ID behavior, (b) migrates all v3/ID
  fixtures until all 1,017 tests pass, (c) indexes release-audit completion
  lookup, and (d) restores Black/terminology gates; then run a new clean fast
  oracle under the resulting commit.

**Disposition: suitable only after specified fixes.**
