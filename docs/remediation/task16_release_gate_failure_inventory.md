# Task 16 release-gate failure inventory

Status: Task 4 HGB and TrueSkill authenticated-v3 mutation evidence complete

Review basis: `docs/reviews/Farkle_II_remediation_readiness_review.md`

Date: 2026-08-03

## Inventory provenance

This inventory was absent from the reviewed worktree at the start of Task 1.
The scoped entries below reconstruct the Task 1 failure family from readiness
review findings R3 and P3 and from the required pre-change focused run. The
pre-change command reproduced eight helper-ID failures. It did not access any
completed fast-run tree.

## Task 1 public-helper ID contract

The ordered strategy input is the identity source for public simulation
helpers. A caller-supplied canonical integer `strategy_id` is preserved.
Missing IDs receive the smallest nonnegative canonical integers not reserved
by any supplied ID, assigned in input order. IDs identify input positions, so
repeated objects and semantically equivalent strategies receive distinct local
IDs. Caller objects are copied rather than mutated. Duplicate caller-supplied
IDs fail before gameplay because canonical rows require distinct seated IDs.

Resolution occurs before serial or process-worker dispatch. The high-level
tournament runner resolves IDs before constructing its strategy manifest; the
low-level tournament helper and worker initializer use the same policy. Thus
serial and parallel execution return or aggregate the same IDs for the same
ordered inputs. The production grid's existing manifest IDs are all preserved.

`validate_simulation_row`, the Arrow raw-row schema, and artifact ingest remain
strict canonical boundaries. Descriptive, null, boolean, mixed-type, and
out-of-domain IDs are not accepted in persisted canonical rows.

## Owned failure dispositions

| Failure or finding | Pre-change cause | Disposition |
| --- | --- | --- |
| R3: strict IDs regressed public helpers | Descriptive `ThresholdStrategy` values reached strict row validation | Resolved by pre-row helper ID resolution; validation unchanged |
| P3: strict boundary regressed elsewhere | Canonical boundary evidence passed, but supported helpers failed | Resolved; strict scalar, pandas, Arrow, row, and ingest boundaries remain closed |
| `test_play_helpers_consistency` | Its private raw-row call supplied no IDs, then its public-helper assertion was never reached | Private raw-row fixture now supplies canonical IDs; ordinary public-helper coverage remains explicit |
| `test_parallel_simulation` | Both equivalent un-ID'd inputs produced the same descriptive string | Resolved; ordered positions receive local IDs 0 and 1 before pool dispatch |
| `test_simulate_many_games_from_seeds_matches` | Seed-list helper emitted a descriptive ID | Resolved; both seed APIs use the same resolver and preserve replay equality |
| `test_simulate_many_games_deterministic_counts` | Serial helper failed during row validation | Resolved without changing the expected seeded winner counts |
| `test_simulate_many_games_from_seeds_parallel` | Process workers received un-ID'd strategies | Resolved; parent-resolved IDs are identical in workers |
| `test_shuffle_rows_preserve_turns_and_rng_coordinates` | Direct tournament worker initialization retained descriptive IDs | Resolved; worker initialization applies the common ordered-input policy |
| `test_seed_reproducible` | Ordinary integration helper use failed before reproducibility comparison | Resolved; repeated calls return identical canonical integer-ID rows |
| `test_run_tournament_process_executor` | Throughput sampling failed before process dispatch | Resolved; IDs are assigned before sampling, and one/two-worker checkpoints agree |

## Added boundary and compatibility evidence

- Both public batch helpers accept ordinary descriptive strategies and return
  deterministic integer seat/winner IDs.
- Repeated objects and distinct-but-equivalent strategies are position-distinct
  and do not mutate caller objects.
- Supplied canonical IDs are preserved; local IDs avoid the complete supplied
  ID set; duplicate supplied IDs fail explicitly.
- Serial and multiprocessing results are frame-equal for the same seed and
  ordered inputs. Direct tournament process-executor checkpoints agree across
  worker counts.
- The high-level runner manifests custom helper-local IDs before tournament
  execution.
- Persisted-row validation rejects null, nonnumeric, numeric-string/mixed,
  boolean, negative, and greater-than-int32 IDs. Existing pandas/Arrow tests
  retain nullable and physical-type rejection evidence.
- Existing safety-limit tests retain null winner/rank/loss-margin semantics.
- Existing deterministic counts, seed-list equality, RNG-contract, compiled
  subprocess, and raw-simulation oracles show unchanged RNG-v2 replay.

## Focused commands and results

| Command family | Result |
| --- | --- |
| Pre-change helper/safety/tournament/integration/strategy-ID pytest set | 8 failures reproduced, all listed above |
| Post-change same focused family plus high-level manifest-ID test | 78 passed |
| Full `tests/unit/simulation` plus strategy-ID/RNG contracts and selected compiled/raw integration oracles | Task 1 paths passed; 3 pre-existing authenticated-v3 runner-fixture failures remain outside this task |
| Targeted Ruff on changed Python files | Passed |
| Targeted Mypy on three changed source files | Passed |
| Targeted Pyright on changed source and test files | Passed with 0 errors and 0 warnings |

The full pytest suite was not run, as required by the bounded task.

## Task 2 authenticated-v3 fixture contract

`tests/helpers/artifact_sidecars.py` is the approved shared entry point for
positive release-contract fixtures. It uses production configuration, canonical
path, Arrow schema, sidecar, immutable-manifest, and completion primitives; it
does not implement a validator or artifact lifecycle of its own. Explicit v2
helpers remain only for tests whose subject is rejection or compatibility of
old contracts.

The representative pre-change command proved that the ingest case stopped on
an obsolete completion payload, HGB stopped on a missing strategy-manifest
sidecar, and TrueSkill stopped on string strategy IDs. The existing lifecycle
control was already valid. After consolidation, the representative cases reach
their intended consumers with artifact contract 3 and the required RNG,
outcome, derived-schema, estimand, conditioning, and stage method versions.

### Shared migration recipe for Tasks 3 and 4

1. Create an isolated root with `make_authenticated_v3_config`; apply any
   public test-specific config changes and refresh `assign_config_sha` before
   publication. Use production `SeedRunContext` and `RootPairRunContext`
   constructors for pair layouts rather than adding a pair-fixture lifecycle.
2. Build strategies with canonical integer IDs and publish their full
   production `build_strategy_manifest` output through
   `publish_v3_strategy_manifest`.
3. Publish small Parquet inputs with `publish_v3_parquet`. Supply every exact
   upstream path so production sidecars bind source bytes, source sidecars,
   canonical scope/path, actual Arrow schema, method contract, clean code, and
   scoped config identity.
4. For simulation/ingest fixtures, publish one canonical row shard, immutable
   row manifest, workload identity, and authenticated completion through
   `publish_v3_simulation_run` / `runner.write_simulation_done`.
5. Run the intended consumer once and assert the unmutated control is accepted.
   Only then mutate one property. Use `mutate_artifact_bytes` for a raw-byte
   corruption, `mutate_json_identity_leaf` for one stored scope/source/method,
   manifest, or completion leaf, or republish one changed Parquet value/schema
   through `publish_v3_parquet` when all surrounding authentication must remain
   valid.
6. Assert the named consumer oracle: an exact semantic error for malformed
   ingest, `complete_stale` for lifecycle identity changes, or observed
   recomputation for HGB/TrueSkill freshness. A setup-side missing-sidecar,
   legacy-completion, string-ID, or noncanonical-path error is not an acceptable
   substitute.

### Representative evidence

- Ingest first accepted a canonical one-row control, then an independently
  authenticated shard whose only schema mutation replaced `winner_seat` with
  retired `winner`; ingest rejected the exact noncanonical-column condition.
- Root-stage lifecycle first resolved to `complete_valid`; changing only source
  bytes while retaining its sidecar resolved the same completion to
  `complete_stale`.
- HGB first ran once and skipped an unchanged authenticated rerun; republishing
  only the target value with a fresh valid source sidecar caused recomputation.
- TrueSkill first rated and sealed canonical integer-ID rows, then skipped a
  valid unchanged cell; republishing only the source row set caused cell
  recomputation.

### Task 2 focused commands and results

| Command family | Result |
| --- | --- |
| New authenticated-v3 helper tests plus the selected ingest, lifecycle, HGB-target, and TrueSkill-input adopters | 7 passed |
| Expanded lifecycle/HGB/TrueSkill mutation-matrix probe | Lifecycle and TrueSkill cases passed; the pre-existing HGB `features`, `code`, and `method` mutation mechanics remain for Task 4 |
| Targeted Ruff on the six changed Python modules | Passed |
| Targeted Mypy on the shared helper module | Passed |
| Targeted Pyright on the shared helper, its tests, and four adopters | Passed with 0 errors and 0 warnings |

No full pytest suite or production-scale pipeline was run. No completed
fast-run tree was accessed or modified.

## Task 3 non-model fixture migration

Task 3 repaired the remaining non-HGB/non-TrueSkill release-gate fixtures by
starting from authenticated artifact-contract-v3 controls and retaining strict
production validation. The repair changed tests only; no Task 3 production
defect was found and no statistical estimand changed.

### Assigned failure dispositions

| Family | Pre-change evidence | Disposition |
| --- | --- | --- |
| Remaining ingest integration | 15 failures stopped in an obsolete string-output completion payload before the advertised row corruption | Resolved. Every parameter first ingests an independent authenticated control, then publishes one independently authenticated corrupted shard and reaches its named semantic oracle. |
| Combine and curate | Seven combine tests lacked curated-input sidecars; one curate test lacked authenticated ingest inputs | Resolved with canonical by-k/concat-k publication and a production simulation-to-ingest control for curate. The obsolete flat combine-partition assertion now uses the canonical by-k path. |
| All-player metrics and game statistics | Three all-player and ten game-statistics tests stopped on missing curated/combined sidecars | Resolved with authenticated curated and concat-k sources, integer strategy columns, and v3 completion-location assertions. The attempted-exposure estimands and expected numerical results are unchanged. |
| Configuration and installed CLI | The current Task 3 baseline reproduced no failure in the assigned config/CLI command | Reclassified as already green: 72 tests passed without Task 3 edits. Strict installed CLI parsing remains intact. |
| Runner/orchestration lifecycle | Three runner tests attempted v3 promotion of unsealed bytes, asserted legacy completion metadata, or exposed an obsolete mock signature | Resolved. Fresh outputs are explicitly sealed, completion assertions inspect authenticated v3 output identities, canonical integer IDs are used, and mocks preserve the public helper signature. |
| Remaining simulation/tournament helpers | No additional failure remained after the Task 1 helper-ID repair and the three runner migrations above | Reclassified with evidence: the complete non-model simulation/tournament family passed. |

The pre-change focused runs therefore reproduced 39 Task 3 fixture failures:
15 ingest, eight combine/curate, 13 all-player/game-statistics, and three
runner lifecycle failures. All 39 are resolved as fixture migrations. No
expected exception text was changed merely to accept an earlier failure.

### Ingest negative-oracle evidence

The ingest matrix now keeps each corruption independent and asserts a specific
oracle after its clean one-row control is accepted:

- exact schema: retired or missing columns and a noncanonical string strategy
  physical type;
- internal coordinates: root, k, shuffle, and deterministic batch;
- internal versions: RNG scheme, RNG purpose namespace, and outcome schema;
- contiguous unique game indices: a gap and a duplicate key are distinct;
- canonical identities: repeated seated IDs, string-typed IDs, and a negative
  integer ID are distinct cases;
- outcome invariants: invalid winner, invalid ranks, inconsistent termination,
  victory margin, and loss margin are distinct cases.

The uncorrupted control is ingested and counted before each mutation. The
corrupted shard is then republished through the v3 fixture primitive, so its
schema/byte/source/completion authentication is valid and cannot mask the
intended semantic failure.

### Lifecycle evidence

Simulation and root-stage controls resolve to `complete_valid` before their
grid/input, output, sidecar, stage-config, code, or method mutation. Changed
source bytes, changed source sidecars, wrong scope, and wrong source identity
remain fail-closed in the authenticated-contract and release-identity v3
checks. Missing or legacy completion records remain non-valid/stale rather
than being promoted. Runtime-only worker changes remain non-staling. Task 3
did not weaken scope, source, method, or lifecycle validation.

### Task 3 focused commands and results

| Command family | Result |
| --- | --- |
| `tests/integration/test_ingest_row_shards.py` | 20 passed |
| Combine/branches/curate focused set | 13 passed |
| All-player metrics and game-statistics focused set | 45 passed |
| Unit config, unit CLI, and installed CLI focused set | 72 passed |
| Full unit simulation plus run-tournament integration | 211 passed |
| Stage completion, runner, authenticated lifecycle, stage runner, and seed workflows | 98 passed |
| Authenticated-contract and release-identity v3 supporting lifecycle checks | 38 passed |
| Combined unique Task 3 owned set | 397 passed in 120.8 seconds |
| Targeted Ruff on seven Task 3 Python files | Passed |
| Targeted Mypy on seven Task 3 Python files | Passed |
| Targeted Pyright on seven Task 3 Python files | Passed with 0 errors and 0 warnings |

The full pytest suite, HGB suites, TrueSkill suites, and production-scale
pipelines were not run. No completed fast-run tree was accessed or modified.

### Remaining ownership and artifact disposition after Task 3

The remaining fixture migration belongs to Task 4: HGB and TrueSkill
freshness, streaming, and orchestration. The release-audit performance repair
and repository-wide formatting/terminology gates remain separate readiness
review findings R2 and R4; Task 3 neither reclassifies nor repairs them.

Task 3 changes only test fixtures and test documentation. They do not make any
existing completed artifact current. The Task 1 production revision and every
future release revision still participate in code freshness, so completed
fast-run artifacts remain stale as release evidence and must not be promoted,
re-sidecarred, or used as a migration source.

## Task 4 HGB and TrueSkill mutation evidence

Task 4 migrated the remaining HGB and TrueSkill release-gate tests to accepted
artifact-contract-v3 controls. Every mutation case now accepts its unmodified
control first, changes one advertised property, and reaches the HGB or
TrueSkill consumer or freshness classifier. No HGB estimator, TrueSkill update
rule, candidate-family estimand, freeze-first rule, or
`mu_softmax_heuristic` name/limitation changed.

The pre-change focused commands reproduced five HGB and fifteen TrueSkill
failures. Their oracle-reached dispositions are explicit below.

### HGB assigned failure dispositions

| Assigned failure | Pre-change stopping point | Oracle reached after Task 4 |
| --- | --- | --- |
| `test_hgb_authenticated_completion_recomputes_on_contract_mutation[features]` | Raw manifest bytes were overwritten while the old sidecar remained, so validation stopped before HGB freshness | A full canonical manifest with the same integer ID and one changed feature is republished authentically; HGB accepts the control and recomputes on the changed feature-source identity. |
| `...[code]` | A mapping-shaped dirty-code fixture produced output identities that could not complete | A production `CodeIdentity` control completes and skips unchanged; changing only the clean commit identity causes HGB recomputation and a valid new completion. |
| `...[method]` | The legacy freshness dictionary changed, but v3 completion classification did not consult the current HGB method constants, so the stale cell skipped | V3 HGB output/completion classification now compares the recorded HGB, HGB-RNG, and fold-construction versions with the wrapper-owned current values; a method-version change reaches HGB and recomputes. |
| `test_hgb_force_recomputes_and_corrupt_bytes_cannot_be_blessed` | The fixture attempted to publish non-Parquet bytes with a newly fabricated Parquet sidecar and failed during setup | A valid HGB artifact is accepted first; changing only its bytes while retaining its bound sidecar makes completion stale and HGB recomputes. No sidecar is fabricated. |
| `test_configuration_run_writes_heldout_artifacts_and_sidecars` | The real HGB run stopped at a strategy manifest with no v3 sidecar | The full production strategy manifest and performance source are published with canonical integer IDs and v3 identities; the real held-out HGB run completes and every output binds both exact upstream identities. |

The expanded HGB matrix also proves both a missing strategy-manifest sidecar
and a one-leaf mutated strategy-manifest sidecar are rejected by `hgb_feat.run`
after the corresponding valid control reached and completed HGB. Target-value,
feature-source, hyperparameter, output-byte, code, method, and output-sidecar
changes each invalidate reuse independently.

### TrueSkill assigned failure dispositions

| Assigned failure | Pre-change stopping point | Oracle reached after Task 4 |
| --- | --- | --- |
| `test_run_trueskill_writes_only_root_k_ratings` | Descriptive `A`/`B` IDs failed canonical row classification | Canonical integer-ID completed rows stream successfully and publish only root/k ratings. |
| `test_run_trueskill_with_seed_suffix` | Descriptive IDs failed before suffix publication | Canonical integer-ID completed rows publish the exact seed-suffixed root/k rating. |
| `test_players_and_ranks_use_only_completed_canonical_rows` | Numeric strings failed before the completed-only oracle | Int32 IDs reach classification; only the two completed rows yield ranks and the safety-limit row yields no ranked update. |
| `test_safety_limit_rows_cannot_carry_ranks_or_become_draws` | String IDs masked the intended safety-rank rejection | A canonical-ID safety row reaches the outcome oracle and is rejected specifically for carrying non-null ranks. |
| `test_rate_block_worker_resumes_from_checkpoint` | String IDs stopped resumed streaming | An authenticated canonical curate source resumes at the declared batch, performs exactly the remaining completed update, and removes its checkpoint files. |
| `test_all_completed_ratings_are_unchanged` | String IDs stopped the reference comparison | An authenticated completed-only stream performs two updates and matches the independent sequential TrueSkill reference. |
| `test_mixed_support_excludes_safety_and_retains_prior_only_strategy` | String IDs stopped support accounting | Four canonical attempts produce two updates and two separately counted safety exclusions; the zero-completed-support strategy remains `prior_only_unrated` at the prior and receives no rank/update. |
| `test_trueskill_corruption_cannot_be_blessed_and_missing_sidecar_recovery_is_bound` | Rating/source/completion paths were outside the canonical v3 stage | Canonical source, rating, and completion controls seal first; missing sidecar, changed output bytes, and changed completion identity are independently rejected without blessing bytes. |
| `test_root_pair_trueskill_aggregates_complete_root_k_cells` | String `strategy` output schema failed the pair consumer | Four canonical root/k rating cells with exact authenticated sources aggregate successfully with complete support and exact upstream hashes. |
| `test_run_trueskill_root_rejects_incomplete_configured_k_support` | Missing curate sidecars masked the missing-cell oracle | All configured curate inputs authenticate first; the root orchestrator reaches and rejects the missing configured rating cell. |
| `test_run_trueskill_root_rejects_extra_k_cells` | Missing curate sidecars masked the extra-cell oracle | The configured source authenticates first; the root orchestrator reaches and rejects exactly extra cell `(11, 4)`. |
| `test_percentile_contribution_requires_complete_root_k_support` | Rating paths were outside canonical stages | Production root and pair contexts publish four canonical rating cells; the pair contribution accepts complete support and excludes the incomplete strategy from the frozen descriptive contribution. |
| `test_percentile_contribution_excludes_prior_only_rows` | Rating paths were outside canonical stages | Canonical authenticated cells reach screening; `prior_only_unrated` rows are excluded while evidence-backed integer IDs retain their percentile results. |
| `test_rating_cell_contract_does_not_repair_a_present_corrupt_sidecar` | The first attempted publication was already a negative/noncanonical setup | A valid canonical rating sidecar is accepted first; changing only its stored scope leaf is rejected, and the corrupt bytes are not repaired or replaced. |
| `test_tau_order_and_heldout_diagnostics` | Unauthenticated rating/game sources stopped diagnostic output publication | Canonical authenticated rating and game sources reach diagnostics; chronological, reversed-order, tau-zero, and unchanged `mu_softmax_heuristic` evidence publishes with authenticated method version 1. |

The TrueSkill cell matrix now proves its valid sealed control matches before
each mutation. Independently changed row-source content, Arrow schema,
source-conditioning sidecar, hyperparameters, output bytes, code identity,
TrueSkill method version, output sidecar, and completion freshness identity all
make the prior cell non-matching and force replay. Canonical int32 strategy
columns remain mandatory at persisted rating and game-row boundaries; the
existing null, string, floating, and boolean rejection controls remain strict.

### Task 4 focused commands and results

| Command family | Result |
| --- | --- |
| Pre-change `tests/unit/analysis/test_hgb_feat.py` | 5 failures reproduced; all stopped at the obsolete setup or missed method freshness described above. |
| Pre-change four-module TrueSkill family | 15 failures reproduced; all assigned stopping points are listed above. |
| Post-change HGB focused command | 16 passed. |
| Post-change four-module TrueSkill focused command | 48 passed. |
| Combined repaired HGB and TrueSkill family | 64 passed in 34.4 seconds. |
| Two tests rerun after type-only fixture cleanup | 2 passed. |
| Targeted Ruff on the Task 4 source/helper/test files | Passed. |
| Targeted Mypy on `src/farkle/utils/release_identity.py` | Passed: no issues. |
| Targeted Pyright on the Task 4 source/helper/test files | Passed with 0 errors and 0 warnings after correcting two fixture annotations. |

No full pytest suite or production-scale pipeline was run. No completed
fast-run tree was accessed or modified.

### Task 4 behavior and artifact disposition

Before Task 4, obsolete v2-like paths, missing sidecars, and descriptive IDs
prevented the high-level HGB and TrueSkill mutation evidence from reaching its
advertised oracle; an HGB method bump was also invisible to v3 completion
classification. After Task 4, valid controls are canonical authenticated v3,
every assigned mutation reaches its intended consumer, safety-limit attempts
remain excluded from TrueSkill updates and ranks, and prior-only strategies are
explicitly unrated.

The HGB completion-classification code change and all future release revisions
participate in code and method freshness. Existing completed fast-run artifacts
therefore remain stale for a future release revision and cannot be promoted,
re-sidecarred, or used as migration inputs. Task 4 created only temporary test
artifacts under pytest-owned directories.

## Compatibility and artifact disposition

This restores the pre-ID public-helper compatibility promise while changing
returned strategy fields from descriptive text to deterministic integers.
Consumers that display descriptions must retain their ordered input list or a
manifest; integer IDs are the supported row identity. `_play_game` remains a
strict internal raw-row constructor and does not silently adapt descriptive
IDs.

No completed fast-run artifact was read or modified. Successful production
grid simulations have the same logical strategy IDs because supplied manifest
IDs are preserved. Nevertheless, the code revision participates in artifact
freshness, so existing release artifacts cannot serve as evidence for a future
release commit and must not be re-sidecarred. Ephemeral helper results and
Python/test caches need no migration.

## Task 5 canonical release-audit completion index

Task 5 resolved readiness finding R2 by replacing per-completion recursive
path discovery with one canonical index per audit root. The typed key is the
full completion-schema location `(stage_key, scope, player_count,
relative_path)`. Index construction rejects duplicate locations, invalid or
ambiguous relative paths, and physical scope/k/path mismatches before any
completion lookup. Completion validation prefers the path derived from the
owning stage directory and canonical scope layout, then requires that exact
path to be the unique indexed value.

The validation contract remains fail closed. Every artifact still receives
full byte hashing, Arrow or non-Parquet schema/format validation, adjacent
sidecar validation, accepted global and named method-version checks, and
stage/source identity validation. Completion records still require the exact
authenticated-v3 schema, canonical ordering, exact output identity, and exact
sidecar hash. In-root source and immutable-manifest bindings are now also
resolved through the same index and compared exactly; cross-root identities
remain bound in the pair sidecar and are validated in their separately audited
owning root.

Focused production-backed tests cover canonical resolution, missing outputs,
duplicate keys during index construction, equal basenames in distinct scopes,
scope/k mismatches, traversal and ambiguous path spellings, and independent
byte/schema/sidecar/source mutations. A 128-output structural test observes one
recursive root traversal, 128 indexed locations, and exactly 128 completion
dictionary lookups. This proves bounded linear traversal plus one lookup per
output without a timing-sensitive microbenchmark. The focused result is 17
passed; no pre-existing test failure was merely reclassified or hidden.

The explicitly authorized read-only pair audit of
`data/results_official_fast_20260801_seed_pair_36_37/seed_pair_analysis`
completed with zero failures. It observed 23,485 files, 11,738 authenticated
artifact locations, nine stage completions, and 11,706 outputs in
`h2h_execute.done.json`. Index construction took 26.635 seconds, complete
artifact/completion validation took 119.778 seconds, and total measured wall
time was 147.063 seconds. This is 452.937 seconds (75.5%) inside the explicit
10-minute operational budget. The result is complete rather than a timeout or
partial audit.

No artifact format, statistical product, estimand, or completed fast-tree byte
was changed. The remaining release-audit scaling cost is the intentionally
preserved full byte and schema validation of every artifact; completion lookup
itself is no longer multiplicative in tree size.

## Task 7 release hygiene and fresh-run preparation

Task 7 resolves readiness finding R4 without changing any statistical
estimand, conditioning rule, artifact contract, RNG method, outcome schema,
derived schema, or applicable method version. Black formatted the maintained
tree, and the repository-wide Black check and Ruff gate now pass.

Terminology enforcement now covers maintained production source, runnable
configuration and tooling, current user-facing documentation, and current
artifact/estimand names. Historical reviews, remediation evidence, archived
migration prose, generated metadata, and result trees are outside the
normative scan. External API exceptions are symbol-span-specific; an allowed
`ProcessPoolExecutor` or `multiprocessing.Pool` spelling cannot exempt a
separate project-owned occurrence on the same line. Focused tests prove the
production-path rejection, historical-evidence exclusion, external-symbol
handling, case-insensitive matching, token boundaries, deterministic ordering,
and path rules. The five terminology tests and the standalone gate pass.

The isolated post-repair fast configuration uses roots 38 and 39, with
`sim.seed` fixed to 38 and prefix
`results_post_task16_repaired_fast_20260803`. All other accepted fast-run
settings are unchanged, including RNG diagnostic capacity and H2H capacity.
The public config round-trips to canonical config SHA-256
`8d0cfa0281c303216597dc5eb03a96b69a2f5a5672b70e25fe356575f0f335f2`.
The following prospective paths were resolved without creation and verified
absent:

- `data/results_post_task16_repaired_fast_20260803_seed_pair_38_39`
- `data/results_post_task16_repaired_fast_20260803_seed_pair_38_39/results_post_task16_repaired_fast_20260803_seed_38`
- `data/results_post_task16_repaired_fast_20260803_seed_pair_38_39/results_post_task16_repaired_fast_20260803_seed_39`

No completed result tree was modified or reformatted. The fast configuration
was not run, and the complete pytest suite was not run.

## Next task unblocked

Task 8 owns the clean release gate: run the complete hermetic validation, then
execute and audit the new isolated fast run. Existing completed artifacts
remain stale and may not be promoted or re-sidecarred. Commit and push remain
withheld pending review of the complete Task 7 diff.
