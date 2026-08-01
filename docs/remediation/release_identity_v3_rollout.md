# Release identity v3 rollout

Status: authenticated v3 release path and public rollout switch implemented

Inventory date: 2026-07-30

## Final rollout disposition

This section is authoritative and supersedes the historical pre-rollout
inventory retained below. The public release path now accepts exactly
artifact/RNG/outcome/derived-schema/estimand/conditioning identity
`3/2/2/2/2/2`. Public validation rejects contract 2 and every mixed identity
before orchestration creates a results or log path.

All release producers still call the shared semantic metadata API, but when
the locked public configuration is used that API has only one reachable
publication path: typed authenticated-v3 atomic publication. Its contract-2
branch remains solely for explicit compatibility and negative fixtures.
Consumers validate exact v3 bytes, canonical location/scope, actual Arrow or
non-Parquet format identity, complete version/method identity, source sidecar
identity, and immutable manifest identity. A contract-2 source cannot be
promoted or consumed by adding a new sidecar.

| Release stage | Producer disposition | Consumer and completion disposition |
| --- | --- | --- |
| simulation | Strategy/workload/checkpoint, row/metric shards, and manifest roots publish as v3 with RNG 2, outcome 2, tournament method 2, run/code identity, and game-profile identity. | Ingest validates the authenticated simulation lifecycle and immutable manifests. Simulation completion owns every required artifact and manifest sidecar. |
| ingest | Per-k raw rows and stream manifests publish as authenticated v3. | Curate validates exact sources and schema. Completion is v3-only. |
| curate | Per-k curated rows and manifests publish as authenticated v3. | Combine and all direct row consumers validate exact v3 sources. Completion is v3-only. |
| combine | Per-k partitions now live under canonical `by_k/{k}p`; concat rows and its manifest use `concat_ks`. All publish as v3. | Metrics, game statistics, RNG diagnostics, and other row consumers validate the correct per-k or concat identity. Partition and stage completions are v3-only. |
| metrics: all-player, performance, and seat diagnostics | Every per-k, across-k, concat, and diagnostic product publishes as v3 with the applicable aggregation, uncertainty, support, and conditioning method identity. | HGB, screening, root stability, candidate selection, and reporting validate exact sources and typed methods. Nested and stage completions are v3-only. |
| game statistics and exact-roll diagnostics | Per-k game statistics and rare-event products, concat products, across-k summaries, and exact-roll outputs publish in their canonical scopes as v3. | Internal aggregation validates v3 sources; final root health requires all nested and stage completions. The game-stat cache scope contains only current configuration fields. |
| RNG diagnostics | Diagnostic output publishes with RNG-diagnostic method 3 and an authenticated concat source. Its freshness binds the public cap and normalized lags, and method metadata records effective capacity and tracked/skipped groups. | It is terminal diagnostic evidence; root lifecycle and final health require its v3 completion when enabled. |
| root TrueSkill | Ratings, diagnostics, and candidate contribution publish with TrueSkill method 3 and completed-game-only update conditioning. | Pair TrueSkill and candidate freeze require the exact typed TrueSkill identity. Completion is v3-only. |
| HGB | Feature/model/diagnostic outputs publish with HGB method 2, RNG method 2, and fold-construction method 1. | Screening and lifecycle consumers require exact v3 identities. Completion is v3-only. |
| descriptive screening | Across-k screening outputs publish as v3 without changing the descriptive estimand. | Candidate freeze continues to use the frozen win-rate and TrueSkill contribution family, not this output as a new selection edge. Root lifecycle still requires authenticated completion. |
| root stability | All fixed-design cross-root products publish with root-stability method 2 and exact root-owned source identities. | Candidate freeze, dominance, and reporting validate the cross-seed identities. Completion is v3-only. |
| pair TrueSkill | The pair-owned candidate contribution publishes as v3 with both root identities. | Candidate freeze requires its full v3 method, conditioning, source, and stage identity. |
| candidate freeze | Membership and the family manifest publish as v3. The family hash binds exact win-rate and TrueSkill source/stage/method identities. | H2H planning requires the authenticated frozen family. The multiplicity family is unchanged. |
| H2H immutable planning | Power plan and coordinate-sorted block schedule publish as v3 with H2H/power method 2, family hash, schedule hash, alpha, power, practical/equivalence settings, and RNG/outcome identity. | Execution and inference require the exact frozen family, plan, and schedule identities. A v2 block or plan is incompatible. The operational cap can authorize a previously frozen plan without changing its immutable schedule identity. |
| H2H execution and replacements | Execution state, every root/order block, and aggregate root/order counts publish as v3. Blocks bind the family, schedule, coordinate, source schedule, RNG namespace, completed target, attempt cap, and replacement attempts. | Recovery and fast finalization validate the exact immutable block set and sidecars; no aggregate can substitute for a missing block. Completion owns state, counts, and every block sidecar. |
| H2H inference and candidate viability | Counts, plan, schedule, execution state, and family are exact authenticated inputs. Combined counts, pairwise inference, root diagnostics, and agreement publish as v3 with inference method 2. | Dominance/agreement/reporting require the exact family/schedule and typed score-test, interval, Holm, practical/equivalence, and viability identity. The estimand remains completed games only; safety-limit attempts and operationally nonviable candidates cannot create claims. |
| dominance | Edges, SCC cycles, condensation fronts, and summary publish as v3 with dominance method 1. | Reporting validates their exact H2H/family/schedule source identities. |
| agreement | Selection-conditioned pairs and summary publish as v3 with structure-agreement method 1. | Reporting validates the exact frozen-family and method identities. Selection conditioning and statistical estimands are unchanged. |
| migration inventory and reporting | Migration report version 3, structure report contract 4, Markdown, plot, and JSON publish as v3. Retired artifact contents remain unread. | Reporting completion owns all final products and sidecars. No report is reusable when any source, method, version, or sidecar identity differs. |
| global lifecycle, audit, and health | Root/pair stage state is derived only from authenticated v3 completions. Run contexts are self-hashed, revalidated against adjacent active configs, and bind parent lifecycle roots and code identity. | The read-only release audit requires explicit fresh roots, rejects missing/orphan/mixed sidecars and incompatible completion inventories, and cannot succeed by discovering an old tree. `pipeline_health.json` remains operational non-evidence, but `complete_success` is gated on the authenticated graph audit and records accepted identity, run-context hashes, code identity, audit roots/failures, and release eligibility. Production requires release-clean Git identity. |

### Public switch and fast-run isolation

- `ArtifactContractConfig` defaults and all genuinely runnable checked-in
  configurations use `3/2/2/2/2/2`; applicable fixed method versions are added
  to each stage's `VersionIdentity`.
- The public lock is equality to artifact contract 3, schema 2, estimand 2,
  conditioning 2, RNG scheme 2, outcome schema 2, and the existing locked
  statistical configuration. There is no contract-2 transition on the release
  path.
- `configs/fast_config.yaml` preserves seeds `34` and `35`, keeps
  `sim.seed == sim.seed_list[0]`, adds conditioning 2, and uses
  `results_post_fast_config_remediated_20260730`.
- The prospective pair root
  `data/results_post_fast_config_remediated_20260730_seed_pair_34_35` and child
  roots ending in `_seed_34` and `_seed_35` were resolved read-only and did
  not exist at rollout time. The fast pipeline was not run.
- Contract-2 artifacts, schema-4/v2 completions, and mixed descendants are
  intentionally invalidated. The new prefix prevents path collision; equality
  validation prevents identity compatibility with old artifacts even if bytes
  are copied.

### Verification disposition

Focused tests prove v2-block rejection, wrong source/family/schedule rejection,
missing-sidecar completion failure, mixed v2/v3 release-audit rejection, exact
freshness identity, configuration lock/round trip, and the fully authenticated
tiny workflow. The Task 14 raw-simulation-through-final-report oracle passes
under the public `3/2/2/2/2/2` defaults, including a no-force replay and final
graph audit.

The next real fast-run preflight is intentionally deferred. It must not run
until these changes are reviewed, committed, and the repository is clean.

## Historical Task 15B implementation update

The source inventory below records the pre-15B baseline. The following
implementation status supersedes its v2 descriptions for the explicit
artifact-contract-3 path. Contract-2 behavior remains available only as a
compatibility path while the public defaults and runnable configurations stay
unchanged.

Task 15B is complete through frozen candidate-family construction:

- `authenticated_contract` now supports exact non-Parquet format identities
  for JSON, JSONL, UTF-8 text/YAML, PNG, pickle, and opaque binary artifacts,
  while Parquet continues to require its actual ordered Arrow field/type/
  nullability identity.
- The dual-mode publication boundary translates the already-implemented
  Tasks 0-14 semantic metadata into typed v3 version, method, stage-config,
  code, canonical path/scope, source-artifact, and immutable-manifest
  identities. It requires the complete global tuple `3/2/2/2/2/2` for
  artifact/RNG/outcome/derived-schema/estimand/conditioning and adds applicable
  tournament, RNG-diagnostic, TrueSkill, HGB, root-stability, candidate-family,
  and stage-operation method versions.
- Contract-3 source capture validates current source bytes, source sidecar
  bytes, canonical location/scope, actual schema or format identity, and the
  complete version tuple before a consumer reads or publishes. A contract-2
  sidecar cannot satisfy a contract-3 source request, and missing contract-3
  sidecars cannot be backfilled onto cached bytes.
- Simulation publishes authenticated strategy/workload/checkpoint/shard
  artifacts plus immutable coordinate-sorted row and metric manifest roots.
  Ingest validates the authenticated simulation lifecycle and those source
  identities before reading.
- Ingest, curation, concatenation, all-player metrics, performance, seat
  diagnostics, game statistics/roll diagnostics, RNG diagnostics, root and
  pair TrueSkill, HGB, descriptive screening, root stability, and candidate
  freezing publish v3 sidecars through the shared atomic lifecycle. Their
  ordinary downstream reads now validate the full v3 identity; sharded
  dependencies are bound by immutable manifest roots.
- Shared stage completion now constructs `AuthenticatedCompletion` only after
  every declared output has a valid v3 sidecar or immutable-manifest sidecar.
  Classification revalidates output bytes, sidecars, schemas/formats,
  stage-config/code/version identities, and available source/manifest
  provenance. Mutation therefore makes the stage stale and invalidates cached
  descendants through the existing stage dependency runner.
- Focused tests in `tests/unit/utils/test_release_identity_v3.py` cover source
  byte mutation, wrong scope, wrong schema, every required global identity
  dimension, v2-as-v3 substitution, invalid completion inventories, and
  simulation publication/completion.

Unresolved downstream work is deliberately outside Task 15B:

- Task 15C must migrate `h2h_power`, `h2h_execute`, `h2h_inference`,
  `h2h_digest`, `agreement`, migration inventory, and final reporting. The
  generic non-Parquet v3 primitives required by that work now exist, but those
  stages still use their v2 publishers and schema-4 completion path.
- Task 15D must make v3 the public default, migrate all runnable configuration
  files and documentation as one change, enforce release-clean identity before
  output creation, migrate final root/pair health and run-context gates, and
  replace the filename-based release audit with a validated v3 graph audit.
- `configs/fast_config.yaml` remains user-owned and was not modified or run by
  Task 15B. Any incomplete explicit v3 tuple is rejected before v3
  publication.

## Scope and release-path entry

This inventory is source-derived. It does not inspect, repair, delete, or
authenticate the known-bad completed fast-run tree identified in
`post_fast_config_remediation_contract.md`.

The public two-root release path is:

```text
farkle --config configs/fast_config.yaml two-seed-pipeline
  -> farkle.cli.main.main
  -> farkle.orchestration.two_seed_pipeline.run_pipeline
  -> for each root:
       simulation.runner.run_tournament
       analysis.build_root_stage_plan / StageRunner.run
  -> analysis.build_root_pair_stage_plan / StageRunner.run
  -> two_seed_pipeline._write_pipeline_health
```

`build_root_stage_plan()` executes `ingest`, `curate`, `combine`, `metrics`,
`game_stats`, `rng_diagnostics`, `trueskill`, `hgb`, and `screening`.
`rng_diagnostics` is enabled because the fast config inherits
`analysis.disable_rng_diagnostics = false`. The single-root tail is present in
the root registry layout but is not executed for a two-root run.

`build_root_pair_stage_plan()` executes `root_stability`, pair `trueskill`
contribution, `candidate_freeze`, `h2h_power`, `h2h_execute`,
`h2h_inference`, `h2h_digest`, `agreement`, and `reporting`.

### Current fast-config identity

The checked-in `configs/fast_config.yaml` is intentionally treated as
user-owned input and is not changed by this inventory. Its effective identity
is:

| Dimension | Effective value |
| --- | ---: |
| root seeds | `34`, `35` |
| player counts | `2`, `4`, `5` |
| artifact contract | `3` (explicit) |
| derived schema | `2` (explicit) |
| estimand | `2` (explicit) |
| RNG scheme / generator | `2` / `PCG64DXSM` |
| outcome schema / tournament method | `2` / `2` |
| baseline / k-support / weighting | `1` / `1` / `1` (inherited defaults) |
| conditioning / multiplicity / candidate family | `1` / `1` / `1` (inherited defaults) |

This is not runnable as release evidence. Loading succeeds, but
`AppConfig.validate_statistical_contract()` rejects it because
`_validate_statistical_contract()` is locked to artifact contract 2. If that
lock is bypassed, the first v2 sidecar publication constructs a sidecar with
the configured value 3 and then rejects it against
`artifact_contract.ARTIFACT_CONTRACT_VERSION == 2`.

The Task 5 adoption guide requires the eventual v3 version identity to include
artifact contract 3, RNG 2, outcome 2, derived schema 2, estimand 2, and
conditioning 2, plus applicable named method versions. Therefore the omitted
fast-config `conditioning_version` is also a release blocker; it currently
inherits 1.

## Identity mechanisms and classification

### Still-v2 artifact publication

`farkle.utils.artifact_contract` is the publication and validation layer used
by every release-path derived analysis producer. Its
`ARTIFACT_CONTRACT_VERSION` is 2. `make_artifact_sidecar()` records declared
scope, operation, free-form method metadata, config hash, source path strings,
selected manifest hashes, and exact output bytes. Publication goes through
`write_artifact_with_sidecar_atomic()`,
`publish_staged_artifact_with_sidecar()`, the wrappers in
`farkle.utils.artifacts`, or streaming sinks in `farkle.utils.writer`.

`validate_artifact_sidecar()` authenticates the adjacent sidecar's structure,
artifact basename, byte length, and content SHA-256. It does **not**
authenticate canonical physical location, actual Arrow field order/types/
nullability, exact ordinary source bytes and source-sidecar bytes, a typed
stage identity, or equality to the active config/code identity unless the
consumer separately asks for an individual field.

`ensure_artifact_sidecar_atomic()` can publish a missing v2 sidecar for
pre-existing bytes. That recovery rule is forbidden for v3 release evidence.

### Partially migrated lifecycle

`farkle.utils.stage_completion` publishes schema-4 `.done.json` stamps. These
stamps bind stage-scoped config, the global freshness dictionary, code and run
lineage, exact input/output bytes, and adjacent sidecar hashes. This is useful
hardening, but it is not `AuthenticatedCompletion`:

- path identities have ordinal roles such as `input_0000`, not canonical
  `CanonicalArtifactLocation` identities;
- semantic sidecar validation calls the v2 validator;
- outputs without sidecars are accepted;
- a successful stamp can therefore bind raw or v2 artifacts and report
  `complete_valid`.

`StageRunner.run()` accepts those stamps through `resolve_stage_state()`.
Consequently every release-path stage can currently complete without an
authenticated v3 artifact graph.

TrueSkill root/k cell seals and HGB freshness are additional partial
migrations. They bind exact bytes, code, parameters, and lineage, but their
published artifacts still have v2 sidecars and their final stage completion is
still schema-4 `write_stage_done()`.

### Available authenticated-v3 primitives

`farkle.utils.authenticated_contract` defines contract 3:
`CanonicalArtifactLocation`, actual `ArrowSchemaIdentity`,
`StageConfigIdentity`, `CodeIdentity`, `VersionIdentity`, typed
`MethodContract`, `StageIdentity`, `SourceArtifactIdentity`,
`ManifestRootIdentity`, `AuthenticatedSidecar`, and
`AuthenticatedCompletion`. Its publication/validation entry points are
`publish_authenticated_parquet_atomic()`,
`publish_immutable_manifest_atomic()`,
`validate_authenticated_artifact()`, and
`classify_authenticated_lifecycle()`.

At the inventory baseline, no release-path scientific stage called those
publisher/lifecycle entry points and ordinary publication was Parquet-only.
The Task 15B implementation update above supersedes that limitation: the
in-scope stages now use the authenticated lifecycle and the generic contract
has format-appropriate non-Parquet identities. The H2H/reporting consumers
listed under Task 15C have not yet adopted them.

## 1. Tournament and upstream analysis through candidate-family freeze

Unless a row says otherwise, every named derived artifact has exactly one
adjacent `.sidecar.json` produced and validated by the v2 mechanism above.
Manifests, checkpoints, active configuration, and schema-4 completion stamps
are control artifacts and do not currently have v3 sidecars.

### Tournament simulation

| Item | Current contract |
| --- | --- |
| Producer | `simulation.runner.run_tournament()` -> `run_single_n()` -> `simulation.run_tournament.run_tournament()`; completion is `runner.write_simulation_done()` |
| Published artifacts | Root `strategy_manifest.parquet`; for each `k`, `simulation_workload_plan.json`, `{k}p_checkpoint.pkl`, optional `{k}p_checkpoint.parquet`, `{k}p_metrics.parquet`, `rows/rows_*.parquet`, `rows/manifest.jsonl`, optional metric chunks/manifest, and `simulation.done.json` |
| Publication | Atomic raw Parquet/checkpoint/manifest writers; **no artifact sidecars**. `simulation.done.json` is a schema-4 `write_stage_done()` stamp over expanded output files. |
| Accepted identity | `simulation_is_complete()` uses `resolve_stage_state()` with the per-root/k stage-config hash. It authenticates recorded bytes, code, lineage, workload plan, and strategy manifest, but has no canonical v3 location/schema/source identity. |
| Consumers/validators | `_run_one_seed()` and `seed_has_completion_markers()` gate reuse with `simulation_is_complete()`. Ingest `_canonical_row_shards()` instead parses selected completion fields and manifest coordinates, checks exact raw Arrow schema and row semantics, but does not validate shard content hashes or the simulation lifecycle identity itself. |
| Completion | One `{k}_players/simulation.done.json`; `_root_lifecycle_identity()` includes its file hash in the root lifecycle root. |
| Versions/methods | RNG 2/PCG64DXSM; outcome schema 2; tournament method 2; workload plan version 1; optional game-profile contract 1; stage cache key 4; global fast identity shown above. |
| Status | **Partially migrated**: authenticated schema-4 lifecycle, no v3 artifacts. |
| Focused tests | `tests/unit/simulation/test_runner_branches.py`; `tests/unit/orchestration/test_authenticated_lifecycle_migration.py::test_simulation_completion_mutation_matrix`; `tests/integration/test_ingest_row_shards.py`; `tests/integration/test_raw_simulation_oracle.py::test_authenticated_raw_two_root_oracle`. |

### Root stages

| Stage | Producer and published artifacts | Consumers and validation | Completion, versions, status, and focused tests |
| --- | --- | --- | --- |
| `ingest` | `analysis.ingest.run()` and `_process_block()` publish `by_k/{k}p/{k}p_ingested_rows.raw.parquet` plus its stream manifest `{k}p_ingested_rows.raw.manifest.jsonl`. `_ensure_ingested_rows_sidecar()` may backfill a missing v2 sidecar. | `curate.run()` checks existence and exact expected Arrow schema but does not validate the ingest sidecar. Ingest validates raw schema/coordinates/outcome fields, not a v3 simulation source identity. | `ingest.done.json` binds raw outputs/manifests and v2 sidecars. Outcome 2, tournament 2, RNG 2, derived schema/estimand from config. **Still-v2 output; partially migrated completion.** Tests: `tests/integration/test_ingest_row_shards.py`, `tests/unit/analysis/test_curate.py`. |
| `curate` | `analysis.curate.run()` copies each raw input to `by_k/{k}p/game_rows.parquet`, writes per-k `manifest.jsonl`, and publishes/backfills v2 sidecars via `_curated_rows_sidecar()` / `_ensure_curated_rows_sidecar()`. | `combine.run()` discovers paths and checks Parquet readability/schema/row counts; it does not validate curated sidecar identity before use. TrueSkill and game stats also consume curated rows by path/schema. | `curate.done.json`; global versions only, no typed method version. **Still-v2 output; partially migrated completion.** Test: `tests/unit/analysis/test_curate.py::test_curate_publishes_and_backfills_row_sidecars_without_recopying`. |
| `combine` | `analysis.combine.run()`, `_write_partitioned_dataset()`, and `_write_concatenated_rows_from_partitions()` publish `concat_ks/all_ingested_rows_partitioned/{k}p_part-00000.parquet`, partition manifests/stamps, `concat_ks/all_ingested_rows.parquet`, and `all_ingested_rows.manifest.jsonl`. Parquet products have v2 sidecars. | `metrics.check_pre_metrics()` validates only the concat artifact's v2 scope/operation plus columns and row-count manifest. Seat analysis validates by-k v2 scope/operation plus columns. Game stats/RNG/TrueSkill use combined or curated paths and schemas without v3 source capture. | Partition stamps and `combine.done.json`; global versions. **Still-v2 output; partially migrated completion.** Tests: `tests/unit/analysis/test_combine.py`, especially `test_combine_writes_partitioned_dataset_and_partition_done`; `tests/unit/analysis/test_checks.py`. |
| `metrics` | `analysis.metrics.run()` coordinates three producers. `all_player_metrics.build_all_player_batch_metrics()` writes per-k `all_player_batch_metrics.parquet` plus stream manifests. `performance.build_canonical_performance()` writes per-k `performance.parquet`, across-k `performance_equal_k.parquet`, `performance_bootstrap.parquet`, `performance_control_contrasts.parquet`, and diagnostic `player_count_effects.parquet`. `seat_analysis.build_canonical_seat_analysis()` writes per-k `seat_batch_counts.parquet`, `seat_effects.parquet`, `seat_population_effects.parquet`, across-k standardized seat effects, and three diagnostics (`seat_exposure_mixture`, `seat_selfplay_p1`, `seat_mirrored_games`). | Performance validates all-player v2 scope and all-attempt conditioning plus a hand-coded schema. Seat analysis validates combined by-k v2 scope/source-scope/operation/support plus columns. HGB validates performance sidecars only for `scope=by_k`. Screening does not validate its performance/bootstrapping input sidecars. Root stability validates all-player v2 scope/conditioning plus schema. Candidate freeze/reporting validate selected performance operation/scope, not complete v3 identity. | Nested stamps plus `metrics.done.json`, all schema-4/v2-sidecar. All-player method parameters bind outcome 2 and tournament 2; estimand 2/schema 2 from fast config; all-attempt conditioning; equal-k aggregation; deterministic-batch uncertainty. **Still-v2 outputs; partially migrated completion.** Tests: `test_all_player_metrics.py`, `test_performance.py`, `test_seat_analysis.py`, `test_metrics_wiring.py`, `test_safety_limit_root_analysis.py`. |
| `game_stats` | `analysis.game_stats.run()` publishes per-k `by_k/{k}p/game_stats.{k}p.parquet`; row-preserving `concat_ks/game_length.parquet` and `margin_stats.parquet`; across-k `game_length_strategy_conditioned_equal_k_mean.parquet`, `margin_strategy_conditioned_equal_k_mean.parquet`, and `rare_events.parquet`; per-k rare-event shard Parquet/stats JSON; and exact-roll diagnostics from `roll_enumeration.run()`: `roll_outcome_distribution_exact.parquet` and `roll_summary_exact.parquet`. `rare_events_details.parquet` is disabled by fast config. | Screening depends on stage completion but does not read these files. Reporting reaches safety-limit values through performance, not these artifacts. Within game stats, aggregation relies on schema-4 stamps and v2 sidecars. No downstream consumer captures authenticated source identities. | Nested per-k/aggregate/rare-event/roll stamps and `game_stats.done.json`; outcome 2 applies to rows; exact roll selection rule is `production_max_immediate_score_v1`; equal-k estimand/conditioning comes from sidecars/config. **Still-v2 outputs; partially migrated completion.** Tests: `test_game_stats.py`, `test_game_stats_branches.py`, `test_roll_enumeration.py`, `test_safety_limit_root_analysis.py`. |
| `rng_diagnostics` | `analysis.rng_diagnostics.run()` publishes `diagnostics/rng_diagnostics.parquet`. | Reads `combine` concat rows by path and required columns; no input-sidecar validation. Output is not consumed statistically, but its completion state participates in final root health. | `rng_diagnostics.done.json`; cache key 5; diagnostic method version 3 (`_DIAGNOSTIC_METHOD_VERSION`), RNG 2, tournament-player namespace, descriptive zero-centered band, and cap/normalized-lag freshness plus typed capacity metadata. **Still-v2 output; partially migrated completion.** Tests: `test_rng_diagnostics_branches.py::test_one_frame_semantic_sequence_matches_hand_oracle`, `test_fragmented_batches_and_seats_match_one_frame_oracle`, and `test_rng_cap_lags_freshness_and_typed_metadata`; lifecycle health is covered by orchestration tests. |
| root `trueskill` | `analysis.trueskill.run()` -> `run_trueskill.run_trueskill_root()` publishes per-k `ratings_{k}_seed{root}.parquet`, auxiliary rating artifacts, `across_k/candidate_percentile_contribution.parquet`, and `diagnostics/screening_diagnostics.parquet`. | Root-pair `trueskill.run_root_pair()` and `build_percentile_contribution()` validate rating v2 scope, operation, conditioning, player support, and exact method dictionary. Candidate freeze validates the contribution likewise. | Root/k `_ShardDoneStamp` binds ordered input/rating/sidecar hashes and freshness, then final contribution/diagnostic schema-4 stamps. TrueSkill method/cell version 3; diagnostic method 1; outcome 2 completed-game-only conditioning with safety-limit exclusion. **Partially migrated seals but still-v2 artifacts and stage completion.** Tests: `test_run_trueskill_streaming.py::test_trueskill_corruption_cannot_be_blessed_and_missing_sidecar_recovery_is_bound`, `test_trueskill_cell_freshness_binds_parameter_code_and_method`, `test_trueskill_screening.py`, `test_trueskill_orchestration.py`. |
| `hgb` | `analysis.hgb_feat.run()` -> `run_hgb.run_hgb()` publishes per-k feature importance, held-out predictions, and fold metrics; `concat_ks/heldout_feature_importance_concat.parquet`; across-k `feature_importance_overall.parquet`, `future_simulation_proposals.parquet`, and `hgb_importance.json`. | Validates input performance only as a v2 `by_k` artifact and validates strategy-ID fields; ordinary strategy manifest has no sidecar. No release consumer uses HGB outputs directly after the stage; screening only depends on stage order. | `hgb.done.json` binds exact inputs/outputs/sidecar bytes and `_hgb_freshness_key()`. HGB method 2, HGB RNG method 2, fold construction 1, target `win_rate`, whole-strategy folds. **Partially migrated freshness, still-v2 artifacts/completion.** Tests: `test_hgb_feat.py::test_hgb_authenticated_completion_recomputes_on_contract_mutation` and `test_configuration_run_writes_heldout_artifacts_and_sidecars`. |
| `screening` | `analysis.screening.run()` publishes stage-root `descriptive_screening.parquet` and `.json`, each with a v2 sidecar. | Candidate freeze does not consume these files; it consumes canonical performance and TrueSkill contributions. Thus this is a required completed stage but not a candidate-family data edge. | `screening.done.json`; estimand 2, all-attempt conditioning, equal-k method, descriptive joint-batch uncertainty; no named method version. **Still-v2 output; partially migrated completion.** Tests: `tests/unit/analysis/test_screening.py`. |

### Pair stages through candidate freeze

| Stage | Producer and artifacts | Consumers/validators | Completion, versions, status, and tests |
| --- | --- | --- | --- |
| `root_stability` | `root_stability.build_two_root_stability()` publishes per-k `cross_seed/performance_root_combination_{k}p.parquet`, across-k combined performance, discrepancies, joint discrepancy, rank stability, top-N stability, bootstrap inclusion, control movement, shortlist changes, matched-count convergence, and half drift. | Reads each root/k all-player artifact with v2 scope/conditioning plus hand-coded schema. Candidate freeze, dominance, and reporting validate only selected v2 operations/scopes and columns. | `root_stability.done.json`; method 2, all-attempt fixed-design conditioning, equal-k or declared weights, deterministic-batch/bootstrap diagnostics. **Still-v2 outputs; partially migrated completion.** Tests: `test_root_stability.py::test_two_root_combination_and_stability_contract` plus scope/support negatives. |
| pair `trueskill` | `trueskill.run_root_pair()` -> `build_percentile_contribution()` republishes the pair-owned `candidate_percentile_contribution.parquet` from all root/k ratings. | Candidate freeze requires v2 operation `equal_root_k_percentile_mean`, combined-root seed scope, exact TrueSkill method dictionary and conditioning, plus columns/support. | `trueskill_percentile_contribution.done.json`; TrueSkill method 3. **Still-v2 output; partially migrated completion.** Tests: `test_trueskill_screening.py::test_percentile_contribution_requires_complete_root_k_support`, `test_candidate_family.py::test_candidate_family_rejects_prechange_trueskill_contract`. |
| `candidate_freeze` | `candidate_family.freeze_h2h_candidate_family()` publishes `h2h_2p/candidate_family.parquet` and `candidate_family.json`. The JSON embeds output `family_hash` and selected source metadata; notably only the TrueSkill source includes a sidecar hash in `source_identity`. | H2H planning `_load_frozen_family()` validates both v2 sidecars for scope/operation, then checks candidate membership and family hashes. It does not authenticate canonical locations, all source bytes/sidecars, or a v3 stage identity. | `candidate_freeze.done.json`; candidate-family version currently 1, estimand/schema 2, descriptive screening selection. **Still-v2 output; partially migrated completion.** Tests: `test_candidate_family.py`, especially provenance, prechange TrueSkill, cap, and scope negatives. |

### Task 15B exact implementation targets

Task 15B owns migration from simulation through candidate freeze:

1. Common API adoption:
   `utils.authenticated_contract.ArtifactIdentity`,
   `_current_artifact_identity()`, `publish_authenticated_parquet_atomic()`,
   `capture_source_artifact()`, `publish_immutable_manifest_atomic()`,
   `write_authenticated_completion_atomic()`, and
   `classify_authenticated_lifecycle()`.
   Add typed helpers for the raw simulation manifest roots needed by ingest.
2. Simulation:
   `simulation.runner.write_simulation_done()`,
   `simulation_is_complete()`, `_completion_output_files()`, and
   `simulation.run_tournament` row/metric manifest publication.
3. Root publishers:
   `ingest._ingested_rows_sidecar()`, `_ensure_ingested_rows_sidecar()`,
   `_canonical_row_shards()`, `run()`; `curate._curated_rows_sidecar()`,
   `_ensure_curated_rows_sidecar()`, `run()`; combine
   `_write_partitioned_dataset()`, `_write_concatenated_rows_from_partitions()`,
   `run()`.
4. Metrics:
   `all_player_metrics.build_all_player_batch_metrics()`;
   `performance._read_batch_metrics()`, `_write_frame()`,
   `build_canonical_performance()`; `seat_analysis._validate_source()`,
   `_write_batch_counts()`, `_write_frame()`,
   `build_canonical_seat_analysis()`; `metrics.run()`.
5. Remaining root analyses:
   `game_stats._write_scoped_game_stats()`,
   `_ensure_rare_event_sidecars()`, `_compute_k_game_stats()`, `run()`;
   `roll_enumeration.run()`; `rng_diagnostics.run()`;
   `run_trueskill._seal_rating_cell_completion()`,
   `run_trueskill_root()`; `trueskill_screening.publish_rating_cell_contract()`,
   `build_percentile_contribution()`, `build_screening_diagnostics()`;
   `hgb_feat.run()`, `run_hgb._write_hgb_frame()`, `run_hgb.run_hgb()`;
   `screening.run()`.
6. Pair upstream:
   `root_stability._read_cell()`, `_write_frame()`,
   `build_two_root_stability()`; `trueskill.run_root_pair()`; and
   `candidate_family._load_win_rate_contribution()`,
   `_load_trueskill_contribution()`, `freeze_h2h_candidate_family()`.

Every ordinary consumer named above must use exact logical-role source
identities and owning configs. Every sharded consumer must use an immutable,
coordinate-sorted manifest root. Remove all release-path calls to
`ensure_artifact_sidecar_atomic()`; no old bytes may be promoted.

## 2. H2H, downstream analysis, and reporting

| Stage | Producer and published artifacts | Consumers and current validator | Completion, versions, status, and focused tests |
| --- | --- | --- | --- |
| `h2h_power` | `h2h_schedule.plan_h2h_schedule()` publishes immutable `power_plan.json` and `block_manifest.parquet`. | Execute validates v2 scope/operation and separately checks plan/schedule/family hashes and method fields. Inference revalidates v2 scope/operation. These are strong logical checks but not canonical v3 artifact/source identities. | `h2h_power.done.json`; H2H method 2, conditional exact first-crossing power v2, score test ID `independent_two_proportion_score_v1`, RNG 2, outcome 2, configured alpha/power/deltas/cap. **Still-v2 outputs; partially migrated completion.** Tests: `test_h2h_schedule.py` power/allocation/immutability/cap tests. |
| `h2h_execute` | `execute_h2h_schedule()` publishes mutable `execution_state.json`, per-coordinate `blocks/pair_*_root_*_order_*.parquet`, and `root_order_counts.parquet`. | `_read_authenticated_block()` is named “authenticated” but uses the v2 validator plus expected block IDs/hashes/coordinates. `_completed_execution_is_recoverable()` validates v2 output/state/block sidecars and source path strings. Inference validates counts/plan/manifest/state v2 scope/operation, then joins logical cells to the schedule. | `h2h_execute.done.json`; schema-4 stamp binds state/count/block bytes, but semantic `sidecar_artifacts` at final write lists only state and aggregate count. H2H method 2, RNG 2, outcome 2, completed-game target with safety attempts retained, attempt cap 2.0, viability 0.99. **Still-v2 outputs; partially migrated completion.** Tests: `test_h2h_schedule.py` checkpoint, interrupted-sidecar, noncompletion-cap, and throttling tests. |
| `h2h_inference` | `h2h_inference.run_h2h_inference()` publishes combined order counts, pairwise inference, root diagnostics, and root decision agreement. | Dominance validates pairwise v2 scope/operation and hand-coded columns/support. Agreement/reporting validate selected output operations only. | `h2h_inference.done.json`; H2H method 2, score-test v1 ID, interval method `independent_two_proportion_score_inversion_v1`, Holm family, simultaneous bounds, completed-game conditioning. **Still-v2 outputs; partially migrated completion.** Tests: `test_h2h_inference.py` schedule mismatch, partial state, Holm, no-test, equivalence, and family-size tests. |
| `h2h_digest` | `dominance.build_dominance_outputs()` publishes dominance edges, SCC cycles, condensation fronts, and `dominance_summary.json`. | Agreement does not consume digest outputs. Reporting validates summary/front/cycle operation strings, but not full stage/source/method identity. | `h2h_digest.done.json`; H2H/family conditioning is recorded in free-form v2 metadata; no separate dominance method version. **Still-v2 outputs; partially migrated completion.** Tests: `test_dominance.py` graph, cycle, invariance, and incomplete-family negatives. |
| `agreement` | `structure_agreement.run()` publishes `selection_conditioned_pairs.parquet` and `agreement_summary.json`. | Reporting validates only expected operation and later root-scope agreement. | `structure_agreement.done.json`; selection-conditioned finite-family estimand, no named method version. **Still-v2 outputs; partially migrated completion.** Tests: `test_structure_agreement.py`. |
| `reporting` | `structure_reporting.run()` first calls `migration_audit.run()`, then publishes `structure_report.json`, `structure_report.md`, `tournament_screening_scores.png`, and the v2-sidecarred `migration_report.json`. | This is the terminal consumer. `_read_json()`/`_read_frame()` generally validate v2 operation only; `_performance_frame()` adds scope/operation/support; root scope equality is checked for selected sources. `migration_audit.run()` may reuse an existing report if its payload equals a freshly computed filename/size inventory and its v2 operation validates. | `structure_reporting.done.json` binds all four outputs and source bytes; report contract version 4; migration report version 2. **Still-v2 outputs; partially migrated completion.** Tests: `test_structure_reporting.py::test_reporting_writes_sidecar_validated_json_markdown_and_plot`, migration-audit tests, and the structure/simulation-to-report oracles. |

### Task 15C exact implementation targets

Task 15C owns the H2H tail and final delivery:

1. **Dependency completed by Task 15B:** `utils.authenticated_contract` now has
   explicit non-Parquet format identities and atomic publishers/validators for
   JSON, JSONL, UTF-8 text/Markdown/YAML, PNG, pickle, and opaque binary bytes.
   Actual Arrow schemas remain mandatory for Parquet.
2. Migrate `h2h_schedule._load_frozen_family()`,
   `_planning_sidecar_common()`, `plan_h2h_schedule()`,
   `_write_execution_state()`, `_read_authenticated_block()`,
   `_write_block()`, `_completed_execution_is_recoverable()`, and
   `execute_h2h_schedule()`. The immutable family, plan, and block-manifest
   roots and every block coordinate must be distinct typed identities.
3. Migrate `h2h_inference._read_counts()`, `_write_frame()`, and
   `run_h2h_inference()` with typed score-test, interval, ordinary/simultaneous
   alpha, Holm family, practical/equivalence, conditioning, family, and
   schedule fields.
4. Migrate `dominance._read_inference()`,
   `_read_tournament_screening_scores()`, `_write_parquet()`, and
   `build_dominance_outputs()`.
5. Migrate `structure_agreement._read_frame()` and `run()`.
6. Migrate `migration_audit.run()` and
   `structure_reporting._read_json()`, `_read_frame()`,
   `_performance_frame()`, `_by_k_vectors()`, `_write_text()`,
   `_write_plot()`, and `run()`. The migration inventory must remain a
   reporting-stage output and must never read retired artifact contents.
7. Replace every stage's `write_stage_done()`/`stage_is_up_to_date()` pair with
   authenticated completion/classification. Completion output inventories must
   include both immutable and mutable stage-owned artifacts exactly as
   appropriate; a valid H2H aggregate cannot stand in for unauthenticated
   block coordinates.

## 3. Global configuration, completion, and release-audit gates

### Configuration and runnable configs

| Target | Current state | Required v3 gate |
| --- | --- | --- |
| `config.ArtifactContractConfig` | Defaults artifact/estimand/schema to `2/1/1`; conditioning and all remaining derived dimensions default to 1. | After all producers migrate, default to artifact 3, schema 2, estimand 2, conditioning 2. Preserve intentionally chosen remaining public versions unless their methods change. |
| `config._validate_statistical_contract()` | Hard lock `artifact_contract_version == 2`. | Lock equality to 3 and the accepted complete version tuple; reject absent/unknown/mixed identity before path creation. |
| `AppConfig.freshness_key()` | Includes all version dimensions, RNG/outcome/tournament identity, k support, weights, and labels. | Use it only as input to typed `VersionIdentity`/`StageIdentity`; do not treat a free-form dictionary as authenticated method identity. |
| `configs/fast_config.yaml` | Explicit `3/2/2`, conditioning inherited as 1. | Preserve current user-owned seed/version edits during implementation; add conditioning 2 only as an intentional Task 15D config migration. |
| `configs/default_config.yaml`, `configs/farkle_mega_config.yaml` | Runnable release-audit configs explicitly use artifact 2, estimand 1, schema 1. | Migrate all runnable configs atomically to the accepted v3 tuple. |
| `configs/blank_config.yaml` | Null template values. | Update guidance/template consistently, but do not classify it as runnable. |
| `docs/config_reference.md` | Documents artifact contract 2. | Update only with the Task 15D public switch. |

### Orchestration and completion gates

| Target | Current behavior and gap |
| --- | --- |
| `cli.main.main()` | Loads config, applies overrides, and dispatches without calling `validate_statistical_contract()`. Release-invalid fast config therefore reaches output-producing orchestration. Validation must run before `_resolve_log_file()` creates a log parent or any writer creates paths. |
| `two_seed_pipeline.run_pipeline()` | Resolves `CodeIdentityPolicy.DEVELOPMENT_DIRTY` unconditionally. There is no public release-clean mode and CLI overrides are not passed to `write_run_context_atomic()`. |
| `run_contexts.write_run_context_atomic()` / `load_run_context()` | Canonically self-hashed run-context payload and parent lifecycle roots are good partial infrastructure, but `run_context.json` is not a v3 artifact/completion and the orchestrator does not validate it again before final health. |
| `seed_utils.write_active_config()` | Round-trips public YAML and writes `active_config.done.json` containing hashes, but that marker is not an authenticated completion and does not bind code/run context. |
| `stage_completion.write_stage_done()` / `resolve_stage_state()` | Schema-4 exact-byte lifecycle accepts v2 or no sidecars. It must not be the release-valid classifier after rollout. Compatibility use, if retained, must be unreachable from public v3 orchestration. |
| `analysis.stage_runner.StageRunner.run()` | Accepts any `complete_valid` returned by the schema-4 resolver. It must receive v3 stage definitions/locations and call `classify_authenticated_lifecycle()`. Required outputs in `StagePlanItem` are presently incomplete for some stages (for example H2H power declares only the plan, execute only state/count); authenticated completion must own the authoritative exact inventory. |
| `two_seed_pipeline._root_lifecycle_identity()` | Hashes schema-4 completion files after checking schema-4 state. Parent roots therefore can bless an all-v2 analysis graph. It must derive the lifecycle root from validated v3 completion identities. |
| `two_seed_pipeline._current_plan_states()` | Same schema-4 acceptance issue. |
| `two_seed_pipeline._write_pipeline_health()` | Writes an unauthenticated JSON file. `complete_success` is based on schema-4 states and has no release-audit proof. Health must bind the validated root/pair lifecycle roots, run contexts, release-clean code identity, and audit result; publish it as authenticated final evidence or explicitly keep it non-evidence and add a separate release attestation. |
| pipeline manifests | `manifest.py` schema version 2 authenticates event shape, not scientific artifact identity. Pair/root manifests are operational logs and must not substitute for v3 completion. |

Every current successful `write_stage_done()` call is a completion marker that
can succeed without authenticated v3 artifacts. There is no exception among
the enabled scientific stages. Simulation passes raw files without sidecars;
all analysis stages pass v2 sidecars.

### Release audit

`analysis.release_audit` still expects artifact contract 2:

- it imports `sidecar_path()` and `validate_artifact_sidecar()` from the v2
  module;
- `audit_sidecar_completeness()` infers canonical artifacts from suffix and
  directory-name heuristics, then validates v2 adjacent sidecars;
- it does not validate canonical `ArtifactIdentity`, actual Arrow schema,
  ordinary source/manifest roots, stage identity, completion inventory, run
  context, root/pair lifecycle roots, code cleanliness, or final health;
- `audit_runnable_configs()` calls the contract-2 configuration lock.

`scripts/check_structure_release.py` audits `default_config.yaml`,
`fast_config.yaml`, and `farkle_mega_config.yaml`. It currently fails on the
fast config's value 3 while the other two runnable configs remain old v2
identities. `tests/unit/analysis/test_release_audit.py` constructs and expects
artifact contract 2. The integration oracle helper
`tests/helpers/tournament_analysis_oracle.py` also explicitly asserts
artifact/estimand/schema `2/1/1`.

### Task 15D exact implementation targets

1. Public version switch:
   `config.ArtifactContractConfig`, `AppConfig.freshness_key()`,
   `_validate_statistical_contract()`, `load_app_config()` call sites, all three
   runnable configs, `blank_config.yaml`, and `docs/config_reference.md`.
2. Pre-write CLI/release policy:
   `cli.main.main()`, `two_seed_pipeline.run_pipeline()`,
   `run_contexts.write_run_context_atomic()`, and `seed_utils.write_active_config()`.
   Resolve one `RELEASE_CLEAN` identity for release evidence and pass it through
   all stage identities.
3. Stage runner/lifecycle:
   `analysis.stage_runner.StagePlanItem`, `StageRunner.run()`;
   `analysis.build_root_stage_plan()`, `_h2h_tail_plan()`,
   `build_root_pair_stage_plan()`; and
   `two_seed_pipeline._current_plan_states()`,
   `_root_lifecycle_identity()`, `_run_one_seed()`, and final health
   publication.
4. Release audit:
   replace `release_audit.audit_sidecar_completeness()` with a graph audit over
   expected canonical locations, typed methods/versions, exact sources or
   immutable manifest roots, authenticated completions, run contexts,
   lifecycle roots, and release-clean code identity. Update
   `run_release_audits()` and `scripts/check_structure_release.py` so the audit
   cannot report success without an explicitly supplied fresh artifact root;
   never default to or discover the known-bad tree.
5. Focused tests:
   migrate `test_authenticated_contract.py`,
   `test_stage_completion.py`, `test_stage_runner.py`,
   `test_stage_state.py`,
   `test_authenticated_lifecycle_migration.py`,
   `test_seed_workflows.py`, `test_main_cli.py`, and
   `test_release_audit.py`; update all fixture helpers that hard-code v2,
   especially `tests/helpers/artifact_sidecars.py`,
   `raw_simulation_oracle.py`, and `tournament_analysis_oracle.py`.
   Finish with `test_raw_simulation_oracle.py`,
   `test_structure_toy_oracle.py`, and
   `test_simulation_to_report_oracle.py` using only a new temporary output
   root.

Task 15D must be the atomic public switch. Until 15B and 15C remove every
release-path v2 publisher/consumer, changing only the defaults or lock would
make the pipeline fail during publication and would not create v3 evidence.

## 4. Compatibility-only code not reachable from the release path

The following may remain only if clearly isolated and tested as non-release
compatibility:

| Code | Reachability decision |
| --- | --- |
| `utils.random.spawn_seeds()` | Legacy reduced-width external-boundary helper. Tournament and H2H release RNGs use full semantic-coordinate constructors and do not call it. |
| `rng_diagnostics._iter_melted_batches()` and `_collect_diagnostics_streaming()` | Compatibility/test helpers. `rng_diagnostics.run()` uses `_iter_prepared_batches()` and `_collect_diagnostics_streaming_compact()` with the canonical global merge. |
| `simulation.run_tournament.WinTotals.add()` legacy-test-double branch | Compatibility behavior for test doubles; it must not define persisted release outcome identity. |
| v2 fixture constructors in `tests/helpers/artifact_sidecars.py` and old-version negative fixtures | Test-only after migration. Positive release-oracle fixtures must move to v3; explicitly stale v2 fixtures may remain as rejection tests. |
| old-manifest rejection paths in `utils.manifest.validate_manifest_contract()` and retired-key rejection in `config.apply_dot_overrides()` | Fail-closed compatibility guards; they do not publish or consume old artifacts as current inputs. |

`farkle.utils.artifact_contract` is **not** compatibility-only today: it is
imported by every release-path analysis producer, completion validation, and
release audit. It may be retained after rollout only if no public orchestration
or release audit can reach it. Likewise `migration_audit.run()` is on the
release path because reporting publishes its inventory; only its retired
artifact *contents* are deliberately unreachable.

Method identifiers ending in `_v1` are not automatically legacy artifact
contracts. `independent_two_proportion_score_v1`,
`independent_two_proportion_score_inversion_v1`, and
`production_max_immediate_score_v1` are current release-path method
identities. They must be carried as typed method versions in v3 unless the
underlying method changes; they must not be silently renamed merely to remove
the digit 1.

## Acceptance boundary for Tasks 15B-15D

The rollout is complete only when:

1. no enabled producer, consumer, completion check, final health check, or
   release audit imports the v2 artifact contract;
2. every declared output has a canonical physical identity, exact bytes,
   format-appropriate schema/format identity, typed versions/method, exact
   sources or immutable manifest roots, and exactly one adjacent v3 sidecar;
3. every enabled stage publishes an authenticated completion last, and only
   `classify_authenticated_lifecycle() == complete_valid` can skip or contribute
   to root/pair health;
4. the public version tuple is accepted consistently by defaults, runnable
   configs, active-config round trips, sidecars, completions, consumers, tests,
   and the release audit;
5. wrong path, Arrow type/nullability/order, source byte, source sidecar,
   manifest coordinate, method parameter, version, code identity, output byte,
   sidecar byte, or completion inventory independently fails closed;
6. the clean fast oracle runs only in a new configured output root, and neither
   implementation nor validation reads or modifies the known-bad artifact
   tree.
