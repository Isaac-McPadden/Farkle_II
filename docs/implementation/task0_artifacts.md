# Task 0: artifact and durable-processing specification

Status: frozen design for `first_release_task0_v1`, 2026-09-24. All new paths
and schemas below are **planned**, not claims about existing files or APIs.
See [statistical/support contract](task0_contract.md) for meanings and
[question map](task0_question_map.md) for coverage. No new canonical scope.

## 1. Ownership and paths

All expressions resolve from the owning run's AppConfig. Define
`M(k)=cfg.scope_dir('metrics', ArtifactScope.BY_K, k=k)` and
`H(k)=cfg.scope_dir('hgb', ArtifactScope.BY_K, k=k)`.
Path getters must be non-mutating. Task 2/7 adds named AppConfig getters wrapping
these exact expressions; never discover outputs by globs or hard-coded stage
numbers. Existing `cfg.ingested_rows_curated(k)` is the source of new row
reductions, with authenticated upstream lineage/plan/strategy manifest.

| ID | Planned canonical path expression | Owner / completion |
| --- | --- | --- |
| F | `M(k) / 'outcome_frequencies.parquet'` | Tasks 1–2, metrics child `outcome_distributions` |
| B | `M(k) / 'outcome_batch_stats.parquet'` | Same child |
| I | `M(k) / 'outcome_batch_inventory.parquet'` | Same child; certificate for complete batches |
| S | `M(k) / 'outcome_support.parquet'` | Same child; support for every strategy/view |
| Q | `M(k) / 'outcome_summaries.parquet'` | Task 2, same child |
| U | `M(k) / 'outcome_precision.parquet'` | Task 3, metrics child `outcome_precision` |
| B7 | `M(k) / 'seat_behavior_batch.parquet'` | Task 7, metrics child `seat_behavior` |
| Q7 | `M(k) / 'seat_behavior_summaries.parquet'` | Same child |
| O7 | `M(k) / 'turn_opportunity_batch.parquet'` | Same child; **game**, not exposure, accumulation |
| O | `M(k) / 'turn_opportunity.parquet'` | Same child |
| OA | `cfg.scope_dir('metrics', ArtifactScope.ACROSS_K) / 'turn_opportunity_equal_k.parquet'` | Task 7; complete configured k support |
| R1 | `H(k) / 'parameter_responses.parquet'` | Task 7, HGB child `parameter_responses` |
| R2 | `H(k) / 'parameter_nonadditivity.parquet'` | Same child |
| RI | `H(k) / 'parameter_response_support.parquet'` | Same child; exact matched strategy identities |
| PE | `cfg.scope_dir('rng_diagnostics', ArtifactScope.DIAGNOSTICS) / 'preflight_evidence.json'` | Task 5; separate validation workload evidence |
| MC | `cfg.scope_dir('rng_diagnostics', ArtifactScope.DIAGNOSTICS) / 'monitor_coverage.json'` | Task 5; declared required checks/disabled optional checks |
| MR | `cfg.scope_dir('rng_diagnostics', ArtifactScope.DIAGNOSTICS) / 'production_monitor.parquet'` | Task 5; root/batch monitoring results |
| G | `pair_cfg.scope_dir('reporting', ArtifactScope.DIAGNOSTICS) / 'analysis_guide.json'` and `'analysis_guide.md'` | Task 8; same resolved directory, separate sidecars |

Checkpoint roots are predetermined too: `M(k)/'_outcome_units'` for distribution
source slices/merges, `M(k)/'_outcome_precision_ranges'` for bootstrap range/tiles,
and `M(k)/'_seat_behavior_units'` for Task 7 reductions. Each owns an immutable
`plan.json`, the partition helper's `manifest.jsonl`, and `units/` outputs with
adjacent typed unit stamps. Unit relative paths are
`batch-{b:08d}/slice-{i:08d}.parquet`,
`merge-{generation:04d}/group-{i:08d}.parquet`, or
`range-{start:08d}-{stop:08d}/tile-{i:08d}.parquet`, as applicable. Independent
partial-statistic product families use named subdirectories beneath the same
slice coordinate. Unit plans enumerate exact paths rather than scanning them.
These files are authenticated resumability state, not additional top-level
stage completions. Their final manifests and retained units are included in the
owning child's authenticated output inventory when referenced by final sidecars;
never delete a referenced unit and leave a supposedly complete graph. Scratch
only relocates staging, not these canonical recovery authorities.

For a single-root permitted analysis use its reporting owner and explicitly
label single-root coverage. Do not fabricate second-root evidence. PE records
authenticated bounded validation evidence and its applicability to the current
run; validation games never enter tournament input rows. Supporting preflight
files, if required by Task 5, live in that owner's `diagnostics/preflight/`
subtree and are explicitly inventoried, not located by searching past results.

Root metrics completion enumerates F/B/I/S/Q/U/B7/Q7/O7/O/OA and their sidecars
when the respective features have been implemented/enabled by the declared
release contract. HGB completion enumerates R1/R2/RI. RNG completion enumerates
PE/MC/MR even when optional exhaustive lag analysis is deliberately omitted;
Task 5 must revise the current whole-stage disable behavior coherently. Reporting
and final byte-deep audit bind all required additions. No empty success marker
stands in for a missing producer. Child/unit completions use existing helper
namespaces and must not masquerade as new top-level canonical stages.

## 2. Physical schema conventions

These are Arrow types, independent of pandas inference. Every field is non-null
unless marked `?`. Empty products use the identical typed schema. Strings used
as statuses/metric IDs have closed enumerations in this contract; dictionary
encoding is a physical optimization, not a different logical type.

- `R`: `root_seed:int64, k:int16`.
- `K`: R plus `strategy:int32, seat:int16`. Actual seats are `1..k`; final
  all-seat views use `seat=0`. There are no negative or padded-seat keys.
- `BK`: R plus `deterministic_batch_id:int32, strategy:int32, seat:int16`.
  Batch artifacts retain **actual seats only**; overall batch vectors are sums.
- Counts and integer sums: int64, nonnegative, checked overflow (fail rather
  than wrap). Ratios/estimates: float64, null for undefined values; never NaN or
  infinity as a substitute for status. Integer enum-scale fields: int8.
- Row ordering is ascending identity fields in the order defined here, then
  metric/status/level fields as listed; numeric ordering for strategy/seat/values.
  Artifacts specify complete primary keys and prohibit duplicate keys.

### F: exact outcome frequencies

Fields: K, `distribution:string, value:int64, value_scale:int8, count:int64`.
Key: K + distribution + value. Only strictly positive count rows are stored.
Scale is 2 for `midrank`, 1 otherwise; the represented value is value/scale.

| distribution | Support / total count per K |
| --- | --- |
| `completed_score` | C |
| `competition_rank` | C |
| `midrank` | C; values are doubled integers |
| `losing_margin` | L, including zero-margin official losers |
| `winning_margin` | W |
| `stopped_score` | S |

Every final overall histogram equals the sum of its seat histograms, bin for
bin. A zero-support distribution has no positive F rows and an explicit S row.
Overall views are derived in reduction, not independent scans. Recorded engine
ranks remain unchanged upstream; do not overwrite them with competition ranks.

### B, I, S: sufficient statistics and completeness

B fields: BK, `attempted:int64, completed:int64, safety:int64, wins:int64,
losses_completed:int64, deficit_sum:int64, victory_margin_sum:int64,
support_status:string`. Key BK. Status is `observed` when attempted>0,
`verified_zero` when attempted=0 and I has certified that batch.
There are exactly `B_count * roster_unique_count * k` rows, with zeros streamed
only after completeness is certified. This is a row-count identity, not a license
to allocate a dense Python object grid in memory. S has K and the same seven
counts/sums/status fields, aggregated over batches; includes seat=0 and 1..k.

I fields: R, `deterministic_batch_id:int32, shuffle_start:int64,
shuffle_stop_exclusive:int64, games_per_shuffle:int64, expected_games:int64,
observed_games:int64, completed_games:int64, safety_games:int64,
expected_support_rows:int64, emitted_support_rows:int64,
coordinate_digest:string, source_identity_sha256:string,
roster_identity_sha256:string, support_status:string`.
Key: R + batch. Status is only `complete_valid` for publishable I rows.
Missing/invalid batch failures are separate failure records, never successful
I rows. Digest is SHA256 over validated game keys in canonical order using a
documented fixed-width little-endian encoding `<qhqi` (root,k,shuffle,game).
Range validation plus uniqueness precedes digest construction. Sidecars bind
the exact plan, sources, source sidecars/manifests, and roster identities;
digests are not a substitute for that authentication.

### Q and U: summaries and selected precision

Q fields: K, `quantity:string, statistic:string, conditioning:string,
denominator:int64, estimate:float64?, status:string`.
Key: K + quantity + statistic + conditioning.
Quantities are the six F distributions and the five summary IDs from section 3
of the main contract, plus `attempted_win_rate`, `attempted_loss_rate`, and
`safety_rate`. For F distributions, statistic is `mean`, `q05`, `q25`, `q50`,
`q75`, `q95`; for rates and selected mean IDs it is `estimate`.
Completed/conditional/stopped support determines the denominator explicitly.
Status is `ok`, `no_completed_support`, `no_conditional_support`, or
`verified_zero` (no attempted exposures). A value of zero with positive support
is an observed estimate, not missingness.
Conditioning values are `completed`, `completed_loss`, `completed_win`,
`safety_limit`, and `all_attempted`, determined by the quantity. Zero safety
support with positive attempted support uses `no_conditional_support`.

U fields: K, `summary_id:string, estimate:float64?, denominator:int64,
batch_count:int32, contributing_batches:int32, confidence:float64,
replicates_requested:int32, replicates_valid:int32, replicates_invalid:int32,
ci_low:float64?, ci_high:float64?, interval_width:float64?, mcse:float64?,
replicate_min:float64?, replicate_max:float64?, status:string, method_id:string`.
Key: K + summary_id. `method_id=outcome_batch_percentile_v1`. Summary IDs are
the four primary estimators plus `completed_loss_rate`. Status rules and
precedence follow main-contract section 3 in listed order; no attempted data
uses `verified_zero`. Requested R is recorded even when support policy skips
resampling; valid/invalid are zero then. When executed, valid+invalid=R. The
completed-loss row mirrors the win row's status and counts, estimate/bounds
are complemented, MCSE and width unchanged.

### B7, Q7, O7, O, OA: behavior and opportunity

B7 key/fields: BK + `termination_status:string` (`completed` or `safety_limit`),
`exposures:int64, zero_turn_exposures:int64, zero_round_exposures:int64,
turn_round_mismatch_count:int64, score_sum:int64, turns_sum:int64,
rounds_sum:int64, score_per_turn_sum:float64, score_per_round_sum:float64`;
and `<field>_sum:int64` for each of `rolls,farkles,hot_dice,highest_turn,
smart_one_uses,n_smart_one_dice,smart_five_uses,n_smart_five_dice`.
Emit both termination rows for every BK, including certified zero strata, after
I validates. Condition-specific exposure sums reconcile exactly to B. Existing
source fields are required in actual seats; a null is a data failure. Zero sums
with zero exposures are storage sufficient statistics, not observed zero means.

Q7 fields: K, `conditioning:string, metric:string, numerator:float64,
denominator:int64, estimate:float64?, status:string`.
Key: K + conditioning + metric. Conditioning is completed/safety_limit/all_attempted.
All exact underlying integer totals remain in B7 even if cast for this display
table. Metrics are `points_per_turn`, `mean_game_points_per_turn`,
`mean_game_points_per_round`, `round_proxy_gap` (exact game-rate minus proxy),
`turn_round_mismatch_rate`, `zero_turn_rate`, `zero_round_rate`,
`mean_turns`, `mean_rounds`, `mean_score`, `mean_highest_turn`, and
`<field>_per_turn`/`<field>_per_exposure` for the seven behavior count fields
other than highest_turn. Additionally `smart_one_dice_per_use` and
`smart_five_dice_per_use`. Status `ok` or `no_denominator`; estimates are null
when denominator=0. Highest-turn remains a per-exposure mean. Proxy gap numerator
is the difference of the two game-rate sums, denominator exposures.

O7 fields: R + batch, `attempted_games:int64, completed_games:int64,
safety_games:int64, completed_turn_range_sum:int64,
completed_total_turns:int64`. Key R + batch. Each game contributes once.
O fields: R, `seat:int16, completed_games:int64, turns_sum:int64?,
mean_turns:float64?, turn_share:float64?, mean_turn_range:float64?,
turn_share_imbalance:float64?, status:string`.
Key R + seat. Actual-seat rows contain turns_sum/mean_turns/share; the seat=0
row contains total turns, mean_turn_range/imbalance and null mean_turns/share.
Non-applicable quantities are null. With zero completed support all estimates
are null (`no_completed_support`); otherwise status `ok`.
OA fields: `root_seed:int64, required_ks:list<int16>, weights:list<float64>,
mean_turn_range_equal_k:float64?, status:string`. One row per root; weights and
values follow sorted k. Required valid k with zero completion yields
`no_completed_support`; missing required k blocks publication.

### R1, R2, RI: response tables with explicit matched support

Use the selection/matching rules in main-contract section 4. Model target is
all-attempt win probability; response values use probability units, not percent.
R1 fields: R, `parameter:string, level:int32, matched_backgrounds:int64,
observed_mean:float64?, oof_predicted_mean:float64?, support_id:string,
status:string`. Key R + parameter + level. Means average matched backgrounds
equally. A required HGB cell that legitimately declares insufficient fitting
support may have null predicted means with `prediction_unavailable`; this does
not hide missing artifacts that its completion promises.

R2 fields: R, `parameter_a:string, parameter_b:string, a0:int32, a1:int32,
b0:int32, b1:int32, matched_backgrounds:int64, observed_00:float64?,
observed_10:float64?, observed_01:float64?, observed_11:float64?,
observed_nonadditivity:float64?, oof_mean_00:float64?, oof_mean_10:float64?,
oof_mean_01:float64?, oof_mean_11:float64?, support_id:string, status:string`.
Key R + pair + levels. There is deliberately **no OOF interaction column**.
Statuses are `ok`, `unestimable_common_support`, or `prediction_unavailable`.
For unestimable rows, matched_backgrounds=0 and all estimates null. Parameters
without enough active levels get an RI status record even if R1/R2 has no rows.

RI fields: R, `support_id:string, product:string, specification_json:string,
background_id:string?, corner_or_level:string?, strategy:int32?, fold:int16?,
status:string`. Product is one_way/two_way/excluded_type. JSON is canonical
sorted-key JSON describing selected levels/parameters and matching attributes.
The immutable selection policy and source hashes bind support IDs (SHA256 of
canonical selection+matched IDs). Background ID is the canonical attribute-key
hash, always checked against its exact attributes. Row key is R + support_id +
background_id + corner_or_level; sentinel unestimable/excluded records have null
background/strategy/fold. Normal rows enumerate exact matched IDs and folds, so
support does not rely on a digest alone. This small metadata product has a
deterministic size bound of selected response cells times unique roster size;
stream it if a future configured grid is large. Never enumerate all raw games.
Canonical `favor_dice_or_score` maps to HGB's `favor_score` using the existing
feature conversion; booleans are level 0/1. All other matching attributes retain
their canonical types. When varying favor_score, exclude its corresponding
favor_dice_or_score attribute from the held-fixed background key; do not
accidentally hold the selected parameter fixed under its other name.
For excluded types, the specification records the exact
type and excluded strategy IDs/count; its support ID distinguishes each record.

### PE, MC, MR, G: coverage/results envelopes

PE/MC are typed JSON objects (`schema_version:1`) with `contract_id`,
`root_seed`, `required_ks`, `method_id`, `code_identity`, `config_identity`,
`rng_identity`, `workload_identity`, `environment`, `checks` (ordered array),
`source_artifacts` (exact inventory), `status`, and `applicability`.
Identity fields are non-null canonical strings/digests; required_ks is an array
of distinct sorted positive integers; environment/applicability are versioned
objects with named fields specified by Task 5's recipe. Status is
`complete_valid`, `investigate`, `insufficient_evidence`, or `omitted_by_design`
(only where the declared check is optional). Root evidence may contain a
statistical investigation warning while preserving successful execution;
deterministic failure never becomes complete_valid.
Each check has `check_id, required, coverage, minimum_support, fault_recipe_id,
clean_fixture_count, fault_fixture_count, clean_alert_count, detected_fault_count,
blind_spots, result_artifact`. Unsupported counts are null, not invented zeros.
Task 5 defines nonempty recipe/method identities and applicability rules before
publishing success. An optional omission has a reason; required omission fails.

MR fields: R, `check_id:string, batch_start:int32?, batch_stop_exclusive:int32?,
support:int64, statistic:float64?, reference_low:float64?, reference_high:float64?,
effect_threshold:float64?, result:string, reason:string, method_id:string`.
Key R + check + range. Result is `no_concern_detected`, `investigate`,
`insufficient_evidence`, or `omitted_by_design`. Statistical warnings do not
silently stop/replan simulations. Deterministic integrity failures remain
failures, outside a successfully complete MR publication. Scope omission is
not a passed diagnostic. Do not claim calibrated false-positive control solely
from a small clean-fixture count.

G JSON has `schema_version, contract_id, run_identity, question_groups,
methodological_notes, artifact_dictionary, coverage, limitations`. Each question
has workbook cells, status, estimand/conditioning, evidence paths and columns,
actual summary values and support, precision method, and claim limits. Markdown
is a rendering of that JSON; no independent result calculations or expected-result
prose. Task 8 reuses existing H2H/root reports and the coverage map here.

## 3. Bounded accumulation, resume, and identities

Use [byte-aware readers](../../src/farkle/utils/arrow_batches.py),
[partitioned stages](../../src/farkle/utils/partitioned_stage.py),
[stream manifests](../../src/farkle/utils/streaming_loop.py),
[artifact publication](../../src/farkle/utils/artifacts.py),
[atomic_path](../../src/farkle/utils/writer.py), and
[completion](../../src/farkle/utils/stage_completion.py). Verify their contracts
before implementing; do not create a second publication framework.

Use a deterministic external sorted reduction of fixed-width records. Let
Q be `cfg.resources.stage_batch_bytes['analysis']`; budget projected input,
record accumulation, sorting work, and writer buffers explicitly within that
working allocation, with at most Q/4 each. Account for Python/Arrow overhead in
the separate scheduler worker estimate; Q is not an OS process-tree limit.
Tiny Q configurations that cannot hold one complete game and minimum buffers
fail with an actionable error. Avoid Python dictionaries growing with all values
or the full strategy/seat/batch Cartesian product. Share source projection across
new consumers when useful; do not change existing performance inputs.

Task 1B publishes an immutable source-slice plan bound to source/plan identities:
root/k/batch plus canonical row-group and row-offset ranges, split greedily by
projected decoded bytes, including a logical range crossing row groups. A
large statistical batch may contain many durable slices. Source slices never
split one game's k-player calculation. Each slice produces sorted frequency
runs and partial sufficient statistics, then authenticates its unit outputs.
Cap the number of live records before sorting; a broad score range increases
run count, not unbounded resident memory. Persist the unit's exact consumed
source coordinates to detect holes/duplicates at the batch merge.

Merge with bounded fan-in (at most 16, reduced further to fit Q), deterministic
source/unit/key order, and checked int64 count addition. Runs/merge units are
checkpoint state; reuse authenticated completed ones. Batch merge validates
coordinate coverage, emits I and explicit B/B7 zero cells in bounded roster
chunks, and refuses a missing unit. Final reduce emits F/S/Q and behavior outputs
in stable key order. Durable units use `atomic_path` and existing adjacent
stamps; no partial canonical file is a resume authority. An interrupted slice
replays only that slice; merge work checkpoints by planned run group.

Bootstrap ranges use exact replicate IDs in half-open ranges of at most 100.
Process bounded strategy/seat tiles, retain B-length vectors, regenerate the
same root/k/replicate draw for every tile, and publish typed range/tile shards.
Never allocate R*B*all_strategies*all_seats. Per-cell replicate estimates may
be memory-mapped/bounded externally sorted for quantiles. Final precision
requires a validated complete range/tile inventory; partial replicates are not
a final interval. Parallelize independent root/k units or tiles/ranges through
existing process policies; suppress nested pools. The two-root orchestrator
and within-tournament scheduling are unchanged.

Use child method IDs `outcome_distributions_v1`, `outcome_batch_percentile_v1`,
`seat_behavior_v1`, `turn_opportunity_v1`, `parameter_response_v1`; schema version
1 for each new artifact family. Sidecars bind exact logical/Arrow schemas,
conditioning, complete support/weights, method/config/code/RNG identities, and
sources. In Task 2/3/5/7/8 advance affected stage freshness and enumerate outputs
in final audit/graph ownership. Do not relabel old v2/v3 bytes as new products.
Statistical identity includes summary selection, confidence/R, sparse-support
rules, and A89 selection/matching; workers, scratch path, and I/O slice byte
ceilings are execution choices. Changing physical slice layout may require
rebuilding incompatible unfinished unit plans; it cannot change final logical
results or simulation coordinates. A valid finished logical artifact remains
subject to current repository-wide freshness rules.

Canonical publication is data, adjacent sidecar, then successful completion.
Use authenticated final manifests for partition/range inventories. Temporary
growing files can use configured scratch; a destination-local atomic bridge
handles cross-volume publication. Existing valid child outputs must survive
stale-parent replay. Read-only source/path discovery does not create directories.

## 4. Required focused acceptance fixtures for later tasks

| Fixture | Required result |
| --- | --- |
| Complete scheduled batch, strategy has no exposure in one seat | Explicit B zero row after I certification; not a missing observation |
| Delete a batch, source partition, or one expected game | Fail; no I/final success, no zero-fill |
| Duplicate a game and delete another (same total count) | Fail exact coordinate/uniqueness validation |
| Delete a published B support row or a required actual-seat value | Fail; do not regenerate zero from a consumer join |
| Same strategy occupies two seats | Count two exposures and one game; never deduplicate by strategy/game |
| Scores 10000/8000/8000/5000 | Competition ranks 1/2/2/4; doubled midranks 2/5/5/8; one official win |
| Official losing player ties winner | Zero losing margin retained in L and histogram |
| Safety stop with valid scores, zero turns/score where permitted | S/stopped products only; no invented winner/rank/margin |
| C>0,L=0; separately C=0 | Zero deficit vs undefined losing severity; all completed summaries unavailable when C=0 |
| Batch vectors include zero-loss batches | Shared draws; ratio of resampled sums; no dropping or averaging conditional batch means |
| Positive denominators but degenerate/bootstrap-invalid support | Status and null CI as specified; no [0,0] certainty claim |
| Changed I/O batch size, workers, or interruption points | Identical logical counts/summaries, fixed statistical batches and replicate draws |
| Broad values/high-k support, deliberately tiny resource ceiling | Bounded accumulation and slice/merge resume; measure working memory and scratch |
| Missing/invalid bootstrap shard | Resume that unit; withhold final interval until inventory complete |
| Additive/nonadditive observed parameter quartets; impossible smart-one/five quartet | Correct contrast arithmetic; unsupported combination unestimable |
| Different fold-model offsets on an otherwise additive grid | No cross-model subtraction presented as a within-model interaction |
| Missing required k vs valid k with no completed games | Failure vs unavailable equal-k opportunity summary, never renormalized weights |
| Disabled optional lag diagnostic vs lost required monitor output | Omission with reason vs incomplete/failed stage |

These are future implementation acceptance checks, not tests claimed to pass
by this documentation change. Task 10 additionally exercises the real Windows
launcher/storage setup, authorization, corruption/recovery, and final deep audit.
