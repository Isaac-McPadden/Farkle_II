# Farkle II adversarial post-remediation synthesis

## 1. Executive verdict

Commit `6be5f5fa11df77155621bfc81188c7515f38f8de` implements most of the intended statistical formulas and report-language boundaries correctly, but the completed `fast_config.yaml` run is not valid release evidence and must not be used to authorize a full production run.

Two release-blocking root causes are confirmed:

1. A game that reaches the 200-round safety cap is still assigned a unique stable-sort winner and rank order. In the completed tournament, all 2,981 safety-cap games were tied and all 2,981 were credited to P1. In H2H, six comparisons among strategies 0, 1, 20, and 21 contain 47,376 nominal games recorded as P1 wins; a direct replay of pair 0/root 32/order 0/game 0 ended 0-0 after 200 rounds with both players marked `hit_max_rounds`, yet returned P1 as winner. The H2H aggregate discards termination status and reports simultaneous bounds of approximately `[-0.002544, 0.002544]` from zero demonstrated completed games.
2. The workflow has no uniform authenticated lifecycle. A no-force two-root run skips simulation from marker existence alone; completed TrueSkill cells ignore their inputs and hyperparameters and can re-authenticate altered bytes; HGB uses an mtime-only cache; stage dependencies are mainly path/size/mtime; every one of 11,586 sidecars records `code_revision: "unknown"`; and the pair configuration is internally contradictory and cannot be reloaded.

Additional confirmed High defects independently require correction: tournament game coordinates are truncated to 32 bits and collide across deterministic batches and k cells; metric-only interruption can checkpoint completed coordinates before their sufficient statistics; strategy-conditioned game statistics include `winner_strategy` as an extra seat and therefore duplicate every winner; the artifact validator accepts wrong physical scopes, fictitious declared columns, and unbound sources; and the public CLI silently ignores the documented post-subcommand seed override.

The strongest positive result is narrow but important: given the stored counts, the main performance and H2H arithmetic recalculates exactly. Chance baselines, Wilson workload checks, batch MCSE/t intervals, equal-k aggregation, joint vector resampling, TrueSkill percentile contribution, raw-count root combination, the seat-adjusted H2H estimator, constrained-null score p-values, Holm adjustment, Bonferroni practical bounds, dominance digestion, unresolved comparisons, and conservative report language are mostly implemented as designed. Correct arithmetic over fabricated or unauthenticated inputs is not sufficient for scientific validity.

**Primary disposition: suitable only after specified fixes.** This is not a call for a wholesale statistical redesign. The central estimands can be retained, but the termination, RNG, lifecycle/provenance, metric-resume, game-statistics, and artifact-boundary fixes below are release gates, followed by a fresh fast oracle run from the reviewed code.

## 2. Exact revision, configuration, outputs, and review scope

### Reviewed identity

| Item | Exact value |
|---|---|
| Checked-out commit | `6be5f5fa11df77155621bfc81188c7515f38f8de` |
| Commit date/subject | `2026-07-18T23:58:40-06:00`, `Fixed sidecar and audit issues` |
| Requested config | `configs/fast_config.yaml` |
| Config file SHA-256 | `bc10e0c4033c45dc4c496cfdfe586fd359b737d6c8fedaead306c13a8525fa74` |
| Recorded root/effective config hash | `1909cc1f0539f10bbfb766d9f50cb216bd07514d7580d92e6f1e064c46e48fcc` |
| Pair-sidecar context hash | `134bbb7f68d34b9cf4f20302ad15fe0e1d222d24005b0b77a888f5371b7dcb6a` |
| Persisted pair config SHA-256 | `ee3215182a6521ad8ce0c5bf713b4be952e6e5b9a8c5f09467ebc201735c282e` |
| Run root | `data/results_seed_pair_32_33` |
| RNG roots / k support | roots 32 and 33; k = 2, 4, 5 |
| Strategy support | 80 configurations, IDs 0-79; root manifests byte-logically identical |
| Recorded health | `complete_success` in `pipeline_health.json` |
| Actual current lifecycle exception | both `rng_diagnostics.done.json` stage hashes disagree with the current stage hashes |

The working tree began with `docs/reviews/` untracked; those files are the eight specialist reports supplied for this review. No application source, test, configuration, specialist report, or existing artifact was changed. This synthesis is the only file written.

### Representative output identities

| Output | Bytes | SHA-256 |
|---|---:|---|
| `pipeline_health.json` | 551 | `298b1af8fa248a57efd6ec48e98bea371a0bb5d4035be20a293a36f60af05669` |
| root-32 `performance_equal_k.parquet` | 14,248 | `19fd938bb0b17e308ba6fca77e5d64b06c096e8e87354928d7911f7aaff6c3d5` |
| pair TrueSkill contribution | 6,486 | `4918c6e2801d2d6eed8186e187c5edbc41ce9b8cbcd358b2a030e48c43453fd8` |
| `candidate_family.json` | 3,226 | `dea8b2349581551160a10c636f3e63cb30abfba2a670e3a443fb8caa747005ba` |
| `power_plan.json` | 2,447 | `bbf6e515330889dc2de6f684eb40b687391a08b27946b6bb920cb3f8d59d12e9` |
| H2H `root_order_counts.parquet` | 406,189 | `688d320ce9238f082d54f5115eb20d12caffce18d0be267d91d3fa5be6d389ff` |
| H2H `pairwise_inference.parquet` | 323,678 | `e61dddcb7d05809904f628f9ca3d2b4715fc802b24907e4ba05d6f2958a9d426` |
| `structure_report.json` | 1,291,768 | `a4fbabaa5e46477b384eefa624670a17ec4d2c33b8755d224ec47cbdadff4d5f` |

### Material reviewed

- All eight reports in `docs/reviews/post_fast_config/`.
- Current source under `src/farkle`, relevant tests and release scripts, `configs/fast_config.yaml`, and governing documentation under `docs/` plus `cli_args.md`.
- Both root simulation trees, all root analysis stages, the pair analysis tree, representative and aggregate sidecars, completion stamps, manifests, health state, the 11,400-block H2H inventory, and the final reports.
- Independent read-only Parquet/JSON recalculations, negative probes in temporary directories, a stored H2H coordinate replay, a CLI parser probe, configuration round-trip probe, lifecycle probes, exact-power counterexample, and the regression suite.

## 3. What was not reviewed or could not be verified

- The review did not replay all 653,600 tournament games or 22,503,600 H2H games. It replayed one decisive stored H2H abort coordinate and relied on full stored-row/count scans for the remaining empirical checks.
- It did not retrain every HGB fold or replay every TrueSkill update from raw rows. Fold separation and percentile contribution were checked from source/artifacts; stale-cache behavior was directly probed.
- It did not establish that the existing bytes were produced by the checked-out commit. The artifacts make that impossible: all sidecars say `code_revision: unknown`, artifact timestamps precede the reviewed commit, and code identity is absent from cache keys.
- It did not infer a population of RNG roots from two roots, strategy performance outside the finite configured grid, or causal effects from HGB. Those claims are outside the design.
- It did not validate production-scale runtime, memory, storage-provider failure rates, or 100-150 finalist report usability beyond the completed 76-finalist run.
- The existing fast run contains no configured equivalence margin, controls, mandatory diagnostics, candidate cap, H2H cycles, or shuffle-seed collision. Those branches require separate fixtures.

## 4. Major remediation-contract fulfillment

| Contract | Status | Synthesis |
|---|---|---|
| Finite-grid conditionality | Fulfilled | Final Markdown/JSON explicitly condition results on the simulated finite grid. |
| Six canonical scopes | Failed | Six combine partitions, twelve rare-event shard/stat files, and two HGB concatenations have physical-scope/sidecar-scope disagreement; mixed artifacts also blur row unions and across-k summaries. |
| Complete-support equal-k performance `mean_k(win_rate-1/k)` | Formula fulfilled; evidence invalid | Complete support and arithmetic are exact. Abort wins bias inputs and 32-bit collisions violate the declared independent-k construction. |
| Wilson workload resolution separate from batch MCSE/t and across-k uncertainty | Fulfilled | Separate columns and computations exist; Wilson is not used for winner inference. |
| Joint batch-vector resampling for nonlinear diagnostics | Fulfilled | One selected batch vector is applied across all strategies within k, and k is resampled separately. Test protection is weak. |
| Screening/TrueSkill/HGB/robustness separated from formal H2H inference | Mostly fulfilled | Final reporting is conservative. Root-stability artifacts still emit unadjusted `statistically_*` labels, and TrueSkill calls a mu-softmax heuristic a calibration output. |
| TrueSkill contribution from complete-cell within-root/k percentiles | Fulfilled computationally | Six-cell percentile means reproduce exactly; stale/corruption handling invalidates provenance. |
| HGB as held-out predictive association, not causation | Fulfilled statistically; artifact/lifecycle failed | Whole configurations are held out and interpretation says association, but HGB can be stale and its scope/quantity metadata are wrong. |
| Two roots as fixed-design reproducibility | Mostly fulfilled | Final language avoids root-population inference; root-stability significance labels cross the intended descriptive boundary. |
| Seat-adjusted `d_AB=0.5(q_AB-q_BA)` | Fulfilled for stored counts | Recalculated exactly; aborts make some stored counts non-Bernoulli outcomes. |
| Constrained-null independent two-proportion score inference | Fulfilled for valid Bernoulli counts | Formula, boundary handling, and selected oracle agree; allocation validation is insufficient and abort counts are invalid. |
| Holm decisions separated from Bonferroni practical bounds | Fulfilled | All 2,850 p-values enter one Holm family; practical bounds use separate Bonferroni score intervals. |
| Equivalence only by explicit-margin simultaneous containment | Fulfilled for configured branch | Equivalence is disabled here; no nonsignificant pair is called equivalent. Zero-completion pairs would be dangerous if enabled. |
| Exact implemented-test power over configured first-seat scenarios | Target met for this run; global minimum claim failed | 1,974 games/root/order gives 0.8003606089 and 1,973 gives 0.7996629815. Binary search is invalid for nonmonotone discrete power in the admitted domain. |
| Cycles, incomparability, unresolved comparisons retained | Fulfilled | 535 unresolved pairs remain, no forced total order, SCC/front logic preserves cycles in toy tests, and unique best requires direct practical dominance. |
| Exact `n_turns`; rounds only a diagnostic proxy | Fulfilled | Exact turn-weighted and game-weighted returns recalculate; rounds remain separately labelled. |
| Deterministic, atomic, resumable, provenance-bound state | Failed | Coordinate sorting and atomic publication are strong, but game RNG truncation, existence-only simulation completion, metric checkpoint ordering, weak caches, unknown code revision, and pair-config lineage violate the contract. |

## 5. Severity-ranked confirmed findings

### B1. Safety-cap games are fabricated as P1 wins and H2H reports precision from non-games

- **Severity / confidence / classification:** Blocker; High; confirmed correctness and statistical-validity defect.
- **Primary evidence:** `FarkleGame.play` sets `max_rounds_hit` and then stable-sorts scores and assigns unique ranks (`src/farkle/game/engine.py:408-480`). `_play_game` requires exactly one rank-1 winner (`src/farkle/simulation/simulation.py:359-413`). Tournament accumulation increments that winner (`src/farkle/simulation/run_tournament.py:214-238`). H2H reduces rows to P1/P2 counts and requires those counts to equal scheduled games, without a termination field (`src/farkle/analysis/h2h_schedule.py:768-830`). Full artifact scans found 2,981 tied tournament aborts, all credited to P1. The six affected H2H pairs occupy 24 blocks and 47,376 nominal games in `04_h2h_execute/h2h_2p/root_order_counts.parquet`; each block is all P1 wins. Direct replay of pair 0/root 32/order 0/game 0 (seed `10022994078775845994`) returned scores 0-0, 200 rounds, both max-round flags set, ranks 1/2, and winner P1. `pairwise_inference.parquet` reports `q_ab=q_ba=1`, `d_ab=0`, and simultaneous bounds about ±0.002544 for all six pairs.
- **Affected outputs and claims:** Tournament wins, chance deltas, MCSE inputs, seat effects, winner-conditioned summaries, screening, TrueSkill, HGB targets, root stability, candidate family, H2H counts/inference/power interpretation, dominance, agreement, and final reports.
- **Practical consequence:** Arbitrary seat-order tie breaking is treated as strategy evidence. For six finalist comparisons, apparent near-zero effects with narrow intervals come from no demonstrated completed games. Existing H2H aggregates cannot be repaired because abort state was discarded.
- **Smallest reasonable remediation:** Introduce explicit completion/termination status. Do not assign a winner or rank 1 to tied safety-cap aborts. Retain attempted, completed, tied, and aborted counts. Define all tournament denominators explicitly. For H2H, infer only from completed games; zero-completed comparisons are unresolved. If power targets completed games, deterministically schedule replacement coordinates while retaining attempts.
- **Focused regression/oracle test:** Force a 0-0 two-player max-round game through engine, tournament accumulation, all-player metrics, H2H block reduction, inference, and report. Assert no winner/rank 1, attempted=1, completed=0, aborted=1, no Bernoulli inference, and an unresolved comparison. Add a mixed completed/aborted denominator hand oracle.
- **Full-run regeneration:** **Yes.** Regenerate simulation, root analyses, candidate selection, all H2H blocks, inference, and reports.

### B2. The workflow does not authenticate completed work to inputs, method, configuration, or code

- **Severity / confidence / classification:** Blocker; High; confirmed cross-cutting provenance, cache-validity, and lifecycle defect.
- **Primary evidence:** Simulation completion is only `simulation.done.json.exists()` (`src/farkle/simulation/runner.py:231-239`), and the two-root orchestrator writes the active config before using that existence-only skip (`src/farkle/orchestration/two_seed_pipeline.py:185-200`). Changing the in-memory score thresholds to `[999]` left `seed_has_completion_markers` true. The marker contains no config, strategy-manifest, code, or output content hashes. TrueSkill's version-1 stamp contains only shard key/path/rows/time and skips before checking rows or environment (`src/farkle/analysis/run_trueskill.py:315-367,558-614`). `publish_rating_cell_contract` catches every contract error and republishes the current bytes (`src/farkle/analysis/trueskill_screening.py:41-83`); a temporary probe changed mu from 25 to 148, observed a hash failure, then saw the function preserve 148 and make the artifact valid. HGB reuses by mtime/existence/self-hash only (`src/farkle/analysis/hgb_feat.py:34-91`). Stage inputs use path/size/mtime fingerprints (`src/farkle/utils/stage_completion.py:126-154,236-250`). All 11,586 sidecars use `code_revision: unknown`. Pair stamps have `config_sha: null`; pair active config includes private runtime fields and fails `load_app_config` (`src/farkle/orchestration/run_contexts.py:46-84,137-160`; `src/farkle/config.py:1605-1611`).
- **Affected outputs and claims:** Every skip/resume claim; especially simulation conditioning, TrueSkill contribution/family selection, HGB evidence, pair-stage lineage, and the assertion that the reviewed commit produced the run.
- **Practical consequence:** Old, altered, or method-incompatible bytes can be presented under a new configuration. Code-only statistical corrections do not invalidate caches. The current artifact set cannot prove which source revision produced it.
- **Smallest reasonable remediation:** Adopt one canonical lifecycle for simulation and every root/pair stage. Bind stage-scoped configuration, method and RNG versions, code identity, exact upstream artifact/sidecar identities (or authenticated immutable-manifest roots), and required outputs. Missing-sidecar finalization must require an independent valid completion identity; a mismatched sidecar must fail or recompute. Persist a reloadable public config plus a separate run-context/lineage manifest.
- **Focused regression/oracle test:** A matrix that changes one simulation grid value, input byte, TrueSkill parameter, HGB parameter, code/method identity, output byte, and sidecar; each must become stale or recompute. Unchanged inputs must skip. `--force` must bypass completed cells. Pair config must load-round-trip and reproduce declared parent/context hashes.
- **Full-run regeneration:** **Yes for release evidence.** Existing bytes cannot be retroactively bound to the reviewed code. After the contract fix, run the fresh fast oracle from a clean output root.

### H1. Tournament coordinates are narrowed to collision-prone 32-bit game and shuffle seeds

- **Severity / confidence / classification:** High; High; confirmed RNG correctness defect.
- **Primary evidence:** Full `(root,k,shuffle,game)` coordinates are reduced with `dtype=np.uint32` (`src/farkle/simulation/run_tournament.py:192-201`), then used as the new root for all seat RNGs (`src/farkle/simulation/simulation.py:325-384`). Shuffle identity is narrowed similarly (`run_tournament.py:635-648,895-909`). Artifact scan found within-cell duplicate excesses 2,2,0 for root 32 k=2/4/5 and 5,1,2 for root 33, all 12 collision groups crossing deterministic batches. Across all k there were 11 duplicate excess values in root 32 and 14 in root 33; 7 and 6 groups respectively crossed k. This resolves the disagreement in report 07: explicit k in the pre-hash coordinate does not create disjoint downstream streams after the coordinate is collapsed to the same 32-bit root.
- **Affected outputs and claims:** Tournament outcomes, deterministic-batch MCSE, independent-k analytic uncertainty, joint resampling inputs, root reproducibility, and future shuffle-resume identity.
- **Practical consequence:** Nominally distinct games reuse all player streams; collision probability grows quadratically with workload. The fast fraction is small, but the scheme does not scale to production volumes and violates the declared independence construction.
- **Smallest reasonable remediation:** Derive each seat generator directly from the full semantic tournament coordinate plus seat. Do not use a generated integer as a new root. Version the RNG scheme. Use full coordinates, not a truncated seed, as shuffle recovery identity.
- **Focused regression/oracle test:** Generate all production-scale `(root,k,shuffle,game,seat)` identities across multiple roots and assert semantic-coordinate uniqueness; compare logical outputs across worker counts and interruption/resume. Include a deliberately colliding 32-bit pair to prove the new path remains distinct.
- **Full-run regeneration:** **Yes.** This is an RNG-scheme change; regenerate all tournament-derived outputs and the finalist/H2H tail.

### H2. Metric-only interruption checkpoints ownership before sufficient statistics

- **Severity / confidence / classification:** High; High; confirmed resumability correctness defect; not triggered by this completed row-producing fast path.
- **Primary evidence:** The parent updates wins and stores returned sums only in `collected_metric_chunks` (`src/farkle/simulation/run_tournament.py:1083-1143`), marks coordinates complete, and may checkpoint the old `metric_sums` (`:1166-1192`). Reduction occurs only after the processing loop (`:1194-1199`). On metric-only resume without row or metric-chunk manifests, completed coordinates are skipped and their sums cannot be reconstructed.
- **Affected outputs and claims:** Winner-conditioned sums, means, variances, and checkpoint-based metrics for allowed metric-only runs interrupted after an ownership checkpoint.
- **Practical consequence:** Wins may be complete while score/roll/farkle sufficient statistics omit pre-interruption work, without an exposed mismatch.
- **Smallest reasonable remediation:** Merge each returned payload into checkpoint state before marking coordinates complete and before checkpoint publication, or durably publish a replayable metric chunk before ownership advances.
- **Focused regression/oracle test:** Interrupt a two-shuffle run after the first persisted block with row and metric-chunk directories disabled. Resume and compare every logical checkpoint field with an uninterrupted run.
- **Full-run regeneration:** **No for this fast run on this defect alone.** Regenerate any result produced through the affected metric-only interruption path.

### H3. Strategy-conditioned game statistics double-count every winner

- **Severity / confidence / classification:** High; High; confirmed estimator/data-selection defect.
- **Primary evidence:** Both game-stat paths select every column ending in `_strategy` (`src/farkle/analysis/game_stats.py:598-675,2167-2237`), including top-level `winner_strategy` in addition to `P#_strategy`. In root-32/k=2, the published observation identity is exactly `raw_exposures + raw_wins`: strategy 27 is 7,064 = 4,300 + 2,764; strategy 78 is 7,013 = 4,300 + 2,713; strategy 0 is 4,495 = 4,300 + 195.
- **Affected outputs and claims:** Per-strategy game length, margin, quantile, close-game and rare-event rows; their concat/equal-k summaries and any report consuming them. Population rows updated once per game are not affected by this specific duplication.
- **Practical consequence:** Better strategies receive more duplicate observations, so summaries are neither game-containing-strategy nor player-game estimands.
- **Smallest reasonable remediation:** Select only anchored seat columns such as `^P[1-9][0-9]*_strategy$`; declare the observational unit; reject unexpected identity columns.
- **Focused regression/oracle test:** Use two hand-built games and assert each strategy's observation count equals its seat exposure count, not exposure plus wins, through rare-event and across-k outputs.
- **Full-run regeneration:** **Game-statistics stage and its consumers only** after the local fix; B1/B2/H1 independently require a full fresh run.

### H4. The artifact contract validates bytes, not the claimed semantic schema, scope, or sources

- **Severity / confidence / classification:** High; High; confirmed artifact-contract and provenance defect.
- **Primary evidence:** `validate_artifact_sidecar` checks sidecar fields, artifact name, size, SHA-256, and optional caller expectations, but never opens Parquet or validates declared columns/dtypes/sources (`src/farkle/utils/artifact_contract.py:587-619`). A temporary artifact with only `wrong_column:int8` and `consistency_columns=["required_but_absent"]` was accepted. A full scan found 20 physical-scope disagreements: six combine partitions beneath `concat_ks` labelled `by_k`, twelve rare-event shard/stat files beneath `across_k` labelled `by_k`, and two HGB concatenations beneath `across_k` labelled `concat_ks`. `release_audit` validates without deriving expected scope from the path (`src/farkle/analysis/release_audit.py:52-89`) and passes them. HGB also writes an active `feature_importance_long.parquet` in both roots, resolving report 07's contrary filename claim. Most `source_artifacts` are un-hashed path strings and most input-manifest-hash lists are empty. Pair config/pair hashes disagree as described in B2.
- **Affected outputs and claims:** Scope-based consumer safety, schema compatibility, source lineage, HGB quantity/scope claims, release-audit assurance, and reproducibility of pair artifacts.
- **Practical consequence:** A row union may be mistaken for an across-k estimate, malformed schemas can validate, and upstream substitution does not invalidate a sidecar.
- **Smallest reasonable remediation:** Derive and enforce scope from canonical path; store canonical Arrow schema fingerprints; bind source artifact and sidecar hashes plus expected operation/scope; split mixed-scope payloads; make typed method contracts carry concrete family/schedule/multiplicity parameters.
- **Focused regression/oracle test:** For each canonical scope, place a valid byte artifact in the wrong directory and require rejection. Test missing/wrong column/dtype/nullability, changed source bytes, wrong family/schedule hash, and mixed-scope JSON.
- **Full-run regeneration:** **No simulation regeneration solely for placement/schema metadata.** Relocate or regenerate the 20 affected artifacts and republish semantically complete sidecars. Existing release evidence still needs a fresh full run under B2.

### H5. The installed CLI silently ignores documented post-subcommand seed overrides and other unknown arguments

- **Severity / confidence / classification:** High; High; confirmed public-interface correctness defect.
- **Primary evidence:** Seed options are global (`src/farkle/cli/main.py:58-73`); the `two-seed-pipeline` subparser has only `--force` (`:242-251`); `main` calls `parse_known_args` and discards unknowns (`:298-303`). Parsing the documented `two-seed-pipeline --seed-pair 42 43` order produced `seed_pair=None` and unknown `['--seed-pair','42','43']`. `cli_args.md` documents that exact order.
- **Affected outputs and claims:** Root selection, artifact paths, run conditioning, and any invocation containing a mistyped option.
- **Practical consequence:** A costly run can silently use configured roots rather than requested roots.
- **Smallest reasonable remediation:** Use strict parsing, put command-owned controls on the subparser or a deliberately shared parent, and eliminate/delegate the duplicate parser.
- **Focused regression/oracle test:** Invoke the installed entry point with the documented order and assert roots 42/43; arbitrary unknown options must exit nonzero before any output is written.
- **Full-run regeneration:** **No for this run**, because its artifacts consistently identify roots 32/33. Audit or regenerate any run whose invocation relied on ignored post-subcommand options.

### H6. Release testing has no real raw-simulation-to-report oracle

- **Severity / confidence / classification:** High; High; confirmed missing validation evidence, not itself an estimator defect.
- **Primary evidence:** `tests/integration/test_structure_toy_oracle.py` manufactures batch metrics and ratings and replaces H2H simulation with `_toy_block_runner` (`:37-56,80-200,237-242`). Tournament integration heavily substitutes its grid/workers/workload and makes weak checkpoint assertions. No test executes `fast_config.yaml`. The current suite passed 830 collected tests when the non-hermetic terminology test was excluded, while all blockers and High defects above remained. The terminology gate fails because it scans the untracked specialist reports.
- **Affected outputs and claims:** Confidence that cross-stage semantic handoffs, lifecycle state, simulation outcome rules, scope, and report claims survive refactoring.
- **Practical consequence:** Schema-compatible but scientifically invalid workflows can remain fully green.
- **Smallest reasonable remediation:** Add a tiny, fully configured two-root CLI workflow using actual simulation, ingest, metrics, TrueSkill, HGB, a small frozen H2H family, inference, and reporting without replacing stage implementations. Add a read-only designated-fast-run audit command.
- **Focused regression/oracle test:** The end-to-end fixture itself, with hand-calculated termination counts, turns, equal-k estimates, hashes, H2H orientation, incomplete comparisons, lifecycle states, and report language.
- **Full-run regeneration:** **No.** This is a gate improvement; it must run successfully after correctness fixes.

### M1. H2H exact-power allocation search assumes a false global monotonicity

- **Severity / confidence / classification:** Medium; High; confirmed planning defect; current fast allocation meets target.
- **Primary evidence:** `_minimum_block_games` binary-searches exact power (`src/farkle/analysis/h2h_schedule.py:280-344`). With one root, family alpha 0.20, effect 0.10, seat scenarios 0/0.03/0.06, and target 0.40, block size 1 has worst power 0.5128, size 2 has 0.15148192, size 13 has 0.37912708, and size 14 has 0.40972647; the function returns 14 rather than the true minimum 1.
- **Affected outputs and claims:** The claimed smallest exact allocation, cap blocking, and projected workload for supported configurations.
- **Practical consequence:** Over-allocation and false cap failures are possible; no evidence of underpowering in the fast plan.
- **Smallest reasonable remediation:** Search without assuming monotonicity or prove/enforce a parameter region and lower bound that excludes earlier crossings.
- **Focused regression/oracle test:** Brute-force the full admitted small-n domain and compare the first target crossing, including the counterexample above.
- **Full-run regeneration:** **No for current H2H counts**, because 1,974 meets target and 1,973 does not. Regenerate plans/schedules affected by a changed allocation.

### M2. H2H inference validates balance only after pooling roots

- **Severity / confidence / classification:** Medium; High; confirmed latent validation defect; not triggered in the fast artifact.
- **Primary evidence:** `_read_counts` checks roots and duplicate cells but not exact planned support (`src/farkle/analysis/h2h_inference.py:292-367`); `_pairwise_estimates` requires equality only after root combination (`:370-437`). Root allocations `(AB,BA)=(100,200)` and `(200,100)` can cancel to equal pooled totals while confounding order with root. The fast run has exactly 1,974 games in every root/order block.
- **Affected outputs and claims:** H2H estimand and power if a malformed/recovered counts table has compensating root/order imbalance.
- **Practical consequence:** Root outcome differences can masquerade as order effects.
- **Smallest reasonable remediation:** Join counts to the immutable block manifest and require every pair/root/order identity, seat mapping, hash, and exact planned support before pooling.
- **Focused regression/oracle test:** Reject the compensated-imbalance fixture; accept the exact Cartesian schedule.
- **Full-run regeneration:** **No for the current fast run.** Recompute malformed H2H aggregates if detected.

### M3. H2H plan immutability and block-sidecar recovery do not meet the documented lifecycle

- **Severity / confidence / classification:** Medium; High; confirmed lifecycle defect.
- **Primary evidence:** The power-plan payload embeds mutable cap/authorization state and is rewritten when the cap changes (`src/farkle/analysis/h2h_schedule.py:592-676`), contrary to `docs/data_artifacts.md`. `_valid_existing_block` and missing-final-stamp authentication allow sidecar errors to abort rather than classify one coordinate for deterministic regeneration (`h2h_schedule.py:841-969`).
- **Affected outputs and claims:** Audit history of a blocked plan and resumability after an interrupted block-sidecar publication.
- **Practical consequence:** Cap history is overwritten; one unauthenticated block among 11,400 can require manual intervention.
- **Smallest reasonable remediation:** Keep immutable statistical design separate from mutable authorization. Treat a missing/invalid block sidecar as a pending coordinate and regenerate that block atomically; never bless existing unauthenticated bytes.
- **Focused regression/oracle test:** Raise only the cap and assert plan bytes/hash unchanged while authorization changes separately. Interrupt between block data and sidecar publication and assert only that coordinate replays.
- **Full-run regeneration:** **No** unless a published plan was overwritten or a block lacked valid authentication.

### M4. Descriptive root stability emits unadjusted significance classifications

- **Severity / confidence / classification:** Medium; High; confirmed claim-boundary defect.
- **Primary evidence:** `_classification` emits `statistically_above_below_practical` when an ordinary normal interval excludes zero (`src/farkle/analysis/root_stability.py:139-150`) across many strategy/root/k cells without a multiplicity rule. Final reporting does not repeat the label, but canonical cross-seed artifacts retain it.
- **Affected outputs and claims:** Root stability/discrepancy artifacts and consumers that treat labels as formal inference.
- **Practical consequence:** A descriptive fixed-design layer presents formal-looking unadjusted significance before finalist H2H.
- **Smallest reasonable remediation:** Remove significance classification; retain estimates, MCSE, practical-distance, and explicitly descriptive reproducibility flags.
- **Focused regression/oracle test:** Ensure root-stability artifacts contain no significance/rejection labels even when an ordinary interval excludes zero.
- **Full-run regeneration:** **Root-stability artifacts and reports only.**

### M5. Canonical RNG autocorrelation diagnostics do not use their declared global order

- **Severity / confidence / classification:** Medium; High; confirmed diagnostic-method defect.
- **Primary evidence:** The canonical path uses `_iter_prepared_batches` and `_collect_diagnostics_streaming_compact` (`src/farkle/analysis/rng_diagnostics.py:143-169`). Each Arrow batch is sorted and then `game_seed` is dropped (`:240-250`); the compact collector processes one seat column at a time (`:282-326`). The sequence is neither globally seed-sorted nor seed-merged across seats. Both current RNG done stamps also have stage hashes different from the current code's expected hashes.
- **Affected outputs and claims:** `05_rng/diagnostics/rng_diagnostics.parquet` lag correlations and top-level completion status.
- **Practical consequence:** Values do not estimate autocorrelation in the sequence named by the module; `complete_success` also overstates current freshness.
- **Smallest reasonable remediation:** Globally merge-sort semantic coordinates, merge seats before group updates, and define a zero-centered reference band or accurately named estimate interval. Recompute health from canonical states.
- **Focused regression/oracle test:** Split a known ordered sequence across Arrow batches/seats and compare streaming output to a one-frame oracle; assert current-stage freshness before `complete_success`.
- **Full-run regeneration:** **RNG diagnostic stages and health only**, after the RNG scheme rerun required by H1.

### M6. TrueSkill “calibration” is a mu-only softmax heuristic

- **Severity / confidence / classification:** Medium; High; confirmed method/claim mismatch; potentially reasonable as a renamed descriptive heuristic.
- **Primary evidence:** Held-out probabilities are `softmax((mu-max(mu))/beta)` and ignore sigma (`src/farkle/analysis/trueskill_screening.py:266-313`), yet outputs include `top_probability_calibration_gap` and governing docs call them predictive-calibration results.
- **Affected outputs and claims:** Held-out log loss, Brier score, top probability, and “calibration” language; percentile candidate contribution is not affected by this formula.
- **Practical consequence:** Values describe an arbitrary mu-ranking link, not predictive calibration of the fitted TrueSkill model.
- **Smallest reasonable remediation:** Either use a documented model-consistent predictive approximation incorporating mu and sigma or rename the outputs/method contract to `mu_softmax_heuristic` and remove calibration claims.
- **Focused regression/oracle test:** Two fixtures with equal mu/different sigma and different mu/equal sigma must distinguish model-consistent probabilities from the current heuristic; verify exact label/method contract.
- **Full-run regeneration:** **Diagnostic TrueSkill outputs only** if the estimator changes; metadata/report regeneration if only renamed.

## 6. Plausible risks requiring further evidence

### P1. Ingest does not authenticate row coordinates and gameplay invariants inside shards

- **Severity / confidence / classification:** Medium; High; plausible corruption risk / missing validation; no violation observed in the fast artifacts.
- **Primary evidence:** `src/farkle/analysis/ingest.py:106-207,214-261,422-445` validates manifest containers/counts and pads missing canonical columns, but does not universally require internal root/k/shuffle/batch identity, exact game-index support, unique row keys, winner/rank/margin consistency, or distinct seated strategies.
- **Affected outputs and claims:** Every downstream count and source-lineage claim if a shard is swapped, duplicated, or internally malformed.
- **Practical consequence:** A structurally readable but semantically wrong shard can receive a valid sidecar.
- **Smallest reasonable remediation:** Stream-validate identity/outcome invariants and reserve nullable padding for later-seat columns only.
- **Focused regression/oracle test:** Swapped shard, duplicated game key, wrong internal shuffle, invalid winner, missing identity column, repeated strategy, and bad margin fixtures must fail.
- **Full-run regeneration:** **Only if audit finds malformed inputs.**

### P2. Zero-exposure and missing batch-cell semantics are not consistently defined

- **Severity / confidence / classification:** Medium; Medium; assumption-sensitive contract gap; not triggered in the rectangular fast run.
- **Primary evidence:** Performance rejects nonpositive exposure and its pivot can fill absent strategy/batch cells with zero (`src/farkle/analysis/performance.py:60-88,261-281`), while sidecars say missing cells fail and root stability applies stricter rectangular checks. Fast data are exactly 80 strategies × 100 batches with 43 exposures/cell.
- **Affected outputs and claims:** Sparse-batch estimates and joint vector resampling outside this run.
- **Practical consequence:** Legitimate zero-game rows may abort, or absent cells may silently change replication support.
- **Smallest reasonable remediation:** Specify zero-exposure exclusion separately from missing rectangular support; enforce the chosen rule consistently and record excluded rows.
- **Focused regression/oracle test:** Adding a declared zero-exposure row must have a specified no-effect or failure result; missing strategy/batch cells must fail if rectangular support is required.
- **Full-run regeneration:** **No for this fast run.**

### P3. Production Numba behavior and identifier coercion lack strong release evidence

- **Severity / confidence / classification:** Low; Medium; missing validation/hardening.
- **Primary evidence:** `tests/conftest.py:74-113,268-284` globally replaces `numba.jit/njit` in pytest. Clean imports compile representative kernels, but no normal end-to-end test uses production JIT behavior. Several consumers coerce or drop malformed strategy IDs rather than enforcing one shared non-null physical type.
- **Affected outputs and claims:** Platform-specific compiled scoring/strategy behavior and cross-family joins on malformed inputs.
- **Practical consequence:** JIT-only failures or silent support loss can escape normal CI.
- **Smallest reasonable remediation:** Add a clean-subprocess compiled-kernel/small-simulation job and central canonical strategy-ID validation.
- **Focused regression/oracle test:** Hand-checkable compiled scoring plus tiny seeded simulation; nullable/nonnumeric/mixed-ID artifact boundary fixtures.
- **Full-run regeneration:** **Only if compiled/Python behavior or canonical IDs change.**

## 7. Statistical-method traceability

| Output | Estimand / target | Estimator or test | Uncertainty / decision | Main assumptions | Implementation | Independent validation status |
|---|---|---|---|---|---|---|
| Workload plan / Wilson fields | Per-strategy per-k binomial resolution | Worst-case full Wilson width, rounded to 100 batches | Wilson interval only; no selection decision | Planned exposure applies to completed outcome estimand | `simulation/workload_planner.py`; `performance.py` | Arithmetic/artifact fields match; abort denominator policy invalidates interpretation |
| All-player returns | Unconditional player-game score return | Sum(score)/sum(turns); mean(score/turns); separate rounds proxy | Descriptive sufficient statistics | Exact positive `n_turns`; explicit abort policy | `all_player_metrics.py` | Exact formulas and sufficient statistics matched; aborts also carry false wins |
| Per-k performance | Strategy win probability within fixed root/k tournament | Raw wins/exposures; delta from `1/k` | Batch rate SD/√B and t interval; Wilson separate | Deterministic batches approximate independent MC units; completed-game semantics | `performance.py:91-141` | Strategy-2 values match ≤1.11e-16; outcome/RNG assumptions fail |
| Across-k performance | Equal importance over configured k of chance-adjusted rates | Complete-support arithmetic mean | `sqrt(sum MCSE_k²)/3`, normal interval | Complete support; independent k streams | `performance.py:144-258` | Formula exact; 13 cross-k collision groups refute strict stream separation |
| Joint nonlinear diagnostics | Rank/top-N/shortlist/control stability | Resample whole batch vectors within k | Empirical descriptive distribution | Rectangular support; k resampling independence | `performance.py:261-397` | Implementation and stored selected values agree; discriminating unit test missing |
| Seat effects | Strategy/population seat rate versus `1/k` | Raw seat wins/exposures; common-seat equal-k mean | Descriptive | Recorded seat and valid winner | `seat_analysis.py` | Arithmetic spot checks pass; abort P1 wins contaminate effects |
| Game statistics | Game or strategy-exposure lengths/margins/events | Counts, sums, histograms, binned quantiles | Descriptive | Exactly one intended strategy observation per seat/game | `game_stats.py` | Population rows plausible; strategy observations proven `exposure+wins` |
| RNG diagnostics | Lag association in declared ordered sequence | Streaming Pearson lag correlation | Approximate diagnostic band only | Exact global semantic order | `rng_diagnostics.py` | Failed: canonical collector loses global/seat-merged order |
| TrueSkill ratings/contribution | Sequential root/k rating; cross-cell screening contribution | TrueSkill updates; within-cell mu percentile mean | Sigma is model state; diagnostics only | Ordered valid outcomes; complete six-cell support | `run_trueskill.py`; `trueskill_screening.py` | Percentile mean exact; cache/authenticity fails; calibration label mismatched |
| HGB | Held-out predictive association on finite grid | Whole-strategy folds; held-out MAE/R²; permutation importance | Fold/repeat variability | Finite noisy targets; no causal interpretation | `run_hgb.py` | Design visible and artifact support complete; no full independent retraining; cache/scope invalid |
| Two-root stability | Fixed-design reproducibility across roots 32/33 | Combine raw counts within k; discrepancies/ranks/bootstrap | MC diagnostics, no root-population inference | Two fixed RNG domains; valid root outcomes | `root_stability.py` | Raw-count combination exact; unadjusted significance labels violate layer boundary |
| Candidate family | Union of top performance and TrueSkill contributions | Protected union and optional balanced-tail contraction | No inference | Complete support and authentic source identities | `candidate_family.py` | 75+75, 74 overlap, 76 union and family hash coherent; sources contaminated/unauthenticated |
| H2H power | Power of implemented score rejection under configured scenarios | Exact joint-binomial probability | Bonferroni planning alpha | Independent completed Bernoulli games; valid global minimum search | `h2h_schedule.py` | Fast target/previous n exact; nonmonotone counterexample disproves minimum algorithm |
| H2H primary effect | `d_AB=.5(q_AB-q_BA)` over roots balanced within order | Constrained-null independent two-proportion score test | Holm zero-effect decisions; ordinary and Bonferroni score intervals | Exact planned root/order cells; completed independent games | `h2h_inference.py` | All 2,850 values/Holm decisions match exactly; six pairs have zero demonstrated completed games |
| Dominance/report | Partial order and permitted finite-grid claims | Typed practical/statistical edges, SCCs/fronts, direct dominance | Unresolved/equivalent pairs create no edge | Valid H2H decisions; selection conditioning explicit | `dominance.py`; `structure_reporting.py` | 535 unresolved retained; no unique-best claim; upstream validity fails |

## 8. Artifact-contract traceability

| Artifact family | Expected scope / owner | Observed identity and sidecar state | Lifecycle/source binding | Status |
|---|---|---|---|---|
| Simulation rows/checkpoints | Root/k simulation | Complete row/manifests and markers; no derived sidecar contract on final marker | Marker existence only; no code/config/output hashes | Failed |
| Ingest/curate by-k rows | `by_k` root stages | Adjacent self-hash sidecars; curated copies and row counts agree | Initial manifests do not content-hash every shard; ingest semantic checks incomplete | Partial |
| Combine partitions/union | by-k partitions then `concat_ks` row union | Main union is row-preserving; six nested partitions physically under `concat_ks` but sidecars say `by_k` | Source paths mostly un-hashed | Partial / scope failed |
| Per-k and across-k performance | `by_k`, `across_k` | Correct scopes and recalculable formulas | Sidecar self-hash valid; code unknown and upstream content weak | Numerically valid, provenance failed |
| Game/RNG diagnostics | `by_k`/`across_k`/`diagnostics` as declared | Twelve rare-event nested scope mismatches; RNG artifact stale under current stage hash | Source paths, not complete content graph | Failed/partial |
| Root TrueSkill | `by_k` | Sidecars accurately call sigma screening-only | Private weak stamps; mismatched bytes can be reauthenticated | Failed |
| HGB | by-k predictions; concat union; across-k summary; diagnostics/proposals | Held-out outputs present; two concat files physically in `across_k`; generic quantity labels wrong | mtime cache ignores HGB config/method | Failed |
| Cross-seed stability | `cross_seed` | Root and combined scopes/conditioning largely correct | Pair config lineage null/contradictory; unadjusted significance labels | Partial |
| Candidate family | `h2h_2p`, pair-owned | 76 members, family hash `07a2f716…41733` coherent in payloads | Sidecar does not independently bind all concrete source identities | Partial |
| Power/schedule | `h2h_2p`, pair-owned | 2,850 pairs, 11,400 cells, schedule hash `b0fd81bb…b3c09` coherent | Plan mixes immutable design and mutable cap state; config SHA null | Partial |
| H2H blocks/execution | `h2h_2p`, pair-owned | 11,400 adjacent sidecars; aggregate/block counts conserve | Missing block sidecar is fail-closed but not coordinate-resumable | Partial |
| Inference/dominance/agreement | `h2h_2p` | Family/count/decision fields internally coherent | Sidecars omit concrete multiplicity/family/schedule parameters; code unknown | Numerically valid, provenance failed |
| Reporting/migration | `diagnostics` | Conservative report language; migration says zero ignored | Migration scans only first-root `results_root`, not whole pair tree | Partial |
| Pair active configuration | Pair root | `config_sha:null`; private fields serialized; cannot reload | Root hash, context hash, and pipeline hash lineage not explicit | Failed |

## 9. Fast-run empirical recalculation, mismatches, and tolerances

### Conservation and invalid outcomes

| Root | k | Games | Aborts | Tied aborts | Abort wins credited to P1 | Within-cell duplicate game-seed excess |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2 | 172,000 | 1,479 | 1,479 | 1,479 | 2 |
| 32 | 4 | 86,000 | 3 | 3 | 3 | 2 |
| 32 | 5 | 68,800 | 0 | 0 | 0 | 0 |
| 33 | 2 | 172,000 | 1,498 | 1,498 | 1,498 | 5 |
| 33 | 4 | 86,000 | 1 | 1 | 1 | 1 |
| 33 | 5 | 68,800 | 0 | 0 | 0 | 2 |

Every root/k cell has 4,300 shuffles, 80 strategies, 100 deterministic batches, and 4,300 exposures per strategy. Total tournament games are 653,600; non-aborted games are 650,619; recorded wins are nevertheless 653,600. Integer conservation checks used tolerance 0.

### Independent strategy-2 calculation

| Root | k | Wins/exposures | Win rate | Chance delta | Batch MCSE | Wilson 95% interval | Maximum artifact error |
|---:|---:|---:|---:|---:|---:|---|---:|
| 32 | 2 | 2502/4300 | 0.581860465116279 | 0.081860465116279 | 0.007027300964633 | [0.567050865753240, 0.596523932885580] | 1.11e-16 |
| 32 | 4 | 1307/4300 | 0.303953488372093 | 0.053953488372093 | 0.006496298947627 | [0.290385589230508, 0.317871356074818] | 0 |
| 32 | 5 | 998/4300 | 0.232093023255814 | 0.032093023255814 | 0.006349365943628 | [0.219717261179214, 0.244947034184648] | 2.78e-17 |
| 33 | 2 | 2556/4300 | 0.594418604651163 | 0.094418604651163 | 0.007386155148018 | [0.579664930434886, 0.609003729358753] | 1.11e-16 |
| 33 | 4 | 1349/4300 | 0.313720930232558 | 0.063720930232558 | 0.006680705345839 | [0.300023682668367, 0.327750710201818] | 0 |
| 33 | 5 | 989/4300 | 0.230000000000000 | 0.030000000000000 | 0.006279417768482 | [0.217665963268021, 0.242816021903371] | 0 |

Root-32 equal-k score/MCSE are `0.05596899224806202` and `0.003828247448616921`; root-33 values are `0.06271317829457365` and `0.003924541010666778`. Both match artifacts exactly. Combining roots from raw counts gives k deltas `0.08813953488372095`, `0.058837209302325555`, `0.031046511627906975`, and equal-k `0.059341085271317824`, exact to stored precision.

Independent within-cell mu percentile ranks for strategy 2 are `[.4625,.5125,.5,.6125,.7375,.575]`; their six-cell mean `0.5666666666666668` matches the pair contribution (`0.566667` displayed) and complete-support count 6/6.

### H2H recalculation

- Schedule/count inventory: 76 candidates, 2,850 unordered pairs, 11,400 root/order blocks, 1,974 games/block, and 22,503,600 nominal games. Identities and integer totals matched exactly.
- From primitive inference columns, all 2,850 `q_ab`, `q_ba`, `d_ab`, constrained-null p-values, Holm adjusted p-values, and reject flags matched with maximum absolute error 0 and zero decision mismatches.
- Pair 1063 (22 vs 66) independently gives `q_ab=2174/3948=0.5506585612968592`, `q_ba=1992/3948=0.5045592705167173`, `d_ab=0.023049645390070927`, `z=4.102616374896638`, `p=4.085042673815476e-05`, Holm adjusted p `0.021854978304912797`, simultaneous d interval `[-0.00219397790262554, 0.04818753143624031]`, and `unresolved`.
- Decision counts are 832 practical-A, 877 practical-B, 348 statistical-only-A, 258 statistical-only-B, and 535 unresolved. Equivalence is disabled.
- Fast power is `0.8003606088973029` at 1,974/root/order and `0.7996629814552753` at 1,973. This verifies the current allocation meets the configured target, not the global-minimum algorithm.

### Artifact mismatches

- 11,586/11,586 sidecars have matching artifact self-hashes and `code_revision: unknown`.
- 20 physical-scope/sidecar-scope mismatches were found.
- Pair `active_config.done.json` has `config_sha:null`; `pipeline_health.json` has `1909cc…`; pair sidecars have `134bbb…`.
- Current expected versus recorded RNG stage hashes are root 32 `bae762…` vs `e304a8…` and root 33 `287add…` vs `e1e5cf…`; therefore top-level `complete_success` is not a current canonical-state summary.
- No tolerance was allowed for counts, IDs, hashes, membership, or decisions. Floating-point formula checks used observed machine-level agreement (maximum 1.11e-16), well inside a review tolerance of 1e-12.

## 10. Test-suite adequacy by the defects it could permit

The current non-terminology suite passed all 830 collected tests in 92.8 seconds. That is useful regression coverage, but it permits the following defect classes:

| Defect tests permit | Why current tests permit it | Required discriminating test |
|---|---|---|
| Safety-cap abort becomes a winner | Engine test checks 200 rounds/turns only (`test_engine.py:331-347`) | End-to-end no-winner/abort-denominator oracle |
| Existence-only simulation reuse | Runner test explicitly treats marker existence as completion (`test_runner_branches.py:78-105`) | Change config/output bytes and require stale/recovery |
| 32-bit stream collisions | Small uniqueness fixtures do not approach birthday-collision scale | Production-scale semantic-coordinate uniqueness and deliberate collision fixture |
| TrueSkill/HGB stale reuse and corruption blessing | Completed-cell branch/hyperparameter changes are not tested; HGB “fresh” fixture reinforces mtime behavior | Change source/parameter/bytes and require recomputation; unchanged skip |
| Metric-only resume loses sums | Resume tests retain replayable rows/chunks or do not compare every sufficient statistic | Interrupted metric-only logical equality oracle |
| Winner duplication in game stats | Fixtures do not include/assert exclusion of top-level `winner_strategy` | Exposure-count identity across per-k/rare-event outputs |
| Wrong scope/schema/source validates | Tests check self-hash and caller-supplied expected values, not canonical path/schema graph | Wrong-directory, absent-column, dtype, nullability, changed-source tests |
| Wrong nonlinear resampling dependence | Test mainly checks repeatability | Correlated batch-vector oracle that fails independent-per-strategy resampling |
| Wrong t/Wilson/across-k interval family | Few exact endpoint/df oracles | Three-batch hand t interval plus distinct Wilson/normal endpoints |
| Root diagnostics change method | Mostly shapes/bounds/determinism | Hand discrepancy/bootstrap/convergence identities and no significance labels |
| TrueSkill raw-mu averaging | Existing fixture does not force cell-specific scales | Affine-shifted cell fixture with hand percentile mean |
| HGB leakage | Tests do not independently record train/held-out strategy sets | Spy estimator asserting disjoint whole-strategy folds and retained fold variation |
| H2H compensated root/order imbalance | Only pooled order imbalance is rejected | Exact block-manifest join and compensated-imbalance rejection |
| Power nonmonotonicity | Only returned n and n-1 are checked | Brute-force first crossing over small admitted domains |
| Numba-only divergence | Pytest disables JIT globally | Clean-subprocess compiled scoring and tiny simulation |
| Cross-stage semantic break | Toy oracle manufactures upstream artifacts and H2H outcomes | Tiny actual two-root CLI workflow from simulation through report |

The full terminology gate is non-hermetic: `scripts/check_terminology.py` fails on accurate quotations in the untracked specialist reports. Coverage is configured at 90% but is not enforced by the documented plain-pytest release command; the specialist coverage run measured 86.09%. These are gate defects, not evidence that higher line coverage alone would catch the statistical defects.

## 11. Claim-language boundaries

- **Screening versus inference:** Final reporting correctly calls tournament leaders descriptive, treats TrueSkill as screening contribution, HGB as predictive association, and H2H as selection-conditioned. Root-stability `statistically_*` labels violate this otherwise-good separation and should be removed.
- **Significance versus practical dominance:** H2H correctly separates Holm zero-effect rejection from Bonferroni simultaneous practical bounds and typed graph edges. A statistical-only edge is not called practical dominance.
- **Nonsignificance versus equivalence:** The run correctly retains 535 unresolved comparisons and calls none equivalent because no equivalence margin is configured. Equivalence code is structurally gated by explicit-margin simultaneous containment. Abort-only pairs show why completed support must also be required before equivalence is ever enabled.
- **Reproducibility versus population inference:** Final root language is fixed-design reproducibility and makes no seed-population t/random-effects claim. Root-stability significance labels are inconsistent with that boundary even if interpreted as MC-only.
- **Association versus causation:** HGB outputs explicitly state `predictive_association_not_causal`, and whole strategy configurations are held out. Sidecar quantity/scope errors and stale caches weaken provenance, but no causal claim was found.
- **Finite-grid versus universal superiority:** Final report begins with finite-grid conditionality and makes no universal strategy claim. With k={2,4,5}, H2H is correctly labelled an external two-player finalist diagnostic and no primary multi-k unique-best claim is permitted.

## 12. Minimal prioritized remediation plan by root cause

### Release-gating correctness fixes

1. **Outcome semantics:** Represent completed/tied/aborted termination explicitly; remove rank/winner fabrication; define tournament and H2H denominators; deterministically replace attempts if power targets completed games.
2. **RNG identity:** Derive player streams from full semantic coordinates and increment the RNG scheme.
3. **Authenticated lifecycle/provenance:** Unify simulation/root/pair lifecycle; bind code, method, scoped config, upstream content, output sidecars, family/schedule identities; make pair config reloadable; forbid corruption re-authentication.
4. **Resume ordering:** Commit sufficient statistics before coordinate ownership in metric-only mode.
5. **Game-stat column selection:** Restrict to canonical seat columns and regenerate affected summaries.
6. **Artifact semantics:** Enforce physical scope, Arrow schema, typed method parameters, and source hashes; correct HGB/rare-event/combine placements.
7. **CLI strictness:** Reject unknown arguments and support the documented seed override position.
8. **End-to-end oracle:** Add a tiny actual workflow and designated fast-run audit before rerunning fast configuration.

### Next statistical/operational fixes

9. Replace nonmonotone exact-power binary search; validate each H2H cell against the immutable manifest; separate immutable design from mutable cap state; make invalid block sidecars coordinate-resumable.
10. Remove root-stability significance labels, repair RNG diagnostic ordering, and either implement or rename TrueSkill predictive calibration.

### Optional cleanup and hardening

11. Clarify zero-exposure/missing-cell policy, strengthen ingest invariants, add compiled-Numba CI, standardize strategy-ID physical type, correct the roll-limit off-by-one, label binned margin quantiles, fix seat sidecar conditioning, scope stage-definition lookup, honor documented worker owners, expand migration inventory, and update CLI/RNG documentation and hermetic release gates.

## 13. Recommended focused tests for each remediation

| Remediation | Minimum focused test set |
|---|---|
| Outcome semantics | Engine abort; tournament accumulator; mixed abort/completion denominator; H2H zero-completed unresolved; report/equivalence guard |
| Full-coordinate RNG | Production-scale uniqueness; cross-k/cross-batch deliberate 32-bit collision; worker-count and resume logical identity |
| Lifecycle/provenance | Config/input/code/method/output mutation matrix; missing vs mismatched sidecar; `--force`; pair config/hash round trip |
| Metric resume | Interrupted versus uninterrupted equality with no row/chunk manifests, including every sum and square sum |
| Game stats | Anchored seat-column selection; per-strategy observations equal exposures; rare-event and across-k conservation |
| Artifact semantics | Wrong scope path, schema field/type/nullability, stale source hash, wrong family/schedule/multiplicity parameters |
| CLI | Installed console invocation in documented order; arbitrary unknown argument fails before write |
| End-to-end | Tiny actual two-root simulation-to-report fixture with hand counts/hashes/lifecycle/claims |
| H2H planning/validation | Brute-force global first crossing; compensated imbalance; immutable cap resume; interrupted block-sidecar recovery |
| Diagnostic claims | No root significance labels; globally ordered lag oracle; model-consistent or explicitly heuristic TrueSkill probabilities |

## 14. Data-regeneration implications

| Correctness fix | Minimum regeneration if applied alone | Implication for this fast run |
|---|---|---|
| Safety-cap outcome semantics | Simulation onward, including all H2H | Full regeneration mandatory |
| RNG scheme/full coordinates | Simulation onward | Full regeneration mandatory |
| Code/config/source-bound lifecycle | Fresh run required for auditable release evidence | Full regeneration mandatory; old bytes cannot be retroactively attributed |
| Metric-only resume | Only affected interrupted metric-only runs | No independent fast-run regeneration, but superseded by full rerun |
| Game-stat winner duplication | Game-stat stage and consumers | Regenerate affected diagnostics; superseded by full rerun |
| Artifact scope/schema/source contract | Relocate/regenerate affected artifacts and sidecars | Republish/regenerate at least 20 mismatched artifacts; superseded by full rerun |
| CLI parsing | None for known-correct root invocation | Audit runs invoked with post-subcommand overrides |
| H2H power minimum | Plan/schedule/H2H only when chosen allocation changes | Current allocation can remain for this defect alone |
| H2H cell validation | Invalid aggregate/inference only | Current exact 1,974-cell support needs no change for this defect alone |
| Root/RNG/TrueSkill diagnostic fixes | Respective diagnostic stages and reports | Regenerate those stages; RNG-scheme rerun already supersedes them |

The practical release action is one clean fast run after fixes 1-8, not piecemeal repair of this tree. Only after that oracle passes should production-scale simulation begin.

## 15. Limitations to document even after correctness fixes

- Results are conditional on the finite configured 80-strategy grid, target score, k support, seating/randomization design, and safety-cap policy; they do not establish universal strategy superiority.
- Equal-k performance gives each configured k equal importance. It is not an exposure-weighted population claim and changes if the supported k set or declared weights change.
- Batch MCSE and analytic across-k intervals quantify Monte Carlo error under the declared deterministic-batch/RNG design. They are not post-selection confidence statements about a strategy population.
- Joint resampling is a descriptive stability analysis over simulation batches, not a replacement for formal finalist inference.
- TrueSkill is sequential and scale-local to each root/k cell; sigma is model state, not a sampling standard error. Percentile contribution is descriptive.
- HGB estimates held-out predictive association on a small, noisy finite grid; it does not identify causal feature effects and currently does not propagate target-rate measurement error.
- Two roots provide fixed-design reproducibility only. They do not identify a distribution of roots, between-root heterogeneity, or population-level seed uncertainty.
- H2H inference is conditioned on the frozen candidate family, the two-player game, the configured first-seat scenarios, multiplicity family, practical margin, and completed-game termination policy.
- Holm rejection establishes a zero-effect decision, not practical dominance. Nonsignificance is not equivalence. Equivalence requires a predeclared margin and simultaneous interval containment.
- Cycles, incomparability, and unresolved comparisons are valid outcomes. Within-front display order and round-robin summaries are noninferential.
- Exact power is exact for the implemented discrete test and configured scenarios, not robust to unmodeled dependence, abort mechanisms, or scenario misspecification.
- The fast configuration is an integration oracle, not production evidence about rank stability, runtime, storage behavior, or final precision.

## 16. Final disposition

**suitable only after specified fixes**

Production runs should remain blocked until B1, B2, H1-H6, and the artifact-affecting parts of M1-M3 are addressed, focused oracles pass, and a clean `fast_config.yaml` run produced by the reviewed revision validates end to end. The remaining Medium/Low diagnostic and documentation items may be sequenced immediately afterward where they do not alter the production estimand, but their limitations must be explicit.
