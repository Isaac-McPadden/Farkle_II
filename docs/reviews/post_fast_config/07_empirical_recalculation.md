# 07 — Empirical recalculation of the completed fast run

## Verdict

**Questionable within this review's scope.** The selected artifacts are numerically sound on every independently recalculated tournament, root-combination, TrueSkill, HGB-fold, and H2H quantity below. I found no statistical arithmetic defect in those results. The qualification is material: the artifacts cannot be bound to the checked-out commit, the persisted pair configuration is not reloadable and loses the global configuration SHA, and the current lifecycle resolver calls both RNG-diagnostics stages `complete_stale` while the top-level health file calls the run `complete_success`. Thus the artifacts are a coherent audit dataset, but they are not sufficient post-remediation evidence that commit `6be5f5f` produced one wholly `complete_valid` run.

The fast sample is not used here to assess strategy superiority, rank stability, production precision, or power beyond checking the implemented identities and planning arithmetic.

## Scope and independent method

This was review-only. I did not call the production aggregation or inference functions being checked. Temporary calculations were piped directly to the repository Python 3.12 venv and did not create repository files. I read curated game rows and immutable H2H block files, formed counts and estimates independently with pandas/NumPy/SciPy/statsmodels primitives, and compared the results to the published Parquet/JSON artifacts. Exact deterministic results were compared at zero or machine-epsilon tolerance.

Relevant implementation locations are:

- per-k estimation, Wilson checks, deterministic-batch MCSE, equal-k propagation, Pareto/maximin, and joint resampling: `src/farkle/analysis/performance.py:91-141`, `144-258`, and `261-397`;
- all-player return inputs and formulas: `src/farkle/analysis/all_player_metrics.py:149-190` and `202-232`;
- seat effects and common-support standardization: `src/farkle/analysis/seat_analysis.py:237-365`;
- coordinate-owned PCG64DXSM streams: `src/farkle/utils/random.py:18-40` and `63-121`;
- within-cell TrueSkill percentiles: `src/farkle/analysis/trueskill_screening.py:86-145`;
- held-out HGB fold construction: `src/farkle/analysis/run_hgb.py:277-350`;
- raw-count root combination and fixed-design diagnostics: `src/farkle/analysis/root_stability.py:126-271` and `305-370`;
- H2H score test, score inversion, Holm adjustment, and decisions: `src/farkle/analysis/h2h_inference.py:62-133`, `136-289`, and `417-530`;
- H2H scenario mapping and exact power: `src/farkle/analysis/h2h_schedule.py:69-90`, `98-226`, and `229-344`.

Focused current-code tests passed: 57 tests across performance, all-player metrics, seats, TrueSkill screening, HGB, root stability, H2H scheduling/inference, and dominance.

## Run identification and freshness

The selected run is `data/results_seed_pair_32_33`, not the neighboring `data/results_seed_pair_30_31` or standalone `data/results_seed_30` trees.

| Item | Observed value |
|---|---|
| Requested config | `configs/fast_config.yaml` |
| Selected persisted config | `data/results_seed_pair_32_33/active_config.yaml` |
| Base config SHA | `1909cc1f0539f10bbfb766d9f50cb216bd07514d7580d92e6f1e064c46e48fcc` |
| RNG roots | 32 and 33 |
| Player counts | 2, 4, 5 |
| Strategy grid | 80 unique IDs, 0–79, identical manifests in both roots |
| Canonical root artifact families | `by_k/`, `concat_ks/`, `across_k/`, `diagnostics/` |
| Pair artifact families | `cross_seed/`, `across_k/`, `h2h_2p/`, `diagnostics/` |
| Pair analysis root | `data/results_seed_pair_32_33/seed_pair_analysis` |
| Pipeline health | `complete_success`; both root simulation/analysis entries and the pair workflow say `complete` |
| H2H execution state | `complete_valid`, 11,400/11,400 blocks |
| Checked-out revision | `6be5f5fa11df77155621bfc81188c7515f38f8de` (`6be5f5f`) |

`fast_config.yaml` has `analysis.n_jobs: 4`, while the resolved per-root active configs have 12. That is expected orchestration policy, not unexplained drift: roots run sequentially (`orchestration.parallel_seeds: false`), and `_derive_per_seed_job_budgets` assigns the 12 simulation workers to analysis for each root (`src/farkle/orchestration/two_seed_pipeline.py:86-112`, `125-140`). Other statistical configuration agrees.

The adjacent `results_seed_pair_30_31` is older (last modified 2026-07-17), names roots 30/31, has config SHA `3ef8…`, and still points its pair workflow at the retired generic `analysis` directory. The selected tree is newer (pair completion 2026-07-18 15:23 MDT), names roots 32/33, has config SHA `1909…`, and uses canonical `seed_pair_analysis`. Those coordinates, config hashes, stage hashes, family hash `07a2…`, and schedule hash `b0fd…` distinguish the selected run; timestamps alone were not used.

No selected-tree filename resurrects `pooled`, `long`, `ratings_k_weighted`, `ratings_pooled`, or `players=0` output. The concatenated artifacts are row-preserving: each root's per-k row counts `[172000, 86000, 68800]` sum to 326,800, exactly the row count of `02_combine/concat_ks/all_ingested_rows.parquet`.

## Findings

### F1 — The completed artifacts are not revision-bound to the checked-out commit

- **Severity:** High
- **Confidence:** High
- **Evidence:** Every one of the 11,586 sidecars beneath the selected tree records `code_revision: "unknown"`. The sidecar factory defaults the field to that value (`src/farkle/utils/artifact_contract.py:286-341`). The newest top-level run files are timestamped 2026-07-18 15:23 MDT, while checked-out commit `6be5f5f` is dated 2026-07-18 23:58 MDT and changes ingest, curate, game statistics, H2H scheduling, TrueSkill orchestration, sidecars, and release audit. The sidecars' artifact SHA-256 values all match their files, but none proves which source revision produced those bytes.
- **Consequence:** Exact agreement with these artifacts verifies the artifacts' arithmetic, not that current commit `6be5f5f` generates them. This directly weakens a post-remediation acceptance claim.
- **Smallest reasonable remediation:** Automatically bind the exact Git revision (and a dirty-state indicator) into every derived sidecar and completion stamp, reject `unknown` for release evidence, then rerun `fast_config.yaml` from `6be5f5f` or a later reviewed commit.

### F2 — Pair-level configuration provenance is incomplete and the persisted config cannot be reloaded

- **Severity:** Medium
- **Confidence:** High
- **Evidence:** `data/results_seed_pair_32_33/active_config.done.json` has `config_sha: null`; every pair-stage completion stamp examined also has `config_sha: null`. In current construction, `pair_base = replace(first.config, sim=pair_sim)` drops the init-false `config_sha`, after which `RunContextConfig.from_base` preserves the resulting `None` (`src/farkle/orchestration/run_contexts.py:47-85` and `137-160`). Separately, `effective_config_dict` serializes all subclass dataclass fields and removes only `config_sha` and `_stage_layout` (`src/farkle/config.py:1605-1611`). Consequently the pair `active_config.yaml` begins with `_analysis_root_override` and `_root_input_layout_override`; `load_app_config` rejects it with `ValueError: Unknown top-level config section(s): '_analysis_root_override', '_root_input_layout_override'`. The writer publishes exactly that mapping (`src/farkle/orchestration/seed_utils.py:79-94`).
- **Consequence:** A reviewer cannot replay the pair workflow from its advertised active config, and the pair stamps do not carry the base config identity `1909…`. Pair sidecars do consistently carry `134bbb…`, the exact effective pair-context hash, so this is not an artifact-hash mismatch; it is a lineage/replay defect.
- **Smallest reasonable remediation:** Preserve the base `config_sha` across the pair `replace`, and serialize only public `AppConfig` fields into `active_config.yaml`; keep path/layout overrides in a separate run-context manifest. Add a load-round-trip test for pair active configs.

### F3 — Top-level completion disagrees with the current canonical lifecycle

- **Severity:** Medium
- **Confidence:** High
- **Evidence:** `pipeline_health.json` says `complete_success`. Reconstructing the current two-root worker policy from `fast_config.yaml` and passing each root stamp to `resolve_stage_state` makes ingest, curate, combine, metrics, game stats, and screening `complete_valid`, and all nine pair stages resolve `complete_valid`. Both RNG-diagnostics stamps resolve `complete_stale`: root 32 recorded stage SHA `e304a8…` versus current `bae762…`; root 33 recorded `e1e5cf…` versus current `287add…`. The resolver explicitly returns `complete_stale` on a stage-SHA mismatch (`src/farkle/utils/stage_completion.py:203-251`).
- **Consequence:** The selected run is not wholly `complete_valid` under the current commit even though its summary says complete. The stale cells are diagnostics rather than inputs to the tournament/H2H arithmetic audited below, but the whole-run completion claim is wrong.
- **Smallest reasonable remediation:** Regenerate the two RNG-diagnostics stages under the reviewed revision and republish pipeline health from canonical lifecycle states. Health generation should not preserve `complete_success` when any enabled stage resolves stale.

### F4 — Strategy identifier types vary across canonical artifact families

- **Severity:** Low
- **Confidence:** High
- **Evidence:** The strategy manifest and performance artifacts use integer IDs, while TrueSkill contribution and H2H artifacts store the same IDs as strings; `_load_rating_frame` explicitly casts to string (`src/farkle/analysis/trueskill_screening.py:92-99`). All 80 IDs converted one-to-one in this run, the 76-member frozen family was complete, and no audited merge introduced missing values.
- **Consequence:** There is no observed numerical error, but future consumers can silently miss joins unless they repeat the same coercion.
- **Smallest reasonable remediation:** Standardize canonical strategy IDs to one declared physical type, preferably integer, or make the conversion an explicit schema/sidecar contract validated at every cross-family boundary.

## Raw counts and conservation

I expanded every player seat from each curated `01_curate/by_k/<k>p/game_rows.parquet` and counted games, winners, exposures, seats, strategies, roots, and deterministic batches without reading the production metric outputs.

| Root | k | Games | Games/batch | Wins | Player exposures | Exposures/batch for strategy 18 | Strategy 18 wins/exposures |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2 | 172,000 | 1,720 | 172,000 | 344,000 | 43 | 2,700 / 4,300 |
| 32 | 4 | 86,000 | 860 | 86,000 | 344,000 | 43 | 1,366 / 4,300 |
| 32 | 5 | 68,800 | 688 | 68,800 | 344,000 | 43 | 1,134 / 4,300 |
| 33 | 2 | 172,000 | 1,720 | 172,000 | 344,000 | 43 | 2,605 / 4,300 |
| 33 | 4 | 86,000 | 860 | 86,000 | 344,000 | 43 | 1,393 / 4,300 |
| 33 | 5 | 68,800 | 688 | 68,800 | 344,000 | 43 | 1,075 / 4,300 |

For all six cells:

- `wins = games`, `exposures = k × games`, and each seat has exactly `games` population exposures;
- there are exactly 100 batches and 80 strategies; every strategy has 43 exposures in every batch;
- no duplicate `(root_seed, k, shuffle_index, game_index)` coordinate exists, including across cells;
- every `winner_strategy` equals the strategy in `winner_seat`;
- no player record has a missing strategy/score or nonpositive `n_turns`/`n_rounds`;
- both strategy manifests are byte-logically identical, with no duplicate strategy ID or string;
- all six 8,000-row all-player batch artifacts have unique `(root,k,batch,strategy)` keys and positive exposures;
- all six 80-row performance artifacts have unique strategy keys and no missing required estimate.

The selected audit strategy is ID 18, `Strat(250,4)[SD][FOFS][OR][HR]`.

## Per-k performance, Wilson interval, and batch uncertainty

For root 32, strategy 18:

| k | Wins | Exposures | Win rate | Chance `1/k` | Chance delta | Batch MCSE |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2,700 | 4,300 | 0.627906976744 | 0.5 | 0.127906976744 | 0.007241850643 |
| 4 | 1,366 | 4,300 | 0.317674418605 | **0.25** | 0.067674418605 | 0.007893146578 |
| 5 | 1,134 | 4,300 | 0.263720930233 | 0.2 | 0.063720930233 | 0.007318540117 |

The four-player baseline is therefore correctly 0.25, not a pooled or fixed 0.5 baseline.

For the root-32/k=4 Wilson check, `x=1366`, `n=4300`, `z=1.95996398454`. Direct Wilson arithmetic gives `[0.303926851782, 0.331747460340]`, full width `0.027820608558`; all values agree with `03_metrics/by_k/4p/performance.parquet` within `2.3e-16`.

The 100 root-32/k=4 batch win counts for strategy 18, each divided by 43 to obtain its batch rate, are:

```text
12,14,14,16,15,21,14,11,17,13,13,14,12,4,12,16,18,11,16,10,
16,11,19,18,5,10,12,13,13,9,11,13,12,11,18,16,12,11,17,14,
13,13,16,10,13,12,16,8,17,9,14,15,16,9,12,15,17,14,15,17,
16,8,14,15,15,16,21,14,13,16,12,10,14,13,11,15,13,14,13,13,
14,19,12,11,12,12,17,13,6,19,6,12,10,21,20,12,15,16,17,21
```

Their mean is `1366/(100×43) = 0.317674418605`; sample SD is `0.078931465777`; `MCSE = s/sqrt(100) = 0.007893146578`; `t_(0.975,99) = 1.984216951509`; and the interval is `[0.302012703364, 0.333336133845]`. It matches the production artifact exactly. Across every strategy/root/k, wins, exposures, rates, deltas, Wilson bounds/widths, MCSEs, and t intervals matched with maximum absolute error `2.22e-16`.

Zero-game rows were absent. No estimate was formed with zero support.

## Equal-k result, propagated MCSE, Pareto/maximin, and player-count diagnostics

For root 32, strategy 18, the complete-support delta vector is

```text
[0.127906976744, 0.067674418605, 0.063720930233]
```

The equal-k score is its unweighted mean, `0.086434108527`. The independent-k MCSE is

```text
sqrt(0.007241850643² + 0.007893146578² + 0.007318540117²) / 3
= 0.004324442072.
```

With normal critical value `1.95996398454`, the analytic interval is `[0.077958357814, 0.094909859241]`. All four values match `03_metrics/across_k/performance_equal_k.parquet` within `2.8e-17`. The independent-k assumption is justified for Monte Carlo error by the RNG design: `k` is an explicit SeedSequence entropy coordinate (`src/farkle/utils/random.py:63-93`), so different k cells use disjoint deterministic PCG64DXSM streams. This is a design assumption about RNG domains, not an empirical estimate of heterogeneity.

The independent Pareto calculation found root-32 frontier IDs `[11, 15, 17, 35, 37, 55]`, exactly the published set. Strategy 55 has delta vector `[0.156046511628, 0.112325581395, 0.105813953488]`; no other vector weakly exceeds it in all k with one strict improvement, so it is Pareto. Its minimum `0.105813953488` is also the largest per-strategy minimum, making ID 55 the unique root-32 maximin leader, again exactly as published. Strategy 18 is correctly neither Pareto nor maximin.

For root 32, strategy 18, finite chance-relative log odds are `[0.523248143765, 0.334132370096, 0.359576604967]` for k `[2,4,5]`; the k=2 minus k=4 contrast is `0.189115773669`. For the 75 strategies finite in both k=2 and k=4, independent rank inputs give Spearman `0.962763271361` and Kendall `0.852654443469`, matching the relevant player-count diagnostic. Structural nulls in the long diagnostic schema are expected for fields not applicable to a row type; no required estimate was lost in a merge.

## Joint batch-vector resampling

I independently recreated the version-1 coordinate entropy `[scheme, purpose=400, root, k, ..., replicate]`, initialized PCG64DXSM, and resampled 100 batch rows with replacement for each root/k/replicate. One selected index vector was applied to the full `100×80` wins matrix and the full `100×80` exposures matrix. This is whole-vector resampling, not a separate draw per strategy.

For root 32 replicate 0, the first 12 selected batch indices were:

| k | First 12 indices |
|---:|---|
| 2 | 20, 38, 73, 67, 24, 94, 54, 78, 29, 14, 49, 13 |
| 4 | 83, 42, 97, 53, 1, 72, 35, 12, 65, 83, 37, 36 |
| 5 | 69, 31, 16, 45, 41, 61, 49, 87, 45, 45, 11, 44 |

Using those same vectors across strategies gave replicate-0 equal-k scores `0.093410852713` for ID 18 and `0.071627906977` for ID 19; ID 18 ranked 20. Across all 2,000 replicates, ID 18 had mean rank `22.8115`, rank SD `1.934158150204`, top-75 inclusion `1.0`, and shortlist inclusion `0.015`. Every published bootstrap rank mean, rank SD, top-75 probability, and shortlist probability for all 80 strategies matched exactly. `controls: []`, so the zero-row control-contrast artifact is correct and no control contrast exists to recalculate.

## Seat effects and returns

For root 32/k=4, strategy 18:

| Seat | Wins | Exposures | Win rate | Effect vs 0.25 |
|---:|---:|---:|---:|---:|
| 1 | 350 | 1,072 | 0.326492537313 | 0.076492537313 |
| 2 | 359 | 1,100 | 0.326363636364 | 0.076363636364 |
| 3 | 338 | 1,041 | 0.324687800192 | 0.074687800192 |
| 4 | 319 | 1,087 | 0.293468261270 | 0.043468261270 |

Population seat counts for the same cell are `(23064,22321,20943,19672)` wins, each over 86,000 exposures; the effects are `(0.018186046512, 0.009546511628, -0.006476744186, -0.021255813953)`. Strategy and population results match the two seat artifacts exactly.

Common-support seats are 1 and 2 because `min(k)=2`. For strategy 18, seat 1's within-k effects are `[0.158845612389, 0.076492537313, 0.101219512195]`, whose equal-k mean is `0.112185887299`; the published standardized result is identical. Population seat 1 similarly standardizes to `0.020748062016`.

For root 32/k=4, strategy 18, direct player-row inputs were 4,300 exposures, `sum(final_score)=35,283,550`, and `sum(n_turns)=72,946`. Therefore:

- exact turn-weighted return: `35,283,550 / 72,946 = 483.694102486771`;
- exact game-weighted mean: `mean(final_score/n_turns) = 497.985742879358`;
- rounds proxy: `mean(final_score/n_rounds) = 506.938454317150`;
- proxy discrepancy: `506.938454317150 - 497.985742879358 = 8.952711437792`;
- turn/round mismatches: 1,399 of 4,300 exposures.

These agree with the summed raw fields in `all_player_batch_metrics.parquet` to at most `2.3e-13`. This also confirms that the exact measure uses `n_turns`; the rounds value remains a separately reported proxy.

## TrueSkill and HGB wiring

For strategy 18, independent average-tie percentile ranks from the six within-root/k `mu` vectors were:

```text
root 32: [0.9125, 1.0000, 0.9750]
root 33: [0.5625, 0.5625, 0.9500]
```

The six-cell equal mean is `0.827083333333`, minimum `0.5625`, with 6/6 complete cells and contribution rank 7. It exactly matches `seed_pair_analysis/01_trueskill/across_k/candidate_percentile_contribution.parquet`. Raw `mu` was not aggregated across cells, and `sigma` was not used as a sampling error.

For HGB root 32/k=4, independent reconstruction of the purpose-600 permutation assigned fold 0 held-out IDs `[10,2,23,25,27,31,38,45,50,56,57,63,68,69,73,75]`. The other 64 IDs form the disjoint training complement; intersection is zero and union is all 80 strategies. All five folds contain 16 distinct held-out strategies, every strategy appears in exactly one held-out row, and all reconstructed fold assignments match `heldout_predictive_scores_4p.parquet`. Root 33/k=4 has the same 16/64 and exactly-once properties. This confirms held-out strategy-configuration separation; I did not treat the predictive associations as causal or require model retraining equality.

## Two-root combination and reproducibility diagnostics

For strategy 18, raw-count combination is:

| k | Root 32 wins/exposures | Root 33 wins/exposures | Combined wins/exposures | Combined delta |
|---:|---:|---:|---:|---:|
| 2 | 2700/4300 | 2605/4300 | 5305/8600 | 0.116860465116 |
| 4 | 1366/4300 | 1393/4300 | 2759/8600 | 0.070813953488 |
| 5 | 1134/4300 | 1075/4300 | 2209/8600 | 0.056860465116 |

These match each `performance_root_combination_<k>p.parquet` row exactly. No root estimate was averaged in place of raw-count combination.

The independently calculated root-32 and root-33 equal-k scores for ID 18 are `0.086434108527` and `0.076589147287`, difference `0.009844961240`. Their independently calculated MCSEs are `0.004324442072` and `0.004112292786`, so the expected discrepancy MCSE is `sqrt(mcse32²+mcse33²)=0.005967558227`, and the standardized discrepancy is `1.649746993074`; all match `root_discrepancies.parquet`. Stable descending-score/integer-ID ranks are 22 and 26, movement 4. Across all 80 strategies, Spearman is `0.982208157525`, Kendall is `0.896835443038`, median movement is 2, 95th percentile movement is 10.05, and maximum movement is 13, exactly matching `root_rank_stability.parquet`. These remain fixed-design reproducibility diagnostics, not seed-random-effects inference.

## H2H blocks, inference, decisions, and power

### Block/order conservation

Reading the 11,400 immutable block Parquets directly found:

- 2,850 unordered pairs, two roots, and two orders, exactly four blocks per pair;
- 1,974 games per block and 22,503,600 games total;
- no duplicate `(pair_id, root_seed, order)` or `block_id`;
- every block complete, and `wins_seat1 + wins_seat2 = games_completed` in every block;
- order 0 always maps `A→seat1, B→seat2`; order 1 always maps `B→seat1, A→seat2`;
- one family hash, one schedule hash, RNG scheme 1, and H2H game namespace 202.

Independent aggregation of the block rows matched all 11,400 `root_order_counts.parquet` rows exactly. Combining roots within order produced 3,948 games per order for every pair. Independent score calculations over all 2,850 pairs had zero error for `q_AB`, `q_BA`, `d_AB`, pooled-null rate, z, unadjusted p, Holm position, adjusted p, and rejection. Required H2H numeric fields were finite, and every ordinary/simultaneous interval contained its point estimate.

### Selected pair 1063: strategy 22 versus 66

Direct block counts are `n_AB=n_BA=3948`, `x_AB=2174` seat-1 A wins in order AB, and `x_BA=1992` seat-1 B wins in order BA. Thus:

```text
q_AB = 2174/3948 = 0.550658561297
q_BA = 1992/3948 = 0.504559270517
d_AB = 0.5(q_AB-q_BA) = 0.023049645390
balanced A wins = 2174 + (3948-1992) = 4130
balanced A rate = 4130/7896 = 0.523049645390
balanced A rate - 0.5 = 0.023049645390 = d_AB
```

Under the constrained equality null, `p0=(2174+1992)/(3948+3948)=0.527608915907`,

```text
z = (q_AB-q_BA) / sqrt[p0(1-p0)(1/3948+1/3948)]
  = 4.102616374897,
p = 2*NormalSF(|z|) = 4.085042673815476e-05.
```

Sorting the complete family of 2,850 independently recalculated p-values with a stable order puts this pair at Holm position 2,316. The running-max Holm adjustment is `0.021854978305`, exceeding family alpha 0.02, so it is not rejected.

Direct statsmodels score inversion, not the production wrapper, gives the ordinary difference interval `[0.019963862299, 0.072178338706]`, hence d interval `[0.009981931149, 0.036089169353]`. Bonferroni alpha is `0.02/2850 = 7.017543859649123e-06`; simultaneous difference bounds are `[-0.004387955805, 0.096375062872]`, hence d bounds `[-0.002193977903, 0.048187531436]`. They match the artifact exactly. Because the simultaneous lower bound is not above practical delta `0.03`, the upper is not below `-0.03`, Holm does not reject, and equivalence is disabled (`delta_equivalence: null`), the correct classification is **unresolved**. Nonsignificance was not mislabeled as equivalence.

### Power-plan scenario

The frozen family has 76 candidates, so `C(76,2)=2850` pairs and planning alpha `0.02/2850`. For target effect `d=0.03` and seat-1 advantage 0, the implemented mapping is `q_AB=0.5+0+0.03=0.53`, `q_BA=0.5+0-0.03=0.47`. Allocation is equal: 1,974 games in each root/order block, two roots, 3,948 games per order, 7,896 per pair.

I independently enumerated the full joint Binomial(3948,0.53) × Binomial(3948,0.47) grid and summed probability wherever the implemented pooled score statistic strictly exceeds the two-sided critical value. Achieved power is `0.8003606088973029`, exactly the plan. Repeating at 1,973 games per root/order (3,946 per order) gives `0.7996629814552753`, also exactly the reported previous-size value and below target 0.80. This confirms the selected block size, probability mapping, planning multiplicity, and implemented rejection rule for the worst reported scenario.

## Artifact integrity and expected absences

- All 11,586 sidecars have a corresponding artifact; every recorded size and SHA-256 matches; every listed source path exists.
- Sidecar config hashes split consistently into 152 root artifacts with base hash `1909…` and 11,434 pair artifacts with effective pair-context hash `134bbb…`.
- No duplicate/missing strategy, root, k, batch, H2H pair, H2H order, or immutable block was found.
- No impossible win/exposure count, interval ordering failure, or game/win/exposure/seat conservation failure was found.
- No required numeric NaN exists in performance or H2H inference. Nulls in diagnostic-type-specific columns, across-k `k`, and disabled equivalence fields are semantically expected.
- The zero-row `performance_control_contrasts.parquet` is expected because `screening.controls` is empty. The zero-row self-play and cycle artifacts are also schema-valid for this realized run and were not treated as missing evidence.
- Recalculations sorted by explicit coordinates/IDs. Shuffling input rows before grouping does not change the arithmetic; stable ID tie-breaks are explicit in performance, root ranking, and Holm ordering. TrueSkill remains intentionally sequential within each root/k cell.

In short, the empirical quantities and decision logic checked here agree independently and tightly. The release blocker is provenance/completion credibility, not a discovered numerical error.
