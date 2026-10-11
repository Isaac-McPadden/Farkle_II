# Task 0: workbook coverage and methodological decisions

Source: `Farkle Questions.xlsx`, `Sheet1`, identity recorded in
[Task 0 contract](task0_contract.md). Reconciled against source revision
`7db67ccc94a143ebd16c335563513e89a84a62aa` on 2026-09-24.

**Existing** means the computation is present in source, not that a production
result has been generated or qualified. **Planned** means a definite deliverable
under this contract. **Partial** identifies both coverage and the outstanding
exclusion. **Deferred/excluded** is a deliberate first-release boundary.
No excluded question is counted as answered by a neighboring statistic.

## Artifact/source legend

Existing paths below are AppConfig getters, resolved against the correct root
or pair owner. Column names are representative evidence fields; Task 8's generated
dictionary must enumerate the actual published fields and method metadata.

| Alias | Existing output/source |
| --- | --- |
| EP | `cfg.performance_by_k_path(k)`: `win_rate`, `chance_delta`, `batch_mcse`, attempted/completed support, batch/Wilson interval columns; [performance.py](../../src/farkle/analysis/performance.py) |
| EK | `cfg.performance_across_k_path()`: `equal_k_score`, `complete_support`, `worst_k`, `pareto_member`, `maximin_value`; `cfg.performance_player_count_effects_path()` for contrasts/rank agreement |
| ES | `cfg.screening_path()`, `cfg.performance_bootstrap_path()`, `cfg.h2h_candidate_family_path()` for descriptive order, top-N resampling and actual frozen membership |
| ET | `cfg.trueskill_rating_path(k)`, `cfg.trueskill_screening_diagnostics_path()`, `cfg.structure_agreement_summary_path()`; [trueskill_screening.py](../../src/farkle/analysis/trueskill_screening.py) |
| EH | `cfg.h2h_pairwise_inference_path()`: `d_ab`, `q_ab`, `q_ba`, ordinary/simultaneous bounds, Holm decisions, viability, planned power/support; `cfg.h2h_power_plan_path()` and `cfg.h2h_order_counts_path()` |
| ED | `cfg.h2h_dominance_edges_path()`, `cfg.h2h_cycle_groups_path()`, `cfg.h2h_dominance_fronts_path()`, `cfg.h2h_dominance_summary_path()` |
| EE | `cfg.seat_effects_by_k_path(k)`, `cfg.seat_population_by_k_path(k)`, `cfg.seat_standardized_across_k_path()`: wins/exposures, `seat_effect`, completed diagnostics |
| EB | `cfg.metrics_all_player_batch_path(k)`: return sums, `turn_return_turn_weighted`, `turn_return_game_weighted_exact`, proxy/mismatch diagnostics and behavioral moments |
| EG | `cfg.scope_dir('game_stats', ArtifactScope.CONCAT_KS) / 'game_length.parquet'` and `'margin_stats.parquet'`; [game_stats.py](../../src/farkle/analysis/game_stats.py) uses explicit observational units and conditioning |
| ER | `cfg.root_discrepancies_path()`, `cfg.root_rank_stability_path()`, `cfg.root_top_n_stability_path()`, `cfg.root_shortlist_changes_path()`, convergence/half-drift paths; H2H root diagnostics/agreement paths |
| EM | `cfg.hgb_importance_path(k)`, `cfg.hgb_predictive_scores_path(k)`, `cfg.hgb_fold_metrics_path(k)`, `cfg.hgb_future_proposals_path()`; [run_hgb.py](../../src/farkle/analysis/run_hgb.py) |
| EV | Current row/artifact/recovery checks, exact roll enumeration, RNG diagnostics, completion and final release audit; see [testing map](../codex_context/testing_and_review_map.md) |

Planned aliases F/B/I/S/Q/U/B7/Q7/O7/O/OA/R1/R2/RI/PE/MC/MR/G refer to the
exact paths and schemas in [artifact specification](task0_artifacts.md).

## Complete question-to-output map

Every substantive B/C entry has a resolution below. An em dash means those
cells contain no methodological note, not that a method decision is missing.

| Workbook cell | Research target | Coverage and output/owner | Methodological notes resolved / claim boundary |
| --- | --- | --- | --- |
| A3 | Strategy win probability by k | Existing EP; Task 8 G | B3: wins/attempted exposures is primary; completed-only probability has a different denominator; report support and MC precision |
| A4 | Advantage over 1/k | Existing EP `chance_delta`; G | B4:C4: probability-point difference from baseline; score multimodality does not determine win-probability uncertainty. Distinguish outcome spread from estimator uncertainty and preserve batch dependence |
| A5 | Rankings, top-N, finalists | Existing EP/ES | B5: descriptive rankings/selection; not simultaneous superiority or equality |
| A6 | Consistently good performance | Partial EK/ER/EE, planned F/Q/G | B6: supported filters are k, root, seat, completion; arbitrary opponent-composition robustness is deferred; do not select favorable filters and generalize |
| A7 | TrueSkill versus win-rate screening | Existing ET/agreement | B7: ranking differences do not isolate opponent effects; model assumptions, update order and completed-game conditioning also differ; sigma is not an ordinary sampling SE |
| A11 | Performance changes with k | Existing EP/EK/player-count effects; G | B11: compare complete per-k estimates/chance deltas; many observations do not by themselves make a pattern meaningful |
| A12 | k specialists | Existing EP/EK; G identifies `worst_k` and relative profile | B12: describe strong/weak k cells and support; no new anomaly/significance test is promised |
| A13 | Robust strategies across k | Existing EK/ER | B13: robustness is across declared complete k support; not evidence of arbitrary-opponent robustness |
| A14 | Equal-k chance-corrected comparison | Existing EK | Mean of `p_k-1/k`, absolute probability points; not relative lift |
| A15 | Pareto and maximin | Existing EK | Descriptive finite-grid criteria; not unique-best inference |
| A16 | Unequal-data pooling artifacts | Existing EP/EK plus G weighting explanation | B16: show per-k exposure weights and contrast with explicit equal-k estimand; this is a weighting/sensitivity issue, not a generic extra hypothesis test. G can compute the exposure-weighted diagnostic from EP counts and label it secondary |
| A20 | Full final-score distribution | Planned F/Q, Tasks 1–2 | Completed exposures; stopped-score distribution separate |
| A21 | Finish-rank frequencies | Planned F competition/midrank, Q | Preserve recorded engine ranks/winner; tied descriptive first places do not add wins |
| A22 | Losing severity | Planned F/Q/U | Conditional on official completed losses; display L/C with severity |
| A23 | Consistency versus volatile outcomes | Planned F/Q/G | Empirical shape/quantiles and mean deficit; no scalar universal risk ranking or unpromised tail CI |
| A24 | Every player's score/rank/margin | Planned F/B/S/Q | All actual-seat exposures including repeated strategy IDs; overall is count-weighted seat sum |
| A28 | Configured finite opponent mixture | Existing EP | Tournament estimand; no claim of universal best against arbitrary opponents |
| A29 | Sensitivity to particular opponents | Deferred | No dedicated composition experiment or identified sensitivity estimator |
| A30 | Strength dependent on favorable matchups | Partial EP/EK; composition component deferred | Across-k robustness does not answer multiplayer composition dependence |
| A31 | Randomness with exact matchup fixed | Deferred | Existing repeated-matchup rounds lag diagnostics do not estimate winner entropy/probabilities |
| A32 | Tournament ranking versus finalist H2H | Existing agreement/EH/ED; G | Selection-conditioned comparison of different estimands |
| A40 | Balanced H2H winners/losers | Existing EH | Frozen family, completed support, both seat orders and roots; no-test family members retained |
| A41 | Seat-adjusted effect | Existing EH `d_ab` | Balanced A win probability is `0.5+d_ab`; retain order-specific evidence |
| A42 | Statistical wins/losses | Existing EH Holm decisions | Configured superiority family alpha; viability required |
| A43 | Practically important differences | Existing EH/ED | Simultaneous bound must cross practical threshold; point estimate beyond it is insufficient |
| A44 | Unresolved comparisons | Existing EH, planned Task 4 flags/G | Distinguish wide intervals, insufficient completed support, and operational nonviability |
| A45 | Equivalence | Planned Task 4 independent flag and configured margin | Interval strictly inside configured equivalence bounds; can coexist with statistical advantage; no transitive equivalence grouping |
| A46 | Dominance fronts | Existing ED | Partial order; same front does not establish equivalence |
| A47 | Cycles, reversals, incomparability | Existing ED and H2H root diagnostics | Preserve cycles and unresolved edges; display order is not inferred total order |
| A48 | Power and resolution | Existing plan/EH plus Tasks 3–4/G wording | Separate configured detection power under alternatives, attained support, interval width, decisions. Do not estimate observed power from the observed effect |
| A52 | Seat advantage | Existing EE/EH | Report k-specific rates and support; balanced allocations do not imply absence of seat effect |
| A53 | Seat advantage changes with k | Existing EE standardized/population products | Complete common support and explicit weights; k-specific seats are not silently pooled |
| A54 | Seat effect on win/score/rank/margin | Existing EE plus planned F/Q/U | Win inference and descriptive outcome differences have separately labeled precision; no new formal all-seat contrast family |
| A55 | H2H seat-order differences | Existing EH order counts/q_ab/q_ba | Retain both orders and completion support |
| A56 | Balancing versus stratified inference | Existing EH method; G explains | Keep order-stratified score inference; exact balance does not eliminate the nuisance seat effect |
| A57 | Global turn-opportunity imbalance | Planned O7/O/OA, Task 7 | Completed-game mean turn range, seat shares and equal-k root-specific headline; game counted once |
| A61 | Sum(score)/sum(turns) | Existing EB overall; planned B7/Q7 by seat | Turn-weighted rate, explicit completion/attempt conditioning; zero total turns means undefined |
| A62 | Mean(score/turns) | Existing EB overall; planned B7/Q7 by seat | Exposure-weighted game-rate mean; preserve explicit safety zero-turn convention; differs from A61 |
| A63 | Score/round proxy and error | Existing EB; planned Q7 by seat | Labeled proxy, exact-minus-proxy gap and mismatch prevalence, not an interchangeable turn count |
| A67 | Turn-to-point efficiency | Existing EB plus planned Q7 | Two weighting interpretations; terminal averages, not causal efficiency |
| A68 | Game length by strategy | Existing EG plus planned Q7 mean rounds/turns | Strategy-conditioned exposure view versus one-row-per-game view must be labeled; repeats count according to that unit |
| A69 | Behavior of successful strategies | Existing EB/EM plus planned Q7/G | Compare opportunity-normalized rolls/farkles/hot-dice/heuristic counts; descriptive association |
| A70 | Faster scoring, survival, risk, states | Partial Q7; state/risk-mechanism component excluded | Terminal totals cannot identify state-specific decisions, survival mechanisms, or causal reasons for winning |
| A71 | Behavior changes by k/seat | Planned B7/Q7 | New seat-keyed aggregation from rows; cannot recover seat behavior from current all-player sums |
| A75 | Reproducibility under second root | Existing ER/G | Fixed-design two-root challenge; not a random population of seeds |
| A76 | Per-k rankings/top-N/finalist stability | Existing ER/ES | Rank/top-N movement, inclusion and shortlist changes with complete support |
| A77 | Across-k stability | Existing ER/EK | Explicit common k weighting and root estimates |
| A78 | H2H directions/structure reproducibility | Existing H2H root diagnostics/agreement/ED | Root diagnostics are labeled; no invented independently powered root conclusions |
| A79 | Movement versus Monte Carlo variation | Existing ER joint reference/bootstrap evidence | Descriptive precision/reference frequencies; not heterogeneity or multiple-testing inference across two roots |
| A80 | Dependence from shuffle/batching/order | Existing EV plus planned PE/MC/MR | Task 5 must demonstrate declared capability; batch-only monitoring cannot rule out within-batch dependence |
| A81 | Fixed-matchup entropy/randomness | Deferred | Omitted, not answered by root agreement or lag bands |
| A87 | Parameter associations | Existing EM; planned R1/G | Held-out predictive associations on finite grid, not causal parameter effects |
| A88 | Associations change across k | Existing per-k EM; planned R1 | Compare root/k products; do not pool different targets/support silently |
| A89 | Interactions/nonlinear regions | Planned R1/R2/RI, Task 7 | Ten bounded one-way tables and three declared pairs; valid matched backgrounds; observed nonadditivity; no mixed-fold model-interaction claim or exhaustive search |
| A90 | Useful out-of-sample prediction | Existing EM fold metrics/predictions | Surface MAE/R2 and support, including weak/negative results; training fit is not evidence |
| A91 | Proposed new strategies | Existing EM future proposals | Candidates for independent simulation; no performance claim before testing |
| A97 | Expected control-strategy behavior | Excluded as a prescribed-ranking program | Keep rule-derived checks, exact enumeration, synthetic method fixtures and integrity oracles; no declared strong/weak ranking gate |
| A98 | Rules/grid/mixture/version conditioning | Existing provenance plus planned G | State complete design and code/config identity, including attempt/completion conditioning |
| A99 | Missing cells or unequal support | Existing support checks plus planned I/B/S/RI | Verified zero requires certified coverage; missing required data fails. No zero-fill, silent population restriction, or weight renormalization |
| A100 | Diagnostics detect faults | Existing EV plus planned Task 5 PE/MC/MR | Fault-to-check matrix, fault/clean fixtures, support, false-alert evidence, and blind spots; completion alone is insufficient |
| A101 | Uncertainty matches claim | Existing declared methods plus planned U/G | New intervals use four specified estimators and one complementary rate; pointwise Monte Carlo precision; no blanket interval coverage claim |
| A102 | Honest unresolved results | Existing EH/ED plus planned G | Undefined support, missing inputs, equivalence, unresolved comparisons and display rankings remain distinct |

Workbook framing statements are preserved as constraints: A34 separates the
tournament and finalist-H2H estimands; A38 restricts H2H inference to the frozen
family; A59 introduces the three scoring-rate views; A83 limits the two-root
interpretation; A93 restricts HGB to prediction/association and future candidates.
Headings A1/A9/A18/A26/A36/A50/A65/A73/A85/A95 and C1 are labels, not additional
unmapped research questions. All notes in B3/B4/B5/B6/B7/B11/B12/B13/B16 and C4
are resolved in their corresponding rows.

The mapping freezes scope and methods, not expected scientific findings. Task 8
must generate actual results from authenticated outputs and retain these named
exclusions even when nearby analyses succeed.
