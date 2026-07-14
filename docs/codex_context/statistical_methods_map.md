# Codex Statistical Methods Map

Generated for Codex orientation. Treat this file as a cache, not authority.
For any statistical review, verify the listed source files directly and
separate implementation fidelity from model validity.

- Sources inspected: `utils/stats.py`, `analysis/metrics.py`,
  `analysis/performance.py`, `analysis/screening.py`, `analysis/seed_summaries.py`,
  `analysis/root_stability.py`, `analysis/run_trueskill.py`,
  `analysis/run_bonferroni_head2head.py`, `analysis/h2h_analysis.py`,
  `analysis/agreement.py`

## Review Frame

For every statistical claim, review in this order:

1. Intended estimator or decision rule.
2. Implementation mapping from formula to code.
3. Model assumptions and independence assumptions.
4. Artifact/report interpretation.
5. Tests, hand checks, and missing counterexamples.

## Wilson Binomial Confidence Interval

- Code: `src/farkle/utils/stats.py::wilson_ci`.
- Used by: `analysis/seed_summaries.py`, boundary handling in `analysis/meta.py`.
- Inputs: successes `k`, trials `n`, two-sided `alpha`.
- Formula: Wilson score interval with
  `z = norm.ppf(1 - alpha / 2)`,
  denominator `1 + z^2 / n`, center `p + z^2 / (2n)`, and Wilson margin.
- Edge behavior: rejects `n <= 0`; callers such as `seed_summaries` use
  `(0, 1)` for zero-game summaries.
- Tests: `tests/test_stats_wilson.py`, `tests/unit/utils/test_stats.py`,
  `tests/unit/analysis/test_seed_summaries.py`.
- Review risks: zero-game policy, integer count coercion, whether Wilson vs
  normal intervals are consistently labeled in downstream artifacts.

## Metrics Win-Rate Uncertainty

- Code: `src/farkle/analysis/metrics.py::_add_win_rate_uncertainty`.
- Inputs: per-strategy `wins`, `games`, `win_rate`.
- Formula: normal approximation `se = sqrt(p * (1 - p) / n)`,
  `win_rate_ci_lo = p - 1.96 * se`, `win_rate_ci_hi = p + 1.96 * se`,
  clipped to `[0, 1]`.
- Outputs: `se_win_rate`, `win_rate_ci_lo`, `win_rate_ci_hi`.
- Tests: `tests/unit/analysis/test_metrics.py`,
  `tests/unit/analysis/test_metrics_branches.py`,
  `tests/integration/test_metrics_stage.py`.
- Review risks: normal approximation differs from Wilson intervals in seed
  summaries; zero-game handling leaves CI equal to win rate; verify denominator
  is games for the intended Bernoulli unit.

## Canonical Performance Estimators

- Code: `src/farkle/analysis/all_player_metrics.py` and
  `src/farkle/analysis/performance.py`.
- Per-k estimand: raw wins divided by raw player-game exposures, with chance
  baseline `1/k` and `chance_delta = win_rate - 1/k`.
- Per-k uncertainty: full-sample Wilson interval plus batch
  `MCSE = sample_sd(batch win rates) / sqrt(number of batches)` and a t interval.
- Across-k estimand: equal-k mean of chance delta over complete configured
  support only. Its analytic variance is the sum of squared equal weights times
  the independent-k batch-MCSE variances.
- Robustness: exact finite-grid Pareto membership and a separately identified
  maximin descriptive leader.
- Resampling: each replicate selects complete deterministic-batch vectors, so
  all strategies share the same selected batches within k. Outputs summarize
  ranks, top-N membership, practical-shortlist inclusion, and declared-control
  contrasts.
- Tests: `tests/unit/analysis/test_all_player_metrics.py` and
  `tests/unit/analysis/test_performance.py` contain hand-computed return,
  baseline, MCSE, equal-k variance, support, and determinism checks.

## Aggregation Across Player Counts

- Code: `analysis/metrics.py::_compute_weighted_metrics`,
  `analysis/seed_summaries.py::_build_combined_seed_summary`,
  `utils/aggregation.py`.
- Aggregation modes: `game-count`, `equal-k`, and `config`.
- Game-count: row weights are game counts.
- Equal-k: each player count contributes equal total mass by scaling rows by
  `games / total_games_for_k`.
- Config: configured per-k weights are normalized by total games for that k;
  missing k weights are treated as zero after warning.
- Tests: `tests/unit/analysis/test_metrics*.py`,
  `tests/unit/analysis/test_seed_summaries.py`,
  `tests/unit/utils/test_stage_io_and_aggregation.py`.
- Review risks: combined rows can mean "across k" rather than "across seeds";
  artifact names with `combined` need scope-specific interpretation.

## Descriptive Performance Screening

- Code: `src/farkle/analysis/screening.py`.
- Inputs: validated per-k performance, complete-support equal-k performance,
  and joint deterministic-batch resampling summaries.
- Evidence retained: root and per-k chance deltas, equal-k score order,
  bootstrap top-N and shortlist inclusion, declared controls and mandatory
  diagnostics, Pareto membership, and the separate maximin leader.
- Practical bands compare observed scores with explicitly configured
  thresholds. They are descriptive and are not tests of equality, final tiers,
  or unique-best decisions.
- Tests: `tests/unit/analysis/test_screening.py`.

## Screening Workload Planning

- Code: `src/farkle/simulation/workload_planner.py`,
  `src/farkle/simulation/runner.py::_plan_workload_from_config`.
- Method: find the smallest shuffle count whose worst-case full 95% Wilson
  interval width meets `screening.resolution_delta`, then round upward to equal
  contiguous batches with the configured minimum batch size.
- Output: required shuffles and games, batch construction, achieved resolution,
  operational cap state, and projected runtime.
- Tests: `tests/unit/simulation/test_workload_planner.py`,
  `tests/unit/simulation/test_runner_wrapper.py`.
- H2H power remains a distinct procedure and must match the final H2H test.

## Canonical Seat Effects

- Code: `src/farkle/analysis/seat_analysis.py`.
- Inputs: validated normalized per-k row partitions with one root and complete
  configured k support.
- Canonical counts: raw wins and player-game exposures for every
  `(root, k, deterministic batch, strategy, seat)` cell.
- Within-k estimands: strategy-specific and population-wide seat win rates
  minus the exact `1/k` chance baseline.
- Across-k estimand: equal-k or declared-k weighted effects only for identical
  common k support. Missing strategy/k/seat cells are not silently reweighted.
- Secondary diagnostic: the exposure-weighted cross-k mixture reports its own
  exposure-weighted chance baseline and is not interchangeable with the
  standardized estimand.
- Additional diagnostics: self-play P1 effects and paired mirrored two-player
  P1-win differences, with unmatched orientations retained as support counts.
- Tests: `tests/unit/analysis/test_seat_analysis.py`.

## TrueSkill Screening Ratings

- Code: `src/farkle/analysis/trueskill.py`,
  `src/farkle/analysis/run_trueskill.py`.
- Inputs: curated rows with strategy columns and ranking information.
- Update rule: streams games and calls `trueskill.TrueSkill(...).rate(...)`
  with ranks from `P#_rank`, `seat_ranks`, or fallback winner-plus-tied-losers.
- Hyperparameters: `cfg.trueskill.beta`, `tau`, `draw_probability`.
- Resume: per-block checkpoints and done stamps.
- Canonical model outputs remain per root and k. Model sigma is retained only
  as TrueSkill state and is not propagated into a formal cross-k uncertainty.
- Candidate contribution: convert `mu` to a percentile rank independently in
  every root/k cell, require complete cell support, and average those
  percentiles with equal cell weight. The contribution contains no sigma.
- Diagnostics: compare the canonical result with a tau-zero replay and reversed
  game order; report held-out log loss, Brier score, top-prediction calibration,
  and their uniform baselines.
- Tests: `tests/unit/analysis/test_run_trueskill_*.py`,
  `tests/unit/analysis/test_analytics_trueskill.py`, and
  `tests/unit/analysis/test_trueskill_screening.py`.
- Review risks: TrueSkill is a sequential rating model, not a classical
  independent-binomial estimator. Game-order and tau sensitivity remain
  diagnostics, and sigma is not a sampling standard error.

## HGB Predictive Associations

- Code: `src/farkle/analysis/run_hgb.py` and
  `src/farkle/analysis/hgb_feat.py`.
- Inputs: canonical per-k performance estimates and the immutable current-run
  strategy manifest. TrueSkill ratings are not an HGB dependency.
- Evaluation: assign strategy configurations to deterministic coordinate-RNG
  folds, fit on the remaining configurations, and calculate predictions and
  permutation importance only on the held-out configurations.
- Output: per-fold MAE/R2, per-strategy out-of-sample predictions, mean
  association importance, between-fold variability, permutation-repeat
  variability, and exact finite-grid support.
- Interpretation: option and interaction results are predictive associations
  on the simulated finite grid, never causal effects.
- Candidate generation: a full-grid exploratory fit may draft valid one-option
  mutations in `future_simulation_proposals.parquet`. Drafts have no current
  strategy ID and are excluded from every current-run analysis; they require a
  new simulation manifest and direct simulation.
- Tests: `tests/unit/analysis/test_run_hgb_functionality.py`,
  `test_run_hgb_helpers.py`, and `test_hgb_feat.py`.

## Head-To-Head Simulation And Holm Decisions

- Candidate freeze code: `src/farkle/analysis/candidate_family.py`.
- Canonical candidate inputs: the combined-root complete-support across-k
  win-rate contribution and the complete root/k TrueSkill percentile
  contribution. Explicitly labelled single-root counterparts are also accepted.
- Default family: top 75 from each method plus configured controls and mandatory
  diagnostics. With no cap, retain the complete union. With a cap, reduce both
  nonprotected method cutoffs simultaneously until the family fits; never spend
  the remaining slots according to incomparable cross-method score scales.
- Provenance: every source rank, shared membership, admission reason, cutoff
  round, removed tail, overlap statistic, family content hash, and projected
  pair/root/order workload is written before scheduling.
- Tests: `tests/unit/analysis/test_candidate_family.py` covers balanced-tail
  contraction, protected candidates, no-cap behavior, sidecar scope, family-hash
  replay, and explicit single-root labelling.
- Power and scheduling code: `src/farkle/analysis/h2h_schedule.py`.
- Planning test: the two-sided independent two-proportion score procedure at
  Bonferroni `family_alpha / pair_count`; the final multiplicity method remains
  Holm. The planner finds the smallest equal root/order block size meeting
  target power in the worst configured common seat-effect scenario and reports
  both `0.03` and `0.04` sensitivity grids.
- Allocation: every unordered pair receives the same count in each root/order
  cell. Single-root execution remains exactly balanced between orders and is
  explicitly labelled.
- RNG and resume: each immutable block owns
  `(root, pair, order, game_index)` coordinate streams. Atomic block artifacts
  are skipped only after their sidecars and family/schedule identity validate.
- Tests: `tests/unit/analysis/test_h2h_schedule.py` covers minimum power,
  root/order equality, cap stop/resume, coordinate separation, single-root
  labelling, and block-level resume.
- Inference code: `src/farkle/analysis/h2h_inference.py`.
- Estimand: `d_AB = 0.5 * (q_AB - q_BA)`, where each rate is the seat-1 win
  rate in its independent ordering after raw counts combine across roots within
  that order.
- Test and intervals: equality uses the constrained-null two-proportion score
  statistic. Ordinary and Bonferroni simultaneous limits invert that same score
  procedure and are divided by two to report the `d_AB` scale.
- Multiplicity and decisions: Holm adjusted p-values determine statistical-only
  advantages; simultaneous bounds determine practical dominance. Equivalence
  requires an explicit configured margin and simultaneous containment;
  everything else is unresolved.
- Checked alias: exact order balance makes the combined A-win rate equal to
  `0.5 + d_AB`; it is retained only as a verified point estimate, not as a
  one-sample test.
- Tests: `tests/unit/analysis/test_h2h_inference.py` verifies the constrained
  null statistic, score interval, raw root combination, alias identity, Holm,
  practical/statistical/unresolved classes, optional equivalence, and balance
  rejection.
- Dominance code: `src/farkle/analysis/dominance.py`.
- Graphs: practical edges require simultaneous practical bounds; statistical
  edges require Holm rejection. Unresolved and equivalent comparisons create no
  edge in either graph.
- Cycles and fronts: strongly connected groups remain explicit and are
  contracted only for condensation-DAG construction. Repeated zero-indegree
  layers define partial fronts; they do not imply a total inferential order.
- Display order: round-robin mean, practical wins/losses, tournament score, and
  stable identifier order rows within a front only. The artifact explicitly
  marks that order noninferential.
- Unique best: permitted only for direct practical dominance over every other
  frozen finalist; paths, front membership, and descriptive scores are
  insufficient.
- Tests: `tests/unit/analysis/test_dominance.py` covers cycles, condensation
  fronts, unresolved comparisons, direct unique-best evidence, complete pair
  support, and identifier-renaming invariance.
- Code: `src/farkle/analysis/head2head.py`,
  `src/farkle/analysis/run_bonferroni_head2head.py`,
  `src/farkle/analysis/h2h_analysis.py`.
- Simulation: each pair is split across both seat orders (`a_b` and `b_a`) and
  writes pairwise, ordered-pairwise, and self-play symmetry artifacts.
- Legacy note: the older runner and post-processor still contain one-sample
  binomial artifacts pending migration cleanup. They are not canonical inputs
  to `h2h_inference.py` and cannot satisfy its sidecar contract.
- Tie policies: `neutral_edge` marks ties non-significant; `simulate_game`
  gives deterministic non-significant tie-break direction from a seeded RNG.
- Tests: `tests/unit/analysis/test_run_bonferroni_head2head*.py`,
  `tests/unit/analysis/test_head2head.py`,
  `tests/unit/analysis/test_h2h_analysis.py`,
  `tests/unit/analysis/test_agreement_ties.py`.
- Review risks: one-sided stored pair p-values differ from two-sided
  post-H2H decisions; pairwise tests are dependent because strategies share
  tournament context; final graph ranking can be partial or cyclic.

## Two-Root Combination And Stability

- Code: `src/farkle/analysis/root_stability.py`.
- Inputs: sidecar-validated unconditional batch sufficient statistics for every
  configured `(root, k, batch, strategy)` cell, with exactly two roots and
  complete identical support.
- Within-k combined estimand: total wins across both roots divided by total
  player-game exposures across both roots. Root-specific estimates are retained
  beside the combined estimate.
- Across-k estimand: the declared weighted mean of `win_rate - 1/k`, calculated
  only after within-k combination and only over complete configured support.
  Its MCSE uses the declared independent-k variance sum.
- Root discrepancy: `root_a chance_delta - root_b chance_delta`, standardized by
  `sqrt(MCSE_a^2 + MCSE_b^2)` and separately scaled by the configured practical
  stability threshold.
- Joint diagnostic: deterministic batch-vector resampling calibrates the maximum
  absolute standardized discrepancy across all per-k and across-k cells.
- Descriptive stability: Spearman and Kendall rank correlation, top-10/25/50 and
  candidate-cutoff overlap, control movement, and practical-shortlist changes.
- Time stability: recompute at declared matched cumulative batch fractions and
  compare contiguous first and second halves within each root.
- Interpretation: the roots are independent RNG domains for one fixed design,
  not a sampled root population. The module does not use random-effects
  estimators, root variance components, or two-root population intervals.
- Tests: `tests/unit/analysis/test_root_stability.py` includes hand-computed raw
  count combination, batch-MCSE, support, sidecar, and deterministic-bootstrap
  checks.

## Agreement Metrics

- Code: `src/farkle/analysis/agreement.py`.
- Methods compared: TrueSkill, frequentist, and combined head-to-head when
  available.
- Rank metrics: Spearman and Kendall on common strategy sets.
- Tier metrics: adjusted Rand index and normalized mutual information on common
  tier maps.
- Stability: per-seed score standard deviations for common strategies.
- Tests: `tests/unit/analysis/test_agreement_payload.py`,
  `tests/unit/analysis/test_agreement_ties.py`.
- Review risks: agreement is descriptive, not proof of correctness; missing
  strategies are summarized as coverage, and tied scores are warned but allowed.

## RNG Diagnostics

- Code: `src/farkle/analysis/rng_diagnostics.py`.
- Intended use: optional interseed diagnostics over game seeds, win indicators,
  and game length to flag ordering artifacts.
- Output fields are `diagnostic_band_lower` and `diagnostic_band_upper`. The
  sidecar records the `1.96/sqrt(n)` approximation and explicitly states that
  inclusion within a reference band does not establish independence.
- Tests: `tests/unit/analysis/test_rng_diagnostics.py`,
  `tests/unit/analysis/test_rng_diagnostics_branches.py`.
- Review risks: diagnostics can flag suspicious patterns but cannot prove RNG
  independence or simulation validity by themselves.

## Exact Roll Enumeration

- Code: `src/farkle/analysis/roll_enumeration.py`.
- Method: enumerate every one of the `6**d` ordered outcomes for each dice count
  from one through six and evaluate it with the production scoring engine.
- Selection rule: `production_max_immediate_score_v1`, with zero score for a
  farkle. Outputs include the exact score/scoring-dice distribution, farkle and
  hot-dice probabilities, expected immediate score, score quantiles, and
  expected scoring dice.
- No RNG or sampling interval is involved; sidecars label the finite
  enumeration and ordered-outcome denominator.
- Tests: `tests/unit/analysis/test_roll_enumeration.py`.

## High-Value Missing Hand Checks

- Meta-analysis: two strategies, two seeds, one boundary-rate case, expected Q,
  I2, tau2, fixed/random switch, and CI values.
- Head-to-head: three pairs with known p-values and Holm adjusted p-values,
  including one tie and one reversed direction.
- TrueSkill: tiny deterministic ordered game stream confirming rank extraction,
  checkpoint resume equivalence, and precision-aggregation arithmetic.
- Metrics vs seed summaries: same wins/games through normal CI and Wilson CI to
  document intentional interval differences.
