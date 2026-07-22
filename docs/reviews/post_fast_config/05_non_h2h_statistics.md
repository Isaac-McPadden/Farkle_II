# 05 - Non-H2H statistical estimators, diagnostics, screening, and models

## Scope, evidence, and verdict

This review covers commit `6be5f5fa11df77155621bfc81188c7515f38f8de` and the completed
`configs/fast_config.yaml` run at `data/results_seed_pair_32_33`. It excludes H2H
estimation and dominance except where final report language distinguishes H2H from
tournament screening. I inspected implementation, tests, sidecars, and produced
artifacts and independently recomputed selected values directly from curated rows and
batch sufficient statistics without calling the production aggregation functions.

The central performance formulas are implemented correctly: per-`k` rates use raw wins
and player-game exposures against `1/k`; deterministic-batch MCSE is
`s_batch/sqrt(B)` with `B-1` t degrees of freedom; canonical across-`k` performance is
the complete-support equal-`k` mean of `win_rate - 1/k`; and the analytic across-`k`
MCSE is the independent-`k` variance sum. Joint resampling preserves complete
cross-strategy batch vectors. TrueSkill percentile aggregation, HGB held-out strategy
folds, and raw-count two-root combination also recalculate as designed.

Those successes do not make the non-H2H evidence sound. Maximum-round aborts are
recorded as arbitrary wins, 32-bit seed narrowing violates the batch and cross-`k`
independence assumptions, and strategy-conditioned game statistics systematically
double-count winners. Additional diagnostics and artifact contracts have material
method/label defects. **Verdict: unsound within this review's scope.**

One evidence limitation applies throughout: reviewed sidecars say
`code_revision: "unknown"`, so the completed artifacts are not cryptographically bound
to the reviewed commit. The independent identities below establish behavioral agreement
for key outputs, but do not repair that provenance gap.

## Findings

### F1 - Maximum-round aborts are assigned arbitrary winners and contaminate every downstream tournament method

- **Severity:** High
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** At the safety cap, `FarkleGame.play` sets `max_rounds_hit`, then
  stable-sorts players by score and assigns distinct ranks even when scores tie
  (`src/farkle/game/engine.py:444-468`). `_play_game` requires exactly one rank-1
  player and publishes that player as the winner
  (`src/farkle/simulation/simulation.py:382-395`); the tournament then increments the
  corresponding strategy's win count (`src/farkle/simulation/run_tournament.py:229-238`).
  All-player metrics retain both the win and abort exposure
  (`src/farkle/analysis/all_player_metrics.py:177-190`), and performance sums them
  without a completion filter (`src/farkle/analysis/performance.py:100-107`).

  An independent scan of all six curated fast-run cells found 653,600 games, 2,981
  maximum-round aborts, and 2,981 tied aborts: root 32 had 1,479/3/0 aborts at
  `k=2/4/5`, while root 33 had 1,498/1/0. Every one nevertheless has a non-null
  `winner_seat`. The all-player artifacts retain 2,958, 12, 0 abort player-exposures
  for root 32 and 2,996, 4, 0 for root 33, confirming that the diagnostic count is
  wired but that those games still enter win and return estimates.
- **Consequence:** Per-`k` win rates, batch MCSE inputs, seat effects, screening,
  root-stability diagnostics, TrueSkill updates/calibration, HGB targets, and
  strategy-conditioned game summaries include arbitrary outcomes. The effect is not
  an uncertainty-resolution problem: additional simulation of the same rule converges
  more precisely on the arbitrary tie-break policy. Retaining an abort count beside an
  estimate does not remove the bias.
- **Smallest reasonable remediation:** Give termination status and unresolved/tied
  completion explicit semantics. Do not assign a winner or rank 1 to a tied safety-cap
  abort; retain attempted, completed, aborted, and tied counts; exclude unresolved games
  from completed-game win denominators; declare the return estimand's abort policy; and
  rebuild every affected tournament, metrics, screening, model, root-stability, and
  report artifact. Add zero-score safety-cap tests through the full non-H2H pipeline.

### F2 - Seed narrowing violates the independence assumptions used by batch and across-k uncertainty

- **Severity:** High
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** Tournament games have full semantic coordinates, but
  `coordinate_seed` is narrowed to `uint32` before `_play_game`
  (`src/farkle/simulation/run_tournament.py:192-215`). `_play_game` then treats that
  narrowed integer as a new root, and player streams depend only on it, `k`, and seat
  (`src/farkle/simulation/simulation.py:325-355`). This discards the original root,
  shuffle, and game coordinates. By contrast, the RNG utility can consume all of those
  coordinates directly (`src/farkle/utils/random.py:63-121`).

  The fast artifacts contain 25 duplicate-seed excesses across the two roots. Twelve
  are within a root/`k` cell and cross deterministic batches; thirteen are cross-`k`
  collisions (seven in root 32 and six in root 33). For example, root-32 game seed
  `2558242864` is reused at `(k, shuffle, game, batch)=(2,441,8,10)` and
  `(4,1338,0,31)`. Thus the per-`k` batch units are not strictly disjoint RNG domains,
  and the actual data construction does not satisfy the independent-`k` assumption
  declared by `performance_equal_k.parquet.sidecar.json` as
  `uncertainty_method: independent_k_variance_sum`.
- **Consequence:** The formulas in `performance.py:223-237` and
  `root_stability.py:245-264` match their stated estimators, but their independence
  premise is false for the produced data. The fast-run collision fraction is small and
  does not visibly move the hand calculation, but collision probability grows
  quadratically with workload, directly threatening production MCSEs, analytic
  intervals, joint resampling, and two-root convergence diagnostics.
- **Smallest reasonable remediation:** Construct player generators directly from the
  complete tournament coordinate plus seat. At minimum, stop narrowing to 32 bits.
  Version the RNG contract, invalidate affected simulations, and test uniqueness and
  logical independence at production-scale coordinate counts. The across-`k` sidecar
  should state the concrete coordinate separation that justifies its variance rule.

### F3 - Strategy-conditioned game statistics count every winner twice

- **Severity:** High
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** `_compute_k_game_stats` selects every schema column ending in
  `_strategy` (`src/farkle/analysis/game_stats.py:598-605`). The curated schema contains
  both the seat columns and top-level `winner_strategy`. The loop then adds game length
  and margins once for every selected column (`game_stats.py:644-675`). The rare-event
  shard repeats the same broad selection and melts `winner_strategy` as another seat
  (`game_stats.py:2167-2169,2215-2237`).

  In root-32 `k=2`, every checked strategy satisfies
  `game_stats observations = performance raw_exposures + raw_wins`. Strategy 27 has
  7,064 reported observations = 4,300 seat exposures + 2,764 wins; strategies 78, 79,
  0, and 2 similarly report 7,013, 6,931, 4,495, and 6,802. The affected artifact is
  `results_seed_32/analysis/04_game_stats/by_k/2p/game_stats.2p.parquet`; the defect
  propagates to both `concat_ks` tables, equal-`k` game-length/margin tables, and
  `across_k/rare_events.parquet`. Its `summary_level="game"` rare-event rows are
  actually repeated strategy exposures, including an extra winner row, despite the
  sidecar claiming `replication_unit: game`.
- **Consequence:** Strategy-conditioned means, quantiles, standard deviations, close-game
  rates, rare-event rates, and cross-`k` summaries are winner-weighted. Better strategies
  receive more duplicate observations, so these outputs cannot describe either games
  containing a strategy or player-game exposures. Population game-length rows that are
  updated once per source row are not affected by this particular duplication.
- **Smallest reasonable remediation:** Select only canonical seat columns with an
  anchored pattern such as `^P[1-9][0-9]*_strategy$`; state whether the unit is a game
  containing a strategy or a player-game exposure; rebuild all game-statistics and
  rare-event artifacts; and assert that per-strategy observations equal canonical seat
  exposures for tournament rows.

### F4 - Root stability performs unadjusted significance classification inside a descriptive screening layer

- **Severity:** Medium
- **Confidence:** High
- **Classification:** Confirmed design-contract defect
- **Evidence:** `_classification` labels an effect
  `statistically_above_below_practical` when an ordinary normal interval excludes zero
  (`src/farkle/analysis/root_stability.py:139-150`). That label is applied separately to
  every strategy/root/`k` and across-`k` row (`root_stability.py:153-200,220-270`) with no
  multiplicity rule. The fast cross-seed artifacts contain this label 5, 11, 16, and 4
  times in the `k=2`, `k=4`, `k=5`, and across-`k` tables, respectively. The discrepancy
  artifact carries those classifications and reports 14 root-to-root classification
  changes. Sidecars otherwise describe these products as fixed-design reproducibility
  diagnostics.
- **Consequence:** Ordinary Monte Carlo intervals are converted into formal-looking
  significance claims before H2H and across a large selected family. This conflicts with
  the governing rule that tournament screening and robustness are descriptive and that
  formal finalist inference is H2H-only. The final structure report does not repeat this
  label, but the canonical cross-seed artifacts do.
- **Smallest reasonable remediation:** Remove statistical-significance classifications
  from root stability. Retain point estimates, MCSEs, practical-distance fields, and
  explicitly descriptive reproducibility flags. Do not repair this by adding a generic
  multiple-testing threshold; formal strategy inference belongs in H2H.

### F5 - RNG autocorrelation artifacts do not use the globally ordered sequence they claim

- **Severity:** Medium
- **Confidence:** High
- **Classification:** Confirmed diagnostic-method defect
- **Evidence:** The module promises diagnostics ordered by `game_seed`
  (`src/farkle/analysis/rng_diagnostics.py:1-7`). It sorts each Arrow batch separately,
  then drops `game_seed` (`rng_diagnostics.py:235-250`). The compact collector processes
  batches sequentially and, within each batch, feeds all P1 observations before all P2
  observations rather than merging seat exposures by seed
  (`rng_diagnostics.py:282-326`). Therefore its lagged sequence is neither globally
  seed-sorted nor even seed-sorted across seats.

  Independently sorting all 4,300 root-32/`k=2` exposures for strategy 2 by `game_seed`
  gives lag-1 win-indicator autocorrelation `-0.0248465686`. The produced
  `results_seed_32/analysis/05_rng/diagnostics/rng_diagnostics.parquet` reports
  `-0.005726...` for the same strategy, support, metric, and lag. The stored
  `diagnostic_band_lower/upper` are centered on the observed correlation
  (`rng_diagnostics.py:747-760`), although the sidecar calls them a reference band;
  the note's phrase “values inside the band” is consequently ambiguous because the
  estimate is always inside its own interval.
- **Consequence:** These values do not diagnose autocorrelation under their declared
  ordering, so they cannot even serve as reliable descriptive evidence about that
  sequence. Their explicit no-independence claim appropriately limits interpretation,
  but does not make a misordered estimator correct.
- **Smallest reasonable remediation:** Stream a globally merge-sorted coordinate order
  and merge seat exposures before updating group lag buffers, or define and publish a
  different exact order. Store either a zero-centered reference band or clearly label an
  estimate-centered approximate interval, with an explicit `zero_inside_interval` field.

### F6 - The TrueSkill “calibration” scores use an undocumented mu-softmax heuristic, not TrueSkill predictive probabilities

- **Severity:** Medium
- **Confidence:** High
- **Classification:** Confirmed method/claim mismatch
- **Evidence:** Held-out fitting correctly freezes ratings after the first 80% of games,
  but prediction converts only the fitted `mu` values to
  `softmax((mu-max(mu))/beta)` (`src/farkle/analysis/trueskill_screening.py:266-301`).
  It ignores every rating's `sigma` and does not use the TrueSkill performance-difference
  distribution. The resulting numbers feed log loss, Brier score, top probability, and
  `top_probability_calibration_gap` (`trueskill_screening.py:302-315`). The sidecar calls
  this `descriptive_replay_and_heldout_prediction`, and the artifact column calls the
  last quantity a calibration gap, without declaring the heuristic link.
- **Consequence:** The fast values (for example root-33/`k=2` held-out log loss
  `0.499292` and calibration gap `0.00746`) are valid scores for an arbitrary softmax
  ranking heuristic, not held-out predictive calibration of the fitted TrueSkill model.
  Tau-zero and reversed-order rank diagnostics remain legitimate descriptive replays,
  subject to F1's contaminated game outcomes.
- **Smallest reasonable remediation:** Either compute model-consistent predictive
  probabilities that incorporate both `mu` and `sigma` under an explicitly documented
  multiplayer approximation, or rename all outputs and sidecar methods as
  `mu_softmax_heuristic` and remove “calibration” claims.

### F7 - HGB statistical content is mostly sound, but active paths and sidecars misstate scope and quantity

- **Severity:** Medium
- **Confidence:** High
- **Classification:** Confirmed artifact-contract defect
- **Evidence:** HGB does hold out entire strategy configurations, evaluates MAE/R2 and
  permutation importance on held-out rows, retains fold/repeat variability, and labels
  importance `predictive_association_not_causal`
  (`src/farkle/analysis/run_hgb.py:256-373`). However, its generic writer hard-codes
  `weighted_quantity="win_rate"` for every output (`run_hgb.py:141-176`). It writes the
  concatenated per-`k` importance table under the physical `across_k` directory while
  declaring `scope="concat_ks"`, and preserves the retired active filename
  `feature_importance_long.parquet` (`run_hgb.py:651-694`). The produced sidecar confirms
  that path/scope disagreement; `feature_importance_overall.parquet` has an
  `association_importance_mean` schema but likewise claims weighted quantity `win_rate`.
  Missing player-count coordinates are still converted to `players=0`
  (`run_hgb.py:503-521`), preserving a retired sentinel path even though canonical fast
  inputs avoided it.
- **Consequence:** Consumers cannot infer statistical scope or estimand from location or
  sidecar, and malformed inputs can resurrect a forbidden cross-`k` sentinel. The model
  results themselves remain associative, out-of-sample diagnostics over the noisy finite
  simulated grid; they support no causal or universal strategy claim.
- **Smallest reasonable remediation:** Put row-preserving importance concatenation in
  `concat_ks`, use a non-retired name, pass artifact-specific quantities and methods to
  the writer, and reject missing/null/nonconfigured player counts. Assert physical scope,
  sidecar scope, table coordinates, and weighted quantity together.

### F8 - Margin “medians” are unlabeled bin midpoints and can be impossible game values

- **Severity:** Low
- **Confidence:** High
- **Classification:** Confirmed documentation/estimator defect
- **Evidence:** Margin values are accumulated in width-25 bins
  (`src/farkle/analysis/game_stats.py:97-109,1345-1364`) and quantiles return the selected
  bin midpoint (`game_stats.py:1383-1405`). The output nevertheless uses unqualified
  names `median_margin_runner_up` and `median_score_spread`
  (`game_stats.py:743-752`), while the sidecar only says `uncertainty_method: descriptive`
  and does not record bin width or approximation error. Farkle score margins are on the
  score lattice, yet the fast root-32/`k=2` artifact reports medians such as `3712.5` and
  `3662.5`, which cannot be observed margins.
- **Consequence:** Means and population SDs use exact sums and are not binned, but median
  columns look exact while carrying a systematic midpoint approximation. Equal-`k`
  summaries inherit it.
- **Smallest reasonable remediation:** Use an exact sparse integer histogram, or rename
  the fields as approximate binned quantiles and bind the width, representative rule,
  and worst-case error in the sidecar.

### F9 - Seat diagnostic sidecars erase the conditioning and pairing that define the outputs

- **Severity:** Low
- **Confidence:** High
- **Classification:** Confirmed provenance/documentation defect
- **Evidence:** Strategy-seat rows are conditional on strategy, seat, root, and `k`;
  self-play rows condition on all seats using the same strategy; mirrored rows pair
  opposite two-player orientations within deterministic batch queues
  (`src/farkle/analysis/seat_analysis.py:237-253,368-443`). The single writer nevertheless
  labels every seat artifact `conditioning="unconditional"` and
  `replication_unit="deterministic_shuffle_batch"`
  (`seat_analysis.py:451-486`). The fast self-play artifact is correctly empty because
  tournament shuffles use unique configurations. The mirrored artifact retains paired
  and both unpaired counts, but its sidecar does not state that pairs are queue-matched
  within batch rather than common-RNG game pairs.
- **Consequence:** The table schemas preserve most needed counts, and within-`k`/across-`k`
  formulas are correct, but a sidecar-only consumer can make the wrong population and
  pairing claim.
- **Smallest reasonable remediation:** Give each output its exact conditioning,
  observational unit, and pairing rule. Reserve “paired” for a declared matching design,
  or call the current statistic a within-batch count-matched orientation contrast.

### F10 - Single-root performance does not enforce the rectangular batch support its sidecars claim

- **Severity:** Low
- **Confidence:** High
- **Classification:** Confirmed latent validation defect; not triggered by the fast run
- **Evidence:** `_read_batch_metrics` rejects duplicates and nonpositive rows but does not
  require every strategy in every batch (`src/farkle/analysis/performance.py:60-88`).
  `_batch_arrays` silently fills missing strategy/batch cells with zero wins and zero
  exposures (`performance.py:261-281`), even though written sidecars declare
  `missing_cell_policy="fail"` (`performance.py:591-614`). Root stability correctly
  rejects the same nonrectangular condition (`src/farkle/analysis/root_stability.py:283-302`).
  The fast cells are rectangular: 80 strategies x 100 batches at every root/`k`.
- **Consequence:** A partial or malformed single-root batch artifact can yield a per-`k`
  estimate with reduced support and a joint resampling distribution with implicit zero
  cells rather than a hard failure.
- **Smallest reasonable remediation:** Validate identical batch IDs and exactly one
  positive-exposure row for every strategy/batch before any per-`k` estimate or pivot;
  add a missing-cell test parallel to root stability.

## Estimand and method inventory

| Important output | Target and observational/replication unit | Estimator / uncertainty | Assumptions, support, multiplicity, and permitted claim |
|---|---|---|---|
| Workload plan and Wilson fields | Resolution of each strategy's per-`k` binomial win proportion; player-game exposure, with one exposure per strategy per shuffle | Worst-case full 95% Wilson width chooses 4,265 shuffles, rounded to 4,300 = 100 contiguous batches of 43; observed Wilson intervals are recomputed from raw wins/exposures | Planning/resolution diagnostic only; no significance decision. Fast plans report achieved worst-case width `0.0298758`; all per-`k` rows meet `0.03`. |
| All-player batch metrics | Unconditional player-game performance/behavior within root/`k`/batch/strategy; player-game exposures | Exact turn-weighted return `sum(score)/sum(n_turns)`; exact game-weighted return `mean(score/n_turns)`; diagnostic rounds proxy `mean(score/n_rounds)` plus gap/mismatch and abort counts; no interval | Positive turns/rounds required. Abort exposures retained but included, causing F1. Raw sufficient statistics allow correct aggregation; these metrics are not promoted to a user-facing overall return report. |
| Per-`k` performance | Conditional strategy win probability over the configured tournament population and design; deterministic batch is the MC replication unit | `sum(wins)/sum(exposures)`, chance delta against `1/k`; MCSE `sd(batch wins/batch exposures, ddof=1)/sqrt(B)` and t interval with `df=B-1`; Wilson shown separately | Positive exposures required. No multiplicity or inferential winner claim. Fast `k=4` baseline is exactly `0.25`; every strategy has 4,300 exposures and 100 batches. |
| Across-`k` performance | Equal importance over configured `k={2,4,5}` of chance-adjusted rates | Complete-support arithmetic mean of per-`k` deltas; MCSE `sqrt(sum(mcse_k^2))/3`; normal interval | No exposure/game/batch/support weighting. Requires all configured `k`. Independent-`k` variance assumption is formula-consistent but violated by F2's seed collisions. Ordinary intervals are MC uncertainty, not post-selection inference. |
| Joint batch resampling and control contrasts | Stability of nonlinear ranks, exact-size top-N, practical-distance shortlist, and configured control contrasts | Within each `k`, resample whole batch rows and retain the full cross-strategy wins/exposure vector; independently resample distinct `k`; 2,000 deterministic coordinate-owned replicates | Complete strategy support; descriptive bootstrap distribution, no hypothesis-test or multiplicity claim. Fast controls are empty, so the contrast artifact correctly has zero rows. Ties at exact top-N are resolved by strategy ID but that convention is not stated in the sidecar. |
| Pareto, maximin, and descriptive screening | Multi-`k` partial-order robustness and worst-`k` performance on complete support | Exact nondominance over per-`k` chance deltas; maximin is maximum minimum delta; screening uses point-score order, configured practical bands, and resampling stability | Descriptive only. Pareto keeps exact ties. Maximin publishes one lowest-ID representative when maxima tie rather than the full co-leader set; this is a display convention that should be explicit. The final report correctly calls both descriptive. |
| Player-count effects | Relative odds versus the appropriate chance baseline, pairwise `k` contrast, within-`k` spread, and cross-`k` ordering agreement | Finite `logit(win_rate)-logit(1/k)`; Spearman/Kendall on common finite strategy support | Boundary rates 0/1 are unavailable, not continuity-corrected; root 32 has 13 such unavailable strategy/`k` rows. Rank correlations are ordering diagnostics, not magnitude agreement or causal effects. No multiplicity. |
| Seat effects and diagnostics | Within-`k` strategy-seat and population-seat win probability versus `1/k`; equal-`k` standardized effects on seats common to all `k`; separate exposure mixture; self-play P1 and count-matched reversed-orientation diagnostics | Raw wins/exposures; equal-`k` mean of within-`k` effects; separate exposure-weighted diagnostic | Complete common `k`/seat support. No uncertainty or multiplicity. Formulas and separate mixture are correct, but F1 contaminates wins and F9 mislabels conditioning/pairing. |
| Game lengths, margins, close/rare events | Per-game population summaries and strategy-conditioned player-game summaries; intended unit stated as game | Exact counts/sums for means and SDs; exact round histograms; binned margin quantiles; no interval | Winner duplication F3 invalidates strategy-conditioned and rare-event outputs; F8 affects median labels. Population game-length rows remain descriptive. |
| RNG and exact roll diagnostics | Lagged association under a declared row order; exact scorer distribution over all `6**d` ordered outcomes for `d=1..6` | Pearson lag autocorrelation with approximate `1.96/sqrt(n)` interval; exact enumeration has no uncertainty | RNG output explicitly cannot establish independence, but F5 violates its order. Roll enumeration is finite/exact and uses production scoring; no multiplicity or inferential claim. |
| TrueSkill | Sequential relative rating within each root/`k`; candidate contribution is mean within-cell `mu` percentile over every required root/`k` cell | Canonical per-cell TrueSkill update; descriptive percentile mean; tau-zero/order replays; held-out predictive scores | Raw `mu` is never averaged across cells; `sigma` remains model state, not SE. Pair contribution has all 80 strategies in all six root/`k` cells and independently reproduces. F1 affects inputs and F6 invalidates the calibration label. No formal inference. |
| HGB | Predictive association between strategy features and simulated per-`k` win rates on the finite grid; strategy configuration is the held-out unit | Five deterministic configuration folds; held-out MAE/R2; held-out permutation importance; fold and repeat variability retained | Out-of-sample and associative, not causal. Complete fast support is 80 configurations per `k`. It does not propagate MC measurement error in target rates, so performance describes prediction of this finite noisy run. F7 affects artifact contracts. |
| Two-root stability | Fixed-design reproducibility across roots 32 and 33, with combined performance as raw count combination within `k` before across-`k` aggregation | Root and combined ratio estimates; batch MCSE; raw/standardized discrepancies; joint max bootstrap, rank correlations/movement, top-N inclusion, shortlist changes, prefixes, and half drift | Exactly two fixed RNG domains and complete root/`k` support. No root-population t interval, random-effects variance, seed generalization, or heterogeneity output was found. Joint resampling preserves vectors. F2 compromises exact independence; F4 introduces forbidden significance labels. |
| Final report | Conditional description over the finite configured grid; combined two-root score is the tournament source | Point leaders, Pareto/maximin, stability, and separately selection-conditioned H2H results | Report says tournament leaders are descriptive, retains unresolved comparisons, and makes no unique-best claim in this fast run. It does not turn tournament intervals into post-selection inference. |

## Independent fast-artifact recalculation

The following checks read curated parquet rows and primitive batch columns directly and
did not call `build_canonical_performance`, `_estimate_one_k`, `_across_k_estimates`, or
the root-stability aggregation functions.

1. **Root 32, strategy 2:** Direct seat expansion gives, for `k=2/4/5`,
   wins/exposures `2502/4300`, `1307/4300`, and `998/4300`. Rates are
   `0.5818604651`, `0.3039534884`, and `0.2320930233`; subtracting chance baselines
   `0.5`, `0.25`, and `0.2` gives `0.0818604651`, `0.0539534884`, and
   `0.0320930233`. These exactly match the three `by_k/*/performance.parquet` rows.

2. **Batch MCSE:** Computing 100 batch ratios directly for those same rows gives
   MCSEs `0.007027300965`, `0.006496298948`, and `0.006349365944`. The production
   artifacts match. With `B=100`, the code uses `t_(0.975,99)` as required
   (`src/farkle/analysis/performance.py:108-112`). Every fast batch contains exactly
   43 exposures per strategy, so the aggregate ratio and unweighted mean batch rate
   coincide in this run.

3. **Across `k`:** The unweighted mean of the three chance deltas is
   `0.05596899224806202`. Propagation gives
   `sqrt((0.007027300965^2 + 0.006496298948^2 + 0.006349365944^2)/9)` =
   `0.003828247448616921`. Both equal the root-32
   `across_k/performance_equal_k.parquet` row. No exposure or support count enters.

4. **Two roots:** Summing root-32 and root-33 raw counts before division gives strategy
   2 combined rates `5058/8600`, `2656/8600`, and `1987/8600`; their chance deltas
   average to `0.059341085271317824`, exactly the `combined_roots` cross-seed row.
   Root-specific rows remain adjacent in all produced combination artifacts.

5. **Returns:** From root-32/`k=2` strategy-2 raw sufficient statistics, exact
   turn-weighted return is `483.02963205`, exact game-weighted return is
   `504.22379211`, the rounds proxy is `508.92612453`, and proxy-minus-exact is
   `4.70233242`; turn/round mismatch prevalence is `0.19534884`. This confirms both
   exact estimators use `n_turns` and the proxy uses `n_rounds`. The artifact retains
   zero abort exposures for this specific strategy/cell; other cells exhibit F1.

6. **TrueSkill contribution:** Independently ranking `mu` within each of the six
   root/`k` rating tables and averaging percentiles produces strategies 31, 11, and 71
   at `0.93125`, `0.895833...`, and `0.88125`, exactly matching
   `seed_pair_analysis/01_trueskill/across_k/candidate_percentile_contribution.parquet`.
   Every strategy has `rating_cells_present=rating_cells_required=6`.

## Separation of the three uncertainty products

The three required products remain computationally and structurally distinct:

1. **Wilson resolution:** `simulation_workload_plan.json` records the planning target
   and achieved worst-case full width; per-`k` performance exposes
   `wilson_interval_*`, `wilson_interval_width`, and `wilson_resolution_met`. Screening
   does not use Wilson bounds for significance or selection.
2. **Per-`k` Monte Carlo uncertainty:** the same per-`k` table separately exposes
   `batch_mcse` and `batch_interval_*`, computed from deterministic batch rates with a
   t critical value. Its sidecar's combined phrase `wilson_and_batch_t_interval` is less
   precise than separate method fields would be, but the columns and implementation are
   unambiguous.
3. **Analytic across-`k` uncertainty:** the across-`k` table exposes
   `equal_k_mcse`/`equal_k_interval_*`; its sidecar says
   `independent_k_variance_sum`, and implementation uses a normal critical value. The
   formula is correct for the equal-`k` estimator. F2 prevents accepting the declared
   independence assumption for the actual run.

Joint batch resampling is a fourth, descriptive nonlinear-stability product rather than
a replacement for any of the above. Produced sidecars call it
`joint_deterministic_batch_resampling`, and reports do not mislabel it as an ordinary
hypothesis test.

## Validation performed

- Independent scans/recalculations described above, including aborts, seed collisions,
  returns, game-stat observation identities, RNG autocorrelation, and TrueSkill cells.
- Focused test command:
  `.venv/Scripts/python -m pytest tests/unit/analysis/test_performance.py tests/unit/analysis/test_all_player_metrics.py tests/unit/analysis/test_seat_analysis.py tests/unit/analysis/test_root_stability.py tests/unit/analysis/test_trueskill_screening.py tests/unit/analysis/test_hgb_feat.py tests/unit/analysis/test_game_stats.py tests/unit/analysis/test_rng_diagnostics_branches.py -q`
- Result: **42 passed**. The pass is evidence of current behavior, not a rebuttal to the
  uncovered estimator, support, or artifact-contract defects.

## Final verdict

**Unsound within this review's scope.** The canonical performance arithmetic and most
method separation are sound in isolation, but the completed non-H2H evidence is not fit
for scientific interpretation because arbitrary abort winners contaminate core outcomes,
actual RNG construction violates uncertainty assumptions, and game-statistics outputs
systematically double-count winners. Medium and low findings further prevent several
diagnostic/model artifacts from supporting the claims their names and sidecars imply.
