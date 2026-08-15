# Codex Statistical Methods Map

This is a review index, not proof. Verify formulas and sidecars in the named
modules and use hand-checkable tests before accepting a claim.

## Workload and RNG

- `simulation/workload_planner.py`: smallest workload satisfying maximum full
  Wilson width, rounded to exactly 100 equal contiguous batches with at least
  30 shuffles each.
- `utils/random.py`: scheme-2 `SeedSequence` namespaces and explicit
  `PCG64DXSM`; tournament and H2H seat streams derive directly from full-width
  semantic coordinates, never scalar fingerprints, worker order, or draw history.
- `simulation/run_tournament.py`: immutable root/k/shuffle-index ownership,
  block-level recovery by semantic coordinate, and explicit
  attempted/completed/safety-limit counters.

## Returns and performance

- `analysis/all_player_metrics.py`: k/schema-aware byte-bounded Arrow projection
  with stable source-row/seat accumulation over all attempted player-game
  exposures. A safety-limit attempt counts as a loss for every participant; explicit
  completed/safety-limit support; exact turn- and game-denominated returns;
  rounds proxy, mismatch prevalence, and maximum-round abort exposures. A
  zero-turn/zero-round safety attempt has zero game-weighted return and no
  turn-weighted return; completed games still require positive denominators.
- `analysis/performance.py`: compact memory-mapped root/k count matrices feed
  primary per-attempt win rates relative to `1/k`, a
  labelled completed-only diagnostic, per-attempt Wilson checks and batch MCSE
  `s_batch/sqrt(B)`, complete-support equal-k performance, declared
  alternatives, controls, Pareto, maximin, and joint batch-vector resampling.
  Bootstrap RNG is owned by root/k/replicate coordinates; fixed replicate-ID
  ranges run in parallel, publish atomic result shards, resume by authenticated
  range, and reduce in global replicate order before final publication.
- Player-count diagnostics include finite chance-relative log odds, pairwise k
  contrasts, within-k spread, and cross-k Spearman/Kendall agreement. Boundary
  logits are unavailable.
- `analysis/screening.py`: descriptive evidence only. Screening does not imply
  equality, final tiers, or unique-best status, and consumes the canonical
  all-attempt performance conditioning.

## Seat, game, and RNG diagnostics

- `analysis/seat_analysis.py`: attempted/completed/safety-limit exposures and
  wins by root, batch, strategy, k, and seat; per-attempt within-k effects
  relative to `1/k`; common-support cross-k standardization; separate exposure
  mixture and self-play outputs; completed-only mirrored effects.
- `analysis/game_stats.py`: attempted-game lengths and multi-target events,
  completed-only winner/margin products, and explicit observational units.
  Strategy-conditioned counts use only canonical `P<seat>_strategy` columns.
- `analysis/rng_diagnostics.py`: method-v4 exact eligibility and lag association
  through stable authenticated external partitions. Strategy groups use seat
  exposures ordered by the RNG-v2 tournament-player coordinate and report win
  and rounds. Matchup groups use one game per sorted participant-ID multiset,
  ordered by the game coordinate, and report rounds only. Their compact digest
  controls partitioning and display, while the full sorted participant vector
  controls identity, membership, merge, and grouping. Fixed max-lag rings are
  allocated only after eligibility. Cap selection uses semantic priority;
  exhaustion is `blocked_by_cap`. Zero-centered descriptive references use
  lagged-pair support and make no independence conclusion.
- `analysis/roll_enumeration.py`: exact enumeration of all `6**d` ordered
  outcomes for dice counts one through six.

## TrueSkill and HGB

- `analysis/run_trueskill.py`: sequential canonical ratings per root/k only.
- `analysis/trueskill_screening.py`: complete-cell percentile-rank candidate
  contribution plus tau-zero and order diagnostics. Held-out descriptive scores
  use `mu_softmax_heuristic = softmax(mu / beta)`, explicitly ignore sigma, and
  are not TrueSkill predictive probabilities.
- TrueSkill sigma is model state, not a sampling standard error and not a
  cross-k uncertainty input.
- `analysis/run_hgb.py`: held-out strategy-configuration predictions and
  permutation association importance with fold variability and finite-grid
  support. Full-grid fits may propose only future simulation rows.

## Two-root stability

- `analysis/root_stability.py`: combine total wins and exposures within k, then
  compute cross-k estimates. Retain root estimates, raw/standardized
  discrepancies, joint diagnostics, rank correlations, 95th-percentile rank
  movement, bootstrap top-N inclusion by root, convergence, and half drift.
- Interpretation is fixed-design reproducibility across two RNG domains. No
  root-population interval, heterogeneity model, significance/rejection
  classification, or multiple-testing inference is permitted. MCSE and
  intervals describe Monte Carlo precision; practical threshold positions and
  joint-bootstrap reference exceedances remain descriptive. Half-drift retains
  raw differences but leaves MCSE and standardized drift unavailable when a
  half contains only one deterministic batch.

## H2H

- `analysis/candidate_family.py`: top 75 from canonical performance and
  TrueSkill, plus protected controls/diagnostics; simultaneous balanced-tail
  contraction; complete provenance and family hash.
- `analysis/h2h_schedule.py`: Bonferroni planning threshold, exhaustive exact
  first-crossing allocation for the implemented two-proportion score rejection
  rule over every locked seat scenario, and equal root/order completed support.
  Power is conditional on reaching completed support. Blocks bind family and
  schedule hashes to stable RNG coordinates; cap authorization/progress is
  mutable state outside the byte-immutable plan and manifest.
- `analysis/h2h_inference.py`: `d_AB = 0.5(q_AB-q_BA)`, constrained-null score
  test, score-inversion intervals, Holm decisions, simultaneous practical
  bounds, optional equivalence, and labelled root diagnostics.
- `analysis/dominance.py`: separate practical/statistical graphs, SCC cycles,
  condensation fronts, noninferential within-front order, and direct practical
  dominance over all finalists for unique best.

## Reporting and agreement

- `analysis/structure_agreement.py`: method overlap, common-population rank
  agreement, admission counts, selection-conditioned H2H agreement, and root
  H2H stability.
- `analysis/structure_reporting.py`: sidecar-validated JSON/Markdown/plot
  reports. The reporting completion binds the migration inventory as an output
  and reports TrueSkill's completed-game/safety-limit conditioning explicitly.
  Claim language is constrained by support, unresolved comparisons, cycles,
  configured equivalence, and direct-dominance evidence.

## Primary oracles

- `tests/unit/analysis/test_performance.py`
- `tests/unit/analysis/test_safety_limit_root_analysis.py`
- `tests/unit/analysis/test_root_stability.py`
- `tests/unit/analysis/test_h2h_schedule.py`
- `tests/unit/analysis/test_h2h_inference.py`
- `tests/unit/analysis/test_dominance.py`
- `tests/integration/test_structure_toy_oracle.py`
- `tests/integration/test_simulation_to_report_oracle.py`
