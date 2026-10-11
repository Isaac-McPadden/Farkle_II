# First scientific release: Task 0 implementation contract

Date: 2026-09-24. Contract ID: `first_release_task0_v1`.
Source inspected: `7db67ccc94a143ebd16c335563513e89a84a62aa`.

Task 0 is complete as a design/source-reconciliation deliverable. The products
below are specifications for later tasks, **not implemented runtime features**.
No simulation targets, production configuration, RNG streams, executable stage
order, or existing artifact contracts change in this task. This document is the
implementation handoff for the updated September 24 backlog, not the historical
remediation tasks that also use numbers 0–10.

Read with [question coverage and workbook notes](task0_question_map.md) and
[artifact, schema, and publication specifications](task0_artifacts.md).

## 1. Source reconciliation and task ownership

The checkout matches the revision inspected by the backlog and September 8
report. No intervening source implementation can be credited. The pre-existing
untracked `docs/project_status_2026-09-08.md` is left untouched.

| Backlog task | Verified existing foundation | Remaining implementation |
| --- | --- | --- |
| 0 | Current source, config, workbook, and review reconciled here | Design complete; later tasks must implement and validate it |
| 1–2 | [All-player metrics](../../src/farkle/analysis/all_player_metrics.py) retain moments; [engine](../../src/farkle/game/engine.py) and [row schema](../../src/farkle/utils/schema_helpers.py) retain scores, official ranks, winner, turns, and termination | Exact score/rank/margin frequencies, support certificate, batch sufficient statistics, summaries, durable publication |
| 3 | [Performance](../../src/farkle/analysis/performance.py) has attempted-exposure Wilson/batch intervals and joint resampling; completed win rate is a point diagnostic | Completed-outcome intervals specified in section 3; existing attempted intervals cannot substitute |
| 4 | [H2H inference](../../src/farkle/analysis/h2h_inference.py) has simultaneous bounds and equivalence machinery | Mega config equivalence is null; exclusive decision chain hides equivalence after Holm rejection; add independent flags and downstream handling |
| 5 | [RNG diagnostics](../../src/farkle/analysis/rng_diagnostics.py), [root stability](../../src/farkle/analysis/root_stability.py), deterministic/recovery tests exist | Bounded preflight, demonstrated monitor capability, lightweight production checks, and independent optional exhaustive coverage |
| 6 | [H2H schedule](../../src/farkle/analysis/h2h_schedule.py) freezes the plan separately from mutable execution state | Interactive/noninteractive plan-specific authorization; refresh currently derives authorization from configured cap and restricts changes after progress |
| 7 | [Seat analysis](../../src/farkle/analysis/seat_analysis.py) has seat wins/exposures; all-player behavior omits seat; [HGB](../../src/farkle/analysis/run_hgb.py) has held-out predictions, importance, and fold metrics | Seat behavior, exact opportunity summary, matched response/nonadditivity products; no existing canonical interaction product found |
| 8 | [Structure reporting](../../src/farkle/analysis/structure_reporting.py) covers H2H/screening/agreement and claim constraints | Automatic ten-group guide, support/status dictionary, method-note resolutions, new outputs and diagnostic coverage |
| 9 | Shared streaming, partition, sidecar, and completion helpers exist | Seat builder still uses `pq.read_table(count_path).to_pandas()`; [metrics](../../src/farkle/analysis/metrics.py) still calls seat analysis with `force=True`; narrow these paths |
| 10 | Existing Windows launcher, scratch publication, integration/recovery oracles, final release audit | Qualify all additions together and refresh capacity; prior focused tests do not establish this |

Empty-output schema inference and the curate schema manifest's JSON-object
representation remain narrow follow-ups where touched (Task 9), not grounds for
rewriting the artifact framework. Current configs retain 100 deterministic
batches, 2,000 screening bootstrap replicates, 95% screening intervals, H2H
target power 0.80, family alpha 0.02, and practical delta 0.03. These are checked
in [mega config](../../configs/farkle_mega_config.yaml), not hard-coded reporting
constants. Neither the historic capacity estimate nor this contract authorizes
a production run.

## 2. Observation, support, and denominator contract

The game key is `(root_seed, k, shuffle_index, game_index)`. A player exposure
adds the actual seat `1..k`. Strategy IDs use `STRATEGY_ID_ARROW_TYPE` (int32).
Two seats carrying the same strategy are two legitimate exposures, not a
duplicate. Padded columns for seats greater than k are not observations.
The deterministic batch is the simulation plan's
`shuffle_index // shuffles_per_batch`; I/O batch sizes cannot redefine it.

For a fixed root/k/strategy and either a specified seat or all seats, define:

| Symbol | Definition |
| --- | --- |
| A | All attempted player-game exposures |
| C | Completed exposures |
| S | Safety-limit exposures; `A = C + S` |
| W | Official completed wins |
| L | Completed losses; `C = W + L` |
| D | Sum of winner-score minus player-score over completed losses |
| V | Sum of winner-minus-runner-up margin over official wins |

All-attempt loss count is `A-W = L+S`. Completed win/loss rates are `W/C`
and `L/C`; attempted rates are `W/A` and `(A-W)/A`. Safety rate is `S/A`.
Only completed scores produce ranks/margins. A tied descriptive first rank
does not create a win; a tied official loser contributes a legitimate zero to D.
Scores at safety termination have their own frequencies and never become final
completed outcomes. Preserve official ranks; descriptive competition rank is
`1 + count(scores strictly greater)` and midrank adds `(tie_count-1)/2`.
Store twice the midrank as an integer.

**Represent verified zero-exposure cells explicitly. Treat missing required
batches, partitions, or records as missing support or validation failures;
never silently convert them to zero counts.**

Zero certification requires all of the following:

1. Resolve the expected roots, k values, strategy roster/multiplicities, shuffle
   ranges, games per shuffle, and deterministic batch ranges from authenticated
   current-run inputs and the workload plan. Do not infer expectations from
   whichever files or rows survived.
2. Authenticate every required upstream artifact/manifest and its completion.
   During the existing forward row scan validate exact game coordinates,
   uniqueness, contiguous expected coverage, and batch assignment. Counts alone
   cannot detect one missing coordinate replaced by a duplicate.
3. Validate all k actual seats and required fields, termination semantics, and
   roster membership. Check `A=k*attempted_games`, `C=k*completed_games`,
   `S=k*safety_games`, and `W=completed_games` after summing all strategies/seats.
   Check each strategy's all-seat exposure multiplicity against the resolved
   scheduled roster; never assume every unique strategy occurs just once.
4. Only after a complete batch is certified may an unobserved roster/seat cell
   be explicitly emitted with zero counts and `support_status=verified_zero`.
   Nonzero cells have `support_status=observed`. If a schedule requires positive
   exposure for a cell, zero is a failure, not a valid zero cell.
5. After publication, the dense batch support grid must contain every expected
   roster/seat cell exactly once. A missing row is corruption; consumers cannot
   recreate it with `fillna(0)` or `reindex(..., fill_value=0)`.

An absent *value bin* in an authenticated complete sparse frequency table means
zero observations at that value. An absent required partition or support cell
does not. A complete conditional distribution with zero support is undefined,
with an explicit support row and no positive frequency rows. Unsupported model
parameter combinations are `unestimable`, not zero observations or zero effect.

Any input or coverage failure blocks publication of successful completion and
of final estimates for that root/k. Valid checkpoints survive; failure evidence
may be reported separately. No computation silently drops bad batches or changes
the requested population. The fixture matrix in the artifact specification
makes these distinctions testable.

## 3. Frozen initial uncertainty specification

First-release additional intervals are limited to the following. Every summary
is separate by root, k, strategy, and seat view; `seat=0` denotes the all-seat
view, calculated by summing actual-seat batch numerators/denominators first.

| Summary ID | Estimator | Batch inputs | Conditioning |
| --- | --- | --- | --- |
| `completed_win_rate` | sum(W) / sum(C) | W, C | Completed exposures |
| `mean_deficit_completed` | sum(D) / sum(C) | D, C | Completed exposures, zero on wins |
| `mean_losing_margin` | sum(D) / sum(L) | D, L | Official completed losses |
| `mean_winning_margin` | sum(V) / sum(W) | V, W | Official completed wins |

`completed_loss_rate` and its interval are the complements of the completed
win estimate/interval, `[1-upper, 1-lower]`; no second resampling run is needed.
When `C>0,L=0`, deficit is zero but losing severity is undefined. When `C=0`,
all four completed-outcome estimates are undefined. The identity
`D/C = (L/C)*(D/L)` is checked whenever L is positive.

Use a **two-sided pointwise percentile bootstrap of deterministic batches**,
with confidence `cfg.screening.interval_confidence` and replicate count
`cfg.screening.bootstrap_replicates` (current production: 0.95 and 2,000).
Do not silently alter these public settings or automatically increase games or
replicates. This is approximate Monte Carlo precision conditional on the fixed
rules/grid/opponent design, not simultaneous coverage or a new ranking test.
Batch exchangeability/independence is a design assumption; a clean monitor does
not prove it. Investigate dependence warnings and qualify reported precision.

For replicate j, sample B indices with replacement from all B certified batches
within each root/k; do not discard batches with zero conditional events.
Use the same multiplicity vector for all strategies, seats, metrics, and
numerators/denominators in that root/k. Recompute ratios from summed resampled
numerators/denominators, never the unweighted mean of batch ratios. Independent
root/k streams do not pair batches just because they have the same integer ID.
No new pooled-root or cross-k outcome intervals are promised in this release.
Existing cross-k and root-stability estimators retain their own contracts.

In Task 3 add the unused permanent purpose `OUTCOME_BOOTSTRAP = 402` to
[RandomPurpose](../../src/farkle/utils/random.py), after rechecking for collision.
Use `(root_seed, k, replicate_index=j)` and defaults of zero for unused
coordinates, PCG64DXSM, j in `[0,R)`. Do not change existing namespace values or
simulation RNG identity. This allocation is a **future** method requirement,
not a namespace implemented by Task 0. Config seeds own every random draw.

Bootstrap endpoints use linear interpolation at `(1-confidence)/2` and
`1-(1-confidence)/2` over sorted replicate estimates (NumPy `method='linear'`).
Report bootstrap SD with `ddof=1` as approximate MCSE, valid/invalid replicate
counts, total B, contributing batches, conditional support, and interval width.
Use these conservative publication rules, which are disclosure policies rather
than a theorem of coverage:

- Missing/invalid input: fail the unit; never compute an interval.
- Zero denominator in the original estimate: null estimate/interval/MCSE,
  `no_conditional_support` (or `no_completed_support` for C).
- Fewer than 30 certified batches or fewer than 30 batches with a positive
  denominator: retain the estimate, withhold interval/MCSE, `insufficient_batches`.
- R below 2,000: bounded integration output is permitted, but withhold the new
  interval/MCSE and record `insufficient_replicates`; do not silently override R.
- Any zero-denominator replicate: do not redraw, drop it, or condition on valid
  replicates; withhold interval/MCSE and record `undefined_resamples` plus counts.
- A constant empirical replicate distribution: retain the estimate and observed
  range, withhold interval/MCSE, `degenerate_resamples`. In particular, observing
  no wins/losses does not prove that the true event probability is zero.
- Otherwise publish the interval/MCSE with `ok`. These checks do not guarantee
  adequate tail or rare-event precision; the report must state this limit.

No tail-quantile, frequency-bin, behavior, rank, or interaction intervals are
added initially. Publish descriptive means and quantiles q=.05,.25,.50,.75,.95
from exact frequency counts; define `Q(q)=min{x:F(x)>=q}` (inverse empirical CDF,
no interpolation). Empty support gives null. Thus **batch-indexed histograms
are not required canonical outputs** for this version: compact batch sums/counts
suffice for all selected intervals. Future quantile intervals require a new
retention/method contract before implementation, not independent draws from a
pooled histogram. Exact frequencies mean exact empirical counts, not known
population probabilities.

Implementation references: [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
describes percentile intervals and shared-index resampling;
[NumPy quantile](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)
defines interpolation conventions. The cluster choice and support policies
above are project decisions, not claims supplied by those API references.

## 4. Task 7 retained data and definite outputs

### Seat behavior and opportunity

Read existing compatible curated rows once per planned source slice. Retain
separate completed and safety-limit exposure counts, score/turn/round sums,
game-wise score/turn and score/round sums, mismatch counts, and terminal behavior
totals keyed by root/k/batch/strategy/seat/termination. Required actual-seat
fields cannot be null. Counts and integer sums are int64 with checked overflow;
ratio sums are float64 accumulated in canonical row/seat order.

Primary behavior tables use completed games. Also publish stopped behavior as
safety-conditioned diagnostics; all-attempt sums combine both explicitly and
must reconcile with compatible existing all-player aggregates. Zero-turn safety
exposures with zero score contribute zero to the all-attempt mean-of-game-ratios
under the existing convention, contribute zero turns to ratio-of-sums, and have
their count displayed. Positive score with zero turns, or any completed exposure
with nonpositive turns/rounds, fails validation. Never divide an unavailable
value by an invented positive denominator.

Publish score/actual-turn as both ratio-of-sums and mean-of-game-ratios; score per
round is a labeled proxy with mean gap and mismatch prevalence. Publish rolls,
farkles, hot dice, and smart-one/five use counts per actual turn; also show
their per-exposure means. Highest-turn is a per-exposure mean, not a count rate.
Smart dice counts additionally have dice per recorded heuristic use (null when
use count is zero). Table labels say these are terminal descriptive associations,
not state-level risk or causal mechanisms. No new behavior confidence intervals.

Freeze the requested global opportunity summary as follows:

- Per root/k, on completed games, publish `mean_turn_range` = mean over games of
  `max_s(n_turns_s)-min_s(n_turns_s)`, in turns. Retain its batch sum and number
  of completed **games**, counted once regardless of repeated strategies.
- Publish per-seat mean actual turns per completed game and seat turn share
  `sum(turns_s)/sum(all seat turns)`, with completed-game/turn support.
- Publish `turn_share_imbalance = 0.5*sum_s(abs(share_s-1/k))` per root/k.
  This is a descriptive opportunity measure, not a seat-adjusted win estimator.
- The headline overall value is the **equal-k mean of mean_turn_range**, separately
  for each root, using every configured k. Missing a required k fails; a valid
  k with no completed games makes the overall value unavailable. Publish the
  explicit weights `1/len(configured_ks)`; no exposure-weighted substitution and
  no pooling the two roots. Show the per-k measures beside it.

### Supported parameter responses and A89 nonadditivity

Use authenticated per-k `performance.win_rate` (all-attempt target, matching
HGB), canonical strategy metadata, and held-out predictions/fold metrics.
Do not condition this target on completion or silently replace it with the new
completed-outcome rates. Work separately by root/k, using every configured
strategy with valid required inputs; missing required predictions/performance
is a failure, not permission to alter common support.
An authenticated HGB result explicitly declaring insufficient fitting support
may yield unavailable prediction columns; observed supported contrasts remain
available. Distinguish that declared condition from lost promised output rows.

One-parameter response tables cover the ten existing HGB `FEATURE_SPECS` in
their declared order. Use valid active values only, sorted numerically. Retain
at most 25 values per parameter: all when there are <=25, otherwise indices
`floor(i*(m-1)/24)` for i=0..24. For each parameter, hold all other canonical
attributes fixed and use backgrounds supporting **all selected levels**;
average equally across these matched backgrounds at every level. If none exist,
emit an explicit unestimable support record. Do not silently switch to marginal
averages over different backgrounds. Retain all endpoints for inspection.

Two-parameter tables are restricted prospectively to these three pairs:

1. `score_threshold`, `dice_threshold` (both consideration flags active).
2. `auto_hot_dice`, `run_up_score`.
3. `smart_five`, `auto_hot_dice`.

Use the same level-selection rule and adjacent selected-level rectangles. For
each rectangle require all four configurations at identical other attributes.
Use a common set of matched backgrounds for its four corners, equally weighted.
The primary probability-scale contrast is
`p(a1,b1)-p(a1,b0)-p(a0,b1)+p(a0,b0)`, averaged over those backgrounds.
Report all four observed means and counts, not just the contrast. Different
rectangles may have different support; publish support identities to expose this.
There are at most 250 one-way level rows and 1,728 pair-contrast rows per root/k;
support records and bounded matched-ID inventories are additional, explicitly
counted artifacts. No data-dependent search for the largest interaction.

Use manifest attributes and canonical strategy rules, not parsing display names
or inventing settings. Inactive threshold sentinels are not numerical levels.
Do not count equivalent inactive settings as independent strategies/backgrounds.
Use the generator's canonical representation; unrecognized strategy types get
an explicit unsupported-type record. `smart_one=True` requires `smart_five=True`;
`require_both=True` requires both consideration flags. Missing legal quartets
are unestimable, not zero interaction. Current constraints are in
[strategies.py](../../src/farkle/simulation/strategies.py).

Primary contrasts use **observed simulation estimates**. Out-of-fold predicted
means are shown separately with fold membership; never subtract predictions
from different fitted folds and label the result an interaction within one
model. No additional model fitting is required for this product. A future
single-model interaction would require all corners evaluated by the same model
and explicit training/held-out status. Weak predictive scores limit interpretation.
All responses/contrasts are exploratory finite-grid associations; no causal,
family-wise significance, or universal nonlinearity claim is made. Support tables
must distinguish this limited three-pair exploration from an exhaustive search.

## 5. Dependency and completion handoff

- [x] Current implementation/configuration reconciled with all backlog tasks.
- [x] Every workbook question and substantive B/C note mapped to output/exclusion.
- [x] Estimands, support semantics, paired units, interval methods, and retained
  inputs frozen before Tasks 1–2.
- [x] Artifact schemas, canonical path expressions, durability, and acceptance
  fixtures specified in the companion artifact contract.
- [x] Seat behavior, opportunity weights, and bounded A89 outputs made definite.

Next is **Task 9**, then Tasks 1–2; Task 3 implements section 3. Task 7 may share
the validated source scan and support inventory, but has its own authenticated
child completion and cannot invalidate valid distribution units merely because
its consumer is interrupted. Tasks 4–6 can proceed independently after reading
their source anchors. Task 5 still owns the actual monitor workload, statistical
alert recipe, and measured capability; Task 0 freezes its result envelope, not
an unvalidated diagnostic. Task 8 generates the guide; Task 10 qualifies the
integrated implementation and capacity. No later task is marked complete here.

Scope remains: no broad opponent-composition experiment, fixed-matchup entropy,
state/event instrumentation, prescribed strategy rankings, automatic extra games,
polished figure suite, broad authentication redesign, or H2H storage optimization.
The H2H equivalence tolerance remains the planned 0.03; planned detection power
must be read from configuration and never described as power to establish
practical dominance/equivalence. Code/config freshness rules remain in force.

## 6. Source records and validation boundary

Inputs were read from `S:/Libraries/Downloads`; SHA256 identities:

| File | SHA256 |
| --- | --- |
| `Farkle_II_Implementation_Todos (1).md` | `427a6a70579e632886360f16eec038c66e6379a0115d9bbd30a5617fdc23be80` |
| `Farkle Questions.xlsx` | `5bdecfbd895ce62e3047abcc3bcee52f9b5f3028c7e7a6c017461b3286d3502c` |
| `project_status_2026-09-08.md` | `a901c455e93dfa18de546dd4419bd4a638af1266ef65a501cc5fafab04ac3282` |

The conversation's explicit zero-versus-missing correction is incorporated in
section 2. The source attachments are unchanged. Validation for this task checks
question/note coverage, source references, canonical scope/path expressions,
configuration assumptions, schema consistency, and hand-calculated semantic
examples. This is design validation, not evidence of implemented producers,
calibrated statistical coverage, a new 39-test run, or production readiness.

Validation performed on 2026-09-24 using the repository Python 3.12 `.venv`:

- Parsed the workbook's XML read-only and checked exact set equality with the
  coverage table: **62 question/scoring-rate targets, no duplicates or omissions**;
  all **10 substantive B/C note cells** have explicit resolutions.
- Verified the three attachment hashes, all relative source/document links,
  referenced existing AppConfig getter names, the declared canonical stage/scope
  combinations, current configuration values, and that RNG namespace 402 is unused.
- Checked hand calculations: competition/midranks for 10000/8000/8000/5000;
  batch counts C=(2,4,2), W=(2,1,0), D=(0,600,200) give mean deficit 100 and
  conditional losing mean 160 (an unweighted mean of nonempty batch means would
  incorrectly give 150); shared-index resampling preserves ratio-of-sums.
- Checked missing-coordinate and same-count duplicate/hole examples; additive,
  nonadditive, and mixed-fold-offset contrast arithmetic; the actual strategy
  constructor's forbidden smart-one/five combination; and level/output bounds.
- Checked Markdown whitespace with `git diff --check` and a direct scan of the
  new files. No runtime source/config edits or production execution were made;
  a full software test/static suite is not evidence required for this design-only
  change. Future implementation tests are listed separately, not claimed here.
