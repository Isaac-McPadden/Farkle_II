# Adversarial post-remediation review: simulation data correctness

## Scope, evidence, and verdict

Reviewed commit `6be5f5fa11df77155621bfc81188c7515f38f8de` and the completed
`configs/fast_config.yaml` run at `data/results_seed_pair_32_33`. This review
traced tournament and H2H data from strategy assignment through game execution,
row production, checkpointing, ingest, curation, concatenation, and sufficient-
statistic aggregation. It did not modify source, tests, configuration, or
existing artifacts. Temporary probes ran in memory or under the system
temporary directory.

The pipeline health file says `complete_success`, and 254 focused game,
scoring, simulation, resume, all-player, H2H-schedule, ingest, and integration
tests passed. Those facts show that the implemented path runs. They do not make
the generated outcomes valid.

**Verdict: unsound within this review's scope.** Normal completed games are
represented consistently, row identity survives the canonical data path, and
the published sufficient statistics reproduce exactly. However, maximum-round
aborts are converted into ordinary P1 wins. The fast tournament contains 2,981
such false wins, and H2H contains at least six finalist comparisons whose
47,376 nominal games are all 0-0 safety-cap aborts credited to seat 1. H2H then
reports very narrow intervals from zero completed games and discards the abort
state needed to repair the result. Separate RNG-collision and metric-only
resume defects also violate the claimed simulation design.

## Implementation trace

1. `generate_strategy_grid` builds distinct numeric strategy IDs, and the
   tournament worker permutes the whole 80-strategy grid once per shuffle
   (`src/farkle/simulation/simulation.py:48-214` and
   `src/farkle/simulation/run_tournament.py:174-212`). Contiguous groups of
   `k` permuted strategies become tables; consequently every strategy has one
   exposure per shuffle and no table should repeat a strategy.
2. Each table obtains a coordinate-derived game seed, and `_play_game` creates
   one `FarklePlayer` per seat, runs `FarkleGame.play`, requires one rank-1
   player, and flattens all players' statistics
   (`src/farkle/simulation/simulation.py:325-413`). Dice are generated in
   `[1, 6]`; scoring, farkles, hot dice, banking, and exact `n_turns` are updated
   in `FarklePlayer` (`src/farkle/game/engine.py:75-264`).
3. `FarkleGame.play` performs normal rounds, gives every non-trigger player a
   closing turn, and copies score, rolls, farkles, exact turns, rank, margin,
   heuristic counts, and maximum-round status into `PlayerStats`
   (`src/farkle/game/engine.py:408-480`). Earlier seats can therefore have
   `n_rounds + 1` turns after a later-seat trigger; `n_turns`, not `n_rounds`,
   captures the exact denominator.
4. One immutable row shard is written per shuffle. Its manifest records root,
   k, shuffle, deterministic batch, RNG version, path, and row count
   (`src/farkle/simulation/run_tournament.py:310-400`). Parent-process
   aggregation records winner counts and winner-conditioned auxiliary sums;
   checkpoints are atomically pickled (`src/farkle/simulation/run_tournament.py:422-440,1073-1227`).
5. Ingest requires complete manifest support, restores shuffle-index order,
   normalizes strategy IDs, derives `winner_strategy` and `seat_ranks`, and
   casts to the canonical schema (`src/farkle/analysis/ingest.py:106-207,214-322,361-512`).
   Curate is a sidecar-bound byte copy. Combine pads only later-seat columns and
   performs a streaming row/value/order comparison for both per-k partitions
   and `concat_ks` (`src/farkle/analysis/curate.py:170-209` and
   `src/farkle/analysis/combine.py:26-101,146-325`).
6. All-player metrics iterate every seat, not only winners, and retain raw
   exposures, wins, final scores, exact turns, per-game exact returns, rounds
   proxies, abort exposure, and behavior sums by deterministic batch
   (`src/farkle/analysis/all_player_metrics.py:115-238,241-299`).
7. H2H freezes one row per pair/root/order block, swaps A/B seats by order,
   derives RNG from `(root, k=2, pair, order, game)`, and reduces each block to
   seat-1 and seat-2 win counts (`src/farkle/analysis/h2h_schedule.py:460-502,768-830`).
   This coordinate and allocation wiring is correct. Its handling of aborted
   games is not.

Four direct replays from stored coordinates reproduced every raw row field:
root 32/k=2 shuffle/game `(0,0)`, root 32/k=4 `(0,0)`, root 32/k=5 `(0,0)`,
and the root 32/k=2 abort at `(2,33)`. The last replay ended after 200 rounds at
0-0 and nevertheless returned `winner_seat=P1`, directly demonstrating the
primary defect below.

## Conservation and reconstruction checks

### Tournament rows

| root | k | rows | maximum-round aborts | non-aborted games | recorded wins | player exposures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2 | 172,000 | 1,479 | 170,521 | 172,000 | 344,000 |
| 32 | 4 | 86,000 | 3 | 85,997 | 86,000 | 344,000 |
| 32 | 5 | 68,800 | 0 | 68,800 | 68,800 | 344,000 |
| 33 | 2 | 172,000 | 1,498 | 170,502 | 172,000 | 344,000 |
| 33 | 4 | 86,000 | 1 | 85,999 | 86,000 | 344,000 |
| 33 | 5 | 68,800 | 0 | 68,800 | 68,800 | 344,000 |

The following identities or invariants passed for all 653,600 tournament rows:

- 4,300 shuffles and 80 strategies occur in every root/k cell. Every strategy
  has exactly 4,300 exposures; every table has `k` distinct strategies; every
  shuffle exposes all 80 strategies exactly once; and game indices are exactly
  `0..games_per_shuffle-1`.
- `(root_seed, k, shuffle_index, game_index)` is unique. Winner labels are
  valid seats, `winner_strategy` matches the winning seat, the winning score is
  the row maximum, ranks are a permutation of `1..k`, `seat_ranks` agrees with
  them, and loss margins reproduce `winning_score - player_score`.
- Seat exposures sum to player exposures. Each seat has one exposure per game;
  per-strategy seat counts fluctuate as expected under random permutation but
  sum to 4,300. For example, root 32/k=2 per-strategy P1 exposure ranges from
  2,078 to 2,210 and P2 from 2,090 to 2,222. Seating is randomized rather than
  exactly mirrored in the tournament; the recorded seat fields are sufficient
  for the intended seat diagnostics.
- Every non-aborted game's exact turns have the required final-round pattern:
  a prefix of seats has `n_rounds + 1`, the remaining seats have `n_rounds`,
  and at most `k-1` seats receive the extra turn. All aborted games have exactly
  `n_rounds` turns for every seat. Across the two roots, exact turns differ from
  rounds in 164,601 k=2 games, 125,609 k=4 games, and 107,082 k=5 games, so the
  exact counter materially captures terminating-seat asymmetry.
- Scores are nonnegative multiples of 50, turns are positive, rolls are at
  least turns, and farkles do not exceed turns. All score ties are exactly the
  2,981 maximum-round aborts; no normally completed game tied.
- Zero-score and losing exposures survive. Root 32 alone contains 34,809,
  35,368, and 35,609 zero-score exposures for k=2/4/5. The only zero-score
  exposures not labelled losses are the false P1 abort winners.

The raw manifests contain 4,300 entries and the exact planned row count in
every cell. First/middle/last raw shards in every cell matched ingest values
after the documented type cast. Each ingest parquet is byte-identical to its
curated parquet. Independent hashes over root/k/shuffle/game plus winner fields
match curated rows, the normalized per-k partition, and the corresponding
`concat_ks` segment in all six cells. For example, the root 32/k=2 hash is
`26cd7a0ee4103c06c71010564f1a4f33eee128be325c7551dfc4a81a18d3d603`
at all three layers. Thus `concat_ks` is row-preserving in this run.

### Sufficient statistics and completed checkpoints

All six `all_player_batch_metrics.parquet` files contain the expected 8,000
rows (`100 batches x 80 strategies`) and exactly 43 player exposures in every
batch/strategy cell. An independent exposure-level recalculation found no
mismatch in any raw count, sum, square sum, or behavior field. Every published
turn-weighted, game-weighted exact, and rounds-proxy return also reproduced from
its recorded sufficient statistics.

As concrete totals, root 32/k=2 records 344,000 exposures, 172,000 wins,
2,958 abort exposures, final-score sum 2,773,621,550, turns sum 7,486,409,
farkles sum 3,519,533, and rolls sum 21,903,022 in
`results_seed_32/analysis/03_metrics/by_k/2p/all_player_batch_metrics.parquet`.
The abort exposure count is exactly `2 x 1,479`; analogous `k x abort_games`
identities hold in every cell.

Completed fast-run pickle checkpoints exactly reproduce row-derived wins and
all winner-conditioned sums/square sums, list every shuffle index, and agree
with each published `{k}p_metrics.parquet`. This establishes equality for the
completed row-producing execution, but not for the metric-only interruption
path in finding S3.

### H2H blocks

The schedule and `root_order_counts.parquet` each contain 11,400 unique
`(pair_id, root_seed, order)` and block IDs. Every one of 2,850 pairs has four
blocks, each root/order cell contains 5,625,900 scheduled games, and
`wins_seat1 + wins_seat2 == games_completed == games_required` for all blocks.
Seven sampled immutable block files exactly matched their aggregate rows. These
are file/count conservation results; finding S1 shows why they are not outcome-
validity results.

## Findings

### S1 - Maximum-round aborts become false P1 wins, and H2H erases the state while manufacturing precision

- **Severity:** Blocker
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** On reaching `max_rounds`, `FarkleGame.play` sets
  `max_rounds_hit` but still sorts scores, calls the first stable-sort element
  the winner, assigns unique sequential ranks, and emits a normal `GameMetrics`
  (`src/farkle/game/engine.py:425-480`). `_play_game` then insists on and emits
  exactly one rank-1 winner (`src/farkle/simulation/simulation.py:385-413`). The
  tournament immediately increments that strategy's win and winner-conditioned
  sums (`src/farkle/simulation/run_tournament.py:229-238`), while all-player
  metrics separately mark the same exposure as both a win and an abort
  (`src/farkle/analysis/all_player_metrics.py:177-190`). Performance sums those
  wins and exposures without excluding aborts
  (`src/farkle/analysis/performance.py:101-106`).

  The fast rows contain 2,981 maximum-round aborts, all tied; recorded wins are
  653,600 while non-aborted games are only 650,619. Stable sorting assigns every
  abort to P1. This is not merely an unusual but resolved game result.

  H2H is worse: `_winner_seat_counts` accepts only P1/P2 and `_simulate_block`
  retains only their counts, with no completion/abort/tie field
  (`src/farkle/analysis/h2h_schedule.py:768-830`). Candidate pairs `(0,1)`,
  `(0,20)`, `(0,21)`, `(1,20)`, `(1,21)`, and `(20,21)` have 24 root/order
  blocks and 47,376 games in
  `seed_pair_analysis/04_h2h_execute/h2h_2p/root_order_counts.parquet`; every
  row says `wins_seat1=1974`, `wins_seat2=0`. Replaying game index 0 from all 24
  immutable coordinates produced 24 0-0, 200-round aborts with both
  `P#_hit_max_rounds=True`. Tournament rows independently show that all observed
  meetings among these four strategies abort.

  Nevertheless,
  `seed_pair_analysis/05_h2h_inference/h2h_2p/pairwise_inference.parquet`
  reports `q_ab=q_ba=1`, `d_ab=0`, and simultaneous intervals approximately
  `[-0.002544, 0.002544]` for all six pairs. Those intervals use 7,896 nominal
  wins per pair but zero demonstrated completed games.
- **Consequence:** Tournament chance-relative performance, batch MCSE inputs,
  seat effects, screening/candidate evidence, and winner-conditioned summaries
  include arbitrary outcomes. Formal H2H treats unresolved non-games as
  independent Bernoulli results and can present extreme false precision; if
  equivalence were enabled, these rows could satisfy it. Because H2H discards
  abort counts, existing aggregate outputs cannot be corrected without replay.
- **Smallest reasonable remediation:** Make game completion and termination
  reason explicit. A maximum-round/tied abort must not receive a winner or
  rank-1 exposure. Retain abort/tie counts in tournament and H2H sufficient
  statistics, exclude them from completed-game win denominators, and make H2H
  comparisons with no completed support unresolved/incomparable. If power
  requires a fixed number of completed games, schedule deterministic replacement
  coordinates and record attempted as well as completed counts. Rebuild all
  affected tournament, family, H2H, inference, and report artifacts. Add engine,
  accumulator, and H2H tests for all-zero safety-cap games.

### S2 - Full game coordinates are narrowed to 32-bit seeds, producing repeated player RNG streams across batches

- **Severity:** High
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** Tournament game coordinates are converted to
  `coordinate_seed(..., dtype=np.uint32)`
  (`src/farkle/simulation/run_tournament.py:192-201`). `_play_game` then uses
  that truncated value as the root from which every seat RNG is derived
  (`src/farkle/simulation/simulation.py:325-384`), despite the RNG utility being
  able to consume the complete 64-bit semantic coordinates directly
  (`src/farkle/utils/random.py:63-150`). The canonical fast rows contain 12
  duplicate excess `game_seed` values: root/k 32/2: 2, 32/4: 2, 32/5: 0,
  33/2: 5, 33/4: 1, 33/5: 2. Every collision crosses deterministic batches.
  Root 32/k=2 seed `2963478802`, for example, appears at
  shuffle/game/batch `(194,18,4)` and `(4052,4,94)`.
- **Consequence:** Nominally distinct games reuse all seat RNG streams. The
  deterministic batches used as Monte Carlo replication units are therefore
  not composed of disjoint coordinate streams. Collision risk grows
  quadratically with workload; the small fast-run count does not bound the
  production consequence. Shuffle identity is narrowed similarly at
  `src/farkle/simulation/run_tournament.py:635-648`, creating a future resume-
  identity risk even though no shuffle collision occurred here.
- **Smallest reasonable remediation:** Derive each player's generator directly
  from the full tournament coordinate plus seat, rather than using a generated
  integer as a new root. At minimum use a non-truncated identity. Version this
  RNG-contract change, invalidate affected simulations, and add large-coordinate
  uniqueness tests for both game and shuffle recovery identities.

### S3 - Metric-only interruption/resume silently loses pre-interruption sufficient statistics

- **Severity:** High
- **Confidence:** High
- **Classification:** Confirmed defect; the completed fast run is protected by
  its row manifests, but the allowed metric-only path is not
- **Evidence:** Parent processing updates wins immediately but stores new metric
  payloads only in `collected_metric_chunks`
  (`src/farkle/simulation/run_tournament.py:1083-1143`). It marks shuffle/block
  coordinates complete and periodically checkpoints the old `metric_sums`
  before reducing those collected payloads
  (`src/farkle/simulation/run_tournament.py:1166-1199`). Reduction occurs only
  after the processing loop finishes. On resume without row or metric-chunk
  manifests, completed coordinates are skipped, so their sums cannot be
  recovered (`src/farkle/simulation/run_tournament.py:883-1012`).

  A temporary two-shuffle focused probe interrupted after the first persisted
  block with `collect_metrics=True`, no row directory, and no metric-chunk
  directory. The checkpoint contained one win and completed shuffle `[0]` but
  winning-score sum 0. After resume it contained two wins and score sum 11,000;
  an uninterrupted run of the identical coordinates contained two wins and
  score sum 21,050. The completed coordinate lists were identical.
- **Consequence:** A documented resumable configuration can silently publish
  correct win totals with incomplete means, variances, and winner-conditioned
  sums after interruption. Completion and checkpoint metadata do not expose
  the discrepancy.
- **Smallest reasonable remediation:** Merge each returned block's sums into
  the checkpoint state before marking its coordinates complete and before the
  atomic checkpoint write, or require/persist a replayable metric chunk before
  completion ownership advances. Add an interruption/resume oracle comparing
  every logical checkpoint field with an uninterrupted run when both optional
  shard directories are disabled.

### S4 - Canonical ingest validates shard containers but not row identity or gameplay invariants

- **Severity:** Medium
- **Confidence:** High
- **Classification:** Plausible corruption risk / hardening gap; no violation
  was observed in the fast artifacts
- **Evidence:** `_canonical_row_shards` validates one manifest record per
  shuffle, path uniqueness, reported row counts, root/k metadata, and batch
  arithmetic (`src/farkle/analysis/ingest.py:106-179`). `_iter_shards` checks
  file existence, schema extras, and row count, but not that rows inside the
  file carry the manifest's root/k/shuffle/batch or that game indices are unique
  and complete (`src/farkle/analysis/ingest.py:182-207`). During normalization,
  every missing canonical column is silently added as nullable `NA`
  (`src/farkle/analysis/ingest.py:422-445`). `_fix_winner` derives columns but
  does not validate winner-seat range, rank uniqueness, winner score, distinct
  strategies, or margins (`src/farkle/analysis/ingest.py:214-261`).
- **Consequence:** A duplicated row, swapped shard, wrong internal coordinate,
  missing game key, or internally inconsistent winner can receive a valid
  sidecar and flow into curation/concatenation. Some later consumers fail on
  missing root/k/batch or turns, but shuffle/game identity and several outcome
  inconsistencies are never universally required. Counts can therefore be
  duplicated or joined under false provenance without an ingest failure.
- **Smallest reasonable remediation:** Stream-validate each shard against its
  manifest: constant root/k/shuffle/batch, exact game-index support, unique
  global row keys, valid winner seat/strategy/score, one rank 1, rank/margin
  consistency, and `k` distinct strategies. Reject missing required fields
  instead of padding identity/outcome columns; reserve padding for later-seat
  columns only in `concat_ks`. Add malformed-shard integration tests.

### S5 - Roll-limit enforcement is off by one, and the canonical narrow types have no explicit range guard

- **Severity:** Low
- **Confidence:** High for the off-by-one; medium for future type-capacity impact
- **Classification:** Confirmed edge defect plus hardening suggestion
- **Evidence:** `ROLL_LIMIT` is documented as the maximum permitted rolls and
  set to 1,000, but `take_turn` checks `rolls_this_turn > ROLL_LIMIT` before the
  next roll (`src/farkle/game/engine.py:33-34,227-236`). A continuing turn can
  therefore execute roll 1,001 before the following loop raises. Canonical
  `n_rounds`, rolls, farkles, highest turn, heuristic counts, hot dice, and
  exact turns are mostly `int16` (`src/farkle/utils/schema_helpers.py:15-46`),
  while game/turn logic does not explicitly prove all such totals fit 32,767.
  The fast-run maxima are safe (maximum rolls 1,075 across a whole game,
  highest turn 13,000, rounds/turns 200), and Arrow casting did not overflow.
- **Consequence:** The named roll cap is not the implemented cap. A future
  legal but unusually long completed game can also fail ingest during type
  narrowing rather than producing a controlled validation error; silent
  overflow was not observed.
- **Smallest reasonable remediation:** Change the pre-roll check to `>=`, add
  an exact boundary test, and either widen accumulative seat counters to
  `int32` or validate documented upper bounds before Arrow conversion.

## Qualifications and passed controls

- Scoring lookup covers straights, three pairs, two triplets, four-kind plus
  pair, n-of-a-kind, and single ones/fives before Smart discard logic
  (`src/farkle/game/scoring_lookup.py:27-172` and
  `src/farkle/game/scoring.py:548-692`). Focused scoring/property tests passed,
  and artifact scores obeyed the 50-point lattice. Final rows do not retain
  individual dice rolls or move choices, so produced artifacts alone cannot
  independently replay every legal-move decision; source/tests and coordinate
  replays are the available evidence.
- All-player rather than winner-only accumulation is implemented correctly for
  the data it receives. Losses, ordinary zero scores, exact turns, rounds,
  farkles, rolls, behavior counters, and abort exposures are preserved. The
  defect is that aborted rows also contain fabricated winners.
- The correct `1/k` chance baseline appears in every per-k performance artifact
  (for example root 32/k=2 has baseline 0.5, 172,000 raw wins, and 344,000 raw
  exposures). Player-count wiring is therefore correct downstream of the
  invalid abort labeling.
- H2H schedule identity, root/order balance, seat swapping, immutable block
  publication, and block-to-aggregate conservation all passed. H2H does not
  reuse the tournament permutation schedule. Its unsoundness here is outcome
  termination and insufficient statistics, not pair/root/order wiring.
- No duplicate canonical row keys, missing required values, negative counters,
  out-of-range ranks, strategy repeats within games, or concat joins were found
  in the completed fast artifacts.

The findings do not judge fast-run strategy effectiveness, rank stability, or
production precision. They establish whether the analysis received valid game
outcomes and sufficient statistics. For maximum-round and affected H2H cells,
it did not.

## Final verdict

**Unsound within this review's scope.** The completed fast artifacts are
internally well conserved, but maximum-round games are converted into false
wins, the affected H2H aggregates discard abort state, tournament RNG streams
can collide after 32-bit seed narrowing, and an allowed metric-only resume can
silently omit pre-interruption sums. The simulation data cannot support the
intended downstream claims until those defects are remediated and the affected
artifacts are regenerated.
