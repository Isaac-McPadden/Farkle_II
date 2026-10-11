# Task 5B capability protocol, version 1

Frozen before the full seeded assessment, 2026-09-28. Parent contract:
`first_release_task0_v1`. Read [Task 5A](task5a_contract.md) and the executable
[recipe](../../configs/validation/task5b_capability.json). This protocol does
not implement production preflight or replace exhaustive diagnostics.

## Observable quantities and warning rule

Method `batch_lag_half_drift_descriptive_v1` evaluates complete, certified,
ordered root/k/strategy all-seat batch vectors. The six candidate responses are
W/A, S/A, W/C, D/C, D/L, and V/W. Completed loss rate adds no separate test.
For an ordered batch-ratio vector x of length B:

- Lag statistic: Pearson correlation of x[0:B-1] and x[1:B], centering the two
  lag segments separately. Warning if `abs(rho) > max(0.5, 4/sqrt(B-1))`.
- Drift statistic: `(mean(second half)-mean(first half))/SD(all B, ddof=1)`.
  The first half ends at floor(B/2). Warning if absolute drift exceeds 1.5.

These are intentionally conservative **descriptive effect thresholds**, not
normal-reference tests or calibrated error probabilities. The support term in
the lag threshold is a policy heuristic, not a confidence band. Threshold
equality does not warn. Values inside the threshold mean only that this check
did not detect a concern of its declared magnitude. Current 1.96/sqrt(n) band
excursions are not automatic failures. The thresholds never change in response
to the assessment outcomes; a revision requires a new recipe and assessment.

Only B=30 and B=100 are assessed by v1. Fewer than 30 batches, any other B,
any zero denominator, constant vectors, or undefined statistics yield
`insufficient_evidence`. No batch is dropped, no temporal gap is closed, and
no replacement value is imputed. Lag can be undefined while drift is defined;
retain separate results. Negative/nonfinite/malformed counts, missing or
duplicate/out-of-order batch IDs, failed A=C+S or C=W+L identities, and nonzero
conditional sums without support are deterministic integrity failures.
Upstream certification remains responsible for full game coordinates, roster,
actual seats, and total k-exposure conservation; a summary-only validator cannot
recover missing games from internally consistent forged summaries.

The proposed production cell selection is at most 16 evenly spaced canonical
strategy IDs from the frozen sorted roster, for every required root/k, seat=0.
For N>16 select index `floor(i*(N-1)/15)` for i=0..15; for N<=16 select all.
Selection is before reading outcomes. Within-strategy seats, unselected
strategies, higher lags, other ordering, and rare-event-tail dependence are not
covered. Task 5C must bind selection and its actual scope to source identities.

## Frozen assessment workload

The JSON recipe fixes 64 seeds (51001..51064), B in {30,100}, 16 synthetic
sentinel series, and six metrics per sentinel. Each fixture family has 96
series and 192 check opportunities. Each B/scenario has 64 families. There are
768 fixture units across six scenarios, with one bounded unit resident per
worker and at most two process workers, each with one native thread.

All randomness uses existing INDEXED_SEED purpose 1 and PCG64DXSM. Fixture
coordinates are `(root_seed=seed, game_index=sentinel, replicate_index=B)`;
unused components are zero. Clean/fault counterparts deliberately share their
Gaussian innovations. Four latent components generate the count vector: each
batch has A=10,000; S=clip(round(10+3*z1),0,A); C=A-S;
W=clip(round(C/2+100*z0),1,C-1); L=C-W;
D=L*max(0,round(1000+50*z2)); V=W*max(0,round(600+40*z3)).
Values are integer sufficient statistics with explicit conditioning. They are
synthetic strategy-level summaries, **not jointly simulated Farkle populations**.
They do not certify game-level k-exposure conservation or realistic performance
at all player counts. Their paired clean outputs test the declared arithmetic
and sensitivity under this model.

For each fault, inject only sentinel zero's four latent components; other
sentinels stay clean. Retain every statistic, threshold, support, and result,
including warnings in uninjected sentinels. Detection credit requires the
intended check to warn in the injected sentinel, not an unrelated false alert.

| Fixture | Construction / strength | Required support | Expected response / blind spot |
| --- | --- | --- | --- |
| Clean | Independent standard Gaussian latent innovations | B=30,100; positive denominators | Count all family/check warnings; no universal no-fault guarantee |
| Positive serial | Stationary AR(1), coefficient +0.85, 200-step burn-in | Same | Lag warning in injected sentinel |
| Negative serial | AR(1), coefficient -0.85, same burn-in | Same | Absolute lag warning; drift need not warn |
| Half drift | Add -2 latent SD in first half, +2 in second; 4-SD total step | Same | Half-drift warning in injected sentinel |
| Weak serial | AR(1), coefficient +0.2 | Same | Record detection/misses, no power promise |
| Weak drift | Total step 0.5 latent SD | Same | Record detection/misses, no power promise |
| Integrity faults | Missing/duplicate/reordered batches; negative/fractional counts; broken completion or win/loss sums; unsupported nonzero margins | Frozen expected inventory | Explicit failure; statistical labels cannot substitute |
| Short/degenerate support | B=29; B=31 outside v1 evidence; zero-denominator batch; constant vector | Exact known construction | Insufficient evidence, not pass |
| Hidden within-batch dependence | Same 100 batch win totals, 1,000 observations/batch; randomized order versus all zeros then all ones | Identical batch inputs | Identical batch results despite strong game-level lag association |
| Shared root fault | Copy the same step-contaminated summaries into two nominal roots | Identical root summaries | Between-root difference zero; both within-root monitors warn |

The within-batch example uses configured seed 59001; the common-root example
uses 59002. The latter is intentional shared corruption, not a claim that the
underlying generators independently produced identical datasets. Existing
certified-publication fault tests separately verify missing-game, duplicate/hole,
and wrong-roster failure; their checks complement this summary-level matrix.

## Acceptance criteria fixed before assessment

At the primary B=100, each strong fault scenario must trigger its intended
check in at least 58 of 64 injected sentinels (at least 90%). At most 4 of 64
clean families may have any warning (a bounded fixture acceptance limit of
6.25%, **not an estimated/calibrated population guarantee**). B=30 results and
both weak-fault results are fully disclosed, without a minimum detection claim.
All nine integrity injections must fail explicitly, all four support cases
must withhold appropriately, the hidden within-batch example must expose the
blind spot, and the common-root example must contradict an independence claim
based solely on root agreement. Failure means `capability_gate_failed` and
requires a revised claim/recipe or a disclosed limitation; running is not enough.

Report clean any-warning counts/rates at the family level and total check
opportunities, plus scenario-specific intended-check detections and misses.
The metrics share numerators/denominators and checks share series; these are not
independent Bernoulli trials. Do not present 12,288 clean check opportunities as
12,288 independent calibration replications. There are 64 seeded clean families
per B under one bounded model. Across production roots/k the number of families
is larger, and no dataset-wide family-wise guarantee is claimed. This is separate
from, and conveys none of, H2H's 2% superiority error control.

## Evidence, recovery, and application boundary

Run `.venv/Scripts/python.exe scripts/task5b_capability.py` from the repository.
Output belongs to recipe path `data/task5b-capability-v1`. The plan binds recipe,
source hashes, clean/dirty code identity, RNG scheme, Python/NumPy, and platform.
Every seed/B/scenario is an atomic JSON unit with a payload hash and plan hash;
the summary inventories exact unit paths/hashes. Compatible runs validate/reuse
units; absent/corrupt units are recomputed individually; a changed identity
requires explicit `--force`. `--workers 1` permits serial replay of the same
scientific plan. Never pool validation outputs into production inference.

This standalone evidence graph is traceable capability evidence, not PE/MC/MR,
a production completion, or execution of Task 5A's all-k correctness matrix.
Task 5C must authenticate/validate applicability and require a compatible
successful capability record before emitting substantive statistical labels.
The arithmetic helper alone cannot grant that applicability.

Warnings must record the statistic, threshold, support, and coverage and surface
an investigation message. They do not add games/replicates, change candidates,
stop statistical sampling, suppress honest uncertainty limitations, or make
an independence claim. Current production behavior remains unchanged in 5B.
