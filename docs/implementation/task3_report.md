# Task 3: achieved precision at the fixed game targets

Implemented against the September 24 `first_release_task0_v1` contract and
Task 3 in `S:/Libraries/Downloads/Farkle_II_Implementation_Todos (1).md`.
Existing uncommitted Task 9/1A/1B/2 work is preserved. No production config,
game target, simulation rule, existing RNG coordinate, or H2H decision changes.

## Published precision

The metrics child `outcome_precision` publishes
`cfg.outcome_precision_path(k)` (`metrics/by_k/<k>p/outcome_precision.parquet`).
Its explicit Arrow schema is the frozen U schema. Every roster strategy has
five rows for each actual seat and seat=0:

| Summary | Retained calculation |
| --- | --- |
| Completed win rate | sum(W)/sum(C) |
| Completed loss rate | Complement of completed win, including `[1-high, 1-low]` |
| Mean completed deficit | sum(D)/sum(C), with wins contributing zero |
| Mean losing margin | sum(D)/sum(L) |
| Mean winning margin | sum(V)/sum(W) |

The existing performance intervals target attempted exposures; its
`win_rate_given_completion` is a point diagnostic. Those intervals cannot supply the
new completed-loss interval. Existing attempted-performance and H2H outputs
remain available under their original conditioning and multiplicity contracts.
Overall views sum actual-seat vectors before calculating ratios. Repeated
strategy appearances remain separate player exposures. Safety attempts enter
neither completed denominators nor conditional win/loss denominators.

The implementation uses configured `screening.interval_confidence` and
`screening.bootstrap_replicates`; neither setting is increased. For each
root/k/replicate, `OUTCOME_BOOTSTRAP=402` generates B replacement indices with
PCG64DXSM and the existing coordinate helper. All strategies, seats, numerators,
denominators and metrics in that root/k use the same multiplicities. All B
certified batches remain eligible for sampling, including zero-event batches.
Root/k streams remain separate. No root-pooled or cross-k outcome interval is
introduced. RNG scheme version and existing purpose values are unchanged.

Endpoints are NumPy linear percentile quantiles; MCSE is bootstrap SD with
ddof=1. U records support, contributing/total batches, requested/valid/invalid
replicates, observed replicate range, interval width, and method/status.
The frozen precedence is enforced: verified zero exposure; no completed or
conditional support; fewer than 30 total or contributing batches; fewer than
2,000 requested replicates; any undefined replicate; constant replicate
distribution; otherwise `ok`. Withheld CI/MCSE fields are null, never `[0,0]`.
Skipped resampling records zero valid/invalid replicates while retaining R.
An executed undefined draw is retained and counted, never redrawn or dropped.
The loss row mirrors the win row's status/counts and preserves its width/MCSE.

## Durable execution and authentication

The producer requires authenticated outcome publication, workload and roster
support. It validates the exact dense B grid, inventory totals, schemas,
nonnegative checked int64 counts, and independent S totals. A missing row,
duplicate replacing another row, invalid zero certificate, or missing batch
fails rather than generating a zero observation.

`cfg.outcome_precision_ranges_dir(k)` owns the immutable `plan.json` and
three existing-partition-executor products:

- `vectors/units/tile-*.parquet`: bounded aligned B vectors, including seat=0.
- `units/range-<start>-<stop>/tile-*.parquet`: exact half-open replicate ranges
  of at most 100, with typed nullable estimates and explicit skipped cells.
- `summaries/units/tile-*.parquet`: final per-tile U reductions.

Each product has an authenticated manifest and adjacent typed unit stamps.
`publication.json` inventories every unit and identity. Validation checks that
the inventory matches the immutable plan's full tile/range coverage. Final U
is published only after complete validation. Metrics completion enumerates U,
the plan and publication inventory. Graph snapshots and byte-deep release
audits retain the referenced units, excluding quarantine and execution scratch.
Metrics cache-key version advances from 4 to 5; its shared method registry adds
`outcome_precision_method_version=1`.

Projected Parquet reads, vector tiles, range arrays, row buffers and footer/plan
metadata have explicit admission against the analysis byte allowance. Final
quantiles read one cell's R values from disposable disk scratch; no resident
R-by-B-by-all-strategy-by-all-seat array is allocated. Source B row spans use
intersecting row groups with bounded decoding. Tile/range jobs use the existing
process policy and nested-pool suppression. Scratch follows configured staging,
including the existing atomic publication bridge.

Valid children survive interruption, shard/sidecar corruption elsewhere, and
worker-count changes. Complete logical results can be reused with a smaller
byte allowance. Unfinished work retains its frozen layout; a smaller allowance
requires explicit force to repartition. Repartition preserves logical results.
Repository-wide configuration freshness still applies: in particular, changing
`io.temp_dir` after upstream publication currently makes those sources stale.

## Evidence

The combined outcome, metrics, registry, release-audit, graph, partition,
completion and RNG suite passed **184 tests in 146.03 seconds**. A subsequent
focused zero-denominator-draw test passed, bringing the distinct focused checks
to **185**, including **18 Task 3 tests**. Tests cover a direct sampled-index
oracle for all 2,000 replicates, unequal batch denominators, zero-loss batches,
shared seat/metric draws, complement arithmetic, policy precedence, missing
replicates/support, re-signed grid corruption, scratch interruption, smaller
tile repartition, actual Windows-spawn repair, and graph/audit rejection of a
corrupt range. After the full-workflow oracle caught missing top-level JSON
consistency-column declarations, those declarations and a focused regression
assertion were added. All **18 Task 3 tests passed again in 95.06 seconds** on
that corrected source. The statistical oracle reconstructs everything from retained
batch vectors without reading game rows.

`python -m scripts.benchmark_outcome_precision --output <path>` reproduces the
finite-fixture measurement in [task3_memory.json](task3_memory.json). It uses
120 strategies, k=12, 30 batches, 1,200 games and 2,000 replicates. Before
measurement it declares a 16 MiB analysis allowance and 128 MiB incremental RSS
measurement allowance. Source/outcome preparation is excluded from timing.
The final-source precision publication used about **28.5 MiB incremental RSS**,
**90.41 seconds**, and **4.75 seconds** for authenticated reuse. Its 7,800 U rows contain 480 `ok`,
5,880 `insufficient_batches`, and 1,440 `no_conditional_support` statuses.
Retained checkpoints occupy about 5.34 MiB. Source/script hashes accompany the
measurement. These are fixture results, not a production-capacity approval.

Repository Ruff and Mypy over all 90 source files pass. Targeted Black/Pyright
cover all 12 Python files changed in this task. Full repository Black/Pyright
and the entire test suite are not claimed.

The corrected real simulation-to-report integration test passed: both roots,
pair H2H/reporting, artifact/statistical oracles, exact completion inventories,
final byte-deep release audit, and byte-stable ordinary replay. The workflow
measured **235.979 seconds**; pytest reported **1 passed in 239.95 seconds**.
The existing 240-second workflow assertion was unchanged and passed. This has
little timing margin and is not a production
capacity claim. The initial attempt failed the independent oracle's JSON
consistency-column check; it is not counted as passing evidence.

Integration used the isolated clean snapshot
`C:/Users/imcpa/AppData/Local/Temp/farkle-task3-verify-pnt4cw48`, commit
`6a5b1374a41e94128f25d266d6e4c4b1368befec`. All eight Task 3 production modules
match the working copy byte-for-byte. This qualification snapshot did not
commit, reset, or change the working branch. Final benchmark source identities
also match the delivered source. Summary evidence is retained in
[task3_validation.json](task3_validation.json).

## Interpretation and next dependency

These are approximate **pointwise Monte Carlo precision** intervals conditional
on the fixed rules/grid/opponent design, with batch exchangeability and
independence assumed. They do not provide simultaneous coverage across U,
new ranking tests, dependence validation, or adequate rare-tail precision merely
because the publication policy passes. No frequency-bin, quantile, behavior,
rank, interaction, pooled-root, or cross-k outcome intervals were added.

Keep four quantities separate: H2H planned detection power under its stated
alternatives (`planned_target_power_conditional_on_completed_support` and
`planned_worst_scenario_power_conditional_on_completed_support`), attained
completed support, observed interval precision, and actual H2H inferential
decisions. Task 3 never estimates post-hoc "achieved power" from observed
effects. Its sparse-support statuses do not request additional games. Existing
family-wise-controlled H2H claims retain their own method and multiplicity rules.

Next is Task 4's independent H2H equivalence/superiority flags and reporting.
The automatic cross-product guide remains Task 8; joint release qualification
and refreshed production capacity remain Task 10. No production run or historical
artifact migration was performed.
