# RNG diagnostic method-v4 migration

Method v4 replaces the global SQLite copy/sort and in-memory matchup-strategy
accumulator graph with a deterministic external-partition pipeline.

## Statistical semantics

- Strategy diagnostics retain ordered seat exposures keyed by
  `(strategy_id, k)`, with Pearson lag correlations for `win_indicator` and
  `n_rounds`.
- Matchup diagnostics now use one ordered game per canonical sorted
  participant-ID multiset and `k`, and report `n_rounds` only. The former
  `matchup × strategy × k` expansion repeated game lengths and redundantly
  represented participant identity.
- Structural eligibility is `observations >= min(requested_lag) + 2`, the least
  support capable of a Pearson correlation at one requested lag. Larger lags
  may report `insufficient_pairs`; constant sequences report `zero_variance`.
- Zero-centered `1.96/sqrt(lagged_pairs)` bands remain descriptive references,
  not tests of independence.

## Bounded execution and lifecycle

Projected fixed-width columns are scanned in byte-bounded batches. Source row
groups route compact integer records into stable hash partitions and receive
hash-bound unit stamps. Eligibility partitions perform exact external
sort/reduction before lag state exists. A deterministic semantic priority
selects eligible matchups under the configured cap while retaining only a
bounded frontier in memory.

The 64-bit BLAKE2 matchup digest is a compact deterministic partition and
display value, not a complete identity. Every count, eligibility, selection,
observation, merge, and output grouping comparison also carries the full
canonical key: `k` plus the sorted participant-ID multiset padded to the fixed
stage width. Digest equality therefore cannot merge distinct matchups, while
different seat orders of the same matchup aggregate into one group.

The second route contains selected observations only. Each independent result
partition externally sorts its own observations, then processes one group at a
time with online moments and a fixed NumPy ring per applicable metric up to the
maximum lag. The shared resource policy caps processes by the central logical-
CPU budget and per-stage scheduling estimates, limits native threads, and uses
the configured process-tree warning, explicit aggregate hard limit, and host-
available reserve. Those execution values do not alter diagnostic estimands or
freshness.

Completed partitions are reusable across crashes, schedules, and worker counts.
A final manifest is published only after every planned unit validates. Final
Parquet/JSON artifacts have canonical authenticated-v3 sidecars, and stage
completion is published last. Arrow readers and external-merge memmaps are
closed explicitly before worker temporary directories are removed; if cleanup
also fails during exception unwinding, the processing exception remains the
reported failure.

## Output and completion migration

Former `summary_level="matchup_strategy"` rows become
`summary_level="matchup"`. Their `strategy` is null; `matchup_id` is a compact
stable integer, `participant_strategy_ids` records the canonical participants,
and `matchup` is a display label built only in bounded result partitions. Every
row adds `estimability_status`.

`rng_diagnostics_summary.json` reports candidate and eligible counts,
observation-count bins, exclusions by reason, deterministic cap exclusions,
status counts by lag and metric, validated partitions, and sampled peak RSS.
Cap exhaustion is `blocked_by_cap`, never successful completion. If no group is
structurally eligible, the diagnostic summary is `not_estimable`; this is a
valid complete calculation because no planned eligible group was discarded.
