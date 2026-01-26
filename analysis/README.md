# Analysis Notes

| Metric | Definition | Symmetry expectation |
| --- | --- | --- |
| `win_rate` / `win_prob` | Fraction of games a strategy wins within a player-count cohort. `win_prob` is written alongside metrics as an alias of `win_rate` for downstream features that expect a probability column. | For symmetric matchups, swapping seats should not change the probability; `win_rate` and `win_prob` should therefore agree when computed from seat-agnostic aggregates. |
| `P1_win_rate` / `P2_win_rate` | Seat-adjusted head-to-head win rates derived from aggregated pairwise results after merging mirrored `(A,B)`/`(B,A)` matchups. | `P1_win_rate + P2_win_rate = 1` for decisive pairs; mirroring the strategies should leave the rates unchanged because results are collapsed by unordered matchup. |

## CLI flags

The `farkle-analyze` entrypoint wires the ingest → curate → combine → metrics → analytics pipeline. Helpful toggles:

- `--game-stats` – run `farkle.analysis.game_stats` after `metrics` (or after `combine` when running that step directly), producing `game_length.parquet`, `margin_stats.parquet`, and `rare_events.parquet` with matching `*.done.json` stamps.
- `--margin-thresholds <ints...>` – override the close-margin cutoffs used by game stats (defaults: `500 1000`).
- `--rare-event-target <int>` – override the score threshold used to flag multi-target rare events (default: `10000`).
- `--rare-event-margin-quantile <float>` – derive the rare-event margin threshold from the margin-of-victory distribution (e.g., `0.001` for the bottom 0.1%).
- `--rare-event-target-rate <float>` – derive the multi-target score threshold from the second-highest score distribution (e.g., `1e-4` for roughly 0.01% of games).

Interseed-only toggles (ignored when `--per-seed-only` is set or `analysis.run_interseed` is `false`):

- `--rng-diagnostics` – run `farkle.analysis.rng_diagnostics` after `combine`, writing `rng_diagnostics.parquet` with a `rng_diagnostics.done.json` freshness stamp.
- `--rng-lags <int>` (repeatable) – provide one or more lag values for RNG autocorrelation diagnostics (defaults to `1`).

To shrink `rare_events.parquet`, favor smaller quantiles/rates (for example `--rare-event-margin-quantile 1e-3` or `--rare-event-target-rate 1e-4`) so fewer games are flagged in the rare-event output.

All pipeline outputs are written atomically with skip-if-fresh stamps so repeat runs are quick when inputs have not changed.
