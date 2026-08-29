# Analysis Notes

This directory contains analysis notes and supporting material. For normal
project usage, prefer the installed `farkle` CLI. The checked-in fast profile
is a two-root integration workflow:

```bash
farkle --config configs/fast_config.yaml two-seed-pipeline
```

## Metric notes

| Metric | Definition | Symmetry expectation |
| --- | --- | --- |
| `win_rate` / `win_prob` | Fraction of games a strategy wins within a player-count cohort. `win_prob` is written alongside metrics as an alias of `win_rate` for downstream consumers that expect a probability column. | For symmetric matchups, swapping seats should not change the probability. |
| `P1_win_rate` / `P2_win_rate` | Seat-adjusted head-to-head win rates derived after collapsing mirrored `(A, B)` and `(B, A)` matchups. | For decisive pairs, `P1_win_rate + P2_win_rate = 1`. |

## Recommended CLI flags

Use these on the installed CLI with a configuration that owns exactly one root:

- `farkle --config <single-root-config.yaml> analyze metrics --compute-game-stats`
  Compute pooled metrics and then run `game_stats`.
- `farkle --config <single-root-config.yaml> analyze pipeline --rng-diagnostics --rng-lags 1 2 4`
  Run preprocessing plus the canonical single-root analytics and H2H tail,
  including RNG diagnostics.
- `farkle --config <single-root-config.yaml> analyze analytics --rng-diagnostics`
  Run the canonical analytics and H2H tail when preprocessing artifacts already
  exist.

Shared analysis overrides:

- `--margin-thresholds <ints...>`
- `--rare-event-target <int>`
- `--rare-event-margin-quantile <float>`
- `--rare-event-target-rate <float>`

## Root ownership

`run` and `analyze` require a configuration with exactly one root in
`sim.seed_list`. Two-root work uses `two-seed-pipeline`, which creates two owned
root contexts and one pair context. Root-local stages write beneath each root's
analysis directory; root-stability and H2H pair stages write beneath
`seed_pair_analysis`.

Stage layout is configuration-dependent. Resolve paths from `AppConfig` or the
authenticated run context rather than assuming fixed folder numbers.

## Output behavior

All pipeline outputs are written atomically and guarded by manifest and
`.done.json` metadata. Repeat runs should therefore skip completed work when
inputs and stage-scoped config hashes are unchanged.
