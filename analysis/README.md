# Analysis Notes

This directory contains analysis notes and supporting material. For normal
project usage, prefer the installed `farkle` CLI:

```bash
farkle --config configs/fast_config.yaml analyze pipeline
```

The module entry point `python -m farkle.analysis.pipeline` still exists, but
it is a legacy/module-facing interface rather than the packaged front door.

## Metric notes

| Metric | Definition | Symmetry expectation |
| --- | --- | --- |
| `win_rate` / `win_prob` | Fraction of games a strategy wins within a player-count cohort. `win_prob` is written alongside metrics as an alias of `win_rate` for downstream consumers that expect a probability column. | For symmetric matchups, swapping seats should not change the probability. |
| `P1_win_rate` / `P2_win_rate` | Seat-adjusted head-to-head win rates derived after collapsing mirrored `(A, B)` and `(B, A)` matchups. | For decisive pairs, `P1_win_rate + P2_win_rate = 1`. |

## Recommended CLI flags

Use these on the installed CLI:

- `farkle analyze metrics --compute-game-stats`
  Compute pooled metrics and then run `game_stats`.
- `farkle analyze pipeline --rng-diagnostics --rng-lags 1 2 4`
  Run the full pipeline and include interseed RNG diagnostics.
- `farkle analyze pipeline --allow-missing-upstream`
  Allow downstream analytics to skip missing mandatory inputs during manual
  debugging.
- `farkle analyze variance --force`
  Recompute variance outputs even when the done-stamp looks fresh.

Shared analysis overrides:

- `--margin-thresholds <ints...>`
- `--rare-event-target <int>`
- `--rare-event-margin-quantile <float>`
- `--rare-event-target-rate <float>`

## Interseed notes

Interseed outputs depend on having either:

- `sim.seed_list` with two seeds, or
- `io.interseed_input_dir` pointing at an upstream analysis root.

When RNG diagnostics are disabled, the interseed stage layout is renumbered.
Do not assume that `00_rng` always exists. Resolve paths from `AppConfig` or
inspect `analysis/config.resolved.yaml`.

## Output behavior

All pipeline outputs are written atomically and guarded by manifest and
`.done.json` metadata. Repeat runs should therefore skip completed work when
inputs and stage-scoped config hashes are unchanged.
