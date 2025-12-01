# Analysis Notes

| Metric | Definition | Symmetry expectation |
| --- | --- | --- |
| `win_rate` / `win_prob` | Fraction of games a strategy wins within a player-count cohort. `win_prob` is written alongside metrics as an alias of `win_rate` for downstream features that expect a probability column. | For symmetric matchups, swapping seats should not change the probability; `win_rate` and `win_prob` should therefore agree when computed from seat-agnostic aggregates. |
| `P1_win_rate` / `P2_win_rate` | Seat-adjusted head-to-head win rates derived from aggregated pairwise results after merging mirrored `(A,B)`/`(B,A)` matchups. | `P1_win_rate + P2_win_rate = 1` for decisive pairs; mirroring the strategies should leave the rates unchanged because results are collapsed by unordered matchup. |
