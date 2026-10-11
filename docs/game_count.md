# Lifetime game-count estimate

Recorded October 11, 2026. Recommended wording: **My PC has played roughly
1.7 billion games of Farkle.**

This estimates completed games represented by surviving evidence, not every
invocation or attempted game.

## Historical baseline

The local Codex conversation **Estimate number of games played**, thread
`019c21e6-14a2-7711-8eab-4f8bd9532226`, reported these results on February 2,
2026 (the final reply is timestamped February 3 at 05:17 UTC):

| Component | Completed-game wins |
| --- | ---: |
| Tournament metrics, including `old_results_seed_20` | 573,057,877 |
| H2H aggregated pairwise wins | 234,380,202 |
| Historical baseline | **807,438,079** |

The user confirmed that generated data was cleaned after this calculation and
that the oldest surviving result folders belong to the subsequent period.
Retain the baseline once; do not recompute it from the later folders.

## Surviving later research runs

Count one tournament metrics aggregate per root/k and one H2H execution
aggregate per run. Raw rows, checkpoints, inference tables and analysis copies
are not additional games. Preserve these counts when removing disposable runs.

| Folder under `data/` | Tournament wins | H2H wins |
| --- | ---: | ---: |
| `results_efficiency_updates_4B_fast_seed_pair_52_53` | 90,762 | 362,208 |
| `results_efficiency_updates_4D_fast_seed_pair_54_55` | 90,757 | 184,320 |
| `results_guardrails_cleanup_1-4_fast_seed_pair_44_45` | 325,275 | 0 |
| `results_guardrails_cleanup_1-4_fast_seed_pair_46_47` | 325,279 | 0 |
| `results_guardrails_cleanup_1-4_fast_seed_pair_48_49` | 650,591 | 23,024,736 |
| `results_official_fast_20260801_seed_pair_36_37` | 650,550 | 23,024,736 |
| `results_post_fast_config_remediated_20260730_seed_pair_34_35` | 650,485 | 23,024,736 |
| `results_post_task16_repaired_fast_20260803_seed_pair_38_39` | 650,445 | 23,024,736 |
| `results_post_task16_repaired_fast_20260803_seed_pair_40_41` | 325,294 | 0 |
| `results_post_task16_repaired_fast_20260803_seed_pair_42_43` | 325,261 | 0 |
| `results_rng_opt_fast_seed_pair_56_57` | 90,747 | 237,060 |
| `results_seed_pair_0_1` | 199,992 | 176,960 |
| `results_seed_pair_100_101` | 99,272,724 | 99,755,010 |
| `results_seed_pair_102_103` | 99,399,144 | 479,161,200 |
| `results_seed_pair_10_11` | 199,992 | 176,960 |
| `results_seed_pair_12_13` | 510,632 | 176,960 |
| `results_seed_pair_14_15` | 510,632 | 176,960 |
| `results_seed_pair_16_17` | 510,632 | 176,960 |
| `results_seed_pair_18_19` | 510,632 | 176,960 |
| `results_seed_pair_20_21` | 510,632 | 0 |
| `results_seed_pair_22_23` | 510,632 | 0 |
| `results_seed_pair_24_25` | 510,632 | 176,960 |
| `results_seed_pair_26_27` | 510,632 | 176,960 |
| `results_seed_pair_28_29` | 510,632 | 176,960 |
| `results_seed_pair_2_3` | 99,996 | 0 |
| `results_seed_pair_2_31` | 199,992 | 176,960 |
| `results_seed_pair_30_31` | 653,600 | 23,209,032 |
| `results_seed_pair_32_33` | 653,600 | 22,503,600 |
| `results_seed_pair_4_5` | 199,992 | 176,960 |
| `results_seed_pair_6_7` | 199,992 | 176,960 |
| `results_seed_pair_8_9` | 199,992 | 176,960 |
| `results_smoke_fix_seed_pair_28_29` | 40 | 0 |
| `results_smoke_minfix_seed_pair_28_29` | 4 | 56 |
| **Later subtotal** | **210,050,194** | **719,811,910** |

The later subtotal is **929,862,104**. With the historical baseline, the estimate
is **1,737,300,183**, or approximately **1.74 billion** completed games.

The planning pass reported 929,950,584 later wins. A byte-identical H2H copy in
`results_smoke_fix_seed_pair_28_29` repeated 88,480 wins already present in
`results_seed_pair_28_29`; it is excluded here. Both aggregates have SHA-256
`c1a8bc479876bddb89501c8a77c39ad0a33b345459d4ab24f4292c9be1943e6f`.
No other byte-identical research aggregates were found in that comparison.

## Development and benchmark candidates

These counts are preserved before pruning. They mix real executions, copies,
and synthetic test counts; they are **not added** without checking provenance.
Their approximately 11.7 million candidate wins do not materially affect the
ballpark. Invalid or missing win columns are excluded rather than set to zero.

| Development folder | Candidate tournament wins | Candidate H2H wins |
| --- | ---: | ---: |
| `data` | 0 | 24 |
| `farkle-task4a-bounded-w1-v2` | 0 | 23,688 |
| `farkle-task4a-bounded-w2` | 0 | 70,668 |
| `farkle-task4a-bounded-w2-v2` | 0 | 70,668 |
| `farkle-task4a-quick-w2` | 0 | 144 |
| `farkle-task4c-pre-fix-12-root_seed_90512` | 2,301 | 0 |
| `farkle-task4c-pre-fix-root_seed_90501` | 2,314 | 0 |
| `farkle-task4c-simulation-execution-v1` | 4,614 | 0 |
| `farkle-task4c-task4d-telemetry-v1` | 4,614 | 0 |
| `farkle-task7b-h2h-bundling-v1` | 0 | 9,823,592 |
| `step_5_5_canary` | 28 | 0 |
| `step_5_5_canary_full` | 28 | 12 |
| `step_5_5_canary_max_k` | 4 | 12 |
| `step_5_5_canary_verified` | 28 | 12 |
| `task10` | 12,674 | 1,436,345 |
| `task7b_candidate_opt_diagnostic` | 0 | 48 |
| `task7b_quick` | 0 | 48 |
| `task7b_topology_diagnostic` | 0 | 246,984 |
| `task8-oracle-acceptance` | 11 | 11 |
| `task8-oracle-completion` | 11 | 11 |
| `task8-oracle-final-acceptance` | 11 | 11 |
| `task8-oracle-final-work` | 11 | 11 |
| `task8-oracle-work` | 5 | 0 |

## Recount recipe and limits

Use the root Python 3.12 venv. Select the exact aggregate paths from configuration
or authenticated run contexts. Tournament files are named `<k>p_metrics.parquet`
under `<k>_players`; project `wins`. H2H execution aggregates are named
`root_order_counts.parquet` or historical `bonferroni_pairwise.parquet`; project
`wins_a`/`wins_b`, or older `wins_seat1`/`wins_seat2`. Do not sum all downstream
representations. A projected read in a Python session is:

```python
from pathlib import Path
import pyarrow.parquet as pq


def summed_wins(path: Path, columns: list[str]) -> int:
    total = 0
    with pq.ParquetFile(path) as source:
        if not set(columns).issubset(source.schema_arrow.names):
            raise ValueError("Missing win columns")
        for batch in source.iter_batches(columns=columns, batch_size=8192):
            for column in batch.columns:
                for value in column.to_pylist():
                    if value is None or value < 0 or int(value) != value:
                        raise ValueError("Invalid win count")
                    total += int(value)
    return total
```

Independent files may use a four-worker executor. This pass used projected
8,192-row batches and four workers; it loaded no game rows. Replace an existing
execution subtotal when recounting; add a new execution only once after checking
for copies. Keep retired subtotals when deleting sources. Do not add the entire
surviving subtotal to this already cumulative estimate again.

Each official win represents one completed game. Safety-limit attempts have no
winner and are absent from this proxy. Unrecorded tests, overwritten runs and
deleted executions cannot be reconstructed. Identical deterministic output does
not prove that a game was physically played only once. Synthetic fixtures are
not actual gameplay. The approximate wording reflects these limitations.
