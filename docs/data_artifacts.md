# Simulation and analysis data artifacts

This catalog enumerates every on-disk artifact produced by the simulation and
analysis pipelines, including who writes it, what it contains, how large it
usually is, and which downstream steps depend on it.

## Simulation stage outputs

| Artifact pattern | Producer | Schema / key fields | Typical volume | Consumers | Notes |
| --- | --- | --- | --- | --- | --- |
| `results/<run>/checkpoint.pkl`<br/>`results/<n>_players/<n>p_checkpoint.pkl` | `farkle.simulation.run_tournament._save_checkpoint` when the tournament loop flushes counters or finishes | Pickle containing `{"win_totals": Counter[str]}` and, when metrics are enabled, nested `metric_sums` / `metric_square_sums` dictionaries keyed by metric label and strategy | ~8 160 strategy rows per table size (10 metric vectors per strategy) because the grid size is fixed in `run_full_field` (`GRID = 8_160`)【F:src/farkle/simulation/run_tournament.py†L331-L345】【F:src/farkle/analysis/run_full_field.py†L140-L205】 | `farkle.simulation.runner.run_tournament` unpickles the file to emit `win_counts.csv`; `analysis.run_full_field` uses presence of the file to detect completed table sizes and to resume interrupted sweeps【F:src/farkle/simulation/runner.py†L102-L128】【F:src/farkle/analysis/run_full_field.py†L80-L118】 | Long-running checkpoint (remains pickle for resumability). Final metric aggregates now land in `<n>p_metrics.parquet`. |
| `results/<run>/win_counts.csv` | `farkle.simulation.runner.run_tournament` after reading the checkpoint | CSV header `strategy,wins` (strategy string, integer win count)【F:src/farkle/utils/sinks.py†L81-L89】 | One row per strategy (~8 160)【F:src/farkle/analysis/run_full_field.py†L140-L205】 | Human inspection; light-weight reporting prior to ingestion | Already text friendly; no change required. |
| `results/<n>_players/<n>p_rows/rows_*.parquet` + `manifest.jsonl` | Worker shards inside `_run_chunk_metrics` when `collect_rows=True` | Arrow schema derived from `_play_one_shuffle`: `game_seed`, winner columns (`winner`, `winner_seat`, scores) and every seat-level metric defined in `analysis_config` (e.g. `P1_score`, `P1_strategy`, ranks, smart-dice counters)【F:src/farkle/simulation/run_tournament.py†L131-L168】【F:src/farkle/analysis/analysis_config.py†L82-L214】【F:src/farkle/simulation/run_tournament.py†L225-L305】 | Shard size is set by shuffle batches; final run produces the totals listed below per table size | `farkle.analysis.ingest` streams either consolidated files or these shards; manifests are append-only NDJSON via `farkle.utils.manifest` for audit/resume【F:src/farkle/analysis/ingest.py†L60-L98】【F:src/farkle/utils/manifest.py†L1-L111】 | Manifests capture `path`, player count, seed, and PID for each shard. |
| `results/<n>_players/<n>p_rows.parquet` | `_concat_row_shards` consolidates shard directory once a table size finishes | Same schema as shards above | Deterministic totals per table size computed by `run_full_field`:<br/>2p 35 683 680 · 3p 21 618 560 · 4p 13 910 760 · 5p 9 640 224 · 6p 7 076 080 · 8p 4 296 240 · 10p 2 904 144 · 12p 2 108 000 (≈97 237 688 rows combined)【F:src/farkle/analysis/run_full_field.py†L28-L76】【F:src/farkle/analysis/run_full_field.py†L191-L259】【23871c†L1-L9】【e1f74f†L1-L1】 | Preferred ingest source; `analysis.ingest` picks these up first via `row_file = block.glob("*p_rows.parquet")`【F:src/farkle/analysis/ingest.py†L60-L86】 | Candidate for long-term canonical raw-game storage. |
| `results/<n>_players/metrics_*.parquet` + `metrics_manifest.jsonl` (optional) | `_run_chunk_metrics` when `metric_chunk_directory` is supplied | Columns `metric`, `strategy`, `sum`, `square_sum` persisted per chunk | One row per `(metric, strategy)` in the chunk (10 metrics × ~8 160 strategies) | Only used for crash-safe aggregation: the driver reloads these files to rebuild the global sums before writing the final checkpoint【F:src/farkle/simulation/run_tournament.py†L513-L546】【F:src/farkle/simulation/run_tournament.py†L599-L604】 | Stored as parquet already; manifest follows NDJSON convention. |
| `results/<n>_players/<n>p_metrics.parquet` | `farkle.simulation.run_tournament.run_tournament` once a table size completes | Arrow schema `metric` (string), `strategy` (string), `sum`, `square_sum` (float64) built from the final aggregates kept in memory or reconstructed from chunk files | Deterministic 10 × ~8 160 rows per table (one record per metric/strategy pair) | Downstream analytics read this instead of unpickling checkpoints; e.g. ingestion notebooks, custom reporting | Canonical metrics export written atomically with snappy compression alongside the checkpoint【F:src/farkle/simulation/run_tournament.py†L620-L651】 |

## Analysis pipeline outputs

### Ingest & curate

| Artifact pattern | Producer | Schema / key fields | Typical volume | Consumers | Notes |
| --- | --- | --- | --- | --- | --- |
| `analysis/data/<n>p/<n>p_ingested_rows.raw.parquet` + `.manifest.jsonl` | `farkle.analysis.ingest._process_block` streams tournament rows into Arrow parquet with atomic shards | Canonical schema from `expected_schema_for(n)`: winner metadata plus `P<i>_{score,farkles,rolls,highest_turn,strategy,rank,loss_margin,smart_*}` fields up to seat `n`【F:src/farkle/analysis/ingest.py†L156-L269】【F:src/farkle/analysis/analysis_config.py†L208-L214】 | Mirrors tournament totals per table size (see above); manifest tracks row counts and provenance metadata | `farkle.analysis.curate` finalises files; `metrics` stage later uses manifest row counts for seat denominators【F:src/farkle/analysis/curate.py†L123-L144】【F:src/farkle/analysis/metrics.py†L215-L227】 | Manifest entries follow the NDJSON format from `farkle.utils.manifest` for append-only provenance【F:src/farkle/utils/manifest.py†L1-L111】. |
| `analysis/data/<n>p/<n>p_ingested_rows.parquet` + `manifest_<n>p.json` | `farkle.analysis.curate.run` promotes raw files and writes JSON manifest with schema hash, compression, and row count | Same seat-augmented schema; manifest stores `row_count`, schema hash and config SHA for reproducibility | Same per-table totals; JSON manifest is tiny | `analysis.combine` streams these; `PipelineCfg.curated_parquet` uses them as canonical per-seat sources【F:src/farkle/analysis/curate.py†L123-L175】【F:src/farkle/analysis/analysis_config.py†L118-L126】 | Manifest enables fast freshness checks and feeds seat-advantage denominators. |
| `analysis/data/all_n_players_combined/all_ingested_rows.parquet` + `.manifest.jsonl` | `farkle.analysis.combine.run` concatenates per-seat files into a 12-seat superset, padding missing seats with nulls | Schema is `expected_schema_for(12)` (all seat columns present)【F:src/farkle/analysis/combine.py†L44-L124】【F:src/farkle/analysis/analysis_config.py†L208-L214】 | ≈97 M rows (sum of all seat counts)【e1f74f†L1-L1】 | `farkle.analysis.metrics` and other analytics read this via `PipelineCfg.curated_parquet`; `metrics` checks manifests for consistency【F:src/farkle/analysis/metrics.py†L98-L144】 | Already Parquet + NDJSON manifest; make snappy the standard compression going forward. |

### Metrics & summaries

| Artifact pattern | Producer | Schema / key fields | Typical volume | Consumers | Notes |
| --- | --- | --- | --- | --- | --- |
| `analysis/metrics.parquet` | `farkle.analysis.metrics.run` | Columns include `strategy`, `n_players`, `games`, `wins`, `win_rate`, `se_win_rate`, `win_rate_ci_lo`, `win_rate_ci_hi`, `win_prob`, and expected-score aggregates【F:src/farkle/analysis/metrics.py†L33-L63】【F:src/farkle/analysis/metrics.py†L167-L209】 | One row per strategy (~8 160) | `farkle.analysis.run_hgb` merges metrics with ratings to build the feature matrix【F:src/farkle/analysis/run_hgb.py†L125-L189】 | Already in parquet; recommended as canonical replacement for any legacy `metrics.pkl`. |
| `analysis/seat_advantage.csv` | `farkle.analysis.metrics.run` | Columns `seat`, `wins`, `games_with_seat`, `win_rate` written from a pandas DataFrame via `write_csv_atomic` | Fixed 12 rows (one per seat) | Readable summary for documentation; also referenced when tracking seat denominators | Text-friendly mirror of the Parquet summary.【F:src/farkle/analysis/metrics.py†L225-L271】 |
| `analysis/seat_advantage.parquet` | Same | Arrow schema `seat` (int32), `wins`/`games_with_seat` (int64), `win_rate` (float64) derived from the same DataFrame as the CSV | 12 rows | Programmatic consumers needing columnar access to seat summaries | Written with snappy compression alongside the CSV for consistency across tooling.【F:src/farkle/analysis/metrics.py†L225-L271】 |
| `analysis/metrics.done.json` | `farkle.analysis.metrics.run` | JSON stamp containing input/output mtimes and sizes for cache invalidation【F:src/farkle/analysis/metrics.py†L260-L276】 | Tiny | Used only for freshness checks before recomputing metrics | Keep as-is; not a data product. |

### Analytics (TrueSkill, head-to-head, HGB)

| Artifact pattern | Producer | Schema / key fields | Typical volume | Consumers | Notes |
| --- | --- | --- | --- | --- | --- |
| `analysis/ratings_<n>.parquet` | `farkle.analysis.run_trueskill._rate_block_worker` writes per-table ratings | Arrow table `{strategy: string, mu: float64, sigma: float64}` | One row per strategy that appeared in the block (≤8 160) | `run_trueskill` pooling phase, downstream ML | Durable parquet already. |
| `analysis/ratings_<n>.json` | Same | JSON mapping `strategy -> {mu, sigma}` for quick inspection【F:src/farkle/analysis/run_trueskill.py†L639-L650】 | Same row count encoded as JSON | Humans / lightweight clients | Text mirror of parquet. |
| `analysis/ratings_<n>.checkpoint.parquet` + `ratings_<n>.ckpt.json` | `run_trueskill` checkpointing loop | Parquet/JSON pair storing interim ratings and resume cursor (`row_group`, `batch_index`, `games_done`)【F:src/farkle/analysis/run_trueskill.py†L575-L637】 | Updated while streaming row groups | `run_trueskill` resume logic | Checkpoints; not final products but must stay in pickle/JSON family. |
| `analysis/ratings_pooled.parquet` & `analysis/ratings_pooled.json` | `run_trueskill` pooling phase combines block outputs into global ratings and tiers | Same schema as per-block parquet; JSON summarises pooled stats and feeds `tiers.json` | One row per strategy (≤8 160) | `analysis/run_hgb` merges this Parquet table directly using pyarrow before fitting the regressor【F:src/farkle/analysis/run_trueskill.py†L775-L822】【F:src/farkle/analysis/run_hgb.py†L121-L150】 | Canonical ratings source consumed without pickle fallbacks. |
| `analysis/tiers.json` | `run_trueskill` | JSON mapping strategies to tier labels【F:src/farkle/analysis/run_trueskill.py†L814-L821】 | One entry per strategy | `run_bonferroni_head2head` loads it to schedule matchups【F:src/farkle/analysis/run_bonferroni_head2head.py†L46-L66】 | Already JSON. |
| `analysis/bonferroni_pairwise.parquet` | `run_bonferroni_head2head` | Arrow schema `{a: string, b: string, wins_a: int64, wins_b: int64, pvalue: float64}` derived from simulated matches | Dense upper triangle of elite strategies (depends on tier size) | Review & reporting plus downstream statistical checks | Real Parquet output written atomically with snappy compression, matching the documented extension.【F:src/farkle/analysis/run_bonferroni_head2head.py†L39-L144】 |
| `analysis/hgb_importance.json` | `farkle.analysis.run_hgb.run_hgb` | JSON mapping feature name to permutation importance (float)【F:src/farkle/analysis/run_hgb.py†L189-L215】 | ≤10 features (strategy parameters) | Model interpretability notebooks | Already JSON; keep. |
| `notebooks/figs/pd_*.png` | `run_hgb.run_hgb` (via `plot_partial_dependence`) | PNG partial dependence plots, up to `MAX_PD_PLOTS` features【F:src/farkle/analysis/run_hgb.py†L218-L237】 | ≤30 images (~100–200 KB each) | Reports / notebooks | Media assets; outside tabular scope. |

### Pipeline scaffolding

| Artifact pattern | Producer | Purpose | Notes |
| --- | --- | --- | --- |
| `analysis/config.resolved.yaml` | `farkle.analysis.pipeline.main` writes the fully resolved configuration | Provenance for analysis runs【F:src/farkle/analysis/pipeline.py†L44-L49】 | Keep YAML for readability. |
| `analysis/manifest.json` | Same | Stores config SHA and run metadata for pipeline orchestration【F:src/farkle/analysis/pipeline.py†L50-L59】 | Append-only JSON manifest. |
| `<artifact>.done.json` (e.g. `tiers.json.done.json`, `bonferroni_pairwise.parquet.done.json`, `hgb_importance.json.done.json`) | `pipeline.write_done` stamps after each analytics step | Records input fingerprints and tool name to skip stale recomputation【F:src/pipeline.py†L60-L157】 | Control metadata; not part of delivered datasets. |

## Pickle artifacts to flag

* **Tournament checkpoints** – `checkpoint.pkl` and `<n>p_checkpoint.pkl` remain essential for resumability and still carry in-memory aggregates for crash recovery【F:src/farkle/simulation/run_tournament.py†L331-L345】. Treat them as runtime checkpoints only; the canonical metrics now live in the adjacent Parquet export.

## Format recommendation matrix

| Data product type | Recommended format |
| --- | --- |
| Large tabular outputs (raw games, per-strategy metrics, ratings) | Parquet with snappy compression for columnar analytics【F:src/farkle/analysis/combine.py†L30-L124】【F:src/farkle/analysis/metrics.py†L247-L256】 |
| Small tabular or human-facing summaries | CSV, optionally mirrored to Parquet for programmatic parity (e.g., seat advantage)【F:src/farkle/analysis/metrics.py†L236-L245】 |
| Append-only logs / indexes | NDJSON manifests written via `farkle.utils.manifest` (already used for shards and streaming writers)【F:src/farkle/simulation/run_tournament.py†L283-L305】【F:src/farkle/analysis/ingest.py†L247-L269】【F:src/farkle/utils/manifest.py†L1-L111】 |
| Checkpoints for resumability | Pickle or JSON depending on payload (retain `_checkpoint.pkl`, `ratings_*.ckpt.json`, etc.) but treat them as transient, not the published dataset【F:src/farkle/simulation/run_tournament.py†L331-L345】【F:src/farkle/analysis/run_trueskill.py†L575-L650】 |

## Recent updates

1. **Per-table metrics now land in Parquet.** The tournament driver writes `<n>p_metrics.parquet` alongside each checkpoint so downstream tools never have to load the pickle payload for canonical aggregates.【F:src/farkle/simulation/run_tournament.py†L620-L651】
2. **Seat advantage ships as CSV *and* Parquet.** Metrics emits both formats atomically from the same DataFrame, keeping the human-readable view while providing a snappy-compressed columnar twin.【F:src/farkle/analysis/metrics.py†L225-L271】
3. **Head-to-head parquet matches its extension.** `run_bonferroni_head2head` now materialises a real Arrow table to `analysis/bonferroni_pairwise.parquet`, eliminating the CSV mismatch.【F:src/farkle/analysis/run_bonferroni_head2head.py†L39-L144】
4. **Ratings consumers use the parquet source.** `run_hgb` ingests `ratings_pooled.parquet` via pyarrow, removing the legacy pickle dependency and simplifying `hgb_feat` orchestration.【F:src/farkle/analysis/run_hgb.py†L121-L189】【F:src/farkle/analysis/hgb_feat.py†L20-L33】

These updates keep checkpoints for resumability while ensuring every canonical data product is published in a consistent, columnar format with NDJSON manifests for append-only logs.
