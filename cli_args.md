# CLI Reference

The package installs a single console entry point named `farkle`. It can also
be invoked with `python -m farkle`.

```text
farkle [GLOBAL OPTIONS] <command> [COMMAND OPTIONS]
```

## Global options

- `--config PATH` - load an `AppConfig` from YAML. The configuration is used by
  the `run`, `analyze`, and `two-seed-pipeline` commands; other subcommands
  ignore it. Omitting the flag falls back to the built-in defaults.
- `--set SECTION.OPTION=VALUE` - override a single field on the loaded config.
  The first segment chooses the top-level section (for example `sim` or
  `analysis`); the second names the attribute inside that dataclass. Values are
  coerced to the existing field type when possible (bool, int, float, or path),
  and the flag may be supplied multiple times.
- `--log-level LEVEL` - set the root logging level before dispatching the
  command. Accepts standard names (`INFO`, `DEBUG`, `WARNING`) or numeric levels.
- `--seed-a INT` / `--seed-b INT` - override the two-seed tuple used by
  `cfg.sim.seed_pair` when running dual-seed orchestration (provide both flags).
- `--seed-pair A B` - override `cfg.sim.seed_pair` with two integer seeds in a
  single flag (mutually exclusive with `--seed-a`/`--seed-b`).

## Subcommands

### `run`
Launch the tournament runner using settings from `cfg.sim`.

Options:
- `--metrics` - force `cfg.sim.expanded_metrics = True` so that extra per-seat
  metrics are collected and persisted.
- `--row-dir PATH` - store per-game rows under the provided directory by
  setting `cfg.sim.row_dir` before execution.
  Results are written to `data/<results_dir_prefix>_seed_<seed>` for single-seed runs.

Example:

```bash
farkle --config configs/fast_config.yaml \
  --set sim.seed=123 \
  --set sim.num_shuffles=200 \
  run --metrics --row-dir data/results/fast_seed_123/rows
```

### `time`
Benchmark simulation throughput with the defaults from
`farkle.simulation.time_farkle.measure_sim_times`. No additional command
options are parsed; only the global logging level applies.

### `watch`
Interactively watch a single game session.

Options:
- `--seed INT` - lock the RNG so repeated runs produce the same game.

### `analyze`
Convenience wrapper around the analysis helpers. Each subcommand uses the
shared `AppConfig`, particularly the `analysis`, `ingest`, `combine`, `metrics`,
and `head2head` sections, to locate inputs and drive processing.

Subcommands:
- `ingest` - convert raw CSV rows into curated parquet shards.
- `curate` - post-process ingested rows and update manifests.
- `combine` - merge curated shards into a consolidated parquet file.
- `metrics` - compute aggregate metrics and engineer useful features from curated data.
- `preprocess` - run `ingest`, `curate`, `combine`, and `metrics` sequentially.
  Optional stages are ordered by their directory prefixes: `04_game_stats` then
  `05_rng` when enabled.
- `two-seed-pipeline` - deprecated alias for the top-level `two-seed-pipeline`
  command. Runs the two-seed simulation + analysis orchestrator for
  `cfg.sim.seed_pair`. This is equivalent to the legacy
  `farkle-two-seed-pipeline` console script and module entry point.
- `analytics` - perform statistical analysis computation steps (TrueSkill, Bonferroni head-to-head, HGB modeling, etc.) according to the configuration.
- `pipeline` - run `preprocess` followed by `analytics` for a full end-to-end pass.
  The analytics suite follows the renumbered stage layout (`06_seed_summaries`,
  `07_variance`, `08_meta`, `09_trueskill`, `10_head2head`, `11_hgb`,
  `12_tiering`, `13_agreement`).

Use `--help` on any subcommand for additional details (for example,
`farkle analyze metrics --help`).

### `two-seed-pipeline`
Run the two-seed simulation + analysis orchestrator for `cfg.sim.seed_pair`.

Example:

```bash
farkle --config configs/fast_config.yaml two-seed-pipeline --seed-pair 42 43
```

#### Handchecking the Pipeline

Use the preset `configs/presets/handcheck_pipeline.yaml` for a quick end-to-end
run over the bundled dummy data. The sample results at `data/results_dummy`
include `4_players`, `5_players`, and `6_players` blocks, each with roughly two
thousand games (<100k as a safety margin).

```bash
farkle --config configs/presets/handcheck_pipeline.yaml analyze pipeline
```

To run the dual-seed orchestration:

```bash
farkle --config configs/fast_config.yaml two-seed-pipeline --seed-pair 42 43
```

The pipeline writes fresh artifacts to `data/results_dummy/analysis_handcheck`.
Per-seat manifests such as `analysis_handcheck/data/4p/manifest_4p.json` record
the ingested `row_count`, while the combined parquet lives at
`analysis_handcheck/data/all_n_players_combined/all_ingested_rows.parquet`.
Inspect those files (or load them in a Parquet viewer) to handcheck the totals.
