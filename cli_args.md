# CLI Reference

The installed console entry point is `farkle`. It is also available through
module execution with `python -m farkle`.

Normal users should prefer `farkle ...`. The legacy module entry point
`python -m farkle.analysis.pipeline` still exists for direct module use and
tests, but it is not the packaged front door.

```text
farkle [GLOBAL OPTIONS] <command> [COMMAND OPTIONS]
```

## Global options

- `--config PATH`
  Load a YAML overlay into `AppConfig`. This applies to `run`, `analyze`, and
  `two-seed-pipeline`.
- `--set SECTION.OPTION=VALUE`
  Override one loaded config value. The flag may be repeated.
- `--log-level LEVEL`
  Set the root logging level before dispatch.
- `--seed-a INT` / `--seed-b INT`
  Override the two-seed tuple used for orchestration commands.
- `--seed-pair A B`
  Override the two-seed tuple in one flag. This is mutually exclusive with
  `--seed-a` and `--seed-b`.

## Commands

### `run`

Launch the tournament runner using `cfg.sim`.

Options:

- `--metrics`
  Force `cfg.sim.expanded_metrics = True`.
- `--row-dir PATH`
  Write full per-game rows to the provided directory.
- `--force`
  Recompute even when resumable run outputs already exist.

Example:

```bash
farkle --config configs/fast_config.yaml \
  --set sim.seed=123 \
  --set sim.num_shuffles=200 \
  run --metrics --row-dir data/results/fast_seed_123/rows
```

### `time`

Benchmark simulation throughput.

Options:

- `--players INT`
- `--n-games INT`
- `--jobs INT`
- `--seed INT`

Example:

```bash
farkle time --players 5 --n-games 5000 --jobs 4 --seed 42
```

### `watch`

Interactively watch a single game.

Options:

- `--seed INT`

### `analyze`

Convenience wrapper around the analysis helpers. Each subcommand uses the
shared `AppConfig` to locate inputs and outputs.

Subcommands:

- `ingest`
  Convert raw simulation outputs into ingest-stage parquet artifacts.
- `curate`
  Finalize ingest outputs into canonical curated row files.
- `combine`
  Merge curated per-player-count files into pooled combined data.
- `metrics`
  Compute pooled metrics and optional game-stat and RNG side outputs.
- `variance`
  Compute variance summaries across seeds.
- `preprocess`
  Run `ingest`, `curate`, `combine`, and `metrics`.
- `pipeline`
  Run preprocess plus the downstream analytics tail.
- `analytics`
  Run the downstream analytics tail only.

Shared option groups:

- `metrics`, `preprocess`, `pipeline`
  - `--compute-game-stats`
  - `--rng-diagnostics`
  - `--rng-lags INT [INT ...]`
  - `--margin-thresholds INT [INT ...]`
  - `--rare-event-target INT`
  - `--rare-event-margin-quantile FLOAT`
  - `--rare-event-target-rate FLOAT`
- `pipeline`, `analytics`
  - `--allow-missing-upstream`
- `variance`
  - `--force`

Examples:

```bash
farkle --config configs/fast_config.yaml analyze metrics --compute-game-stats
```

```bash
farkle --config configs/fast_config.yaml analyze pipeline \
  --compute-game-stats \
  --rng-diagnostics \
  --rng-lags 1 2 4
```

### `two-seed-pipeline`

Run the dual-seed simulation and analysis orchestration pipeline.

Options:

- `--force`
  Recompute even when completion markers exist.

Example:

```bash
farkle --config configs/fast_config.yaml two-seed-pipeline --seed-pair 42 43
```

## Notes

- Stage folder numbers are assigned by `StageLayout` at runtime. Use
  `AppConfig` path helpers or `analysis/config.resolved.yaml` instead of
  assuming fixed folder numbers in downstream scripts.
- `sim.seed_list` is the canonical seed source. `sim.seed` and `sim.seed_pair`
  remain compatibility aliases.
- Run `farkle <command> --help` for the exact parser output for your installed
  version.
