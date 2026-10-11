# CLI Reference

The installed console entry point is `farkle`. It is also available through
module execution with `python -m farkle`.

Normal users should prefer `farkle ...`.

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
- `--h2h-authorize PLAN_SHA256:MAX_ATTEMPTS`
  Authorize the exact frozen H2H plan above its configured cap. Supported by
  `two-seed-pipeline` and `analyze pipeline|analytics`; put this global option
  before the command name. Copy the token printed by the blocked run. This is
  runtime consent, not a configuration override.
- `--resource-limiters {enabled,memory-only,disabled}`
  Select runtime host-resource controls for `run`, `analyze`, and
  `two-seed-pipeline`. `enabled` is the default; `memory-only` retains memory
  controls but leaves CPU/native-thread allocation to the OS; `disabled`
  bypasses host-resource ceilings. Non-default values are rejected by `time`
  and `watch`.

## Commands

After finalist freeze, H2H displays completed-game targets, maximum attempts
including safety replacements, the configured cap, and both budget percentages.
When the maximum exceeds the cap, an interactive terminal asks:
`Override the cap for this H2H plan and continue? [Y/N]`.
Y/y publishes authenticated authorization before execution; N/n or blank input
declines. Invalid input is retried; EOF or Ctrl+C stops without approval.
Noninteractive input never prompts and requires the exact option above.
These operational stops exit with code 3 and preserve completed work.

Resume the same command with the same configuration and output paths. Compatible
saved approval is reused without another prompt, including after partial H2H
execution. Authorization lives at
`cfg.h2h_cap_authorization_path()` alongside execution state; it binds the family,
schedule, exact plan/manifest bytes, and maximum attempt amount. The preset cap,
power target, candidate family, and statistical schedule remain unchanged.
Missing or interrupted approval is not inferred from game progress.

For example, an unattended resume uses the exact token printed by that run:

```text
farkle --config PATH --h2h-authorize PLAN_SHA256:MAX_ATTEMPTS two-seed-pipeline
```

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
farkle --config configs/default_config.yaml \
  --resource-limiters enabled \
  --set sim.seed=123 \
  --set sim.n_jobs=4 \
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
  Finalize ingest outputs using authenticated reflinks where supported and a
  bounded physical-copy fallback.
- `combine`
  Normalize independently resumable by-k partitions and publish the logical
  `concat_ks` manifest. `--force` rewrites all partitions, `--deep-verify`
  rereads and hashes them, and `--materialize PATH` writes a non-canonical
  release compatibility Parquet inside the `concat_ks` scope.
- `metrics`
  Compute pooled metrics and optional game-stat and RNG side outputs.
- `preprocess`
  Run `ingest`, `curate`, `combine`, and `metrics`.
- `pipeline`
  Run preprocess plus the downstream analytics tail.
- `analytics`
  Run the downstream analytics tail only.

Shared option groups:

- `metrics`, `preprocess`
  - `--compute-game-stats`
- `pipeline`, `analytics`
  - `--rng-diagnostics`
  - `--rng-lags INT [INT ...]`
- `metrics`, `preprocess`, `pipeline`
  - `--margin-thresholds INT [INT ...]`
  - `--rare-event-target INT`
  - `--rare-event-margin-quantile FLOAT`
  - `--rare-event-target-rate FLOAT`

Examples:

```bash
farkle --config configs/default_config.yaml --set sim.seed=123 \
  analyze metrics --compute-game-stats
```

```bash
farkle --config configs/default_config.yaml --set sim.seed=123 analyze pipeline \
  --rng-diagnostics \
  --rng-lags 1 2 4
```

### `two-seed-pipeline`

Run the dual-seed simulation and analysis orchestration pipeline.

Options:

- `--force`
  Recompute even when completion markers exist.
- `--seed-a INT` / `--seed-b INT`
  Override the two roots. Both options must be provided together.
- `--seed-pair A B`
  Override the two roots in one flag. This is mutually exclusive with
  `--seed-a` and `--seed-b`.

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
- `run` and `analyze` require a one-root configuration. The checked-in
  `fast_config.yaml` is a two-root integration profile and is intended for
  `two-seed-pipeline`.
- Run `farkle <command> --help` for the exact parser output for your installed
  version.
- Global options must appear before the command name.
