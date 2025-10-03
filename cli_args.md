# CLI Reference

The repository exposes a single console entry point: `farkle`. You can invoke
it directly or via `python -m farkle`.

```text
farkle [GLOBAL OPTIONS] <command> [COMMAND OPTIONS]
```

## Global options

- `--config PATH` - load YAML configuration data and pass it to the chosen
  subcommand. The mapping is unpacked as keyword arguments, so the keys should
  match the function parameters used by that command (for example, the
  arguments of `farkle.simulation.run_tournament.run_tournament`).
- `--set KEY=VALUE` - apply inline overrides to the loaded configuration.
  Keys may include dots to create nested mappings and the values are parsed
  with `yaml.safe_load`, allowing booleans, integers, lists, and other simple
  YAML literals.
- `--log-level LEVEL` - configure the root logger before the command runs.
  The default is `INFO`.

## Subcommands

### `run`
Run a Monte Carlo tournament.

Options:
- `--metrics` - collect per-strategy summary metrics as well as win counts
  (sets `collect_metrics` when calling `run_tournament`).
- `--row-dir PATH` - write per-game rows to the given directory (Parquet
  shards plus a manifest; forwarded as `row_output_directory`).

Example usage:

```bash
# configs/tournament.yaml contains run_tournament keyword arguments such as
# n_players, num_shuffles, checkpoint_path, n_jobs, etc.
farkle --config configs/tournament.yaml \
  --set n_jobs=6 \
  --set global_seed=42 \
  run --metrics --row-dir data/results_seed_42/rows
```

### `time`
Benchmark simulation throughput using the defaults from
`farkle.simulation.time_farkle.measure_sim_times`.

This subcommand does not accept additional command options. Global
configuration files and `--set` overrides are currently ignored; call the
helper from Python for custom benchmarks.

### `watch`
Interactively watch a single game play out.

Options:
- `--seed INT` - seed the RNG for deterministic behaviour.

### `analyze`
Convenience wrapper around the analysis pipeline. Requires configuration
compatible with `farkle.config.AppConfig`.

Subcommands:
- `ingest` - load raw CSV data into Parquet shards.
- `curate` - post-process ingested data and update manifests.
- `combine` - merge curated Parquet shards into a single superset file.
- `metrics` - compute aggregate metrics (including TrueSkill ratings when
  enabled in the configuration).
- `pipeline` - run `ingest`, `curate`, `combine`, and `metrics` in sequence.

Use `--help` on any subcommand for additional details, for example
`farkle analyze metrics --help`.

#### Handchecking the Pipeline

Use the preset `configs/presets/handcheck_pipeline.yaml` when you want a quick end-to-end run over the bundled dummy data. The sample results at `data/results_dummy` only contain `4_players`, `5_players`, and `6_players` blocks, each with roughly two thousand games (<100k as a safety margin).

```bash
farkle --config configs/presets/handcheck_pipeline.yaml analyze pipeline
```

The pipeline writes fresh artifacts to `data/results_dummy/analysis_handcheck`. Per-seat manifests such as `analysis_handcheck/data/4p/manifest_4p.json` record the ingested `row_count`, while the combined parquet lives at `analysis_handcheck/data/all_n_players_combined/all_ingested_rows.parquet`. Review those files (or open them in a Parquet viewer) to handcheck the totals after the run.
