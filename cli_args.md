# CLI Reference

The repository exposes a single console entry point: `farkle`.  You can invoke
it directly or via `python -m farkle`.

```text
farkle [GLOBAL OPTIONS] <command> [COMMAND OPTIONS]
```

## Global options

- `--config PATH` – load YAML configuration data and pass it to the chosen
  subcommand.  The mapping is unpacked as keyword arguments, so the keys should
  match the function parameters used by that command (for example, the
  arguments of `farkle.simulation.run_tournament.run_tournament`).
- `--set KEY=VALUE` – apply inline overrides to the loaded configuration.  The
  option may be repeated.  Keys may include dots to create nested mappings and
  the values are parsed with `yaml.safe_load`, allowing booleans, integers,
  lists, and other simple YAML literals.
- `--log-level LEVEL` – configure the root logger before the command runs.

## Subcommands

### `run`
Run a Monte Carlo tournament.

Options:
- `--metrics` – collect per-strategy summary metrics as well as win counts.
- `--row-dir PATH` – write per-game rows to the given directory (Parquet).

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
Benchmark simulation throughput.  This forwards to
`farkle.simulation.time_farkle.measure_sim_times` and accepts the same options
as that helper.  Use `farkle time --help` for the full list.

### `watch`
Interactively watch a single game play out.

Options:
- `--seed INT` – seed the RNG for deterministic behaviour.

### `analyze`
Convenience wrapper around the analysis pipeline.  Requires configuration
compatible with `farkle.analysis.analysis_config.PipelineCfg`.

Subcommands:
- `ingest` – load raw CSV data into Parquet shards.
- `curate` – post-process ingested data and update manifests.
- `metrics` – compute aggregate metrics (including TrueSkill ratings when
  enabled in the configuration).
- `pipeline` – run `ingest`, `curate`, and `metrics` in sequence.

Example usage:

```bash
farkle --config analysis/pipeline.yaml \
  --set run_trueskill=true \
  analyze pipeline
```

Use `--help` on any subcommand for additional details, e.g.
`farkle analyze metrics --help`.
