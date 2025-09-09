# CLI Reference

This document lists the command line entry points provided by the repository and their options.

## `farkle`
Invoke with `farkle` or `python -m farkle`.

Subcommands:
- `run <config>` – run a tournament using a YAML configuration file where `config` is the path to the file.

## `time-farkle`
Measure timings for single and multiple games.

Options:
- `-n, --n_games` *(default 1000)* – number of games to simulate in the batch.
- `-p, --players` *(default 5)* – number of players per game.
- `-s, --seed` *(default 42)* – master seed for reproducible RNG.
- `-j, --jobs` *(default 1)* – number of parallel processes.

## `run-full-field`
Sweep tournaments across table sizes.

Options:
- `--results-dir PATH` *(default `results_seed_0`)* – directory to store tournament results.
- `--force-clean` – remove existing row directories when final parquets are present.

## `watch-game`
Play a verbose two‑player game. No CLI options; strategies are random.

## `farkle-analyze`
Run parts of the analysis pipeline. Can also be invoked with `python -m farkle.pipeline`.

Options:
- `--config PATH` *(default `analysis_config.yaml`)* – path to pipeline YAML configuration.

Subcommands:
- `ingest`, `curate`, `aggregate`, `metrics`, `analytics`, `all` – run the specified pipeline stage.

## `python -m farkle.ingest`
Ingest raw tournament results.

Options:
- `--config PATH` *(default `analysis_config.yaml`)* – path to pipeline YAML configuration.

## `python -m farkle.run_tournament`
Run a Monte‑Carlo tournament.

Options:
- `--seed INT` *(default 0)* – global RNG seed.
- `--checkpoint PATH` *(default `checkpoint.pkl`)* – pickle output for checkpoints.
- `--jobs INT` – number of worker processes.
- `--ckpt-sec INT` *(default 30)* – seconds between saves.
- `--metrics` – collect per‑strategy means and variances.
- `--num-shuffles INT` *(default 5907)* – number of shuffles to simulate.
- `--row-dir DIR` – write full per‑game rows to directory as parquet.
- `--metric-chunk-dir DIR` – write per‑chunk metric aggregates.
- `--log_level {DEBUG,INFO,WARNING}` *(default INFO)* – logging verbosity.

## `python -m farkle.run_trueskill`
Compute TrueSkill ratings.

Options:
- `--output-seed INT` *(default 0)* – append seed to output filenames.
- `--dataroot PATH` – folder containing `<N>_players` blocks.
- `--root PATH` – output directory (defaults to `<dataroot>/analysis`).
- `--workers INT` – process blocks in parallel.
- `--batch-rows INT` – Arrow batch size for streaming readers.
- `--single-pass-from PATH` – path to aggregated rows for single-pass mode.
- `--no-single-pass` – force legacy per‑N mode.
- `--resume/--no-resume` – resume from checkpoint (default) or start fresh.
- `--no-resume-per-n` – disable per‑N resume.
- `--checkpoint-every-batches INT` *(default 500)* – batches between checkpoints.
- `--checkpoint-path PATH` – where to save checkpoint JSON.
- `--ratings-checkpoint-path PATH` – where to save interim ratings parquet.

## `python -m farkle.run_hgb`
Fit a HistGradientBoostingRegressor to ratings and metrics.

Options:
- `--seed INT` *(default 0)* – random seed for model fitting.
- `-o, --output PATH` – location to write `hgb_importance.json`.
- `--root PATH` *(default `results_seed_0`)* – root directory for inputs.

## `python -m farkle.run_bonferroni_head2head`
Run pairwise Bonferroni-corrected matches between top strategies.

Options:
- `--seed INT` *(default 0)* – base seed for schedule generation.
- `--root PATH` *(default `results_seed_0`)* – directory containing `tiers.json` and outputs.
- `--jobs INT` *(default 1)* – number of worker processes.

