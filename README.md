# Farkle Mk II

Farkle Mk II is a Monte Carlo simulation and analytics toolkit for large
Farkle tournaments. It provides a deterministic simulation engine, a unified
CLI, stage-aware analysis paths, and resumable artifact handling.

## Highlights

- Deterministic single-game and tournament simulation driven by explicit seeds.
- Unified `farkle` CLI for simulation, benchmarking, watching games, analysis,
  and two-seed orchestration.
- Stage-aware `AppConfig` helpers that resolve canonical output paths without
  hard-coding numbered analysis directories.
- Streaming-friendly parquet outputs, append-only manifests, and `.done.json`
  stage stamps for resumable workflows.
- Config-driven analytics including coverage, game stats, TrueSkill,
  frequentist tiering, head-to-head analysis, HGB modeling, variance, meta,
  agreement, and interseed summaries.

## Installation

Requires Python 3.12 or newer.

```bash
pip install farkle
```

## Unified Configuration

All CLI workflows load a single YAML document into `farkle.config.AppConfig`.
The top-level sections are:

- `io`
- `sim`
- `analysis`
- `ingest`
- `combine`
- `metrics`
- `trueskill`
- `head2head`
- `hgb`
- `orchestration`

Important seed semantics:

- `sim.seed_list` is the canonical seed container.
- Single-seed commands expect one seed.
- Two-seed orchestration expects two seeds.
- `sim.seed` and `sim.seed_pair` are still supported for compatibility, but
  `seed_list` is the source of truth when present.

Path semantics:

- `io.results_dir_prefix` is rooted under `data/` unless it is already absolute.
- Single-seed results are written to `data/<results_dir_prefix>_seed_<seed>`.
- Analysis stage folders are assigned by `StageLayout` at runtime. Use
  `AppConfig` helpers such as `cfg.stage_dir("metrics")`,
  `cfg.metrics_input_path("metrics.parquet")`, and
  `cfg.head2head_path("bonferroni_pairwise.parquet")` instead of manual string
  concatenation.

See [docs/config_reference.md](docs/config_reference.md) for a fuller config
summary.

```yaml
# configs/fast_config.yaml
io:
  results_dir_prefix: results/fast

sim:
  n_players_list: [5]
  num_shuffles: 300
  seed_list: [42]
  n_jobs: 6
  expanded_metrics: false
  row_dir: data/results/fast_seed_42/rows

analysis:
  log_level: INFO

ingest:
  n_jobs: 4
  row_group_size: 64000

orchestration:
  parallel_seeds: false
```

Configuration overlays passed with `--config` are merged in order. Inline
overrides use dotted keys that match the dataclass structure, for example:

```bash
farkle --config configs/fast_config.yaml --set sim.n_jobs=12 --set sim.seed=123 run
```

## CLI Commands

```text
farkle [GLOBAL OPTIONS] <command> [COMMAND OPTIONS]
```

Global options:

- `--config PATH` loads one YAML overlay.
- `--set SECTION.OPTION=VALUE` overrides a single config field.
- `--log-level LEVEL` sets the root logging level.
- `--seed-a`, `--seed-b`, and `--seed-pair A B` override the active two-seed
  tuple for orchestration commands.

### `run`

Launch the tournament runner using `cfg.sim`.

- `--metrics` forces `cfg.sim.expanded_metrics = True`.
- `--row-dir PATH` writes full per-game rows to that directory.
- `--force` recomputes even when resumable run artifacts already exist.

```bash
farkle --config configs/fast_config.yaml \
  --set sim.seed=123 \
  --set sim.num_shuffles=200 \
  run --metrics --row-dir data/results/fast_seed_123/rows
```

### `time`

Benchmark simulation throughput.

- `--players INT` sets players per game.
- `--n-games INT` sets the number of simulated games.
- `--jobs INT` sets the parallel worker count.
- `--seed INT` sets the benchmark seed.

### `watch`

Interactively watch a single game.

- `--seed INT` locks the RNG for deterministic replays.

### `analyze`

Run analysis helpers against the current `AppConfig`.

Subcommands:

- `ingest`
- `curate`
- `combine`
- `metrics`
- `variance`
- `preprocess`
- `pipeline`
- `analytics`

Shared analysis flags:

- `metrics`, `preprocess`, and `pipeline` accept
  `--compute-game-stats`, `--rng-diagnostics`, `--rng-lags`,
  `--margin-thresholds`, `--rare-event-target`,
  `--rare-event-margin-quantile`, and `--rare-event-target-rate`.
- `pipeline` and `analytics` accept `--allow-missing-upstream`.
- `variance` accepts `--force`.

The current default single-seed stage layout resolves to:

```text
00_ingest
01_curate
02_combine
03_metrics
04_coverage_by_k
05_game_stats
06_seed_summaries
07_trueskill
08_tiering
09_head2head
10_seed_symmetry
11_post_h2h
12_hgb
13_variance
14_meta
15_h2h_tier_trends
16_agreement
17_interseed
```

Do not hard-code those numbers in scripts. Read `analysis/config.resolved.yaml`
or resolve paths through `AppConfig`.

```bash
farkle --config configs/fast_config.yaml analyze pipeline --compute-game-stats
```

### `two-seed-pipeline`

Run simulations and per-seed analysis for both entries in `sim.seed_list`, then
run the interseed comparison stages.

- `--force` recomputes even when completion markers exist.

```bash
farkle --config configs/fast_config.yaml two-seed-pipeline --seed-pair 42 43
```

Module execution is also available for the orchestration entry point:

```bash
python -m farkle.orchestration.two_seed_pipeline --config configs/fast_config.yaml
```

## Pipeline Metadata

The pipeline writes:

- `active_config.yaml` under each results root.
- `analysis/config.resolved.yaml` for the resolved analysis configuration.
- Append-only manifest files (`manifest.jsonl` by default).
- `.done.json` stage stamps carrying `config_sha`, `stage_config_sha`, and
  `cache_key_version`.

Cache reuse is stage-scoped. Unrelated config edits should not invalidate every
analysis stage.

## Direct Engine Usage

```python
from farkle.game.engine import FarkleGame, FarklePlayer
from farkle.simulation.strategies import ThresholdStrategy

players = [
    FarklePlayer("P1", ThresholdStrategy()),
    FarklePlayer("P2", ThresholdStrategy(score_threshold=450)),
]

game = FarkleGame(players)
summary = game.play()
print(summary.game)
```

## Repository Layout

- `src/farkle` - core package code
- `configs` - configuration presets
- `docs` - project documentation
- `tests` - unit and integration tests
- `notebooks` - exploratory analysis
- `data` - sample datasets and generated outputs

## Development

Install dev dependencies and run checks:

```bash
pip install -e .[dev]
ruff .
black --check .
mypy
pytest
```

Typing notes: `mypy` is configured to check `src/farkle` only. Tests are not
part of the strict typing target.

## License

This project is licensed under the Apache 2.0 License.
