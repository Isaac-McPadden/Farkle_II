# Farkle Mk II

Farkle Mk II is a high-throughput Monte Carlo engine and analytics toolkit for
running large Farkle tournaments. It ships a unified CLI, streaming-friendly
artifacts, and statistical helpers for sizing experiments.

## Highlights

- Deterministic-friendly engine for single games (`src/farkle/game/engine.py`).
- Threshold strategy framework for roll-or-bank heuristics
  (`src/farkle/simulation/strategies.py`).
- Streaming parquet writers and append-only manifests that survive restarts.
- Unified `farkle` CLI with `run`, `time`, `watch`, and `analyze` subcommands.
- Config-driven analysis pipeline with metrics, TrueSkill, and head-to-head
  reporting.

## Installation

Requires Python 3.12 or newer.

```bash
pip install farkle
```

## Unified Configuration

All CLI workflows consume a single YAML document that maps to
`farkle.config.AppConfig`. The top-level keys correspond to dataclass sections:
`io`, `sim`, `analysis`, `ingest`, `combine`, `metrics`, `trueskill`,
`head2head`, and `hgb`. Missing values fall back to sensible defaults.

```yaml
# configs/fast_config.yaml
io:
  results_dir: data/results/fast
  append_seed: true  # append "_seed_<sim.seed>" to results_dir automatically

sim:
  n_players_list: [5]
  num_shuffles: 300
  seed: 42
  n_jobs: 6
  expanded_metrics: false
  row_dir: data/results/fast/rows

analysis:
  results_glob: "*_players"
  log_level: INFO

ingest:
  n_jobs: 4
  row_group_size: 64000
```

Analysis stage directories are numbered automatically by the resolved
``StageLayout``. Use the convenience helpers on :class:`farkle.config.AppConfig`
instead of string concatenation when you need a path (for example
``cfg.head2head_stage_dir`` or ``cfg.metrics_input_path("metrics.parquet")``).

Configuration overlays supplied with `--config` are loaded in order; inline
overrides use dotted keys that match the dataclass structure. For example:
`--set sim.n_jobs=12 --set io.append_seed=false`.

## CLI Commands

```text
farkle [GLOBAL OPTIONS] <command> [COMMAND OPTIONS]
```

**Global options**
- `--config PATH` - load an `AppConfig` from YAML. Used by `run` and `analyze`.
- `--set SECTION.OPTION=VALUE` - override a single field in the loaded config
  (coerced to the field type when possible). May be supplied multiple times.
- `--log-level LEVEL` - set the root logging level before executing the command.

### `run`

Launch the tournament runner using `cfg.sim`. The `--metrics` flag forces
`cfg.sim.expanded_metrics = True`; `--row-dir PATH` updates `cfg.sim.row_dir`
before execution. When `cfg.sim.n_players_list` contains more than one value the
runner sweeps each player count sequentially.

```bash
farkle --config configs/fast_config.yaml \
  --set sim.seed=123 \
  --set sim.num_shuffles=200 \
  run --metrics --row-dir data/results_fast/rows
```

### `time`

Benchmark simulation throughput using the defaults in
`farkle.simulation.time_farkle.measure_sim_times`. No additional command
options are parsed apart from the global logging level.

### `watch`

Interactively watch a single game. `--seed INT` locks the RNG for deterministic
replays.

### `analyze`

Wrapper around the analysis helpers that operate on streaming results. The
subcommands share the same `AppConfig` and read from its `analysis`, `ingest`,
`combine`, `metrics`, `trueskill`, `head2head`, and `hgb` sections. Stages write
into numbered directories under `analysis/`:

- `00_ingest` - convert raw CSV rows into parquet shards.
- `<idx>_curate` - finalize ingested shards and write manifests under `analysis/<idx>_curate` (index chosen automatically by :class:`~farkle.analysis.stage_registry.StageLayout`). Prefer ``cfg.stage_dir("curate")`` over manually concatenating folder numbers.
- `02_combine` - merge curated shards from `01_curate` into a consolidated parquet file.
- `03_metrics` - compute aggregate metrics, including pooled summaries.
- `04_game_stats` / `05_rng` - optional enrichments that keep their numeric
  slots even when skipped.
- `09_trueskill`, `10_head2head`, `11_hgb`, `12_tiering` - analytics stages that
  depend on upstream metrics.
- `pipeline` - run `ingest`, `curate`, `combine`, and `metrics` sequentially
  before branching into downstream analytics.

```bash
farkle --config configs/farkle_mega_config.yaml analyze pipeline
```

## Direct Engine Usage

The engine remains importable for bespoke experiments:

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
- `configs` - configuration presets for simulations and analysis
- `tests` - unit and integration tests
- `notebooks` - exploratory notebooks and reports
- `data` - small sample datasets and cached artifacts

## Development

Install dev dependencies and run checks:

```bash
pip install -e .[dev]
ruff .
black --check .
mypy
```

## License

This project is licensed under the Apache 2.0 License.
