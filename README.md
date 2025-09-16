# Farkle Mk II

+ Fast Monte Carlo engine built to stream 100M+ games with bounded RAM.
+ • CLI: unified `farkle` command with subcommands: run, time, watch, analyze
+ • Streaming Parquet shards (Snappy), atomic temp→rename, append-only manifest
+ • Resume-safe restarts; structured logs per worker

## Features
- Pure engine for single games (`engine.py`)
- Lookup table based scoring with Smart Five and Smart One helpers (`scoring.py`, `scoring_lookup.py`)
- Threshold strategy framework for roll or bank decisions (`strategies.py`)
- Batch simulation utilities for exploring strategy grids (`simulation.py`)
- Streaming output for large runs (``utils.parallel``)
- Command line interface: `farkle [--config FILE] <command>`
- Statistical helper to size experiments (`stats.py`)

## Installation
Requires Python 3.12 or newer.

```bash
pip install farkle
```

## Big Run Quick-Start (streaming & resume-safe)

Create a YAML file with the tournament parameters:

```yaml
# configs/tournament.yaml
n_players: 5
num_shuffles: 300
global_seed: 42
n_jobs: 6
checkpoint_path: data/checkpoints/seed_42.pkl
```

Then launch a run:

```bash
# Override selected values without editing the file
farkle --config configs/tournament.yaml \
  --set num_shuffles=500 \
  --log-level INFO \
  run --metrics --row-dir data/results_seed_42/rows
```

You can silence progress logs with `--log-level WARNING`.

Use the API directly:

```python
from farkle import FarklePlayer, ThresholdStrategy, FarkleGame

players = [
    FarklePlayer('P1', ThresholdStrategy()),
    FarklePlayer('P2', ThresholdStrategy(score_threshold=400))
]

game = FarkleGame(players)
metrics = game.play()
print(metrics)
```

## Strategy Variables
The `ThresholdStrategy` dataclass controls how a player decides
whether to keep rolling or bank points. Key options:

- `score_threshold` - minimum points to collect in a turn before banking.
- `dice_threshold` - bank when remaining dice fall to this number or lower.
- `consider_score` - enable the score threshold check.
- `consider_dice` - enable the dice threshold check.
- `require_both` - if true, wait for both score and dice conditions;
  otherwise stop when either triggers.
- `smart_five` - re-roll single fives when allowed by the thresholds.
- `smart_one` - re-roll single ones; valid only if `smart_five` is true.
- `favor_dice_or_score` - choose whether to favor the score or dice threshold when both are met.
- `auto_hot_dice` - automatically roll again if every die scores.
- `run_up_score` - in the final round, keep rolling even after taking the lead.

## Repository Layout
- `src/farkle` - core package code
- `tests` - unit and integration tests
- `notebooks` - sample notebooks and HTML reports
- `data` - small datasets used in examples
- `experiments` - configuration files for larger runs

## Code Checks
Install dev dependencies and run the linters and type checker:

```bash
pip install -e .[dev] mypy
ruff .
black --check .
mypy
```
## License
This project is licensed under the Apache 2.0 License.

## Project Usage
Run `farkle --config cfg.yml run` to simulate tournaments from a configuration file or use
the API as shown above. See the unit tests and module-level docstrings for more
examples.

## TrueSkill Ratings
Compute ratings for a directory of tournament results:

```bash
farkle --config analysis/pipeline.yaml analyze pipeline
```

This scans `data/results_seed_0` for blocks and writes rating files and
`tiers.json` to `data/results_seed_0/analysis` by default. The pipeline includes
`combine`, which merges curated per-seat Parquet shards into a single
`all_ingested_rows.parquet` superset used by downstream analytics.
