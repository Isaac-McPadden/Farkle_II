# Farkle_II
The fully tested, software packaged, upgraded version of my FarkleProject

A fast Monte Carlo engine and strategy toolkit for the dice game Farkle. This package provides:

A pure engine for simulating single games (engine.py).

Flexible scoring logic with smart-discard heuristics (scoring.py, scoring_lookup.py).

Strategy abstractions to decide when to roll or bank (strategies.py).

High-level batch simulation and grid-sweep utilities (simulation.py).

Command-line interface for configuring and running tournaments (farkle_cli.py, __main__.py).

Streaming I/O for writing large simulations to CSV without blowing memory (farkle_io.py).

Statistical helper to size experiments for given power (stats.py).

A clean public API surface and versioning (__init__.py).

File Overview

engine.py

Classes:

FarklePlayer: owns a threshold-based strategy (ThresholdStrategy), tracks score and per-turn stats, and implements the turn loop (take_turn).

FarkleGame: drives multiple players in rounds until someone reaches the target score and applies final-round rules.

Dependencies:

Imports default_score from scoring.py (which uses scoring_lookup.py).

Uses ThresholdStrategy from strategies.py to decide when to continue rolling.

scoring_lookup.py

Builds a table-driven lookup for scoring any combination of dice counts.

Defines fine-grained rule functions (straights, sets, singles) and a pipeline to produce a fast evaluate function.

Exposes build_score_lookup_table for instant lookup (used by scoring.py).

scoring.py

Provides pure, side-effect-free scoring:

score_roll_cached → raw score using the lookup table.

decide_smart_discards → optional Smart-5/Smart-1 heuristics to reroll low-value singles if they don’t force banking.

apply_discards → final score, dice used, and dice to reroll.

default_score → master function orchestrating the above.

strategies.py

Defines ThresholdStrategy: a dataclass encapsulating all roll/bank thresholds and flags (smart-fives, hot-dice, run-up, etc.).

Implements decide(...) to return whether to roll based on current turn state.

Convenience factories:

random_threshold_strategy for random grid sampling.

parse_strategy/__str__ for log parsing and round-tripping.

simulation.py

High-level utilities for batch sweeps:

generate_strategy_grid: build Cartesian products of thresholds and flags → list of ThresholdStrategy + metadata DataFrame.

simulate_one_game / _play_game: wrappers to run a single game with reproducible RNG seeds.

simulate_many_games: parallel batch runner returning a DataFrame.

aggregate_metrics: aggregate summary statistics from simulation results.

farkle_io.py

Streaming variant of batch simulation:

simulate_many_games_stream: producer/consumer pattern using multiprocessing.Pool and a writer process to append results to CSV incrementally.

Avoids holding large DataFrame in memory.

farkle_cli.py & __main__.py

Provides a console entry point (farkle run <config.yml>).

Loads YAML config with:

strategy_grid parameters for generate_strategy_grid.

sim parameters for streaming runner (n_games, out_csv, seed, n_jobs).

stats.py

games_for_power: calculates required games per strategy to detect a performance delta with given statistical power (Bonferroni or Benjamini–Hochberg correction).

__init__.py

Exposes the public API: FarklePlayer, GameMetrics, ThresholdStrategy, generate_strategy_grid, simulate_many_games_stream, and games_for_power.

Versioning support via pyproject.toml or package metadata.

How It All Connects

Scoring Layer: scoring_lookup.py builds raw-lookup tables. scoring.py uses these tables plus smart-discard heuristics to compute turn-by-turn outcomes without side effects.

Strategy Layer: strategies.py encapsulates roll/bank decision logic, reading current turn state and thresholds.

Engine Layer: engine.py wires players (with RNGs) and strategies to play full games, handling final-round rules and aggregating per-player metrics.

Simulation Layer: simulation.py leverages the engine to bulk-run games across large grids of strategies, producing tidy pandas.DataFrame results.

I/O Layer: farkle_io.py streams simulation results directly to CSV for memory efficiency. farkle_cli.py ties YAML-based configuration to streaming simulations in a user-friendly CLI.

Statistics: stats.py helps plan experiment sizes to achieve statistical significance.

With this modular design, you can mix and match components:

Use engine.py and scoring.py for interactive or real-time play.

Use simulation.py for offline Monte Carlo sweeps.

Swap in different ThresholdStrategy instances (or your own strategy implementations) without touching the engine.

Scale out via simulate_many_games_stream and the CLI for large-scale batch runs.

For detailed usage examples, refer to the docstrings in each module and the unit tests in tests/.

import farkle as far

Any of these work:
python -m farkle run cfg.yml
python -m farkle.cli run cfg.yml
farkle run cfg.yml               # installed entry-point


Dice threshold -> I must have at least n dice to keep rolling (Inclusive Down)

Score threshold -> I stop at this number or higher (Inclusive Up)

5.3 Stat-power vs player-count
n_games ≥ 2·(z_α + z_β)² · p(1-p) / δ²
