You are assisting with a Monte Carlo simulation for determining the best strategy or strategies in the dice game farkle.

Secondary goals are to produce professional software that performs the simulation, writes and pipes the data, and performs rigorous statistical analysis of game outcomes with randomized match groupings.

When operating in VSCode, use the venv located in the root as it has all dependencies installed and uses Python 3.12.

## General Principles

- **Deterministic:**  
  All randomness must be controlled through explicit seeds from config files. No hidden RNG. Avoid using Python's included random module as the RNG algorithm they use does not scale well.

- **Idempotent:**  
  Analyses should skip recomputation if outputs already exist *unless* a `--force` flag is passed.

- **Resumable:**  
  Long-running operations must write periodic checkpoints using the `atomic_path` helper.  
  Never leave partial files.

- **Stable File Paths:**  
  Write outputs to predetermined paths based on config (`results_dir`, `analysis_subdir`, `io.*`).  
  Never guess locations.

- **Use Existing Helpers:**
  There are many reusable tools in src/farkle/utils/ that should be checked before implementing ad-hoc helpers for basic and general computation processes (reads, atomic writes, logging, rng, streaming, etc)

- **Parallel Process When Sensible:**
  If it is possible to process something in parallel, always do so unless there is a logical reason not to. 
  For example: In run_trueskill, order matters within a set of games between k players so that should not be performed in parallel.  However, there can be multiple k's for game size so different sets of games between different numbers of k players can and should be performed in parallel.

- **Limit RAM Usage:**
  Always prefer streaming small amounts of data when processing.  Do not put a whole data set in memory, run calculations, then write it all out at the end unless the data set and calculations are tiny (if you need a heuristic, less than 5 seconds of processing time).  Streaming reads and writing atomically have helper functions in src/farkle/utils/ to assist with this in addition to their usefulness in resumability.