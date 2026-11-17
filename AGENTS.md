# Farkle Mk II

Compare configured Farkle strategies with deterministic Monte Carlo simulation
and statistical analysis of randomized match groupings.

## Engineering invariants

- Derive every RNG stream directly from explicit configuration seeds and full
  semantic coordinates using `PCG64DXSM`. Preserve stream identity across
  worker counts, scheduling, and resume.
- Resolve output paths through `AppConfig` (`results_dir`, `analysis_subdir`,
  `io.*`). Reuse helpers in `src/farkle/utils/` for I/O, logging, RNG, and
  streaming.
- Reuse valid outputs unless `--force` is passed. Enforce freshness checks
  and preserve incompatible artifacts.
- Stream large inputs within byte bounds and publish outputs atomically. Use
  `atomic_path` for periodic checkpoints and preserve exact resume from durable
  units. Treat process-tree memory ceilings as operational controls.
- Run roots and player counts sequentially. Parallelize independent stage
  units within resource and ordering constraints.

## Development and references

Use the root Python 3.12 `.venv`. The project uses Ruff, Pyright/Pylance,
Mypy for `src/`, and Black when formatting is needed.

- To locate source ownership or understand pipeline, statistical, or recovery
  contracts, use the [contributor orientation](docs/codex_context/context_prompt.md).
- Select checks appropriate to the change. For focused tests and release gates,
  use the [testing and review map](docs/codex_context/testing_and_review_map.md).
- Before changing retained protocols, fixtures, or evidence, check the
  [protected dependencies](docs/codex_context/metadata.md#protected-dependencies).
