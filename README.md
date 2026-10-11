# Farkle Mk II

A deterministic Monte Carlo simulation and statistical analysis toolkit for
comparing a finite configured grid of Farkle strategies. The PC has played
**roughly 1.7 billion games**; see the [estimate and counting basis](docs/game_count.md).

## Install

Use Python 3.12 or newer. Local development uses the root venv:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

## Use

All workflows load validated YAML into `farkle.config.AppConfig`.
The fast preset is a reduced-resolution **two-root integration profile**:

```powershell
farkle --config configs/fast_config.yaml two-seed-pipeline
```

`run` and `analyze` require exactly one root in `sim.seed_list`.
`two-seed-pipeline` runs both root workflows, then one pair workflow.
Long operations publish atomically and resume compatible work without
`--force`. Read [operations](docs/operations.md) before a full workload.

## References

- [Documentation index](docs/index.md), [CLI](docs/cli.md) and [rules](docs/rules.md).
- [Configuration](docs/config_reference.md), [statistical methods](docs/statistical_methods.md)
  and [data artifacts](docs/data_artifacts.md).
- [RNG](docs/rng_contract.md), [turn/row contract](docs/turn_and_row_contract.md)
  and [resource controls](docs/os_memory_enforcement.md).
- [Contributor orientation](docs/codex_context/context_prompt.md),
  [testing/review](docs/codex_context/testing_and_review_map.md) and
  [development history](docs/development_history.md).

## Core contracts

Full-coordinate NumPy PCG64DXSM streams own randomness. Simulation and analysis
are atomic, idempotent, resumable and byte bounded. Paths come from AppConfig;
canonical derived artifacts require compatible adjacent sidecars.

Primary tournament rates use all attempted player exposures; completed outcomes
retain distinct conditioning. Complete configured support and explicit weights
are required across player counts. Screening, prediction, H2H inference,
dominance and display order remain separate.

## Development

Source lives under `src/farkle/`: game, simulation, analysis, orchestration,
CLI and shared utils. Runnable presets are in `configs/`; unit/integration
oracles are in `tests/`; maintained checks and benchmarks are in `scripts/`.
Exploratory notebooks are historical research records.

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m black --check .
.\.venv\Scripts\python -m mypy src
.\.venv\Scripts\python -m pyright
.\.venv\Scripts\python scripts/check_rng_contract.py
.\.venv\Scripts\python scripts/check_terminology.py
```

See the testing map for focused targets and the explicit-artifact-root release
audit. Runtime-read recipes, frozen protocols and test-dependent historical
datasets are retained; do not delete them as task debris.

## License

Apache License 2.0.
