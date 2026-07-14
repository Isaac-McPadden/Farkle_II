# Farkle Mk II

Farkle Mk II is a deterministic Monte Carlo simulation and statistical
analysis toolkit for comparing Farkle strategies over a finite configured
strategy grid.

## Core contracts

- NumPy `PCG64DXSM` streams are derived from stable experiment coordinates.
- Simulation and analysis work is resumable and written atomically.
- Paths come from `AppConfig`; consumers validate canonical scope and sidecars.
- Large row sets are streamed or partitioned.
- Chance baselines are `1/k` for k-player games.
- Cross-k performance requires complete configured support.
- Screening, H2H inference, dominance, and display ordering remain distinct.

See [configuration](docs/config_reference.md),
[artifacts](docs/data_artifacts.md), [RNG](docs/rng_contract.md), and
[turn/row](docs/turn_and_row_contract.md) for the full contracts.

## Installation

Python 3.12 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

## Configuration

All workflows load YAML into `farkle.config.AppConfig`. Current top-level
sections are:

`io`, `sim`, `analysis`, `ingest`, `combine`, `trueskill`, `head2head`, `hgb`,
`orchestration`, `rng`, `screening`, `batching`, `robustness`,
`artifact_contract`, and `k_aggregation`.

Example:

```yaml
sim:
  seed_list: [42, 43]
  n_players_list: [2, 4]

screening:
  practical_delta_by_k: {2: 0.03, 4: 0.03}
  delta_across_k: 0.03

k_aggregation:
  method: equal-k
```

Unknown and retired keys are rejected with migration guidance. Inline
overrides use current dotted keys:

```powershell
farkle --config configs/fast_config.yaml --set sim.n_jobs=8 run
```

## CLI

```text
farkle [GLOBAL OPTIONS] <command> [COMMAND OPTIONS]
```

Commands:

- `run`: simulate the configured root and player counts.
- `time`: benchmark simulation throughput.
- `watch`: replay one interactive deterministic game.
- `analyze ingest|curate|combine|metrics|preprocess`: run a focused root stage.
- `analyze pipeline`: run the complete standalone-root workflow, including its
  explicitly labelled H2H tail.
- `analyze analytics`: run the same canonical standalone-root analysis from
  existing inputs.
- `two-seed-pipeline`: run both root workflows, then execute the root-pair tail
  exactly once at the pair analysis root.

Global root overrides are `--seed-a`, `--seed-b`, and `--seed-pair A B`.
Long workflows accept `--force` where recomputation is supported.

## Workflow ownership

The root workflow is:

1. ingest;
2. curate;
3. row concatenation;
4. all-player metrics and performance;
5. game, seat, RNG, and roll diagnostics;
6. TrueSkill and HGB;
7. descriptive screening;
8. stop.

The root-pair workflow is:

1. raw-count root combination and stability;
2. TrueSkill candidate contribution;
3. candidate freeze;
4. H2H power planning and block execution;
5. seat-adjusted inference;
6. dominance digestion;
7. agreement;
8. reporting.

A standalone root appends the same H2H tail under its own `h2h_2p` scope and
labels outputs `single_root`. A two-root run never executes H2H independently
inside either root workflow.

## Artifacts

Canonical derived scopes are `by_k`, `concat_ks`, `across_k`, `cross_seed`,
`diagnostics`, and `h2h_2p`. Every derived artifact has exactly one adjacent
hash-bound sidecar. The only lifecycle states are `not_started`,
`partial_resumable`, `complete_valid`, `complete_stale`, and
`blocked_by_cap`.

Old on-disk results may remain for inspection. Current consumers ignore them;
the migration report inventories them without deleting user data.

## Development

Use the repository virtual environment:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m black --check .
.\.venv\Scripts\python -m mypy src
.\.venv\Scripts\python -m pyright
.\.venv\Scripts\python scripts/check_terminology.py
.\.venv\Scripts\python scripts/check_structure_release.py
```

The interrupted/resumed two-root structural oracle is
`tests/integration/test_structure_toy_oracle.py`.

## Repository layout

- `src/farkle/game`: rules and game engine
- `src/farkle/simulation`: workload planning and simulation
- `src/farkle/analysis`: canonical estimators, diagnostics, H2H, and reports
- `src/farkle/orchestration`: root and root-pair contexts
- `src/farkle/utils`: RNG, atomic I/O, sidecars, manifests, and shared helpers
- `configs`: validated runnable presets
- `tests`: unit and integration oracles
- `docs`: contracts and Codex orientation

## License

Apache License 2.0.
