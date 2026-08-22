# Task 5A bounded production-capacity benchmark plan

Status: published before Task 5A measurement. This is a diagnostic plan, not a
production-run authorization.

## Ownership and bounds

The driver owns only `data/farkle-task5a-production-capacity-v1` and requires
`.task5a-owned.json` before reuse or cleanup. It never writes beneath a configured
production, fast-integration, or historical result root. Per-case JSON checkpoints
are written atomically and are reused unless `--force` is supplied. The final
evidence root is retained for review.

The synchronized OneDrive repository tree is the primary location. Task 3A's
measured 6.3% RNG and 2.3% H2H synchronized-tree penalties remain the accepted
storage decision; no local-working-root feature is part of Task 5A.

## Cases and coefficient identification

| Case | Scale/repetitions | Expected wall/RAM/disk | Coefficient identified |
| --- | --- | --- | --- |
| Current executable-plan derivation | Production YAML and code; one deterministic resolution | under 30 s / under 1 GiB / negligible | strategies, roots, `k`, shuffles, games, exposures, row groups, route units, reducers, bootstrap units, candidates, H2H pairs/blocks/checkpoints |
| Task 4D integration reconciliation | Read-only metadata and telemetry from seeds 54/55 | under 30 s / under 1 GiB / no writes | real stage fixed terms, row-group and record anchors, worker topology, RAM, artifact/open/hash inventory |
| Simulation real-path anchors | Accepted Task 4C 1/2/12-worker cases plus both Task 4D roots and all integration `k` | no rerun; measured evidence reuse | pool startup/shutdown, games and player-exposure throughput, worker lifecycle, CPU and RAM |
| RNG source-unit sweep | Accepted Task 3B 256/1,024 row groups, route widths 1/16/32, two repetitions | no rerun; measured evidence reuse | source-unit overhead, route publications, reducer opens, spill creation, routing and reduction |
| RNG production-volume anchor | Both Task 4D roots, 1,800 real row groups, route size 32, real schemas and algorithms | no rerun; measured evidence reuse | count/statistics record density and combined production implementation throughput |
| RNG four-pass topology | Deterministic fan-in-32 topology calculation using current spill policy and production record counts | under 1 s / negligible / no durable data | actual production merge depth and per-pass input/output counts; prevents one/two-pass-only extrapolation |
| H2H cardinality and game cases | Accepted Task 4A targets 1,372/1,974/2,191 plus exceptional completion cases; Task 4D 144-block real integration | no rerun; measured evidence reuse | game execution, attempts/completion, fixed block/checkpoint/publication, pool lifecycle, inference tail |
| Authentication file/byte sweep | Existing Task 4B 9/128/512-artifact cases and Task 4D 4,004-artifact final audit | no rerun; measured evidence reuse | artifact/open and byte-hash costs, snapshot and final-audit terms |
| Model validation | Predict the observed 1,663.156 s Task 4D run from held stage anchors | under 1 s / negligible | residual and interval calibration |

The accepted cases were produced by the current production implementations; their
commits are ancestors of the current clean branch. Task 5A records that provenance
instead of rerunning an expensive Cartesian matrix. The only new root contains
machine-readable derivation/model checkpoints, not simulated scientific results.

## Stop criteria

The model must keep fixed/source-unit/record/game/block/artifact/byte terms separate,
reproduce the fast integration within its declared tolerance, and publish a broad
interval when production-scale extrapolation dominates. Failure to validate yields
`indeterminate_more_evidence_required`. A conservative projection beyond 48 hours
yields `not_capacity_ready`; Task 5B, optimization, production execution, and any
statistical-scope change remain out of scope.
