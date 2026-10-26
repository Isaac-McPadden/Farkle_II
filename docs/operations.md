# Launch and resume operations

First-release operating configuration, qualified October 7, 2026.
Read [qualification history](development_history.md) and
[capacity evidence](implementation/task10_capacity.json)
before scheduling the full workload.

## Fixed scientific settings

The [mega configuration](../configs/farkle_mega_config.yaml) owns
roots 102/103, player counts 2/3/4/5/6/8/10/12, 5,160 strategies,
100 deterministic batches, 95% intervals and 2,000 bootstrap replicates.
The planner resolves 4,300 shuffles per root/k: 39,013,900 tournament games
and 177,504,000 player exposures per root. Roots and player counts remain
sequential. Safety stops count against attempted tournament performance and
remain outside completed outcome distributions and H2H score-test counts.

H2H retains family alpha 0.02, detection power 0.80 at the planned 0.03
alternative, practical threshold 0.03, equivalence tolerance 0.03,
minimum completion rate 0.99 and replacement multiplier 2.0. The maximum
150-candidate envelope has 11,175 pairs and 44,700 root/order blocks,
each targeting 2,191 completed games. This is 97,937,700 completed H2H
games and at most 195,875,400 attempts. The actual frozen family determines
the actual budget. Equivalence tolerance does not promise equivalence power.

The configured H2H cap remains 100,000,000 attempts. An above-cap frozen plan
requires its own approval; the configured cap is never automatically raised.
In a terminal, Y persists authorization for that exact plan and N stops with
`blocked_by_cap` (exit 3). Noninteractive execution without an exact approval
also exits 3 before H2H games start. Tournament/root analysis work completed
before that stop remains resumable. Do not use the envelope projection as an
authorization token.

## Execution and storage

The machine profile requests 16 simulation/analysis workers, eight ingest
workers and automatic H2H workers. The 15-logical-CPU policy admits at most
15 simulation/analysis/H2H workers before pending-unit or memory constraints.
Native threads per worker are one. Byte-bounded readers, queues, checkpoints
and nested-executor suppression remain active.

The strict aggregate Windows Job Object limit is 12 GiB. Scheduler admission
budget and RSS warning are each 8 GiB; host available-memory reserve is 8 GiB.
Those have different meanings. Job committed-memory peak is distinct from RSS.
Boundary setup failure exits 78; an enforced memory limit exits 86. Retain
the default `enabled` resource mode for the qualified operating path.

Results are on S: under the synchronized project data directory. The selected
example below uses a new, explicit prefix because the default
`data/results_seed_pair_102_103` directory already exists. Its existing contents
were not inspected or reused for qualification. Existing incompatible runs
must fail freshness checks; preserve them. Avoid `--force` as a way to bypass
a code/config identity conflict.

For the qualified cross-volume storage option, choose an absolute C: scratch
directory before the first launch and keep it fixed. The CLI authenticates it
in the active configuration and execution context. Data and metadata stage on
scratch, cross-volume copies use bounded destination-local atomic publication,
and completion follows data and sidecar publication. Scratch capacity must be
reviewed alongside durable S: capacity; retained canonical units remain on S:.
The mega config's default scratch remains null; these are explicit launch
overrides, not edits to that configuration.

The high-cost durable scenario uses about 571 GiB before publication/scratch
headroom and other disk activity. Recheck both volumes before launch; historical
free-space measurements reserve no capacity and do not prove adequate space
for all empirical support/compression.

## Future launch and ordinary resume

Finalize a clean, immutable checkout of the intended release first. Changing source,
tests, configuration or other fingerprinted files during a run changes its
authenticated identity. Use the root Python 3.12 venv and explicitly select the
immutable checkout's source; the venv's editable installation otherwise points
at the working tree. The qualified temporary snapshots are evidence, not the
user's finalized release checkout.

The following PowerShell example is documentation only. Set `releaseRoot` to
the finalized checkout; review disk space and the new result prefix first.
Keep the same checkout and options for every ordinary resume.

```powershell
$releaseRoot = 'C:\path\to\finalized-clean-checkout'
$projectRoot = 'S:\Libraries\OneDrive\Documents\Code Projects Parent Folder\Code Projects\Farkle Mk II'
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path $releaseRoot 'src'
Set-Location -LiteralPath $releaseRoot
$launchArgs = @(
    '-m', 'farkle',
    '--config', (Join-Path $releaseRoot 'configs\farkle_mega_config.yaml'),
    '--set', ('io.results_dir_prefix=' + (Join-Path $projectRoot 'data\results_first_release_20261007')),
    '--set', 'io.temp_dir=C:\FarkleScratch\first_release_20261007',
    'two-seed-pipeline'
)
& $python @launchArgs
```

Ordinary resume repeats that command without `--force`. The orchestrator checks
the existing pair's authenticated public configuration and code identity before
publication. Valid durable units resume at their exact coordinates; no-force
replay preserves stable products. Inspection of `pipeline_health.json`,
`pipeline_telemetry.json` and the append-only pipeline manifest locates a stop
without treating partial staging files as completion inputs.

`Pipeline progress summary` records completed work for a simulation or an
analysis partition phase. The JSON fields after ` | ` identify its `stage` and
`scope`, completed/reused units, and worker counts. A stage can have several
partition phases and therefore several summaries; these are operational
measurements, not statistical findings or parent-stage completion stamps.
The same field formatting exposes details on workload, publication, and
finalization messages. Heartbeats describe current work, while the summaries
record completed phase totals. Retained summaries also appear under
`timing_summary.telemetry.completed_progress` in `pipeline_health.json`.
Spawned Windows workers do not inherit the parent's INFO logging handlers;
worker exceptions propagate to the parent, and the parent-owned summaries and
heartbeats provide progress reporting. Log levels and progress intervals still
control which messages are emitted. At INFO, heartbeats keep their readable
single-line form, completion summaries show phase totals, and cleanup/timing
messages omit bulk histories and resource snapshots. Full telemetry remains in
the JSON artifacts; DEBUG logging also includes the full attached fields.
Final release-audit failures are written explicitly to the log with their
artifact paths, as well as retained in `pipeline_health.json`.
Heartbeat `workers=live/created/resolved` counts refer to the current executor;
`requested=0` means automatic worker selection. After process pool shutdown, live
workers become zero while the created count remains available. Intentional
serial execution runs in the parent and creates no worker processes.

For an unattended above-cap stop, read the current immutable power plan's
`schedule_hash` and `maximum_total_attempts`. Explicit approval is
`--h2h-authorize PLAN_SHA256:MAX_ATTEMPTS`, placed before `two-seed-pipeline`.
Approve only the reviewed actual plan. Persisted matching approval is reused
on later resumes, including an unattended replay without the flag. Missing,
partial, altered or mismatched approval is not permission to continue.

Success requires the fresh final root/root/pair byte-deep audit and
`complete_success` health. Bounded integration products remain non-production
and non-release even when that audit passes. For a scientific release, verify
the finalized clean code identity, calculated production qualification, and
`release_eligible: true` in the final audit. Pre-audit reports show
`release_status: pending_final_audit`; unmet requirements and release
ineligibility reasons identify the settings or evidence needing attention.

## Limits that remain

Selected completed-outcome intervals can be withheld for sparse support,
insufficient contributing batches, undefined resamples or degeneracy. Exact
frequency counts and honest support statuses remain useful in those cases.
The default exhaustive lag diagnostic is enabled; deliberate omission removes
only that optional work and leaves preflight and lightweight monitoring required.
Monitor warnings and two-root agreement do not prove independence. No automatic
extra games or automatic diagnostic stopping rule is added.

Broad opponent-composition sensitivity, fixed-matchup entropy, state-level
decisions, prescribed strategy rankings and an automatic figure suite retain
their exclusions. Performance and storage projections are conditional estimates;
qualification does not authorize optimization, change the scientific scope or
start the full run.
