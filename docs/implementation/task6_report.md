# Task 6 — interactive H2H cap authorization

Contract: September 24 `first_release_task0_v1`; implementation backlog Task 6.
Validation evidence is recorded in `task6_validation.json`.

## Behavior and ownership

The immutable H2H power plan now completes independently of operational consent.
Its completed-game targets, maximum replacement-attempt allocations, candidate
family, schedule hash, RNG coordinates, and bytes are unchanged by approval.
The H2H executor resolves consent before submitting any games. The CLI displays
target completed games, maximum attempts including safety replacements, the
configured cap, percentage of cap required, and percentage over cap. A 10,300
attempt budget against 10,000 displays 103.00% required and 3.00% over.

Only an exceeded cap prompts, with the specified text:
`Override the cap for this H2H plan and continue? [Y/N]`.
Y/y authorizes exactly the frozen maximum. N/n and blank input decline; invalid
input repeats the prompt. EOF and keyboard interruption stop without approval.
The CLI requires terminal stdin for prompting. Library calls are noninteractive
by default; callers can scope the same explicit policy with
`cap_authorization_policy`.

Unattended execution supplies the global option
`--h2h-authorize PLAN_SHA256:MAX_ATTEMPTS` before `two-seed-pipeline` or
`analyze pipeline|analytics`. The blocked run prints its exact token. A different
hash or amount cannot approve this plan. Consent is scoped to the invocation
and never enters public YAML, config hashing, or RNG identity. Resume uses the
same original configuration; no cap edit or `--force` is needed.

Refusal and unavailable input use `H2HAuthorizationRequired`, stage/run telemetry
and pipeline health identify `blocked_by_cap`, and the CLI exits 3 without a
traceback. Downstream inference/reporting do not execute after the stop, even
when a stage runner otherwise allows continuation on error. Completed upstream
work and authenticated H2H blocks remain available for exact resume.

## Durable authorization lifecycle

The new non-mutating AppConfig getter resolves
`cfg.h2h_2p_dir('h2h_execute') / 'cap_authorization.json'`. This is a derived
artifact in the existing `h2h_2p` scope with one adjacent authenticated sidecar,
owned by execution. Its schema version is 1, with:

- `family_hash`, `schedule_hash`, `total_block_count`, `maximum_total_attempts`;
- `power_plan_sha256` and `block_manifest_sha256` for exact immutable source bytes;
- `configured_total_game_cap`, `authorized_total_game_cap`;
- `authorization_source`: `configured_cap`, `interactive`, or `explicit_override`;
- `execution_authorization`: `ready` or `blocked_by_cap`.

Within-cap permission uses the configured amount (null retains existing
unlimited semantics). An override's amount must be an integer exactly equal to
the frozen maximum, not an unlimited or global grant. Sidecars additionally bind
canonical ownership, code/config identity, method identity, and exact sources.
Current authenticated run lineage and supplied code identity are checked on reuse.
An authenticated record for a different family/plan fails closed.

Authorization publishes through the existing atomic data/sidecar helper and is
read back before execution. An interruption after data but before a compatible
sidecar leaves no valid approval; resume requires fresh consent above the cap.
An interruption after both publish reuses consent even if execution state has
not yet refreshed. No completed blocks are deleted in either case.

Planning replay and every state checkpoint consult the authorization artifact.
Refresh retains all existing progress fields, including with partial work, and
never replaces explicit approval with the unchanged preset cap. Execution state
records `cap_authorization_sha256` and has the authorization as a source artifact.
Even renewed consent for the same amount refreshes that exact source binding.
Damaged state publication can be reconstructed from independently authenticated
block prefixes; damaged or missing authorization cannot be reconstructed from
progress. The existing per-block storage and deterministic replacement algorithm
remain unchanged.

Execution completion owns the authorization artifact alongside state, aggregate
counts, and blocks. Snapshot inventories retain it. The fresh byte-deep release
audit checks exact plan bindings, sufficient authorized amount, agreement with
execution state, and authorization provenance. Missing final execution stamps
can still be finalized without replaying valid block data.

The execution cache-key version advances from 5 to 6 in both root-pair and
single-root layouts. H2H statistical method version 2, power cache version 5,
and all scientific/RNG identities remain unchanged. Existing code/config
freshness rules still apply; this does not migrate historical artifacts.

## Validation scope

Focused coverage includes under/equal/over-cap cases; the 103% example; lowercase,
invalid, blank, N, EOF, interruption, and noninteractive input; explicit token
matching; repeated planning and execution; missing consent with partial work;
state/source corruption; authorization publication boundaries with and without
partial execution; preserved block bytes/timestamps; and byte-deep audit.
CLI policy reset, exit status, stage stopping, and pair health are also tested.

The raw simulation-to-report oracle uses a preset cap of 23 against its frozen
24-attempt plan, accepts one interactive approval, runs required preflights and
batch monitors, and exercises final root/root/pair byte-deep audit. Its ordinary
no-force replay has no interactive policy, proving persisted authorization reuse
and stable workflow bytes. This is a bounded validation fixture, not production.

The older structural toy oracle required a fixture correction: its pair config
lacked root RNG stage mappings and required Task 5C evidence. It now supplies
explicit `insufficient_evidence` synthetic inputs through the existing reporting
test helper. It does not claim to run preflight or establish RNG capability;
the full raw oracle continues to run real preflight.

Initial development checks exposed the v3-specific exception type at interrupted
approval publication; recovery now handles both artifact contract variants.
Source edits during early working-tree tests also invalidated dirty-code
identities. Final qualification therefore uses independent frozen checkouts and
the repository Python 3.12 venv. No working-repository commit, reset, or cleanup
of existing changes is performed.

Final validation passed:

- 192 focused regression cases in 477.192 seconds.
- 28 expanded authorization boundary cases in 135.236 seconds. These overlap
  the focused suite and are not an additional 28 unique cases.
- The full raw simulation-to-report oracle, final audit, and no-force replay
  passed in 1,292.647 seconds (1,295.868 seconds including pytest overhead),
  within the 3,840-second ceiling. Required preflight calls used 546.764 seconds.
- Ruff (`src`, `tests`, and `scripts`), Black for 15 touched Python files,
  Mypy for all 95 source files, Pyright for six affected runtime modules,
  public CLI help, and `git diff --check` passed.

All 95 production Python files match the three frozen validation checkouts.
The companion validation JSON records exact snapshot commits and evidence
hashes. The full oracle emitted only the existing pytest `record_property`
compatibility warning; its measured properties are present in the JUnit XML.
All pre-existing Task 0/9/1A/1B/2/3/4/5A-5C changes remain. No production simulation,
sampling-budget increase, or production cap override is authorized or executed.
Task 7 is the next implementation task; broader integrated capacity and launcher
qualification remains Task 10.

The repository terminology command reports four pre-existing uses of `pooled`
in `outcome_publication.py` (lines 995, 1031, 1080, and 1186). That Task 1B/2
source is unchanged by Task 6; one finding is a durable unit directory name,
so this task does not rename it as cosmetic cleanup. This separate repository
gate is not represented as passing.
