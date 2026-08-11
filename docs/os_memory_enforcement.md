# Aggregate OS Memory Enforcement

The public `farkle run`, `farkle analyze`, and `farkle two-seed-pipeline`
surfaces start through a lightweight supervisor. The supervisor establishes one
aggregate operating-system boundary before the real CLI imports numerical or
analysis modules. `process_tree_warning_threshold_mb` is the cooperative
high-water threshold. `aggregate_memory_hard_limit_mb` is passed directly to
the OS backend as the allocation-spike backstop; it is not derived from a
multiplier.

## Backends

- Windows uses a Job Object with `JOB_OBJECT_LIMIT_JOB_MEMORY` and
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. The real CLI is created suspended,
  assigned to the Job, and resumed only after assignment. Workers and
  grandchildren inherit the Job. Existing enclosing Jobs and nested-assignment
  failures are detected; an enclosing aggregate limit can only reduce the
  effective ceiling.
- Linux, containers, and WSL use a delegated cgroup-v2 child with `memory.max`
  and `memory.oom.group=1` where available. The supervisor enters the child
  cgroup before launching analysis, so descendants inherit membership. The
  effective limit is the minimum of the requested value and all detected
  enclosing `memory.max` values. A retained failed cgroup and its
  `memory.events` are reported for OOM diagnosis.

`RLIMIT_AS` is not used as a substitute because it is per-process and does not
provide the required aggregate process-tree contract.

## Configuration

Official defaults are strict:

```yaml
resources:
  scheduler_memory_budget_mb: 768
  process_tree_warning_threshold_mb: 768
  aggregate_memory_hard_limit_mb: 2304
  minimum_system_available_memory_mb: 1024
  parent_process_memory_mb: 192
  logical_cpu_budget: 0
  native_threads_per_worker: 1
  os_memory_limit_enabled: true
  os_memory_limit_required: true
  allow_unenforced_memory_fallback: false
```

Development fallback requires both `os_memory_limit_required: false` and
`allow_unenforced_memory_fallback: true`. It prints a conspicuous warning and
records `backend: unenforced`; it never claims that the ceiling was enforced.
OS settings and the detected backend/effective limit are execution provenance
and remain excluded from statistical freshness.

Before launch, explicit CPU and memory values are checked against detected
logical CPUs and physical memory. The configured host reserve must also be
currently available. The same reserve is checked at cooperative scheduling
boundaries. Enclosing Job/cgroup limits may reduce the effective hard limit;
requested, resolved, and effective values are separately recorded in the run
context and pipeline health metadata.

Strict setup failure exits with code 78 before the real analysis process starts.
An OS memory event exits the supervisor with code 86 when the backend exposes
enough evidence to classify it. The two-root health artifact is atomically set
to `running` before work begins, so abrupt termination cannot leave a new
successful pipeline marker. Existing authenticated units remain reusable;
temporary and unauthenticated files remain ineligible on resume.

## Bounded production-platform canary

Run the maximum-k, two-concurrent-root canary only in its dedicated directory:

```powershell
.\.venv\Scripts\python -m farkle.utils.os_memory --limit-mb 2304 -- `
  .\.venv\Scripts\python scripts\run_step_5_5_canary.py `
  --workspace data\step_5_5_canary_full --force
```

The canary uses two roots, `k=2` and maximum configured `k=12`, tiny authenticated
game limits/counts, and the real simulation-to-report orchestration. Its final
stdout and `pipeline_health.json` report the selected backend, effective limit,
peak sampled aggregate RSS, and peak sampled native-thread count.

The durable engineering invariants are byte-bounded streaming, atomic durable
units, authenticated manifests, and exact resume after interruption. A fixed
1 GiB process-tree ceiling is not an artifact-validity or statistical invariant.
