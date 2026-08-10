# Aggregate OS Memory Enforcement

The public `farkle run`, `farkle analyze`, and `farkle two-seed-pipeline`
surfaces start through a lightweight supervisor. The supervisor establishes one
aggregate operating-system boundary before the real CLI imports numerical or
analysis modules. The configured process-tree target is a cooperative warning
threshold; the OS boundary is the allocation-spike backstop at
`target_memory_mb * memory_safety_factor`.

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
  target_memory_mb: 768
  memory_safety_factor: 3.0
  os_memory_limit_enabled: true
  os_memory_limit_required: true
  allow_unenforced_memory_fallback: false
```

Development fallback requires both `os_memory_limit_required: false` and
`allow_unenforced_memory_fallback: true`. It prints a conspicuous warning and
records `backend: unenforced`; it never claims that the ceiling was enforced.
OS settings and the detected backend/effective limit are execution provenance
and remain excluded from statistical freshness.

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
