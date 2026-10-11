"""Reproducible Task 9 count-history reduction; no simulation or historical inputs.

Default fixture: 5,160 strategies x 12 seats x 100 batches = 6,192,000 rows.
The aggregation runs in a fresh process. The declared allowance is 128 MiB
incremental RSS above imported-process baseline, with 32 MiB batch/state
allowances. This is a workload qualification, not a universal RSS invariant.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.seat_analysis import _COUNT_SCHEMA, _stream_within_k_frames
from farkle.utils.writer import atomic_path


def _measure(path: Path, batch_bytes: int, budget: int) -> dict[str, object]:
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.wait(0.005):
            peak = max(peak, process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        effects, population = _stream_within_k_frames(path, 12, batch_bytes)
        peak = max(peak, process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    elapsed = time.perf_counter() - start
    rows = pq.ParquetFile(path).metadata.num_rows
    batches = rows // (5160 * 12)
    assert len(effects) == 5160 * 12 and len(population) == 12
    assert (effects["raw_exposures"] == 12 * batches).all()
    assert (effects["raw_wins"] == batches).all()
    assert (effects["raw_safety_limit_exposures"] == 2 * batches).all()
    assert population["raw_exposures"].sum() == rows * 12
    result = {
        "fixture_rows": rows,
        "strategies": 5160,
        "seats": 12,
        "deterministic_batches": batches,
        "batch_and_state_allowance_bytes_each": batch_bytes,
        "declared_incremental_rss_budget_bytes": budget,
        "baseline_rss_bytes": baseline,
        "peak_sampled_rss_bytes": peak,
        "incremental_rss_bytes": peak - baseline,
        "sampling_interval_seconds": 0.005,
        "elapsed_seconds": elapsed,
        "exact_count_checks_passed": True,
        "within_budget": peak - baseline <= budget,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-mib", type=int, default=32)
    parser.add_argument("--budget-mib", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.batches, args.batch_mib, args.budget_mib) < 1:
        parser.error("batch counts and byte allowances must be positive")
    if args.worker:
        result = _measure(args.worker, args.batch_mib * 2**20, args.budget_mib * 2**20)
    else:
        with TemporaryDirectory(prefix="farkle-task9-") as temporary:
            path = Path(temporary) / "seat_counts.parquet"
            strategies = np.repeat(np.arange(5160, dtype=np.int32), 12)
            seats = np.tile(np.arange(1, 13, dtype=np.int16), 5160)
            count = len(strategies)
            with pq.ParquetWriter(path, _COUNT_SCHEMA) as writer:
                for batch in range(args.batches):
                    values = [
                        np.full(count, 11),
                        np.full(count, 12),
                        np.full(count, batch),
                        strategies,
                        seats,
                        np.ones(count, dtype=np.int64),
                        np.full(count, 12),
                        np.full(count, 10),
                        np.full(count, 2),
                    ]
                    table = pa.Table.from_arrays(
                        [
                            pa.array(value, type=field.type)
                            for value, field in zip(values, _COUNT_SCHEMA, strict=True)
                        ],
                        schema=_COUNT_SCHEMA,
                    )
                    writer.write_table(table)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(path),
                    "--batch-mib",
                    str(args.batch_mib),
                    "--budget-mib",
                    str(args.budget_mib),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(str(args.output)) as staged:
            Path(staged).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
