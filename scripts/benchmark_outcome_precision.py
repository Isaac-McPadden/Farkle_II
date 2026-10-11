"""Bounded first-release Task 3 fixture; no production games or target changes.

By default: 120 strategies, k=12, 30 certified batches, four shuffles per
batch, 2,000 replicates, a 16 MiB analysis allowance and a declared 128 MiB
incremental RSS measurement allowance. Source/outcome preparation is excluded
from measurement. This measures finite-fixture resource use, not capacity.
"""

from __future__ import annotations

import argparse
import gc
import json
import threading
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import psutil
import pyarrow.parquet as pq
from tests.helpers.artifact_sidecars import make_authenticated_v3_config
from tests.helpers.outcome_fixture import publish_outcome_source
from tests.unit.analysis.test_outcome_distributions import _row

from farkle.analysis.outcome_precision import build_outcome_precision, precision_is_up_to_date
from farkle.analysis.outcome_publication import build_outcome_distributions
from farkle.config import assign_config_sha
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.writer import atomic_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", type=int, default=120)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--shuffles-per-batch", type=int, default=4)
    parser.add_argument("--batch-mib", type=int, default=16)
    parser.add_argument("--budget-mib", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.strategies < 12 or args.strategies % 12 or args.batches < 30:
        parser.error("strategies must be a positive multiple of 12; at least 30 batches")
    if (
        args.shuffles_per_batch < 1
        or args.batches * args.shuffles_per_batch * args.strategies // 12 > 20000
    ):
        parser.error("positive shuffles per batch and at most 20,000 fixture games required")
    with TemporaryDirectory(prefix="farkle-task3-benchmark-") as temporary:
        cfg = make_authenticated_v3_config(Path(temporary), player_counts=(12,))
        cfg.sim.row_dir = Path("rows")
        cfg.analysis.n_jobs = cfg.ingest.n_jobs = 1
        cfg.screening.bootstrap_replicates = 2000
        cfg.resources.stage_batch_bytes["analysis"] = args.batch_mib * 2**20
        assign_config_sha(cfg)
        games, width = args.strategies // 12, args.shuffles_per_batch
        roster = tuple(range(args.strategies))
        rows = []
        for shuffle in range(args.batches * width):
            for game in range(games):
                ids = list(roster[game * 12 : (game + 1) * 12])
                offset = shuffle % 12
                ids = ids[offset:] + ids[:offset]
                scores = [10000 + shuffle * 13 + seat * (1 + shuffle % 9) for seat in range(12)]
                rows.append(
                    _row(
                        scores,
                        ids,
                        shuffle=shuffle,
                        game=game,
                        width=width,
                        stopped=(shuffle + game) % 29 == 0,
                    )
                )
        publish_outcome_source(cfg, rows, roster=roster, width=width)
        build_outcome_distributions(cfg, 12)
        del rows
        gc.collect()
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
        started = time.perf_counter()
        try:
            outputs = build_outcome_precision(cfg, 12)
        finally:
            elapsed = time.perf_counter() - started
            stop.set()
            sampler.join()
        assert precision_is_up_to_date(cfg, 12)
        table = pq.read_table(cfg.outcome_precision_path(12))
        assert table.num_rows == args.strategies * 13 * 5
        statuses = Counter(table["status"].to_pylist())
        assert statuses["ok"] > 0
        before = {path: sha256_file(path) for path in outputs}
        started = time.perf_counter()
        build_outcome_precision(cfg, 12)
        resume = time.perf_counter() - started
        assert before == {path: sha256_file(path) for path in outputs}
        root = Path(__file__).resolve().parents[1]
        result = {
            "contract_id": "first_release_task0_v1",
            "strategies": args.strategies,
            "k": 12,
            "batches": args.batches,
            "shuffles_per_batch": width,
            "games": args.batches * width * games,
            "replicates": 2000,
            "analysis_batch_mib": args.batch_mib,
            "declared_incremental_rss_allowance_mib": args.budget_mib,
            "baseline_rss_bytes": baseline,
            "peak_rss_bytes": peak,
            "incremental_rss_bytes": peak - baseline,
            "memory_allowance_passed": peak - baseline <= args.budget_mib * 2**20,
            "elapsed_seconds": elapsed,
            "resume_seconds": resume,
            "precision_rows": table.num_rows,
            "resampled_primary_cells": sum(
                row["summary_id"] != "completed_loss_rate"
                and row["replicates_valid"] + row["replicates_invalid"] == 2000
                for row in table.to_pylist()
            ),
            "status_counts": dict(statuses),
            "final_artifact_bytes": sum(p.stat().st_size for p in outputs),
            "retained_checkpoint_bytes": sum(
                p.stat().st_size
                for p in cfg.outcome_precision_ranges_dir(12).rglob("*")
                if p.is_file()
            ),
            "source_sha256": {
                p: sha256_file(root / p)
                for p in (
                    "src/farkle/analysis/outcome_precision.py",
                    "scripts/benchmark_outcome_precision.py",
                )
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(str(args.output)) as staged:
            Path(staged).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
