"""Bounded synthetic first-release cost measurements; never production simulation.

Measure low/high-k distribution and behavior work separately from authenticated
source preparation. The small fixture has distinct score bins, rotating seats,
explicit terminal counters and safety stops. Its timings are coefficients with
disclosed applicability limits, not observed production throughput.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import psutil
import pyarrow.parquet as pq
from tests.helpers.artifact_sidecars import make_authenticated_v3_config
from tests.helpers.outcome_fixture import publish_outcome_source
from tests.unit.analysis.test_outcome_distributions import _row

from farkle.analysis import outcome_publication as publication
from farkle.analysis import seat_behavior
from farkle.analysis.all_player_metrics import build_all_player_batch_metrics
from farkle.config import assign_config_sha
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.writer import atomic_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, choices=(2, 12), required=True)
    parser.add_argument("--strategies", type=int, default=5160)
    parser.add_argument("--shuffles", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.strategies < args.k or args.strategies % args.k:
        parser.error("strategies must be a positive multiple of k")
    if args.shuffles < 2 or args.shuffles % 2:
        parser.error("shuffles must be positive and even")
    if args.shuffles * args.strategies // args.k > 20000:
        parser.error("bounded source preparation permits at most 20,000 fixture games")
    with TemporaryDirectory(prefix="farkle-task10-cost-") as temporary:
        cfg = make_authenticated_v3_config(Path(temporary), player_counts=(args.k,))
        cfg.sim.row_dir = Path("rows")
        cfg.analysis.n_jobs = cfg.ingest.n_jobs = 1
        cfg.resources.stage_batch_bytes["analysis"] = 32 * 2**20
        assign_config_sha(cfg)
        roster = tuple(range(args.strategies))
        games = args.strategies // args.k
        width = args.shuffles // 2
        rows = []
        for shuffle in range(args.shuffles):
            for game in range(games):
                ids = list(roster[game * args.k : (game + 1) * args.k])
                offset = shuffle % args.k
                ids = ids[offset:] + ids[:offset]
                ordinal = shuffle * games + game
                row = _row(
                    [1000000 + ordinal * args.k + seat for seat in range(args.k)],
                    ids,
                    shuffle=shuffle,
                    game=game,
                    stopped=ordinal % 17 == 0,
                    width=width,
                )
                for seat in range(1, args.k + 1):
                    row[f"P{seat}_n_turns"] = 1 + (shuffle + seat) % 3
                    row[f"P{seat}_rolls"] = 2
                    row[f"P{seat}_highest_turn"] = 100
                rows.append(row)
        prepared = time.perf_counter()
        publish_outcome_source(cfg, rows, roster=roster, width=width)
        preparation_seconds = time.perf_counter() - prepared
        del rows
        timings: dict[str, float] = defaultdict(float)
        behavior_timings: dict[str, float] = defaultdict(float)
        original_run = publication._Stages.run

        def timed_run(self, name, *arguments, **kwargs):
            before = time.perf_counter()
            result = original_run(self, name, *arguments, **kwargs)
            timings[name] += time.perf_counter() - before
            return result

        def timed_function(name, function, destination=timings):
            def wrapped(*arguments, **kwargs):
                before = time.perf_counter()
                result = function(*arguments, **kwargs)
                destination[name] += time.perf_counter() - before
                return result

            return wrapped

        baseline = psutil.Process().memory_info().rss
        peak = baseline
        stop = threading.Event()

        def sample() -> None:
            nonlocal peak
            while not stop.wait(0.01):
                peak = max(peak, psutil.Process().memory_info().rss)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        measurements: dict[str, Any] = {}

        def measure(name, producer, validator, unit_dir):
            before = time.perf_counter()
            outputs = producer(cfg, args.k)
            elapsed = time.perf_counter() - before
            assert validator(cfg, args.k)
            identities = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in outputs}
            before = time.perf_counter()
            producer(cfg, args.k)
            resume_seconds = time.perf_counter() - before
            assert identities == {p: (sha256_file(p), p.stat().st_mtime_ns) for p in outputs}
            measurements[name] = {
                "seconds": elapsed,
                "no_force_seconds": resume_seconds,
                "outputs": {
                    p.name: {
                        "bytes": p.stat().st_size,
                        "rows": (
                            pq.ParquetFile(p).metadata.num_rows if p.suffix == ".parquet" else None
                        ),
                    }
                    for p in outputs
                },
                "retained_unit_bytes": sum(
                    p.stat().st_size for p in unit_dir.rglob("*") if p.is_file()
                ),
            }

        try:
            with (
                patch.object(publication._Stages, "run", timed_run),
                patch.object(
                    publication,
                    "_publish_batches",
                    timed_function("publish_batches", publication._publish_batches),
                ),
                patch.object(
                    publication,
                    "_publish_summaries",
                    timed_function("publish_summaries", publication._publish_summaries),
                ),
            ):
                measure(
                    "outcomes",
                    publication.build_outcome_distributions,
                    publication.outcome_is_up_to_date,
                    cfg.outcome_units_dir(args.k),
                )
            before = time.perf_counter()
            build_all_player_batch_metrics(cfg, args.k)
            measurements["existing_all_player_preparation_seconds"] = time.perf_counter() - before
            with (
                patch.object(
                    seat_behavior._SourceWriter,
                    "__call__",
                    timed_function(
                        "source", seat_behavior._SourceWriter.__call__, behavior_timings
                    ),
                ),
                patch.object(
                    seat_behavior._MergeWriter,
                    "__call__",
                    timed_function("merge", seat_behavior._MergeWriter.__call__, behavior_timings),
                ),
                patch.object(
                    seat_behavior._BatchWriter,
                    "__call__",
                    timed_function(
                        "certify_batches", seat_behavior._BatchWriter.__call__, behavior_timings
                    ),
                ),
                patch.object(
                    seat_behavior._SummaryWriter,
                    "__call__",
                    timed_function(
                        "summaries", seat_behavior._SummaryWriter.__call__, behavior_timings
                    ),
                ),
                patch.object(
                    seat_behavior,
                    "_publish_rows",
                    timed_function("publish_rows", seat_behavior._publish_rows, behavior_timings),
                ),
                patch.object(
                    seat_behavior,
                    "_reconcile_all_player",
                    timed_function(
                        "reconcile", seat_behavior._reconcile_all_player, behavior_timings
                    ),
                ),
            ):
                measure(
                    "behavior",
                    seat_behavior.build_seat_behavior,
                    seat_behavior.behavior_is_up_to_date,
                    cfg.seat_behavior_units_dir(args.k),
                )
        finally:
            stop.set()
            sampler.join()
        result = {
            "schema_version": 1,
            "synthetic": True,
            "k": args.k,
            "strategies": args.strategies,
            "batches": 2,
            "shuffles": args.shuffles,
            "games": args.shuffles * games,
            "exposures": args.shuffles * args.strategies,
            "source_preparation_seconds": preparation_seconds,
            "outcome_phase_seconds": dict(timings),
            "behavior_phase_seconds": dict(behavior_timings),
            "measurements": measurements,
            "parent_baseline_rss_bytes": baseline,
            "parent_peak_rss_bytes": peak,
            "analysis_batch_bytes": 32 * 2**20,
            "code_sha256": {
                str(p.relative_to(Path(__file__).resolve().parents[1])): sha256_file(p)
                for p in (
                    Path(__file__).resolve(),
                    Path(publication.__file__),
                    Path(seat_behavior.__file__),
                )
            },
            "limits": [
                "Synthetic broad bins, two batches; production cardinality and I/O are unknown.",
                "Serial coefficients; no unmeasured worker speedup assumed.",
                "Existing all-player work and source preparation are excluded from added cost.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(str(args.output)) as staged:
            Path(staged).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
