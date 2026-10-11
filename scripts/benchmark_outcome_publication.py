"""Bounded Task 1B/2 qualification, with explicit synthetic outcomes and no simulation.

Default: k=12, 5,160 strategies, four shuffles in two aligned batches. Scores
are distinct across games/seats and seats rotate. The producer has a 32 MiB
batch allowance and a declared 128 MiB incremental RSS measurement allowance.
These are fixture measurements, not production capacity or universal RSS claims.
"""

from __future__ import annotations

import argparse
import gc
import json
import struct
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import psutil
import pyarrow.parquet as pq
from tests.helpers.artifact_sidecars import make_authenticated_v3_config
from tests.helpers.outcome_fixture import publish_outcome_source
from tests.unit.analysis.test_outcome_distributions import _row

from farkle.analysis.outcome_publication import build_outcome_distributions, outcome_is_up_to_date
from farkle.config import assign_config_sha
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.writer import atomic_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", type=int, default=5160)
    parser.add_argument("--shuffles", type=int, default=4)
    parser.add_argument("--batch-mib", type=int, default=32)
    parser.add_argument("--budget-mib", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.strategies < 12 or args.strategies % 12 or args.shuffles < 2 or args.shuffles % 2:
        parser.error(
            "strategies must be a positive multiple of 12; shuffles must be positive and even"
        )
    with TemporaryDirectory(prefix="farkle-outcome-benchmark-") as temporary:
        cfg = make_authenticated_v3_config(Path(temporary), player_counts=(12,))
        cfg.sim.row_dir = Path("rows")
        cfg.analysis.n_jobs = cfg.ingest.n_jobs = 1
        cfg.resources.stage_batch_bytes["analysis"] = args.batch_mib * 2**20
        assign_config_sha(cfg)
        roster = tuple(range(args.strategies))
        games = args.strategies // 12
        width = args.shuffles // 2
        rows = []
        for shuffle in range(args.shuffles):
            for game in range(games):
                ids = list(roster[game * 12 : (game + 1) * 12])
                offset = shuffle % 12
                ids = ids[offset:] + ids[:offset]
                ordinal = shuffle * games + game
                scores = [1000000 + ordinal * 12 + seat for seat in range(12)]
                rows.append(
                    _row(
                        scores,
                        ids,
                        shuffle=shuffle,
                        game=game,
                        stopped=ordinal % 17 == 0,
                        width=width,
                    )
                )
        publish_outcome_source(cfg, rows, roster=roster, width=width)
        with cfg.ingested_rows_curated(12).open("rb") as stream:
            stream.seek(-8, 2)
            source_footer_bytes = struct.unpack("<I4s", stream.read(8))[0]
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
            outputs = build_outcome_distributions(cfg, 12)
        finally:
            elapsed = time.perf_counter() - started
            stop.set()
            sampler.join()
        assert outcome_is_up_to_date(cfg, 12)
        support = pq.read_table(cfg.outcome_support_path(12)).to_pandas()
        assert support.loc[support.seat == 0, "attempted"].sum() == args.shuffles * args.strategies
        assert support.loc[support.seat > 0, "attempted"].sum() == args.shuffles * args.strategies
        inventory = pq.read_table(cfg.outcome_batch_inventory_path(12)).to_pandas()
        assert inventory.observed_games.sum() == args.shuffles * games
        started = time.perf_counter()
        build_outcome_distributions(cfg, 12)
        resume = time.perf_counter() - started
        root = Path(__file__).resolve().parents[1]
        modules = [
            "src/farkle/analysis/outcome_publication.py",
            "src/farkle/analysis/outcome_distributions.py",
            "scripts/benchmark_outcome_publication.py",
        ]
        result = {
            "k": 12,
            "strategies": args.strategies,
            "shuffles": args.shuffles,
            "batches": 2,
            "games": args.shuffles * games,
            "exposures": args.shuffles * args.strategies,
            "source_bytes": cfg.ingested_rows_curated(12).stat().st_size,
            "stage_batch_bytes": args.batch_mib * 2**20,
            "declared_incremental_rss_allowance_bytes": args.budget_mib * 2**20,
            "source_footer_bytes": source_footer_bytes,
            "baseline_rss_bytes": baseline,
            "peak_sampled_rss_bytes": peak,
            "incremental_rss_bytes": peak - baseline,
            "within_declared_allowance": peak - baseline <= args.budget_mib * 2**20,
            "sample_interval_seconds": 0.005,
            "publication_seconds": elapsed,
            "authenticated_resume_seconds": resume,
            "checkpoint_bytes": sum(
                p.stat().st_size for p in cfg.outcome_units_dir(12).rglob("*") if p.is_file()
            ),
            "outputs": {
                p.name: {"bytes": p.stat().st_size, "rows": pq.ParquetFile(p).metadata.num_rows}
                for p in outputs[:5]
            },
            "exact_support_checks_passed": True,
            "source_sha256": {p: sha256_file(root / p) for p in modules},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(str(args.output)) as temporary_output:
            Path(temporary_output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
