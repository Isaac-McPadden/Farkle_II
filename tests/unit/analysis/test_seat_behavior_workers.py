"""Small Windows-spawn and byte-bound qualification for Task 7 source units."""

from __future__ import annotations

import contextlib
import json
import threading
import time

import psutil
import pyarrow.parquet as pq
from tests.unit.analysis.test_outcome_distributions import _row, _table
from tests.unit.utils.test_partitioned_stage import _identity, _resources

from farkle.analysis import seat_behavior as sb
from farkle.analysis.outcome_publication import _Design, _source_units
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.partitioned_stage import run_partitioned_stage


def test_high_k_bounded_source_workers_and_resume(tmp_path, record_property):
    k, games, byte_budget = 12, 48, 2 * 1024**2
    rows = []
    for i in range(games):
        row = _row([50000 - s * 100 + i for s in range(k)], list(range(k)), shuffle=i, width=games)
        row["n_rounds"] = 20
        for seat in range(1, k + 1):
            row[f"P{seat}_n_turns"] = seat + i + 1
            row.update({f"P{seat}_{n}": seat + i for n in sb.BEHAVIORS})
        rows.append(row)
    source = tmp_path / "source.parquet"
    pq.write_table(_table(rows, k), source, row_group_size=7)
    d = _Design(
        11,
        k,
        1,
        games,
        1,
        tuple((i, 1) for i in range(k)),
        str(source),
        (),
        sha256_file(source),
        "a" * 64,
    )
    units = _source_units(d, 6)
    writer = sb._SourceWriter(d, byte_budget)
    parent = psutil.Process()
    stop = threading.Event()
    peaks = {"process_tree_rss_bytes": 0, "children": 0}

    def monitor():
        while not stop.is_set():
            children = parent.children(recursive=True)
            rss = 0
            for process in [parent, *children]:
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    rss += process.memory_info().rss
            peaks["process_tree_rss_bytes"] = max(peaks["process_tree_rss_bytes"], rss)
            peaks["children"] = max(peaks["children"], len(children))
            stop.wait(0.05)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.perf_counter()
    digests = []
    try:
        for workers in (1, 2):
            root = tmp_path / f"workers{workers}"
            result = run_partitioned_stage(
                root=root,
                identity=_identity(),
                unit_source=units.__iter__,
                writer=writer,
                resources=_resources(),
                requested_workers=workers,
                mp_start_method="spawn",
            )
            outputs = [root / "units" / unit.relative_output for unit in units]
            digests.append([sha256_file(p) for p in outputs])
            assert result.completed_units == len(units)
            if workers == 2:
                preserved = {p: p.stat().st_mtime_ns for p in outputs[1:]}
                outputs[0].unlink()
                resumed = run_partitioned_stage(
                    root=root,
                    identity=_identity(),
                    unit_source=units.__iter__,
                    writer=writer,
                    resources=_resources(),
                    requested_workers=workers,
                    mp_start_method="spawn",
                )
                assert resumed.reused_units == len(units) - 1
                assert [sha256_file(p) for p in outputs] == digests[-1]
                assert preserved == {p: p.stat().st_mtime_ns for p in outputs[1:]}
        assert digests[0] == digests[1]
        assert peaks["children"] >= 2
    finally:
        stop.set()
        thread.join()
    scratch_bytes = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    record_property("source_budget_bytes", byte_budget)
    record_property("fixture_games", games)
    record_property("fixture_k", k)
    record_property("source_units", len(units))
    record_property("max_fixed_record_bytes", 6 * (k + 1) * sb.RUN_DTYPE.itemsize)
    record_property("process_tree_peak_rss_bytes", peaks["process_tree_rss_bytes"])
    record_property("scratch_bytes", scratch_bytes)
    record_property("runtime_seconds", round(time.perf_counter() - started, 3))
    (tmp_path / "measurement.json").write_text(
        json.dumps({**peaks, "scratch_bytes": scratch_bytes})
    )
