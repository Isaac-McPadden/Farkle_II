"""Small subprocess probe for aggregate OS memory-boundary tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _allocate(mebibytes: int) -> bytearray:
    allocation = bytearray(mebibytes * 1024 * 1024)
    for offset in range(0, len(allocation), 4096):
        allocation[offset] = 1
    return allocation


def _spawn(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *arguments],
        check=False,
        text=True,
    )


def _tree(args: argparse.Namespace) -> int:
    allocation = _allocate(args.allocate_mb)
    if args.depth > 0:
        child = _spawn(
            [
                "tree",
                "--allocate-mb",
                str(args.allocate_mb),
                "--depth",
                str(args.depth - 1),
                "--hold-seconds",
                str(args.hold_seconds),
            ]
        )
        if child.returncode != 0:
            return child.returncode
    else:
        time.sleep(args.hold_seconds)
    return allocation[0] - 1


def _two_roots(args: argparse.Namespace) -> int:
    parent_allocation = _allocate(args.parent_mb)
    children = [
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "tree",
                "--allocate-mb",
                str(args.root_mb),
                "--depth",
                "0",
                "--hold-seconds",
                str(args.hold_seconds),
            ]
        )
        for _ in range(2)
    ]
    returncodes = [child.wait() for child in children]
    return max([parent_allocation[0] - 1, *returncodes])


def _publication(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completed_temp = output / "unit.complete.tmp"
    completed_temp.write_text("authenticated-complete-unit", encoding="utf-8")
    os.replace(completed_temp, output / "unit.complete")
    partial = output / "unit.partial.tmp"
    with partial.open("wb") as handle:
        handle.write(b"not-authenticated")
        handle.flush()
        os.fsync(handle.fileno())
    child = _spawn(
        [
            "tree",
            "--allocate-mb",
            str(args.allocate_mb),
            "--depth",
            "1",
            "--hold-seconds",
            "2",
        ]
    )
    if child.returncode != 0:
        return child.returncode
    (output / "pipeline.complete").write_text("complete", encoding="utf-8")
    return 0


def _identity(args: argparse.Namespace) -> int:
    payload = {
        "seed": args.seed,
        "values": [(index * 17 + args.seed) % 101 for index in range(1000)],
    }
    print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
    return 0


def _status(_args: argparse.Namespace) -> int:
    print(os.environ["FARKLE_OS_MEMORY_BOUNDARY"])
    return 0


def _partition_worker_status(_value: int) -> str:
    """Pickle-safe worker used to prove inherited aggregate containment."""

    return os.environ["FARKLE_OS_MEMORY_BOUNDARY"]


def _partition_workers(args: argparse.Namespace) -> int:
    from farkle.utils.parallel import process_map

    print(
        json.dumps(
            list(process_map(_partition_worker_status, range(args.workers), n_jobs=args.workers))
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    tree = subparsers.add_parser("tree")
    tree.add_argument("--allocate-mb", type=int, required=True)
    tree.add_argument("--depth", type=int, required=True)
    tree.add_argument("--hold-seconds", type=float, default=1.0)
    roots = subparsers.add_parser("two-roots")
    roots.add_argument("--parent-mb", type=int, required=True)
    roots.add_argument("--root-mb", type=int, required=True)
    roots.add_argument("--hold-seconds", type=float, default=1.0)
    publication = subparsers.add_parser("publication")
    publication.add_argument("--output-dir", required=True)
    publication.add_argument("--allocate-mb", type=int, required=True)
    identity = subparsers.add_parser("identity")
    identity.add_argument("--seed", type=int, default=7)
    subparsers.add_parser("status")
    workers = subparsers.add_parser("partition-workers")
    workers.add_argument("--workers", type=int, default=2)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "tree":
        return _tree(args)
    if args.command == "two-roots":
        return _two_roots(args)
    if args.command == "publication":
        return _publication(args)
    if args.command == "status":
        return _status(args)
    if args.command == "partition-workers":
        return _partition_workers(args)
    return _identity(args)


if __name__ == "__main__":
    raise SystemExit(main())
