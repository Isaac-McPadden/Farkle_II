"""Run the read-only structural release audits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from farkle.analysis.release_audit import run_release_audits

ROOT = Path(__file__).resolve().parents[1]
RUNNABLE_CONFIGS = (
    ROOT / "configs" / "default_config.yaml",
    ROOT / "configs" / "fast_config.yaml",
    ROOT / "configs" / "farkle_mega_config.yaml",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        action="append",
        type=Path,
        default=[],
        help="Canonical analysis root to audit; may be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print release-audit failures and return a CI-compatible status."""

    args = _parser().parse_args(argv)
    failures = run_release_audits(
        ROOT,
        config_paths=RUNNABLE_CONFIGS,
        artifact_roots=args.artifact_root,
    )
    if not failures:
        return 0
    print("Structure release audit failed:", file=sys.stderr)
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
