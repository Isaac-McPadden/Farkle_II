"""Run only the bounded Task 5C validation workload, never production games."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farkle.analysis.rng_preflight import run  # noqa: E402
from farkle.config import assign_config_sha, load_app_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_app_config(ROOT / "configs/farkle_mega_config.yaml")
    cfg.io.results_dir_prefix = ROOT / "data/task5c-validation"
    assign_config_sha(cfg)
    path, result = run(cfg, force=args.force)
    print(json.dumps({"publication": str(path), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
