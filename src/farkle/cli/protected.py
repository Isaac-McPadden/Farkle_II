"""Real CLI process imported only after its OS memory boundary is active."""

import os
from pathlib import Path

if __name__ == "__main__":  # pragma: no cover - protected subprocess entry point
    expected = os.environ.get("FARKLE_LAUNCH_SOURCE")
    if expected is not None and Path(__file__).resolve().parents[2] != Path(expected):
        raise RuntimeError(f"Protected interpreter resolved {__file__} outside {expected}")
    from farkle.cli.main import main

    main()
