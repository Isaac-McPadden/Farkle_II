# pragma: no cover
# pragma: no cover
import importlib.util
import logging
import pickle
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.find_spec("matplotlib")
if spec is not None:
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg", force=True)


def pytest_configure():
    """
    During unit-tests we don't need Numba's jit - disable it so coverage can
    see the Python source lines inside decorated functions.
    """
    try:
        import numba
    except ModuleNotFoundError:
        return

    numba.jit = lambda *a, **k: (lambda f: f)  # type: ignore  # noqa: ARG005
    numba.njit = numba.jit  # keep both symbols


@pytest.fixture
def capinfo(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO)
    return caplog


@pytest.fixture
def tmp_artifacts_with_legacy(tmp_path: Path) -> dict[str, Path]:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = artifacts_dir / "checkpoint.pkl"
    payload = {"win_totals": {"alpha": 1}}
    checkpoint.write_bytes(pickle.dumps(payload))

    metrics_path = artifacts_dir / "metrics.parquet"
    metrics_df = pd.DataFrame(
        {
            "metric": ["wins"],
            "strategy": ["alpha"],
            "sum": [1.0],
            "square_sum": [1.0],
        }
    )
    try:
        metrics_df.to_parquet(metrics_path, index=False)
    except Exception:
        metrics_path.write_text(metrics_df.to_csv(index=False))

    return {"checkpoint": checkpoint, "metrics": metrics_path}
