# pragma: no cover
import os
import importlib.machinery
import importlib.util
import logging
import pickle
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

if "tomllib" not in sys.modules:
    try:
        import tomli as _tomli
    except ModuleNotFoundError:
        def _load_toml(fh):
            raw = fh.read()
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            project: dict[str, str] = {}
            in_project = False
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("[project]"):
                    in_project = True
                    continue
                if in_project:
                    if stripped.startswith("["):
                        break
                    if stripped.startswith("version"):
                        _, _, value = stripped.partition("=")
                        project["version"] = value.strip().strip('"').strip("'")
                        break
            return {"project": project} if project else {}

        stub = types.ModuleType("tomllib")
        stub.load = _load_toml  # type: ignore[attr-defined]
        sys.modules["tomllib"] = stub
    else:
        sys.modules["tomllib"] = _tomli


def _identity_jit(*jit_args, **jit_kwargs):
    if jit_args and callable(jit_args[0]) and len(jit_args) == 1 and not jit_kwargs:
        return jit_args[0]

    def _decorator(func):
        return func

    return _decorator


try:
    import numba  # type: ignore[import-not-found]
except ModuleNotFoundError:
    numba = types.SimpleNamespace(jit=_identity_jit, njit=_identity_jit)  # type: ignore[assignment]
    sys.modules["numba"] = numba  # type: ignore[assignment]
else:
    numba.jit = _identity_jit  # type: ignore[assignment]
    numba.njit = _identity_jit  # type: ignore[assignment]

spec = importlib.util.find_spec("matplotlib")
if spec is not None:
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg", force=True)
else:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("pyplot")
    matplotlib.pyplot = pyplot  # type: ignore[attr-defined]
    matplotlib.use = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    matplotlib.__spec__ = importlib.machinery.ModuleSpec("matplotlib", loader=None)  # type: ignore[attr-defined]
    pyplot.__spec__ = importlib.machinery.ModuleSpec("matplotlib.pyplot", loader=None)  # type: ignore[attr-defined]
    pyplot.close = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)

sklearn_spec = importlib.util.find_spec("sklearn")
if sklearn_spec is None:
    sklearn = types.ModuleType("sklearn")
    ensemble = types.ModuleType("ensemble")
    inspection = types.ModuleType("inspection")

    class _DummyHGB:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):  # noqa: D401 - behavior irrelevant for tests
            return self

        def predict(self, X):  # noqa: D401 - behavior irrelevant for tests
            return [0.0] * len(X)

    ensemble.HistGradientBoostingRegressor = _DummyHGB  # type: ignore[attr-defined]
    sklearn.ensemble = ensemble  # type: ignore[attr-defined]
    sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)  # type: ignore[attr-defined]
    ensemble.__spec__ = importlib.machinery.ModuleSpec("sklearn.ensemble", loader=None)  # type: ignore[attr-defined]

    class _DummyPDP:  # pragma: no cover - simple stub
        def __init__(self) -> None:
            class _Fig:
                def savefig(self, path, format="png") -> None:  # noqa: D401 - stub
                    Path(path).touch()

            self.figure_ = _Fig()

        @classmethod
        def from_estimator(cls, *args, **kwargs):  # noqa: D401 - stub
            return cls()

    def _dummy_permutation_importance(model, X, y, *args, **kwargs):  # noqa: D401 - stub
        n = len(getattr(X, "columns", []))
        return {
            "importances_mean": [0.0] * n,
            "importances_std": [0.0] * n,
        }

    inspection.PartialDependenceDisplay = _DummyPDP  # type: ignore[attr-defined]
    inspection.permutation_importance = _dummy_permutation_importance  # type: ignore[attr-defined]
    sklearn.inspection = inspection  # type: ignore[attr-defined]
    inspection.__spec__ = importlib.machinery.ModuleSpec("sklearn.inspection", loader=None)  # type: ignore[attr-defined]
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.ensemble", ensemble)
    sys.modules.setdefault("sklearn.inspection", inspection)


def pytest_configure():
    """
    During unit-tests we don't need Numba's jit - disable it so coverage can
    see the Python source lines inside decorated functions.
    """
    try:
        import numba  # noqa: F401  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return

    numba.jit = _identity_jit  # type: ignore[assignment]
    numba.njit = _identity_jit  # keep both symbols  # type: ignore[assignment]


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    prev = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(prev)


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
