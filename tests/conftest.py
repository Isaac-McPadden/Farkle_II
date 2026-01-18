# tests/conftest.py
"""Shared pytest fixtures and compatibility shims for the test suite."""

# pragma: no cover
# ruff: noqa: ARG005 ARG003 ARG002 ARG001
import importlib
import importlib.machinery
import importlib.util
import logging
import os
import pickle
import random
import sys
import types
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from freezegun import freeze_time

from farkle.config import AppConfig, IOConfig, SimConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))
TEST_PATH = PROJECT_ROOT / "tests"
if TEST_PATH.exists():
    sys.path.insert(0, str(TEST_PATH))

if "tomllib" not in sys.modules:
    tomllib_spec = importlib.util.find_spec("tomllib")
    tomli_spec = importlib.util.find_spec("tomli")
    if tomllib_spec is not None:
        sys.modules["tomllib"] = importlib.import_module("tomllib")
    elif tomli_spec is not None:
        sys.modules["tomllib"] = importlib.import_module("tomli")
    else:

        def _load_toml(fh):
            """Minimal TOML loader that only extracts a version string.

            Args:
                fh: File-like object pointing at a ``pyproject.toml`` file.

            Returns:
                Mapping with a ``project`` section containing a ``version`` key
                when present, otherwise an empty mapping.
            """

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


def _identity_jit(*jit_args, **jit_kwargs):
    """Return the decorated function unchanged when Numba is unavailable.

    Args:
        *jit_args: Positional arguments that may include the decorated callable.
        **jit_kwargs: Keyword arguments normally passed to :func:`numba.jit`.

    Returns:
        The original callable when used as a decorator or a passthrough
        decorator that yields the input function.
    """

    if jit_args and callable(jit_args[0]) and len(jit_args) == 1 and not jit_kwargs:
        return jit_args[0]

    def _decorator(func):
        """Passthrough decorator preserving the wrapped function.

        Args:
            func: Function being wrapped.

        Returns:
            The unmodified ``func`` reference.
        """

        return func

    return _decorator


numba_spec = importlib.util.find_spec("numba")
if numba_spec is None:
    numba = types.ModuleType("numba")
    numba.jit = _identity_jit  # type: ignore[attr-defined]
    numba.njit = _identity_jit  # type: ignore[attr-defined]
    sys.modules["numba"] = numba
else:
    numba = importlib.import_module("numba")
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
    metrics = types.ModuleType("metrics")

    class _DummyHGB:  # pragma: no cover - simple stub
        """Lightweight Histogram Gradient Boosting stub used in tests."""

        def __init__(self, *args, **kwargs):
            """Accept arbitrary initialization arguments without storing state."""

        def fit(self, *args, **kwargs):  # noqa: D401 - behavior irrelevant for tests
            """Mimic estimator fit by returning self.

            Returns:
                The stub instance so chained calls behave like scikit-learn.
            """

            return self

        def predict(self, X):  # noqa: D401 - behavior irrelevant for tests
            """Produce zero-valued predictions matching the input length.

            Args:
                X: Iterable of samples to score.

            Returns:
                List of float predictions aligned to ``X`` length.
            """

            return [0.0] * len(X)

    ensemble.HistGradientBoostingRegressor = _DummyHGB  # type: ignore[attr-defined]
    sklearn.ensemble = ensemble  # type: ignore[attr-defined]
    sklearn.metrics = metrics  # type: ignore[attr-defined]
    sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)  # type: ignore[attr-defined]
    ensemble.__spec__ = importlib.machinery.ModuleSpec("sklearn.ensemble", loader=None)  # type: ignore[attr-defined]
    metrics.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)  # type: ignore[attr-defined]

    class _DummyPDP:  # pragma: no cover - simple stub
        """Simple partial dependence display stub with a file-writing figure."""

        def __init__(self) -> None:
            class _Fig:
                """Placeholder figure object that writes an empty file."""

                def savefig(self, path, format="png") -> None:  # noqa: D401 - stub
                    """Create an empty image file path to satisfy callers.

                    Args:
                        path: Destination file path.
                        format: Image format extension.

                    Returns:
                        None
                    """

                    Path(path).touch()

            self.figure_ = _Fig()

        @classmethod
        def from_estimator(cls, *args, **kwargs):  # noqa: D401 - stub
            """Construct a stub display irrespective of estimator inputs.

            Returns:
                A new :class:`_DummyPDP` instance.
            """

            return cls()

    def _dummy_permutation_importance(model, X, y, *args, **kwargs):  # noqa: D401 - stub
        """Return deterministic zero-valued permutation importance results.

        Args:
            model: Estimator being inspected.
            X: Feature matrix with column metadata.
            y: Target vector.
            *args: Additional positional arguments ignored in the stub.
            **kwargs: Additional keyword arguments ignored in the stub.

        Returns:
            Dictionary mirroring scikit-learn's importance structure filled with zeros.
        """

        n = len(getattr(X, "columns", []))
        return {
            "importances_mean": [0.0] * n,
            "importances_std": [0.0] * n,
        }

    inspection.PartialDependenceDisplay = _DummyPDP  # type: ignore[attr-defined]
    inspection.permutation_importance = _dummy_permutation_importance  # type: ignore[attr-defined]
    sklearn.inspection = inspection  # type: ignore[attr-defined]
    inspection.__spec__ = importlib.machinery.ModuleSpec("sklearn.inspection", loader=None)  # type: ignore[attr-defined]
    metrics.adjusted_rand_score = lambda *args, **kwargs: 0.0  # type: ignore[attr-defined]
    metrics.normalized_mutual_info_score = lambda *args, **kwargs: 0.0  # type: ignore[attr-defined]
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.ensemble", ensemble)
    sys.modules.setdefault("sklearn.inspection", inspection)
    sys.modules.setdefault("sklearn.metrics", metrics)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI flags used by the test suite.

    Args:
        parser: Pytest option parser object to extend.

    Returns:
        None
    """

    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Regenerate stored golden files for deterministic tests.",
    )


@pytest.fixture(scope="session")
def update_goldens(pytestconfig: pytest.Config) -> bool:
    """Expose the ``--update-goldens`` flag to tests.

    Args:
        pytestconfig: Active pytest configuration instance.

    Returns:
        True when golden files should be regenerated.
    """

    return bool(pytestconfig.getoption("--update-goldens"))


@pytest.fixture(autouse=True)
def _freeze_time() -> Generator[None, None, None]:
    """Pin wall-clock time for deterministic file stamps.

    Returns:
        Generator that freezes time while the test executes.
    """

    with freeze_time("2024-01-01 00:00:00", tick=True):
        yield


@pytest.fixture(autouse=True)
def _seed_random_generators(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Force deterministic randomness for every test function.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to set environment vars.

    Returns:
        Generator that seeds randomness before yielding to the test.
    """

    random.seed(1337)
    _ = np.random.default_rng(1337)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    yield


def pytest_configure():
    """Disable Numba JIT during unit tests to preserve coverage reporting."""

    numba.jit = _identity_jit  # type: ignore[assignment]
    numba.njit = _identity_jit  # keep both symbols  # type: ignore[assignment]


@pytest.fixture
def tmp_results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """Provide an isolated working directory for filesystem interactions.

    Args:
        tmp_path: Unique temporary directory provided by pytest.
        monkeypatch: Fixture used to override environment variables.

    Returns:
        Generator yielding the temporary root path while the test runs.
    """

    prev = os.getcwd()
    analysis_root = tmp_path / "analysis" / "data"
    analysis_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FARKLE_RESULTS_DIR", str(tmp_path))
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(prev)


@pytest.fixture
def capinfo(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Set the log capture level to INFO for assertions.

    Args:
        caplog: Pytest logging capture fixture.

    Returns:
        The configured log capture fixture.
    """
    caplog.set_level(logging.INFO)
    return caplog


@pytest.fixture
def sim_artifacts(tmp_path: Path) -> dict[str, Path]:
    """Create synthetic simulation outputs that mirror the expected layout.

    Args:
        tmp_path: Temporary directory root for writing artifacts.

    Returns:
        Mapping of artifact names to their generated paths.
    """

    cfg = AppConfig(
        io=IOConfig(results_dir=tmp_path, append_seed=False),
        sim=SimConfig(n_players_list=[2], expanded_metrics=True),
    )
    n_players = cfg.sim.n_players_list[0]
    n_dir = cfg.n_dir(n_players)
    n_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = cfg.checkpoint_path(n_players)
    payload = {"win_totals": {"alpha": 1}}
    checkpoint.write_bytes(pickle.dumps(payload))

    metrics_path = cfg.metrics_path(n_players)
    metrics_df = pd.DataFrame(
        {
            "strategy": ["alpha"],
            "wins": [1],
            "total_games_strat": [1],
            "win_rate": [1.0],
            "sum_winning_score": [10.0],
            "sq_sum_winning_score": [100.0],
            "sum_n_rounds": [1.0],
            "sq_sum_n_rounds": [1.0],
            "sum_winner_hit_max_rounds": [0.0],
        }
    )
    metrics_df.to_parquet(metrics_path, index=False)

    return {"checkpoint": checkpoint, "metrics": metrics_path}
