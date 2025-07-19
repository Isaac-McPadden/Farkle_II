import importlib.util
import sys
import types
from pathlib import Path

CONF_PATH = Path(__file__).resolve().parents[1] / "conftest.py"
spec = importlib.util.spec_from_file_location("test_conf", CONF_PATH)
assert spec is not None
conf = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(conf)


def test_pytest_configure_patches_numba(monkeypatch):
    dummy = types.SimpleNamespace()
    sentinel_jit = object()
    sentinel_njit = object()
    dummy.jit = sentinel_jit
    dummy.njit = sentinel_njit
    monkeypatch.setitem(sys.modules, "numba", dummy)

    conf.pytest_configure()

    assert dummy.jit is not sentinel_jit
    assert dummy.njit is dummy.jit

    def func():
        return 42

    assert dummy.jit()(func) is func # type: ignore
    assert dummy.njit()(func) is func # type: ignore