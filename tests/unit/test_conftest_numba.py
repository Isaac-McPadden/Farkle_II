import importlib.util
from pathlib import Path

CONF_PATH = Path(__file__).resolve().parents[1] / "conftest.py"
spec = importlib.util.spec_from_file_location("test_conf", CONF_PATH)
assert spec is not None
conf = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(conf)


def test_pytest_configure_patches_numba(monkeypatch):
    sentinel_jit = object()
    sentinel_njit = object()

    # pytest_configure mutates the module object captured in tests/conftest.py,
    # regardless of whether that object came from a real numba install or a
    # fallback shim. Patch that object directly to avoid coupling to import order.
    monkeypatch.setattr(conf.numba, "jit", sentinel_jit)
    monkeypatch.setattr(conf.numba, "njit", sentinel_njit)

    conf.pytest_configure()

    assert conf.numba.jit is conf._identity_jit
    assert conf.numba.njit is conf._identity_jit

    def func():
        return 42

    assert conf.numba.jit()(func) is func  # type: ignore[misc]
    assert conf.numba.njit()(func) is func  # type: ignore[misc]
