# pragma: no cover
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
