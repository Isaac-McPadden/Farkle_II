# src/farkle/__main__.py
"""Command line entry point for the :mod:`farkle` package.

When executed as ``python -m farkle`` this module delegates to the lightweight
launcher, which establishes the aggregate OS memory boundary before importing
the full CLI implementation.
"""

from __future__ import annotations

from farkle.cli.launcher import main as cli_main


def main() -> int:
    """Invoke :func:`farkle.cli.launcher.main`."""

    return cli_main()


if __name__ == "__main__":  # pragma: no cover - direct execution path
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
