# src/farkle/__main__.py
"""Command line entry point for the :mod:`farkle` package.

When executed as ``python -m farkle`` this module simply delegates to
:func:`farkle.farkle_cli.main` which implements the full CLI logic.
"""

from __future__ import annotations

from farkle.farkle_cli import main as cli_main


def main() -> None:
    """Invoke :func:`farkle.farkle_cli.main`."""

    cli_main()


if __name__ == "__main__":  # pragma: no cover - direct execution path
    main()
