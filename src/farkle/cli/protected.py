"""Real CLI process imported only after its OS memory boundary is active."""

from farkle.cli.main import main

if __name__ == "__main__":  # pragma: no cover - protected subprocess entry point
    main()
