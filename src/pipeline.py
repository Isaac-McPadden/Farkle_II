"""Small compatibility layer that re-exports the pipeline's 
main entry point from farkle.pipeline. It exists solely to 
provide a backward-compatible module interface for running 
the analysis pipeline without importing the full package manually.
"""

from farkle.pipeline import main

__all__ = ["main"]
