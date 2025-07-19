import importlib
import sys
from pathlib import Path

import watch_game as wg

import farkle.scoring as scoring

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def test_default_score_patch_handles_discards(monkeypatch):  # noqa: ARG001
    orig = scoring.default_score
    wg._patch_default_score()
    try:
        res = scoring.default_score([1, 1], turn_score_pre=0, return_discards=True)
        assert len(res) == 5
    finally:
        importlib.reload(scoring)
        wg.default_score = orig