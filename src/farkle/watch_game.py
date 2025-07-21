#!/usr/bin/env python3
# pragma: no cover
"""
watch_game.py - run a *single* Farkle game with very chatty logging.

It
 • prints every dice throw,
 • prints every Smart-discard scoring call,
 • prints every keep-rolling / bank decision, and
 • finishes with a tiny summary.

No game-logic is duplicated - we only *wrap* the real engine.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict
from types import MethodType
from typing import Sequence

import numpy as np

from farkle.engine import FarkleGame, FarklePlayer  # :contentReference[oaicite:0]{index=0}
from farkle.scoring import default_score  # :contentReference[oaicite:1]{index=1}
from farkle.strategies import (  # :contentReference[oaicite:2]{index=2}
    ThresholdStrategy,
    random_threshold_strategy,
)

# ── 1.  Plain-text logger ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("watch")


# ── 2.  Tiny helpers ---------------------------------------------------------
# ── helper: dump a ThresholdStrategy in YAML-ish format ─────────────────────
def strategy_yaml(s: ThresholdStrategy) -> str:
    """
    Return something like:

        score_threshold : 600
        dice_threshold  : 3
        smart_five      : true
        smart_one       : false
        consider_score  : true
        consider_dice   : true
        require_both    : false
        auto_hot_dice   : false
        run_up_score    : false
        prefer_score    : true
    """
    
    # dataclass → plain dict (keeps declared order)
    if not isinstance(s, ThresholdStrategy):
        raise TypeError("strategy_yaml expects a ThresholdStrategy")
    d = asdict(s)
    
    # YAML-friendly booleans (lowercase)
    def _fmt(v): 
        return str(v).lower() if isinstance(v, bool) else v
    
    lines = [f"{k:<15}: {_fmt(v)}" for k, v in d.items()]
    return "\n".join(lines)


def _trace_decide(s: ThresholdStrategy, label: str) -> None:
    """Monkey-patch *one* strategy instance so we can watch its calls."""
    original = s.decide

    def traced_decide(self: ThresholdStrategy, **kw):  # same signature  # noqa: ARG001
        keep = original(**kw)
        log.info(
            f"{label} decide(): turn={kw['turn_score']:<4}  "
            f"dice_left={kw['dice_left']}  →  "
            f"{'ROLL' if keep else 'BANK'}"
        )
        return keep

    # bind as *method* so `self` is passed correctly
    s.decide = MethodType(traced_decide, s)  # type: ignore[attr-defined]


def _patch_default_score() -> None:
    """Wrap scoring.default_score so every call is printed once."""
    global default_score  # module alias
    original = default_score

    def traced_default_score(*args, **kw):
        res = original(*args, **kw)  # type: ignore[arg-type]
        pts, used, reroll = res[:3]
        roll = args[0]
        log.info(f"score({roll}) -> pts={pts:<4} used={used} reroll={reroll}")
        return res

    # monkey-patch in the *farkle.scoring* module too
    import farkle.scoring as _scoring_mod
    
    _scoring_mod.default_score = traced_default_score         # type: ignore[assignment]
    default_score = traced_default_score                      # local alias


class TracePlayer(FarklePlayer):
    """Subclass that only adds a noisy _roll()."""

    def _roll(self, n: int) -> Sequence[int]:  # :contentReference[oaicite:3]{index=3}
        faces = super()._roll(n)
        log.info(f"{self.name} rolls {faces}")
        return faces


# ── 3.  High-level entry-point ----------------------------------------------
def watch_game(seed: int | None = None) -> None:
    """
    Play one game between two *random* ThresholdStrategy players
    with full trace output.
    """
    rng = np.random.default_rng(seed)

    # --- make two random strategies -------------------------------------
    s1 = random_threshold_strategy(random.Random(int(rng.integers(2**32))))
    s2 = random_threshold_strategy(random.Random(int(rng.integers(2**32))))
    log.info("P1 strategy\n%s\n", strategy_yaml(s1))
    log.info("P2 strategy\n%s\n", strategy_yaml(s2))

    # monkey-patch the *strategy* layer so we can see decide()
    _trace_decide(s1, "P1")
    _trace_decide(s2, "P2")

    # monkey-patch the *scoring* layer so we can see Smart-discard results
    _patch_default_score()

    # --- wrap players so we can see every dice throw --------------------
    p1 = TracePlayer("P1", s1, rng=np.random.default_rng(rng.integers(2**32)))
    p2 = TracePlayer("P2", s2, rng=np.random.default_rng(rng.integers(2**32)))

    game = FarkleGame([p1, p2], target_score=10_000)
    gm = game.play()

    log.info("\n===== final result =====")
    log.info(
        f"Winner: {gm.winner}  "
        f"score={gm.winning_score}  "
        f"rounds={gm.n_rounds}"
    )


if __name__ == "__main__":
    # run:  python -m farkle.watch_game  (optionally pass a seed)
    watch_game()
