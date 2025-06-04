from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

"""strategies.py
================
Strategy abstractions used by the Farkle simulation engine.  A *strategy*
object decides **whether to continue rolling** based on the current turn
context, independent of dice-scoring rules.

Classes
-------
`ThresholdStrategy` - simple heuristic combining score & dice limits plus
an optional Smart-5 flag.

Functions
---------
`random_threshold_strategy` - convenience generator for Monte-Carlo
sweeps.
"""



__all__: list[str] = [
    "ThresholdStrategy",
    "random_threshold_strategy",
]

DiceRoll = List[int]
"""A list of integers 1-6 representing a single dice roll."""

# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------

@dataclass
class ThresholdStrategy:
    """Threshold-based decision rule.

    Parameters
    ----------
    score_threshold
        Keep rolling until turn score reaches this number (subject to
        *consider_score*).
    dice_threshold
        Bank when dice-left drop to this value or below (subject to
        *consider_dice*).
    smart_five, smart_one
        Enables/Disables Smart-5 and Smart-1 heuristics during scoring.
    consider_score, consider_dice
        Toggle the two conditions to reproduce *score-only*, *dice-only*,
        or *balanced* play styles.
    """

    score_threshold: int = 300
    dice_threshold: int = 2
    smart_five: bool = False   # “throw back lone 5’s first”
    smart_one:  bool = False
    consider_score: bool = True
    consider_dice: bool = True
    require_both: bool = False
    auto_hot_dice: bool = False  
    
    def __post_init__(self):
        # 1) smart_one may never be True if smart_five is False
        if self.smart_one and not self.smart_five:
            raise ValueError(
                "ThresholdStrategy: smart_one=True requires smart_five=True"
            )

        # 2) require_both may only be True if both consider_score and consider_dice are True
        if self.require_both and not (self.consider_score and self.consider_dice):
            raise ValueError(
                "ThresholdStrategy: require_both=True requires both "
                "consider_score=True and consider_dice=True"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decide(
        self,
        *,
        turn_score: int,
        dice_left: int,
        has_scored: bool,
        score_needed: int,          # noqa: ARG002 (vestigial/reserved for richer strats)
        final_round: bool = False,  # (new end-game hooks – harmless defaults)
        score_to_beat: int = 0,
        running_total: int = 0,
    ) -> bool:  # noqa: D401 – imperative name
        """Return **True** to keep rolling, **False** to bank."""

        # --------------------------------- fast exits ---------------------------------
        if not has_scored and turn_score < 500:
            return True                     # must cross the 500-pt entry gate

        # final-round catch-up rule
        if final_round:
            return running_total <= score_to_beat

        # -----------------------------------------------------------------------------  
        want_score = self.consider_score and turn_score < self.score_threshold
        want_dice  = self.consider_dice and dice_left  > self.dice_threshold

        # both score & dice active ----------------------------------------------------
        if self.consider_score and self.consider_dice:
            if self.require_both:
                # stop only when **both** limits are satisfied
                return want_score or want_dice
            else:
                # stop as soon as **either** limit is satisfied
                return want_score and want_dice

        # single-axis strategies ------------------------------------------------------
        if self.consider_score:
            return want_score
        if self.consider_dice:
            return want_dice

        # nothing left to consider → always bank
        return False

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __str__(self) -> str:  # noqa: D401 - magics method
        cs = "S" if self.consider_score else "-"
        cd = "D" if self.consider_dice else "-"
        sf = "*F" if self.smart_five else "-"
        so = "*O" if self.smart_one else "-"
        rb = "AND" if self.require_both else "OR"
        return f"T({self.score_threshold},{self.dice_threshold})[{cs}{cd}][{sf}{so}][{rb}]"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def random_threshold_strategy(rng: random.Random | None = None) -> ThresholdStrategy:
    """Return a random ThresholdStrategy that always satisfies the two constraints."""
    rng_inst = rng if rng is not None else random.Random()

    # pick smart_five first; if it’s False, force smart_one=False
    sf = rng_inst.choice([True, False])
    so = rng_inst.choice([True, False]) if sf else False

    # pick consider_score/dice; if either is False, force require_both=False
    cs = rng_inst.choice([True, False])
    cd = rng_inst.choice([True, False])
    rb = rng_inst.choice([True, False]) if (cs and cd) else False

    return ThresholdStrategy(
        score_threshold=rng_inst.randrange(50, 1000, 50),
        dice_threshold=rng_inst.randint(0, 4),
        smart_five=sf,
        smart_one=so,
        consider_score=cs,
        consider_dice=cd,
        require_both=rb,
    )
