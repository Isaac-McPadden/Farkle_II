from __future__ import annotations

import random
from dataclasses import dataclass

"""strategies.py
================
Strategy abstractions used by the Farkle simulation engine.  A *strategy*
object decides **whether to continue rolling** based on the current turn
context, independent of dice‑scoring rules.

Classes
-------
`ThresholdStrategy` – simple heuristic combining score & dice limits plus
an optional Smart‑5 flag.

Functions
---------
`random_threshold_strategy` – convenience generator for Monte‑Carlo
sweeps.
"""



__all__: list[str] = [
    "ThresholdStrategy",
    "random_threshold_strategy",
]

# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------

@dataclass
class ThresholdStrategy:
    """Threshold‑based decision rule.

    Parameters
    ----------
    score_threshold
        Keep rolling until turn score reaches this number (subject to
        *consider_score*).
    dice_threshold
        Bank when dice‑left drop to this value or below (subject to
        *consider_dice*).
    smart
        Enables Smart‑5 heuristic during scoring.
    consider_score, consider_dice
        Toggle the two conditions to reproduce *score‑only*, *dice‑only*,
        or *balanced* play styles.
    """

    score_threshold: int = 300
    dice_threshold: int = 2
    smart: bool = False
    consider_score: bool = True
    consider_dice: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decide(
        self,
        *,
        turn_score: int,
        dice_left: int,
        has_scored: bool,
        score_needed: int,  # not used but allows richer future strats  # noqa: ARG002
    ) -> bool:  # noqa: D401 – imperative name
        """Decision rule implementation."""
        if not has_scored and turn_score < 500:
            return True  # opening rolls until first 500 pts
        want_score = self.consider_score and turn_score < self.score_threshold
        want_dice = self.consider_dice and dice_left > self.dice_threshold
        if self.consider_score and self.consider_dice:
            return want_score and want_dice
        if self.consider_score:
            return want_score
        if self.consider_dice:
            return want_dice
        return False

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __str__(self) -> str:  # noqa: D401 – magics method
        cs = "S" if self.consider_score else "-"
        cd = "D" if self.consider_dice else "-"
        sm = "*" if self.smart else " "
        return f"T({self.score_threshold},{self.dice_threshold})[{cs}{cd}]{sm}"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def random_threshold_strategy(rng: random.Random | None = None) -> ThresholdStrategy:
    """Return a randomised *ThresholdStrategy* instance."""
    # if caller didn’t pass a PRNG, make a fresh one
    rng_inst: random.Random = rng if rng is not None else random.Random()
    return ThresholdStrategy(
        score_threshold=rng_inst.randrange(50, 1000, 50),
        dice_threshold=rng_inst.randint(0, 4),
        smart=rng_inst.choice([True, False]),
        consider_score=rng_inst.choice([True, False]),
        consider_dice=rng_inst.choice([True, False]),
    )
