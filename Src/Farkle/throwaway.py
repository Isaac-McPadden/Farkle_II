from __future__ import annotations
"""Farkle Simulation Module (v1.1 – strategy grid + smart‑reroll)
================================================================
An evolution of *Farkle Game Code Export.py* focused on **flexibility**
**scalability** and **rich metrics**.  New in v1.1:

* *Smart* single‑5 reroll logic (optional, per‑strategy flag).
* Helper to **generate the full grid** of strategy combinations:
  ``generate_strategy_grid``.
* Default grid matches the original *800‑strategy* experiment
  (17 score × 5 dice × (consider_score×consider_dice fourfold) × 2 smart
  = *680* ≈ *800* after including baseline variants).  Users can override
  the ranges to build meta‑tournaments of arbitrary size.
* ``experiment_size`` convenience to report the number of strategies.

The public API:

* ``ThresholdStrategy`` – parametric strategy including *smart* flag.
* ``generate_strategy_grid`` – returns (``List[ThresholdStrategy]``,
  ``pd.DataFrame``) covering all combos.
* ``simulate_one_game`` / ``simulate_many_games`` – single/batch runs.
* ``aggregate_metrics`` – quick experiment‑level summariser.

Example
-------
>>> from farkle_simulation_module import generate_strategy_grid, simulate_many_games
>>> strats, meta = generate_strategy_grid()  # default 800‑ish grid
>>> print(f"Strategies: {len(strats)}\n", meta.head())
>>> df = simulate_many_games(n_games=5_000, strategies=strats[:5], n_jobs=8)
>>> print(aggregate_metrics(df))
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, List, Sequence, Tuple, Protocol, Optional, Any
import random
import multiprocessing as mp

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    raise ImportError("pandas is required for the simulation module")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
DiceRoll = List[int]
ScoreFunc = Callable[[DiceRoll], Tuple[int, int, int]]  # score, used_dice, reroll_dice

# ---------------------------------------------------------------------------
# Scoring logic (classic Farkle)
# ---------------------------------------------------------------------------

def default_score(dice_roll: DiceRoll) -> Tuple[int, int, int]:
    """Return (*score*, *used_dice*, *reroll_dice*) for *dice_roll*.

    Implements standard Farkle scoring rules.
    """
    counts = Counter(dice_roll)
    score = 0
    used = 0

    # Special combos -------------------------------------------------------
    if len(counts) == 6:
        return 1500, 6, 0  # straight 1‑6
    if len(counts) == 3 and all(v == 2 for v in counts.values()):
        return 1500, 6, 0  # three pairs
    if len(counts) == 2 and set(counts.values()) == {3, 3}:
        return 2500, 6, 0  # two triplets
    if len(counts) == 2 and 4 in counts.values() and 2 in counts.values():
        return 1500, 6, 0  # 4‑of‑a‑kind + pair

    # n‑of‑a‑kind & singles ----------------------------------------------
    for num, cnt in counts.items():
        if cnt >= 3:
            if cnt == 3:
                score += 300 if num == 1 else num * 100
            elif cnt == 4:
                score += 1000
            elif cnt == 5:
                score += 2000
            elif cnt == 6:
                score += 3000
            used += cnt
        elif num == 1:
            score += 100 * cnt
            used += cnt
        elif num == 5:
            score += 50 * cnt
            used += cnt

    reroll = len(dice_roll) - used
    return score, used, reroll


# ---------------------------------------------------------------------------
# Strategy protocol & implementation
# ---------------------------------------------------------------------------

class Strategy(Protocol):
    def decide(self, *, turn_score: int, dice_left: int, has_scored: bool,
               score_needed: int) -> bool:
        """Return **True** to *continue* rolling, **False** to bank points."""

    def __str__(self) -> str: ...


@dataclass
class ThresholdStrategy:
    """Simple heuristic strategy.

    Parameters
    ----------
    score_threshold
        Continue rolling until *turn_score* ≥ this value (if
        ``consider_score`` is True).
    dice_threshold
        Bank if dice left ≤ this value (if ``consider_dice`` is True).
    smart
        If *True*, single 5s (worth only 50 pts) are **ignored** – the die
        is rerolled instead of banked, reducing low‑value stalls.
    consider_score, consider_dice
        Toggle each heuristic independently to replicate the full
        2×2 design used in the original 800‑strategy sweep.
    """

    score_threshold: int = 300
    dice_threshold: int = 2
    smart: bool = False
    consider_score: bool = True
    consider_dice: bool = True

    # ------------------------------------------------------------------
    # Decision rule
    # ------------------------------------------------------------------
    def decide(self, *, turn_score: int, dice_left: int, has_scored: bool,
               score_needed: int) -> bool:
        # Cannot bank before first 500‑pt turn
        if not has_scored and turn_score < 500:
            return True
        want_score = self.consider_score and turn_score < self.score_threshold
        want_dice = self.consider_dice and dice_left > self.dice_threshold
        if self.consider_score and self.consider_dice:
            return want_score and want_dice
        if self.consider_score:
            return want_score
        if self.consider_dice:
            return want_dice
        return False

    def __str__(self) -> str:
        cs = "S" if self.consider_score else "‑"
        cd = "D" if self.consider_dice else "‑"
        sm = "*" if self.smart else " "
        return f"T({self.score_threshold},{self.dice_threshold})[{cs}{cd}]{sm}"


# ---------------------------------------------------------------------------
# Player & game engine
# ---------------------------------------------------------------------------

@dataclass
class FarklePlayer:
    name: str
    strategy: ThresholdStrategy
    score: int = 0
    has_scored: bool = False
    rng: random.Random = field(default_factory=random.Random, repr=False)
    n_farkles: int = 0
    n_rolls: int = 0
    highest_turn: int = 0

    def _roll(self, n: int) -> DiceRoll:
        self.n_rolls += 1
        return [self.rng.randint(1, 6) for _ in range(n)]

    def take_turn(self, score_fn: ScoreFunc, target_score: int) -> None:
        dice = 6
        turn_score = 0
        while dice > 0:
            roll = self._roll(dice)
            roll_score, used, reroll = score_fn(roll)

            # Smart rule – ignore lone 5s (50 pts)
            if self.strategy.smart and roll_score == 50 and used == 1:
                # Treat as *no score*; reroll all dice
                roll_score = 0
                used = 0
                reroll = len(roll)

            if roll_score == 0:
                self.n_farkles += 1
                turn_score = 0
                break

            turn_score += roll_score
            dice = reroll or 6 if used == len(roll) and reroll == 0 else reroll

            if not self.strategy.decide(
                turn_score=turn_score,
                dice_left=dice,
                has_scored=self.has_scored,
                score_needed=max(0, target_score - (self.score + turn_score)),
            ):
                break

        if not self.has_scored and turn_score >= 500:
            self.has_scored = True
        if self.has_scored:
            self.score += turn_score
            self.highest_turn = max(self.highest_turn, turn_score)


@dataclass
class GameMetrics:
    winner: str
    winning_score: int
    n_rounds: int
    per_player: Dict[str, Dict[str, Any]]


class FarkleGame:
    def __init__(self, players: Sequence[FarklePlayer], *, target_score: int = 10_000,
                 score_fn: ScoreFunc = default_score) -> None:
        self.players = list(players)
        self.target_score = target_score
        self.score_fn = score_fn

    def play(self, max_rounds: int = 100) -> GameMetrics:
        final_round_flag = False
        trigger_player: Optional[FarklePlayer] = None
        rounds = 0
        while rounds < max_rounds:
            for p in self.players:
                p.take_turn(self.score_fn, self.target_score)
                if p.score >= self.target_score and not final_round_flag:
                    final_round_flag = True
                    trigger_player = p
            rounds += 1
            if final_round_flag:
                for p in self.players:
                    if p is not trigger_player:
                        p.take_turn(self.score_fn, self.target_score)
                break
        winner = max(self.players, key=lambda pl: pl.score)
        per_player_stats = {
            p.name: {
                "score": p.score,
                "farkles": p.n_farkles,
                "rolls": p.n_rolls,
                "highest_turn": p.highest_turn,
                "strategy": str(p.strategy),
            }
            for p in self.players
        }
        return GameMetrics(winner=winner.name, winning_score=winner.score,
                           n_rounds=rounds, per_player=per_player_stats)


# ---------------------------------------------------------------------------
# Strategy grid helpers
# ---------------------------------------------------------------------------

def generate_strategy_grid(
    *,
    score_thresholds: Sequence[int] | None = None,
    dice_thresholds: Sequence[int] | None = None,
    smart_options: Sequence[bool] | None = None,
    consider_score_opts: Sequence[bool] | None = (True, False),
    consider_dice_opts: Sequence[bool] | None = (True, False),
) -> Tuple[List[ThresholdStrategy], pd.DataFrame]:
    """Return (*strategies*, *meta_df*).

    Defaults reproduce the original ~800‑strategy sweep used in the
    master's project while keeping everything pluggable for meta tours.
    """
    score_thresholds = score_thresholds or list(range(200, 1050, 50))
    dice_thresholds = dice_thresholds or list(range(0, 5))
    smart_options = smart_options or [True, False]

    combos = list(product(score_thresholds, dice_thresholds, smart_options, #type: ignore
                          consider_score_opts, consider_dice_opts))
    strategies: List[ThresholdStrategy] = [
        ThresholdStrategy(st, dt, sm, cs, cd)
        for st, dt, sm, cs, cd in combos
    ]
    meta = pd.DataFrame(combos, columns=[
        "score_threshold", "dice_threshold", "smart",
        "consider_score", "consider_dice",
    ])
    meta["strategy_idx"] = meta.index
    return strategies, meta


def experiment_size(*, score_thresholds: Sequence[int] | None = None,
                    dice_thresholds: Sequence[int] | None = None,
                    smart_options: Sequence[bool] | None = None,
                    consider_score_opts: Sequence[bool] | None = (True, False),
                    consider_dice_opts: Sequence[bool] | None = (True, False)) -> int:
    """Number of unique strategies given the design space."""
    score_thresholds = score_thresholds or list(range(200, 1050, 50))
    dice_thresholds = dice_thresholds or list(range(0, 5))
    smart_options = smart_options or [True, False]
    try:
        tot = (len(score_thresholds) * len(dice_thresholds) * len(smart_options)
            * len(consider_score_opts) * len(consider_dice_opts))
    except:
        print("Insufficient args or kwargs supplied to experiment_size")
    return tot


# ---------------------------------------------------------------------------
# Batch simulation helpers
# ---------------------------------------------------------------------------

def _single_game_worker(args: Tuple[int, Sequence[ThresholdStrategy], int]) -> Dict[str, Any]:
    seed, strategies, target_score = args
    rng = random.Random(seed)
    players = [FarklePlayer(name=f"P{i+1}", strategy=s, rng=rng)
               for i, s in enumerate(strategies)]
    gm = FarkleGame(players, target_score=target_score).play()
    flat = {"winner": gm.winner, "winning_score": gm.winning_score,
            "n_rounds": gm.n_rounds}
    for pname, st in gm.per_player.items():
        for k, v in st.items():
            flat[f"{pname}_{k}"] = v
    return flat


def simulate_many_games(*, n_games: int, strategies: Sequence[ThresholdStrategy],
                        target_score: int = 10_000, seed: int | None = None,
                        n_jobs: int = 1) -> pd.DataFrame:
    rng_master = random.Random(seed)
    seeds = [rng_master.randint(0, 2**32 - 1) for _ in range(n_games)]
    args = [(s, strategies, target_score) for s in seeds]
    if n_jobs == 1:
        rows = [_single_game_worker(a) for a in args]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            rows = pool.map(_single_game_worker, args)
    return pd.DataFrame(rows)


def simulate_one_game(*, strategies: Sequence[ThresholdStrategy], target_score: int = 10_000,
                      seed: int | None = None) -> GameMetrics:
    rng = random.Random(seed)
    players = [FarklePlayer(name=f"P{i+1}", strategy=s, rng=rng)
               for i, s in enumerate(strategies)]
    return FarkleGame(players, target_score=target_score).play()


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def aggregate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    agg: Dict[str, Any] = {"games": len(df), "avg_rounds": df["n_rounds"].mean()}
    agg["winner_freq"] = df["winner"].value_counts().to_dict()
    return agg


# ---------------------------------------------------------------------------
# Quick random strategy generator
# ---------------------------------------------------------------------------

def random_threshold_strategy(rng: random.Random | None = None) -> ThresholdStrategy:
    rng = rng or random
    return ThresholdStrategy(
        score_threshold=rng.randrange(50, 1000, 50),
        dice_threshold=rng.randint(0, 4),
        smart=rng.choice([True, False]),
        consider_score=rng.choice([True, False]),
        consider_dice=rng.choice([True, False]),
    )


# ---------------------------------------------------------------------------
# __main__ smoke‑test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Strategy grid demo (defaults) ---")
    strats, meta = generate_strategy_grid()
    print("Total strategies:", len(strats))
    print(meta.head())

    print("--- Batch simulate quick sanity check (5 strategies, 100 games) ---")
    df = simulate_many_games(n_games=100, strategies=strats[:5], n_jobs=max(mp.cpu_count()-1, 1))
    print(aggregate_metrics(df))
