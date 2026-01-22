# src/farkle/simulation/strategies.py
"""Simulation strategy helpers for the Farkle engine.

Provides the FavorDiceOrScore tie-break enum, the ThresholdStrategy decision
heuristic, and utilities that randomize or parse strategy definitions used by
the simulation pipeline.
"""
from __future__ import annotations

import pickle
import random
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple

import numba as nb
import pandas as pd

from farkle.utils.types import DiceRoll  # noqa: F401 Likely needed for decide(*)

__all__: list[str] = [
    "FavorDiceOrScore",
    "StopAtStrategy",
    "STOP_AT_THRESHOLDS",
    "build_stop_at_strategy",
    "STOP_AT_REGISTRY",
    "ThresholdStrategy",
    "StrategyEncoder",
    "StrategyGridOptions",
    "STRATEGY_TUPLE_FIELDS",
    "DEFAULT_STRATEGY_GRID",
    "STRATEGY_MANIFEST_NAME",
    "build_strategy_manifest",
    "build_strategy_encoder",
    "decode_strategy_id",
    "encode_strategy",
    "iter_strategy_combos",
    "coerce_strategy_ids",
    "normalize_strategy_ids",
    "parse_strategy_identifier",
    "strategy_attributes_from_series",
    "strategy_tuple",
    "random_threshold_strategy",
]


class FavorDiceOrScore(Enum):
    """Tie-break preference when both score and dice targets are hit."""

    SCORE = "score"
    DICE = "dice"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


STOP_AT_THRESHOLDS: tuple[int, ...] = (350, 400, 450, 500)
"""Named stop-at thresholds available via the strategy registry."""

STRATEGY_TUPLE_FIELDS: tuple[str, ...] = (
    "score_threshold",
    "dice_threshold",
    "smart_five",
    "smart_one",
    "consider_score",
    "consider_dice",
    "require_both",
    "auto_hot_dice",
    "run_up_score",
    "favor_dice_or_score",
)

STRATEGY_MANIFEST_NAME = "strategy_manifest.parquet"

DEFAULT_STRATEGY_GRID: dict[str, tuple[object, ...]] = {
    "score_thresholds": tuple(range(200, 1400, 50)),
    "dice_thresholds": tuple(range(0, 5)),
    "smart_five_opts": (True, False),
    "smart_one_opts": (True, False),
    "consider_score_opts": (True, False),
    "consider_dice_opts": (True, False),
    "auto_hot_dice_opts": (False, True),
    "run_up_score_opts": (True, False),
}

StrategyTuple = Tuple[int, int, bool, bool, bool, bool, bool, bool, bool, FavorDiceOrScore]


_STRAT_RE = re.compile(
    r"""
    \A
    Strat\(\s*(?P<score>\d+)\s*,\s*(?P<dice>\d+)\s*\)  # thresholds
    \[
        (?P<cs>[S\-])(?P<cd>[D\-])
    \]
    \[
        (?P<sf>[F\-])  # smart_five block
        (?P<so>[O\-])  # smart_one block
        (?P<fs>FS|FD)
    \]
    \[
        (?P<rb>AND|OR)
    \]
    \[
        (?P<hd>[H\-])(?P<rs>[R\-])
    \]
    \Z
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _decide_continue(
    turn_score,
    dice_left,
    score_threshold,
    dice_threshold,
    consider_score,
    consider_dice,
    require_both,
) -> bool:
    """Return ``True`` to keep rolling based on score/dice thresholds.

    Parameters
    ----------
    turn_score, dice_left
        Current turn score and dice remaining.
    score_threshold, dice_threshold
        Limits governing when to stop rolling.
    consider_score, consider_dice
        Flags enabling the above limits.
    require_both
        When both consider_score and consider_dice flags are set to ``True``,
        if ``require_both`` is ``True`` the function continues when *either*
        limit is still unmet (``OR``);
        if ``False`` it continues only when *both* limits are unmet (``AND``).
    """
    # The function asks if we continue while the thresholds dictate when to stop.
    # As a result, the logic here looks incorrect at first glance but it is correct.
    # want_s and want_d are checking if limits are NOT hit for easier boolean logic.
    want_s = consider_score and turn_score < score_threshold  # want higher score
    want_d = consider_dice and dice_left > dice_threshold  # want to spend more dice
    if consider_score and consider_dice:  # the booleans here are counterintuitive but correct
        return (want_s or want_d) if require_both else (want_s and want_d)
    if consider_score:
        return want_s
    if consider_dice:
        return want_d
    return False


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
    smart_five: bool = False  # “throw back lone 5’s first”
    smart_one: bool = False
    consider_score: bool = True
    consider_dice: bool = True
    require_both: bool = False
    auto_hot_dice: bool = False
    run_up_score: bool = False
    favor_dice_or_score: FavorDiceOrScore = FavorDiceOrScore.SCORE
    strategy_id: int | None = None

    def __post_init__(self):
        # 1) smart_one may never be True if smart_five is False
        if self.smart_one and not self.smart_five:
            raise ValueError("ThresholdStrategy: smart_one=True requires smart_five=True")

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
        score_needed: int,  # noqa: ARG002 (vestigial/reserved for richer strats)
        final_round: bool = False,  # (new end-game hooks - harmless defaults)
        score_to_beat: int = 0,
        running_total: int = 0,
    ) -> bool:  # noqa: D401 - imperative name
        """Decide whether the player should keep rolling this turn.

        Parameters
        ----------
        turn_score : int
            Points accumulated so far during the current turn.
        dice_left : int
            Number of dice that remain available to roll.
        has_scored : bool
            Whether the player has banked any scoring dice this turn.
        score_needed : int
            Margin required to win; retained for compatibility with richer strategies.
        final_round : bool, default False
            True when the table is in the final round and catch-up logic applies.
        score_to_beat : int, default 0
            Opponent score threshold to surpass during the final round.
        running_total : int, default 0
            Player total score before banking the current turn.

        Returns
        -------
        bool
            True to continue rolling; False to bank the current turn score.
        """

        # --------------------------------- fast exits ---------------------------------
        if not has_scored and turn_score < 500:
            return True  # must cross the 500-pt entry gate

        # final-round catch-up rule
        if final_round:
            # Must beat the leader; ties don't win.
            if running_total <= score_to_beat:
                return True  # keep rolling until you beat them

            # Already ahead:
            if not self.run_up_score:
                return False  # auto-bank if not running up

        # self.run_up_score == True -> fall through to normal thresholds

        # -----------------------------------------------------------------------------
        keep_rolling = _decide_continue(
            turn_score,
            dice_left,
            self.score_threshold,
            self.dice_threshold,
            self.consider_score,
            self.consider_dice,
            self.require_both,
        )

        return keep_rolling

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:  # noqa: D401 - magic method
        cs = "S" if self.consider_score else "-"
        cd = "D" if self.consider_dice else "-"
        sf = "F" if self.smart_five else "-"
        so = "O" if self.smart_one else "-"
        rb = "AND" if self.require_both else "OR"
        hd = "H" if self.auto_hot_dice else "-"
        rs = "R" if self.run_up_score else "-"
        fs = "FS" if self.favor_dice_or_score is FavorDiceOrScore.SCORE else "FD"
        return f"Strat({self.score_threshold},{self.dice_threshold})[{cs}{cd}][{sf}{so}{fs}][{rb}][{hd}{rs}]"


@dataclass
class StopAtStrategy(ThresholdStrategy):
    """Named strategy that banks once a turn score crosses a fixed level."""

    label: str = ""
    heuristic: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not re.match(r"stop_at_\d+(?:_heuristic)?\Z", self.label):
            raise ValueError(f"Invalid stop-at strategy label: {self.label!r}")

    def __str__(self) -> str:  # noqa: D401 - magic method
        return self.label


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def _coerce_options(options: Sequence[Any] | None, fallback: Iterable[Any]) -> tuple[Any, ...]:
    """Return an immutable tuple using ``fallback`` when ``options`` is None."""
    return tuple(fallback) if options is None else tuple(options)


def _favor_options(sf: bool, cs: bool, cd: bool) -> tuple[FavorDiceOrScore, ...]:
    """Return valid FavorDiceOrScore choices for the given flag combination."""
    if cs and cd:
        return (FavorDiceOrScore.SCORE, FavorDiceOrScore.DICE) if sf else (FavorDiceOrScore.SCORE,)
    if cs:
        return (FavorDiceOrScore.SCORE,)
    if cd:
        return (FavorDiceOrScore.DICE,)
    return (FavorDiceOrScore.SCORE,)


def iter_strategy_combos(
    *,
    score_thresholds: Sequence[int],
    dice_thresholds: Sequence[int],
    smart_five_opts: Sequence[bool],
    smart_one_opts: Sequence[bool],
    consider_score_opts: Sequence[bool],
    consider_dice_opts: Sequence[bool],
    auto_hot_dice_opts: Sequence[bool],
    run_up_score_opts: Sequence[bool],
    inactive_score_threshold: int,
    inactive_dice_threshold: int,
    allowed_smart_pairs: set[tuple[bool, bool]] | None = None,
) -> Iterable[StrategyTuple]:
    """Iterate over strategy parameter tuples that respect flag constraints."""
    for sf in smart_five_opts:
        smart_one_candidates = [
            so
            for so in smart_one_opts
            if (sf or not so) and (allowed_smart_pairs is None or (sf, so) in allowed_smart_pairs)
        ]
        if not smart_one_candidates:
            continue

        for so in smart_one_candidates:
            for cs in consider_score_opts:
                score_values = score_thresholds if cs else [inactive_score_threshold]

                for cd in consider_dice_opts:
                    dice_values = dice_thresholds if cd else [inactive_dice_threshold]
                    rb_values = [True, False] if (cs and cd) else [False]
                    favor_choices = _favor_options(sf, cs, cd)

                    for st in score_values:
                        for dt in dice_values:
                            for hd in auto_hot_dice_opts:
                                for rs in run_up_score_opts:
                                    for rb in rb_values:
                                        for ps in favor_choices:
                                            yield (
                                                int(st),
                                                int(dt),
                                                bool(sf),
                                                bool(so),
                                                bool(cs),
                                                bool(cd),
                                                bool(rb),
                                                bool(hd),
                                                bool(rs),
                                                ps,
                                            )


def _sample_favor_score(cs: bool, cd: bool, rng: random.Random) -> FavorDiceOrScore:
    """
    Return the *only* legal value(s) for `favor_dice_or_score`
    given the (consider_score, consider_dice) pair.

        cs  cd   →  favor_dice_or_score
        ─────────────────────────
        T   F       True    (always favor score)
        F   T       False   (always favor dice)
        T   T       rng     (tie-break random)
        F   F       rng     (doesn't matter - random)

    Much easier to read than a stacked ternary.
    """
    if cs == cd:  # (T,T) or (F,F)   →  free choice
        return rng.choice([FavorDiceOrScore.SCORE, FavorDiceOrScore.DICE])
    return FavorDiceOrScore.SCORE if cs else FavorDiceOrScore.DICE


def random_threshold_strategy(rng: random.Random | None = None) -> ThresholdStrategy:
    """Create a randomized threshold strategy consistent with engine rules.

    Parameters
    ----------
    rng : random.Random | None, default None
        Source of randomness. A new generator is created when ``None`` is provided.

    Returns
    -------
    ThresholdStrategy
        Strategy instance populated with randomly sampled thresholds and flags that
        respect required invariants.
    """

    rng_inst = rng if rng is not None else random.Random()

    # pick smart_five first; if it’s False, force smart_one=False
    sf = rng_inst.choice([True, False])
    so = rng_inst.choice([True, False]) if sf else False

    # pick consider_score/dice; if either is False, force require_both=False
    cs = rng_inst.choice([True, False])
    cd = rng_inst.choice([True, False])
    rb = rng_inst.choice([True, False]) if (cs and cd) else False
    fs = _sample_favor_score(cs, cd, rng_inst)

    return ThresholdStrategy(
        score_threshold=rng_inst.randrange(50, 1000, 50),
        dice_threshold=rng_inst.randint(0, 4),
        smart_five=sf,
        smart_one=so,
        consider_score=cs,
        consider_dice=cd,
        require_both=rb,
        favor_dice_or_score=fs,
    )


def build_stop_at_strategy(
    threshold: int,
    *,
    heuristic: bool = False,
    inactive_dice_threshold: int | None = None,
) -> StopAtStrategy:
    """Create a stop-at strategy with an optional heuristic variant."""

    if threshold not in STOP_AT_THRESHOLDS:
        raise ValueError(f"Unregistered stop-at threshold: {threshold}")

    dice_threshold = -1 if inactive_dice_threshold is None else inactive_dice_threshold
    label = f"stop_at_{threshold}" + ("_heuristic" if heuristic else "")

    return StopAtStrategy(
        score_threshold=threshold,
        dice_threshold=dice_threshold,
        smart_five=heuristic,
        smart_one=heuristic,
        consider_score=True,
        consider_dice=False,
        require_both=False,
        auto_hot_dice=heuristic,
        run_up_score=False,
        favor_dice_or_score=FavorDiceOrScore.SCORE,
        label=label,
        heuristic=heuristic,
    )


STOP_AT_REGISTRY: dict[str, Callable[..., StopAtStrategy]] = {
    **{
        f"stop_at_{threshold}": partial(build_stop_at_strategy, threshold)
        for threshold in STOP_AT_THRESHOLDS
    },
    **{
        f"stop_at_{threshold}_heuristic": partial(
            build_stop_at_strategy, threshold, heuristic=True
        )
        for threshold in STOP_AT_THRESHOLDS
    },
}


def strategy_tuple(strategy: ThresholdStrategy) -> StrategyTuple:
    """Return the canonical tuple representation for a ThresholdStrategy."""
    return tuple(
        getattr(strategy, field) for field in STRATEGY_TUPLE_FIELDS
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class StrategyGridOptions:
    """Normalized option grid inputs for strategy encoders and manifests."""

    score_thresholds: tuple[int, ...]
    dice_thresholds: tuple[int, ...]
    smart_five_opts: tuple[bool, ...]
    smart_one_opts: tuple[bool, ...]
    consider_score_opts: tuple[bool, ...]
    consider_dice_opts: tuple[bool, ...]
    auto_hot_dice_opts: tuple[bool, ...]
    run_up_score_opts: tuple[bool, ...]
    include_stop_at: bool = False
    include_stop_at_heuristic: bool = False

    @property
    def inactive_score_threshold(self) -> int:
        return min(self.score_thresholds) - 1

    @property
    def inactive_dice_threshold(self) -> int:
        return min(self.dice_thresholds) - 1

    @classmethod
    def from_inputs(
        cls,
        *,
        score_thresholds: Sequence[int] | None = None,
        dice_thresholds: Sequence[int] | None = None,
        smart_five_opts: Sequence[bool] | None = None,
        smart_one_opts: Sequence[bool] | None = None,
        consider_score_opts: Sequence[bool] = (True, False),
        consider_dice_opts: Sequence[bool] = (True, False),
        auto_hot_dice_opts: Sequence[bool] = (False, True),
        run_up_score_opts: Sequence[bool] = (True, False),
        include_stop_at: bool = False,
        include_stop_at_heuristic: bool = False,
    ) -> "StrategyGridOptions":
        return cls(
            score_thresholds=_coerce_options(
                score_thresholds, DEFAULT_STRATEGY_GRID["score_thresholds"]
            ),
            dice_thresholds=_coerce_options(
                dice_thresholds, DEFAULT_STRATEGY_GRID["dice_thresholds"]
            ),
            smart_five_opts=_coerce_options(
                smart_five_opts, DEFAULT_STRATEGY_GRID["smart_five_opts"]
            ),
            smart_one_opts=_coerce_options(
                smart_one_opts, DEFAULT_STRATEGY_GRID["smart_one_opts"]
            ),
            consider_score_opts=_coerce_options(
                consider_score_opts, DEFAULT_STRATEGY_GRID["consider_score_opts"]
            ),
            consider_dice_opts=_coerce_options(
                consider_dice_opts, DEFAULT_STRATEGY_GRID["consider_dice_opts"]
            ),
            auto_hot_dice_opts=_coerce_options(
                auto_hot_dice_opts, DEFAULT_STRATEGY_GRID["auto_hot_dice_opts"]
            ),
            run_up_score_opts=_coerce_options(
                run_up_score_opts, DEFAULT_STRATEGY_GRID["run_up_score_opts"]
            ),
            include_stop_at=include_stop_at,
            include_stop_at_heuristic=include_stop_at_heuristic,
        )


@dataclass(frozen=True)
class StrategyEncoder:
    """Deterministic encoder/decoder for strategy tuples."""

    options: StrategyGridOptions
    tuples: tuple[StrategyTuple, ...]
    tuple_to_id: Mapping[StrategyTuple, int]

    def encode_tuple(self, combo: StrategyTuple) -> int:
        """Return the integer identifier for a strategy tuple."""
        return int(self.tuple_to_id[combo])

    def decode_id(self, strategy_id: int) -> dict[str, Any]:
        """Return the attribute dict for ``strategy_id``."""
        combo = self.tuples[int(strategy_id)]
        return dict(zip(STRATEGY_TUPLE_FIELDS, combo, strict=True))

    def encode_strategy(self, strategy: ThresholdStrategy) -> int:
        """Return the integer identifier for a ThresholdStrategy."""
        return self.encode_tuple(strategy_tuple(strategy))


def _iter_encoder_combos(options: StrategyGridOptions) -> Iterable[StrategyTuple]:
    """Yield strategy tuples in deterministic order for encoding."""
    yield from iter_strategy_combos(
        score_thresholds=options.score_thresholds,
        dice_thresholds=options.dice_thresholds,
        smart_five_opts=options.smart_five_opts,
        smart_one_opts=options.smart_one_opts,
        consider_score_opts=options.consider_score_opts,
        consider_dice_opts=options.consider_dice_opts,
        auto_hot_dice_opts=options.auto_hot_dice_opts,
        run_up_score_opts=options.run_up_score_opts,
        inactive_score_threshold=options.inactive_score_threshold,
        inactive_dice_threshold=options.inactive_dice_threshold,
    )

    if options.include_stop_at:
        for threshold in STOP_AT_THRESHOLDS:
            strat = build_stop_at_strategy(
                threshold, inactive_dice_threshold=options.inactive_dice_threshold
            )
            yield strategy_tuple(strat)

    if options.include_stop_at_heuristic:
        for threshold in STOP_AT_THRESHOLDS:
            strat = build_stop_at_strategy(
                threshold,
                heuristic=True,
                inactive_dice_threshold=options.inactive_dice_threshold,
            )
            yield strategy_tuple(strat)


def build_strategy_encoder(
    *,
    score_thresholds: Sequence[int] | None = None,
    dice_thresholds: Sequence[int] | None = None,
    smart_five_opts: Sequence[bool] | None = None,
    smart_one_opts: Sequence[bool] | None = None,
    consider_score_opts: Sequence[bool] = (True, False),
    consider_dice_opts: Sequence[bool] = (True, False),
    auto_hot_dice_opts: Sequence[bool] = (False, True),
    run_up_score_opts: Sequence[bool] = (True, False),
    include_stop_at: bool = False,
    include_stop_at_heuristic: bool = False,
) -> StrategyEncoder:
    """Build a deterministic encoder for the provided strategy grid options."""
    options = StrategyGridOptions.from_inputs(
        score_thresholds=score_thresholds,
        dice_thresholds=dice_thresholds,
        smart_five_opts=smart_five_opts,
        smart_one_opts=smart_one_opts,
        consider_score_opts=consider_score_opts,
        consider_dice_opts=consider_dice_opts,
        auto_hot_dice_opts=auto_hot_dice_opts,
        run_up_score_opts=run_up_score_opts,
        include_stop_at=include_stop_at,
        include_stop_at_heuristic=include_stop_at_heuristic,
    )
    return _build_strategy_encoder_cached(options)


@lru_cache(maxsize=None)
def _build_strategy_encoder_cached(options: StrategyGridOptions) -> StrategyEncoder:
    """Cached builder keyed by a frozen StrategyGridOptions."""

    tuples: list[StrategyTuple] = []
    tuple_to_id: dict[StrategyTuple, int] = {}
    for combo in _iter_encoder_combos(options):
        if combo not in tuple_to_id:
            tuple_to_id[combo] = len(tuples)
            tuples.append(combo)
    return StrategyEncoder(options=options, tuples=tuple(tuples), tuple_to_id=tuple_to_id)


def encode_strategy(strategy: ThresholdStrategy, encoder: StrategyEncoder) -> int:
    """Return the deterministic ID for ``strategy`` using ``encoder``."""
    return encoder.encode_strategy(strategy)


def decode_strategy_id(strategy_id: int, encoder: StrategyEncoder) -> dict[str, Any]:
    """Return strategy attribute mapping for ``strategy_id`` using ``encoder``."""
    return encoder.decode_id(strategy_id)


def build_strategy_manifest(strategies: Sequence[ThresholdStrategy]) -> pd.DataFrame:
    """Return a manifest DataFrame mapping strategy IDs to attributes."""
    rows: dict[int, dict[str, Any]] = {}
    for strat in strategies:
        if strat.strategy_id is None:
            continue
        sid = int(strat.strategy_id)
        if sid in rows:
            continue
        attrs: dict[str, Any] = dict(
            zip(STRATEGY_TUPLE_FIELDS, strategy_tuple(strat), strict=True)
        )
        attrs["strategy_id"] = sid
        attrs["strategy_str"] = str(strat)
        if isinstance(attrs["favor_dice_or_score"], FavorDiceOrScore):
            attrs["favor_dice_or_score"] = attrs["favor_dice_or_score"].value
        rows[sid] = attrs

    manifest = pd.DataFrame(rows.values())
    if not manifest.empty:
        manifest = manifest.sort_values("strategy_id", kind="mergesort").reset_index(drop=True)
    return manifest


def normalize_strategy_ids(series: pd.Series) -> pd.Series:
    """Coerce a series of strategy identifiers into nullable integers."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def coerce_strategy_ids(series: pd.Series) -> pd.Series:
    """Return strategy identifiers with numeric IDs coerced but legacy strings preserved."""
    normalized = normalize_strategy_ids(series)
    return normalized.astype(object).where(normalized.notna(), series)


def parse_strategy_identifier(
    value: Any,
    *,
    encoder: StrategyEncoder | None = None,
    manifest: pd.DataFrame | None = None,
    parse_legacy: Callable[[str], dict] | None = None,
) -> ThresholdStrategy:
    """Return a ThresholdStrategy from an identifier (id or legacy string)."""
    if isinstance(value, int) and not isinstance(value, bool) or isinstance(value, str) and value.isdigit():
        strategy_id = int(value)
    else:
        strategy_id = None

    if strategy_id is not None:
        attrs: dict[str, Any] | None = None
        if encoder is not None:
            attrs = encoder.decode_id(strategy_id)
        elif manifest is not None and not manifest.empty:
            match = manifest.loc[manifest["strategy_id"] == strategy_id]
            if not match.empty:
                attrs = {str(k): v for k, v in match.iloc[0].to_dict().items()}
        if attrs is None:
            raise KeyError(f"strategy_id {strategy_id} missing from manifest/encoder")
        attrs = {k: v for k, v in attrs.items() if k in STRATEGY_TUPLE_FIELDS}
        if "favor_dice_or_score" in attrs and not isinstance(
            attrs["favor_dice_or_score"], FavorDiceOrScore
        ):
            attrs["favor_dice_or_score"] = (
                FavorDiceOrScore.SCORE
                if attrs["favor_dice_or_score"] == FavorDiceOrScore.SCORE.value
                else FavorDiceOrScore.DICE
            )
        return ThresholdStrategy(**attrs, strategy_id=strategy_id)

    if isinstance(value, str) and value in STOP_AT_REGISTRY:
        return STOP_AT_REGISTRY[value]()

    if parse_legacy is None:
        raise ValueError(f"Cannot parse legacy strategy identifier: {value!r}")
    strategy = ThresholdStrategy(**parse_legacy(str(value)))
    return strategy


def strategy_attributes_from_series(
    strategies: pd.Series,
    *,
    manifest: pd.DataFrame | None = None,
    parse_legacy: Callable[[str], dict] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of strategy attributes for a mixed identifier series."""
    if parse_legacy is None:
        parse_legacy = parse_strategy_for_df
    id_series = normalize_strategy_ids(strategies)
    attrs_frames: list[pd.DataFrame] = []
    if manifest is not None and not manifest.empty and id_series.notna().any():
        manifest_indexed = manifest.set_index("strategy_id")
        ids = id_series.dropna().astype(int)
        mapped = manifest_indexed.reindex(ids.values)
        mapped = mapped.reset_index(drop=True)
        mapped.index = ids.index
        attrs_frames.append(mapped[list(STRATEGY_TUPLE_FIELDS)])

    missing_mask = id_series.isna() & strategies.notna()
    if parse_legacy is not None and missing_mask.any():
        legacy_attrs = strategies[missing_mask].apply(parse_legacy).apply(pd.Series)
        legacy_attrs = legacy_attrs.reindex(columns=STRATEGY_TUPLE_FIELDS)
        attrs_frames.append(legacy_attrs)

    if not attrs_frames:
        return pd.DataFrame(columns=STRATEGY_TUPLE_FIELDS)

    combined = pd.concat(attrs_frames).sort_index()
    return combined.reindex(columns=STRATEGY_TUPLE_FIELDS)


def _parse_strategy_flags(s: str) -> dict[str, Any]:
    """Return a mapping of strategy fields parsed from ``s``."""

    m = _STRAT_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse strategy string: {s!r}")

    return {
        "score_threshold": int(m.group("score")),
        "dice_threshold": int(m.group("dice")),
        "smart_five": m.group("sf") == "F",
        "smart_one": m.group("so") == "O",
        "consider_score": m.group("cs") == "S",
        "consider_dice": m.group("cd") == "D",
        "require_both": m.group("rb") == "AND",
        "auto_hot_dice": m.group("hd") == "H",
        "run_up_score": m.group("rs") == "R",
        "favor_dice_or_score": (
            FavorDiceOrScore.SCORE if m.group("fs") == "FS" else FavorDiceOrScore.DICE
        ),
    }


def parse_strategy(s: str) -> ThresholdStrategy:
    """Parse a serialized strategy string into a ThresholdStrategy instance.

    Parameters
    ----------
    s : str
        Strategy literal produced by ``ThresholdStrategy.__str__``, e.g.
        ``'Strat(300,2)[SD][FO][AND][H-]'``.

    Returns
    -------
    ThresholdStrategy
        Strategy configured with the thresholds and flags encoded in ``s``.

    Notes
    -----
    Intended for analysis and tooling rather than the simulation hot path.
    """
    # 1) Top-level pattern:  Strat(score_threshold, dice_threshold)
    # 2) Four bracketed blocks:
    #    [<consider_score><consider_dice>]
    #    [<smart_five><smart_one>]
    #    [AND|OR]
    #    [<auto_hot><run_up>]
    #
    #   where:
    #     consider_score = "S" or "-"
    #     consider_dice  = "D" or "-"
    #     smart_five     = "F" or "-"
    #     smart_one      = "O" or "-"
    #     auto_hot       = "H" or "-"
    #     run_up_score   = "R" or "-"
    #     require_both   = "AND" or "OR"
    #     favor_dice_or_score   = "FS" or "FD"
    #
    # Example literal: "Strat(300,2)[SD][FOFD][AND][H-]"

    flags = _parse_strategy_flags(s)
    return ThresholdStrategy(**flags)


def parse_strategy_for_df(s: str) -> dict:
    """Convert a strategy string into a dictionary of parsed attributes.

    Parameters
    ----------
    s : str
        Strategy literal in the ``Strat(score,dice)[...][...]`` format produced by
        ``ThresholdStrategy.__str__``.

    Returns
    -------
    dict
        Mapping of ThresholdStrategy field names to their parsed values, suitable
        for expansion into a DataFrame row.

    Notes
    -----
    Designed for analysis and log processing, not the simulation hot path.
    """
    # 1) Top-level pattern:  Strat(score_threshold, dice_threshold)
    # 2) Four bracketed blocks:
    #    [<consider_score><consider_dice>]
    #    [<smart_five><smart_one>]
    #    [AND|OR]
    #    [<auto_hot><run_up>]
    #
    #   where:
    #     consider_score = "S" or "-"
    #     consider_dice  = "D" or "-"
    #     smart_five     = "F" or "-"
    #     smart_one      = "O" or "-"
    #     auto_hot       = "H" or "-"
    #     run_up_score   = "R" or "-"
    #     require_both   = "AND" or "OR"
    #     favor_dice_or_score   = "FS" or "FD"
    #
    # Example literal: "Strat(300,2)[SD][FOFD][AND][H-]"

    return _parse_strategy_flags(s)


def load_farkle_results(
    pkl_path: str | Path,
    *,
    parse_strategy: Callable[[str], dict] = parse_strategy_for_df,
    manifest: pd.DataFrame | None = None,
    ordered: bool = True,
) -> pd.DataFrame:
    """
    Load a pickled Counter {strategy_str: wins} and explode it into a
    “full-fat” DataFrame with every strategy flag broken out.

    Warning
    -------
    Unpickling data from untrusted sources can execute arbitrary code.
    Only load pickle files from trusted sources.

    Parameters
    ----------
    pkl_path : str | Path
        Path to the pickle file produced by `run_tournament_2.py`.
        The pickle must contain a `collections.Counter` or plain dict
        whose keys are strategy strings and whose values are win counts.
    parse_strategy : callable, default ``parse_strategy_for_df``
        Function that converts one strategy string to a dict of columns.
        Override only if you have a different parser.
    ordered : bool, default True
        Whether to return the columns in a logical, pre-defined order.

    Returns
    -------
    pd.DataFrame
        Columns:
          strategy, wins,
          score_threshold, dice_threshold,
          consider_score, consider_dice, require_both, favor_dice_or_score,
          smart_five, smart_one, auto_hot_dice, run_up_score
    """
    # ------------------------------------------------------------------
    # 1) Load the Counter
    # ------------------------------------------------------------------
    pkl_path = Path(pkl_path).expanduser().resolve()
    counter = pickle.loads(pkl_path.read_bytes())

    # ------------------------------------------------------------------
    # 2) Counter → two-column DataFrame
    # ------------------------------------------------------------------
    base_df = (
        pd.Series(counter, name="wins")
        .reset_index(drop=False)
        .rename(columns={"index": "strategy"})
    )

    # ------------------------------------------------------------------
    # 3) Explode strategy identifiers into individual columns
    # ------------------------------------------------------------------
    if manifest is not None and not manifest.empty:
        flags_df = strategy_attributes_from_series(
            base_df["strategy"], manifest=manifest, parse_legacy=parse_strategy
        )
    else:
        flags_df = base_df["strategy"].apply(parse_strategy).apply(pd.Series)

    full_df = pd.concat([base_df, flags_df], axis=1)

    # ------------------------------------------------------------------
    # 4) Nice-to-have column ordering
    # ------------------------------------------------------------------
    if ordered:
        col_order = [
            "strategy",
            "wins",
            "score_threshold",
            "dice_threshold",
            "consider_score",
            "consider_dice",
            "require_both",
            "smart_five",
            "smart_one",
            "favor_dice_or_score",
            "auto_hot_dice",
            "run_up_score",
        ]
        full_df = full_df[col_order].sort_values(by="wins", ascending=False, ignore_index=True)

    return full_df
