"""tests/test_run_tournament_integration.py
================================================
End -to -end integration tests for the *parallel* Monte -Carlo tournament.

The canonical configuration explores 8 160 strategies and simulates many
millions of games  - far too expensive for unit -tests.  We therefore:

1.  Monkey -patch farkle.simulation.generate_strategy_grid so it returns a
    **deterministic list of 10 random ThresholdStrategy objects**.
2.  Shrink the timing/size constants (shuffles, games, etc.) to run in under a
    second.
3.  Keep the **real** :class:concurrent.futures.ProcessPoolExecutor so we
    still spawn worker processes and exercise the serialization logic.

Both the direct API and the CLI wrapper are covered.
"""

from __future__ import annotations

import importlib
import pickle
import random
from pathlib import Path
from typing import List, Sequence  # noqa: F401

import pytest
import yaml

pytest.importorskip("pydantic")

import farkle.simulation.simulation as sim
from farkle.cli import main as cli_main
from farkle.simulation.run_tournament import TournamentConfig
from farkle.simulation.strategies import ThresholdStrategy

###############################################################################
# 1. A tiny deterministic grid (10 strategies)
###############################################################################


def _tiny_strategy_grid(seed: int = 0) -> List[ThresholdStrategy]:
    """Return *exactly* 10 valid ThresholdStrategy objects.

    We need valid combinations, i.e. smart_one may only be *True* if
    smart_five is also *True*.
    """

    rng = random.Random(seed)
    strategies: List[ThresholdStrategy] = []
    while len(strategies) < 10:
        smart_five = rng.choice([True, False])
        smart_one = rng.choice([True, False]) and smart_five  # enforce rule
        try:
            strategies.append(
                ThresholdStrategy(
                    score_threshold=rng.randrange(500, 5000, 250),
                    dice_threshold=rng.randint(1, 6),
                    smart_five=smart_five,
                    smart_one=smart_one,
                )
            )
        except ValueError:
            # Paranoia  - should never trip because we enforce the rule above
            continue
    return strategies


###############################################################################
# 2. Worker init  - must be *top -level* so Windows can pickle it
###############################################################################


def _init_worker_small(  # pragma: no cover
    strategies: Sequence[ThresholdStrategy],
    cfg: object,
    *_extra: object,            # ← swallow optional 3rd positional arg
) -> None:
    """Accepts 2 or 3 positional arguments so we can reuse it after the row-queue
    parameter was added.
    Executed in every spawned process.

    We re-import the module *inside* the worker so we mutate the *child's*
    globals, **not** the parent's.  With the spawn method each worker starts
    with a fresh interpreter.
    """

    import importlib as _imp  # local import to avoid leak in parent

    _rt = _imp.import_module("farkle.simulation.run_tournament")
    _rt.TournamentConfig.games_per_shuffle = property(lambda self: 2)  # type: ignore  # noqa: ARG005
    _rt._STATE = _rt.WorkerState(list(strategies), cfg, None)  # type: ignore


###############################################################################
# 3. Parent -side monkey -patch helper
###############################################################################


def _apply_fast_patches(monkeypatch: pytest.MonkeyPatch, rt) -> TournamentConfig:  # type: ignore
    """Patch **everything** needed for a lightning -fast run.

    * constants (shuffles, games, timing)
    * worker initializer
    * *and* the generate_strategy_grid global so the *parent* passes only
      10 strategies to every worker via initargs.
    """

    # Compute once and reuse → deterministic
    _tiny_grid = _tiny_strategy_grid()

    # 3a. Constants in parent (affects chunking logic)
    monkeypatch.setattr(rt, "DESIRED_SEC_PER_CHUNK", 0.1, raising=False)
    monkeypatch.setattr(rt, "CKPT_EVERY_SEC", 1, raising=False)

    # 3b. Replace the worker -initializer
    monkeypatch.setattr(rt, "_init_worker", _init_worker_small, raising=True)

    # 3c. Make *this* interpreter use the tiny grid → the list also travels to
    #      every worker because run_tournament passes it via *initargs*.
    monkeypatch.setattr(rt, "generate_strategy_grid", lambda: (_tiny_grid, None), raising=True)

    monkeypatch.setattr(rt.TournamentConfig, "games_per_shuffle", property(lambda self: 2), raising=False)  # noqa: ARG005

    return rt.TournamentConfig(
        n_players=rt.TournamentConfig().n_players,
        num_shuffles=rt.NUM_SHUFFLES,
        desired_sec_per_chunk=rt.DESIRED_SEC_PER_CHUNK,
        ckpt_every_sec=rt.CKPT_EVERY_SEC,
    )


###############################################################################
# 4. Test 1  - direct call (multi -process path)
###############################################################################


def test_run_tournament_process_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Patch the strategy grid **before** importing the target so its global
    # alias is already the tiny version at import -time.
    monkeypatch.setattr(sim, "generate_strategy_grid", lambda: (_tiny_strategy_grid(), None))

    rt = importlib.import_module("farkle.simulation.run_tournament")
    cfg = _apply_fast_patches(monkeypatch, rt)

    ckpt = tmp_path / "ppool.pkl"
    kwargs = {
        "global_seed": 123,
        "checkpoint_path": ckpt,
        "n_jobs": 2,
        "num_shuffles": 2,
    }
    kwargs["config"] = cfg
    rt.run_tournament(**kwargs)

    # --- assertions ---
    assert ckpt.exists(), "checkpoint not created"
    with ckpt.open("rb") as fh:
        data = pickle.load(fh)

    wins = data["win_totals"] if isinstance(data, dict) else data
    assert wins, "empty win counter"

    expected = {str(s) for s in _tiny_strategy_grid()}
    assert set(wins.keys()).issubset(expected)


###############################################################################
# 5. Test 2  - CLI facade
###############################################################################


def test_run_tournament_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(sim, "generate_strategy_grid", lambda: (_tiny_strategy_grid(), None))

    rt = importlib.reload(importlib.import_module("farkle.simulation.run_tournament"))
    _apply_fast_patches(monkeypatch, rt)
    monkeypatch.setattr(cli_main, "run_tournament", rt.run_tournament)

    ckpt = tmp_path / "cli.pkl"

    cfg = {
        "global_seed": 42,
        "checkpoint_path": str(ckpt),
        "n_jobs": 2,
        "num_shuffles": 2,
    }
    cfg_path = tmp_path / "cli.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cli_main.main(["--config", str(cfg_path), "run"])

    # --- assertions ---
    assert ckpt.exists(), "CLI failed to create checkpoint"
    with ckpt.open("rb") as fh:
        data = pickle.load(fh)

    wins = data["win_totals"] if isinstance(data, dict) else data
    expected = {str(s) for s in _tiny_strategy_grid()}
    assert set(wins.keys()).issubset(expected)
