"""
Fast, side-effect-free sanity checks for run_tournament.py
(all run in < 1 s).

We monkey-patch the heavy helpers so no real games are played.
"""

from __future__ import annotations

import logging
import pickle
import types  # noqa: F401
from collections import Counter
from pathlib import Path

import numpy as np  # noqa: F401 | Potentially imports something that needs it
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pyarrow")

import farkle.simulation.run_tournament as rt
from farkle.simulation.strategies import ThresholdStrategy

# --------------------------------------------------------------------------- #
# Mini test doubles ? replace expensive pieces with cheap determinism
# --------------------------------------------------------------------------- #


def _mini_strats(n: int = 6):
    """Return deterministic Strategy objects with distinct __str__()."""
    return [ThresholdStrategy(50 + 50 * i, i % 3, True, True) for i in range(n)]


def fake_play_shuffle(seed: int) -> Counter[str]:
    # pretend player at index (seed % len(strats)) always wins
    strats = _mini_strats(12)
    return Counter({str(strats[seed % len(strats)]): 1})


@pytest.fixture(autouse=True)
def fast_helpers(monkeypatch):
    """
    Keep tests lightning-fast **and** remember the real `_play_shuffle` so
    individual tests can restore it when they *need* the real behavior.
    """
    strats = _mini_strats(12)
    monkeypatch.setattr(
        rt, "generate_strategy_grid", lambda *a, **kw: (strats, None), raising=True  # noqa: ARG005
    )

    # save the genuine function on the module *once*
    if not hasattr(rt, "_ORIG_PLAY_SHUFFLE"):
        rt._ORIG_PLAY_SHUFFLE = rt._play_shuffle  # type: ignore

    monkeypatch.setattr(rt, "_play_shuffle", fake_play_shuffle, raising=True)


@pytest.mark.parametrize(
    ("n_players", "pattern"),
    [
        (1, r"n_players must be ≥2"),
        (8, r"n_players must divide [\d,]+"),
    ],
)
def test_run_tournament_invalid_player_counts(n_players: int, pattern: str) -> None:
    cfg = rt.TournamentConfig()
    cfg.num_shuffles = 1

    with pytest.raises(ValueError, match=pattern):
        rt.run_tournament(config=cfg, n_players=n_players)


def test_init_worker_rejects_bad_player_counts(monkeypatch) -> None:
    strats = _mini_strats(3)
    cfg = rt.TournamentConfig(n_players=7)
    monkeypatch.setattr(rt, "_STATE", None, raising=False)

    with pytest.raises(ValueError, match=r"n_players must divide [\d,]+"):
        rt._init_worker(strats, cfg)


def test_run_chunk_logs_and_propagates(monkeypatch, caplog) -> None:
    class BoomError(RuntimeError):
        pass

    def boom(_seed: int):
        raise BoomError("boom")

    monkeypatch.setattr(rt, "_play_shuffle", boom, raising=True)

    with caplog.at_level(logging.ERROR, logger=rt.LOGGER.name), pytest.raises(BoomError):
        rt._run_chunk([123])

    assert any("Shuffle failed" in rec.getMessage() for rec in caplog.records)
    logged = [rec for rec in caplog.records if "Shuffle failed" in rec.getMessage()][0]
    assert logged.stage == "simulation"
    assert logged.shuffle_seed == 123


def test_coerce_counter_and_metric_sums_helpers() -> None:
    wins = rt._coerce_counter({"1": "2", "A": 3})
    assert wins == Counter({1: 2, "A": 3})

    sums = rt._coerce_metric_sums({"winning_score": {"5": "4.5"}})
    assert sums is not None
    assert sums["winning_score"][5] == pytest.approx(4.5)
    assert sums["winner_rolls"] == {}


def test_coerce_counter_rejects_invalid_payload() -> None:
    with pytest.raises(TypeError, match="Unexpected win_totals payload type"):
        rt._coerce_counter([("A", 1)])


def test_measure_throughput_uses_spawned_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[int] = []

    monkeypatch.setattr(rt.urandom, "spawn_seeds", lambda n, seed=0: [11, 22, 33], raising=True)
    monkeypatch.setattr(
        rt,
        "_play_game",
        lambda seed, strategies: observed.append(seed) or {"winner": "p0"},
        raising=True,
    )

    perf_state = {"idx": 0, "vals": [10.0, 10.5]}

    def fake_perf() -> float:
        value = perf_state["vals"][perf_state["idx"]]
        perf_state["idx"] += 1
        return value

    monkeypatch.setattr(rt.time, "perf_counter", fake_perf, raising=True)

    value = rt._measure_throughput(_mini_strats(2), sample_games=3, seed=7)
    assert value == pytest.approx(6.0)
    assert observed == [11, 22, 33]


def test_manifest_int_set_parses_valid_values(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                '{"shuffle_seed": "5"}',
                '{"shuffle_seed": "bad"}',
                '{"shuffle_seed": 3}',
                '{"other": 9}',
            ]
        )
        + "\n"
    )

    assert rt._manifest_int_set(manifest_path, "shuffle_seed") == {3, 5}


def test_run_tournament_non_metrics_chunk_execution_and_corrupt_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    strats = _mini_strats(4)
    monkeypatch.setattr(rt, "generate_strategy_grid", lambda *a, **k: (strats, None), raising=True)
    monkeypatch.setattr(rt, "_measure_throughput", lambda sample: 8.0, raising=True)
    monkeypatch.setattr(rt.urandom, "spawn_seeds", lambda n, seed=0: list(range(n)), raising=True)

    chunk_calls: list[list[int]] = []

    def fake_process_map(func, iterable, *, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*initargs)
        for item in iterable:
            chunk_calls.append(list(item[1]))
            yield func(item)

    monkeypatch.setattr(rt.parallel, "process_map", fake_process_map)
    monkeypatch.setattr(rt, "_play_shuffle", lambda seed: Counter({f"W{seed}": 1}), raising=True)

    cfg = rt.TournamentConfig(n_players=2, num_shuffles=5, desired_sec_per_chunk=1, ckpt_every_sec=999)
    ckpt = tmp_path / "checkpoint.pkl"
    rt.run_tournament(
        config=cfg,
        num_shuffles=cfg.num_shuffles,
        checkpoint_path=ckpt,
        n_jobs=1,
        collect_metrics=False,
    )

    # shuffles_per_chunk = max(1, int(1 * 8.0 // (4 // 2))) = 4 for this fixture.
    assert chunk_calls == [[0, 1, 2, 3], [4]]
    payload = pickle.loads(ckpt.read_bytes())
    assert payload["win_totals"] == Counter({"W0": 1, "W1": 1, "W2": 1, "W3": 1, "W4": 1})

    ckpt.write_bytes(pickle.dumps(123))
    with pytest.raises(AttributeError):
        rt.run_tournament(
            config=cfg,
            num_shuffles=cfg.num_shuffles,
            checkpoint_path=ckpt,
            n_jobs=1,
            collect_metrics=False,
        )
